# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Regression tests for the ported workload cost models."""

import unittest
from dataclasses import replace

from hyper_parallel.distributed_data import BackboneFlopsConfig, DefaultCostModel, SampleMetadata, WorkloadCost


def _backbone_config() -> BackboneFlopsConfig:
    """Return a small one-layer dense backbone with explicit dimensions."""
    return BackboneFlopsConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        mlp_layer_types=("dense",), kv_lora_rank=4, q_lora_rank=None,
        qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=4,
    )


class TestCostModel(unittest.TestCase):
    """Verify metadata passthrough, backbone estimates, and dimension validation."""

    def test_default_preserves_explicit_metadata_cost(self) -> None:
        """Without a model configuration, the caller's cost components remain intact."""
        cost = WorkloadCost(io=1, encoder=2, llm=3)
        metadata = SampleMetadata(pack_tokens=4, cost=cost)

        self.assertEqual(DefaultCostModel()(metadata), cost)

    def test_backbone_estimate_preserves_token_footprint(self) -> None:
        """Forward projections, MLP and attention costs use independent sample dimensions."""
        model = DefaultCostModel(_backbone_config())
        metadata = SampleMetadata(pack_tokens=4, features={"P": 2, "D": 2})

        self.assertEqual(model.forward_flops(metadata), 5184.0)
        self.assertEqual(model(metadata), WorkloadCost(llm=15552.0))
        self.assertEqual(metadata.pack_tokens, 4)

    def test_backbone_rejects_boolean_noninteger_and_nonpositive_dimensions(self) -> None:
        """Python booleans must not be accepted as integer model dimensions."""
        config = _backbone_config()
        for value in (True, False, 8.0, 0, -1):
            with self.subTest(hidden_size=value), self.assertRaisesRegex(ValueError, "hidden_size"):
                replace(config, hidden_size=value)
