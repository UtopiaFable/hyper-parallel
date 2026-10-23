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
"""Backend tensor operations for differentiable chunk delta-rule math."""
from typing import Any


def chunk_kda_stateful(query: Any, key: Any, value: Any, gate: Any, beta: Any,
                       initial_state: Any, scale: float, chunk_size: int = 64,
                       summary_only: bool = False, recompute: bool = True) -> tuple:
    """Run native chunk math using lazily loaded backend tensor operations."""
    from mindspore import mint as tensor_ops  # pylint: disable=C0415
    from hyper_parallel.custom_ops.experimental.chunk_delta import chunk_delta_rule  # pylint: disable=C0415
    return chunk_delta_rule(tensor_ops, query, key, value, gate, beta, initial_state,
                            scale, chunk_size, summary_only, recompute)
