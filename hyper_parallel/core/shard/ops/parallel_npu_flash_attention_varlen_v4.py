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
"""Register the public three-output V4 FA API with the existing TND layout rules."""
from typing import Callable, Tuple

from hyper_parallel.core.shard.ops.parallel_ms_flash_attention_score import FlashAttentionScoreDistributedOp


def _normalize_varlen_v4_args(query, key, value, atten_mask=None, actual_seq_qlen=None,
                             actual_seq_kvlen=None, scale_value=1.0, head_num=1, sparse_mode=3,
                             pre_tokens=9223372036854775807, next_tokens=0, inner_precise=0,
                             tnd_softmax_out=True):
    """Normalize the 13 positional slots used by the V4 DFunction."""
    return (query, key, value, atten_mask, actual_seq_qlen, actual_seq_kvlen,
            scale_value, head_num, sparse_mode, pre_tokens, next_tokens,
            inner_precise, tnd_softmax_out), {}


class NpuFlashAttentionVarLenV4DistributedOp(FlashAttentionScoreDistributedOp):
    """Reuse stock FA partitioning while preserving the V4 three-output contract.

    Distributed statistics require TND order. Band mode works with complete
    documents and head sharding; the inherited sequence-partition validation
    deliberately rejects band CP until per-document query offsets are supported.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Adapt V4 slots to the canonical FlashAttentionScore signature."""
        normalized, _ = _normalize_varlen_v4_args(*args, **kwargs)
        query, key, value, mask, qlen, kvlen, scale, heads, mode, pre, next_, precise, tnd = normalized
        canonical = (query, key, value, None, None, None, mask, None,
                     qlen, kvlen, heads, 1.0, scale, pre, next_, precise, "TND", mode)
        local_args, local_kwargs, cache_values = super().preprocess(canonical, {})
        return local_args, local_kwargs, cache_values + [tnd]

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Drop stock FA's reserved softmax_out slot from the inferred layouts."""
        self._check_partial_inputs(cache_values[:3])
        if cache_values[4] is not True:
            raise ValueError(
                f"For {self.op_name}, distributed statistics should use TND order, but got tnd_softmax_out=False.")
        outputs, _ = super().infer_layout(cache_values[:4])
        return ((outputs[0], outputs[1], outputs[3]), None)

    @staticmethod
    def use_tnd_v4_kernel() -> bool:
        """The local function is already V4; never invoke the public API recursively."""
        return False

    def get_expand_impl(self, func: Callable, infer_result: tuple, cache_values: list) -> Callable:
        """Translate canonical local calls back to V4 and preserve all window inputs."""
        def _four_output_kernel(query, key, value, _shift, _drop, _padding, mask, _prefix,
                                qlen, kvlen, heads, _keep, scale, pre, next_, precise, _layout, mode):
            softmax_max, softmax_sum, attention = func(
                query, key, value, mask, qlen, kvlen, scale, heads, mode, pre, next_, precise, True)
            # The reserved slot is removed before DTensor wrapping; no device allocation is needed.
            return softmax_max, softmax_sum, None, attention

        expanded = super().get_expand_impl(_four_output_kernel, infer_result, cache_values[:4])

        def _three_output_kernel(*args):
            outputs = expanded(*args)
            return outputs[0], outputs[1], outputs[3]

        return _three_output_kernel
