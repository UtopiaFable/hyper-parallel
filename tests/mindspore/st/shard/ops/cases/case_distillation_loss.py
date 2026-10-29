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
"""DTensor MSE, KL and LogSoftmax against standalone MindSpore primitives."""
import mindspore as ms
from mindspore import mint

from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.custom_ops.experimental import npu_flash_attention_varlen_v4
from tests.shard_ops.framework import CompareSpec, InputSpec, OpShardCase, register


def _mse(input_tensor, target, reduction):
    return mint.nn.functional.mse_loss(input_tensor, target, reduction=reduction)


def _kl(input_tensor, target, reduction):
    return mint.nn.functional.kl_div(input_tensor, target, reduction=reduction, log_target=True)


def _log_softmax(input_tensor):
    return mint.nn.functional.log_softmax(input_tensor, dim=-1, dtype=ms.float32)


for name, function in (("mse", _mse), ("kl", _kl)):
    for loss_reduction in ("none", "sum", "mean", "batchmean") if name == "kl" else ("none", "sum", "mean"):
        for shard_dim in (0, 1):
            register(OpShardCase(
                name=f"distill_{name}_{loss_reduction}_dim{shard_dim}", fn=function,
                inputs=[InputSpec(shape=(8, 16), init="randn", seed=21),
                        InputSpec(shape=(8, 16), init="randn", seed=22)],
                placements=[(Shard(shard_dim),), (Shard(shard_dim),)],
                kwargs={"reduction": loss_reduction}, mesh_shape=(2,), mesh_dim_names=("dp",), tags=("npu_level1",),
                compare=CompareSpec.allclose(atol=2e-5, rtol=2e-5),
            ))

register(OpShardCase(
    name="distill_log_softmax_dtype", fn=_log_softmax,
    inputs=[InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=31)],
    placements=[(Shard(0),)], mesh_shape=(2,), mesh_dim_names=("dp",), tags=("npu_level1",),
    compare=CompareSpec.allclose(atol=2e-5, rtol=2e-5),
))

register(OpShardCase(
    name="distill_log_softmax_vocab", fn=_log_softmax,
    inputs=[InputSpec(shape=(8, 16), dtype="float16", init="randn", seed=32)],
    placements=[(Shard(1),)], mesh_shape=(2,), mesh_dim_names=("tp",), tags=("npu_level1",),
    compare=CompareSpec.allclose(atol=2e-5, rtol=2e-5),
))


def _v4_window(query, key, value):
    mask = mint.triu(mint.ones((2048, 2048), dtype=ms.uint8), diagonal=1)
    return npu_flash_attention_varlen_v4(
        query, key, value, atten_mask=mask, actual_seq_qlen=(17, 26, 51), actual_seq_kvlen=(17, 26, 51),
        head_num=4, scale_value=32 ** -.5, sparse_mode=4, pre_tokens=4, next_tokens=0)


register(OpShardCase(
    name="distill_v4_window_heads", fn=_v4_window,
    inputs=[InputSpec(shape=(51, 4, 32), dtype="float16", init="randn", seed=51),
            InputSpec(shape=(51, 4, 32), dtype="float16", init="randn", seed=52),
            InputSpec(shape=(51, 4, 16), dtype="float16", init="randn", seed=53)],
    placements=[(Shard(1),)] * 3, mesh_shape=(2,), mesh_dim_names=("tp",), tags=("npu_level1",),
    compare=CompareSpec.allclose(atol=3e-3, rtol=3e-3),
))
