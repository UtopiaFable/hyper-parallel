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
"""Two-NPU gradient worker for distillation primitives and TND SWA head TP."""
import numpy as np
import mindspore as ms
import mindspore.communication.management as communication
from mindspore import mint

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.custom_ops.experimental import npu_flash_attention_varlen_v4
from tests.mindspore.st.swa.test_swa_interfaces import _inputs, _reference


def test_swa_distillation_gradients():
    """Compare vocabulary-sharded KL/MSE and head-sharded FA gradients to full math."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    communication.init()
    rank = communication.get_rank()
    mesh = init_device_mesh("npu", (2,), mesh_dim_names=("tp",))
    rng = np.random.default_rng(318)
    full_input = ms.Tensor(rng.normal(size=(8, 16)).astype(np.float32))
    full_target = ms.Tensor(rng.normal(size=(8, 16)).astype(np.float32))
    vocab_slice = slice(rank * 8, (rank + 1) * 8)
    local_input = ms.Tensor(full_input.asnumpy()[:, vocab_slice])
    local_target = ms.Tensor(full_target.asnumpy()[:, vocab_slice])
    target = DTensor.from_local(local_target, mesh, (Shard(1),))

    def reference_kl(input_tensor):
        return mint.nn.functional.kl_div(
            mint.nn.functional.log_softmax(input_tensor, -1),
            mint.nn.functional.log_softmax(full_target, -1), log_target=True, reduction="none")

    def distributed_kl(input_tensor):
        distributed = DTensor.from_local(input_tensor, mesh, (Shard(1),))
        losses = mint.nn.functional.kl_div(
            mint.nn.functional.log_softmax(distributed, -1),
            mint.nn.functional.log_softmax(target, -1), log_target=True, reduction="none")
        return losses.to_local().sum()

    np.testing.assert_allclose(distributed_kl(local_input).asnumpy(),
                               reference_kl(full_input).asnumpy()[:, vocab_slice].sum(), atol=2e-5, rtol=2e-5)
    actual_grad = ms.grad(distributed_kl)(local_input)
    expected_grad = ms.grad(lambda x: reference_kl(x).sum())(full_input)
    np.testing.assert_allclose(actual_grad.asnumpy(), expected_grad.asnumpy()[:, vocab_slice], atol=2e-5, rtol=2e-5)

    def distributed_mse(input_tensor):
        distributed = DTensor.from_local(input_tensor, mesh, (Shard(1),))
        return mint.nn.functional.mse_loss(distributed, target, reduction="none").to_local().sum()

    mse_grad = ms.grad(distributed_mse)(local_input)
    np.testing.assert_allclose(mse_grad.asnumpy(), 2 * (local_input - local_target).asnumpy(), atol=1e-6)

    query, key, value = _inputs()
    head_slice = slice(rank * 2, (rank + 1) * 2)
    local_qkv = tuple(ms.Tensor(tensor.asnumpy()[:, head_slice]) for tensor in (query, key, value))
    sensitivity = ms.Tensor(rng.normal(size=(51, 4, 16)).astype(np.float32))
    local_sensitivity = ms.Tensor(sensitivity.asnumpy()[:, head_slice])
    mask = mint.triu(mint.ones((2048, 2048), dtype=ms.uint8), diagonal=1)

    def distributed_fa(q, k, v):
        q, k, v = tuple(DTensor.from_local(tensor, mesh, (Shard(1),)) for tensor in (q, k, v))
        output = npu_flash_attention_varlen_v4(
            q, k, v, atten_mask=mask, actual_seq_qlen=(17, 26, 51), actual_seq_kvlen=(17, 26, 51),
            head_num=4, scale_value=32 ** -.5, sparse_mode=4, pre_tokens=4, next_tokens=0)[2]
        return (output.to_local().astype(ms.float32) * local_sensitivity).sum()

    def reference_fa(q, k, v):
        return (_reference(q, k, v, (17, 26, 51), 4, 0) * sensitivity).sum()

    actual_grads = ms.grad(distributed_fa, grad_position=(0, 1, 2))(*local_qkv)
    expected_grads = ms.grad(reference_fa, grad_position=(0, 1, 2))(query, key, value)
    for actual, expected in zip(actual_grads, expected_grads):
        np.testing.assert_allclose(actual.asnumpy(), expected.asnumpy()[:, head_slice], atol=4e-3, rtol=2e-2)
