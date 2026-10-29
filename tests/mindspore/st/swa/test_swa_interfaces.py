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
"""TND window outputs/gradients against independent document-local math.

Run this file directly with pytest on one NPU. No model or training launcher
is involved. Documents deliberately have unequal lengths and Q/K and V have
different head dimensions, matching expanded MLA rather than assuming GQA.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, ops

from hyper_parallel.custom_ops.experimental import npu_flash_attention_varlen_v4
from tests.common.mark_utils import arg_mark


def _inputs(total_tokens=51):
    rng = np.random.default_rng(772)
    return tuple(ms.Tensor((rng.normal(size=shape) * scale).astype(np.float16))
                 for shape, scale in (((total_tokens, 4, 32), .2), ((total_tokens, 4, 32), .2),
                                      ((total_tokens, 4, 16), 1.)))


def _reference(query, key, value, ends, left, right):
    outputs = []
    start = 0
    for end in ends:
        q = mint.permute(query[start:end].astype(ms.float32), (1, 0, 2))
        k = mint.permute(key[start:end].astype(ms.float32), (1, 2, 0))
        v = mint.permute(value[start:end].astype(ms.float32), (1, 0, 2))
        positions = np.arange(end - start)
        delta = positions[:, None] - positions[None, :]
        allowed = ms.Tensor((delta <= left) & (delta >= -right))
        scores = mint.matmul(q, k) * (32 ** -.5)
        scores = mint.where(allowed, scores, mint.full_like(scores, -1e9))
        outputs.append(mint.permute(mint.matmul(mint.softmax(scores, -1), v), (1, 0, 2)))
        start = end
    return mint.cat(outputs, dim=0)


def _attention(query, key, value, ends, left, right, backend):
    mask = ms.Tensor(np.triu(np.ones((2048, 2048), np.uint8), k=1))
    if backend == "v4":
        return npu_flash_attention_varlen_v4(
            query, key, value, atten_mask=mask, actual_seq_qlen=ends, actual_seq_kvlen=ends,
            head_num=4, scale_value=32 ** -.5, sparse_mode=4, pre_tokens=left, next_tokens=right)[2]
    return ops.flash_attention_score(
        query, key, value, 4, attn_mask=mask, actual_seq_qlen=ends, actual_seq_kvlen=ends,
        scalar_value=32 ** -.5, input_layout="TND", sparse_mode=4,
        pre_tokens=left, next_tokens=right)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
@pytest.mark.parametrize("backend", ["native", "v4"])
@pytest.mark.parametrize("dtype", [ms.float16, ms.bfloat16])
@pytest.mark.parametrize("ends,left,right", [((17, 26, 51), 1, 0), ((17, 26, 51), 4, 0),
                                           ((521, 530, 1081), 511, 0), ((17, 26, 51), 3, 2)])
def test_tnd_document_window_forward_backward(backend, dtype, ends, left, right):
    """Verify packed-document boundaries, window endpoints and Q/K/V gradients."""
    ms.set_context(mode=ms.PYNATIVE_MODE)
    query, key, value = tuple(tensor.astype(dtype) for tensor in _inputs(ends[-1]))
    sensitivity = ms.Tensor(np.random.default_rng(91).normal(size=(ends[-1], 4, 16)).astype(np.float32))
    actual_fn = lambda q, k, v: _attention(q, k, v, ends, left, right, backend)
    reference_fn = lambda q, k, v: _reference(q, k, v, ends, left, right)
    actual = actual_fn(query, key, value).astype(ms.float32).asnumpy()
    expected = reference_fn(query, key, value).asnumpy()
    atol = 2e-2 if dtype == ms.bfloat16 else 2e-3
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=3e-3)
    actual_grads = ms.grad(lambda q, k, v: (actual_fn(q, k, v).astype(ms.float32) * sensitivity).sum(),
                           grad_position=(0, 1, 2))(query, key, value)
    reference_grads = ms.grad(lambda q, k, v: (reference_fn(q, k, v) * sensitivity).sum(),
                              grad_position=(0, 1, 2))(query, key, value)
    for actual_grad, expected_grad in zip(actual_grads, reference_grads):
        grad_atol = 4e-2 if dtype == ms.bfloat16 else 4e-3
        np.testing.assert_allclose(actual_grad.astype(ms.float32).asnumpy(),
                                   expected_grad.astype(ms.float32).asnumpy(), atol=grad_atol, rtol=2e-2)
    # Altering another document must never affect the first document, even for a huge window.
    first_end = ends[0]
    changed_value = mint.cat((value[:first_end], value[first_end:] + 50), dim=0)
    unchanged = actual_fn(query, key, changed_value).astype(ms.float32).asnumpy()[:first_end]
    np.testing.assert_array_equal(unchanged, actual[:first_end])
