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
"""Attention-scope storage reuse, nested replay and shared derivative lifetimes."""
from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from tests.common.mark_utils import arg_mark
from hyper_parallel.core.context_parallel import dsa_interval_runtime as runtime_module
from hyper_parallel.core.context_parallel.dsa_interval_runtime import DSAIntervalRuntime


def mesh() -> SimpleNamespace:
    """One logical rank with complete local documents; no distributed setup needed."""
    return SimpleNamespace(get_group=lambda: None, get_local_rank=lambda: 0, size=lambda: 1)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              _indices: object, _scale: float, **kwargs: object) -> tuple:
    """Position-local reference exercises exact key/value and rotary derivative routes."""
    rope = kwargs['query_rope'].sum(-1, keepdim=True) + kwargs['key_rope'].sum(-1, keepdim=True)
    output = query + key + value + rope
    stats = output.sum(-1).unsqueeze(0)
    return output, stats, stats


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scopes_reuse_storage_without_reusing_another_layer_or_gradient() -> None:
    """A borrowed callback sees the active layer; backward does not need its expired cache."""
    runtime = DSAIntervalRuntime(mesh(), share_key_value=True)
    meta = SimpleNamespace(_hp_compact_ends=(4, 8), _hp_fold_pairs=None)
    query = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3).requires_grad_()
    key = torch.ones(8, 1, 3, requires_grad=True)
    second_key = torch.full((8, 1, 3), 7., requires_grad=True)
    qr = torch.ones(8, 2, 2, requires_grad=True)
    kr = torch.ones(8, 1, 2, requires_grad=True)
    indices = torch.zeros(8, 1, 1, dtype=torch.int32)
    with patch.object(runtime_module, 'exchange_tnd_kv', wraps=runtime_module.exchange_tnd_kv) as send:
        with runtime.scope():
            outer = runtime.attention(attention, query, key, key, indices, 1., qr, kr, meta, meta, 2)[0]
            with runtime.scope():
                inner = runtime.attention(attention, query, second_key, second_key, indices,
                                          1., qr, kr, meta, meta, 2)[0]
            restored = runtime.attention(attention, query, key, key, indices, 1., qr, kr, meta, meta, 2)[0]
        assert send.call_count == 4  # main KV and rotary once per independent scope
    torch.testing.assert_close(restored, outer)
    assert not torch.equal(inner, outer)
    (outer.sum() + inner.sum() + restored.sum()).backward()
    torch.testing.assert_close(query.grad, torch.full_like(query, 3.))
    torch.testing.assert_close(key.grad, torch.full_like(key, 8.))
    torch.testing.assert_close(second_key.grad, torch.full_like(second_key, 4.))
    torch.testing.assert_close(qr.grad, torch.full_like(qr, 9.))
    torch.testing.assert_close(kr.grad, torch.full_like(kr, 18.))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scope_restores_state_after_an_exception() -> None:
    """A failed attention invocation cannot lend its cached values to a later call."""
    runtime = DSAIntervalRuntime(mesh(), share_key_value=True)
    meta = SimpleNamespace(_hp_compact_ends=(4, 8), _hp_fold_pairs=None)
    q = torch.ones(8, 2, 3)
    k = torch.ones(8, 1, 3)
    indices = torch.zeros(8, 1, 1, dtype=torch.int32)
    qr, kr = torch.ones(8, 2, 2), torch.ones(8, 1, 2)
    with pytest.raises(RuntimeError, match='deliberate'):
        with runtime.scope():
            runtime.attention(attention, q, k, k, indices, 1., qr, kr, meta, meta, 2)
            raise RuntimeError('deliberate failure')
    with pytest.raises(RuntimeError, match='outside'):
        runtime.attention(attention, q, k, k, indices, 1., qr, kr, meta, meta, 2)
    with runtime.scope():
        actual = runtime.attention(attention, q, 3*k, 3*k, indices, 1., qr, kr, meta, meta, 2)[0]
    torch.testing.assert_close(actual, torch.full_like(actual, 11.))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_indexer_derivatives_receive_the_upstream_loss_scale() -> None:
    """Precomputed KL derivatives remain connected through the explicit bank input."""
    runtime = DSAIntervalRuntime(mesh(), share_key_value=True)
    meta = SimpleNamespace(_hp_compact_ends=(4, 8), _hp_fold_pairs=None)
    query = torch.ones(8, 2, 3)
    key = torch.ones(8, 1, 3)
    qi = torch.ones_like(query, requires_grad=True)
    ki = torch.ones_like(key, requires_grad=True)
    weights = torch.ones(8, 2, requires_grad=True)
    indices = torch.zeros(8, 1, 1, dtype=torch.int32)
    stats = torch.ones(2, 8, 1)
    qr, kr = torch.ones(8, 2, 2), torch.ones(8, 1, 2)

    def derivatives(*args: object) -> tuple:
        """Return independently known projection gradients for each compact block."""
        # The KL backend takes host IntArray attributes. Reusing the selector's
        # device metadata here would make MindSpore call Tensor.asnumpy().
        assert isinstance(args[11], tuple) and args[11] == (4,)
        assert isinstance(args[12], tuple) and args[12] == (4,)
        return (torch.ones_like(args[2]), 2 * torch.ones_like(args[3]),
                3 * torch.ones_like(args[4]), torch.tensor([2.]))

    with patch.object(runtime_module.platform.custom_ops,
                      'npu_sparse_lightning_indexer_grad_kl_loss', side_effect=derivatives):
        with runtime.scope():
            loss = runtime.indexer_loss(query, key, qi, ki, weights, indices, stats, stats,
                                        1., qr, kr, meta, meta)
        (7 * loss).backward()
    torch.testing.assert_close(loss, torch.tensor(4.))
    torch.testing.assert_close(qi.grad, torch.full_like(qi, 7.))
    torch.testing.assert_close(ki.grad, torch.full_like(ki, 14.))
    torch.testing.assert_close(weights.grad, torch.full_like(weights, 21.))
