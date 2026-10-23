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
"""Stateful chunk KDA compared with an independent token recurrence on CPU."""
from types import SimpleNamespace

import pytest
import torch

from hyper_parallel.custom_ops.experimental import chunk_kda_state_summary, chunk_kda_with_state
from hyper_parallel.core.context_parallel import RecurrentContextParallel


def recurrence(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               gate: torch.Tensor, beta: torch.Tensor, state: torch.Tensor) -> tuple:
    """Compute the original token-level delta rule without WY or affine summaries."""
    outputs = []
    for token in range(query.shape[1]):
        state = state * gate[:, token].exp().unsqueeze(-1)
        delta = beta[:, token].unsqueeze(-1) * (
            value[:, token] - (key[:, token].unsqueeze(-2) @ state).squeeze(-2))
        state = state + key[:, token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((query[:, token].unsqueeze(-2) @ state).squeeze(-2) * query.shape[-1] ** -.5)
    return torch.stack(outputs, dim=1), state


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 64])
@pytest.mark.parametrize("recompute", [False, True])
def test_chunk_outputs_state_and_all_gradients(chunk_size, recompute):
    """Include nonzero initial state, final-state loss, unequal Dk/Dv and tail chunks."""
    torch.manual_seed(203)
    query, key = (torch.nn.functional.normalize(torch.randn(2, 19, 2, 3), dim=-1)
                  for _ in range(2))
    value = torch.randn(2, 19, 2, 5) * .1
    gate = -torch.rand_like(key) * .2
    beta = torch.rand(2, 19, 2) * .5
    state = torch.randn(2, 2, 3, 5) * .1
    args = tuple(x.requires_grad_() for x in (query, key, value, gate, beta, state))
    actual, final = chunk_kda_with_state(*args, scale=3 ** -.5, chunk_size=chunk_size, recompute=recompute)
    expected, ref_state = recurrence(*args)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(final, ref_state, atol=2e-6, rtol=2e-4)
    weight, state_weight = torch.randn_like(actual), torch.randn_like(final)
    grads = torch.autograd.grad((actual * weight).sum() + (final * state_weight).sum(), args, retain_graph=True)
    refs = torch.autograd.grad((expected * weight).sum() + (ref_state * state_weight).sum(), args, retain_graph=True)
    for grad, reference in zip(grads, refs):
        torch.testing.assert_close(grad, reference, atol=3e-6, rtol=3e-4)
    trans, bias = chunk_kda_state_summary(key, value, gate, beta, chunk_size, recompute)
    torch.testing.assert_close(trans @ state + bias, ref_state, atol=2e-6, rtol=2e-4)
    summary_grads = torch.autograd.grad(((trans @ state + bias) * state_weight).sum(), args[1:])
    state_grads = torch.autograd.grad((ref_state * state_weight).sum(), args[1:])
    for grad, reference in zip(summary_grads, state_grads):
        torch.testing.assert_close(grad, reference, atol=3e-6, rtol=3e-4)


def test_noncausal_exponentials_do_not_overflow():
    """Large negative gates never evaluate exp(-cumulative_gate)."""
    torch.manual_seed(17)
    key = torch.nn.functional.normalize(torch.randn(1, 19, 1, 3), dim=-1)
    value, gate, beta = torch.randn(1, 19, 1, 5), torch.full_like(key, -5), torch.ones(1, 19, 1) * .5
    output, state = chunk_kda_with_state(key, key, value, gate, beta, None, 3 ** -.5, 64)
    expected, ref_state = recurrence(key, key, value, gate, beta, torch.zeros(1, 1, 3, 5))
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(state, ref_state, atol=2e-6, rtol=2e-4)


class _AffineBoundary(torch.nn.Module):
    """A non-attention affine module exercising the generic CP contract."""

    def state_summary(self, value: torch.Tensor) -> tuple:
        """Describe a square-to-rectangular state update."""
        return torch.eye(2), value

    def forward(self, value: torch.Tensor, initial_state: torch.Tensor = None) -> torch.Tensor:
        """Consume the supplied state without changing the module's parameters."""
        return value + initial_state


def test_affine_boundary_cp1_and_validation():
    """CP1 is differentiable; malformed boundaries fail before communication."""
    mesh = SimpleNamespace(ndim=1, rank_list=(0,), size=lambda: 1, get_local_rank=lambda: 0)
    module = _AffineBoundary()
    assert RecurrentContextParallel().apply(module, mesh) is module
    value = torch.randn(2, 3, requires_grad=True)
    torch.testing.assert_close(module(value), value)
    torch.testing.assert_close(module(value=value, initial_state=None), value)
    torch.testing.assert_close(torch.autograd.grad(module(value).sum(), value)[0], torch.ones_like(value))
    with pytest.raises(ValueError, match="explicit initial"):
        module(value, initial_state=torch.zeros_like(value))
    with pytest.raises(ValueError, match="summary"):
        RecurrentContextParallel().apply(torch.nn.Identity(), mesh)
    with pytest.raises(ValueError, match="names"):
        RecurrentContextParallel(summary_method=1)
    invalid = _AffineBoundary()
    invalid.state_summary = lambda x: (torch.zeros(2, 3), x)
    RecurrentContextParallel().apply(invalid, mesh)
    with pytest.raises(ValueError, match="compatible square"):
        invalid(value)
