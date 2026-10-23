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
"""Run with torchrun: affine sequence CP with more ranks than heads, including CP3."""
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel import DeviceMesh
from hyper_parallel.core.context_parallel import RecurrentContextParallel
from hyper_parallel.core.context_parallel import context_parallel as cp_module
from hyper_parallel.custom_ops.experimental import chunk_kda_state_summary, chunk_kda_with_state


class DeltaCore(torch.nn.Module):
    """Expose the generic affine recurrent module contract."""

    def state_summary(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      gate: torch.Tensor, beta: torch.Tensor) -> tuple:
        """Return this sequence's state map without assuming an incoming state."""
        del query
        return chunk_kda_state_summary(key, value, gate, beta, chunk_size=4)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                gate: torch.Tensor, beta: torch.Tensor, initial_state: torch.Tensor = None) -> torch.Tensor:
        """Run the original local sequence with the CP-provided boundary state."""
        return chunk_kda_with_state(query, key, value, gate, beta, initial_state, 3 ** -.5, 4)[0]


def recurrence(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               gate: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Independent serial recurrence for the full global sequence."""
    state = query.new_zeros((2, 1, 3, 5))
    outputs = []
    for token in range(query.shape[1]):
        state = state * gate[:, token].exp().unsqueeze(-1)
        delta = beta[:, token].unsqueeze(-1) * (
            value[:, token] - (key[:, token].unsqueeze(-2) @ state).squeeze(-2))
        state = state + key[:, token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((query[:, token].unsqueeze(-2) @ state).squeeze(-2) * 3 ** -.5)
    return torch.stack(outputs, dim=1)


def test_recurrent_context_parallel() -> None:
    """Check outputs, all five input gradients and fixed-size communication buffers."""
    dist.init_process_group("gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh("cpu", torch.arange(size), mesh_dim_names=("cp",))
    torch.manual_seed(712)
    qk_shape, v_shape = (2, size * 5, 1, 3), (2, size * 5, 1, 5)
    full = tuple(x.requires_grad_() for x in (
        torch.randn(qk_shape) * .2, torch.randn(qk_shape) * .2,
        torch.randn(v_shape) * .2, -torch.rand(qk_shape) * .1,
        torch.rand(qk_shape[:-1]) * .7))
    core = RecurrentContextParallel().apply(DeltaCore(), mesh)
    for only_last_loss in (False, True):
        local = tuple(x[:, rank * 5:(rank + 1) * 5].detach().requires_grad_() for x in full)
        weights = torch.randn(v_shape)
        # Generate the same global loss on every process.
        dist.broadcast(weights, src=0)
        if only_last_loss:
            weights[:, :(size - 1) * 5] = 0
        buffers = []
        # Observe the instance used by the boundary, even if a pytest fixture
        # has refreshed the global platform factory after module collection.
        platform = cp_module.platform
        exchange = platform.differentiable_all_to_all_single

        def counted(tensor: torch.Tensor, *args: Any) -> torch.Tensor:
            """Record that each collective exchanges only state-sized buffers."""
            buffers.append(tensor.numel())
            return exchange(tensor, *args)

        platform.differentiable_all_to_all_single = counted
        actual = core(*local)
        platform.differentiable_all_to_all_single = exchange
        expected = recurrence(*full)
        torch.testing.assert_close(actual, expected[:, rank * 5:(rank + 1) * 5], atol=2e-6, rtol=2e-4)
        grads = torch.autograd.grad((actual * weights[:, rank * 5:(rank + 1) * 5]).sum(), local)
        refs = torch.autograd.grad((expected * weights).sum(), full)
        for grad, ref in zip(grads, refs):
            torch.testing.assert_close(grad, ref[:, rank * 5:(rank + 1) * 5], atol=3e-6, rtol=3e-4)
        assert len(buffers) == math.ceil(math.log2(size)) + 1
        assert max(buffers) <= 2 * 1 * 3 * (3 + 5)
    report_dir = Path(__file__).resolve().parents[3] / "output"
    report_dir.mkdir(exist_ok=True)
    report = {"world": size, "heads": 1, "pass": True,
              "rounds": len(buffers), "max_buffer_elements": max(buffers)}
    (report_dir / f"kda_scan_torch_cp{size}_rank{rank}.json").write_text(
        json.dumps(report), encoding="utf-8")
    print("RECURRENT CP PASS", size, rank, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    test_recurrent_context_parallel()
