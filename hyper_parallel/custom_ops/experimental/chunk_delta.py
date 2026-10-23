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
"""Differentiable chunk delta-rule math with explicit affine state boundaries.

This path uses FP32 matmuls and recomputes each local chunk in backward. It does
not call the fused zero-state KDA backward, whose state gradients are unsupported.
Only local sequences and fixed-size state matrices are materialized.
"""
from functools import partial
from typing import Any, Optional

from hyper_parallel.platform import get_platform

platform = get_platform()


def _identity(reference: Any, tensor_ops: Any) -> Any:
    """Construct an identity on the input device without a host transfer."""
    positions = tensor_ops.cumsum(tensor_ops.ones_like(reference), dim=0)
    return (positions[:, None] == positions[None, :]).float()


def _inverse_unit_lower(lower: Any, tensor_ops: Any) -> Any:
    """Invert I+L using the finite nilpotent product in logarithmic depth."""
    identity = _identity(lower[0, 0, 0], tensor_ops)
    inverse = identity + tensor_ops.zeros_like(lower)
    power, degree = -lower, 1
    while degree < lower.shape[-1]:
        inverse = tensor_ops.matmul(inverse, identity + power)
        degree *= 2
        if degree < lower.shape[-1]:
            power = tensor_ops.matmul(power, power)
    return inverse


def _chunk_factors(key: Any, value: Any, gate: Any, beta: Any, tensor_ops: Any) -> tuple:
    """Build stable WY factors and the affine map for one local chunk."""
    key, value, gate = (x.swapaxes(1, 2).float() for x in (key, value, gate))
    beta = beta.swapaxes(1, 2).float().unsqueeze(-1)
    cumulative = tensor_ops.cumsum(gate, dim=2)
    positions = tensor_ops.cumsum(tensor_ops.ones_like(key[0, 0, :, 0]), dim=0)
    causal = positions[:, None] >= positions[None, :]
    strict = positions[:, None] > positions[None, :]
    difference = cumulative.unsqueeze(3) - cumulative.unsqueeze(2)
    decay = tensor_ops.exp(tensor_ops.where(causal[None, None, :, :, None],
                                            difference, tensor_ops.zeros_like(difference)))
    lower = tensor_ops.sum(key.unsqueeze(3) * key.unsqueeze(2) * decay, dim=-1) * beta * strict
    inverse = _inverse_unit_lower(lower, tensor_ops)
    exp_gate = tensor_ops.exp(cumulative)
    weight = tensor_ops.matmul(inverse, beta * key * exp_gate)
    update = tensor_ops.matmul(inverse, beta * value)
    tail_key = key * tensor_ops.exp(cumulative[:, :, -1:, :] - cumulative)
    identity = _identity(key[0, 0, 0], tensor_ops)
    transition = exp_gate[:, :, -1, :, None] * identity - tensor_ops.matmul(tail_key.swapaxes(-1, -2), weight)
    bias = tensor_ops.matmul(tail_key.swapaxes(-1, -2), update)
    return transition, bias, weight, update, exp_gate, key, decay, causal


def _summary_step(tensor_ops: Any, key: Any, value: Any, gate: Any, beta: Any,
                  transition: Any, bias: Any) -> tuple:
    """Compose this chunk after the accumulated affine map."""
    local_a, local_b = _chunk_factors(key, value, gate, beta, tensor_ops)[:2]
    return tensor_ops.matmul(local_a, transition), tensor_ops.matmul(local_a, bias) + local_b


def _output_step(tensor_ops: Any, scale: float, query: Any, key: Any, value: Any,
                 gate: Any, beta: Any, state: Any) -> tuple:
    """Compute output and final state for a chunk with an arbitrary input state."""
    trans, bias, weight, update, exp_gate, key, decay, causal = _chunk_factors(
        key, value, gate, beta, tensor_ops)
    query = query.swapaxes(1, 2).float()
    aqk = tensor_ops.sum(query.unsqueeze(3) * key.unsqueeze(2) * decay, dim=-1) * causal
    new_value = update - tensor_ops.matmul(weight, state)
    output = (tensor_ops.matmul(query * exp_gate, state) + tensor_ops.matmul(aqk, new_value)) * scale
    return output.swapaxes(1, 2), tensor_ops.matmul(trans, state) + bias


def _validate(key: Any, value: Any, gate: Any, beta: Any, chunk_size: int) -> None:
    """Validate the dense BSHD input boundary, including the tail-chunk contract."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if len(key.shape) != 4 or min(key.shape) < 1:
        raise ValueError("Chunk delta rule requires nonempty dense [B,S,H,D] keys.")
    if tuple(gate.shape) != tuple(key.shape) or tuple(beta.shape) != tuple(key.shape[:3]):
        raise ValueError("Decay must match keys and beta must have shape [B,S,H].")
    if len(value.shape) != 4 or tuple(value.shape[:3]) != tuple(key.shape[:3]):
        raise ValueError("Values must match key batch, sequence and head dimensions.")


def chunk_delta_rule(tensor_ops: Any, query: Optional[Any], key: Any, value: Any,
                     gate: Any, beta: Any, initial_state: Optional[Any], scale: float,
                     chunk_size: int, summary_only: bool, recompute: bool) -> tuple:
    """Run local stateful KDA or return its input-independent affine state map.

    Args:
        tensor_ops: Backend tensor math operations (Torch or MindSpore mint).
        query: BSHD queries; unused when computing only a state summary.
        key: BSHD keys, normalized by the caller.
        value: BSHD values.
        gate: BSHD channel-wise natural-log decays, expected nonpositive.
        beta: BSH update gates.
        initial_state: Optional FP32 [B,H,Dk,Dv] state.
        scale: Query scaling factor.
        chunk_size: Local chunk length; incomplete tail chunks are supported.
        summary_only: Return (A,B) describing S_out=A@S_in+B.
        recompute: Recompute each chunk's intermediates in backward.

    Returns:
        Affine (A,B) summary or (BSHD output, final state), with FP32 states.
    """
    _validate(key, value, gate, beta, chunk_size)
    key_zero, value_zero = key[:, 0].float(), value[:, 0].float()
    zero_state = key_zero.unsqueeze(-1) * value_zero.unsqueeze(-2) * 0
    if summary_only:
        transition = _identity(key_zero[0, 0], tensor_ops) + key_zero.unsqueeze(-1) * 0
        state = zero_state
        step = partial(_summary_step, tensor_ops)
    else:
        if query is None or tuple(query.shape) != tuple(key.shape):
            raise ValueError("Queries must match keys for stateful KDA output.")
        state = zero_state if initial_state is None else initial_state.float()
        if tuple(state.shape) != tuple(zero_state.shape):
            raise ValueError("initial_state must have shape [B,H,Dk,Dv].")
        transition = None
        step = partial(_output_step, tensor_ops, scale)
    outputs = []
    for start in range(0, key.shape[1], chunk_size):
        stop = min(start + chunk_size, key.shape[1])
        args = (key[:, start:stop], value[:, start:stop], gate[:, start:stop], beta[:, start:stop])
        args = args + (transition, state) if summary_only else (query[:, start:stop],) + args + (state,)
        result = platform.checkpoint(step, *args, use_reentrant=False) if recompute else step(*args)
        if summary_only:
            transition, state = result
        else:
            output, state = result
            outputs.append(output)
    return (transition, state) if summary_only else (tensor_ops.cat(outputs, dim=1).to(value.dtype), state)
