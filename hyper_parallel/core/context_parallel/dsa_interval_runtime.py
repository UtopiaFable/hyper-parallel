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
"""Raw-tensor sparse TND DSA execution over document-interval KV exchange."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.context_parallel.tnd_kv_exchange import (
    plan_tnd_kv_exchange, exchange_tnd_kv, bridge_tnd_kv_bank, block_views, merge_tnd_kv_gradients,
)

platform = get_platform()
_ACTIVE = ContextVar("dsa_interval_attention_call", default=None)
_MAX_TOKENS = 9223372036854775807


def _local(value: Any) -> Any:
    """Preserve the gradient-bearing local tensor without fabricating replication."""
    return value.to_local() if isinstance(value, DTensor) else value


@dataclass
class _Call:
    """One attention invocation; no activation cache survives its scope."""
    runtime: Any
    signature: Any = None
    plan: Any = None
    metadata: Any = None
    banks: dict = field(default_factory=dict)


class _IntervalKL(platform.Function):
    """Route precomputed indexer-only derivatives through local projection inputs."""

    @staticmethod
    def forward(ctx: Any, state: Any, query: Any, key: Any, qi: Any, ki: Any, weights: Any,
                indices: Any, maximum: Any, summation: Any, scale: float, qr: Any, kr: Any) -> Any:
        """Compute compact derivatives; the key bank bridge applies the true loss scaling."""
        with platform.no_grad():
            result = state.runtime._kl(state, query, key, qi, ki, weights, indices, maximum,
                                       summation, scale, qr, kr)
        ctx.save_for_backward(*result[:3])
        return result[3][0]

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple:  # pylint: disable=arguments-differ
        """The teacher and selection remain nondifferentiable, as in the original loss."""
        with platform.no_grad():
            dqi, dki, dw = ctx.saved_tensors
            dqi.mul_(gradient)
            dki.mul_(gradient)
            dw.mul_(gradient)
            return (None, None, None, dqi, dki, dw, None, None, None, None, None, None)


class DSAIntervalRuntime:
    """A model's interval transport, with fresh attention scopes and cached host geometry.

    The caller must install this runtime on the complete self-attention scope,
    including any borrowed IndexShare loss. All tensor slots use the same mesh
    and two-chunk input ownership. Only folded, sparse, causal TND is supported.
    """

    def __init__(self, mesh: Any, share_key_value: bool = False) -> None:
        """Bind the model-level sequence group; activations live only in call scopes."""
        self.group = mesh.get_group()
        self.rank = mesh.get_local_rank()
        self.world = mesh.size()
        self.share_key_value = share_key_value
        self._metadata = OrderedDict()

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Restore the preceding scope on success, exceptions and nested replay."""
        state = _Call(self)
        token = _ACTIVE.set(state)
        try:
            yield
        finally:
            state.banks.clear()
            _ACTIVE.reset(token)

    def _state(self, query: Any, qmeta: Any, kmeta: Any) -> _Call:
        state = _ACTIVE.get()
        if state is None or state.runtime is not self:
            raise RuntimeError("Interval DSA kernel called outside its self-attention scope.")
        ends = getattr(qmeta, "_hp_compact_ends", None)
        pairs = getattr(qmeta, "_hp_fold_pairs", None)
        if ends is None or getattr(kmeta, "_hp_compact_ends", None) != ends:
            raise ValueError("Interval DSA requires matching global compact metadata at model input.")
        if getattr(kmeta, "_hp_fold_pairs", None) != pairs:
            raise ValueError("Interval DSA query and key metadata carry different ownership mappings.")
        if pairs is None:
            pairs = tuple((2*r, 2*self.world-1-2*r) for r in range(self.world))
        signature = (ends, pairs)
        if state.plan is not None:
            if state.signature != signature or query.shape[0] != state.plan.local_rows:
                raise ValueError("Interval DSA geometry changed inside one attention invocation.")
            return state
        if len(pairs) != self.world:
            raise ValueError("Interval DSA ownership does not match its sequence group size.")
        plan = plan_tnd_kv_exchange(ends, pairs)[self.rank]
        if query.shape[0] != plan.local_rows:
            raise ValueError("Interval DSA query rows do not match combined TP/CP ownership.")
        metadata = self._metadata.get(signature)
        if metadata is None:
            metadata = tuple(tuple(platform.from_numpy(np.asarray(values, np.int32))
                                   for values in (block.query_ends, block.key_ends)) for block in plan.blocks)
            self._metadata[signature] = metadata
            if len(self._metadata) > 16:
                self._metadata.popitem(last=False)
        else:
            self._metadata.move_to_end(signature)
        state.signature, state.plan, state.metadata = signature, plan, metadata
        return state

    def _bank(self, state: _Call, slot: str, value: Any, differentiable: bool = False) -> Any:
        signature = (tuple(value.shape), value.dtype)
        saved = state.banks.get(slot)
        if saved is None:
            with platform.no_grad():
                bank = exchange_tnd_kv(value, state.plan, self.group).detach()
            state.banks[slot] = (signature, bank)
        else:
            previous, bank = saved
            if previous != signature:
                raise ValueError(f"Interval DSA reused incompatible tensor slot {slot}.")
        return bridge_tnd_kv_bank(value, bank, state.plan, self.group) if differentiable else bank

    def indexer(self, function: Any, query: Any, key: Any, weights: Any, qmeta: Any, kmeta: Any,
                topk: int, enhance: bool = False) -> tuple:
        """Run the unchanged selector on two compact key prefixes."""
        query, key, weights = map(_local, (query, key, weights))
        state = self._state(query, qmeta, kmeta)
        with platform.no_grad():
            keys = block_views(self._bank(state, "indexer_k", key), state.plan)
            results = []
            width = state.plan.local_rows // 2
            for slot, (cq, ck) in enumerate(state.metadata):
                options = {"actual_seq_lengths_query": cq, "actual_seq_lengths_key": ck,
                           "layout_query": "TND", "layout_key": "TND", "sparse_count": topk,
                           "return_value": not enhance}
                if enhance:
                    options.update(sparse_block_size=1, sparse_block_mode=0)
                results.append(function(query.narrow(0, slot*width, width), keys[slot],
                                        weights.narrow(0, slot*width, width), **options))
            indices = platform.cat([result[0] for result in results], dim=0)
            scores = None if enhance else platform.cat([result[1] for result in results], dim=0)
        return indices, scores

    def attention(self, function: Any, query: Any, key: Any, value: Any, indices: Any, scale: float,
                  qr: Any, kr: Any, qmeta: Any, kmeta: Any, mode: int) -> tuple:
        """Exchange each semantic bank once while preserving independent reverse paths."""
        query, key, value, indices, qr, kr = map(_local, (query, key, value, indices, qr, kr))
        state = self._state(query, qmeta, kmeta)
        keys = block_views(self._bank(state, "main_kv", key, True), state.plan)
        value_slot = "main_kv" if self.share_key_value else "value"
        values = block_views(self._bank(state, value_slot, value, True), state.plan)
        ropes = block_views(self._bank(state, "key_rope", kr, True), state.plan)
        results = []
        width = state.plan.local_rows // 2
        for slot, (cq, ck) in enumerate(state.metadata):
            results.append(function(query.narrow(0, slot*width, width), keys[slot], values[slot],
                indices.narrow(0, slot*width, width), scale, query_rope=qr.narrow(0, slot*width, width),
                key_rope=ropes[slot], actual_seq_lengths_query=cq, actual_seq_lengths_kv=ck,
                layout_query="TND", layout_kv="TND", attention_mode=mode, return_softmax_lse=True))
        return (platform.cat([row[0] for row in results], dim=0),
                platform.cat([row[1] for row in results], dim=1),
                platform.cat([row[2] for row in results], dim=1))

    def indexer_loss(self, query: Any, key: Any, qi: Any, ki: Any, weights: Any, indices: Any,
                     maximum: Any, summation: Any, scale: float, qr: Any, kr: Any,
                     qmeta: Any, kmeta: Any) -> Any:
        """Return the local loss with the same projection-only gradient contract."""
        query, key, qi, ki, weights, indices, maximum, summation, qr, kr = map(
            _local, (query, key, qi, ki, weights, indices, maximum, summation, qr, kr))
        state = self._state(query, qmeta, kmeta)
        # Keep the bank as a real autograd input. Its independent bridge returns
        # gradients only after the receiver's upstream scalar has been applied.
        key_bank = self._bank(state, "indexer_k", ki, True)
        return _IntervalKL.apply(state, query, key, qi, key_bank, weights, indices,
                                 maximum, summation, scale, qr, kr)

    def _kl(self, state: _Call, query: Any, key: Any, qi: Any, ki: Any, weights: Any, indices: Any,
            maximum: Any, summation: Any, scale: float, qr: Any, kr: Any) -> tuple:
        """Compute compact KL derivatives without moving key gradients before loss scaling."""
        keys = block_views(self._bank(state, "main_kv", key), state.plan)
        ropes = block_views(self._bank(state, "key_rope", kr), state.plan)
        index_keys = block_views(ki, state.plan)
        results = []
        width = state.plan.local_rows // 2
        for slot, block in enumerate(state.plan.blocks):
            results.append(platform.custom_ops.npu_sparse_lightning_indexer_grad_kl_loss(
                query.narrow(0, slot*width, width), keys[slot], qi.narrow(0, slot*width, width),
                index_keys[slot], weights.narrow(0, slot*width, width), indices.narrow(0, slot*width, width),
                maximum.narrow(1, slot*width, width), summation.narrow(1, slot*width, width), scale,
                # KL consumes host IntArray attributes, unlike the selector's
                # device length tensors. Keep its immutable host coordinates:
                # a reused Tensor may have moved to Ascend, making the backend
                # convert it with asnumpy() in the middle of distributed work.
                qr.narrow(0, slot*width, width), ropes[slot], block.query_ends, block.key_ends,
                "TND", 3, _MAX_TOKENS, _MAX_TOKENS))
        dki = merge_tnd_kv_gradients(tuple(row[1] for row in results), state.plan, True)
        return (platform.cat([row[0] for row in results], dim=0), dki,
                platform.cat([row[2] for row in results], dim=0), results[0][3] + results[1][3])
