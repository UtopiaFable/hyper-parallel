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
"""Experimental document-interval KV exchange on raw local tensors.

This module is independent of DTensor replication. A returned bank contains only
its receiver's interval union; callers must use block_views and the matching
compact document ends. It is not a full-sequence replicated tensor.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import logging
from numbers import Integral
import os
from threading import Lock, get_native_id
from time import time_ns
from typing import Any, Sequence

from hyper_parallel.platform import get_platform
from hyper_parallel.core.shard.ops.tnd_kv_plan import plan_tnd_kv

platform = get_platform()
_TRACE_ENABLED = os.environ.get("HP_TND_KV_TRACE", "0") == "1"
_TRACE_START = int(os.environ.get("HP_TND_KV_TRACE_START", "1"))
_TRACE_END = int(os.environ.get("HP_TND_KV_TRACE_END", "0"))
_TRACE_COUNTERS = {}
_TRACE_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TNDKVRoute:
    """One rank's immutable element-independent all-to-all routing plan.

    Routes contain (source local row, destination bank row, row count).
    Self traffic stays local. All communication participants use the same ends
    and pair mapping, so counts do not require an extra request collective.
    """
    rank: int
    full_len: int
    local_rows: int
    pairs: tuple
    blocks: tuple
    intervals: tuple
    routes_out: tuple
    routes_in: tuple
    send_rows: tuple
    recv_rows: tuple
    global_remote_rows: int
    document_ends: tuple = ()

    @property
    def bank_rows(self) -> int:
        """Number of distinct KV rows materialized on this receiver."""
        return sum(stop - start for start, stop in self.intervals)

    @property
    def remote_rows(self) -> int:
        """Rows transmitted by this source, excluding local copies."""
        return sum(self.send_rows)


def _union(intervals: tuple) -> tuple:
    """Merge touching and overlapping natural intervals."""
    result = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return tuple(result)


def _route(source: tuple, target: tuple, width: int) -> tuple:
    """Intersect source ownership with the destination's deduplicated bank."""
    result = []
    for slot, chunk in enumerate(source):
        offset = 0
        for start, stop in target:
            low, high = max(start, chunk * width), min(stop, (chunk + 1) * width)
            if low < high:
                result.append((slot * width + low - chunk * width, offset + low - start, high - low))
            offset += stop - start
    return tuple(result)


def plan_tnd_kv_exchange(ends: Sequence[int], pairs: Sequence[Sequence[int]]) -> tuple:
    """Build routes for a folded TND sequence, including arbitrary dynamic pairs.

    Finite host integer sequences are copied into immutable tuples before
    entering the cache; malformed structures raise ValueError.

    Args:
        ends: Global document cumulative ends, including optional repeated ends.
        pairs: Rank-major natural chunk IDs, exactly two chunks per rank.

    Returns:
        One TNDKVRoute per rank, ready to reuse for every compatible KV tensor.

    Raises:
        ValueError: Ownership or document metadata is invalid.
    """
    if isinstance(ends, (str, bytes)) or isinstance(pairs, (str, bytes)):
        raise ValueError("Document ends and ownership pairs must be integer sequences.")
    try:
        ends = tuple(ends)
        rows = []
        for pair in pairs:
            if isinstance(pair, (str, bytes)):
                raise ValueError("Each ownership pair must be a two-element integer sequence.")
            rows.append(tuple(pair))
        pairs = tuple(rows)
    except TypeError as exc:
        raise ValueError("Document ends and ownership pairs must be integer sequences.") from exc
    values = (*ends, *(chunk for pair in pairs for chunk in pair))
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError("Document ends and chunk IDs must be integers, excluding booleans.")
    return _plan_tnd_kv_exchange(tuple(int(value) for value in ends),
                                 tuple(tuple(int(chunk) for chunk in pair) for pair in pairs))


@lru_cache(maxsize=16)
def _plan_tnd_kv_exchange(ends: tuple, pairs: tuple) -> tuple:
    """Cache normalized, immutable routing geometry only."""
    size = len(pairs)
    if (not size or any(len(pair) != 2 for pair in pairs)
            or any(isinstance(x, bool) or not isinstance(x, int) for pair in pairs for x in pair)
            or sorted(x for pair in pairs for x in pair) != list(range(2 * size))
            or not ends or ends[-1] % (2 * size)):
        raise ValueError("Partial TND KV requires a complete two-chunk ownership partition.")
    full_len = ends[-1]
    width = full_len // (2 * size)
    blocks = tuple(plan_tnd_kv(ends, full_len, tuple((chunk * width, (chunk + 1) * width)
                                                  for chunk in pair)) for pair in pairs)
    unions = tuple(_union(tuple(block.key_interval for block in row)) for row in blocks)
    routes = tuple(tuple(_route(pair, union, width) for union in unions) for pair in pairs)
    global_rows = sum(sum(length for _, _, length in routes[src][dst])
                      for src in range(size) for dst in range(size) if src != dst)
    result = []
    for rank in range(size):
        incoming = tuple(routes[src][rank] for src in range(size))
        outgoing = routes[rank]
        sends = tuple(sum(n for _, _, n in row) if dst != rank else 0 for dst, row in enumerate(outgoing))
        receives = tuple(sum(n for _, _, n in row) if src != rank else 0 for src, row in enumerate(incoming))
        result.append(TNDKVRoute(rank, full_len, 2 * width, pairs, blocks[rank], unions[rank],
                                 outgoing, incoming, sends, receives, global_rows, ends))
    return tuple(result)


def _zeros(ref: Any, rows: int) -> Any:
    """Allocate only a local or compact segment, never a global sequence buffer."""
    return platform.zeros((rows,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)


def _cat(parts: list, ref: Any) -> Any:
    """Concatenate nonempty segments, including zero-traffic ranks."""
    return platform.cat(parts, dim=0) if parts else _zeros(ref, 0)


@lru_cache(maxsize=128)
def _trace_digest(values: tuple) -> str:
    """Fingerprint immutable host geometry without reading any device tensor."""
    return sha256(repr(values).encode("utf-8")).hexdigest()[:32]


def _trace_exchange(plan: TNDKVRoute, direction: str, group: Any, ref: Any,
                    columns: int, packed_rows: int, sends: tuple, receives: tuple) -> Any:
    """Record actual element splits; host return does not imply device completion."""
    key = (str(group), plan.rank)
    with _TRACE_LOCK:
        sequence = _TRACE_COUNTERS.get(key, 0) + 1
        _TRACE_COUNTERS[key] = sequence
    if sequence < _TRACE_START or (_TRACE_END and sequence > _TRACE_END):
        return None
    if len(sends) != len(plan.pairs) or len(receives) != len(plan.pairs) or sum(sends) != packed_rows:
        raise ValueError("Traced interval KV buffer and per-peer row counts disagree.")
    record = {
        "event": "enter", "host_time_ns": time_ns(), "thread": get_native_id(),
        "group": str(group), "sequence": sequence,
        "global_rank": os.environ.get("RANK_ID", os.environ.get("RANK")),
        "rank": plan.rank, "world": len(plan.pairs), "direction": direction,
        "columns": columns, "dtype": str(ref.dtype), "full_rows": plan.full_len,
        "transport": "native_sync",
        "ends_digest": _trace_digest(plan.document_ends), "pairs_digest": _trace_digest(plan.pairs),
        "owned_chunks": plan.pairs[plan.rank], "bank_intervals": plan.intervals,
        "send_counts": [count * columns for count in sends],
        "recv_counts": [count * columns for count in receives],
        "input_elements": packed_rows * columns, "output_elements": sum(receives) * columns,
    }
    _LOGGER.warning("TND_KV_TRACE %s", json.dumps(record, separators=(",", ":")))
    return {name: record[name] for name in ("group", "sequence", "rank", "direction")}


def _exchange(parts: list, ref: Any, sends: tuple, receives: tuple, group: Any,
              plan: TNDKVRoute, direction: str) -> Any:
    """Exchange a 1-D buffer with splits expressed in flattened element counts.

    MindSpore requires 1-D inputs and element counts; Torch accepts N-D inputs
    and counts dim-0 rows. Flattening makes each row one element on both
    backends, so these split counts have the same meaning.
    """
    packed = _cat(parts, ref)
    columns = 1
    for dim in ref.shape[1:]:
        columns *= dim
    trace = (_trace_exchange(plan, direction, group, ref, columns, packed.shape[0], sends, receives)
             if _TRACE_ENABLED else None)
    received = platform.differentiable_all_to_all_single(
        packed.reshape(-1), [n * columns for n in sends], [n * columns for n in receives], group)
    # The native synchronous operator owns the stream wait, with no lazy wrapper.
    # Some backends expose a one-element allocation for an empty receive.
    # Resolve the collective first, then discard that storage-only sentinel.
    received = received.reshape(-1).narrow(0, 0, sum(receives) * columns)
    if trace is not None:
        _LOGGER.warning("TND_KV_TRACE %s", json.dumps(
            {**trace, "event": "host_return", "host_time_ns": time_ns(), "thread": get_native_id()}))
    return received.reshape((sum(receives),) + tuple(ref.shape[1:])).contiguous()


def _sum_segments(segments: list, rows: int, ref: Any, fp32_accumulation: bool = False) -> Any:
    """Sum overlapping contributions in deterministic consumer-rank order."""
    edges = sorted({0, rows, *(edge for start, tensor in segments
                              for edge in (start, start + tensor.shape[0]))})
    parts = []
    for start, stop in zip(edges[:-1], edges[1:]):
        values = [tensor.narrow(0, start - left, stop - start) for left, tensor in segments
                  if left <= start and stop <= left + tensor.shape[0]]
        value = values[0] if values else _zeros(ref, stop - start)
        if fp32_accumulation:
            value = value.float()
        for other in values[1:]:
            value = value + (other.float() if fp32_accumulation else other)
        parts.append(value.to(ref.dtype) if fp32_accumulation else value)
    return _cat(parts, ref)


def _receive_bank(local: Any, plan: TNDKVRoute, group: Any) -> Any:
    """Send requested source fragments and assemble a natural-order compact bank."""
    parts = [local.narrow(0, start, length) for peer, routes in enumerate(plan.routes_out)
             if peer != plan.rank for start, _, length in routes]
    remote = (_exchange(parts, local, plan.send_rows, plan.recv_rows, group, plan, "forward")
              if plan.global_remote_rows else None)
    pieces = []
    offset = 0
    for peer, routes in enumerate(plan.routes_in):
        for source, target, length in routes:
            tensor = local.narrow(0, source, length) if peer == plan.rank else remote.narrow(0, offset, length)
            pieces.append((target, tensor))
            if peer != plan.rank:
                offset += length
    return _cat([tensor for _, tensor in sorted(pieces, key=lambda item: item[0])], local)


def _return_gradients(bank_grad: Any, plan: TNDKVRoute, group: Any, fp32_accumulation: bool = False) -> Any:
    """Reverse the exact routes, then sum all consumers on each source owner."""
    parts = [bank_grad.narrow(0, target, length) for peer, routes in enumerate(plan.routes_in)
             if peer != plan.rank for _, target, length in routes]
    remote = (_exchange(parts, bank_grad, plan.recv_rows, plan.send_rows, group, plan, "backward")
              if plan.global_remote_rows else None)
    pieces = []
    offset = 0
    for peer, routes in enumerate(plan.routes_out):
        for source, target, length in routes:
            tensor = (bank_grad.narrow(0, target, length) if peer == plan.rank
                      else remote.narrow(0, offset, length))
            pieces.append((source, tensor))
            if peer != plan.rank:
                offset += length
    return _sum_segments(pieces, plan.local_rows, bank_grad, fp32_accumulation)


class _PartialKV(platform.Function):
    """Autograd boundary: only immutable routing metadata survives forward."""

    @staticmethod
    def forward(ctx: Any, local: Any, plan: TNDKVRoute, group: Any) -> Any:
        """Materialize the receiver's unique KV rows."""
        ctx.plan, ctx.group = plan, group
        # The outer custom Function owns the first-order adjoint. Do not record
        # an additional autograd graph for its nested collective Function.
        with platform.no_grad():
            return _receive_bank(local, plan, group)

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple:  # pylint: disable=arguments-differ
        """Every participating rank enters the reverse exchange, including zero receivers."""
        with platform.no_grad():
            return _return_gradients(gradient, ctx.plan, ctx.group), None, None


def exchange_tnd_kv(local: Any, plan: TNDKVRoute, group: Any) -> Any:
    """Return a differentiable compact bank from an actual local KV shard.

    All group ranks call this in the same tensor-slot order. The group rank order
    must match plan.pairs; the local token order must match plan.pairs[plan.rank].
    Only first-order differentiation is supported by this transport.
    """
    if local.ndim < 1 or local.shape[0] != plan.local_rows:
        raise ValueError("Local KV rows do not match the partial exchange ownership plan.")
    return _PartialKV.apply(local, plan, group)


def _block_offset(block: Any, plan: TNDKVRoute) -> int:
    """Locate one causal query block or fail before a partial view can escape."""
    start, stop = block.key_interval
    query_start, query_stop = block.query_interval
    if not 0 <= start <= query_start < query_stop == stop <= plan.full_len:
        raise ValueError("Compact KV block has invalid causal query coordinates.")
    offset = 0
    for left, right in plan.intervals:
        if left <= start and stop <= right:
            return offset + start - left
        offset += right - left
    raise ValueError("Compact KV block is not covered by the receiver interval union.")


def block_views(bank: Any, plan: TNDKVRoute) -> tuple:
    """Return compact KV views in Q-block order, sharing overlapping bank rows."""
    if bank.ndim < 1 or bank.shape[0] != plan.bank_rows:
        raise ValueError("Received KV bank does not match the interval union.")
    return tuple(bank.narrow(0, _block_offset(block, plan), block.key_interval[1] - block.key_interval[0])
                 for block in plan.blocks)


def merge_tnd_kv_gradients(gradients: tuple, plan: TNDKVRoute,
                           fp32_accumulation: bool = False) -> Any:
    """Merge explicit block gradients into their compact bank without communication."""
    if not gradients or len(gradients) != len(plan.blocks):
        raise ValueError("One explicit gradient is required per nonempty compact Q block.")
    with platform.no_grad():
        pieces = []
        for gradient, block in zip(gradients, plan.blocks):
            start, stop = block.key_interval
            if gradient.ndim < 1 or gradient.shape[0] != stop - start:
                raise ValueError("KL gradient does not match its compact KV block.")
            pieces.append((_block_offset(block, plan), gradient))
        return _sum_segments(pieces, plan.bank_rows, gradients[0], fp32_accumulation)


def reduce_tnd_kv_gradients(gradients: tuple, plan: TNDKVRoute, group: Any,
                            fp32_accumulation: bool = False) -> Any:
    """Return first-order KL key gradients to owners without recording another graph.

    Already-computed gradients are final first-order values; no higher-order
    graph is recorded. Automatic first-order consumers use exchange_tnd_kv.
    """
    with platform.no_grad():
        bank_grad = merge_tnd_kv_gradients(gradients, plan, fp32_accumulation)
        return _return_gradients(bank_grad, plan, group, fp32_accumulation)


def reduce_tnd_kv_bank_gradients(gradient: Any, plan: TNDKVRoute, group: Any,
                                fp32_accumulation: bool = False) -> Any:
    """Return a merged bank gradient before exposing dependent parameter gradients.

    Custom backward functions call this after applying their upstream scalar,
    without adding a separate autograd node for the reverse collective.
    """
    if gradient.ndim < 1 or gradient.shape[0] != plan.bank_rows:
        raise ValueError("Explicit KV bank gradient does not match the receiver interval union.")
    with platform.no_grad():
        return _return_gradients(gradient, plan, group, fp32_accumulation)


class _CachedBankBridge(platform.Function):
    """Share forward storage while keeping each consumer's reverse independent."""

    @staticmethod
    def forward(ctx: Any, local: Any, bank: Any, plan: TNDKVRoute, group: Any) -> Any:
        """Retain no cache or activation references on the gradient bridge."""
        del local
        ctx.plan, ctx.group = plan, group
        return bank

    @staticmethod
    def backward(ctx: Any, gradient: Any) -> tuple:  # pylint: disable=arguments-differ
        """Accumulate source-local contributions in FP32, retaining BF16 wire data."""
        with platform.no_grad():
            return _return_gradients(gradient, ctx.plan, ctx.group, True), None, None, None


def bridge_tnd_kv_bank(local: Any, bank: Any, plan: TNDKVRoute, group: Any) -> Any:
    """Attach an independent first-order reverse to an already exchanged, detached bank."""
    if local.shape[0] != plan.local_rows or bank.shape[0] != plan.bank_rows:
        raise ValueError("Shared interval KV bank does not match its source and receiver shapes.")
    return _CachedBankBridge.apply(local, bank, plan, group)
