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
"""Deterministic sample-level balancing with capacity-aware sequence bins."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any

from hyper_parallel.distributed_data.cost_model import CostModel, resolve_cost_model
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    OversizedPolicy,
    PackingBinPlan,
    PackingConstraints,
    SampleKey,
    WorkloadCost,
)


@dataclass
class _MutableBin:
    data_rank: int
    pack_index: int
    samples: list[BufferedSampleMetadata] = field(default_factory=list)
    pack_tokens: int = 0
    packing_costs: dict[str, float] = field(default_factory=dict)
    cost: WorkloadCost = WorkloadCost()
    oversized: bool = False


class DynamicPackingPlanner:
    """Place a frozen step sample set across capacity-constrained DP bins.

    With balancing enabled, longest-processing-time placement minimizes the
    maximum estimated backbone workload, then variance and load range. The
    original rank-local packing is retained when greedy placement fails or the
    complete objective does not improve. Without balancing, metadata costs and
    token-first ordering retain the native packing behavior.
    """

    def __init__(
            self,
            *,
            data_parallel_size: int,
            seq_len: int,
            local_batch_size: int,
            oversized_policy: OversizedPolicy = "error",
            packing_budgets: Mapping[str, float] | None = None,
            enable_balancing: bool = False,
            model_config: Any = None,
            cost_model: CostModel | None = None,
            validate: bool = True,
            min_balance_gain: float = 0.0,
    ) -> None:
        """Initialize fixed constructor dimensions.

        Args:
            data_parallel_size: Number of independent DP Data Constructors.
            seq_len: Capacity of one packed sequence.
            local_batch_size: Packed sequences produced by each constructor.
            oversized_policy: ``error`` or explicit singleton overflow.
            packing_budgets: Additive hard caps on each bin's named
                ``SampleMetadata.packing_costs``; independent of cost estimates.
            enable_balancing: Use cost-first placement with the default or
                explicitly supplied workload model.
            model_config: Backbone dimensions required by the default cost model
                when balancing is enabled and no callback is supplied.
            cost_model: Deterministic CPU callback evaluated on frozen samples.
                Native packing otherwise retains metadata-provided costs.
            validate: Audit metadata types and generated plan membership/order.
                Disable for trusted local packing; placement capacities still apply.
            min_balance_gain: Native packing's minimum relative reduction in the
                maximum dominant rank cost. Ignored when balancing is enabled.
        """
        for name, value in (
                ("data_parallel_size", data_parallel_size),
                ("seq_len", seq_len),
                ("local_batch_size", local_batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, but got {value!r}.")
        if oversized_policy not in ("error", "single"):
            raise ValueError("oversized_policy must be 'error' or 'single'.")
        if not enable_balancing and (
                not isinstance(min_balance_gain, (int, float))
                or isinstance(min_balance_gain, bool)
                or not 0.0 <= min_balance_gain < 1.0
        ):
            raise ValueError("min_balance_gain must be in [0, 1).")
        self.data_parallel_size = data_parallel_size
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.oversized_policy = oversized_policy
        self._validate = validate
        self._constraints = PackingConstraints(seq_len, oversized_policy, packing_budgets)
        self.enable_balancing = enable_balancing
        self.cost_model = resolve_cost_model(cost_model, model_config) if enable_balancing else cost_model
        if self.cost_model is not None and not callable(self.cost_model):
            raise ValueError("cost_model must be callable: SampleMetadata -> WorkloadCost.")
        self.min_balance_gain = 0.0 if enable_balancing else float(min_balance_gain)
        self.last_sample_costs: dict[SampleKey, WorkloadCost] = {}

    @property
    def distributed_bin_count(self) -> int:
        """Return sequence bins required for one distributed yield."""
        return self.data_parallel_size * self.local_batch_size

    @property
    def distributed_token_budget(self) -> int:
        """Return maximum non-oversized tokens in one distributed yield."""
        return self.distributed_bin_count * self.seq_len

    def plan(
            self,
            samples: Sequence[BufferedSampleMetadata],
            *,
            reference_bins: Sequence[Sequence[SampleKey]],
            step: int,
    ) -> DistributedPackingPlan:
        """Balance every selected sample exactly once.

        Args:
            samples: Exactly the sample occurrences selected for this step.
                Metadata input order does not affect placement.
            reference_bins: Known-feasible original grouping of those occurrences,
                ordered by data rank and then local bin index. Used for cost
                comparison and fallback; every sample key must occur once.
            step: Zero-based distributed-yield index.

        Returns:
            Full plan containing exactly the selected sample keys.
        """
        self._validate_plan_request(samples, reference_bins, step)
        samples = self._estimate_samples(samples)
        ordered = self._validate_and_order(samples)
        bins = [
            _MutableBin(data_rank=data_rank, pack_index=pack_index)
            for data_rank in range(self.data_parallel_size)
            for pack_index in range(self.local_batch_size)
        ]
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        bins, rank_costs = self._place_samples(ordered, reference_bins, bins, rank_costs)
        if self.enable_balancing:
            original_bins, reference_costs = self._reference_bins_in_order(samples, reference_bins)
            scale = max((cost.llm for cost in (*reference_costs, *rank_costs)), default=0.0)
            if self._objective_score(rank_costs, scale) >= self._objective_score(reference_costs, scale):
                bins, rank_costs = original_bins, reference_costs
        elif self.min_balance_gain:
            original_bins, reference_costs = self._reference_bins_in_order(samples, reference_bins)
            if self._reference_is_better(reference_costs, rank_costs):
                bins, rank_costs = original_bins, reference_costs
        local_batches = self._freeze_bins(bins)
        if self._validate:
            self._validate_conservation(local_batches, samples)
        plan_id = self._plan_id(step, local_batches)
        return DistributedPackingPlan(
            plan_id=plan_id,
            step=step,
            seq_len=self.seq_len,
            local_batches=local_batches,
            rank_costs=tuple(rank_costs),
            # Placement enforces capacities; conservation was checked above.
            validate=False,
        )

    def _estimate_samples(
            self, samples: Sequence[BufferedSampleMetadata],
    ) -> Sequence[BufferedSampleMetadata]:
        """Attach balancing estimates without mutating the reader's metadata."""
        self.last_sample_costs = {}
        if self.cost_model is None:
            self.last_sample_costs = {item.key: item.metadata.cost for item in samples}
            return samples
        estimated_samples = []
        for item in samples:
            cost = self.cost_model(item.metadata)
            if self._validate and not isinstance(cost, WorkloadCost):
                raise ValueError(f"cost_model must return WorkloadCost for sample {item.key}, but got {type(cost)}.")
            # Physical metadata and membership are unchanged from the input.
            # Only cost changes; do not replay recursive feature/key validation.
            metadata = copy(item.metadata)
            object.__setattr__(metadata, "cost", cost)
            estimated_item = copy(item)
            object.__setattr__(estimated_item, "metadata", metadata)
            estimated_samples.append(estimated_item)
            self.last_sample_costs[item.key] = cost
        return tuple(estimated_samples)

    def _reference_bins_in_order(
            self,
            samples: Sequence[BufferedSampleMetadata],
            reference_bins: Sequence[Sequence[SampleKey]],
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        """Materialize the canonical bins in their original rank order."""
        samples_by_key = {item.key: item for item in samples}
        bins: list[_MutableBin] = []
        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        rank_tokens = [0] * self.data_parallel_size
        for index, key_bin in enumerate(reference_bins):
            data_rank = index // self.local_batch_size
            packing_bin = _MutableBin(data_rank=data_rank, pack_index=index % self.local_batch_size)
            for key in key_bin:
                self._place(packing_bin, samples_by_key[key], rank_costs, rank_tokens)
            bins.append(packing_bin)
        return bins, rank_costs

    def _reference_is_better(
            self, reference_costs: Sequence[WorkloadCost], balanced_costs: Sequence[WorkloadCost],
    ) -> bool:
        reference_max = max((cost.dominant for cost in reference_costs), default=0.0)
        balanced_max = max((cost.dominant for cost in balanced_costs), default=0.0)
        if reference_max <= 0.0:
            return True
        gain = (reference_max - balanced_max) / reference_max
        return gain < self.min_balance_gain

    def _validate_plan_request(
            self,
            samples: Sequence[BufferedSampleMetadata],
            reference_bins: Sequence[Sequence[SampleKey]],
            step: int,
    ) -> None:
        if not samples:
            raise ValueError("Step samples must not be empty.")
        if self._validate and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
            raise ValueError(f"step must be a non-negative integer, but got {step!r}.")
        if len(reference_bins) != self.distributed_bin_count:
            raise ValueError(
                f"Step selection expected {self.distributed_bin_count} reference bins, "
                f"but got {len(reference_bins)}."
            )
        if len(samples) < self.distributed_bin_count:
            raise ValueError(
                f"Step selection has {len(samples)} samples for {self.distributed_bin_count} non-empty bins."
            )
        if self._validate:
            if any(not packing_bin for packing_bin in reference_bins):
                raise ValueError("reference_bins must contain non-empty bins.")
            sample_keys = {item.key for item in samples}
            if len(sample_keys) != len(samples):
                raise ValueError("Step samples must have unique SampleKey values.")
            reference_keys = tuple(key for packing_bin in reference_bins for key in packing_bin)
            if len(reference_keys) != len(samples) or set(reference_keys) != sample_keys:
                raise ValueError("Reference bins must contain every selected sample exactly once.")

    def _place_samples(
            self,
            ordered: Sequence[BufferedSampleMetadata],
            reference_bins: Sequence[Sequence[SampleKey]],
            bins: list[_MutableBin],
            rank_costs: list[WorkloadCost],
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        rank_tokens = [0 for _ in range(self.data_parallel_size)]
        seed_items = ordered[:self.distributed_bin_count]
        remaining_items = ordered[self.distributed_bin_count:]
        for item in seed_items:
            selected = min(
                (packing_bin for packing_bin in bins if not packing_bin.samples),
                key=lambda packing_bin: self._placement_score(
                    packing_bin, item, rank_costs, rank_tokens, seeding=True
                ),
            )
            self._place(selected, item, rank_costs, rank_tokens)

        for item in remaining_items:
            feasible = [packing_bin for packing_bin in bins if self._fits(packing_bin, item)]
            if not feasible:
                if self.enable_balancing:
                    bins, rank_costs = self._reference_bins_in_order(ordered, reference_bins)
                else:
                    bins, rank_costs = self._place_reference_bins(ordered, reference_bins)
                break
            selected = min(
                feasible,
                key=lambda packing_bin: self._placement_score(
                    packing_bin, item, rank_costs, rank_tokens, seeding=False
                ),
            )
            self._place(selected, item, rank_costs, rank_tokens)
        return bins, rank_costs

    def _place_reference_bins(
            self,
            samples: Sequence[BufferedSampleMetadata],
            reference_bins: Sequence[Sequence[SampleKey]],
    ) -> tuple[list[_MutableBin], list[WorkloadCost]]:
        """Balance known-feasible reference packs when sample-level packing fails."""
        samples_by_key = {item.key: item for item in samples}
        scored_bins = []
        for original_index, key_bin in enumerate(reference_bins):
            items = tuple(samples_by_key[key] for key in key_bin)
            cost = sum((item.metadata.cost for item in items), WorkloadCost())
            tokens = sum(item.metadata.pack_tokens for item in items)
            scored_bins.append((original_index, items, cost, tokens))
        scored_bins.sort(key=lambda item: (-item[2].dominant, -item[2].total, -item[3], item[0]))

        rank_costs = [WorkloadCost() for _ in range(self.data_parallel_size)]
        rank_tokens = [0 for _ in range(self.data_parallel_size)]
        rank_bins: list[list[_MutableBin]] = [[] for _ in range(self.data_parallel_size)]
        for _, items, cost, tokens in scored_bins:
            eligible_ranks = [
                data_rank
                for data_rank in range(self.data_parallel_size)
                if len(rank_bins[data_rank]) < self.local_batch_size
            ]
            data_rank = min(
                eligible_ranks,
                key=lambda rank: (
                    (rank_costs[rank] + cost).dominant,
                    (rank_costs[rank] + cost).total,
                    rank_tokens[rank] + min(tokens, self.seq_len),
                    rank,
                ),
            )
            packing_bin = _MutableBin(data_rank=data_rank, pack_index=len(rank_bins[data_rank]))
            for item in items:
                self._place(packing_bin, item, rank_costs, rank_tokens)
            rank_bins[data_rank].append(packing_bin)
        return [packing_bin for bins in rank_bins for packing_bin in bins], rank_costs

    @staticmethod
    def _validate_conservation(
            local_batches: Sequence[Sequence[PackingBinPlan]],
            samples: Sequence[BufferedSampleMetadata],
    ) -> None:
        """Reject any balanced plan that drops or duplicates a selected key."""
        selected_keys = tuple(item.key for item in samples)
        planned_keys = tuple(
            key
            for local_batch in local_batches
            for packing_bin in local_batch
            for key in packing_bin.sample_keys
        )
        planned_set = set(planned_keys)
        selected_set = set(selected_keys)
        if len(planned_keys) != len(planned_set) or planned_set != selected_set:
            missing = sorted(selected_set - planned_set)
            unexpected = sorted(planned_set - selected_set)
            raise ValueError(
                "Balanced placement must conserve the frozen step sample set exactly; "
                f"missing={missing}, unexpected={unexpected}."
            )

    def _validate_and_order(
            self,
            candidates: Sequence[BufferedSampleMetadata],
    ) -> tuple[BufferedSampleMetadata, ...]:
        for item in candidates:
            self._constraints.validate_sample(item)
        return tuple(sorted(candidates, key=self._ordering_key))

    def _ordering_key(self, item: BufferedSampleMetadata) -> tuple:
        if self.enable_balancing:
            return (-item.metadata.cost.llm, -item.metadata.pack_tokens, item.key)
        return (
            -item.metadata.pack_tokens,
            -item.metadata.cost.dominant,
            -item.metadata.cost.total,
            item.key,
        )

    def _fits(self, packing_bin: _MutableBin, item: BufferedSampleMetadata) -> bool:
        return self._constraints.fits(packing_bin.pack_tokens, packing_bin.packing_costs, item)

    def _placement_score(
            self,
            packing_bin: _MutableBin,
            item: BufferedSampleMetadata,
            rank_costs: Sequence[WorkloadCost],
            rank_tokens: Sequence[int],
            *,
            seeding: bool,
    ) -> tuple[float, ...]:
        data_rank = packing_bin.data_rank
        remaining_capacity = self.seq_len - min(
            self.seq_len,
            packing_bin.pack_tokens + item.metadata.pack_tokens,
        )
        if self.enable_balancing:
            return (
                rank_costs[data_rank].llm,
                float(remaining_capacity),
                float(rank_tokens[data_rank]),
                float(data_rank),
                float(packing_bin.pack_index),
            )
        projected_cost = rank_costs[data_rank] + item.metadata.cost
        projected_rank_tokens = rank_tokens[data_rank] + min(item.metadata.pack_tokens, self.seq_len)
        placement_phase = 0.0 if seeding else 1.0
        return (
            projected_cost.dominant,
            projected_cost.total,
            float(projected_rank_tokens),
            float(remaining_capacity),
            placement_phase,
            float(data_rank),
            float(packing_bin.pack_index),
        )

    @staticmethod
    def _objective_score(rank_costs: Sequence[WorkloadCost], scale: float) -> tuple[float, ...]:
        if scale == 0.0:
            return (0.0, 0.0, 0.0)
        loads = [cost.llm / scale for cost in rank_costs]
        mean = math.fsum(loads) / len(loads)
        variance = math.fsum((load - mean) ** 2 for load in loads) / len(loads)
        return (max(loads), variance, max(loads) - min(loads))

    def _place(
            self,
            packing_bin: _MutableBin,
            item: BufferedSampleMetadata,
            rank_costs: list[WorkloadCost],
            rank_tokens: list[int],
    ) -> None:
        if not self._fits(packing_bin, item):
            raise ValueError(f"Sample {item.key} cannot fit within this bin's token and stage budgets.")
        packing_bin.samples.append(item)
        packing_bin.pack_tokens += item.metadata.pack_tokens
        packing_bin.packing_costs = self._constraints.add_costs(packing_bin.packing_costs, item)
        packing_bin.cost = packing_bin.cost + item.metadata.cost
        packing_bin.oversized = packing_bin.pack_tokens > self.seq_len
        data_rank = packing_bin.data_rank
        rank_costs[data_rank] = rank_costs[data_rank] + item.metadata.cost
        rank_tokens[data_rank] += min(item.metadata.pack_tokens, self.seq_len)

    def _freeze_bins(
            self,
            bins: Sequence[_MutableBin],
    ) -> tuple[tuple[PackingBinPlan, ...], ...]:
        local_batches = []
        for data_rank in range(self.data_parallel_size):
            rank_bins = []
            for packing_bin in bins:
                if packing_bin.data_rank != data_rank:
                    continue
                rank_bins.append(PackingBinPlan(
                    sample_keys=tuple(item.key for item in packing_bin.samples),
                    pack_tokens=packing_bin.pack_tokens,
                    oversized=packing_bin.oversized,
                    validate=False,
                ))
            local_batches.append(tuple(rank_bins))
        return tuple(local_batches)

    def _plan_id(
            self,
            step: int,
            local_batches: tuple[tuple[PackingBinPlan, ...], ...],
    ) -> str:
        stable_plan = {
            "step": step,
            "seq_len": self.seq_len,
            "local_batch_size": self.local_batch_size,
            "data_parallel_size": self.data_parallel_size,
            "bins": [
                {
                    "data_rank": data_rank,
                    "packs": [
                        {
                            "samples": [
                                [key.reader_rank, key.dataset_index]
                                for key in packing_bin.sample_keys
                            ],
                        }
                        for packing_bin in local_batch
                    ],
                }
                for data_rank, local_batch in enumerate(local_batches)
            ],
        }
        positions = [
            key.global_sample_position
            for local_batch in local_batches
            for packing_bin in local_batch
            for key in packing_bin.sample_keys
        ]
        if any(positions):
            stable_plan["global_sample_positions"] = positions
        encoded = json.dumps(stable_plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]


__all__ = ["DynamicPackingPlanner", "OversizedPolicy"]
