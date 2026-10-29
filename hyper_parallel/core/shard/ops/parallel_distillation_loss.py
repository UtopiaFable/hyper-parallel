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
"""Layout rules for MindSpore distillation primitives.

Losses operate on equally shaped and equally distributed inputs. Reduction is
deferred through Partial placements, so local token means are never mislabeled
as globally replicated losses. LogSoftmax gathers only its normalization axis
when sharded, rather than silently normalizing a local vocabulary shard.
"""
from copy import deepcopy
from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.shard.ops.parallel_ops import DistributedOp


def _normalize_pointwise_loss_args(input_tensor, target, reduction="mean", log_target=False):
    """Keep the primitive reduction enum/string unchanged for the local call."""
    return (input_tensor, target, reduction, log_target), {}


def _normalize_log_softmax_args(input_tensor, dim=-1, dtype=None):
    """Normalize all three LogSoftmaxExt slots, including optional dtype."""
    return (input_tensor, dim, dtype), {}


class PointwiseLossDistributedOp(DistributedOp):
    """MSELossExt and KLDiv for aligned DTensors, with none/sum/mean reduction."""

    _MS_PRIMITIVE_OP_NAMES = frozenset({"MSELossExt", "KLDiv"})
    _REDUCTIONS = {0: "sum", 1: "mean", 2: "none"}

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Unwrap inputs while retaining reduction and log_target in dispatch."""
        normalized, _ = _normalize_pointwise_loss_args(*args, **kwargs)
        input_tensor, target, reduction, log_target = normalized
        local_args = tuple(value.to_local() if hasattr(value, "_layout") else value
                           for value in normalized[:2]) + (reduction,)
        if self.op_name == "KLDiv":
            local_args += (log_target,)
        cache_values = [getattr(value, "layout", None) for value in normalized[:2]]
        cache_values += [input_tensor.shape, target.shape, reduction]
        return local_args, {}, cache_values

    @staticmethod
    def _validate_input_layouts(layouts: list, shapes: list, op_name: str) -> None:
        """Require identical global shapes, meshes and placements; reject broadcasting."""
        if any(layout is None for layout in layouts) or layouts[0] != layouts[1]:
            raise ValueError(
                f"For {op_name}, input layouts should be identical DTensor layouts, but got {layouts}.")
        if tuple(shapes[0]) != tuple(shapes[1]):
            raise ValueError(f"For {op_name}, input shapes should be identical, but got {shapes}.")

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Preserve elementwise layout or mark reduced mesh axes Partial.

        Rules:
            Inputs must be non-Partial DTensors with identical shapes/layouts.
            Reductions are none, sum or mean (strings or MindSpore enums).
            Broadcast targets must be explicitly expanded and distributed first.
        """
        self._check_partial_inputs(cache_values[:2])
        self._validate_input_layouts(cache_values[:2], cache_values[2:4], self.op_name)
        reduction = self._REDUCTIONS.get(cache_values[4], cache_values[4])
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(
                f"For {self.op_name}, reduction should be none, sum or mean, but got {cache_values[4]}.")
        input_layout = cache_values[0]
        if reduction == "none":
            return ((deepcopy(input_layout),), None)
        output = Layout(mesh_shape=input_layout.mesh_shape, alias_name=input_layout.alias_name,
                        rank_list=input_layout.rank_list)()
        for mapping in input_layout.alias_tensor_map:
            aliases = mapping if isinstance(mapping, (tuple, list)) else (mapping,)
            for alias in aliases:
                if alias != "None":
                    output.set_partial_by_dev_axis(alias, "avg" if reduction == "mean" else "sum")
        return ((output,), None)


class LogSoftmaxDistributedOp(DistributedOp):
    """LogSoftmaxExt preserving dtype with a normalization-axis gather fallback."""

    _MS_PRIMITIVE_OP_NAMES = frozenset({"LogSoftmaxExt"})

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Preserve dim and optional dtype positional slots of the MS primitive."""
        normalized, _ = _normalize_log_softmax_args(*args, **kwargs)
        input_tensor, dim, dtype = normalized
        return (input_tensor.to_local(), dim, dtype), {}, [input_tensor.layout, dim]

    @staticmethod
    def _validate_input_layouts(layout: Layout, dim: int, op_name: str) -> None:
        """Require a valid normalization dimension."""
        ndim = len(layout.alias_tensor_map)
        if isinstance(dim, bool) or not isinstance(dim, int) or not -ndim <= dim < ndim:
            raise ValueError(f"For {op_name}, dim should be an axis in [-{ndim}, {ndim}), but got {dim}.")

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Preserve input shards; only Partial inputs and invalid axes are unsupported."""
        self._check_partial_inputs(cache_values[:1])
        self._validate_input_layouts(cache_values[0], cache_values[1], self.op_name)
        return ((deepcopy(cache_values[0]),), None)

    @staticmethod
    def get_expand_impl(func: Callable, infer_result: tuple, cache_values: list) -> Optional[Callable]:
        """Gather a sharded normalization axis, apply the native op, and restore shards.

        This correctness path keeps other tensor dimensions sharded and is
        differentiable through the existing redistribution machinery. It
        materializes a full normalization axis per token chunk; a fused
        vocabulary-parallel implementation can replace it independently.
        """
        del infer_result
        layout, dim = cache_values
        mapping = layout.alias_tensor_map[dim]
        if mapping == "None":
            return None
        target_map = list(layout.alias_tensor_map)
        target_map[dim] = "None"
        target = Layout(layout.mesh_shape, layout.alias_name, rank_list=layout.rank_list)(*target_map)

        def _expanded(input_tensor, axis, dtype):
            distributed = DTensor.from_local_with_layout(input_tensor, layout)
            gathered = distributed.redistribute(layout.mesh, target.placements)
            output = func(gathered.to_local(), axis, dtype)
            distributed_output = DTensor.from_local_with_layout(output, target)
            return distributed_output.redistribute(layout.mesh, layout.placements).to_local()

        return _expanded
