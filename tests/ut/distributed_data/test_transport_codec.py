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
"""Tests for internal payload serialization and sample conservation."""

import pickle
import unittest
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.distributed_data.topology import DataTopology
from hyper_parallel.distributed_data.transport import (
    DataGroups,
    DataPlaneTransport,
    ModelParallelTransport,
    _decode_model_batch,
    _encode_model_batch,
    create_data_groups,
    _decode_payload_segment,
    _decode_received_payloads,
    _encode_payload_segment,
)


class TestPayloadCodec(unittest.TestCase):
    """Verify route round trips and conservation checks at payload merge."""

    def test_round_trip_preserves_order_keys_and_payloads(self) -> None:
        """A valid route segment round-trips without changing payload order."""
        items = (
            (SampleKey(0, 7), {"tokens": [1, 2], "image": b"jpeg"}),
            (SampleKey(4, 3), ("caption", 9)),
        )

        encoded = _encode_payload_segment(items)

        self.assertEqual(_decode_payload_segment(encoded), items)
        self.assertEqual(_encode_payload_segment(()), b"")
        self.assertEqual(_decode_payload_segment(b""), ())

    def test_wire_format_is_plain_pickle(self) -> None:
        """All routes use the same codec without a second framing/checksum pass."""
        items = ((SampleKey(0, 1), {"value": 3}),)

        self.assertEqual(_encode_payload_segment(items), pickle.dumps(items, protocol=pickle.HIGHEST_PROTOCOL))

    def test_decoder_propagates_pickle_errors(self) -> None:
        """Deserialization retains the original exception instead of wrapping it."""
        with self.assertRaises(pickle.UnpicklingError):
            _decode_payload_segment(b"invalid pickle")

    def test_receive_rejects_duplicates_within_and_across_routes(self) -> None:
        """Merging into a dictionary must not silently overwrite an occurrence."""
        key = SampleKey(1, 5)
        for segments in (
                (_encode_payload_segment(((key, "first"), (key, "second"))),),
                (_encode_payload_segment(((key, "first"),)), _encode_payload_segment(((key, "second"),))),
        ):
            with self.subTest(segment_count=len(segments)), self.assertRaisesRegex(ValueError, "duplicate payload"):
                _decode_received_payloads(b"".join(segments), tuple(map(len, segments)))

    def test_receive_preserves_empty_routes_and_repeated_dataset_indices(self) -> None:
        """Distinct sampling occurrences may legitimately refer to the same index."""
        items = ((SampleKey(0, 3, 0), "first"), (SampleKey(0, 3, 1), "second"))
        segments = (b"", _encode_payload_segment(items[:1]), b"", _encode_payload_segment(items[1:]))

        result = _decode_received_payloads(b"".join(segments), tuple(map(len, segments)))

        self.assertEqual(result, dict(items))


class TestDataPlaneTransport(unittest.TestCase):
    """Verify control metadata and payload bytes use their designated groups."""

    @staticmethod
    def _two_rank_topology() -> DataTopology:
        return DataTopology.from_layout(
            mesh_shape=(2,),
            mesh_dim_names=("dp",),
            rank_list=(0, 1),
            global_rank=0,
            dp_dim_names=("dp",),
        )

    def test_cpu_communication_device_reuses_gloo_control_group(self) -> None:
        """An explicit CPU device must not inherit an accelerator WORLD backend."""
        with (
                patch("hyper_parallel.distributed_data.transport.dist.is_available", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.is_initialized", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.get_world_size", return_value=2),
                patch("hyper_parallel.distributed_data.transport.dist.get_rank", return_value=0),
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="hccl") as get_backend,
                patch("hyper_parallel.distributed_data.transport.dist.new_group", return_value="gloo") as new_group,
        ):
            groups = create_data_groups(
                self._two_rank_topology(),
                (0, 1),
                0,
                cpu_backend="gloo",
                payload_backend=None,
                communication_device="cpu",
                enable_payload_exchange=True,
            )

        self.assertEqual(groups.control_group, "gloo")
        self.assertIs(groups.payload_group, groups.control_group)
        self.assertEqual(new_group.call_count, 1)
        get_backend.assert_not_called()

    def test_control_plane_rejects_accelerator_backend_before_group_creation(self) -> None:
        """Object control collectives must use a CPU-capable backend."""
        with (
                patch("hyper_parallel.distributed_data.transport.dist.is_available", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.is_initialized", return_value=True),
                patch("hyper_parallel.distributed_data.transport.dist.get_world_size", return_value=2),
                patch("hyper_parallel.distributed_data.transport.dist.get_rank", return_value=0),
                patch("hyper_parallel.distributed_data.transport.dist.new_group") as new_group,
                self.assertRaisesRegex(ValueError, "cpu_backend must support CPU tensors"),
        ):
            create_data_groups(
                self._two_rank_topology(),
                (0, 1),
                0,
                cpu_backend="hccl",
                payload_backend="hccl",
                communication_device="npu:0",
                enable_payload_exchange=True,
            )

        new_group.assert_not_called()

    def test_payload_a2a_uses_payload_group_after_cpu_size_exchange(self) -> None:
        """Variable split sizes stay on control while payload bytes use the payload group."""
        groups = DataGroups(
            data_plane_ranks=(0, 1),
            control_group="control",
            payload_group="payload",
            model_parallel_group=None,
            planner_rank=0,
            distributed=True,
        )
        transport = DataPlaneTransport(groups, global_rank=0, communication_device="cpu")
        expected = {
            SampleKey(0, 0): {"sample": 0},
            SampleKey(0, 1): {"sample": 1},
        }
        prepared = transport.prepare_exchange({
            0: ((SampleKey(0, 0), expected[SampleKey(0, 0)]),),
            1: ((SampleKey(0, 1), expected[SampleKey(0, 1)]),),
        })
        collective_groups = []

        def fake_all_to_all(output: Any, input_tensor: Any, **kwargs: Any) -> None:
            """Copy local buffers while recording the selected process group."""
            collective_groups.append(kwargs["group"])
            output.copy_(input_tensor)

        with patch(
                "hyper_parallel.distributed_data.transport.dist.all_to_all_single",
                side_effect=fake_all_to_all,
        ):
            received = transport.exchange_prepared(prepared)

        self.assertEqual(received, expected)
        self.assertEqual(collective_groups, ["control", "payload"])

    def test_singleton_exchange_still_rejects_duplicate_occurrences(self) -> None:
        """Local bypass and multi-rank A2A share the same payload merge contract."""
        transport = DataPlaneTransport(DataGroups((0,), None, None, None, 0, False), global_rank=0)
        key = SampleKey(0, 1)
        prepared = transport.prepare_exchange({0: ((key, "first"), (key, "second"))})

        with self.assertRaisesRegex(ValueError, "duplicate payload"):
            transport.exchange_prepared(prepared)

    def test_unknown_target_is_not_silently_dropped(self) -> None:
        """Payload routes outside the fixed data plane still fail before A2A."""
        transport = DataPlaneTransport(DataGroups((0,), None, None, None, 0, False), global_rank=0)

        with self.assertRaisesRegex(ValueError, "outside the data plane"):
            transport.prepare_exchange({1: ((SampleKey(0, 1), "sample"),)})

    def test_missing_control_group_fails_at_construction(self) -> None:
        """Check immutable group configuration once rather than on every collective."""
        groups = DataGroups((0, 1), None, None, None, 0, False)

        with self.assertRaisesRegex(ValueError, "initialized process group"):
            DataPlaneTransport(groups, global_rank=0)
        self.assertFalse(DataPlaneTransport(groups, global_rank=2).is_member)


class TestModelParallelTransport(unittest.TestCase):
    """Verify tensor leaves use direct collectives while metadata stays serialized."""

    @staticmethod
    def _topology() -> DataTopology:
        return DataTopology.from_layout(
            mesh_shape=(1, 2), mesh_dim_names=("dp", "mp"), rank_list=(0, 1),
            global_rank=0, dp_dim_names=("dp",),
        )

    def test_codec_preserves_nested_structure_and_tensor_values(self) -> None:
        """Tensor leaves are removed from the object payload and restored in order."""
        batch = {"input_ids": torch.tensor([[1, 2]]), "meta": ["caption", None]}

        schema, tensors = _encode_model_batch(batch)
        restored = _decode_model_batch(schema, [tensor.clone() for tensor in tensors])

        self.assertEqual(restored["meta"], ["caption", None])
        self.assertTrue(torch.equal(restored["input_ids"], batch["input_ids"]))

    def test_broadcast_sends_tensor_leaves_directly(self) -> None:
        """Model-group object broadcast carries only structure and direct broadcast carries tensor data."""
        groups = DataGroups(
            data_plane_ranks=(0, 1), control_group=None, payload_group=None,
            model_parallel_group="model", planner_rank=0, distributed=True,
        )
        transport = ModelParallelTransport(self._topology(), groups)
        with patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list") as object_broadcast, \
                patch("hyper_parallel.distributed_data.transport.dist.broadcast") as tensor_broadcast, \
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"):
            transport.broadcast({"input_ids": torch.tensor([1, 2]), "label": "text"})

        object_broadcast.assert_called_once()
        tensor_broadcast.assert_called_once()
        self.assertEqual(object_broadcast.call_args.kwargs["group"], "model")
        self.assertEqual(tensor_broadcast.call_args.kwargs["group"], "model")

    def test_receiver_rebuilds_nested_batch_and_eof(self) -> None:
        """Directly broadcast tensors populate the receiver's local structure."""
        topology = DataTopology.from_layout(
            mesh_shape=(1, 2), mesh_dim_names=("dp", "mp"), rank_list=(0, 1),
            global_rank=1, dp_dim_names=("dp",),
        )
        groups = DataGroups((0,), None, None, "model", 0, True)
        transport = ModelParallelTransport(topology, groups)
        batch = {"inputs": (torch.tensor([1, 2]), [torch.tensor([[3.0]]), "text"])}
        schema, tensors = _encode_model_batch(batch)
        pending = iter(tensors)

        def broadcast_schema(payload: list, **_kwargs: Any) -> None:
            """Supply the constructor's schema to this receiver."""
            payload[0] = schema

        def broadcast_tensor(tensor: torch.Tensor, **_kwargs: Any) -> None:
            """Populate tensor leaves in encoder order."""
            tensor.copy_(next(pending))

        with (
                patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list",
                      side_effect=broadcast_schema),
                patch("hyper_parallel.distributed_data.transport.dist.broadcast",
                      side_effect=broadcast_tensor) as broadcast,
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="gloo"),
        ):
            received = transport.broadcast(None)
            self.assertEqual(broadcast.call_count, 2)
            schema, _ = _encode_model_batch(None)
            self.assertIsNone(transport.broadcast(None))
            self.assertEqual(broadcast.call_count, 2)

        self.assertTrue(torch.equal(received["inputs"][0], batch["inputs"][0]))
        self.assertTrue(torch.equal(received["inputs"][1][0], batch["inputs"][1][0]))
        self.assertEqual(received["inputs"][1][1], "text")

    def test_device_backend_mismatch_still_fails_before_tensor_broadcast(self) -> None:
        """User tensors must remain compatible with their collective backend."""
        groups = DataGroups((0,), None, None, "model", 0, True)
        transport = ModelParallelTransport(self._topology(), groups)
        with (
                patch("hyper_parallel.distributed_data.transport.dist.broadcast_object_list"),
                patch("hyper_parallel.distributed_data.transport.dist.broadcast") as broadcast,
                patch("hyper_parallel.distributed_data.transport.dist.get_backend", return_value="hccl"),
                self.assertRaisesRegex(ValueError, "incompatible"),
        ):
            transport.broadcast({"input_ids": torch.tensor([1, 2])})

        broadcast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
