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
"""Layout and argument contracts for distillation operator registrations."""
from unittest.mock import Mock, patch

import pytest
import numpy as np

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.shard.ops.parallel_distillation_loss import (
    LogSoftmaxDistributedOp, PointwiseLossDistributedOp,
)
from hyper_parallel.core.shard.ops.parallel_npu_flash_attention_varlen_v4 import (
    NpuFlashAttentionVarLenV4DistributedOp,
)
from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op


@pytest.fixture(name="layouts")
def layout_factory():
    """Construct topology-only layouts without creating communication groups."""
    with patch("hyper_parallel.core.dtensor.device_mesh.platform") as mock_platform:
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = 4
        mock_platform.tensor_to_numpy.side_effect = lambda tensor: (
            tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor))
        layout = Layout((2, 2), ("data", "model"), rank_list=list(range(4)))
        yield layout


@pytest.mark.parametrize("name", ["MSELossExt", "KLDiv", "LogSoftmaxExt", "npu_flash_attention_varlen_v4"])
def test_yaml_registration(name):
    """Use the registry populated by normal package initialization."""
    assert get_distributed_op(name) is not None


@pytest.mark.parametrize("name", ["MSELossExt", "KLDiv"])
@pytest.mark.parametrize("mapping", [("data", "None"), ("None", "model"), ("data", "model"), ("None", "None")])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean", 0, 1, 2])
def test_loss_layout(layouts, name, mapping, reduction):
    """Elementwise losses preserve shards; global reductions become Partial."""
    layout = layouts(*mapping)
    op = PointwiseLossDistributedOp(name)
    outputs, _ = op.infer_layout([layout, layout, (8, 16), (8, 16), reduction])
    output = outputs[0]
    if reduction in ("none", 2):
        assert output == layout
        assert output is not layout
    else:
        assert len(output.alias_tensor_map) == 0
        assert output.is_partial() == any(alias != "None" for alias in mapping)
    # Local kernels already implement none/sum/mean; no expanded computation is needed.
    assert op.get_expand_impl(None, (outputs, None), []) is None


def test_loss_rejects_bad_layouts_and_broadcast(layouts):
    """Do not silently run global targets or Partial activations as local losses."""
    layout = layouts("data", "None")
    op = PointwiseLossDistributedOp("KLDiv")
    with pytest.raises(ValueError, match="identical DTensor"):
        op.infer_layout([layout, layouts("None", "model"), (8, 16), (8, 16), "none"])
    with pytest.raises(ValueError, match="identical"):
        op.infer_layout([layout, layout, (8, 16), (1, 16), "none"])
    partial = layouts("None", "None")
    partial.set_partial_by_dev_axis("data", "sum")
    with pytest.raises(ValueError, match="Partial"):
        op.infer_layout([partial, partial, (8, 16), (8, 16), "none"])
    with pytest.raises(ValueError, match="reduction"):
        op.infer_layout([layout, layout, (8, 16), (8, 16), "invalid"])


def test_loss_argument_slots(layouts):
    """Preserve log_target and enum reduction, without passing a KL slot to MSE."""
    layout = layouts("data", "None")
    tensor = Mock(_layout=layout, layout=layout, shape=(8, 16))
    for name, expected_args in (("MSELossExt", 3), ("KLDiv", 4)):
        op = PointwiseLossDistributedOp(name)
        args, kwargs, cache = op.preprocess((tensor, tensor), {"reduction": 2, "log_target": True})
        assert len(args) == expected_args
        assert args[2] == 2
        assert kwargs == {}
        assert cache[-1] == 2
        if name == "KLDiv":
            assert args[3] is True


def test_log_softmax_dtype_axis_and_errors(layouts):
    """Keep dtype slot and use a gathered normalization axis for vocabulary shards."""
    layout = layouts("data", "None")
    op = LogSoftmaxDistributedOp("LogSoftmaxExt")
    tensor = Mock(layout=layout)
    args, kwargs, cache = op.preprocess((tensor, -1, "float32"), {})
    assert args[1:] == (-1, "float32")
    assert kwargs == {}
    assert op.infer_layout(cache)[0][0] == layout
    assert op.get_expand_impl(None, None, cache) is None
    for axis in (2, -3, None, True):
        with pytest.raises(ValueError, match="dim"):
            op.infer_layout([layout, axis])
    sharded = layouts("None", "model")
    result = op.infer_layout([sharded, -1])
    assert result[0][0] == sharded
    assert callable(op.get_expand_impl(None, result, [sharded, -1]))


def test_varlen_v4_preserves_window_and_three_outputs(layouts):
    """Head sharding adjusts head_num without dropping window or document lengths."""
    layout = layouts("None", "model", "None")
    tensor = Mock(_layout=layout, layout=layout, shape=(20, 4, 32))
    op = NpuFlashAttentionVarLenV4DistributedOp("npu_flash_attention_varlen_v4")
    args, kwargs, cache = op.preprocess((tensor, tensor, tensor), {
        "actual_seq_qlen": (7, 20), "actual_seq_kvlen": (7, 20),
        "head_num": 4, "sparse_mode": 4, "pre_tokens": 3, "next_tokens": 0,
    })
    assert kwargs == {}
    assert args[8:10] == ((7, 20), (7, 20))
    assert args[13:15] == (3, 0)
    result = op.infer_layout(cache)
    assert len(result[0]) == 3
    local_outputs = (Mock(), Mock(), Mock())
    kernel = Mock(return_value=local_outputs)
    expanded = op.get_expand_impl(kernel, result, cache)
    assert callable(expanded)
    with patch("hyper_parallel.core.shard.ops.parallel_ms_flash_attention_score._get_lb_override",
               return_value=(None, None)):
        assert expanded(*args) == local_outputs
    local_args = kernel.call_args.args
    assert local_args[7:11] == (2, 4, 3, 0)
    assert local_args[-1] is True
    with pytest.raises(ValueError, match="TND order"):
        op.infer_layout(cache[:4] + [False])
