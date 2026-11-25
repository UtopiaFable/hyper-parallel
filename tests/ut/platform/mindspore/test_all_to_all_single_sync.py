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
"""Synchronous native AllToAllV flags and row/element conversion contracts."""
from unittest.mock import patch

import numpy as np
import pytest
import mindspore as ms

from tests.common.mark_utils import arg_mark
from tests.ut.platform.mindspore._ensure_mindspore_platform import ensure_mindspore_platform_default

ensure_mindspore_platform_default()

from hyper_parallel.platform.mindspore.platform import MindSporePlatform  # pylint: disable=C0413


@pytest.mark.parametrize('receives', [[3, 4], [0, 0]])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_native_sync_preserves_row_contract_and_selects_sync(receives: list) -> None:
    """The sixth flag selects variable splits; async_op=False is a separate argument."""
    source = ms.Tensor(np.arange(10, dtype=np.float32).reshape(5, 2))
    primitive_output = ms.Tensor(np.arange(max(1, 2*sum(receives)), dtype=np.float32))
    with patch('hyper_parallel.platform.mindspore.platform.get_group_size', return_value=2), \
         patch('hyper_parallel.platform.mindspore.platform.inner_comm_all_to_all_v_op',
               return_value=primitive_output) as primitive:
        output = MindSporePlatform.differentiable_all_to_all_single(source, [1, 4], receives, 'test')
    args = primitive.call_args.args
    np.testing.assert_array_equal(args[0].asnumpy(), np.arange(10, dtype=np.float32))
    assert args[1:] == ('test', [2, 8], [2*n for n in receives], 2, False)
    assert primitive.call_args.kwargs == {'async_op': False}
    assert output.shape == (sum(receives), 2)
    np.testing.assert_array_equal(output.asnumpy().reshape(-1), np.arange(2*sum(receives), dtype=np.float32))


@pytest.mark.parametrize('sends', [[1], [-1, 6], [True, 4], [1, 1]])
@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_native_sync_rejects_invalid_splits_before_dispatch(sends: list) -> None:
    """Local shape/count failures must not enter a collective on one rank."""
    source = ms.Tensor(np.zeros((5, 2), np.float32))
    with patch('hyper_parallel.platform.mindspore.platform.get_group_size', return_value=2), \
         patch('hyper_parallel.platform.mindspore.platform.inner_comm_all_to_all_v_op') as primitive:
        with pytest.raises(ValueError):
            MindSporePlatform.differentiable_all_to_all_single(source, sends, [0, 0], 'test')
        primitive.assert_not_called()
