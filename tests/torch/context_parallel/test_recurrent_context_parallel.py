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
"""CPU distributed launchers for the head-independent affine state scan."""
import os

import pytest

from tests.torch.utils import torchrun_case


@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_recurrent_cp_cpu(ranks: int) -> None:
    """Run non-power-of-two and power-of-two CP groups with one attention head."""
    worker = os.path.join(os.path.dirname(__file__), "recurrent_context_parallel.py")
    torchrun_case(worker, "test_recurrent_context_parallel", num_proc=ranks)
