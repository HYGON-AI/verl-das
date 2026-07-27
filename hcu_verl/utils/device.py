# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from verl.utils.device import is_torch_npu_available


def get_visible_devices_keyword() -> str:
    if os.environ.get("HIP_VISIBLE_DEVICES", None):
        return "HIP_VISIBLE_DEVICES"
    
    return "CUDA_VISIBLE_DEVICES" if not is_torch_npu_available(check_device=False) else "ASCEND_RT_VISIBLE_DEVICES"