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

from verl.utils.device import get_torch_device, is_npu_available

def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return f"GPU-{get_torch_device().get_device_properties(device_id).uuid}"