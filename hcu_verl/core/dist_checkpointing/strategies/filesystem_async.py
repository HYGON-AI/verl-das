# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path
from typing import List, Tuple

import torch

WriteBucket = Tuple[Path, str, Tuple[list, list]]  # represents writes to a single file


@staticmethod
def preload_tensors(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
    """
    Preloads tensors in `state_dict` to host memory via CPU memory.

    Args:
        write_buckets (List): List of `WriteBucket` objects that define what to
            save in a checkpoint.
        non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
    """
    result = []

    for bucket in write_buckets:
        file_name, storage_key, (bytes_data, tensor_data) = bucket
        tensor_list = []
        for item, tensor in tensor_data:
            # we belive these tensors are detached from the model trainers
            tensor_list.append((item, tensor.to("cpu", non_blocking=False)))
            # This is required for `PersistentAsyncCaller` to remove reference
            del tensor
        result.append((file_name, storage_key, (bytes_data, tensor_list)))
    if non_blocking:
        torch.cuda.synchronize()
    return result