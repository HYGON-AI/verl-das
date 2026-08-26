# Copyright (c) 2026 BAAI. All rights reserved.
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

from verl.plugin.platform.platform_rocm import PlatformROCm


def rollout_env_vars(self) -> dict[str, str]:
    # Extend CUDA's rollout env vars with ROCm-specific ones. SGLANG_USE_AITER
    # routes SGLang's non-attention kernels (RMSNorm/RoPE/MoE/quant) through AITER.
    # Default to "0" but honor an explicit user override (e.g. SGLANG_USE_AITER=0
    # to fall back to vLLM kernels).
    return {
        **super(PlatformROCm, self).rollout_env_vars(),
        "SGLANG_USE_AITER": os.environ.get("SGLANG_USE_AITER", "0"),
    }