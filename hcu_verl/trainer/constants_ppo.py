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
from verl.utils.device import get_device_capability

VERL_PATH = os.getenv('VERL_PATH')
ENV = {
    "mlnx": {
        "NCCL_IB_HCA": "mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1",
        "ROCSHMEM_MAX_NUM_CONTEXTS": "48",
        "ROCSHMEM_HEAP_SIZE": "2684354560",
        "ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX": "288",
        "ROCSHMEM_TOPO_FILE_FORCE": f"{VERL_PATH}/examples/topo/topo.config",
        "ROCSHMEM_ALLOWED_IBV_DEVICES": "mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9",
    },

    "shca": {
        "NCCL_PXN_DISABLE": "0",
        "NCCL_PLUGIN_P2P": "ib",
        "NCCL_NET_PLUGIN": "shca",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_IB_HCA": "shca_0:1,shca_1:1,shca_2:1,shca_3:1",
        "RCCL_PXN_GPU_BALANCE": "1",
        "RCCL_NET_PLANE": "shca_0,shca_3|shca_1,shca_2",
        "SHCA_DEBUG_MASK": "0",
        "SHCA_CMR_LOG_LEVEL": "1",
        "UCX_IB_NUM_PATHS": "1",   
    },
}

_major, _ = get_device_capability()
# Opt-in GB200 NCCL WAR: set TLLM_DISABLE_NVLS_MNNVL=1 in the launch shell to disable
# both NCCL_NVLS_ENABLE and NCCL_MNNVL_ENABLE on Blackwell. Required by async-RL
# Megatron on GB200 nodes without IMEX (mbridge all_gather raises NCCL 801).
_gb200_nccl_env = {}
if (_major or 0) >= 10 and os.environ.get("TLLM_DISABLE_NVLS_MNNVL", "0") == "1":
    _gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        # "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        **_gb200_nccl_env,
        
        # hcu add
        "TORCH_CPP_LOG_LEVEL": "fatal",
        "GLOG_minloglevel": "3",
        "GPU_MAX_HW_QUEUES": "10",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    },
}

def apply_env(env_dict):
    net_type = os.getenv("NET_TYPE", None)
    if net_type is not None:
        assert net_type in env_dict.keys(), \
            f"Expected NET_TYPE is one of (mlnx, shca), but got {net_type}. " \
            f"Please set NET_TYPE env correctly." 

        PPO_RAY_RUNTIME_ENV["env_vars"].update(env_dict[net_type])
    else:
        pass

apply_env(ENV)
