#!/bin/bash
set -o pipefail

for para in $*
do
    if [[ $para == --model_name* ]];then
        model_name=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
SAVE_CKPT_PATH=${VERL_PATH}/examples/grpo_trainer
LOG_PATH=${model_name}-`date +%F-%H%M`.log
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export VERL_PATH=${VERL_PATH}
export TENSORBOARD_DIR=${SAVE_CKPT_PATH}/tensorboard
export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/third_party/verl:${VERL_PATH}/third_party/Megatron-LM:$PYTHONPATH
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libgalaxyhip.so

RED="\033[31m"
RESET="\033[0m"
DEEPSEEK_MODELS=(
    deepseek_7b_llm_fsdp_vllm
    deepseek_7b_llm_math_fsdp_vllm
)

MOONLIGHT_MODELS=(
    moonlight_16b_a3b_megatron_vllm
)

QWEN2_5_MODELS=(
    qwen2_5_0.5b_fsdp_vllm
    qwen2_5_vl_7b_fsdp2_vllm
    qwen2_5_vl_7b_megatron_vllm
)

QWEN3_MODELS=(
    qwen3_8b_fsdp_vllm
    qwen3_8b_megatron_vllm
    qwen3_vl_8b_fsdp2_vllm
    qwen3_vl_8b_megatron_vllm
    qwen3_235b_megatron_vllm
)

QWEN3_5_MODELS=(
    qwen3_5_9b_megatron_vllm
    qwen3_5_27b_fsdp2_vllm
    qwen3_5_27b_megatron_vllm
    qwen3_5_35b_a3b_fsdp2_vllm
    qwen3_5_35b_a3b_megatron_vllm
)

SEED_OSS_MODELS=(
    seed_oss_36b_fsdp2_vllm
)

SUPPORTED_MODELS=(
    "${DEEPSEEK_MODELS[@]}"
    "${MOONLIGHT_MODELS[@]}"
    "${QWEN2_5_MODELS[@]}"
    "${QWEN3_MODELS[@]}"
    "${QWEN3_5_MODELS[@]}"
    "${SEED_OSS_MODELS[@]}"
)

env_args=(
    -x PATH
    -x LIBRARY_PATH
    -x LD_LIBRARY_PATH
    -x PYTHONPATH
    -x HYHAL_PATH
    -x ROCM_PATH
    -x PYTHONWARNINGS
    -x TRANSFORMERS_VERBOSITY
    -x VLLM_CUDART_SO_PATH
)

if [[ -z "$model_name" ]]; then
    echo -e "${RED}Missing argument: --model_name${RESET}"
    echo "Currently supported models:"
    for m in "${SUPPORTED_MODELS[@]}"; do
        echo "  - $m"
    done
    echo -e "\nExample:"
    echo "  bash run.sh --model_name=${SUPPORTED_MODELS[0]}"
    exit 1 
elif [[ ${model_name} == "qwen3_235b_megatron" ]]; then
    export NCCL_NVLS_ENABLE=0
    export VLLM_USE_V1=1
    env_args+=(
        -x NCCL_NVLS_ENABLE
        -x VLLM_USE_V1
    )
elif [[ ${model_name} == "moonlight_16b_a3b_megatron_vllm" ]]; then
    export VLLM_ROCM_USE_AITER=True
    export VLLM_ROCM_USE_AITER_MOE=True
    export VLLM_ROCM_USE_AITER_FP8BMM=False
    export VLLM_HCU_USE_FUSE_MOE_GATE=False
    env_args+=(
        -x VLLM_ROCM_USE_AITER
        -x VLLM_ROCM_USE_AITER_MOE
        -x VLLM_ROCM_USE_AITER_FP8BMM
        -x VLLM_HCU_USE_FUSE_MOE_GATE
    )
fi

# These variables should be modified
export NET_TYPE="" # please choose one of {mlnx, shca}.
PORT="" # The port which you set in your docker
HOST_FILE=""
DATA_PATH=""
HF_MODEL_PATH=""
MCORE_MODEL_PATH=""
PROFILING="" # If you want to profiling, please choose one of {torch}

# pstart ray
head_ip=$(awk '{print $1}' ${HOST_FILE} | head -n 1)
mpirun -v \
    --allow-run-as-root \
    --bind-to none \
    --hostfile ${HOST_FILE} \
    --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args "-p ${PORT}" \
    ${env_args[@]} \
    bash ../scripts/pstart_ray.sh ${head_ip} ${HOST_FILE} 2>&1 | tee ${LOG_PATH}
ray_status=${PIPESTATUS[0]}
if [[ ${ray_status} -ne 0 ]]; then
    echo -e "${RED}Ray startup failed with status ${ray_status}${RESET}"
    exit "${ray_status}"
fi

# hcu verl patch
cp ${VERL_PATH}/hcu_verl/patch_init.py ${VERL_PATH}/third_party/verl/verl/__init__.py

bash run_${model_name}.sh \
    --data_path=${DATA_PATH} \
    --host_file=${HOST_FILE} \
    --hf_model_path=${HF_MODEL_PATH} \
    --mcore_model_path=${MCORE_MODEL_PATH} \
    --save_ckpt_path=${SAVE_CKPT_PATH} \
    --profiling=${PROFILING} 2>&1 | tee -a ${LOG_PATH}
training_status=${PIPESTATUS[0]}
exit "${training_status}"
