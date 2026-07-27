#!/bin/bash
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
QWEN3_MODELS=(
    qwen3_0.6b_fsdp2_vllm
    qwen3_0.6b_fsdp2_sglang
)

SUPPORTED_MODELS=(
    "${QWEN3_MODELS[@]}"
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
wait

# hcu verl patch
cp ${VERL_PATH}/hcu_verl/patch_init.py ${VERL_PATH}/third_party/verl/verl/__init__.py

bash run_${model_name}.sh \
    --data_path=${DATA_PATH} \
    --host_file=${HOST_FILE} \
    --hf_model_path=${HF_MODEL_PATH} \
    --mcore_model_path=${MCORE_MODEL_PATH} \
    --save_ckpt_path=${SAVE_CKPT_PATH} \
    --profiling=${PROFILING} 2>&1 | tee -a ${LOG_PATH}
wait
