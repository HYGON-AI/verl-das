#!/bin/bash
for para in $*
do  
    if [[ $para == --hf_model_path* ]];then
        hf_model_path=${para#*=}
    elif [[ $para == --mcore_model_path* ]];then
        mcore_model_path=${para#*=}
    fi
done

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export GLOG_minloglevel=3
export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/third_party/verl:${VERL_PATH}/third_party/Megatron-LM:$PYTHONPATH

# single gpu
python ${VERL_PATH}/third_party/verl/scripts/converter_hf_to_mcore.py \
       --hf_model_path ${hf_model_path} \
       --output_path ${mcore_model_path}

# multi gpus
# torchrun --nproc_per_node 8 \
#          --nnodes 1 \
#          --node_rank 0 \
#          ${VERL_PATH}/verl/scripts/converter_hf_to_mcore.py \
#          --hf_model_path ${hf_model_path} \
#          --output_path ${mcore_model_path}