set -x

CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))

export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/third_party/verl:${VERL_PATH}/third_party/Megatron-LM:$PYTHONPATH
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

nproc_per_node=8
save_path=$CURRENT_DIR/output

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/public/home/yuhui1/data/gsm8k_processe/train.parquet \
    data.val_files=/public/home/yuhui1/data/gsm8k_processe/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen3/Qwen3-8B \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir="$save_path" \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen3-8b-instruct \
    trainer.total_epochs=10 \
    trainer.logger='["console"]' $@