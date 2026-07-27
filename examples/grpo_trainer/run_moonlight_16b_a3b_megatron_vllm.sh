#!/bin/bash

# Very important! The following are things you must do before running:
# 1、please add these params to config.json after downloading from huggingface:
# "rope_scaling": {
#     "beta_fast": 32,
#     "beta_slow": 1,
#     "factor": 40,
#     "mscale": 1.0,
#     "mscale_all_dim": 1.0,
#     "original_max_position_embeddings": 4096,
#     "type": "yarn"
#   },
# 2、convert HF model to megatron core format:
# cd ../scripts
# bash converter_hf_to_mcore.sh --hf_model_path=${hf_model_path}/Moonlight-16B-A3B --mcore_model_path=${mcore_model_path}/Moonlight-16B-A3B-mcore

for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=${para#*=}
    elif [[ $para == --host_file* ]];then
        host_file=${para#*=}
    elif [[ $para == --hf_model_path* ]];then
        hf_model_path=${para#*=}
    elif [[ $para == --mcore_model_path* ]];then
        mcore_model_path=${para#*=}
    elif [[ $para == --save_ckpt_path* ]];then
        save_ckpt_path=${para#*=}
    elif [[ $para == --profiling* ]];then
        profiling=${para#*=}
    fi
done

# dependency: vllm==0.18.1, transformers==4.57.3, kernels==0.11.0, Megatron-LM==0.17.1
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
NNODES=$( (awk '{print $1}' ${host_file} | sort -u | wc -l) || echo 3 )

# ===================================== Data Config =====================================
train_path=${data_path}/gsm8k/train.parquet
test_path=${data_path}/gsm8k/test.parquet
train_batch_size=192
max_prompt_length=1024
max_response_length=2048

DATA_CONFIG=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.trust_remote_code=True
)

# ===================================== Actor Model & Optim Config =====================================
actor_lr=1e-6
ppo_mini_bsz=64
ppo_micro_bsz_per_gpu=16
actor_tp=4
actor_pp=3
actor_ep=8
actor_etp=1

ACTOR_CONFIG=(
    model_engine=megatron
    actor_rollout_ref.model.path=${hf_model_path}/Moonlight-16B-A3B
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_bsz_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${mcore_model_path}/Moonlight-16B-A3B-mcore
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

# ===================================== Ref Config =====================================
log_prob_micro_bsz_per_gpu=16

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${mcore_model_path}/Moonlight-16B-A3B-mcore
)

# ===================================== Rollout Config =====================================
rollout_tp=4
rollout_gpu_mem_util=0.2
n_resp_per_prompt=5
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ===================================== Trainer Config =====================================
project_name=GRPO-Moonlight-16B-A3B-BASE-GSM8K
exp_name=GRPO-Moonlight-16B-A3B-BASE-Megatron-vLLM
n_gpus_per_node=8

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_gpus_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
)

# ===================================== Profiler Config =====================================
PROFILE_CONFIG=(
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks=[0,4]
    actor_rollout_ref.actor.profiler.all_ranks=False
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=['cuda','cpu']
    actor_rollout_ref.ref.profiler.enable=True
    actor_rollout_ref.ref.profiler.ranks=[0,4]
    actor_rollout_ref.ref.profiler.all_ranks=False
    actor_rollout_ref.ref.profiler.tool_config.torch.contents=['cuda','cpu']
    global_profiler.tool=${profiling}
    global_profiler.steps=[3]
    global_profiler.save_path=${VERL_PATH}/examples/grpo_trainer/torch_prof
)

# Conditionally Add Torch Profiling Configuration
if [[ $profiling == "torch" ]]; then
    TRAINER_CONFIG+=(${PROFILE_CONFIG[@]})
fi

# Main GRPO Training Command
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${REF_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \