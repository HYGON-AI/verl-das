#!/bin/bash

set -euo pipefail

data_path=${data_path:-${VERL_HCU_DATA_ROOT}}
hf_model_path=${hf_model_path:-${VERL_HCU_MODEL_ROOT}}
mcore_model_path=${mcore_model_path:-${VERL_HCU_MODEL_ROOT}}
export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libgalaxyhip.so

# ===================================== Data Config =====================================
train_file=${data_path}/gsm8k/train.parquet
test_file=${data_path}/gsm8k/test.parquet
train_prompt_bsz=128
max_prompt_length=$((128 * 1))
max_response_length=$((256 * 1))

DATA_CONFIG=(
    data.train_files=${train_file}
    data.val_files=${test_file}
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# ===================================== Actor Model & Optim Config =====================================
train_prompt_mini_bsz=64
use_kl_loss=True
kl_loss_coef=0.001
param_offload=True
optimizer_offload=True

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/deepseek-llm-7b-chat
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload}
)

# ===================================== Ref Config =====================================
REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# ===================================== Rollout Config =====================================
n_resp_per_prompt=5
gen_tp=2
gpu_memory_utilization=0.3
enable_sleep=False

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ===================================== Trainer Config =====================================
project_name='GRPO-deepseek-llm-7b-chat-BASE-GSM8K'
exp_name='GRPO-deepseek-llm-7b-chat-BASE-FSDP-vLLM'
ngpus_per_node=8

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.nnodes=1
    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.device='cuda'
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.test_freq=5
    trainer.save_freq=-1
    trainer.total_training_steps=5
)

# Main GRPO Training Command
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${REF_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \
