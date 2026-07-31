#!/bin/bash
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

CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
NNODES=$( (awk '{print $1}' ${host_file} | sort -u | wc -l) || echo 1 )

# ===================================== Data Config =====================================
train_file=${data_path}/gsm8k/train.parquet
test_file=${data_path}/gsm8k/test.parquet
train_prompt_bsz=1024
max_prompt_length=$((512 * 1))
max_response_length=$((1024 * 1))

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
train_prompt_mini_bsz=256
use_kl_loss=True
kl_loss_coef=0.001
param_offload=True
optimizer_offload=True

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen2.5-0.5B-Instruct
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload}
)

# ===================================== Ref Config =====================================
REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload}
)

# ===================================== Rollout Config =====================================
n_resp_per_prompt=5
gen_tp=1
gpu_memory_utilization=0.5
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
)

# ===================================== Algorithm Config =====================================
adv_estimator=grpo
kl_coef=0.0001

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# ===================================== Trainer Config =====================================
project_name='GRPO-Qwen2.5-0.5B-Instruct-BASE-GSM8K'
exp_name='GRPO-Qwen2.5-0.5B-Instruct-BASE-FSDP-vLLM'
ngpus_per_node=8

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.device='cuda'
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.test_freq=5
    trainer.save_freq=20
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
    --config-name=ppo_trainer \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${REF_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \