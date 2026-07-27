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

# dependency: vllm==0.18.1, transformers==5.7.0, Megatron-LM==0.17.1
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
NNODES=$( (awk '{print $1}' ${host_file} | sort -u | wc -l) || echo 1 )

# ===================================== Data Config =====================================
train_file=${data_path}/geo3k/train.parquet
test_file=${data_path}/geo3k/test.parquet
train_batch_size=64
max_prompt_length=8192
max_response_length=24576

DATA_CONFIG=(
    data.train_files=${train_file}
    data.val_files=${test_file}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation=error
    data.filter_overlong_prompts=True
)

# ===================================== Actor Model & Optim Config =====================================
use_dynamic_bsz=False
lr=1e-6
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=1
ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
kl_loss_coef=0.01
kl_loss_type=low_var_kl
train_tp=4
train_pp=1
train_cp=1

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen3.5-9B
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.optim.lr=${lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.grad_offload=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
)

# ===================================== Ref Config =====================================
REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.param_offload=False
)

# ===================================== Rollout Config =====================================
gen_tp=4
rollout_gpu_mem_util=0.2
rollout_n=5
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.free_cache_engine=False
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ===================================== Trainer Config =====================================
project_name=GRPO-Qwen3.5-9B-BASE-GEO3K
exp_name=GRPO-Qwen3.5-9B-BASE-Megatron-vLLM
ngpus_per_node=8

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger="['console']"
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${ngpus_per_node}
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