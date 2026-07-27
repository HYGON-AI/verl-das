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
train_file=${data_path}/gsm8k/train.parquet
train_file_2=${data_path}/math/train.parquet
test_file=${data_path}/gsm8k/test.parquet
test_file_2=${data_path}/math/test.parquet
train_files="['${train_file}','${train_file_2}']"
test_files="['${test_file}','${test_file_2}']"
train_batch_size=128
max_prompt_length=1024
max_response_length=2048

DATA_CONFIG=(
    data.train_files=${train_files}
    data.val_files=${test_files}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# ===================================== Actor Model & Optim Config =====================================
actor_lr=1e-6
ppo_mini_batch_size=128
ppo_max_token_len_per_gpu=24576
actor_tp=4
actor_pp=1

ACTOR_CONFIG=(
    model_engine=megatron
    actor_rollout_ref.model.path=${hf_model_path}/Qwen3-8B
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
)

# ===================================== Rollout Config =====================================
rollout_tp=2
rollout_gpu_mem_util=0.3
rollout_n=1
max_num_tokens=$((max_prompt_length + max_response_length + 1))
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ===================================== Distillation Config =====================================
teacher_world_size=4
teacher_tp=2
teacher_gpu_mem_util=0.3
distillation_loss_mode=forward_kl_topk
distillation_topk=64
use_policy_gradient=False

DISTILL_CONFIG=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${teacher_world_size}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path=${hf_model_path}/Qwen3-32B
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.teacher_models.teacher_model.inference.free_cache_engine=${enable_sleep}
    +distillation.teacher_models.teacher_model.inference.enable_sleep_mode=${enable_sleep}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

# ===================================== Trainer Config =====================================
project_name='DISTILLATION-Qwen3-8B-BASE-GSM8K-MATH'
exp_name='DISTILLATION-Qwen3-8B-BASE-Megatron-vLLM'
ngpus_per_node=4

TRAINER_CONFIG=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
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
    global_profiler.save_path=${VERL_PATH}/examples/on_policy_distillation_trainer/torch_prof
)

# Conditionally Add Torch Profiling Configuration
if [[ $profiling == "torch" ]]; then
    TRAINER_CONFIG+=(${PROFILE_CONFIG[@]})
fi

# Main On_Policy_Distillation Training Command
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${DISTILL_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \