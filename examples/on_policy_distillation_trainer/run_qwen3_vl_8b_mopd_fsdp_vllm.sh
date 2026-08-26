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

# dependency: vllm==0.25.1, transformers==5.12.1
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
NNODES=$( (awk '{print $1}' ${host_file} | sort -u | wc -l) || echo 1 )

# ===================================== Data Config =====================================
gsm8k_train_file=${data_path}/gsm8k/train.parquet
gsm8k_test_file=${data_path}/gsm8k/test.parquet
geo3k_train_file=${data_path}/geo3k/train.parquet
geo3k_test_file=${data_path}/geo3k/test.parquet
train_files="['${gsm8k_train_file}','${geo3k_train_file}']"
test_files="['${gsm8k_test_file}','${geo3k_test_file}']"
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
    data.truncation=error
    data.shuffle=True
    data.image_key=images
)

# ===================================== Actor Model & Optim Config =====================================
use_dynamic_bsz=True
actor_lr=1e-6
ppo_mini_batch_size=128
ppo_max_token_len_per_gpu=12288
actor_param_offload=True
actor_optimizer_offload=True

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen3-VL-8B-Instruct
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

# ===================================== Rollout Config =====================================
rollout_tp=2
rollout_gpu_mem_util=0.3
n_resp_per_prompt=1
max_num_tokens=$((max_prompt_length + max_response_length + 1))
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
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
teacher_num_replicas_gsm8k=1
teacher_num_replicas_geo3k=1
teacher_tp=2
teacher_world_size=$(((teacher_num_replicas_gsm8k + teacher_num_replicas_geo3k) * teacher_tp))
teacher_gpu_mem_util=0.3

DISTILL_CONFIG=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${teacher_world_size}
    distillation.nnodes=${NNODES}
    distillation.teacher_key=data_source
    +distillation.teacher_models.gsm8k.key=openai/gsm8k
    +distillation.teacher_models.gsm8k.model_path=${hf_model_path}/Qwen3-32B
    +distillation.teacher_models.gsm8k.num_replicas=${teacher_num_replicas_gsm8k}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.gsm8k.inference.max_model_len=${max_num_tokens}
    +distillation.teacher_models.gsm8k.inference.free_cache_engine=${enable_sleep}
    +distillation.teacher_models.gsm8k.inference.enable_sleep_mode=${enable_sleep}
    +distillation.teacher_models.geo3k.key=hiyouga/geometry3k
    +distillation.teacher_models.geo3k.model_path=${hf_model_path}/Qwen3-VL-32B-Instruct
    +distillation.teacher_models.geo3k.num_replicas=${teacher_num_replicas_geo3k}
    +distillation.teacher_models.geo3k.inference.name=vllm
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.geo3k.inference.max_model_len=${max_num_tokens}
    +distillation.teacher_models.geo3k.inference.free_cache_engine=${enable_sleep}
    +distillation.teacher_models.geo3k.inference.enable_sleep_mode=${enable_sleep}
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

# ===================================== Trainer Config =====================================
project_name='DISTILLATION-Qwen3-VL-8B-Instruct-BASE-GSM8K-GEO3K'
exp_name='DISTILLATION-Qwen3-VL-8B-Instruct-BASE-MOPD-FSDP-vLLM'
ngpus_per_node=4

TRAINER_CONFIG=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=200
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
    global_profiler.save_path=${VERL_PATH}/examples/on_policy_distillation_trainer/torch_prof
)

# Conditionally Add Torch Profiling Configuration
if [[ $profiling == "torch" ]]; then
    TRAINER_CONFIG+=(${PROFILE_CONFIG[@]})
fi

# Main On_Policy_Distillation Training Command
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${DISTILL_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \