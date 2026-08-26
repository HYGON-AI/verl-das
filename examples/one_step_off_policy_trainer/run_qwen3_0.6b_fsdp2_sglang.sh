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

# dependency: sglang==0.5.15, transformers==5.12.1, kernels==0.14.1
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
NNODES=$( (awk '{print $1}' ${host_file} | sort -u | wc -l) || echo 1 )

# ===================================== Data Config =====================================
train_file=${data_path}/gsm8k/train.parquet
test_file=${data_path}/gsm8k/test.parquet
train_batch_size=1152
max_prompt_length=512
max_response_length=1024

DATA_CONFIG=(
    data.train_files=${train_file}
    data.val_files=${test_file}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# ===================================== Actor Model & Optim Config =====================================
lr=1e-6
ppo_mini_batch_size=192
ppo_micro_batch_size_per_gpu=32
kl_loss_coef=0.001
kl_loss_type=low_var_kl

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen3-0.6B
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.actor.optim.lr=${lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type}
    actor_rollout_ref.actor.entropy_coeff=0
)

# ===================================== Ref Config =====================================
ref_log_prob_micro_bsz=32

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_bsz}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# ===================================== Rollout Config =====================================
gen_tp=2
rollout_gpu_mem_util=0.6
rollout_n=5
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.max_model_len=2048
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=64
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=False
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ===================================== Critic Config =====================================
CRITIC_CONFIG=(
    critic.strategy=fsdp2
)

# ===================================== Trainer Config =====================================
project_name='One_Step_Off_Policy-Qwen3-0.6B-BASE-GSM8K'
exp_name='One_Step_Off_Policy-Qwen3-0.6B-BASE-FSDP2-SGLANG'
ngpus_per_node=8
n_gpus_rollout=2
n_gpus_training=$((ngpus_per_node - n_gpus_rollout))

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.val_before_train=False
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.save_freq=-1
    trainer.test_freq=5
    trainer.total_epochs=2
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${n_gpus_training}
    trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
    rollout.nnodes=${NNODES}
    rollout.n_gpus_per_node=${n_gpus_rollout}
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
    global_profiler.save_path=${VERL_PATH}/examples/one_step_off_policy_trainer/torch_prof
)

# Conditionally Add Torch Profiling Configuration
if [[ $profiling == "torch" ]]; then
    TRAINER_CONFIG+=(${PROFILE_CONFIG[@]})
fi

# Main One_Step_Off_Policy Training Command
python3 -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path=config \
    --config-name=one_step_off_ppo_trainer \
    hydra.searchpath=[file://${VERL_PATH}/third_party/verl/verl/trainer/config] \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${REF_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${CRITIC_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \