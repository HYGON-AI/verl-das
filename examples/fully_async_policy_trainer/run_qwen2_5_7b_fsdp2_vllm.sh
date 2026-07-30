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

# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
CURRENT_DIR=$( cd $( dirname $0 ) && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
NNODES=$( (awk '{print $1}' ${host_file} | sort -u | wc -l) || echo 1 )
rollout_mode="async"
rollout_name="vllm"  # sglang or vllm
return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ===================================== Data Config =====================================
train_file=${data_path}/dapo-math-17k/train.parquet
test_file=${data_path}/aime-2024/train.parquet
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
train_prompt_bsz=0
gen_prompt_bsz=1

DATA_CONFIG=(
    data.train_files=${train_file}
    data.val_files=${test_file}
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    data.gen_batch_size=${gen_prompt_bsz}
    data.return_raw_chat=${return_raw_chat}
)

# ===================================== Actor Model & Optim Config =====================================
train_prompt_mini_bsz=32
use_dynamic_bsz=True
loss_agg_mode="token-mean"
sp_size=2
fsdp_size=4

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen2.5-Math-7B
    actor_rollout_ref.model.use_remove_padding=True
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
)

# ===================================== Ref Config =====================================
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

REF_CONFIG=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

# ===================================== Rollout Config =====================================
n_resp_per_prompt=16
gen_tp=1
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

# ===================================== Reward Config =====================================
REWARD_CONFIG=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=$((1024 * 4))
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${max_response_length}
)

# ===================================== Critic Config =====================================
CRITIC_CONFIG=(
    critic.strategy=fsdp2
)

# ===================================== Trainer Config =====================================
project_name='Fully_Async_Policy-Qwen2.5-Math-7B-BASE-DAPO-Math-17k-AIME-2024'
exp_name='Fully_Async_Policy-Qwen2.5-Math-7B-BASE-FSDP2-vLLM'
n_gpus_training=4
n_gpus_rollout=4
total_rollout_steps=$(( 512 * 100 ))

TRAINER_CONFIG=(
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.val_before_train=True
    trainer.save_freq=-1
    trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
    trainer.resume_mode=auto
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${n_gpus_training}
    trainer.total_epochs=10
    trainer.test_freq=10
    rollout.nnodes=${NNODES}
    rollout.n_gpus_per_node=${n_gpus_rollout}
    rollout.total_rollout_steps=${total_rollout_steps}
)

# ===================================== Async Training Config =====================================
staleness_threshold=0.1
trigger_parameter_sync_step=4
require_batches=4
partial_rollout=True

ASYNC_CONFIG=(
    async_training.staleness_threshold=${staleness_threshold}
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
    async_training.require_batches=${require_batches}
    async_training.partial_rollout=${partial_rollout}
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
    global_profiler.save_path=${VERL_PATH}/examples/fully_async_policy_trainer/torch_prof
)

# Conditionally Add Torch Profiling Configuration
if [[ $profiling == "torch" ]]; then
    TRAINER_CONFIG+=(${PROFILE_CONFIG[@]})
fi

# Main Fully_Async_Policy Training Command
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_trainer \
    hydra.searchpath=[file://${VERL_PATH}/third_party/verl/verl/trainer/config] \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${REF_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${REWARD_CONFIG[@]} \
    ${CRITIC_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \
    ${ASYNC_CONFIG[@]} \