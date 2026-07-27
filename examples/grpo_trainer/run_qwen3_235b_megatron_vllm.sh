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
train_file=${data_path}/DAPO-Math-17k/dapo-math-17k.parquet
test_file=${data_path}/AIME-2024/aime-2024.parquet
train_prompt_bsz=32
max_prompt_length=$((1024 * 2))
max_response_length=$((1204 * 8))

DATA_CONFIG=(
    data.train_files=${train_file}
    data.val_files=${test_file}
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='left'
)

# ===================================== Actor Model & Optim Config =====================================
use_dynamic_bsz=True
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
train_prompt_mini_bsz=16
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 10 / 10))
optimizer_offload_fraction=1
train_tp=4
train_pp=16
train_cp=1
train_ep=4
train_etp=2
param_offload=True
optimizer_offload=True
grad_offload=True

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen3-235B
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=5
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=${param_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${optimizer_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${grad_offload}
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=True
)

# ===================================== Ref Config =====================================
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=${param_offload}
)

# ===================================== Rollout Config =====================================
n_resp_per_prompt=8
gpu_memory_utilization=0.65
gen_tp=32
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
)

# ===================================== Algorithm Config =====================================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# ===================================== Reward Config =====================================
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

REWARD_CONFIG=(
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
)

# ===================================== Trainer Config =====================================
project_name='GRPO-Qwen3-235B-BASE-DAPO-Math-17k-AIME-2024'
exp_name='GRPO-Qwen3-235B-BASE-Megatron-vLLM'
ngpus_per_node=8

TRAINER_CONFIG=(
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.device='cuda'
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
    trainer.resume_mode=disable
    trainer.log_val_generations=100
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
    ${REWARD_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \