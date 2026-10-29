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
train_file_2=${data_path}/math/train.parquet
test_file=${data_path}/gsm8k/test.parquet
test_file_2=${data_path}/math/test.parquet
train_files="['${train_file}','${train_file_2}']"
test_files="['${test_file}','${test_file_2}']"
train_batch_size=16
max_prompt_length=512
max_response_length=8192

DATA_CONFIG=(
    data.train_files=${train_files}
    data.val_files=${test_files}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=64
    data.truncation=error
)

# ===================================== Actor Model & Optim Config =====================================
actor_lr=1e-6
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=1
train_sp=4

ACTOR_CONFIG=(
    actor_rollout_ref.model.path=${hf_model_path}/Qwen3-8B
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.strategy=fsdp2
)

# ===================================== Ref Config =====================================
REF_CONFIG=(
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${train_sp}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# ===================================== Rollout Config =====================================
gen_tp=2
n_resp_per_prompt=5
gpu_memory_utilization=0.4
enable_sleep=True

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.free_cache_engine=${enable_sleep}
    +actor_rollout_ref.rollout.enable_sleep_mode=${enable_sleep}
    actor_rollout_ref.rollout.drafter.enable=True
    actor_rollout_ref.rollout.drafter.enable_drafter_training=True
    actor_rollout_ref.rollout.drafter.model_path=${hf_model_path}/dspark_qwen3_8b_block7
    actor_rollout_ref.rollout.drafter.rollout.spec_steps=1
    actor_rollout_ref.rollout.drafter.rollout.spec_topk=1
    actor_rollout_ref.rollout.drafter.rollout.spec_verify_tokens=7
    actor_rollout_ref.rollout.drafter.speculative_algorithm=DSPARK
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_sgl=False
    actor_rollout_ref.rollout.drafter.training.collect_hidden_states_from_old_logprob=True
    actor_rollout_ref.rollout.drafter.training.collect_interval_steps=5
    actor_rollout_ref.rollout.drafter.training.draft_update_weights_bucket_megabytes=512
    actor_rollout_ref.rollout.drafter.training.draft_update_pause_generation=True
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_before=False
    actor_rollout_ref.rollout.drafter.training.draft_update_flush_after=True
    actor_rollout_ref.rollout.drafter.training.dspark_block_size=7
    actor_rollout_ref.rollout.drafter.training.dspark_num_anchors=32
    actor_rollout_ref.rollout.drafter.training.dspark_max_window=512
    actor_rollout_ref.rollout.drafter.training.dspark_loss_mode=restricted_ce
    actor_rollout_ref.rollout.drafter.training.dspark_loss_decay_gamma=7
    actor_rollout_ref.rollout.drafter.training.dspark_num_target_layers=5
    actor_rollout_ref.rollout.drafter.training.dspark_num_hidden_layers=5
    actor_rollout_ref.rollout.drafter.training.dspark_markov_rank=256
    actor_rollout_ref.rollout.drafter.training.dspark_markov_head_type=vanilla
    actor_rollout_ref.rollout.drafter.training.dspark_ce_loss_alpha=0.1
    actor_rollout_ref.rollout.drafter.training.dspark_l1_loss_alpha=0.9
    actor_rollout_ref.rollout.drafter.training.dspark_confidence_loss_alpha=0.0
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log=True
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_first_n=2
    actor_rollout_ref.rollout.drafter.training.dspark_debug_log_interval=100
    actor_rollout_ref.rollout.drafter.training.hidden_state_window_tokens_per_sample=512
    actor_rollout_ref.rollout.drafter.training.max_collect_samples_per_step_per_replica=16
    actor_rollout_ref.rollout.drafter.training.max_collect_tokens_per_step_per_replica=16384
    actor_rollout_ref.rollout.drafter.training.old_logprob_hidden_capture_impl=forward_hook
    actor_rollout_ref.rollout.drafter.training.publish_async=True
    actor_rollout_ref.rollout.drafter.training.publish_dtype=bf16
    actor_rollout_ref.rollout.drafter.training.step=10
    actor_rollout_ref.rollout.drafter.training.target_lm_head_row_restricted_sync=False
    actor_rollout_ref.rollout.drafter.training.training_interval_steps=5
)

# ===================================== Algorithm Config =====================================
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ===================================== Ray Config =====================================
ray_num_cpus=64
ray_worker_soft_limit=8

RAY_CONFIG=(
    ray_kwargs.ray_init.num_cpus=${ray_num_cpus}
    +ray_kwargs.ray_init._system_config.prestart_worker_first_driver=false
    +ray_kwargs.ray_init._system_config.num_workers_soft_limit=${ray_worker_soft_limit}
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

# ===================================== Trainer Config =====================================
project_name='GRPO-Qwen3-8B-Drafter-Dspark-BASE-GSM8K-MATH'
exp_name='GRPO-Qwen3-8B-Drafter-Dspark-BASE-FSDP2-vLLM'
ngpus_per_node=8

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger="['console']"
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=6
    trainer.total_training_steps=100
    trainer.val_before_train=False
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
PYTHONUNBUFFERED=1 python3 -m verl_speco.main \
    ${DATA_CONFIG[@]} \
    ${ACTOR_CONFIG[@]} \
    ${REF_CONFIG[@]} \
    ${ROLLOUT_CONFIG[@]} \
    ${ALGORITHM_CONFIG[@]} \
    ${RAY_CONFIG[@]} \
    ${TRAINER_CONFIG[@]} \
