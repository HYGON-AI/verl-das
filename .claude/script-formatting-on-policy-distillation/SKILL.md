---
name: script-formatting-on-policy-distillation
description: Convention for formatting on-policy distillation trainer shell scripts — parameter grouping, bash arrays, section headers
---

**Why:** User wants consistent, readable on-policy distillation training scripts with logical grouping. Messy flat parameter lists are hard to maintain. This trainer extends GRPO with a teacher model for knowledge distillation.

**Rules:**

1. **Standard preamble**: Every script MUST start with this exact preamble block after the shebang and `set -xeuo pipefail` (if used). 

   ```bash
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
   ```

   If the original script has a rollout mode block, preserve it right after `NNODES`:
   ```bash
   rollout_mode="async"
   rollout_name="vllm"  # sglang or vllm
   return_raw_chat="False"
   if [ "$rollout_mode" = "async" ]; then
       export VLLM_USE_V1=1
       return_raw_chat="True"
   fi
   ```

2. **Variable extraction**: Every literal value becomes a variable defined inside its corresponding config section (under the `=====` header, before the array). Compute derived values (e.g., `$((a + b))`) as variables too. No variable definitions should appear before the first `=====` section — only the preamble and `NNODES` stay at the top. If the original script has a rollout mode block, preserve it after `NNODES` as-is.

   Key distillation-specific variables to extract:
   - `teacher_world_size` — number of GPUs for teacher inference (defaults to a fraction of `ngpus_per_node`)
   - `teacher_tp` — tensor parallelism for teacher model
   - `teacher_gpu_mem_util` — GPU memory utilization for teacher vLLM
   - `distillation_loss_mode` — loss mode (e.g., `k1`, `k2`, `k3`, `kl`)
   - `distillation_topk` — top-K for distillation loss
   - `use_policy_gradient` — whether to use policy gradient (`True`/`False`)

3. **Section headers**: Group by category using `# ===================================== Name =====================================`. Standard categories:
   - Data Config (files, lengths, batch sizes)
   - Actor Model & Optim Config (LR, megatron/fsdp, KL, dynamic bsz)
   - Ref Config (log prob settings, megatron/fsdp for ref)
   - Rollout Config (generation params: n, temperature, top_p, top_k, GPU memory, TP)
   - Algorithm Config (adv_estimator, KL settings)
   - **Distillation Config** (teacher model path, inference backend, TP, GPU memory, max_model_len, loss mode, topk, policy gradient, sleep mode)
   - Reward Config (reward_manager, reward_kwargs)
   - Critic Config (strategy)
   - Trainer Config (project_name, exp_name, ngpus_per_node, logger, epochs, nnodes, n_gpus, default_local_dir)
   - NCCL Config (nccl_timeout — standalone params that don't fit other sections)
   - Profiler Config (torch profiler settings)

   The **Distillation Config** section MUST be placed between Algorithm Config and Trainer Config. It is the defining characteristic of on-policy distillation scripts.

4. **Bash arrays**: Each section's params go in a `*_CONFIG=(...)` array. One param per line, 4-space indent. Variables referenced with `${var}` syntax. Hydra `+` prefix preserved for overrides.

   The distillation array MUST be named `DISTILL_CONFIG`.

5. **Final command**: `python3 -m verl.trainer.main_ppo` with `--config-path`, `--config-name`, then `${ARRAY[@]}` expansions. **Every line including the last one MUST end with `\`**, and there must be no blank line after the command. On-policy distillation scripts do NOT include `hydra.searchpath`.
The `--config-name` depends on the training backend:
   - fsdp/fsdp2: `--config-name=ppo_trainer`
   - megatron: `--config-name=ppo_megatron_trainer`

   ```
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
   ```

   `DISTILL_CONFIG` MUST come after `ALGORITHM_CONFIG` and before `TRAINER_CONFIG`.

6. **Naming conventions**:
   - Array names: UPPERCASE with `_CONFIG` suffix (e.g., `DATA_CONFIG`, `ACTOR_CONFIG`, `DISTILL_CONFIG`)
   - Variable names: snake_case (e.g., `train_prompt_bsz`, `n_resp_per_prompt`, `teacher_world_size`)
   - Path variables end in `_path` (e.g., `train_path`, `hf_model_path`)

7. **Path prefixes**: The following variables MUST use path prefixes from the preamble:
   - `train_file` and `test_file` MUST be prefixed with `${data_path}/`:
     ```
     train_file=${data_path}/gsm8k/train.parquet
     test_file=${data_path}/gsm8k/test.parquet
     ```
   - **Student model**: `actor_rollout_ref.model.path` MUST directly reference `${hf_model_path}/...` without an intermediate variable:
     ```
     actor_rollout_ref.model.path=${hf_model_path}/Qwen3-8B
     ```
   - **Teacher model**: `distillation.teacher_models.teacher_model.model_path` MUST directly reference `${hf_model_path}/...` without an intermediate variable:
     ```
     distillation.teacher_models.teacher_model.model_path=${hf_model_path}/Qwen3-32B
     ```
   - `trainer.default_local_dir` MUST use `${save_ckpt_path}/ckpts/${project_name}/${exp_name}`:
     ```
     trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
     ```

8. **Project and experiment names**: Every script MUST define `project_name` and `exp_name`. If the original script doesn't have them, auto-generate following this pattern:

   - **project_name**: `DISTILLATION-<ModelName>-BASE-<Dataset>`
     - `<ModelName>` is the student model name as it appears in the model path (e.g., `Qwen3-8B`). Do NOT strip suffixes like `-Instruct`, `-Chat`, `-Base`.
     - `<Dataset>` is identified from the data file parameters (`train_file` / `TEST_FILE` / `TRAIN_FILE`). Extract the dataset name from the path (e.g., `${data_path}/gsm8k/train.parquet` → `GSM8K`, `${data_path}/dapo-math-17k.parquet` → `DAPO-Math-17k`). If multiple datasets are used, join them with `-` (e.g., `GSM8K-MATH`).
   - **exp_name**: `DISTILLATION-<ModelName>-BASE-<TrainBackend>-<InferBackend>`
     - `<ModelName>`: same as above.
     - `<TrainBackend>`: identified from the script filename. Check the filename for these keywords (case-insensitive, check `fsdp2` before `fsdp`):
       - contains `megatron` → `Megatron`
       - contains `fsdp2` → `FSDP2`
       - contains `fsdp` (but not `fsdp2`) → `FSDP`
     - `<InferBackend>`: identified from the `rollout.name` parameter (student rollout):
       - `rollout.name=vllm` → `vLLM` (lowercase `v`, uppercase `LLM`)
       - `rollout.name=sglang` → `SGLANG`

   These go in the Trainer section before `TRAINER_CONFIG`.

9. **Standard Profiler Config**: The following block MUST be the last section before the python command. Place it after all other config arrays but before the final python invocation:

   ```
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
   ```

   Note: `global_profiler.save_path` uses `on_policy_distillation_trainer`, NOT `grpo_trainer`.

10. **Distillation Config parameters**: The `DISTILL_CONFIG` array MUST include the following groups of parameters. Within each group, order should be preserved:

    **Top-level distillation control:**
    ```
    distillation.enabled=True
    distillation.n_gpus_per_node=${teacher_world_size}
    distillation.nnodes=${NNODES}
    ```

    **Teacher model definition:**
    ```
    distillation.teacher_models.teacher_model.model_path=${hf_model_path}/Qwen3-32B
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    ```

    **Distillation loss settings:**
    ```
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    ```

    **Teacher sleep mode (if `enable_sleep` is defined in Rollout Config):**
    ```
    distillation.teacher_models.teacher_model.inference.free_cache_engine=${enable_sleep}
    +distillation.teacher_models.teacher_model.inference.enable_sleep_mode=${enable_sleep}
    ```

    The teacher inference backend (`teacher_model.inference.name`) and student rollout backend (`rollout.name`) are independent — they can differ (e.g., student uses vLLM while teacher uses SGLANG). When both use the same backend, their sleep mode configurations should match.

    Variables shared across sections:
    - `${max_num_tokens}` (computed in Data Config) is reused for both student `rollout.max_model_len` and teacher `inference.max_model_len`
    - `${enable_sleep}` (defined in Rollout Config) is reused for both student `rollout.free_cache_engine`/`enable_sleep_mode` and teacher `inference.free_cache_engine`/`enable_sleep_mode`
    - `${ppo_max_token_len_per_gpu}` (defined in Actor Config) may be reused for `rollout.log_prob_max_token_len_per_gpu`

**How to apply:** When given a messy verl on-policy distillation script, extract all parameters, categorize them, pull out literals as variables, and rewrite in this grouped array format. Pay special attention to the Distillation Config section — it is the defining feature that distinguishes these scripts from standard GRPO scripts.
