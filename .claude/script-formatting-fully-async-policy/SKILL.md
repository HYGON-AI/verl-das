---
name: script-formatting-fully-async-policy
description: Convention for formatting verl fully-async-policy shell scripts — parameter grouping, bash arrays, section headers
---

**Why:** User wants consistent, readable scripts with logical grouping. Messy flat parameter lists are hard to maintain.

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

3. **Section headers**: Group by category using `# ===================================== Name =====================================`. Standard categories:
   - Data Config (files, lengths, batch sizes)
   - Actor Model & Optim Config (LR, megatron/fsdp, KL, dynamic bsz)
   - Ref Config (log prob settings, megatron/fsdp for ref)
   - Rollout Config (generation params: n, temperature, top_p, top_k, GPU memory, TP)
   - Algorithm Config (adv_estimator, KL settings)
   - Reward Config (reward_manager, reward_kwargs)
   - Critic Config (strategy)
   - Trainer Config (project_name, exp_name, n_gpus_training, n_gpus_rollout, logger, epochs, nnodes, n_gpus, default_local_dir)
   - Async Training Config (staleness, trigger_sync, partial_rollout)
   - NCCL Config (nccl_timeout — standalone params that don't fit other sections)
   - Profiler Config (torch profiler settings)

4. **Bash arrays**: Each section's params go in a `*_CONFIG=(...)` array. One param per line, 4-space indent. Variables referenced with `${var}` syntax. Hydra `+` prefix preserved for overrides.

5. **Final command**: `python3 -m verl.trainer.main_ppo` with `--config-path`, `--config-name`, `hydra.searchpath`, then `${ARRAY[@]}` expansions. **Every line including the last one MUST end with `\`**, and there must be no blank line after the command. The `hydra.searchpath=[file://${VERL_PATH}/third_party/verl/verl/trainer/config]` line MUST be included as the first argument after `--config-name`.
The `--config-name` depends on the training backend:
   - fsdp/fsdp2: `--config-name=fully_async_ppo_trainer`
   - megatron: `--config-name=fully_async_ppo_megatron_trainer`

   ```
   # Main Fully_Async_Policy Training Command
   python3 -m verl.experimental.fully_async_policy.fully_async_main \
       --config-path=config \
       --config-name=fully_async_ppo_trainer \
       hydra.searchpath=[file://${VERL_PATH}/third_party/verl/verl/trainer/config] \
       ${DATA_CONFIG[@]} \
       ...
       ${TRAINER_CONFIG[@]} \
   ```

6. **Naming conventions**:
   - Array names: UPPERCASE with `_CONFIG` suffix (e.g., `DATA_CONFIG`, `ACTOR_CONFIG`)
   - Variable names: snake_case (e.g., `train_prompt_bsz`, `n_resp_per_prompt`)
   - Path variables end in `_path` (e.g., `train_path`, `hf_model_path`)

7. **Path prefixes**: The following variables MUST use path prefixes from the preamble:
   - `train_file` and `test_file` MUST be prefixed with `${data_path}/`:
     ```
     train_file=${data_path}/dapo-math-17k.parquet
     test_file=${data_path}/aime-2024.parquet
     ```
   - `actor_rollout_ref.model.path` MUST directly reference `${hf_model_path}/...` without an intermediate variable:
     ```
     actor_rollout_ref.model.path=${hf_model_path}/Qwen3-30B-A3B
     ```
   - `trainer.default_local_dir` MUST use `${save_ckpt_path}/ckpts/${project_name}/${exp_name}`:
     ```
     trainer.default_local_dir=${save_ckpt_path}/ckpts/${project_name}/${exp_name}
     ```

8. **Project and experiment names**: Every script MUST define `project_name` and `exp_name`. If the original script doesn't have them, auto-generate following this pattern:

   - **project_name**: `Fully_Async_Policy-<ModelName>-BASE-<Dataset>`
     - `<ModelName>` is the original HuggingFace model name as it appears in the model path (e.g., `Qwen2.5-0.5B-Instruct`). Do NOT strip suffixes like `-Instruct`, `-Chat`, `-Base`.
     - `<Dataset>` is identified from the data file parameters (`train_file` / `TEST_FILE` / `TRAIN_FILE`). Extract the dataset name from the path (e.g., `${data_path}/gsm8k/train.parquet` → `GSM8K`, `${data_path}/dapo-math-17k.parquet` → `DAPO-Math-17k`). If multiple datasets are used, join them with `-` (e.g., `GSM8K-MATH`).
   - **exp_name**: `Fully_Async_Policy-<ModelName>-BASE-<TrainBackend>-<InferBackend>`
     - `<ModelName>`: same as above.
     - `<TrainBackend>`: identified from the script filename. Check the filename for these keywords (case-insensitive, check `fsdp2` before `fsdp`):
       - contains `megatron` → `Megatron`
       - contains `fsdp2` → `FSDP2`
       - contains `fsdp` (but not `fsdp2`) → `FSDP`
     - `<InferBackend>`: identified from the `rollout.name` parameter:
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
       global_profiler.save_path=${VERL_PATH}/examples/fully_async_policy_trainer/torch_prof
   )

   # Conditionally Add Torch Profiling Configuration
   if [[ $profiling == "torch" ]]; then
       TRAINER_CONFIG+=(${PROFILE_CONFIG[@]})
   fi
   ```

**How to apply:** When given a messy verl script, extract all parameters, categorize them, pull out literals as variables, and rewrite in this grouped array format.
