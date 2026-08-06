#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_DIR="${SCRIPT_DIR}/../ci"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export VERL_HCU_CI_RUN_ID="${VERL_HCU_CI_RUN_ID:-vllm-grpo-pr-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$}"
export VERL_HCU_CI_TMP_ROOT="${VERL_HCU_CI_TMP_ROOT:-${TMPDIR:-/tmp}/verl-hcu-ci}"
if [[ ! "${VERL_HCU_CI_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: unsafe VERL_HCU_CI_RUN_ID: ${VERL_HCU_CI_RUN_ID}" >&2
    exit 1
fi
VERL_HCU_CI_TMP_ROOT="$(realpath -m -- "${VERL_HCU_CI_TMP_ROOT}")"
if [[ "${VERL_HCU_CI_TMP_ROOT}" == "/" ]]; then
    echo "ERROR: refusing to use the filesystem root as VERL_HCU_CI_TMP_ROOT" >&2
    exit 1
fi
export VERL_HCU_CI_TMP_ROOT
run_dir="${VERL_HCU_CI_TMP_ROOT}/${VERL_HCU_CI_RUN_ID}"
log_root="${VERL_HCU_CI_LOG_DIR:-${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}}"
training_log="${log_root}/vllm-grpo-1step.log"
mkdir -p "${run_dir}" "${log_root}"
exec > >(tee -a "${training_log}") 2>&1

cleanup() {
    bash "${CI_DIR}/cleanup.sh"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# shellcheck disable=SC1091
source "${CI_DIR}/prepare_workspace.sh"
python3 "${CI_DIR}/check_environment.py" runtime --require-data-roots --require-gpus 8

model_path="${VERL_HCU_MODEL_ROOT}/qwen2.5/Qwen2.5-0.5B-Instruct"
train_file="${VERL_HCU_DATA_ROOT}/gsm8k/train.parquet"
test_file="${VERL_HCU_DATA_ROOT}/gsm8k/test.parquet"
for required_path in "${model_path}" "${train_file}" "${test_file}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required PR E2E input is missing: ${required_path}" >&2
        exit 1
    fi
done

export PYTHONWARNINGS=ignore
export TRANSFORMERS_VERBOSITY=error
export VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libgalaxyhip.so

# Mark Ray as owned before VERL starts it so cancellation cleanup cannot affect
# any process that did not originate from this CI run.
touch "${run_dir}/ray-owned"

# This is the HCU-sized equivalent of upstream's AMD ROCm FSDP/vLLM PR case:
# a real GRPO rollout, reward calculation, backward pass and optimizer step.
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer \
    data.train_files="${train_file}" \
    data.val_files="${test_file}" \
    data.train_batch_size=16 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_model_len=256 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=verl-hcu-pr \
    trainer.experiment_name=qwen2.5-0.5b-grpo-vllm-1step \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.device=cuda \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=1

echo "HCU vLLM GRPO one-step E2E passed."
