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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASELINE="${REPO_ROOT}/third_party/verl/verl/experimental/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_sglang_2_6.sh"
MODEL_PATH="${VERL_HCU_MODEL_ROOT:?VERL_HCU_MODEL_ROOT is required}/Qwen3-0.6B"
if [[ ! -e "${MODEL_PATH}" ]]; then
    MODEL_PATH="${VERL_HCU_MODEL_ROOT}/qwen3/Qwen3-0.6B"
fi
TRAIN_FILE="${VERL_HCU_DATA_ROOT:?VERL_HCU_DATA_ROOT is required}/gsm8k/train.parquet"
TEST_FILE="${VERL_HCU_DATA_ROOT}/gsm8k/test.parquet"

for required_path in "${BASELINE}" "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required SGLang one-step input is missing: ${required_path}" >&2
        exit 1
    fi
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export MODEL_PATH
export TRAIN_FILE
export TEST_FILE
export NNODES=1
export NGPUS_PER_NODE=8
export RAY_DATA_HOME="${VERL_HCU_CI_TMP_ROOT:-${TMPDIR:-/tmp}/verl-hcu-ci}/${VERL_HCU_CI_RUN_ID:-local}"

bash "${BASELINE}" \
    "hydra.searchpath=[file://${REPO_ROOT}/third_party/verl/verl/trainer/config]" \
    data.train_batch_size=12 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=64 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=False
