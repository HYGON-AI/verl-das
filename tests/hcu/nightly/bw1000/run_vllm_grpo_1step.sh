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
BASELINE="${REPO_ROOT}/third_party/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh"
MODEL_PATH="${VERL_HCU_MODEL_ROOT:?VERL_HCU_MODEL_ROOT is required}/Qwen2.5-0.5B-Instruct"
if [[ ! -e "${MODEL_PATH}" ]]; then
    MODEL_PATH="${VERL_HCU_MODEL_ROOT}/qwen2.5/Qwen2.5-0.5B-Instruct"
fi
TRAIN_FILE="${VERL_HCU_DATA_ROOT:?VERL_HCU_DATA_ROOT is required}/gsm8k/train.parquet"
TEST_FILE="${VERL_HCU_DATA_ROOT}/gsm8k/test.parquet"

for required_path in "${BASELINE}" "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required vLLM one-step input is missing: ${required_path}" >&2
        exit 1
    fi
done

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export DEVICE=gpu
export INFER_BACKEND=vllm
export MODEL_PATH
export NNODES=1
export NGPUS_PER_NODE=8
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=256
export ROLLOUT_TP=1
export ROLLOUT_N=2
export SAVE_FREQ=-1
export TEST_FREQ=-1
export TOTAL_EPOCHS=1
export PROJECT_NAME=verl-hcu-ci
export EXPERIMENT_NAME=qwen2.5-0.5b-grpo-vllm-1step

bash "${BASELINE}" \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${TEST_FILE}" \
    data.train_batch_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.rollout.n=2 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable
