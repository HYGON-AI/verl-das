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
CI_DIR="${REPO_ROOT}/tests/hcu/ci"
MODEL_PATH="${VERL_HCU_MODEL_ROOT:?VERL_HCU_MODEL_ROOT is required}/Qwen3-0.6B"
if [[ ! -e "${MODEL_PATH}" ]]; then
    MODEL_PATH="${VERL_HCU_MODEL_ROOT}/qwen3/Qwen3-0.6B"
fi
SOURCE_TRAIN_FILE="${VERL_HCU_DATA_ROOT:?VERL_HCU_DATA_ROOT is required}/gsm8k/train.parquet"
SOURCE_TEST_FILE="${VERL_HCU_DATA_ROOT}/gsm8k/test.parquet"

for required_path in "${MODEL_PATH}" "${SOURCE_TRAIN_FILE}" "${SOURCE_TEST_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required SGLang 5-step input is missing: ${required_path}" >&2
        exit 1
    fi
done

export VERL_HCU_CI_RUN_ID="${VERL_HCU_CI_RUN_ID:-nightly-sglang-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$}"
export VERL_HCU_CI_TMP_ROOT="${VERL_HCU_CI_TMP_ROOT:-${TMPDIR:-/tmp}/verl-hcu-ci}"
VERL_HCU_CI_TMP_ROOT="$(realpath -m -- "${VERL_HCU_CI_TMP_ROOT}")"
if [[ "${VERL_HCU_CI_TMP_ROOT}" == "/" ]]; then
    echo "ERROR: refusing to use the filesystem root as VERL_HCU_CI_TMP_ROOT" >&2
    exit 1
fi
export VERL_HCU_CI_TMP_ROOT

run_dir="${VERL_HCU_CI_TMP_ROOT}/${VERL_HCU_CI_RUN_ID}"
data_dir="${run_dir}/data"
TRAIN_FILE="${data_dir}/train.parquet"
TEST_FILE="${data_dir}/test.parquet"
log_root="${VERL_HCU_CI_LOG_DIR:-${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}}"
log_file="${log_root}/sglang.log"
mkdir -p "${run_dir}/pids" "${log_root}"
: > "${log_file}"

cleanup() {
    bash "${CI_DIR}/cleanup.sh"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set +e
(
    set -euo pipefail

    # shellcheck disable=SC1091
    source "${CI_DIR}/prepare_workspace.sh"
    python3 "${REPO_ROOT}/tests/hcu/nightly/prepare_gsm8k_data.py" \
        --train-input "${SOURCE_TRAIN_FILE}" \
        --test-input "${SOURCE_TEST_FILE}" \
        --output-dir "${data_dir}" \
        --min-train-rows 3456 \
        --min-test-rows 8
    echo "main_sha=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    git -C "${REPO_ROOT}" submodule status
    echo "ci_image=${VERL_HCU_CI_IMAGE:-unknown}"
    python3 "${CI_DIR}/check_environment.py" runtime --require-data-roots --require-gpus 8
    python3 - <<'PY'
import importlib.metadata
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
for distribution in ("ray", "sglang", "transformers"):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"{distribution}={version}")
assert torch.cuda.is_available(), "torch.cuda must be available in HCU CI"
assert torch.cuda.device_count() == 8, (
    f"expected 8 HCU devices, found {torch.cuda.device_count()}"
)
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
PY

    touch "${run_dir}/ray-owned"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export WANDB_MODE=disabled
    export TOKENIZERS_PARALLELISM=false
    export PYTHONWARNINGS=ignore
    export TRANSFORMERS_VERBOSITY=error
    export RAY_DATA_HOME="${run_dir}/ray-data"

    python3 -m verl.experimental.one_step_off_policy.main_ppo \
        --config-path=config \
        --config-name=one_step_off_ppo_trainer \
        "hydra.searchpath=[file://${REPO_ROOT}/third_party/verl/verl/trainer/config]" \
        "data.train_files=${TRAIN_FILE}" \
        "data.val_files=${TEST_FILE}" \
        data.train_batch_size=1152 \
        data.max_prompt_length=512 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        "actor_rollout_ref.model.path=${MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.hybrid_engine=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=192 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.layered_summon=True \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        +actor_rollout_ref.rollout.enable_sleep_mode=True \
        +actor_rollout_ref.rollout.engine_kwargs.sglang.page_size=64 \
        +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3 \
        +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=fa3 \
        +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=False \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        critic.strategy=fsdp2 \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        'trainer.logger=["console"]' \
        trainer.project_name=One_Step_Off_Policy-Qwen3-0.6B-BASE-GSM8K \
        trainer.experiment_name=One_Step_Off_Policy-Qwen3-0.6B-BASE-FSDP2-SGLANG \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=2 \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=6 \
        "trainer.default_local_dir=${run_dir}/output/ckpts/One_Step_Off_Policy-Qwen3-0.6B-BASE-GSM8K/One_Step_Off_Policy-Qwen3-0.6B-BASE-FSDP2-SGLANG" \
        rollout.nnodes=1 \
        rollout.n_gpus_per_node=2 \
        trainer.total_training_steps=5
) 2>&1 | tee "${log_file}"
case_status=${PIPESTATUS[0]}
set -e

if [[ ${case_status} -ne 0 ]]; then
    echo "ERROR: SGLang 5-step case exited with status ${case_status}" |
        tee -a "${log_file}" >&2
    exit "${case_status}"
fi
if ! grep -Fq "[HCU_ADAPT] Patch has been applied in worker" "${log_file}"; then
    echo "ERROR: SGLang log is missing the HCU_ADAPT patch marker" |
        tee -a "${log_file}" >&2
    exit 1
fi
if ! grep -Fq "step:5" "${log_file}"; then
    echo "ERROR: SGLang log is missing the console logger step:5 marker" |
        tee -a "${log_file}" >&2
    exit 1
fi

failure_pattern='Error executing job|RayTaskError|AcceleratorError|out of memory|(^|[^[:alpha:]])OOM([^[:alpha:]]|$)|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)|WorkerCrashedError|RayActorError|ActorDiedError'
if grep -Eiq "${failure_pattern}" "${log_file}"; then
    echo "ERROR: SGLang log contains a fatal job, accelerator, OOM, NaN, or worker failure" |
        tee -a "${log_file}" >&2
    grep -Ein "${failure_pattern}" "${log_file}" | tail -n 20 |
        tee -a "${log_file}" >&2
    exit 1
fi
if find "${run_dir}" -type d -name 'global_step_*' -print -quit | grep -q .; then
    echo "ERROR: SGLang 5-step case unexpectedly saved a checkpoint" |
        tee -a "${log_file}" >&2
    exit 1
fi

echo "HCU nightly SGLang 5-step case passed."
