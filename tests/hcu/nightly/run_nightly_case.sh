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

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {vllm|sglang}" >&2
    exit 2
fi

case_name="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_DIR="${SCRIPT_DIR}/../ci"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
accelerator="${VERL_HCU_ACCELERATOR:-bw1000}"
if [[ ! "${accelerator}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: unsafe HCU accelerator label: ${accelerator}" >&2
    exit 2
fi
case_dir="${SCRIPT_DIR}/${accelerator}"
export VERL_HCU_CI_RUN_ID="${VERL_HCU_CI_RUN_ID:-nightly-${case_name}-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$}"
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
mkdir -p "${run_dir}/pids" "${log_root}"
log_file="${log_root}/${case_name}.log"
: > "${log_file}"

cleanup() {
    bash "${CI_DIR}/cleanup.sh"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

case "${case_name}" in
    vllm)
        case_script="${case_dir}/run_vllm_grpo_1step.sh"
        ;;
    sglang)
        case_script="${case_dir}/run_sglang_off_policy_1step.sh"
        ;;
    *)
        echo "ERROR: unsupported nightly HCU case '${case_name}'; expected vllm or sglang" >&2
        exit 2
        ;;
esac

set +e
(
    set -euo pipefail

    # shellcheck disable=SC1091
    source "${CI_DIR}/prepare_workspace.sh"
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
for distribution in ("ray", "vllm", "sglang", "transformers"):
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
    bash "${case_script}"
) 2>&1 | tee "${log_file}"
case_status=${PIPESTATUS[0]}
set -e

if [[ ${case_status} -ne 0 ]]; then
    echo "ERROR: ${case_name} one-step case exited with status ${case_status}" |
        tee -a "${log_file}" >&2
    exit "${case_status}"
fi
if ! grep -Fq "[HCU_ADAPT] Patch has been applied in worker" "${log_file}"; then
    echo "ERROR: ${case_name} log is missing the HCU_ADAPT patch marker" |
        tee -a "${log_file}" >&2
    exit 1
fi
if ! grep -Fq "step:1" "${log_file}"; then
    echo "ERROR: ${case_name} log is missing the console logger step:1 marker" |
        tee -a "${log_file}" >&2
    exit 1
fi

failure_pattern='Error executing job|RayTaskError|AcceleratorError|out of memory|(^|[^[:alpha:]])OOM([^[:alpha:]]|$)|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)|WorkerCrashedError|RayActorError|ActorDiedError'
if grep -Eiq "${failure_pattern}" "${log_file}"; then
    echo "ERROR: ${case_name} log contains a fatal job, accelerator, OOM, NaN, or worker failure" |
        tee -a "${log_file}" >&2
    grep -Ein "${failure_pattern}" "${log_file}" | tail -n 20 |
        tee -a "${log_file}" >&2
    exit 1
fi

echo "HCU nightly ${case_name} one-step case passed."
