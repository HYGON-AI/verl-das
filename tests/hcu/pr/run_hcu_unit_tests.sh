#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_DIR="${SCRIPT_DIR}/../ci"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export VERL_HCU_CI_RUN_ID="${VERL_HCU_CI_RUN_ID:-unit-pr-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$}"
tmp_root="${VERL_HCU_CI_TMP_ROOT:-${TMPDIR:-/tmp}/verl-hcu-ci}"
run_dir="${tmp_root}/${VERL_HCU_CI_RUN_ID}"
log_root="${VERL_HCU_CI_LOG_DIR:-${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}}"
if [[ "${log_root}" != /* ]]; then
    log_root="${REPO_ROOT}/${log_root}"
fi
pytest_log="${log_root}/unit-tests.log"
megatron_log="${log_root}/megatron-kl-loss.log"
mkdir -p "${run_dir}/pids" "${log_root}"
touch "${run_dir}/ray-owned"

# shellcheck disable=SC1091
source "${CI_DIR}/prepare_workspace.sh"
export PYTHONPATH="${REPO_ROOT}/tests${PYTHONPATH:+:${PYTHONPATH}}"
python3 "${CI_DIR}/check_environment.py" runtime --require-gpus 8

cd "${REPO_ROOT}"

set +e
python3 -m pytest -s -x \
    --ignore-glob="*test_special_*.py" \
    tests/ \
    2>&1 | tee "${pytest_log}"
pytest_status=${PIPESTATUS[0]}
set -e
if [[ ${pytest_status} -ne 0 ]]; then
    echo "ERROR: HCU unit tests exited with status ${pytest_status}" >&2
    exit "${pytest_status}"
fi

set +e
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    tests/utils/test_special_megatron_kl_loss_tp.py \
    2>&1 | tee "${megatron_log}"
megatron_status=${PIPESTATUS[0]}
set -e
if [[ ${megatron_status} -ne 0 ]]; then
    echo "ERROR: Megatron KL loss test exited with status ${megatron_status}" >&2
    exit "${megatron_status}"
fi

echo "HCU unit tests passed."
