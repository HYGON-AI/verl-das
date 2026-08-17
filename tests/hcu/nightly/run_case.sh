#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_DIR="${SCRIPT_DIR}/../ci"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

for name in \
    CASE_ENGINE \
    CASE_EXPECTED_STEP \
    CASE_ID \
    CASE_MODEL_PARENT \
    CASE_MODEL_PATH \
    CASE_REQUIRED_GPUS \
    CASE_REQUIRES_GALAXYHIP \
    CASE_SCRIPT; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: ${name} is required for an HCU nightly case" >&2
        exit 1
    fi
done
if [[ ! "${CASE_ID}" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "ERROR: unsafe nightly case id: ${CASE_ID}" >&2
    exit 1
fi
if [[ ! "${CASE_REQUIRED_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid HCU count: ${CASE_REQUIRED_GPUS}" >&2
    exit 1
fi
if [[ ! "${CASE_EXPECTED_STEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid expected training step: ${CASE_EXPECTED_STEP}" >&2
    exit 1
fi
case "${CASE_ENGINE}:${CASE_REQUIRES_GALAXYHIP}" in
    vllm:true|sglang:false) ;;
    *)
        echo "ERROR: unsupported nightly engine configuration" >&2
        exit 1
        ;;
esac

case_script="$(realpath -e -- "${REPO_ROOT}/${CASE_SCRIPT}")"
nightly_root="$(realpath -e -- "${SCRIPT_DIR}/bw1000")"
case "${case_script}" in
    "${nightly_root}"/*.sh) ;;
    *)
        echo "ERROR: nightly script is outside ${nightly_root}: ${case_script}" >&2
        exit 1
        ;;
esac

run_dir="${VERL_HCU_CI_TMP_ROOT}/${VERL_HCU_CI_RUN_ID}"
log_file="${VERL_HCU_CI_LOG_DIR}/${CASE_ID}.log"
mkdir -p "${run_dir}/pids" "${VERL_HCU_CI_LOG_DIR}"
touch "${run_dir}/ray-owned"

# shellcheck disable=SC1091
source "${CI_DIR}/prepare_workspace.sh"
python3 "${CI_DIR}/check_environment.py" \
    runtime --require-data-roots --require-gpus "${CASE_REQUIRED_GPUS}"
test -s "${VERL_HCU_DATA_ROOT}/gsm8k/train.parquet"
test -s "${VERL_HCU_DATA_ROOT}/gsm8k/test.parquet"
test -d "${VERL_HCU_MODEL_ROOT}/${CASE_MODEL_PATH}"
if [[ "${CASE_REQUIRES_GALAXYHIP}" == "true" ]]; then
    test -f /opt/dtk/hip/lib/libgalaxyhip.so
fi

echo "main_sha=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git -C "${REPO_ROOT}" submodule status
echo "ci_image=${VERL_HCU_CI_IMAGE}"
python3 - <<'PY'
import importlib.metadata
import os
import sys

import torch

engine = os.environ["CASE_ENGINE"]
required_gpus = int(os.environ["CASE_REQUIRED_GPUS"])
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
for distribution in ("ray", engine, "transformers"):
    print(f"{distribution}={importlib.metadata.version(distribution)}")
assert torch.cuda.is_available()
assert torch.cuda.device_count() == required_gpus
PY

hf_model_path="${VERL_HCU_MODEL_ROOT}/${CASE_MODEL_PARENT}" \
    bash "${case_script}" 2>&1 | tee "${log_file}"
python3 "${CI_DIR}/check_nightly_result.py" \
    --log "${log_file}" \
    --expected-step "${CASE_EXPECTED_STEP}" \
    --checkpoint-root "${GITHUB_WORKSPACE}/checkpoints"
