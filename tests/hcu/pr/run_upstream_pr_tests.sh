#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
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
export VERL_HCU_CI_RUN_ID="${VERL_HCU_CI_RUN_ID:-upstream-pr-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$}"
log_root="${VERL_HCU_CI_LOG_DIR:-${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}}"
if [[ "${log_root}" != /* ]]; then
    log_root="${REPO_ROOT}/${log_root}"
fi
pytest_log="${log_root}/upstream-pr-tests.log"
mkdir -p "${log_root}"

# shellcheck disable=SC1091
source "${CI_DIR}/prepare_workspace.sh"

cd "${REPO_ROOT}/third_party/verl"

# Keep this list pinned and intentionally small enough for every PR. It combines
# upstream's sanity lane with CPU tests covering protocol, PPO math, config and
# reward-manager behavior that is directly affected by the HCU adaptation.
test_paths=(
    tests/special_sanity
    tests/test_base_config_on_cpu.py
    tests/test_protocol_on_cpu.py
    tests/test_protocol_v2_on_cpu.py
    tests/trainer/ppo/test_core_algos_on_cpu.py
    tests/trainer/ppo/test_metric_utils_on_cpu.py
    tests/utils/test_config_on_cpu.py
    tests/utils/test_import_utils_on_cpu.py
    tests/utils/test_padding_on_cpu.py
    tests/utils/test_temp_env_on_cpu.py
    tests/workers/config/test_actor_config_on_cpu.py
    tests/workers/config/test_critic_config_on_cpu.py
    tests/workers/config/test_engine_config_on_cpu.py
    tests/workers/config/test_model_config_on_cpu.py
    tests/workers/config/test_optim_config_on_cpu.py
    tests/workers/reward_manager/test_registry_on_cpu.py
)

# The pinned upstream v0.8.0 case says the model path is not loaded, but its
# OmegaConf.to_object call does load ~/models/Qwen/Qwen2.5-0.5B. Exclude only
# that broken node until the pinned upstream revision contains the fix.
broken_upstream_node="tests/workers/config/test_model_config_on_cpu.py::TestHFModelConfigCPU::test_target_modules_raises_on_invalid_type"

set +e
python3 -m pytest -q "${test_paths[@]}" \
    --deselect "${broken_upstream_node}" \
    2>&1 | tee "${pytest_log}"
pytest_status=${PIPESTATUS[0]}
set -e

if [[ ${pytest_status} -ne 0 ]]; then
    echo "ERROR: curated upstream PR tests exited with status ${pytest_status}" >&2
    exit "${pytest_status}"
fi

echo "Curated upstream PR tests passed."
