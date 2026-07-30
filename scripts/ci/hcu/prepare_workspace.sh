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

prepare_hcu_workspace() {
    local script_dir repo_root patch_source patch_target joined_python_path patch_tmp
    local -a python_paths

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "${script_dir}/../../.." && pwd)"
    patch_source="${repo_root}/hcu_verl/patch_init.py"
    patch_target="${repo_root}/third_party/verl/verl/__init__.py"

    for safe_path in \
        "${repo_root}" \
        "${repo_root}/third_party/verl" \
        "${repo_root}/third_party/Megatron-LM" \
        "${repo_root}/third_party/VeOmni"; do
        if ! git config --global --get-all safe.directory 2>/dev/null |
            grep -Fqx -- "${safe_path}"; then
            git config --global --add safe.directory "${safe_path}"
        fi
    done

    python3 "${script_dir}/verify_submodules.py" --repo-root "${repo_root}"

    python_paths=(
        "${repo_root}"
        "${repo_root}/third_party/verl"
        "${repo_root}/third_party/Megatron-LM"
        "${repo_root}/third_party/VeOmni"
    )
    joined_python_path="$(IFS=:; printf '%s' "${python_paths[*]}")"
    export PYTHONPATH="${joined_python_path}${PYTHONPATH:+:${PYTHONPATH}}"
    export VERL_PATH="${repo_root}"

    if [[ ! -f "${patch_source}" ]]; then
        echo "ERROR: HCU patch source is missing: ${patch_source}" >&2
        return 1
    fi
    if [[ ! -f "${patch_target}" ]]; then
        echo "ERROR: verl patch target is missing: ${patch_target}" >&2
        return 1
    fi

    if ! cmp -s "${patch_source}" "${patch_target}"; then
        patch_tmp="${patch_target}.hcu-ci.$$"
        if ! cp -- "${patch_source}" "${patch_tmp}"; then
            rm -f -- "${patch_tmp}"
            echo "ERROR: failed to stage HCU patch at ${patch_tmp}" >&2
            return 1
        fi
        if ! chmod --reference="${patch_target}" "${patch_tmp}"; then
            rm -f -- "${patch_tmp}"
            echo "ERROR: failed to preserve mode for HCU patch target" >&2
            return 1
        fi
        if ! mv -f -- "${patch_tmp}" "${patch_target}"; then
            rm -f -- "${patch_tmp}"
            echo "ERROR: failed to install HCU patch target" >&2
            return 1
        fi
    fi

    if ! cmp -s "${patch_source}" "${patch_target}"; then
        echo "ERROR: HCU patch injection verification failed: ${patch_target}" >&2
        return 1
    fi
    if ! grep -Fq "from hcu_verl import verl_adaptor" "${patch_target}"; then
        echo "ERROR: HCU patch target does not load hcu_verl.verl_adaptor" >&2
        return 1
    fi

    echo "HCU workspace prepared at ${repo_root}"
}

prepare_hcu_workspace
