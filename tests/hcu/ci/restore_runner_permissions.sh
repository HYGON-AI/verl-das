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

if [[ -z "${GITHUB_WORKSPACE:-}" || -z "${RUNNER_TEMP:-}" ]]; then
    echo "ERROR: GITHUB_WORKSPACE and RUNNER_TEMP are required" >&2
    exit 1
fi

workspace="$(realpath -m -- "${GITHUB_WORKSPACE}")"
runner_temp="$(realpath -m -- "${RUNNER_TEMP}")"
work_root="$(dirname -- "${runner_temp}")"

case "${workspace}" in
    "${work_root}"/*/*) ;;
    *)
        echo "ERROR: refusing to change ownership outside the runner work root: ${workspace}" >&2
        exit 1
        ;;
esac

owner="$(stat -c '%u:%g' -- "${runner_temp}")"
if [[ ! "${owner}" =~ ^[1-9][0-9]*:[0-9]+$ ]]; then
    echo "ERROR: unsafe runner ownership derived from ${runner_temp}: ${owner}" >&2
    exit 1
fi

chown -R -- "${owner}" "${workspace}"
echo "Restored ${workspace} ownership to ${owner}."
