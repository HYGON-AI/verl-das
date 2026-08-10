#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GITHUB_WORKSPACE:-}" || -z "${RUNNER_TEMP:-}" || -z "${VERL_HCU_CI_IMAGE:-}" ]]; then
    echo "ERROR: GITHUB_WORKSPACE, RUNNER_TEMP, and VERL_HCU_CI_IMAGE are required" >&2
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

if [[ ! "${VERL_HCU_CI_IMAGE}" =~ @sha256:[0-9a-fA-F]{64}$ ]]; then
    echo "ERROR: VERL_HCU_CI_IMAGE must use an immutable sha256 digest" >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is required to restore runner workspace ownership" >&2
    exit 1
fi

docker run --rm \
    --volume "${workspace}:/workspace" \
    "${VERL_HCU_CI_IMAGE}" \
    chown -R -- "${owner}" /workspace
echo "Restored ${workspace} ownership to ${owner}."
