#!/usr/bin/env bash
set -euo pipefail

container="${VERL_HCU_CI_CONTAINER_NAME:-}"
workdir=/workspace
env_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --container-name)
            container="$2"
            shift 2
            ;;
        -w|--workdir)
            workdir="$2"
            shift 2
            ;;
        -e|--env)
            env_args+=(--env "$2")
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "ERROR: unknown exec_container option: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! "${container}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: unsafe VERL_HCU_CI_CONTAINER_NAME: ${container}" >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "ERROR: no container command was provided" >&2
    exit 2
fi

docker exec --workdir "${workdir}" "${env_args[@]}" "${container}" "$@"
