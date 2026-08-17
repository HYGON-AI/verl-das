#!/usr/bin/env bash
set -euo pipefail

container="${VERL_HCU_CI_CONTAINER_NAME:-}"
if [[ ! "${container}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: unsafe VERL_HCU_CI_CONTAINER_NAME: ${container}" >&2
    exit 1
fi
if ! docker ps --all --format '{{.Names}}' | grep -Fqx -- "${container}"; then
    echo "HCU CI container ${container} is already absent."
    exit 0
fi

host_uid="$(id -u)"
host_gid="$(id -g)"
if [[ ! "${host_uid}:${host_gid}" =~ ^[1-9][0-9]*:[0-9]+$ ]]; then
    echo "ERROR: unsafe runner ownership: ${host_uid}:${host_gid}" >&2
    exit 1
fi

status=0
if docker ps --format '{{.Names}}' | grep -Fqx -- "${container}"; then
    docker exec --workdir /workspace "${container}" \
        bash tests/hcu/ci/cleanup.sh || status=1
    docker exec "${container}" \
        chown -R -- "${host_uid}:${host_gid}" /workspace || status=1
else
    echo "ERROR: HCU CI container ${container} stopped before cleanup" >&2
    status=1
fi
docker rm -f "${container}" >/dev/null || status=1

if [[ ${status} -ne 0 ]]; then
    echo "ERROR: HCU CI container cleanup was incomplete" >&2
    exit "${status}"
fi
echo "Cleaned HCU CI container ${container}."
