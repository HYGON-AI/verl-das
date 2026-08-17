#!/usr/bin/env bash
set -euo pipefail

image="${VERL_HCU_CI_IMAGE:-}"
container="${VERL_HCU_CI_CONTAINER_NAME:-}"
workspace="${GITHUB_WORKSPACE:-}"

if [[ ! "${image}" =~ @sha256:[0-9a-fA-F]{64}$ ]]; then
    echo "ERROR: VERL_HCU_CI_IMAGE must use an immutable sha256 digest" >&2
    exit 1
fi
if [[ ! "${container}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: unsafe VERL_HCU_CI_CONTAINER_NAME: ${container}" >&2
    exit 1
fi
if [[ -z "${workspace}" || ! -d "${workspace}" ]]; then
    echo "ERROR: GITHUB_WORKSPACE must reference an existing directory" >&2
    exit 1
fi
workspace="$(realpath -e -- "${workspace}")"
if [[ "${workspace}" == "/" ]]; then
    echo "ERROR: refusing to mount the filesystem root as the workspace" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is required for HCU CI" >&2
    exit 1
fi
if ! docker image inspect "${image}" >/dev/null 2>&1; then
    docker pull "${image}"
fi

volume_args=(
    --volume "${workspace}:/workspace"
    --volume /opt/hyhal:/opt/hyhal:ro
)
if [[ -n "${VERL_HCU_MODEL_ROOT:-}" ]]; then
    if [[ ! -d "${VERL_HCU_MODEL_ROOT}" ]]; then
        echo "ERROR: VERL_HCU_MODEL_ROOT is not a directory: ${VERL_HCU_MODEL_ROOT}" >&2
        exit 1
    fi
    volume_args+=(--volume "${VERL_HCU_MODEL_ROOT}:${VERL_HCU_MODEL_ROOT}:ro")
fi
if [[ -n "${VERL_HCU_DATA_ROOT:-}" ]]; then
    data_dir="${VERL_HCU_DATA_ROOT}/gsm8k"
    if [[ ! -d "${data_dir}" ]]; then
        echo "ERROR: GSM8K data directory is missing: ${data_dir}" >&2
        exit 1
    fi
    volume_args+=(--volume "${data_dir}:${data_dir}:ro")
fi

env_args=(
    --env GITHUB_WORKSPACE=/workspace
    --env VERL_HCU_CI_IMAGE="${image}"
)
for name in \
    GITHUB_RUN_ID \
    GITHUB_RUN_ATTEMPT \
    VERL_HCU_CI_RUN_ID \
    VERL_HCU_CI_LOG_DIR \
    VERL_HCU_CI_TMP_ROOT \
    VERL_HCU_MODEL_ROOT \
    VERL_HCU_DATA_ROOT \
    HF_HUB_OFFLINE \
    TRANSFORMERS_OFFLINE \
    HF_DATASETS_OFFLINE \
    WANDB_MODE \
    TOKENIZERS_PARALLELISM \
    GLOG_minloglevel \
    PYTHONWARNINGS \
    TORCH_CPP_LOG_LEVEL \
    TRANSFORMERS_VERBOSITY \
    VLLM_CUDART_SO_PATH; do
    if [[ -n "${!name:-}" ]]; then
        env_args+=(--env "${name}=${!name}")
    fi
done

docker rm -f "${container}" >/dev/null 2>&1 || true
docker run --detach \
    --name "${container}" \
    --user root \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size=64g \
    --ipc=host \
    "${volume_args[@]}" \
    "${env_args[@]}" \
    --workdir /workspace \
    "${image}" \
    sleep infinity

docker exec "${container}" git config --global --add safe.directory /workspace
echo "Started HCU CI container ${container}."
