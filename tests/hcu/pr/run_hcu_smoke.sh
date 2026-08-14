#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_DIR="${SCRIPT_DIR}/../ci"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export VERL_HCU_CI_RUN_ID="${VERL_HCU_CI_RUN_ID:-smoke-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$}"
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
if [[ "${log_root}" != /* ]]; then
    log_root="${REPO_ROOT}/${log_root}"
fi
environment_log="${log_root}/environment.log"
mkdir -p "${run_dir}" "${log_root}"
exec > >(tee -a "${environment_log}") 2>&1

trap 'exit 130' INT
trap 'exit 143' TERM

# shellcheck disable=SC1091
source "${CI_DIR}/prepare_workspace.sh"
echo "main_sha=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
git -C "${REPO_ROOT}" submodule status
echo "ci_image=${VERL_HCU_CI_IMAGE:-unknown}"
python3 "${CI_DIR}/check_environment.py" runtime --require-gpus 8

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | sed -n '1,80p'
else
    echo "ERROR: rocminfo is required in the HCU CI container" >&2
    exit 1
fi
if command -v hipconfig >/dev/null 2>&1; then
    hipconfig --version
else
    echo "ERROR: hipconfig is required in the HCU CI container" >&2
    exit 1
fi

python3 - <<'PY'
import importlib.metadata
import sys

import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "torch.cuda must be available in HCU CI"
assert torch.cuda.device_count() == 8, (
    f"expected 8 HCU devices, found {torch.cuda.device_count()}"
)
for distribution in ("ray", "vllm", "sglang", "transformers"):
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"{distribution}={version}")
PY

if ! patch_output="$(python3 -c 'import verl; print("[HCU_SMOKE] verl import completed")' 2>&1)"; then
    printf '%s\n' "${patch_output}" >&2
    echo "ERROR: importing the patched verl package failed" >&2
    exit 1
fi
printf '%s\n' "${patch_output}"
if ! grep -Fq "[HCU_ADAPT] Patch has been applied in worker" <<<"${patch_output}"; then
    echo "ERROR: HCU patch marker was not emitted while importing verl" >&2
    exit 1
fi

python3 - <<'PY'
import verl

from hcu_verl.core.dist_checkpointing.strategies.filesystem_async import (
    preload_tensors,
)
from hcu_verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from hcu_verl.utils.flops_counter import _DEVICE_FLOPS
from hcu_verl.utils.megatron_utils import get_model
from hcu_verl.workers.rollout.vllm_rollout.utils import get_device_uuid
from megatron.core.dist_checkpointing.strategies.filesystem_async import (
    FileSystemWriterAsync,
)
from verl.trainer import constants_ppo
from verl.utils import flops_counter, megatron_utils
from verl.workers.rollout.vllm_rollout import utils as vllm_utils

expected_preload = (
    preload_tensors.__func__
    if isinstance(preload_tensors, staticmethod)
    else preload_tensors
)
checks = (
    (
        "verl.trainer.constants_ppo.PPO_RAY_RUNTIME_ENV",
        constants_ppo.PPO_RAY_RUNTIME_ENV,
        PPO_RAY_RUNTIME_ENV,
    ),
    (
        "verl.utils.flops_counter._DEVICE_FLOPS",
        flops_counter._DEVICE_FLOPS,
        _DEVICE_FLOPS,
    ),
    (
        "verl.utils.megatron_utils.get_model",
        megatron_utils.get_model,
        get_model,
    ),
    (
        "verl.workers.rollout.vllm_rollout.utils.get_device_uuid",
        vllm_utils.get_device_uuid,
        get_device_uuid,
    ),
    (
        "FileSystemWriterAsync.preload_tensors",
        FileSystemWriterAsync.preload_tensors,
        expected_preload,
    ),
)
for name, actual, expected in checks:
    assert actual is expected, f"HCU patch identity mismatch: {name}"
    print(f"patch_identity=ok {name}")
PY

touch "${run_dir}/ray-owned"
python3 - <<'PY'
import ray

ray.init(address="local", include_dashboard=False)
resources = ray.cluster_resources()
assert resources.get("GPU", 0) >= 8, f"Ray sees insufficient GPU resources: {resources}"


@ray.remote(num_gpus=1)
class WorkerProbe:
    def inspect(self):
        import torch

        assert torch.cuda.is_available(), "torch.cuda unavailable in Ray worker"
        assert torch.cuda.device_count() >= 1, "Ray worker has no visible HCU device"
        import verl

        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        gpu_ids = tuple(str(value) for value in accelerator_ids.get("GPU", ()))
        return gpu_ids, "ray-worker-verl-import-ok"


actors = [WorkerProbe.remote() for _ in range(8)]
results = ray.get([actor.inspect.remote() for actor in actors])
assignments = [result[0] for result in results]
assert all(len(assignment) == 1 for assignment in assignments), assignments
assert len(set(assignments)) == 8, f"expected 8 unique Ray GPU assignments: {assignments}"
print(results)
for actor in actors:
    ray.kill(actor)
ray.shutdown()
PY
ray stop --force
echo "HCU smoke checks passed."
