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

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

CONFIG_ENV_VARS = (
    "VERL_HCU_CI_RUNNER_LABEL",
    "VERL_HCU_PR_IMAGE",
    "VERL_HCU_VLLM_IMAGE",
    "VERL_HCU_SGLANG_IMAGE",
    "VERL_HCU_MODEL_ROOT",
    "VERL_HCU_DATA_ROOT",
)
IMAGE_ENV_VARS = (
    "VERL_HCU_PR_IMAGE",
    "VERL_HCU_VLLM_IMAGE",
    "VERL_HCU_SGLANG_IMAGE",
)
PR_CONFIG_ENV_VARS = (
    "VERL_HCU_CI_RUNNER_LABEL",
    "VERL_HCU_PR_IMAGE",
)
NIGHTLY_CONFIG_ENV_VARS = (
    "VERL_HCU_CI_RUNNER_LABEL",
    "VERL_HCU_VLLM_IMAGE",
    "VERL_HCU_SGLANG_IMAGE",
    "VERL_HCU_MODEL_ROOT",
    "VERL_HCU_DATA_ROOT",
)
CONFIG_PROFILES = {
    "pr": PR_CONFIG_ENV_VARS,
    "nightly": NIGHTLY_CONFIG_ENV_VARS,
    "all": CONFIG_ENV_VARS,
}
RUNTIME_PATH_ENV_VARS = ("VERL_HCU_MODEL_ROOT", "VERL_HCU_DATA_ROOT")
IMAGE_DIGEST_PATTERN = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")


def visible_device_ids(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def validate_config(
    environment: Mapping[str, str],
    profile: str = "all",
) -> list[str]:
    errors = []
    required_names = CONFIG_PROFILES[profile]

    for name in required_names:
        if not environment.get(name, "").strip():
            errors.append(f"{name} is required")

    for name in IMAGE_ENV_VARS:
        if name not in required_names:
            continue
        image = environment.get(name, "").strip()
        if image and not IMAGE_DIGEST_PATTERN.fullmatch(image):
            errors.append(f"{name} must be pinned with @sha256:<64 hex digits>")

    return errors


def validate_runtime(
    environment: Mapping[str, str],
    required_gpus: int = 0,
    require_data_roots: bool = False,
) -> list[str]:
    errors = []

    if require_data_roots:
        for name in RUNTIME_PATH_ENV_VARS:
            value = environment.get(name, "").strip()
            if not value:
                errors.append(f"{name} is required")
            elif not Path(value).is_dir():
                errors.append(f"{name} must reference an existing directory: {value}")

    if required_gpus < 0:
        errors.append("required_gpus must be zero or greater")
    elif required_gpus:
        visible = environment.get("HIP_VISIBLE_DEVICES", "").strip()
        if visible:
            device_ids = visible_device_ids(visible)
            count = len(device_ids)
            if count != required_gpus:
                errors.append(
                    f"Expected {required_gpus} HCU devices in HIP_VISIBLE_DEVICES, "
                    f"found {count}"
                )
            elif len(set(device_ids)) != count:
                errors.append("HIP_VISIBLE_DEVICES must contain unique HCU device IDs")

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the verl-das HCU CI environment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser(
        "config",
        help="validate repository-level HCU CI variables",
    )
    config_parser.add_argument(
        "--profile",
        choices=tuple(CONFIG_PROFILES),
        default="all",
        help="validate only the variables required by this CI workflow",
    )
    runtime_parser = subparsers.add_parser(
        "runtime",
        help="validate mounted paths and optional HCU visibility",
    )
    runtime_parser.add_argument(
        "--require-data-roots",
        action="store_true",
        help="require existing VERL_HCU_MODEL_ROOT and VERL_HCU_DATA_ROOT directories",
    )
    runtime_parser.add_argument(
        "--require-gpus",
        "--required-gpus",
        dest="required_gpus",
        type=int,
        default=0,
        metavar="COUNT",
        help="require exactly COUNT comma-separated HIP_VISIBLE_DEVICES entries",
    )
    runtime_parser.add_argument(
        "--require-8-gpus",
        action="store_const",
        const=8,
        dest="required_gpus",
        help="require exactly eight visible HCU devices",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "config":
        errors = validate_config(os.environ, profile=args.profile)
    else:
        errors = validate_runtime(
            os.environ,
            required_gpus=args.required_gpus,
            require_data_roots=args.require_data_roots,
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"HCU CI {args.command} validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
