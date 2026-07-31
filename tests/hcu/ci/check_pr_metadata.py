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

import os
import re
import sys
from collections.abc import Sequence

ALLOWED_MODULES = (
    "fsdp",
    "megatron",
    "veomni",
    "sglang",
    "vllm",
    "trtllm",
    "rollout",
    "trainer",
    "tests",
    "training_utils",
    "recipe",
    "hardware",
    "deployment",
    "ray",
    "worker",
    "single_controller",
    "misc",
    "docker",
    "ci",
    "perf",
    "model",
    "algo",
    "env",
    "tool",
    "ckpt",
    "doc",
    "data",
    "cfg",
    "reward",
    "fully_async",
    "one_step_off",
    "hcu",
)
ALLOWED_TYPES = ("feat", "fix", "refactor", "chore", "test")
PROGRESS_PATTERN = r"(?:\[(?P<part>[1-9]\d*)/(?P<total>[1-9]\d*|N)\]\s*)?"
BREAKING_PATTERN = r"(?:\[BREAKING\]\s*)?"
TITLE_PATTERN = re.compile(
    rf"^{PROGRESS_PATTERN}{BREAKING_PATTERN}"
    r"\[(?P<modules>[a-z_]+(?:\s*,\s*[a-z_]+)*)\]\s+"
    rf"(?P<type>{'|'.join(ALLOWED_TYPES)}):\s*(?P<description>\S.*)$",
    re.IGNORECASE,
)


def validate_title(title: str) -> list[str]:
    stripped_title = title.strip()
    match = TITLE_PATTERN.fullmatch(stripped_title)
    if not match:
        return [
            (
                "PR title must match: optional [1/N], optional [BREAKING], "
                "[module[,module]] type: description"
            )
        ]

    modules = tuple(
        module.strip().lower() for module in match.group("modules").split(",")
    )
    invalid_modules = [module for module in modules if module not in ALLOWED_MODULES]
    if invalid_modules:
        return [
            "Invalid PR title module(s): " + ", ".join(sorted(set(invalid_modules)))
        ]

    part = match.group("part")
    total = match.group("total")
    if part and total and total.upper() != "N" and int(part) > int(total):
        return [f"Invalid PR title progress prefix: part {part} exceeds total {total}"]

    return []


def validate_metadata(title: str, body: str) -> list[str]:
    errors = validate_title(title)
    if not body.strip():
        errors.append("PR description must not be empty")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    errors = validate_metadata(
        os.environ.get("PR_TITLE", ""),
        os.environ.get("PR_BODY", ""),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("PR metadata validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
