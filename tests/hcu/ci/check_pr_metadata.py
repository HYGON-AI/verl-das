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

ALLOWED_SCOPES = (
    "attention",
    "kv_cache",
    "kernel",
    "comm",
    "runtime",
    "moe",
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
ALLOWED_TYPES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "test",
    "build",
    "ci",
    "chore",
)
TITLE_PATTERN = re.compile(
    rf"^(?P<type>{'|'.join(ALLOWED_TYPES)})"
    r"\((?P<scope>[a-z][a-z0-9_-]*)\): "
    r"(?P<subject>\S.*)$"
)
MAX_SUBJECT_LENGTH = 72


def validate_title(title: str) -> list[str]:
    stripped_title = title.strip()
    match = TITLE_PATTERN.fullmatch(stripped_title)
    if not match:
        return [
            "PR title must match: type(scope): subject; "
            f"type must be one of {', '.join(ALLOWED_TYPES)}"
        ]

    errors = []
    scope = match.group("scope")
    if scope not in ALLOWED_SCOPES:
        errors.append(f"Invalid PR title scope: {scope}")

    subject = match.group("subject")
    if not subject[0].islower():
        errors.append("PR title subject must start with a lowercase letter")
    if len(subject) > MAX_SUBJECT_LENGTH:
        errors.append(
            f"PR title subject must not exceed {MAX_SUBJECT_LENGTH} characters"
        )

    return errors


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
