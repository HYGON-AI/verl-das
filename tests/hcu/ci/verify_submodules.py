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
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

SUBMODULE_PATHS = (
    "third_party/verl",
    "third_party/Megatron-LM",
    "third_party/VeOmni",
)


def parse_gitlink_sha(output: str) -> str | None:
    fields = output.strip().split()
    if len(fields) < 3 or fields[0] != "160000" or fields[1] != "commit":
        return None
    return fields[2]


def verify_submodules(
    repository_root: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    root = repository_root.resolve()
    errors = []

    for path in SUBMODULE_PATHS:
        gitlink_result = run(
            ["git", "-C", str(root), "ls-tree", "HEAD", "--", path],
            capture_output=True,
            text=True,
            check=False,
        )
        gitlink_sha = (
            parse_gitlink_sha(gitlink_result.stdout)
            if gitlink_result.returncode == 0
            else None
        )
        if gitlink_sha is None:
            errors.append(f"{path} gitlink is missing or invalid")
            continue

        checkout_root = root / path
        if not (checkout_root / ".git").exists():
            errors.append(f"{path} checkout is not initialized")
            continue

        checkout_result = run(
            ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        checkout_sha = (
            checkout_result.stdout.strip() if checkout_result.returncode == 0 else None
        )
        if checkout_sha != gitlink_sha:
            actual = checkout_sha or "missing"
            errors.append(
                f"{path} checkout SHA mismatch: expected {gitlink_sha}, got {actual}"
            )

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify pinned HCU CI submodules.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repository root containing third_party submodules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = verify_submodules(args.repo_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Pinned HCU CI submodules are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
