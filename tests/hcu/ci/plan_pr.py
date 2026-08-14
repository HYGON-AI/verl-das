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

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

RUNTIME_PREFIXES = ("hcu_verl/", "tests/hcu/")
RUNTIME_PATHS = {
    ".github/workflows/pr-test-hcu.yml",
    ".gitmodules",
    "third_party/verl",
}


def classify_paths(paths: Iterable[str]) -> dict[str, object]:
    normalized = sorted(
        {path.strip().replace("\\", "/") for path in paths if path.strip()}
    )
    runtime_paths = [
        path
        for path in normalized
        if path in RUNTIME_PATHS or path.startswith(RUNTIME_PREFIXES)
    ]
    python_paths = [
        path
        for path in normalized
        if path.endswith(".py")
        and not path.startswith("examples/")
        and not path.startswith("tests/special_sanity/")
        and not (path.startswith("tests/") and path.endswith("_on_cpu.py"))
    ]

    runtime = bool(runtime_paths)
    unit = runtime or bool(python_paths)
    profile = "runtime" if runtime else "unit" if unit else "none"
    return {
        "profile": profile,
        "runtime": runtime,
        "unit": unit,
        "runtime_paths": runtime_paths,
        "python_paths": python_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify HCU PR test triggers.")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    plan = classify_paths(sys.stdin)
    print(f"HCU PR profile: {plan['profile']}")
    for name in ("runtime_paths", "python_paths"):
        for path in plan[name]:
            print(f"- {name}: {path}")

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"profile={plan['profile']}\n")
            output.write(f"runtime={str(plan['runtime']).lower()}\n")
            output.write(f"unit={str(plan['unit']).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
