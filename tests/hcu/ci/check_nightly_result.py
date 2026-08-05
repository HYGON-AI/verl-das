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
import re
from pathlib import Path

PATCH_MARKER = "[HCU_ADAPT] Patch has been applied in worker"
FATAL_PATTERN = re.compile(
    r"Error executing job|RayTaskError|AcceleratorError|out of memory|"
    r"(^|[^A-Za-z])OOM([^A-Za-z]|$)|(^|[^A-Za-z])NaN([^A-Za-z]|$)|"
    r"WorkerCrashedError|RayActorError|ActorDiedError",
    flags=re.IGNORECASE | re.MULTILINE,
)


def validate_result(
    log_text: str,
    expected_step: int,
    checkpoint_root: Path,
) -> list[str]:
    errors = []
    if PATCH_MARKER not in log_text:
        errors.append("missing HCU patch marker")
    if f"step:{expected_step}" not in log_text:
        errors.append(f"missing training step marker: step:{expected_step}")
    if FATAL_PATTERN.search(log_text):
        errors.append("log contains a fatal runtime marker")
    if checkpoint_root.exists() and any(checkpoint_root.rglob("global_step_*")):
        errors.append("training checkpoint was unexpectedly created")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an HCU Nightly log.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    errors = validate_result(log_text, args.expected_step, args.checkpoint_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"HCU Nightly reached step:{args.expected_step} without a checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
