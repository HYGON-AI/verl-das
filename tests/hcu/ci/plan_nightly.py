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
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_PATH = ROOT / "tests" / "hcu" / "nightly" / "bw1000" / "cases.json"
NIGHTLY_SCRIPT_ROOT = (ROOT / "tests" / "hcu" / "nightly" / "bw1000").resolve()
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
RELATIVE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
ENGINES = {"vllm", "sglang"}

REQUIRED_FIELDS = {
    "id": str,
    "name": str,
    "engine": str,
    "image_profile": str,
    "script": str,
    "model_parent": str,
    "model_path": str,
    "required_gpus": int,
    "timeout_minutes": int,
    "expected_step": int,
    "requires_galaxyhip": bool,
}


def _validate_relative_path(value: str, field: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or not RELATIVE_PATH_PATTERN.fullmatch(value)
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError(f"{field} must be a safe relative path: {value!r}")


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("nightly case schema_version must be 1")

    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("nightly cases must be a non-empty list")

    seen_ids: set[str] = set()
    validated = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"nightly case {index} must be an object")
        for field, expected_type in REQUIRED_FIELDS.items():
            value = case.get(field)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"nightly case {index} field {field!r} must be "
                    f"{expected_type.__name__}"
                )

        case_id = case["id"]
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(f"invalid nightly case id: {case_id!r}")
        if case_id in seen_ids:
            raise ValueError(f"duplicate nightly case id: {case_id}")
        seen_ids.add(case_id)

        engine = case["engine"]
        if engine not in ENGINES or case["image_profile"] != engine:
            raise ValueError(
                f"nightly case {case_id} must use a matching vllm or sglang image"
            )
        if not case["name"].strip():
            raise ValueError(f"nightly case {case_id} has an empty display name")
        for field in ("script", "model_parent", "model_path"):
            _validate_relative_path(case[field], field)
        if not case["script"].endswith(".sh"):
            raise ValueError(f"nightly case {case_id} script must be a .sh file")

        script_path = (ROOT / case["script"]).resolve()
        if NIGHTLY_SCRIPT_ROOT not in script_path.parents or not script_path.is_file():
            raise ValueError(
                f"nightly case {case_id} script is missing or outside the nightly root"
            )
        for field in ("required_gpus", "timeout_minutes", "expected_step"):
            if case[field] <= 0:
                raise ValueError(
                    f"nightly case {case_id} field {field} must be positive"
                )

        validated.append(dict(case))
    return validated


def select_cases(cases: list[dict[str, Any]], selector: str) -> list[dict[str, Any]]:
    requested = [value.strip() for value in selector.split(",") if value.strip()]
    if not requested or requested == ["all"]:
        return cases
    if "all" in requested:
        raise ValueError("all cannot be combined with another nightly selector")

    known = {case["id"] for case in cases} | {case["engine"] for case in cases}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown nightly selector(s): {', '.join(unknown)}")

    selected = [
        case for case in cases if case["id"] in requested or case["engine"] in requested
    ]
    if not selected:
        raise ValueError("nightly selector did not match any case")
    return selected


def build_matrix(selector: str, path: Path = DEFAULT_CASES_PATH) -> dict[str, Any]:
    return {"include": select_cases(load_cases(path), selector)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HCU nightly case matrix.")
    parser.add_argument(
        "--select",
        default="all",
        help="all, engine, case id, or comma-separated values",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.select, args.cases), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
