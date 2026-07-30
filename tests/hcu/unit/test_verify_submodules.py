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

import importlib.util
from pathlib import Path
from subprocess import CompletedProcess

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "ci" / "hcu" / "verify_submodules.py"
EXPECTED = {
    "third_party/verl": "7aed6b230776f963fa09509c10d9c3a767d1102c",
    "third_party/Megatron-LM": "266f1c97dca477ca8b92c16087506da2000b0b84",
    "third_party/VeOmni": "cbb3e012936912fd9ce063241e1fb77e8d564d2f",
}


def load_module():
    spec = importlib.util.spec_from_file_location("verify_submodules", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_expected_submodule_inventory_is_fixed():
    module = load_module()

    assert module.EXPECTED_SUBMODULES == EXPECTED


def test_parse_gitlink_sha_reads_commit_entry():
    module = load_module()

    assert (
        module.parse_gitlink_sha(
            "160000 commit 7aed6b230776f963fa09509c10d9c3a767d1102c\tthird_party/verl"
        )
        == EXPECTED["third_party/verl"]
    )


def test_verify_submodules_checks_gitlink_and_checkout(tmp_path):
    module = load_module()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["git", "-C", str(tmp_path)] and command[3] == "ls-tree":
            path = command[-1]
            return CompletedProcess(
                command,
                0,
                f"160000 commit {EXPECTED[path]}\t{path}\n",
                "",
            )
        path = Path(command[2]).relative_to(tmp_path).as_posix()
        return CompletedProcess(command, 0, EXPECTED[path] + "\n", "")

    errors = module.verify_submodules(tmp_path, run=run)

    assert errors == []
    assert len(calls) == 6


def test_verify_submodules_reports_gitlink_and_checkout_mismatches(tmp_path):
    module = load_module()
    wrong_sha = "0" * 40

    def run(command, **kwargs):
        if command[3] == "ls-tree":
            path = command[-1]
            return CompletedProcess(
                command,
                0,
                f"160000 commit {wrong_sha}\t{path}\n",
                "",
            )
        return CompletedProcess(command, 0, wrong_sha + "\n", "")

    errors = module.verify_submodules(tmp_path, run=run)

    assert len(errors) == 6
    assert any("gitlink" in error for error in errors)
    assert any("checkout" in error for error in errors)


def test_verify_submodules_reports_missing_checkout(tmp_path):
    module = load_module()

    def run(command, **kwargs):
        path = command[-1]
        if command[3] == "ls-tree":
            return CompletedProcess(
                command,
                0,
                f"160000 commit {EXPECTED[path]}\t{path}\n",
                "",
            )
        return CompletedProcess(command, 128, "", "not a git repository")

    errors = module.verify_submodules(tmp_path, run=run)

    assert len(errors) == 3
    assert all("checkout" in error for error in errors)
