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
MODULE_PATH = ROOT / "tests" / "hcu" / "ci" / "verify_submodules.py"
EXPECTED = {
    "third_party/verl": "1" * 40,
    "third_party/Megatron-LM": "2" * 40,
    "third_party/VeOmni": "3" * 40,
}


def load_module():
    spec = importlib.util.spec_from_file_location("verify_submodules", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mark_submodules_initialized(repository_root):
    for path in EXPECTED:
        marker = repository_root / path / ".git"
        marker.parent.mkdir(parents=True)
        marker.write_text("gitdir: test\n", encoding="utf-8")


def test_expected_submodule_inventory_is_complete():
    module = load_module()

    assert module.SUBMODULE_PATHS == tuple(EXPECTED)


def test_parse_gitlink_sha_reads_commit_entry():
    module = load_module()

    assert (
        module.parse_gitlink_sha(
            f"160000 commit {EXPECTED['third_party/verl']}\tthird_party/verl"
        )
        == EXPECTED["third_party/verl"]
    )


def test_verify_submodules_checks_gitlink_and_checkout(tmp_path):
    module = load_module()
    calls = []
    mark_submodules_initialized(tmp_path)

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


def test_verify_submodules_reports_checkout_mismatches(tmp_path):
    module = load_module()
    wrong_sha = "0" * 40
    mark_submodules_initialized(tmp_path)

    def run(command, **kwargs):
        if command[3] == "ls-tree":
            path = command[-1]
            return CompletedProcess(
                command,
                0,
                f"160000 commit {EXPECTED[path]}\t{path}\n",
                "",
            )
        return CompletedProcess(command, 0, wrong_sha + "\n", "")

    errors = module.verify_submodules(tmp_path, run=run)

    assert len(errors) == 3
    assert all("checkout SHA mismatch" in error for error in errors)


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
