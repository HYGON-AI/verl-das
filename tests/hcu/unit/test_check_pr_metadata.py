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
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tests" / "hcu" / "ci" / "check_pr_metadata.py"
OFFICIAL_MODULES = {
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
}


def load_module():
    spec = importlib.util.spec_from_file_location("check_pr_metadata", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_allowed_modules_match_official_list_plus_hcu():
    module = load_module()

    assert set(module.ALLOWED_MODULES) == OFFICIAL_MODULES | {"hcu"}


@pytest.mark.parametrize(
    "title",
    [
        "[hcu] feat: add HCU CI",
        "[hcu,ci] fix: preserve runtime env",
        "[1/N] [ci] chore: add smoke coverage",
        "[2/3][BREAKING][hcu] refactor: update patch loading",
        "[BREAKING] [tests] test: cover metadata",
    ],
)
def test_validate_metadata_accepts_valid_titles(title):
    module = load_module()

    assert module.validate_metadata(title, "Non-empty PR description.") == []


def test_validate_metadata_rejects_invalid_module():
    module = load_module()

    errors = module.validate_metadata("[unknown] feat: add check", "body")

    assert any("unknown" in error and "module" in error.lower() for error in errors)


def test_validate_metadata_rejects_invalid_type():
    module = load_module()

    errors = module.validate_metadata("[hcu] docs: explain check", "body")

    assert any("type" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "title",
    [
        "[hcu] feat:",
        "[hcu] feat:   ",
        "[1/N] [BREAKING] [hcu] fix:",
    ],
)
def test_validate_metadata_rejects_empty_title_description(title):
    module = load_module()

    assert module.validate_metadata(title, "body")


@pytest.mark.parametrize("body", ["", " ", "\n\t"])
def test_validate_metadata_rejects_blank_pr_body(body):
    module = load_module()

    errors = module.validate_metadata("[hcu] feat: valid title", body)

    assert any("description" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "title",
    [
        "[N/3] [hcu] feat: invalid numerator",
        "[1/0] [hcu] feat: invalid denominator",
        "[1/N] [BREAKING] feat: missing modules",
        "[BREAKING] [1/N] [hcu] feat: wrong prefix order",
    ],
)
def test_validate_metadata_rejects_invalid_prefix_forms(title):
    module = load_module()

    assert module.validate_metadata(title, "body")


def test_cli_reads_pr_title_and_body_from_environment():
    environment = os.environ.copy()
    environment.update(
        {
            "PR_TITLE": "[hcu,ci] feat: add HCU checks",
            "PR_BODY": "Adds local static and HCU container checks.",
        }
    )

    result = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert "passed" in result.stdout


def test_cli_returns_nonzero_for_invalid_metadata():
    environment = os.environ.copy()
    environment.update({"PR_TITLE": "[unknown] docs:", "PR_BODY": "   "})

    result = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode != 0
    assert "ERROR:" in result.stderr
