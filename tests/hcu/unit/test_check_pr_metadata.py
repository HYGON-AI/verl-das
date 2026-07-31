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
ALLOWED_SCOPES = {
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
}


def load_module():
    spec = importlib.util.spec_from_file_location("check_pr_metadata", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_allowed_scopes_match_supported_modules():
    module = load_module()

    assert set(module.ALLOWED_SCOPES) == ALLOWED_SCOPES


@pytest.mark.parametrize(
    "title",
    [
        "ci(hcu): add HCU CI",
        "fix(runtime): preserve runtime env",
        "docs(ci): document smoke coverage",
        "perf(attention): improve kernel latency",
        "build(docker): pin image digest",
        "test(tests): cover metadata",
    ],
)
def test_validate_metadata_accepts_valid_titles(title):
    module = load_module()

    assert module.validate_metadata(title, "Non-empty PR description.") == []


def test_validate_metadata_rejects_invalid_scope():
    module = load_module()

    errors = module.validate_metadata("feat(unknown): add check", "body")

    assert any("unknown" in error and "scope" in error.lower() for error in errors)


def test_validate_metadata_rejects_invalid_type():
    module = load_module()

    errors = module.validate_metadata("style(hcu): explain check", "body")

    assert any("type" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "title",
    [
        "feat(hcu):",
        "feat(hcu):   ",
        "feat(hcu): Add uppercase subject",
        f"feat(hcu): {'a' * 73}",
    ],
)
def test_validate_metadata_rejects_invalid_subject(title):
    module = load_module()

    assert module.validate_metadata(title, "body")


@pytest.mark.parametrize("body", ["", " ", "\n\t"])
def test_validate_metadata_rejects_blank_pr_body(body):
    module = load_module()

    errors = module.validate_metadata("feat(hcu): add valid title", body)

    assert any("description" in error.lower() for error in errors)


@pytest.mark.parametrize(
    "title",
    [
        "[hcu] feat: legacy format",
        "feat: missing scope",
        "feat(hcu) missing colon",
        "feat(hcu):add missing space",
        "FEAT(hcu): add uppercase type",
    ],
)
def test_validate_metadata_rejects_invalid_title_forms(title):
    module = load_module()

    assert module.validate_metadata(title, "body")


def test_cli_reads_pr_title_and_body_from_environment():
    environment = os.environ.copy()
    environment.update(
        {
            "PR_TITLE": "ci(hcu): add PR and nightly validation",
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
    environment.update({"PR_TITLE": "style(unknown): Bad title", "PR_BODY": "   "})

    result = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode != 0
    assert "ERROR:" in result.stderr
