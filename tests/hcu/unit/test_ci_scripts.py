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
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HCU_TEST_DIR = ROOT / "tests" / "hcu"
COPYRIGHT = "Copyright (c) 2026 Hygon Information Technology Co., Ltd."


def read_script(name: str) -> str:
    return (HCU_TEST_DIR / name).read_text(encoding="utf-8")


def read_workflow(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def load_nightly_planner():
    path = HCU_TEST_DIR / "ci" / "plan_nightly.py"
    spec = importlib.util.spec_from_file_location("verl_hcu_plan_nightly_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def workflow_job_blocks(workflow: str) -> dict[str, str]:
    jobs = workflow.split("\njobs:\n", maxsplit=1)[1]
    matches = list(re.finditer(r"(?m)^  ([a-z][a-z0-9-]*):\s*$", jobs))
    return {
        match.group(1): jobs[
            match.start() : matches[index + 1].start()
            if index + 1 < len(matches)
            else len(jobs)
        ]
        for index, match in enumerate(matches)
    }


def test_hcu_sources_follow_header_policy():
    python_files = list(HCU_TEST_DIR.rglob("*.py"))
    shell_files = list(HCU_TEST_DIR.rglob("*.sh"))

    assert python_files and shell_files
    for path in python_files:
        text = path.read_text(encoding="utf-8")
        assert COPYRIGHT in text, path
        assert "Licensed under the Apache License, Version 2.0" in text, path
    for path in shell_files:
        script = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in script, path
        assert "Copyright" not in script, path
        assert not any(
            marker in script for marker in ("SPDX-License-Identifier", "Licensed under")
        ), path


def test_container_lifecycle_keeps_shared_runner_resources_scoped():
    start = read_script("ci/start_container.sh")
    execute = read_script("ci/exec_container.sh")
    cleanup = read_script("ci/cleanup_container.sh")

    assert all(
        item in start
        for item in (
            '--volume "${workspace}:/workspace"',
            '--volume "${VERL_HCU_MODEL_ROOT}:${VERL_HCU_MODEL_ROOT}:ro"',
            '--volume "${data_dir}:${data_dir}:ro"',
            "--device=/dev/kfd",
            "--device=/dev/mkfd",
        )
    )
    assert 'docker exec --workdir "${workdir}"' in execute
    assert all(
        item in cleanup
        for item in (
            "tests/hcu/ci/cleanup.sh",
            'chown -R -- "${host_uid}:${host_gid}" /workspace',
            'docker rm -f "${container}"',
        )
    )
    lifecycle = f"{start}\n{execute}\n{cleanup}"
    assert not any(
        command in lifecycle
        for command in ("docker system prune", "pkill", "killall", "eval ")
    )


def test_workflows_restore_workspace_before_checkout():
    jobs = {
        "pr-test-hcu.yml": "unit",
        "nightly-test-hcu.yml": "plan",
    }

    for workflow_name, job_name in jobs.items():
        job = workflow_job_blocks(read_workflow(workflow_name))[job_name]
        assert job.index("Restore existing workspace ownership") < job.index(
            "Checkout repository"
        )


def test_pr_workflow_uses_paths_and_existing_test_lanes():
    workflow = read_workflow("pr-test-hcu.yml")
    jobs = workflow_job_blocks(workflow)
    paths = (
        "hcu_verl/**",
        "tests/hcu/**",
        ".github/workflows/pr-test-hcu.yml",
        ".gitmodules",
        "third_party/verl",
    )
    commands = (
        "bash tests/hcu/pr/run_upstream_pr_tests.sh",
        "bash tests/hcu/pr/run_hcu_unit_tests.sh",
        "bash tests/hcu/pr/run_hcu_smoke.sh",
        "bash tests/hcu/pr/run_vllm_grpo_1step.sh",
    )

    assert set(jobs) == {"unit", "runtime", "finish"}
    assert all(f'"{path}"' in workflow for path in paths)
    assert all(command in workflow for command in commands)
    assert "needs:\n      - unit" in jobs["runtime"]
    assert "TransferQueue==0.1.9" in jobs["unit"]
    assert "needs.plan" not in workflow
    assert "tests/hcu/ci/plan_pr.py" not in workflow


def test_nightly_manifest_drives_the_job_matrix():
    planner = load_nightly_planner()
    cases = planner.load_cases()

    assert cases
    assert planner.build_matrix("all")["include"] == cases
    for case in cases:
        assert planner.build_matrix(case["id"])["include"] == [case]
    with pytest.raises(ValueError, match="unknown nightly selector"):
        planner.build_matrix("missing-case")

    workflow = read_workflow("nightly-test-hcu.yml")
    assert "python3 tests/hcu/ci/plan_nightly.py" in workflow
    assert "fromJSON(needs.plan.outputs.matrix)" in workflow


def test_model_cases_use_configured_assets_without_downloading():
    planner = load_nightly_planner()
    scripts = [HCU_TEST_DIR / "pr" / "run_vllm_grpo_1step.sh"]
    scripts.extend(ROOT / case["script"] for case in planner.load_cases())

    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "VERL_HCU_MODEL_ROOT" in text, path
        assert "VERL_HCU_DATA_ROOT" in text, path
        assert not any(
            command in text for command in ("hf download", "wget ", "curl ")
        ), path


def test_workflows_use_container_python_and_managed_cleanup():
    pr_workflow = read_workflow("pr-test-hcu.yml")
    nightly_workflow = read_workflow("nightly-test-hcu.yml")
    workflows = pr_workflow + nightly_workflow

    assert "actions/setup-python" not in workflows
    assert "check_environment.py config" in pr_workflow
    assert "check_environment.py config" in nightly_workflow
    assert "--profile" not in workflows
    assert "VERL_HCU_CI_IMAGE" in pr_workflow
    assert "VERL_HCU_CI_IMAGE" in nightly_workflow
    for workflow in (pr_workflow, nightly_workflow):
        assert "tests/hcu/ci/start_container.sh" in workflow
        assert "tests/hcu/ci/cleanup_container.sh" in workflow


def test_finish_jobs_require_success():
    expected_results = {
        "pr-test-hcu.yml": ("UNIT_RESULT", "RUNTIME_RESULT"),
        "nightly-test-hcu.yml": ("PLAN_RESULT", "NIGHTLY_RESULT"),
    }

    for workflow_name, results in expected_results.items():
        finish = workflow_job_blocks(read_workflow(workflow_name))["finish"]
        assert all(
            f'[[ "${{{result}}}" == "success" ]]' in finish for result in results
        )
