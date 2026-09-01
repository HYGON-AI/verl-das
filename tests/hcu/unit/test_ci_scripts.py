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
import json
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


def load_module(filename: str, module_name: str):
    path = HCU_TEST_DIR / "ci" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_nightly_planner():
    return load_module("plan_nightly.py", "verl_hcu_plan_nightly_test")


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


def test_all_hcu_python_files_have_hygon_apache_headers():
    files = list(HCU_TEST_DIR.rglob("*.py"))

    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert COPYRIGHT in text, path
        assert "Licensed under the Apache License, Version 2.0" in text, path


def test_nightly_planner_has_apache_spdx_header():
    planner = read_script("ci/plan_nightly.py")

    assert "SPDX-License-Identifier: Apache-2.0" in planner


def test_hcu_shell_scripts_use_strict_mode_without_license_headers():
    files = list(HCU_TEST_DIR.rglob("*.sh"))

    assert files
    for path in files:
        script = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in script, path
        assert "Copyright" not in script, path
        assert "SPDX-License-Identifier" not in script, path
        assert "Licensed under" not in script, path


def test_prepare_workspace_sets_paths_and_verifies_patch_copy():
    script = read_script("ci/prepare_workspace.sh")

    assert "verify_submodules.py" in script
    assert 'git -C "${repo_root}" submodule sync --recursive' in script
    assert 'git -C "${repo_root}" submodule update --init --recursive' in script
    assert 'patch_source="${repo_root}/hcu_verl/patch_init.py"' in script
    assert 'patch_target="${repo_root}/third_party/verl/verl/__init__.py"' in script
    assert 'cp -- "${patch_source}" "${patch_tmp}"' in script
    assert "cmp -s" in script
    assert 'grep -Fq "from hcu_verl import verl_adaptor" "${patch_target}"' in script
    assert "safe.directory" in script


def test_cleanup_only_targets_resources_owned_by_the_current_run():
    script = read_script("ci/cleanup.sh")

    assert "ray stop --force" in script
    assert '[[ -f "${run_dir}/ray-owned" ]]' in script
    assert "/proc/${candidate_pid}/environ" in script
    assert "VERL_HCU_CI_RUN_ID=${run_id}" in script
    assert '[[ "${tmp_root}" == "/" ]]' in script
    assert "pkill" not in script
    assert "killall" not in script


def test_explicit_container_lifecycle_is_scoped_and_restores_ownership():
    start = read_script("ci/start_container.sh")
    execute = read_script("ci/exec_container.sh")
    cleanup = read_script("ci/cleanup_container.sh")

    assert "@sha256:" in start
    assert '[[ "${workspace}" == "/" ]]' in start
    assert '--volume "${workspace}:/workspace"' in start
    assert '--volume "${VERL_HCU_MODEL_ROOT}:${VERL_HCU_MODEL_ROOT}:ro"' in start
    assert '--volume "${data_dir}:${data_dir}:ro"' in start
    assert "--device=/dev/kfd" in start
    assert "--device=/dev/mkfd" in start
    assert 'docker rm -f "${container}"' in start
    assert 'docker exec --workdir "${workdir}"' in execute
    assert "eval " not in execute
    assert "tests/hcu/ci/cleanup.sh" in cleanup
    assert 'chown -R -- "${host_uid}:${host_gid}" /workspace' in cleanup
    assert 'docker rm -f "${container}"' in cleanup
    assert "docker system prune" not in cleanup


def test_workflows_use_explicit_containers_without_restore_jobs():
    pr_workflow = read_workflow("pr-test-hcu.yml")
    nightly_workflow = read_workflow("nightly-test-hcu.yml")

    for workflow in (pr_workflow, nightly_workflow):
        assert "container:" not in workflow
        assert "restore-after-" not in workflow
        assert "tests/hcu/ci/start_container.sh" in workflow
        assert "tests/hcu/ci/cleanup_container.sh" in workflow
    assert not (HCU_TEST_DIR / "ci" / "restore_runner_permissions.sh").exists()


def test_workflows_restore_workspace_ownership_before_checkout():
    workflows = {
        "pr-test-hcu.yml": ("unit", "VERL_HCU_PR_IMAGE"),
        "nightly-test-hcu.yml": ("plan", "VERL_HCU_VLLM_IMAGE"),
    }

    for name, (job_name, image_variable) in workflows.items():
        job = workflow_job_blocks(read_workflow(name))[job_name]
        restore = job.index("- name: Restore existing workspace ownership")
        checkout = job.index("- name: Checkout repository")

        assert restore < checkout
        assert f"${{{{ vars.{image_variable} }}}}" in job
        assert '"${work_root}"/*/*' in job
        assert "@sha256:" in job
        assert "docker run --rm" in job
        assert "--user root" in job
        assert 'chown -R -- "${owner}" /workspace' in job


def test_nightly_case_manifest_is_the_matrix_source_of_truth():
    planner = load_nightly_planner()
    cases = planner.load_cases()

    assert [case["engine"] for case in cases] == ["vllm", "sglang"]
    assert [case["expected_step"] for case in cases] == [5, 3]
    assert cases[0]["model_parent"] == "vllm-optest-models/deepseek-ai"
    assert cases[0]["model_path"] == (
        "vllm-optest-models/deepseek-ai/deepseek-llm-7b-chat"
    )
    assert planner.build_matrix("all")["include"] == cases
    assert [case["engine"] for case in planner.build_matrix("vllm")["include"]] == [
        "vllm"
    ]
    assert planner.build_matrix(cases[1]["id"])["include"] == [cases[1]]
    with pytest.raises(ValueError, match="unknown nightly selector"):
        planner.build_matrix("missing-case")

    workflow = read_workflow("nightly-test-hcu.yml")
    assert "tests/hcu/ci/plan_nightly.py" in workflow
    assert "fromJSON(needs.plan.outputs.matrix)" in workflow
    assert "max-parallel: 1" in workflow
    assert not (HCU_TEST_DIR / "nightly" / "bw1000" / "ci_cases.yaml").exists()


def test_nightly_runner_validates_inputs_and_training_results():
    runner = read_script("nightly/run_case.sh")

    assert "CASE_REQUIRED_GPUS" in runner
    assert "CASE_EXPECTED_STEP" in runner
    assert "realpath -e" in runner
    assert 'source "${CI_DIR}/prepare_workspace.sh"' in runner
    assert "runtime --require-data-roots --require-gpus" in runner
    assert 'test -s "${VERL_HCU_DATA_ROOT}/gsm8k/train.parquet"' in runner
    assert 'test -s "${VERL_HCU_DATA_ROOT}/gsm8k/test.parquet"' in runner
    assert 'test -d "${VERL_HCU_MODEL_ROOT}/${CASE_MODEL_PATH}"' in runner
    assert "check_nightly_result.py" in runner
    assert 'bash "${case_script}" 2>&1 | tee "${log_file}"' in runner


def test_training_scripts_only_contain_training_configuration():
    scripts = (
        read_script("nightly/bw1000/run_vllm_grpo_5step.sh"),
        read_script("nightly/bw1000/run_sglang_off_policy_3step.sh"),
    )

    for script in scripts:
        assert "prepare_workspace.sh" not in script
        assert "check_environment.py" not in script
        assert "check_nightly_result.py" not in script
        assert "HCU_ADAPT" not in script
        assert "hf download" not in script
        assert "wget " not in script
        assert "curl " not in script


def test_pr_workflow_uses_direct_path_filters_and_three_jobs():
    workflow = read_workflow("pr-test-hcu.yml")
    jobs = workflow_job_blocks(workflow)

    assert list(jobs) == ["unit", "runtime", "finish"]
    assert "pull_request_target:" in workflow
    assert "\n  pull_request:\n" not in workflow
    assert "      - edited\n" not in workflow
    for path in (
        '      - "hcu_verl/**"',
        '      - "tests/hcu/**"',
        '      - ".github/workflows/pr-test-hcu.yml"',
        '      - ".gitmodules"',
        '      - "third_party/verl"',
    ):
        assert path in workflow
    assert "github.event.pull_request.head.repo.full_name" in jobs["unit"]
    assert "HCU execution authorized for" in jobs["unit"]
    assert "github.event.pull_request.author_association" not in jobs["unit"]
    assert "github.event.pull_request.head.sha" in workflow
    assert "needs:\n      - unit" in jobs["runtime"]
    assert "needs.plan" not in workflow
    assert "tests/hcu/ci/plan_pr.py" not in workflow
    assert not (ROOT / ".github" / "workflows" / "hcu_unit_tests.yaml").exists()


def test_pr_lanes_preserve_all_four_test_layers():
    workflow = read_workflow("pr-test-hcu.yml")
    unit = workflow_job_blocks(workflow)["unit"]
    runtime = workflow_job_blocks(workflow)["runtime"]

    assert "bash tests/hcu/pr/run_upstream_pr_tests.sh" in unit
    assert "bash tests/hcu/pr/run_hcu_unit_tests.sh" in unit
    assert "bash tests/hcu/pr/run_hcu_smoke.sh" in runtime
    assert "bash tests/hcu/pr/run_vllm_grpo_1step.sh" in runtime
    assert "timeout-minutes: 10" in unit
    assert "timeout-minutes: 60" in unit
    assert "timeout-minutes: 20" in runtime
    assert "timeout-minutes: 30" in runtime


def test_transfer_queue_is_installed_only_in_the_unit_lane():
    workflow = read_workflow("pr-test-hcu.yml")
    jobs = workflow_job_blocks(workflow)
    patch = (ROOT / "hcu_verl" / "patch_init.py").read_text(encoding="utf-8")

    assert "TransferQueue==0.1.9" in jobs["unit"]
    assert "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple" in jobs["unit"]
    assert "TransferQueue" not in jobs["runtime"]
    assert "_configure_optional_queue_mock" not in patch
    assert "optional_queue_compat" not in patch


def test_smoke_uses_real_resources_without_repeating_hcu_unit_tests():
    script = read_script("pr/run_hcu_smoke.sh")

    assert 'check_environment.py" runtime --require-gpus 8' in script
    assert "assert torch.cuda.is_available()" in script
    assert "assert torch.cuda.device_count() == 8" in script
    assert 'ray.init(address="local", include_dashboard=False)' in script
    assert "@ray.remote(num_gpus=1)" in script
    assert "len(set(assignments)) == 8" in script
    assert "FileSystemWriterAsync.preload_tensors" in script
    assert "tests/hcu/unit" not in script
    assert "trap cleanup EXIT" not in script


def test_case_scripts_use_repository_logs_and_managed_cleanup():
    scripts = (
        read_script("pr/run_hcu_smoke.sh"),
        read_script("pr/run_upstream_pr_tests.sh"),
        read_script("pr/run_hcu_unit_tests.sh"),
        read_script("pr/run_vllm_grpo_1step.sh"),
    )

    expected = "${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}"
    for script in scripts:
        assert expected in script
    assert all("trap cleanup EXIT" not in script for script in scripts)


def test_workflows_validate_only_their_own_configuration_profile():
    pr_workflow = read_workflow("pr-test-hcu.yml")
    nightly_workflow = read_workflow("nightly-test-hcu.yml")

    setup_python = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
    assert setup_python not in pr_workflow
    assert setup_python not in nightly_workflow
    assert "check_environment.py config --profile pr" in pr_workflow
    assert "check_environment.py config --profile nightly" in nightly_workflow


def test_nightly_plan_uses_python_from_the_pinned_ci_image():
    plan = workflow_job_blocks(read_workflow("nightly-test-hcu.yml"))["plan"]

    assert "VERL_HCU_PLAN_IMAGE: ${{ vars.VERL_HCU_VLLM_IMAGE }}" in plan
    assert "--network none" in plan
    assert '--volume "${GITHUB_WORKSPACE}:/workspace:ro"' in plan
    assert "--workdir /workspace" in plan
    assert "python3 tests/hcu/ci/plan_nightly.py" in plan


def test_hcu_workflows_use_bw1100_runners():
    expected_jobs = {
        "pr-test-hcu.yml": ("unit", "runtime", "finish"),
        "nightly-test-hcu.yml": ("plan", "nightly", "finish"),
    }

    for name, job_names in expected_jobs.items():
        workflow = read_workflow(name)
        jobs = workflow_job_blocks(workflow)

        assert "ubuntu-latest" not in workflow
        for job_name in job_names:
            block = jobs[job_name]
            assert "group: ci-general" in block, job_name
            assert "labels: [self-hosted, ci, bw1100]" in block, job_name


def test_actionlint_knows_the_local_runner_label():
    config = (ROOT / ".github" / "actionlint.yaml").read_text(encoding="utf-8")

    assert "self-hosted-runner:" in config
    assert "- bw1000" in config
    assert "- bw1100" in config
    assert "- ci" in config


def test_ci_readme_uses_shared_asset_roots():
    readme = (ROOT / ".github" / "workflows" / "README.md").read_text(encoding="utf-8")

    assert "VERL_HCU_MODEL_ROOT=/ci_public/verl-das/models" in readme
    assert "VERL_HCU_DATA_ROOT=/ci_public/verl-das/data" in readme
    assert "/home/github/tly" not in readme


def test_ci_support_scripts_and_case_launchers_live_under_hcu_tests():
    expected = (
        HCU_TEST_DIR / "ci" / "check_environment.py",
        HCU_TEST_DIR / "ci" / "check_nightly_result.py",
        HCU_TEST_DIR / "ci" / "cleanup.sh",
        HCU_TEST_DIR / "ci" / "cleanup_container.sh",
        HCU_TEST_DIR / "ci" / "exec_container.sh",
        HCU_TEST_DIR / "ci" / "plan_nightly.py",
        HCU_TEST_DIR / "ci" / "prepare_workspace.sh",
        HCU_TEST_DIR / "ci" / "start_container.sh",
        HCU_TEST_DIR / "ci" / "verify_submodules.py",
        HCU_TEST_DIR / "nightly" / "run_case.sh",
        HCU_TEST_DIR / "pr" / "run_hcu_smoke.sh",
        HCU_TEST_DIR / "pr" / "run_hcu_unit_tests.sh",
        HCU_TEST_DIR / "pr" / "run_upstream_pr_tests.sh",
        HCU_TEST_DIR / "pr" / "run_vllm_grpo_1step.sh",
    )

    assert all(path.is_file() for path in expected)
    assert not (ROOT / "scripts" / "ci" / "hcu").exists()


def test_training_cases_use_configured_roots_without_downloads_or_examples():
    script_paths = (
        HCU_TEST_DIR / "pr" / "run_vllm_grpo_1step.sh",
        HCU_TEST_DIR / "nightly" / "bw1000" / "run_vllm_grpo_5step.sh",
        HCU_TEST_DIR / "nightly" / "bw1000" / "run_sglang_off_policy_3step.sh",
    )

    for path in script_paths:
        script = path.read_text(encoding="utf-8")
        assert "VERL_HCU_MODEL_ROOT" in script
        assert "VERL_HCU_DATA_ROOT" in script
        assert "${REPO_ROOT}/examples/" not in script
        assert "hf download" not in script
        assert "wget " not in script
        assert "curl " not in script


def test_complete_gsm8k_data_is_mounted_read_only():
    start = read_script("ci/start_container.sh")
    vllm = read_script("nightly/bw1000/run_vllm_grpo_5step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_3step.sh")

    assert '--volume "${data_dir}:${data_dir}:ro"' in start
    for script in (vllm, sglang):
        assert "train_file=${data_path}/gsm8k/train.parquet" in script
        assert "test_file=${data_path}/gsm8k/test.parquet" in script
        assert "prepare_gsm8k_data.py" not in script


def test_upstream_pr_suite_remains_curated():
    script = read_script("pr/run_upstream_pr_tests.sh")

    assert "tests/special_sanity" in script
    assert "tests/test_protocol_on_cpu.py" in script
    assert "tests/trainer/ppo/test_core_algos_on_cpu.py" in script
    assert "tests/workers/config/test_actor_config_on_cpu.py" in script
    assert "--deselect" in script
    assert "test_target_modules_raises_on_invalid_type" in script
    assert "pinned upstream v" in script


def test_hcu_unit_suite_keeps_pinned_environment_and_exclusions():
    script = read_script("pr/run_hcu_unit_tests.sh")

    assert 'source "${CI_DIR}/prepare_workspace.sh"' in script
    assert 'PYTHONPATH="${REPO_ROOT}/tests${PYTHONPATH:+:${PYTHONPATH}}"' in script
    assert 'check_environment.py" runtime --require-gpus 8' in script
    assert "python3 -m pytest -s -x" in script
    assert "--ignore=tests/utils/test_activation_offload.py" in script
    assert "--ignore=tests/utils/test_fsdp_lora_merge.py" in script
    assert "tests/utils/test_special_megatron_kl_loss_tp.py" in script
    assert "TransferQueue" not in script


def test_pr_vllm_case_remains_a_bounded_real_training_step():
    script = read_script("pr/run_vllm_grpo_1step.sh")

    assert "runtime --require-data-roots --require-gpus 8" in script
    assert "python3 -m verl.trainer.main_ppo" in script
    assert "actor_rollout_ref.rollout.name=vllm" in script
    assert "algorithm.adv_estimator=grpo" in script
    assert "data.train_batch_size=16" in script
    assert "actor_rollout_ref.rollout.n=2" in script
    assert "trainer.total_training_steps=1" in script


def test_finish_jobs_keep_strict_success_semantics():
    pr_finish = workflow_job_blocks(read_workflow("pr-test-hcu.yml"))["finish"]
    nightly_finish = workflow_job_blocks(read_workflow("nightly-test-hcu.yml"))[
        "finish"
    ]

    assert '[[ "${UNIT_RESULT}" == "success" ]]' in pr_finish
    assert '[[ "${RUNTIME_RESULT}" == "success" ]]' in pr_finish
    assert "PLAN_RESULT" not in pr_finish
    assert '== "skipped"' not in pr_finish
    assert '[[ "${PLAN_RESULT}" == "success" ]]' in nightly_finish
    assert '[[ "${NIGHTLY_RESULT}" == "success" ]]' in nightly_finish


def test_nightly_manifest_is_valid_json_with_current_cases():
    path = HCU_TEST_DIR / "nightly" / "bw1000" / "cases.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert [case["id"] for case in document["cases"]] == [
        "vllm-deepseek-7b-grpo-5step",
        "sglang-qwen3-0p6b-off-policy-3step",
    ]
    assert not (HCU_TEST_DIR / "pr" / "ci_cases.yaml").exists()
