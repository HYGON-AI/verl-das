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

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HCU_TEST_DIR = ROOT / "tests" / "hcu"
COPYRIGHT = "Copyright (c) 2026 Hygon Information Technology Co., Ltd."


def read_script(name: str) -> str:
    return (HCU_TEST_DIR / name).read_text(encoding="utf-8")


def test_all_new_python_and_shell_files_have_hygon_apache_headers():
    files = list(HCU_TEST_DIR.rglob("*.py")) + list(HCU_TEST_DIR.rglob("*.sh"))

    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert COPYRIGHT in text, path
        assert "Licensed under the Apache License, Version 2.0" in text, path


def test_all_shell_scripts_enable_strict_mode():
    for path in HCU_TEST_DIR.rglob("*.sh"):
        assert "set -euo pipefail" in path.read_text(encoding="utf-8"), path


def test_prepare_workspace_sets_paths_and_verifies_patch_copy():
    script = read_script("ci/prepare_workspace.sh")

    assert "verify_submodules.py" in script
    assert "third_party/verl" in script
    assert "third_party/Megatron-LM" in script
    assert "third_party/VeOmni" in script
    assert 'patch_source="${repo_root}/hcu_verl/patch_init.py"' in script
    assert 'patch_target="${repo_root}/third_party/verl/verl/__init__.py"' in script
    assert 'cp -- "${patch_source}" "${patch_tmp}"' in script
    assert "cmp -s" in script
    assert 'grep -Fq "from hcu_verl import verl_adaptor" "${patch_target}"' in script
    assert "safe.directory" in script


def test_cleanup_only_targets_owned_processes():
    script = read_script("ci/cleanup.sh")

    assert "ray stop --force" in script
    assert '[[ -f "${run_dir}/ray-owned" ]]' in script
    assert "/proc/${candidate_pid}/environ" in script
    assert "VERL_HCU_CI_RUN_ID=${run_id}" in script
    assert 'tmp_root="$(realpath -m -- "${tmp_root}")"' in script
    assert '[[ "${tmp_root}" == "/" ]]' in script
    assert "pkill" not in script
    assert "killall" not in script


def test_nightly_runner_checks_exit_status_patch_and_fatal_logs():
    script = read_script("nightly/run_nightly_case.sh")

    assert '2>&1 | tee "${log_file}"' in script
    assert "case_status=${PIPESTATUS[0]}" in script
    assert "exec > >(" not in script
    assert "HCU_ADAPT" in script
    assert 'grep -Fq "step:5"' in script
    assert 'grep -Fq "step:1"' not in script
    assert (
        'check_environment.py" runtime --require-data-roots --require-gpus 8' in script
    )
    assert "torch.cuda.is_available()" in script
    assert "torch.cuda.device_count() == 8" in script
    for marker in (
        "Error executing job",
        "RayTaskError",
        "AcceleratorError",
        "out of memory",
        "OOM",
        "NaN",
        "WorkerCrashedError",
        "RayActorError",
        "ActorDiedError",
    ):
        assert marker in script
    assert "failure_pattern='Traceback|" not in script
    assert "worker[^[:cntrl:]]*" not in script
    assert "vllm)" in script
    assert "sglang)" in script
    assert "VERL_HCU_CI_IMAGE" in script
    assert "rev-parse HEAD" in script
    assert "submodule status" in script


def test_smoke_uses_real_torch_and_ray_gpu_resources():
    script = read_script("pr/run_hcu_smoke.sh")

    assert 'check_environment.py" runtime --require-gpus 8' in script
    assert "--require-data-roots" not in script
    assert "assert torch.cuda.is_available()" in script
    assert "assert torch.cuda.device_count() == 8" in script
    assert 'ray.init(address="local", include_dashboard=False)' in script
    assert "num_gpus=8" not in script
    assert 'resources.get("GPU", 0) >= 8' in script
    assert "@ray.remote(num_gpus=1)" in script
    assert "WorkerProbe.remote()" in script
    assert "range(8)" in script
    assert "get_accelerator_ids" in script
    assert "len(set(assignments)) == 8" in script
    assert "torch.cuda.device_count() >= 1" in script
    assert "verl.utils.device.get_visible_devices_keyword" in script
    assert "FileSystemWriterAsync.preload_tensors" in script
    assert "actual is expected" in script


def test_smoke_and_nightly_default_to_repository_ci_logs():
    smoke = read_script("pr/run_hcu_smoke.sh")
    nightly = read_script("nightly/run_nightly_case.sh")

    expected = "${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}"
    assert expected in smoke
    assert expected in nightly
    assert "environment.log" in smoke
    assert "pytest.log" in smoke
    assert "${VERL_HCU_CI_RUN_ID:-smoke-" in smoke
    assert "${VERL_HCU_CI_RUN_ID:-nightly-" in nightly


def test_e2e_scripts_use_pinned_baselines_local_roots_and_offline_mode():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_1step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_1step.sh")

    assert "third_party/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh" in vllm
    assert "Qwen2.5-0.5B-Instruct" in vllm
    assert "qwen2.5/Qwen2.5-0.5B-Instruct" in vllm
    assert "actor_rollout_ref.rollout.name" not in vllm
    assert "grpo_0.6b_gsm8k_fsdp2_sglang_2_6.sh" in sglang
    assert "Qwen3-0.6B" in sglang
    assert "qwen3/Qwen3-0.6B" in sglang
    assert "data.train_batch_size=12" in sglang
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=12" in sglang
    for script in (vllm, sglang):
        assert "VERL_HCU_MODEL_ROOT" in script
        assert "VERL_HCU_DATA_ROOT" in script
        assert "export PYTHONWARNINGS=ignore" in script
        assert "export TRANSFORMERS_VERBOSITY=error" in script
        assert "trainer.total_training_steps=5" in script
        assert "trainer.total_training_steps=1" not in script
        assert "trainer.total_epochs=5" in script
        assert "trainer.total_epochs=1" not in script
        assert "trainer.save_freq=-1" in script
        assert "HF_HUB_OFFLINE=1" in script
        assert "TRANSFORMERS_OFFLINE=1" in script
        assert "hf download" not in script
        assert "wget " not in script
        assert "curl " not in script


def test_nightly_cases_preserve_locally_validated_runtime_config():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_1step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_1step.sh")

    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.5" in vllm
    assert (
        "hydra.searchpath=[file://${REPO_ROOT}/third_party/verl/verl/trainer/config]"
        in sglang
    )


def test_ci_case_inventory_registers_requested_cases():
    pr_cases = (ROOT / "tests" / "hcu" / "pr" / "ci_cases.yaml").read_text(
        encoding="utf-8"
    )
    nightly_cases = (
        ROOT / "tests" / "hcu" / "nightly" / "bw1000" / "ci_cases.yaml"
    ).read_text(encoding="utf-8")

    assert "pr_smoke:" in pr_cases
    assert "nightly_" not in pr_cases
    assert "nightly_vllm:" in nightly_cases
    assert "nightly_sglang:" in nightly_cases


def test_nightly_sglang_does_not_start_after_workflow_cancellation():
    workflow = (ROOT / ".github" / "workflows" / "nightly-test-hcu.yml").read_text(
        encoding="utf-8"
    )

    assert "always() &&" in workflow
    assert "!cancelled() &&" in workflow


def test_pr_hcu_job_is_limited_to_same_repository_changes():
    workflow = (ROOT / ".github" / "workflows" / "pr-test-hcu.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request_target:" in workflow
    assert "\n  pull_request:\n" not in workflow
    assert "authorize-hcu:" in workflow
    assert (
        "check-changes:\n    name: Check changes\n    needs:\n      - authorize-hcu"
        in workflow
    )
    assert "needs.authorize-hcu.result" in workflow
    assert "github.event.pull_request.head.repo.full_name" in workflow
    assert "github.event.pull_request.head.sha" in workflow
    assert "github.repository" in workflow
    assert "github.event.pull_request.author_association" not in workflow
    assert "AUTHOR_ASSOCIATION" not in workflow
    assert "OWNER|MEMBER|COLLABORATOR" not in workflow


def test_pr_hcu_runtime_trigger_only_tracks_hcu_patch_tree():
    workflow = (ROOT / ".github" / "workflows" / "pr-test-hcu.yml").read_text(
        encoding="utf-8"
    )
    classifier = workflow.split("          runtime=false", maxsplit=1)[1].split(
        '          echo "runtime=${runtime}"', maxsplit=1
    )[0]

    assert "hcu_verl/*)" in classifier
    for unrelated_path in (
        "examples/*",
        "third_party/*",
        "requirements.txt",
        ".gitmodules",
        "tests/hcu/*",
        ".github/workflows/*",
    ):
        assert unrelated_path not in classifier


def test_smoke_and_nightly_apply_patch_before_verl_execution():
    smoke = read_script("pr/run_hcu_smoke.sh")
    nightly = read_script("nightly/run_nightly_case.sh")

    prepare_source = 'source "${CI_DIR}/prepare_workspace.sh"'
    assert smoke.index(prepare_source) < smoke.index("import verl")
    assert nightly.index(prepare_source) < nightly.index('bash "${case_script}"')
    assert "HCU_ADAPT" in smoke
    assert "HCU_ADAPT" in nightly


def test_workflows_validate_only_their_own_configuration_profile():
    pr_workflow = (ROOT / ".github" / "workflows" / "pr-test-hcu.yml").read_text(
        encoding="utf-8"
    )
    nightly_workflow = (
        ROOT / ".github" / "workflows" / "nightly-test-hcu.yml"
    ).read_text(encoding="utf-8")

    assert "check_environment.py config --profile pr" in pr_workflow
    assert "check_environment.py config --profile nightly" in nightly_workflow


def test_hcu_runtime_jobs_are_bound_to_bw1000_runners():
    pr_workflow = (ROOT / ".github" / "workflows" / "pr-test-hcu.yml").read_text(
        encoding="utf-8"
    )
    nightly_workflow = (
        ROOT / ".github" / "workflows" / "nightly-test-hcu.yml"
    ).read_text(encoding="utf-8")

    assert pr_workflow.count("\n      - bw1000\n") == 1
    assert nightly_workflow.count("\n      - bw1000\n") == 2
    assert nightly_workflow.count("VERL_HCU_ACCELERATOR: bw1000") == 2
    assert "name: BW1000" not in pr_workflow
    assert "name: BW1000" not in nightly_workflow
    workflow_readme = (ROOT / ".github" / "workflows" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "tests/hcu/nightly/bw1000/ci_cases.yaml" in workflow_readme


def test_ci_support_scripts_live_under_hcu_tests():
    legacy_dir = ROOT / "scripts" / "ci" / "hcu"
    assert not legacy_dir.exists() or not any(legacy_dir.rglob("*"))
    expected = (
        HCU_TEST_DIR / "ci" / "check_environment.py",
        HCU_TEST_DIR / "ci" / "check_pr_metadata.py",
        HCU_TEST_DIR / "ci" / "cleanup.sh",
        HCU_TEST_DIR / "ci" / "prepare_workspace.sh",
        HCU_TEST_DIR / "ci" / "verify_submodules.py",
        HCU_TEST_DIR / "pr" / "run_hcu_smoke.sh",
        HCU_TEST_DIR / "nightly" / "run_nightly_case.sh",
        HCU_TEST_DIR / "nightly" / "bw1000" / "run_vllm_grpo_1step.sh",
        HCU_TEST_DIR / "nightly" / "bw1000" / "run_sglang_off_policy_1step.sh",
    )
    assert all(path.is_file() for path in expected)

    nightly_runner = read_script("nightly/run_nightly_case.sh")
    assert 'accelerator="${VERL_HCU_ACCELERATOR:-bw1000}"' in nightly_runner
    assert 'case_dir="${SCRIPT_DIR}/${accelerator}"' in nightly_runner


def test_nightly_dispatch_preserves_pipeline_failures():
    script = read_script("nightly/run_nightly_case.sh")

    assert "set -euo pipefail" in script
    assert "case_status=${PIPESTATUS[0]}" in script
    assert 'exit "${case_status}"' in script


def test_vllm_case_avoids_unstable_hcu_memory_paths():
    script = read_script("nightly/bw1000/run_vllm_grpo_1step.sh")

    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False" in script
    assert "actor_rollout_ref.rollout.free_cache_engine=False" in script
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=False" in script


def test_nightly_cases_use_vendored_verl_baselines():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_1step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_1step.sh")

    assert "${REPO_ROOT}/third_party/verl/examples/" in vllm
    assert "${REPO_ROOT}/third_party/verl/verl/experimental/" in sglang
    assert "${REPO_ROOT}/examples/" not in vllm
    assert "${REPO_ROOT}/examples/" not in sglang


def test_hcu_ci_does_not_read_or_modify_product_examples():
    shell_examples_ref = "${REPO_ROOT}/" + "examples/"
    python_examples_ref = "ROOT / " + '"examples"'
    for subdir in ("ci", "pr", "nightly"):
        for path in (HCU_TEST_DIR / subdir).rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert shell_examples_ref not in text, path
            assert python_examples_ref not in text, path
