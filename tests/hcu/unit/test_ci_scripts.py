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


def test_nightly_scripts_check_exit_status_patch_and_fatal_logs():
    scripts = (
        read_script("nightly/bw1000/run_vllm_grpo_5step.sh"),
        read_script("nightly/bw1000/run_sglang_off_policy_5step.sh"),
    )

    for script in scripts:
        assert '2>&1 | tee "${log_file}"' in script
        assert "case_status=${PIPESTATUS[0]}" in script
        assert "exec > >(" not in script
        assert "HCU_ADAPT" in script
        assert 'grep -Fq "step:5"' in script
        assert 'grep -Fq "step:1"' not in script
        assert (
            'check_environment.py" runtime --require-data-roots --require-gpus 8'
            in script
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
    assert "verl.utils.device.get_visible_devices_keyword" not in script
    assert "FileSystemWriterAsync.preload_tensors" in script
    assert "actual is expected" in script


def test_smoke_and_nightly_default_to_repository_ci_logs():
    smoke = read_script("pr/run_hcu_smoke.sh")
    nightly_scripts = (
        read_script("nightly/bw1000/run_vllm_grpo_5step.sh"),
        read_script("nightly/bw1000/run_sglang_off_policy_5step.sh"),
    )

    expected = "${REPO_ROOT}/ci-logs/${VERL_HCU_CI_RUN_ID}"
    assert expected in smoke
    assert "environment.log" in smoke
    assert "pytest.log" in smoke
    assert "${VERL_HCU_CI_RUN_ID:-smoke-" in smoke
    for script in nightly_scripts:
        assert expected in script
        assert "${VERL_HCU_CI_RUN_ID:-nightly-" in script


def test_e2e_scripts_use_ci_local_roots_and_offline_mode():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_5step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_5step.sh")

    assert "Qwen2.5-0.5B-Instruct" in vllm
    assert "qwen2.5/Qwen2.5-0.5B-Instruct" in vllm
    assert "Qwen3-0.6B" in sglang
    assert "qwen3/Qwen3-0.6B" in sglang
    for script in (vllm, sglang):
        assert "VERL_HCU_MODEL_ROOT" in script
        assert "VERL_HCU_DATA_ROOT" in script
        assert "export PYTHONWARNINGS=ignore" in script
        assert "export TRANSFORMERS_VERBOSITY=error" in script
        assert "trainer.total_training_steps=5" in script
        assert "trainer.total_training_steps=1" not in script
        assert "trainer.save_freq=-1" in script
        assert "HF_HUB_OFFLINE=1" in script
        assert "TRANSFORMERS_OFFLINE=1" in script
        assert "hf download" not in script
        assert "wget " not in script
        assert "curl " not in script


def test_nightly_cases_match_repository_example_parameters():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_5step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_5step.sh")

    vllm_parameters = (
        "data.train_batch_size=1024",
        "data.max_prompt_length=512",
        "data.max_response_length=1024",
        "actor_rollout_ref.actor.ppo_mini_batch_size=256",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "trainer.total_epochs=15",
    )
    sglang_parameters = (
        "data.train_batch_size=1152",
        "data.max_prompt_length=512",
        "data.max_response_length=1024",
        "actor_rollout_ref.actor.ppo_mini_batch_size=192",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32",
        "actor_rollout_ref.actor.fsdp_config.strategy=fsdp2",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "rollout.n_gpus_per_node=2",
        "trainer.n_gpus_per_node=6",
        "trainer.total_epochs=2",
    )
    for parameter in vllm_parameters:
        assert parameter in vllm
    for parameter in sglang_parameters:
        assert parameter in sglang


def test_vllm_case_uses_hcu_runtime_for_example_sleep_mode():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_5step.sh")

    assert "VLLM_CUDART_SO_PATH=/opt/dtk/hip/lib/libgalaxyhip.so" in vllm
    assert "actor_rollout_ref.rollout.free_cache_engine=True" in vllm
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=True" in vllm
    assert "actor_rollout_ref.rollout.free_cache_engine=False" not in vllm
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=False" not in vllm


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
    nightly_scripts = (
        read_script("nightly/bw1000/run_vllm_grpo_5step.sh"),
        read_script("nightly/bw1000/run_sglang_off_policy_5step.sh"),
    )

    prepare_source = 'source "${CI_DIR}/prepare_workspace.sh"'
    assert smoke.index(prepare_source) < smoke.index("import verl")
    assert "HCU_ADAPT" in smoke
    for script in nightly_scripts:
        assert script.index(prepare_source) < script.index("python3 -m verl")
        assert "HCU_ADAPT" in script


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
    assert "VERL_HCU_ACCELERATOR" not in nightly_workflow
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
        HCU_TEST_DIR / "nightly" / "bw1000" / "run_vllm_grpo_5step.sh",
        HCU_TEST_DIR / "nightly" / "bw1000" / "run_sglang_off_policy_5step.sh",
    )
    assert all(path.is_file() for path in expected)
    assert not (HCU_TEST_DIR / "nightly" / "run_nightly_case.sh").exists()


def test_nightly_workflow_runs_five_step_scripts_directly():
    workflow = (ROOT / ".github" / "workflows" / "nightly-test-hcu.yml").read_text(
        encoding="utf-8"
    )

    assert "bash tests/hcu/nightly/bw1000/run_vllm_grpo_5step.sh" in workflow
    assert "bash tests/hcu/nightly/bw1000/run_sglang_off_policy_5step.sh" in workflow
    assert "run_nightly_case.sh" not in workflow
    assert "5-step" in workflow


def test_nightly_cases_use_standalone_ci_training_commands():
    vllm = read_script("nightly/bw1000/run_vllm_grpo_5step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_5step.sh")

    assert "python3 -m verl.trainer.main_ppo" in vllm
    assert "python3 -m verl.experimental.one_step_off_policy.main_ppo" in sglang
    assert "${REPO_ROOT}/examples/" not in vllm
    assert "${REPO_ROOT}/examples/" not in sglang
    assert "EXAMPLE_SCRIPT" not in vllm
    assert "EXAMPLE_SCRIPT" not in sglang


def test_nightly_cases_do_not_modify_repository_examples():
    shell_examples_ref = "${REPO_ROOT}/" + "examples/"
    python_examples_ref = "ROOT / " + '"examples"'
    for subdir in ("ci", "pr", "nightly"):
        for path in (HCU_TEST_DIR / subdir).rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert shell_examples_ref not in text, path
            assert python_examples_ref not in text, path


def test_nightly_cases_expand_read_only_fixture_data_for_example_batches():
    helper = HCU_TEST_DIR / "nightly" / "prepare_gsm8k_data.py"
    assert helper.is_file()

    vllm = read_script("nightly/bw1000/run_vllm_grpo_5step.sh")
    sglang = read_script("nightly/bw1000/run_sglang_off_policy_5step.sh")
    for script in (vllm, sglang):
        assert 'SOURCE_TRAIN_FILE="${VERL_HCU_DATA_ROOT' in script
        assert 'SOURCE_TEST_FILE="${VERL_HCU_DATA_ROOT' in script
        assert 'data_dir="${run_dir}/data"' in script
        assert '"${REPO_ROOT}/tests/hcu/nightly/prepare_gsm8k_data.py"' in script
        assert '--output-dir "${data_dir}"' in script
        assert 'TRAIN_FILE="${data_dir}/train.parquet"' in script
        assert 'TEST_FILE="${data_dir}/test.parquet"' in script

    assert "--min-train-rows 1024" in vllm
    assert "--min-train-rows 3456" in sglang
