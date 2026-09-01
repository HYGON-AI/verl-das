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
MODULE_PATH = ROOT / "tests" / "hcu" / "ci" / "check_environment.py"
CONFIG_ENV_VARS = (
    "VERL_HCU_MODEL_ROOT",
    "VERL_HCU_DATA_ROOT",
)


def load_module():
    spec = importlib.util.spec_from_file_location("check_environment", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def config_environment() -> dict[str, str]:
    return {
        "VERL_HCU_MODEL_ROOT": "/configuration/host/models",
        "VERL_HCU_DATA_ROOT": "/configuration/host/data",
    }


def runtime_environment(tmp_path: Path) -> dict[str, str]:
    model_root = tmp_path / "models"
    data_root = tmp_path / "data"
    model_root.mkdir()
    data_root.mkdir()
    return {
        "VERL_HCU_MODEL_ROOT": str(model_root),
        "VERL_HCU_DATA_ROOT": str(data_root),
    }


def test_config_inventory_is_exact():
    module = load_module()

    assert module.CONFIG_ENV_VARS == CONFIG_ENV_VARS


def test_validate_config_accepts_nonexistent_host_paths():
    module = load_module()

    assert module.validate_config(config_environment()) == []


@pytest.mark.parametrize("name", CONFIG_ENV_VARS)
def test_validate_config_requires_all_repository_variables(name):
    module = load_module()
    environment = config_environment()
    environment[name] = "  "

    errors = module.validate_config(environment)

    assert f"{name} is required" in errors


@pytest.mark.parametrize(
    "name",
    ["VERL_HCU_MODEL_ROOT", "VERL_HCU_DATA_ROOT"],
)
def test_validate_runtime_requires_existing_model_and_data_roots(tmp_path, name):
    module = load_module()
    environment = runtime_environment(tmp_path)
    environment[name] = str(tmp_path / "missing")

    errors = module.validate_runtime(environment, require_data_roots=True)

    assert any(name in error and "directory" in error for error in errors)


def test_validate_runtime_skips_data_roots_unless_requested():
    module = load_module()

    assert module.validate_runtime({}) == []


@pytest.mark.parametrize(
    "name",
    ["VERL_HCU_MODEL_ROOT", "VERL_HCU_DATA_ROOT"],
)
def test_validate_runtime_requires_nonempty_data_roots_when_requested(name):
    module = load_module()
    environment = {
        "VERL_HCU_MODEL_ROOT": "/models",
        "VERL_HCU_DATA_ROOT": "/data",
    }
    environment[name] = "  "

    errors = module.validate_runtime(environment, require_data_roots=True)

    assert f"{name} is required" in errors


def test_runtime_gpu_check_allows_unset_hip_visibility(tmp_path):
    module = load_module()
    environment = runtime_environment(tmp_path)

    assert module.validate_runtime(environment, required_gpus=8) == []


def test_runtime_gpu_check_validates_hip_count_when_set(tmp_path):
    module = load_module()
    environment = runtime_environment(tmp_path)
    environment["HIP_VISIBLE_DEVICES"] = "0,1,2,3"

    errors = module.validate_runtime(environment, required_gpus=8)

    assert any("8 HCU devices" in error and "found 4" in error for error in errors)


def test_runtime_gpu_check_validates_unique_hip_ids(tmp_path):
    module = load_module()
    environment = runtime_environment(tmp_path)
    environment["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,6"

    errors = module.validate_runtime(environment, required_gpus=8)

    assert any("unique" in error for error in errors)


def test_visible_device_ids_ignores_whitespace_and_empty_entries():
    module = load_module()

    assert module.visible_device_ids("0, 2,,7") == ("0", "2", "7")


@pytest.mark.parametrize("command", ["config", "runtime"])
def test_cli_subcommands_accept_valid_environment(tmp_path, command):
    environment = os.environ.copy()
    environment.update(config_environment())
    runtime = runtime_environment(tmp_path)
    environment.update(runtime)
    arguments = [sys.executable, str(MODULE_PATH), command]
    if command == "runtime":
        arguments.extend(["--require-data-roots", "--require-gpus", "8"])

    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert f"{command} validation passed" in result.stdout


def test_runtime_cli_does_not_require_unmounted_data_roots():
    environment = os.environ.copy()
    environment.pop("VERL_HCU_MODEL_ROOT", None)
    environment.pop("VERL_HCU_DATA_ROOT", None)
    environment.pop("HIP_VISIBLE_DEVICES", None)

    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "runtime",
            "--require-gpus",
            "8",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert "runtime validation passed" in result.stdout


def test_config_cli_accepts_legacy_profile_argument():
    environment = os.environ.copy()
    environment.update(config_environment())

    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "config", "--profile", "pr"],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 0
    assert "config validation passed" in result.stdout
