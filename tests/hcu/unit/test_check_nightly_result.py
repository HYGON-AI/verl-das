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

ROOT = Path(__file__).resolve().parents[3]
CHECKER = ROOT / "tests" / "hcu" / "ci" / "check_nightly_result.py"


def load_checker():
    assert CHECKER.is_file()
    spec = importlib.util.spec_from_file_location("check_nightly_result", CHECKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_valid_log_reaches_expected_step_without_checkpoint(tmp_path):
    checker = load_checker()
    log = (
        "[HCU_ADAPT] Patch has been applied in worker\n"
        "step:1 - training/global_step:1\n"
        "step:5 - training/global_step:5"
    )

    assert checker.validate_result(log, expected_step=5, checkpoint_root=tmp_path) == []


def test_missing_step_and_fatal_runtime_error_are_reported(tmp_path):
    checker = load_checker()
    log = (
        "[HCU_ADAPT] Patch has been applied in worker\n"
        "step:2 - training/global_step:2\n"
        "torch.OutOfMemoryError: HIP out of memory"
    )

    errors = checker.validate_result(log, expected_step=3, checkpoint_root=tmp_path)

    assert "missing training step marker: step:3" in errors
    assert "log contains a fatal runtime marker" in errors


def test_saved_checkpoint_is_rejected(tmp_path):
    checker = load_checker()
    (tmp_path / "checkpoints" / "global_step_3").mkdir(parents=True)
    log = "[HCU_ADAPT] Patch has been applied in worker\nstep:3"

    errors = checker.validate_result(log, expected_step=3, checkpoint_root=tmp_path)

    assert errors == ["training checkpoint was unexpectedly created"]
