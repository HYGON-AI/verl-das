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
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "hcu_verl" / "adaptor" / "patch_utils.py"


@pytest.fixture
def patch_utils_module():
    spec = importlib.util.spec_from_file_location(
        "hcu_patch_utils_for_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_replaces_function_and_is_idempotent(monkeypatch, patch_utils_module):
    target_module = types.ModuleType("hcu_patch_target")
    target_module.value = lambda: "original"
    monkeypatch.setitem(sys.modules, target_module.__name__, target_module)

    patch = patch_utils_module.Patch(
        "hcu_patch_target.value",
        lambda: "patched",
        create_dummy=False,
    )
    patch.apply_patch()
    first_replacement = target_module.value
    patch.apply_patch()

    assert target_module.value() == "patched"
    assert target_module.value is first_replacement
    assert patch.is_applied is True


def test_patch_can_create_missing_module(monkeypatch, patch_utils_module):
    parent = types.ModuleType("hcu_patch_parent")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, parent.__name__, parent)

    patch = patch_utils_module.Patch(
        "hcu_patch_parent.generated.value",
        lambda: "created",
        create_dummy=True,
    )
    patch.apply_patch()

    assert sys.modules["hcu_patch_parent.generated"].value() == "created"
