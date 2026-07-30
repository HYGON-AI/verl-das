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

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "hcu_verl" / "utils" / "device.py"


def load_device_module(monkeypatch, npu_available):
    verl = types.ModuleType("verl")
    verl.__path__ = []
    utils = types.ModuleType("verl.utils")
    utils.__path__ = []
    device = types.ModuleType("verl.utils.device")
    device.is_torch_npu_available = lambda check_device=False: npu_available
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.utils", utils)
    monkeypatch.setitem(sys.modules, "verl.utils.device", device)

    spec = importlib.util.spec_from_file_location("hcu_device_for_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hip_visible_devices_takes_priority(monkeypatch):
    module = load_device_module(monkeypatch, npu_available=True)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")

    assert module.get_visible_devices_keyword() == "HIP_VISIBLE_DEVICES"


def test_cuda_keyword_is_used_when_hcu_visibility_is_unset(monkeypatch):
    module = load_device_module(monkeypatch, npu_available=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    assert module.get_visible_devices_keyword() == "CUDA_VISIBLE_DEVICES"


def test_ascend_keyword_is_preserved_for_npu(monkeypatch):
    module = load_device_module(monkeypatch, npu_available=True)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    assert module.get_visible_devices_keyword() == "ASCEND_RT_VISIBLE_DEVICES"
