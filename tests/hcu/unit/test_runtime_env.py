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
MODULE_PATH = ROOT / "hcu_verl" / "trainer" / "constants_ppo.py"


def load_constants(monkeypatch, capability=(9, 0)):
    verl = types.ModuleType("verl")
    verl.__path__ = []
    utils = types.ModuleType("verl.utils")
    utils.__path__ = []
    device = types.ModuleType("verl.utils.device")
    device.get_device_capability = lambda: capability
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.utils", utils)
    monkeypatch.setitem(sys.modules, "verl.utils.device", device)
    monkeypatch.setenv("VERL_PATH", str(ROOT))
    monkeypatch.delenv("NET_TYPE", raising=False)

    spec = importlib.util.spec_from_file_location("hcu_constants_for_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runtime_env_contains_hcu_worker_settings(monkeypatch):
    module = load_constants(monkeypatch)

    env = module.PPO_RAY_RUNTIME_ENV["env_vars"]
    assert env["TORCH_CPP_LOG_LEVEL"] == "fatal"
    assert env["GLOG_minloglevel"] == "3"
    assert env["GPU_MAX_HW_QUEUES"] == "10"
    assert env["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"
    assert env["HSA_NO_SCRATCH_RECLAIM"] == "1"


def test_runtime_env_applies_selected_network_settings(monkeypatch):
    module = load_constants(monkeypatch)
    monkeypatch.setenv("NET_TYPE", "shca")

    module.apply_env(module.ENV)

    env = module.PPO_RAY_RUNTIME_ENV["env_vars"]
    assert env["NCCL_NET_PLUGIN"] == "shca"
    assert env["NCCL_SOCKET_IFNAME"] == "ib0"


def test_runtime_env_rejects_unknown_network_type(monkeypatch):
    module = load_constants(monkeypatch)
    monkeypatch.setenv("NET_TYPE", "unknown")

    try:
        module.apply_env(module.ENV)
    except AssertionError as error:
        assert "Expected NET_TYPE" in str(error)
    else:
        raise AssertionError("unknown NET_TYPE should be rejected")
