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

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ADAPTOR = ROOT / "hcu_verl" / "adaptor" / "verl_adaptor.py"
EXPECTED_PATCH_TARGETS = {
    "verl.plugin.platform.platform_rocm.PlatformROCm.rollout_env_vars",
    "verl.trainer.constants_ppo.PPO_RAY_RUNTIME_ENV",
    "verl.utils.flops_counter._DEVICE_FLOPS",
    (
        "megatron.core.dist_checkpointing.strategies.filesystem_async."
        "FileSystemWriterAsync.preload_tensors"
    ),
}
EXPECTED_IMPLEMENTATIONS = {
    "hcu_verl/plugin/platform/platform_rocm.py",
    "hcu_verl/trainer/constants_ppo.py",
    "hcu_verl/utils/flops_counter.py",
    "hcu_verl/core/dist_checkpointing/strategies/filesystem_async.py",
}


def registered_patch_targets() -> set[str]:
    tree = ast.parse(ADAPTOR.read_text(encoding="utf-8"), filename=str(ADAPTOR))
    targets = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "register"
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            targets.add(node.args[0].value)
    return targets


def test_patch_target_inventory_is_explicit_and_complete():
    assert registered_patch_targets() == EXPECTED_PATCH_TARGETS


def test_patch_implementation_inventory_exists():
    missing = [path for path in EXPECTED_IMPLEMENTATIONS if not (ROOT / path).is_file()]

    assert missing == []
