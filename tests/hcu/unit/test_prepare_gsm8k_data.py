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

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tests" / "hcu" / "nightly" / "prepare_gsm8k_data.py"


def load_helper():
    assert HELPER.is_file()
    spec = importlib.util.spec_from_file_location("prepare_gsm8k_data", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_copies_reaches_minimum_without_extra_full_copy():
    helper = load_helper()

    assert helper.required_copies(16, 1024) == 64
    assert helper.required_copies(16, 1152) == 72
    assert helper.required_copies(1152, 1152) == 1
    assert helper.required_copies(2048, 1152) == 1


def test_required_copies_rejects_empty_input():
    helper = load_helper()

    with pytest.raises(ValueError, match="input parquet is empty"):
        helper.required_copies(0, 1024)
