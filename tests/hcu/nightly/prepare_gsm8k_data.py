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

import argparse
from pathlib import Path


def required_copies(current_rows: int, minimum_rows: int) -> int:
    if current_rows <= 0:
        raise ValueError("input parquet is empty")
    if minimum_rows <= 0:
        raise ValueError("minimum row count must be positive")
    return max(1, (minimum_rows + current_rows - 1) // current_rows)


def expand_parquet(source: Path, destination: Path, minimum_rows: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if source.resolve() == destination.resolve():
        raise ValueError("source and destination parquet paths must differ")

    table = pq.read_table(source)
    copies = required_copies(table.num_rows, minimum_rows)
    expanded = pa.concat_tables([table] * copies).slice(0, minimum_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(expanded, destination)
    return expanded.num_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare writable GSM8K parquet files for HCU Nightly."
    )
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, required=True)
    parser.add_argument("--min-test-rows", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = expand_parquet(
        args.train_input,
        args.output_dir / "train.parquet",
        args.min_train_rows,
    )
    test_rows = expand_parquet(
        args.test_input,
        args.output_dir / "test.parquet",
        args.min_test_rows,
    )
    print(
        f"Prepared GSM8K CI data: train_rows={train_rows}, "
        f"test_rows={test_rows}, output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
