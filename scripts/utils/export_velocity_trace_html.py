#!/usr/bin/env python3
# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""Export an interactive velocity/reward trace HTML from a trace CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from velocity_trace_html import build_reward_signs, build_velocity_trace_html


def _parse_value(value: str):
    if value == "":
        return ""
    try:
        as_float = float(value)
    except ValueError:
        return value
    if as_float.is_integer() and value.strip().isdigit():
        return int(as_float)
    return as_float


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [{key: _parse_value(value) for key, value in row.items()} for row in reader]


def load_reward_signs(path: Path | None, rows: list[dict]) -> dict[str, int]:
    if path is None:
        return build_reward_signs(rows=rows)
    raw = json.loads(path.read_text())
    if all(str(key).startswith("reward_") for key in raw.keys()):
        return {str(key): int(value) for key, value in raw.items()}
    return build_reward_signs({str(key): float(value) for key, value in raw.items()}, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--html-path", type=Path, default=None)
    parser.add_argument(
        "--reward-signs-json",
        type=Path,
        default=None,
        help="JSON with either reward column signs or cfg reward scales.",
    )
    args = parser.parse_args()

    rows = load_csv(args.csv_path)
    html_path = args.html_path or args.csv_path.with_suffix(".html")
    reward_signs = load_reward_signs(args.reward_signs_json, rows)
    html_path.write_text(build_velocity_trace_html(rows, reward_signs))
    print(html_path)


if __name__ == "__main__":
    main()
