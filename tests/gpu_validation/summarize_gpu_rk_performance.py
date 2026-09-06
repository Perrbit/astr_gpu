#!/usr/bin/env python3
"""Summarize repeated complete-RK timings for an ASTR GPU case."""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize_tgv_performance import read_timings, summarize as summarize_tgv


def summarize(
    rows,
    case_name: str,
    grid: str,
    maxstep: int,
    discard_steps: int,
    sync_mode: str = "explicit",
) -> list[str]:
    if not case_name or any(character.isspace() for character in case_name):
        raise ValueError("case_name must be one non-empty token")
    lines = summarize_tgv(rows, grid, maxstep, discard_steps, sync_mode)
    lines[0] = "# ASTR GPU RK Performance Benchmark"
    lines.insert(2, f"- case: `{case_name}`")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--maxstep", required=True, type=int)
    parser.add_argument("--discard-steps", required=True, type=int)
    parser.add_argument("--sync-mode", choices=("explicit", "selective"), default="explicit")
    args = parser.parse_args()
    lines = summarize(
        read_timings(args.timings),
        args.case_name,
        args.grid,
        args.maxstep,
        args.discard_steps,
        args.sync_mode,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
