#!/usr/bin/env python3
"""Validate and summarize repeated TGV GPU RK timing samples."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Timing:
    label: str
    repeat: int
    rk_samples: int
    median_rk_seconds: float
    min_rk_seconds: float
    max_rk_seconds: float
    wall_seconds: float
    max_memory_mib: int
    max_utilization_percent: int


def read_timings(path: Path) -> list[Timing]:
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = [
            "label",
            "repeat",
            "rk_samples",
            "median_rk_seconds",
            "min_rk_seconds",
            "max_rk_seconds",
            "wall_seconds",
            "max_memory_mib",
            "max_utilization_percent",
        ]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        rows = [
            Timing(
                label=row["label"],
                repeat=int(row["repeat"]),
                rk_samples=int(row["rk_samples"]),
                median_rk_seconds=float(row["median_rk_seconds"]),
                min_rk_seconds=float(row["min_rk_seconds"]),
                max_rk_seconds=float(row["max_rk_seconds"]),
                wall_seconds=float(row["wall_seconds"]),
                max_memory_mib=int(row["max_memory_mib"]),
                max_utilization_percent=int(row["max_utilization_percent"]),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"no timing rows in {path}")
    return rows


def relative_spread(values: list[float]) -> float:
    median = statistics.median(values)
    if median <= 0.0:
        raise ValueError("timing median must be positive")
    return (max(values) - min(values)) / median


def summarize(rows: list[Timing], grid: str, maxstep: int, discard_steps: int) -> list[str]:
    dimensions = [int(value) for value in grid.split(",")]
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise ValueError("grid must contain three positive dimensions")
    labels = {row.label for row in rows}
    if len(labels) != 1:
        raise ValueError(f"one benchmark label is required, found {sorted(labels)}")
    repeats = sorted(row.repeat for row in rows)
    if repeats != list(range(1, len(rows) + 1)):
        raise ValueError(f"repeats are not contiguous: {repeats}")
    expected_samples = maxstep + 1 - discard_steps
    if expected_samples < 1:
        raise ValueError("discard_steps removes every RK sample")
    if any(row.rk_samples != expected_samples for row in rows):
        raise ValueError(
            f"each run must contain {expected_samples} retained RK samples: "
            f"{[row.rk_samples for row in rows]}"
        )

    run_medians = [row.median_rk_seconds for row in rows]
    median = statistics.median(run_medians)
    spread = relative_spread(run_medians)
    cells = dimensions[0] * dimensions[1] * dimensions[2]
    throughput = cells / median
    label = rows[0].label
    return [
        "# TGV 256 GPU Performance Benchmark",
        "",
        f"- label: `{label}`",
        f"- grid: `{grid}` ({cells} cells)",
        f"- configured maxstep: `{maxstep}` ({maxstep + 1} RK advances)",
        f"- discarded in-process warm-up advances per run: `{discard_steps}`",
        f"- measured repeats after process warm-up: `{len(rows)}`",
        "- timing scope: first-stage preparation plus the remaining RK3 advance",
        "- checkpoint and full-field output: disabled during the measured loop",
        "",
        "| repeat | retained RK samples | median RK (s) | min/max RK (s) | wall (s) | peak device memory (MiB) | peak GPU utilization |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {row.repeat} | {row.rk_samples} | {row.median_rk_seconds:.9f} | "
            f"{row.min_rk_seconds:.9f} / {row.max_rk_seconds:.9f} | "
            f"{row.wall_seconds:.6f} | {row.max_memory_mib} | "
            f"{row.max_utilization_percent}% |"
            for row in rows
        ],
        "",
        f"Median complete-RK time across runs: `{median:.9f} s`",
        f"Run-to-run relative spread: `{100.0 * spread:.3f}%`",
        f"Median throughput: `{throughput:.6e} cell-RK/s`",
        f"Maximum sampled device memory: `{max(row.max_memory_mib for row in rows)} MiB`",
        f"Maximum sampled GPU utilization: `{max(row.max_utilization_percent for row in rows)}%`",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--maxstep", required=True, type=int)
    parser.add_argument("--discard-steps", required=True, type=int)
    args = parser.parse_args()
    lines = summarize(
        read_timings(args.timings), args.grid, args.maxstep, args.discard_steps
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
