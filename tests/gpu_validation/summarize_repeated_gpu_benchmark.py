#!/usr/bin/env python3
"""Summarize repeated GPU wall times and device-level monitoring samples."""

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
    np: int
    topology: str
    seconds: float
    max_memory_mib: int
    max_utilization_percent: int


def read_timings(path: Path) -> list[Timing]:
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "label",
            "repeat",
            "np",
            "topology",
            "seconds",
            "max_memory_mib",
            "max_utilization_percent",
        }
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        rows = [
            Timing(
                label=row["label"],
                repeat=int(row["repeat"]),
                np=int(row["np"]),
                topology=row["topology"],
                seconds=float(row["seconds"]),
                max_memory_mib=int(row["max_memory_mib"]),
                max_utilization_percent=int(row["max_utilization_percent"]),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"no timings in {path}")
    return rows


def summarize(rows: list[Timing], grid: str, maxstep: int) -> list[str]:
    dimensions = [int(value) for value in grid.split(",")]
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise ValueError("grid must contain three positive dimensions")
    cells = dimensions[0] * dimensions[1] * dimensions[2]
    rk_advances = maxstep + 1
    groups: dict[str, list[Timing]] = {}
    for row in rows:
        groups.setdefault(row.label, []).append(row)
    for label, group in groups.items():
        repeats = sorted(row.repeat for row in group)
        if repeats != list(range(1, len(group) + 1)):
            raise ValueError(f"{label} repeats are not contiguous: {repeats}")

    lines = [
        "# Curvilinear C10 Repeated GPU Benchmark",
        "",
        f"- grid: `{grid}` ({cells} cells)",
        f"- configured maxstep: `{maxstep}`",
        f"- actual RK advances from nstep=0 through maxstep: `{rk_advances}`",
        "- checkpoint frequency: `9999`",
        "- full-field initialization/output is included in wall time",
        "- memory and utilization are device-level samples, not process-exclusive counters",
        "",
        "| case | NP | topology | repeats | wall time min/median/max (s) | median cell-RK/s | max device memory (MiB) | max GPU utilization |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    medians: dict[int, float] = {}
    for label in sorted(groups, key=lambda value: groups[value][0].np):
        group = groups[label]
        seconds = [row.seconds for row in group]
        median = statistics.median(seconds)
        medians[group[0].np] = median
        throughput = cells * rk_advances / median
        lines.append(
            f"| {label} | {group[0].np} | {group[0].topology} | {len(group)} | "
            f"{min(seconds):.3f} / {median:.3f} / {max(seconds):.3f} | "
            f"{throughput:.6e} | {max(row.max_memory_mib for row in group)} | "
            f"{max(row.max_utilization_percent for row in group)}% |"
        )
    if 1 in medians and 2 in medians:
        scaling = medians[1] / medians[2]
        lines.extend(
            [
                "",
                f"NP=2 speedup relative to NP=1 GPU: `{scaling:.4f}x`",
                f"NP=2 parallel efficiency relative to ideal 2x: `{0.5 * scaling:.4f}`",
            ]
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--maxstep", required=True, type=int)
    args = parser.parse_args()
    lines = summarize(read_timings(args.timings), args.grid, args.maxstep)
    args.summary.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
