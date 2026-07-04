#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingRow:
    case: str
    role: str
    np: int
    topology: str
    seconds: float
    out_dir: str


@dataclass(frozen=True)
class SpeedupRow:
    case: str
    np: int
    topology: str
    gpu_seconds: float
    speedup: float
    out_dir: str


def load_timings(path: Path) -> list[TimingRow]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"case", "role", "np", "topology", "seconds", "out_dir"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

        rows = []
        for raw in reader:
            rows.append(
                TimingRow(
                    case=raw["case"],
                    role=raw["role"],
                    np=int(raw["np"]),
                    topology=raw["topology"],
                    seconds=float(raw["seconds"]),
                    out_dir=raw["out_dir"],
                )
            )
    return rows


def baseline_seconds(rows: list[TimingRow], baseline_case: str) -> float:
    matches = [row.seconds for row in rows if row.case == baseline_case and row.role == "cpu"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one CPU baseline row for {baseline_case}, found {len(matches)}")
    return matches[0]


def compute_speedups(rows: list[TimingRow], baseline_case: str) -> list[SpeedupRow]:
    base = baseline_seconds(rows, baseline_case)
    gpu_rows = [row for row in rows if row.role == "gpu"]
    gpu_rows.sort(key=lambda row: (row.np, row.topology))
    return [
        SpeedupRow(
            case=row.case,
            np=row.np,
            topology=row.topology,
            gpu_seconds=row.seconds,
            speedup=base / row.seconds,
            out_dir=row.out_dir,
        )
        for row in gpu_rows
    ]


def write_summary(
    path: Path,
    *,
    baseline_seconds: float,
    speedups: list[SpeedupRow],
    grid: str,
    deltat: str,
    maxstep: str,
) -> None:
    lines = [
        "# Channel Benchmark Summary",
        "",
        "Parameters:",
        "",
        f"- `GRID={grid}`",
        f"- `DELTAT={deltat}`",
        f"- `MAXSTEP={maxstep}`",
        "- `LFILTER=f`",
        "- `DIFFTERM=t`",
        "- `CHANNEL_FORCE_MODE=fixed`",
        "",
        f"NP=1 CPU baseline: `{baseline_seconds:.3f} s`",
        "",
        "| case | NP | topology | GPU wall time (s) | speedup vs NP=1 CPU |",
        "|---|---:|---|---:|---:|",
    ]
    for row in speedups:
        lines.append(
            f"| {row.case} | {row.np} | {row.topology} | {row.gpu_seconds:.3f} | {row.speedup:.2f}x |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize channel CPU/GPU benchmark timings.")
    parser.add_argument("--timings", required=True, type=Path)
    parser.add_argument("--baseline-case", default="channel_np1_1x1x1")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--deltat", required=True)
    parser.add_argument("--maxstep", required=True)
    args = parser.parse_args()

    rows = load_timings(args.timings)
    base = baseline_seconds(rows, args.baseline_case)
    speedups = compute_speedups(rows, args.baseline_case)
    write_summary(
        args.summary,
        baseline_seconds=base,
        speedups=speedups,
        grid=args.grid,
        deltat=args.deltat,
        maxstep=args.maxstep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
