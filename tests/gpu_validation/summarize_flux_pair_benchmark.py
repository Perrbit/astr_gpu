#!/usr/bin/env python3
"""Summarize paired split and fused upwind-flux timings."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from summarize_tgv_performance import read_timings, relative_spread


def aggregate_timing(path: Path, expected_label: str) -> dict[str, float]:
    rows = read_timings(path)
    if len(rows) < 5:
        raise ValueError("flux-pair benchmark requires at least five repeats")
    if {row.label for row in rows} != {expected_label}:
        raise ValueError(f"expected only {expected_label} timing rows")
    repeats = sorted(row.repeat for row in rows)
    if repeats != list(range(1, len(rows) + 1)):
        raise ValueError(f"repeats are not contiguous: {repeats}")
    rk_samples = {row.rk_samples for row in rows}
    if len(rk_samples) != 1 or next(iter(rk_samples)) < 1:
        raise ValueError("each run must retain one positive RK sample count")
    medians = [row.median_rk_seconds for row in rows]
    median = statistics.median(medians)
    if median <= 0.0:
        raise ValueError("timing median must be positive")
    return {
        "median": median,
        "spread_percent": 100.0 * relative_spread(medians),
        "rk_samples": float(next(iter(rk_samples))),
    }


def summarize(split_path: Path, fused_path: Path, precision_mode: str) -> list[str]:
    split = aggregate_timing(split_path, "split")
    fused = aggregate_timing(fused_path, "fused")
    if split["rk_samples"] != fused["rk_samples"]:
        raise ValueError("split and fused runs retained different RK sample counts")
    speedup = split["median"] / fused["median"]
    reduction = 100.0 * (split["median"] - fused["median"]) / split["median"]
    return [
        "# TGV Upwind Flux-Pair Benchmark",
        "",
        f"Precision mode: `{precision_mode}`.",
        "",
        "| Metric | Split kernels | Fused pair kernel |",
        "|---|---:|---:|",
        f"| Median complete-RK time (s) | {split['median']:.9f} | {fused['median']:.9f} |",
        f"| Run-to-run spread | {split['spread_percent']:.3f}% | {fused['spread_percent']:.3f}% |",
        "",
        f"Fused complete-RK time reduction: `{reduction:.3f}%`",
        f"Split/fused speedup: `{speedup:.5f}x`",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--fused", required=True, type=Path)
    parser.add_argument("--precision-mode", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    lines = summarize(args.split, args.fused, args.precision_mode)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
