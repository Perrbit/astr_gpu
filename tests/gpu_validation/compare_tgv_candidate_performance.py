#!/usr/bin/env python3
"""Compare a TGV GPU optimization candidate with a frozen timing baseline."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from summarize_tgv_performance import Timing, read_timings, relative_spread


def aggregate(rows: list[Timing]) -> dict[str, float]:
    if len(rows) < 5:
        raise ValueError("performance acceptance requires at least five repeats")
    labels = {row.label for row in rows}
    if len(labels) != 1:
        raise ValueError(f"one benchmark label is required, found {sorted(labels)}")
    repeats = sorted(row.repeat for row in rows)
    if repeats != list(range(1, len(rows) + 1)):
        raise ValueError(f"repeats are not contiguous: {repeats}")
    rk_samples = {row.rk_samples for row in rows}
    if len(rk_samples) != 1 or next(iter(rk_samples)) < 1:
        raise ValueError("each run must use one positive RK sample count")
    medians = [row.median_rk_seconds for row in rows]
    median = statistics.median(medians)
    if median <= 0.0:
        raise ValueError("timing median must be positive")
    max_memory_mib = max(row.max_memory_mib for row in rows)
    if max_memory_mib <= 0:
        raise ValueError("peak device memory must be positive")
    return {
        "median": median,
        "spread_percent": 100.0 * relative_spread(medians),
        "max_memory_mib": float(max_memory_mib),
        "rk_samples": float(next(iter(rk_samples))),
    }


def compare(
    baseline_rows: list[Timing],
    candidate_rows: list[Timing],
    min_time_reduction_percent: float,
    max_spread_percent: float,
    max_memory_growth_percent: float,
) -> tuple[list[str], bool]:
    if min_time_reduction_percent < 0.0:
        raise ValueError("minimum time reduction must be non-negative")
    if max_spread_percent < 0.0 or max_memory_growth_percent < 0.0:
        raise ValueError("spread and memory-growth limits must be non-negative")
    baseline = aggregate(baseline_rows)
    candidate = aggregate(candidate_rows)
    if baseline["rk_samples"] != candidate["rk_samples"]:
        raise ValueError(
            "baseline and candidate retained RK sample counts differ: "
            f"{baseline['rk_samples']:.0f} != {candidate['rk_samples']:.0f}"
        )
    time_reduction = 100.0 * (
        baseline["median"] - candidate["median"]
    ) / baseline["median"]
    speedup = baseline["median"] / candidate["median"]
    memory_growth = 100.0 * (
        candidate["max_memory_mib"] - baseline["max_memory_mib"]
    ) / baseline["max_memory_mib"]

    time_pass = time_reduction >= min_time_reduction_percent
    spread_pass = candidate["spread_percent"] <= max_spread_percent
    memory_pass = memory_growth <= max_memory_growth_percent
    passed = time_pass and spread_pass and memory_pass

    lines = [
        "# TGV GPU Candidate Performance Comparison",
        "",
        "| Metric | Baseline | Candidate | Acceptance | Result |",
        "|---|---:|---:|---:|---|",
        f"| Median complete-RK time (s) | {baseline['median']:.9f} | "
        f"{candidate['median']:.9f} | reduction >= "
        f"{min_time_reduction_percent:.3f}% | {'pass' if time_pass else 'fail'} |",
        f"| Run-to-run spread | {baseline['spread_percent']:.3f}% | "
        f"{candidate['spread_percent']:.3f}% | <= {max_spread_percent:.3f}% | "
        f"{'pass' if spread_pass else 'fail'} |",
        f"| Peak sampled memory (MiB) | {baseline['max_memory_mib']:.0f} | "
        f"{candidate['max_memory_mib']:.0f} | growth <= "
        f"{max_memory_growth_percent:.3f}% | {'pass' if memory_pass else 'fail'} |",
        "",
        f"Time reduction: `{time_reduction:.3f}%`",
        f"Speedup: `{speedup:.5f}x`",
        f"Peak-memory growth: `{memory_growth:.3f}%`",
        f"Overall status: `{'pass' if passed else 'fail'}`",
    ]
    return lines, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-time-reduction-percent", type=float, default=3.0)
    parser.add_argument("--max-spread-percent", type=float, default=5.0)
    parser.add_argument("--max-memory-growth-percent", type=float, default=5.0)
    args = parser.parse_args()

    lines, passed = compare(
        read_timings(args.baseline),
        read_timings(args.candidate),
        args.min_time_reduction_percent,
        args.max_spread_percent,
        args.max_memory_growth_percent,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
