#!/usr/bin/env python3
"""Summarize paired FP64 and mixed-workspace complete-RK timings."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_tgv_candidate_performance import aggregate
from summarize_tgv_performance import read_timings


def summarize(
    fp64_path: Path,
    mixed_path: Path,
    fp64_workspace_bytes: int,
    mixed_workspace_bytes: int,
) -> list[str]:
    fp64 = aggregate(read_timings(fp64_path))
    mixed = aggregate(read_timings(mixed_path))
    if fp64["rk_samples"] != mixed["rk_samples"]:
        raise ValueError("FP64 and mixed runs retained different RK sample counts")
    speedup = fp64["median"] / mixed["median"]
    time_change = 100.0 * (mixed["median"] - fp64["median"]) / fp64["median"]
    workspace_reduction = 100.0 * (
        fp64_workspace_bytes - mixed_workspace_bytes
    ) / fp64_workspace_bytes
    return [
        "# TGV Upwind Mixed-Precision Benchmark",
        "",
        "This report measures the candidate without imposing a speedup acceptance threshold.",
        "",
        "| Metric | FP64 workspace | Mixed workspace |",
        "|---|---:|---:|",
        f"| Median complete-RK time (s) | {fp64['median']:.9f} | {mixed['median']:.9f} |",
        f"| Run-to-run spread | {fp64['spread_percent']:.3f}% | {mixed['spread_percent']:.3f}% |",
        f"| Peak sampled device memory (MiB) | {fp64['max_memory_mib']:.0f} | {mixed['max_memory_mib']:.0f} |",
        f"| Flux workspace bytes per rank | {fp64_workspace_bytes} | {mixed_workspace_bytes} |",
        "",
        f"Mixed/FP64 complete-RK time change: `{time_change:+.3f}%`",
        f"FP64/mixed speedup: `{speedup:.5f}x`",
        f"Flux-workspace storage reduction: `{workspace_reduction:.3f}%`",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp64", required=True, type=Path)
    parser.add_argument("--mixed", required=True, type=Path)
    parser.add_argument("--fp64-workspace-bytes", required=True, type=int)
    parser.add_argument("--mixed-workspace-bytes", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    lines = summarize(
        args.fp64,
        args.mixed,
        args.fp64_workspace_bytes,
        args.mixed_workspace_bytes,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
