#!/usr/bin/env python3
"""Summarize MP2 timings and enforce analytical workspace-byte accounting."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_tgv_candidate_performance import aggregate
from summarize_tgv_performance import read_timings


def summarize(
    fp64_path: Path,
    candidate_path: Path,
    fp64_workspace_bytes: int,
    mixed_workspace_bytes: int,
    expected_fp64_workspace_bytes: int,
    expected_mixed_workspace_bytes: int,
    phase: str = "MP2",
) -> list[str]:
    if phase not in {"MP2", "MP3"}:
        raise ValueError(f"unsupported mixed-precision phase: {phase}")
    if fp64_workspace_bytes != expected_fp64_workspace_bytes:
        raise ValueError("measured FP64 reference workspace bytes do not match the analytical value")
    if mixed_workspace_bytes != expected_mixed_workspace_bytes:
        raise ValueError("measured mixed workspace bytes do not match the analytical value")
    fp64_rows = read_timings(fp64_path)
    candidate_rows = read_timings(candidate_path)
    fp64 = aggregate(fp64_rows)
    candidate = aggregate(candidate_rows)
    candidate_name = candidate_rows[0].label
    if fp64["rk_samples"] != candidate["rk_samples"]:
        raise ValueError("FP64 and candidate runs retained different RK sample counts")
    speedup = fp64["median"] / candidate["median"]
    time_change = 100.0 * (candidate["median"] - fp64["median"]) / fp64["median"]
    reduction = 100.0 * (fp64_workspace_bytes - mixed_workspace_bytes) / fp64_workspace_bytes
    return [
        f"# {phase} {candidate_name} Workspace Benchmark",
        "",
        "This candidate has no speedup acceptance threshold.",
        "",
        f"| Metric | GPU FP64 | GPU {candidate_name} |",
        "|---|---:|---:|",
        f"| Median complete-RK time (s) | {fp64['median']:.9f} | {candidate['median']:.9f} |",
        f"| Run-to-run spread | {fp64['spread_percent']:.3f}% | {candidate['spread_percent']:.3f}% |",
        f"| Peak sampled device memory (MiB) | {fp64['max_memory_mib']:.0f} | {candidate['max_memory_mib']:.0f} |",
        *(
            [
                "| Peak sampled GPU utilization | "
                f"{max(row.max_utilization_percent for row in fp64_rows)}% | "
                f"{max(row.max_utilization_percent for row in candidate_rows)}% |"
            ]
            if phase == "MP3"
            else []
        ),
        f"| {candidate_name} workspace bytes | {fp64_workspace_bytes} | {mixed_workspace_bytes} |",
        "",
        f"{candidate_name}/FP64 complete-RK time change: `{time_change:+.3f}%`",
        f"FP64/{candidate_name} speedup: `{speedup:.5f}x`",
        f"{candidate_name}-workspace storage reduction: `{reduction:.3f}%`",
        "The timing result is descriptive; no speedup acceptance threshold is imposed.",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp64", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--fp64-workspace-bytes", required=True, type=int)
    parser.add_argument("--mixed-workspace-bytes", required=True, type=int)
    parser.add_argument("--expected-fp64-workspace-bytes", required=True, type=int)
    parser.add_argument("--expected-mixed-workspace-bytes", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--phase", default="MP2", choices=("MP2", "MP3"))
    args = parser.parse_args()
    lines = summarize(
        args.fp64,
        args.candidate,
        args.fp64_workspace_bytes,
        args.mixed_workspace_bytes,
        args.expected_fp64_workspace_bytes,
        args.expected_mixed_workspace_bytes,
        args.phase,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
