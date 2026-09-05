#!/usr/bin/env python3
"""Summarize one completed CURVE-C21 aggregate regression directory."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def update_max(current, value: float, label: str, path: Path):
    if not math.isfinite(value):
        raise ValueError(f"non-finite value in {path}: {label}={value}")
    if current is None or value > current[0]:
        return value, label, path
    return current


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.input.resolve()
    if not root.is_dir():
        raise SystemExit(f"missing C21 output directory: {root}")

    stage_path = root / "c21_stage_summary.tsv"
    stages = []
    for line in stage_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        stage, status, seconds = line.split("\t")
        stages.append((stage, status, int(seconds)))
    failed = [stage for stage, status, _ in stages if status != "pass"]
    if failed:
        raise SystemExit(f"non-passing C21 stages: {', '.join(failed)}")

    field_max = None
    field_reports = list(root.rglob("flowfield_compare.txt"))
    field_pattern = re.compile(rf"^(\w+)\s+linf=({FLOAT})(?:\s|$)")
    for path in field_reports:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = field_pattern.match(line)
            if match:
                field_max = update_max(
                    field_max, float(match.group(2)), match.group(1), path
                )

    stat_max = None
    stat_reports = list(root.rglob("flowstate_compare.txt"))
    stat_pattern = re.compile(rf"^(\w+)\s+({FLOAT})(?:\s|$)")
    for path in stat_reports:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = stat_pattern.match(line)
            if match and match.group(1) not in {"nstep", "time"}:
                stat_max = update_max(
                    stat_max, float(match.group(2)), match.group(1), path
                )

    boundary_max = None
    invariant_reports = list(root.rglob("*invariants.txt"))
    row_pattern = re.compile(rf"^(lower|upper)\s+(\w+)\s+({FLOAT})(?:\s|$)")
    key_pattern = re.compile(rf"^([A-Za-z0-9_]+):\s*({FLOAT})\s*$")
    def is_residual_key(key: str) -> bool:
        return (
            key.endswith("_profile_linf")
            or key == "xmin_eos_linf"
            or key.endswith("_discrete_normal_velocity_linf")
            or key.endswith("_zero")
            or key.endswith("_target")
        )

    for path in invariant_reports:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = row_pattern.match(line)
            if match:
                boundary_max = update_max(
                    boundary_max,
                    abs(float(match.group(3))),
                    f"{match.group(1)} {match.group(2)}",
                    path,
                )
                continue
            match = key_pattern.match(line)
            if match and is_residual_key(match.group(1)):
                boundary_max = update_max(
                    boundary_max,
                    abs(float(match.group(2))),
                    match.group(1),
                    path,
                )

    jacobian_min = None
    finite_checks = 0
    for path in root.rglob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(rf"^jacobian_min:\s*({FLOAT})\s*$", line)
            if match:
                value = float(match.group(1))
                if not math.isfinite(value) or value <= 0.0:
                    raise SystemExit(f"non-positive Jacobian in {path}: {value}")
                if jacobian_min is None or value < jacobian_min[0]:
                    jacobian_min = value, path
            match = re.match(r"^finite(?:_fields)?:\s*(\w+)\s*$", line)
            if match:
                if match.group(1).lower() != "true":
                    raise SystemExit(f"failed finite-field check in {path}: {line}")
                finite_checks += 1

    memcheck_logs = [root / f"c21_memcheck_{axis}.log" for axis in "xyz"]
    for path in memcheck_logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "ERROR SUMMARY: 0 errors" not in text:
            raise SystemExit(f"memcheck did not report zero errors: {path}")

    residency_path = root / "c15_residency" / "residency_report.txt"
    residency = read_key_values(residency_path)
    if residency.get("status") != "pass":
        raise SystemExit(f"residency report did not pass: {residency_path}")

    benchmark_path = root / "c15_benchmark" / "benchmark_summary.md"
    benchmark = benchmark_path.read_text(encoding="utf-8")
    np1 = re.search(r"\| np1 .*?\| ([0-9.]+ / [0-9.]+ / [0-9.]+) \|", benchmark)
    np2 = re.search(r"\| np2_x .*?\| ([0-9.]+ / [0-9.]+ / [0-9.]+) \|", benchmark)
    speedup = re.search(r"speedup relative to NP=1 GPU: `([0-9.]+x)`", benchmark)
    efficiency = re.search(r"parallel efficiency relative to ideal 2x: `([0-9.]+)`", benchmark)
    if not all((np1, np2, speedup, efficiency)):
        raise SystemExit(f"cannot parse benchmark summary: {benchmark_path}")

    if field_max is None or stat_max is None or boundary_max is None:
        raise SystemExit("missing field, statistic, or boundary comparison evidence")

    git_head = (root / "git_head.txt").read_text(encoding="utf-8").strip()
    solver_diff_sha = (root / "solver_diff_sha256.txt").read_text(
        encoding="utf-8"
    ).strip()
    solver_status = (root / "solver_git_status.txt").read_text(
        encoding="utf-8"
    ).strip()
    solver_status_inline = "; ".join(solver_status.splitlines()) or "clean"
    binary_hashes = (root / "binary_sha256.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    if len(binary_hashes) != 2:
        raise SystemExit("expected exactly two CPU/GPU binary hashes")
    started = (root / "run_started.txt").read_text(encoding="utf-8").strip()
    completed_path = root / "run_completed.txt"
    completed = (
        completed_path.read_text(encoding="utf-8").strip()
        if completed_path.exists()
        else "not recorded"
    )
    total_seconds = sum(seconds for _, _, seconds in stages)

    lines = [
        "# CURVE-C21 Aggregate Regression Report",
        "",
        "- status: `pass`",
        f"- git HEAD: `{git_head}`",
        f"- tracked solver diff SHA-256: `{solver_diff_sha}`",
        f"- solver worktree status: `{solver_status_inline}`",
        f"- CPU binary SHA-256: `{binary_hashes[0].split()[0]}`",
        f"- GPU binary SHA-256: `{binary_hashes[1].split()[0]}`",
        f"- started: `{started}`",
        f"- completed: `{completed}`",
        f"- stages: `{len(stages)}/{len(stages)}` passed in `{total_seconds} s`",
        f"- field reports: `{len(field_reports)}`",
        f"- statistic reports: `{len(stat_reports)}`",
        f"- boundary invariant reports: `{len(invariant_reports)}`",
        f"- explicit finite-field checks: `{finite_checks}`",
        "",
        "## Aggregate Maxima",
        "",
        "| quantity | maximum | field/metric | evidence |",
        "|---|---:|---|---|",
        f"| CPU/GPU field $L_\\infty$ | `{field_max[0]:.16e}` | `{field_max[1]}` | `{relative(field_max[2], root)}` |",
        f"| CPU/GPU statistic max abs | `{stat_max[0]:.16e}` | `{stat_max[1]}` | `{relative(stat_max[2], root)}` |",
        f"| boundary invariant residual | `{boundary_max[0]:.16e}` | `{boundary_max[1]}` | `{relative(boundary_max[2], root)}` |",
        "",
        "## Geometry And Runtime Safety",
        "",
        f"- minimum checked numerical Jacobian: `{jacobian_min[0]:.16e}` from `{relative(jacobian_min[1], root)}`",
        "- representative x/y/z Compute Sanitizer memchecks: `3/3`, each `ERROR SUMMARY: 0 errors`",
        "- required curved open-boundary rejection cases: covered by passing `c20_open_boundaries` stage",
        "- C11 global centered filter on the discontinuous shock case remains a documented closed-negative policy gate and was not rerun",
        "",
        "## Compute-Loop Residency",
        "",
        f"- kernels after the selected first RHS kernel: `{residency['kernels_after_start']}`",
        f"- transfers at or above {residency['large_transfer_threshold_bytes']} B: `{residency['large_h2d_d2h_count']}`",
        f"- H2D: `{residency['h2d_count']}` operations, `{residency['h2d_total_bytes']} B` total, `{residency['h2d_max_bytes']} B` maximum",
        f"- D2H: `{residency['d2h_count']}` operations, `{residency['d2h_total_bytes']} B` total, `{residency['d2h_max_bytes']} B` maximum",
        "",
        "## Current-Machine Performance",
        "",
        "The benchmark uses a `256^3` C10 case, six actual RK advances, three timed repeats, and includes initialization and final output.",
        "",
        f"- GPU NP=1 min/median/max: `{np1.group(1)} s`",
        f"- GPU NP=2 x-slab min/median/max: `{np2.group(1)} s`",
        f"- NP=2 speedup relative to GPU NP=1: `{speedup.group(1)}`",
        f"- NP=2 parallel efficiency: `{float(efficiency.group(1)) * 100.0:.2f}%`",
        "",
        "## Scope Boundary",
        "",
        "This closes the tested static single-block curvilinear, nonreacting, explicit-scheme capability. It does not validate arbitrary curved characteristic/open boundaries, moving or multi-block grids, GPU HDF5, compact schemes, species/chemistry, RANS/LES, immersed boundaries, physical SBLI fidelity, or scaling beyond the available two GPUs.",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
