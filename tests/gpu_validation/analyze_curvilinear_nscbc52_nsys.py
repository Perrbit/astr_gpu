#!/usr/bin/env python3
"""Audit the CUDA work used by the curved non-reflecting bctype=52 path."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


RHS_KERNEL = "nscbc_farfield_y_upper_nonreflecting_rhs_kernel"
VISCOUS_CAPTURE_KERNEL = "nscbc_farfield_y_upper_prediff_capture_kernel"
VISCOUS_CORRECTION_KERNEL = "nscbc_farfield_y_upper_viscous_source_kernel"
FORBIDDEN_KERNELS = (
    "nscbc_farfield_y_upper_mach2_partial_kernel",
    "nscbc_farfield_y_upper_filter_x_kernel",
    "nscbc_farfield_y_upper_filter_z_kernel",
)


def analyze(
    input_path: Path, rk_steps: int, viscous: bool = False
) -> tuple[bool, list[str]]:
    if rk_steps <= 0:
        raise ValueError("rk-steps must be positive")
    with sqlite3.connect(input_path) as connection:
        rows = connection.execute(
            """
            SELECT strings.value, COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS strings ON strings.id = kernel.demangledName
            GROUP BY strings.value
            """
        ).fetchall()

    counts = {str(name): int(count) for name, count in rows}
    rhs_count = sum(count for name, count in counts.items() if RHS_KERNEL in name)
    forbidden = {
        query: sum(count for name, count in counts.items() if query in name)
        for query in FORBIDDEN_KERNELS
    }
    expected_rhs = 3 * rk_steps
    passed = rhs_count == expected_rhs and not any(forbidden.values())
    capture_count = sum(
        count for name, count in counts.items() if VISCOUS_CAPTURE_KERNEL in name
    )
    correction_count = sum(
        count
        for name, count in counts.items()
        if VISCOUS_CORRECTION_KERNEL in name
    )
    if viscous:
        passed = (
            passed
            and capture_count == expected_rhs
            and correction_count == expected_rhs
        )
    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"rk_steps: {rk_steps}",
        "rk_substages_per_step: 3",
        f"expected_nonreflecting_rhs_launches: {expected_rhs}",
        f"observed_nonreflecting_rhs_launches: {rhs_count}",
    ]
    if viscous:
        lines.extend(
            (
                f"expected_viscous_capture_launches: {expected_rhs}",
                f"observed_viscous_capture_launches: {capture_count}",
                f"expected_viscous_correction_launches: {expected_rhs}",
                f"observed_viscous_correction_launches: {correction_count}",
            )
        )
    lines.extend(
        f"forbidden_{query}: {forbidden[query]}" for query in FORBIDDEN_KERNELS
    )
    return passed, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--rk-steps", required=True, type=int)
    parser.add_argument("--viscous", action="store_true")
    args = parser.parse_args()

    passed, lines = analyze(args.input, args.rk_steps, viscous=args.viscous)
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="ascii")
    print(report, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
