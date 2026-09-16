#!/usr/bin/env python3
"""Compare matched TGV flowstate diagnostics from two ASTR runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


EXPECTED_HEADER = ["nstep", "time", "kenergy", "enstophy", "dissipation"]


def read_flowstate(path: Path) -> tuple[list[str], list[list[float]]]:
    flowstate = path / "flowstate.dat" if path.is_dir() else path
    lines = [line.split() for line in flowstate.read_text(encoding="ascii").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty flowstate file: {flowstate}")
    if lines[0] != EXPECTED_HEADER:
        raise ValueError(f"unexpected flowstate header in {flowstate}: {lines[0]}")
    rows = [[float(value) for value in row] for row in lines[1:]]
    if any(len(row) != len(EXPECTED_HEADER) for row in rows):
        raise ValueError(f"invalid flowstate row width in {flowstate}")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"non-finite flowstate value in {flowstate}")
    return lines[0], rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--min-rows", type=int, default=1)
    args = parser.parse_args()

    _, reference = read_flowstate(args.reference)
    _, candidate = read_flowstate(args.candidate)
    if len(reference) != len(candidate):
        raise ValueError(
            f"flowstate row-count mismatch: reference={len(reference)} candidate={len(candidate)}"
        )
    if len(reference) < args.min_rows:
        raise ValueError(f"flowstate has {len(reference)} rows, expected at least {args.min_rows}")

    maximum = {name: 0.0 for name in EXPECTED_HEADER[1:]}
    failures: list[str] = []
    for row_index, (ref_row, candidate_row) in enumerate(zip(reference, candidate, strict=True), 1):
        ref_step = int(ref_row[0])
        candidate_step = int(candidate_row[0])
        if ref_row[0] != ref_step or candidate_row[0] != candidate_step or ref_step != candidate_step:
            failures.append(
                f"row {row_index}: nstep mismatch reference={ref_row[0]} candidate={candidate_row[0]}"
            )
        for column, name in enumerate(EXPECTED_HEADER[1:], 1):
            ref_value = ref_row[column]
            candidate_value = candidate_row[column]
            difference = abs(candidate_value - ref_value)
            maximum[name] = max(maximum[name], difference)
            if not math.isclose(candidate_value, ref_value, rel_tol=args.rtol, abs_tol=args.atol):
                failures.append(
                    f"row {row_index} {name}: reference={ref_value:.16e} "
                    f"candidate={candidate_value:.16e} abs={difference:.16e}"
                )

    status = "fail" if failures else "pass"
    report = [
        f"status: {status}",
        f"rows: {len(reference)}",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
    ]
    report.extend(f"max_abs_{name}: {maximum[name]:.16e}" for name in EXPECTED_HEADER[1:])
    report.extend(failures)
    text = "\n".join(report) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="ascii")
    print(text, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
