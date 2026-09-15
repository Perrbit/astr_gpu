#!/usr/bin/env python3
"""Check the C4 unique-cell global conservation record."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


EXPECTED_COLUMNS = 21


def read_records(path: Path) -> list[list[float]]:
    records: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = [float(field) for field in stripped.split()]
        if len(values) != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}:{line_number}: expected {EXPECTED_COLUMNS} columns, "
                f"found {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite conservation value")
        records.append(values)
    if len(records) < 2:
        raise ValueError(f"{path}: expected an initial and at least one completed record")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-species-change", action="store_true")
    args = parser.parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("conservation tolerances must be non-negative")

    try:
        records = read_records(args.input)
        baseline = records[0]
        if int(baseline[1]) != 0:
            raise ValueError("first conservation record must have phase=0")
        completed = [record for record in records[1:] if int(record[1]) == 1]
        if not completed:
            raise ValueError("no completed phase=1 conservation record")

        mass_scale = max(abs(baseline[2]), 1.0)
        species_scale = max(max(abs(value) for value in baseline[7:12]), 1.0)
        max_closure = max(abs(record[15]) for record in completed)
        max_relative = max(
            abs(record[index]) for record in completed for index in range(16, 20)
        )
        max_species_drift = max(abs(record[20]) for record in completed)
        closure_limit = args.atol + args.rtol * mass_scale
        species_limit = args.atol + args.rtol * species_scale
        passed = (
            max_closure <= closure_limit
            and max_relative <= args.rtol
            and (args.allow_species_change or max_species_drift <= species_limit)
        )
        lines = [
            f"status: {'pass' if passed else 'fail'}",
            f"records: {len(records)}",
            f"max_species_mass_closure: {max_closure:.16e}",
            f"max_relative_mass_energy_element_drift: {max_relative:.16e}",
            f"max_species_absolute_drift: {max_species_drift:.16e}",
            "species_change_gate: "
            + ("disabled" if args.allow_species_change else "enabled"),
            f"closure_limit: {closure_limit:.16e}",
            f"relative_limit: {args.rtol:.16e}",
            f"species_limit: {species_limit:.16e}",
        ]
    except (OSError, ValueError) as error:
        passed = False
        lines = ["status: fail", f"error: {error}"]

    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
