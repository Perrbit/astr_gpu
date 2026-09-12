#!/usr/bin/env python3
"""Freeze MP3 field and statistics tolerances from the NP=1 pilot."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FIELD_PATTERN = re.compile(r"\blinf=([+\-0-9.eE]+)")


def read_field_max(path: Path) -> float:
    values = [
        float(match.group(1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := FIELD_PATTERN.search(line)) is not None
    ]
    if not values:
        raise ValueError(f"{path}: no field linf values")
    return max(values)


def read_stats_max(path: Path) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(
            "metric max_abs max_rel final_cpu final_gpu final_abs final_rel"
        ) + 1
    except ValueError as exc:
        raise ValueError(f"{path}: missing statistics metric table") from exc
    values: list[float] = []
    for line in lines[start:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "first_failure:":
            break
        if len(fields) != 7:
            raise ValueError(f"{path}: malformed statistics row: {line}")
        values.append(float(fields[1]))
    if not values:
        raise ValueError(f"{path}: no statistics max_abs values")
    return max(values)


def frozen_tolerance(observed: float, floor: float, ceiling: float) -> float:
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError("observed error must be finite and nonnegative")
    target = max(floor, 10.0 * observed)
    exponent = math.floor(math.log10(target))
    decade = 10.0**exponent
    scaled = target / decade
    frozen = 10.0 * decade
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        if scaled <= mantissa * (1.0 + 1.0e-12):
            frozen = mantissa * decade
            break
    if frozen > ceiling:
        raise ValueError(
            f"frozen tolerance {frozen:.1e} exceeds ceiling {ceiling:.1e}"
        )
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-report", required=True, type=Path)
    parser.add_argument("--stats-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    field_observed = read_field_max(args.field_report)
    stats_observed = read_stats_max(args.stats_report)
    field_tolerance = frozen_tolerance(field_observed, 1.0e-12, 1.0e-5)
    stats_tolerance = frozen_tolerance(stats_observed, 1.0e-12, 1.0e-5)
    lines = [
        f"CANDIDATE_FIELD_ATOL={field_tolerance:.1e}",
        "CANDIDATE_FIELD_RTOL=0",
        f"CANDIDATE_STATS_ATOL={stats_tolerance:.1e}",
        "CANDIDATE_STATS_RTOL=0",
        f"MP3_PILOT_FIELD_MAX={field_observed:.16e}",
        f"MP3_PILOT_STATS_MAX={stats_observed:.16e}",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
