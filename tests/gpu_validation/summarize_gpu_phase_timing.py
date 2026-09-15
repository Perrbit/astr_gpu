#!/usr/bin/env python3
"""Validate and summarize complete per-rank GPU RK phase timing records."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics


PHASE_ORDER = (
    "prepare",
    "filter",
    "solution_halo",
    "convection",
    "diffusion_flux",
    "diffusion_halo",
    "diffusion_rhs",
    "rk_update",
)
PHASE_COUNTS = {
    "prepare": 1,
    "filter": 3,
    "solution_halo": 3,
    "convection": 3,
    "diffusion_flux": 3,
    "diffusion_halo": 3,
    "diffusion_rhs": 3,
    "rk_update": 3,
}


def _validate_dimensions(ranks: int, steps: int) -> None:
    if ranks < 1 or steps < 1:
        raise ValueError("ranks and steps must be positive")


def parse_phase_log(text: str, ranks: int, steps: int) -> dict[str, list[float]]:
    """Return the slowest-rank duration for every expected phase occurrence."""
    _validate_dimensions(ranks, steps)
    records: dict[tuple[int, int, int, str], float] = {}

    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "ASTR_GPU_PHASE_TIMING":
            continue
        if len(fields) != 6:
            raise ValueError(f"invalid phase timing record: {line}")
        try:
            rank, step, rkstep = map(int, fields[1:4])
            seconds = float(fields[5])
        except ValueError as error:
            raise ValueError(f"invalid phase timing record: {line}") from error

        label = fields[4]
        if label not in PHASE_COUNTS:
            raise ValueError(f"unknown phase label: {line}")
        if rank < 0 or rank >= ranks or step < 0 or step >= steps:
            raise ValueError(f"phase record outside expected range: {line}")
        if (label == "prepare" and rkstep != 0) or (
            label != "prepare" and rkstep not in range(1, 4)
        ):
            raise ValueError(f"invalid phase occurrence: {line}")
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"non-finite or negative phase time: {line}")

        key = (rank, step, rkstep, label)
        if key in records:
            raise ValueError(f"duplicate phase record: {line}")
        records[key] = seconds

    occurrences = [(0, "prepare")]
    occurrences.extend(
        (rkstep, label)
        for rkstep in range(1, 4)
        for label in PHASE_ORDER[1:]
    )
    samples = {label: [] for label in PHASE_ORDER}
    for step in range(steps):
        for rkstep, label in occurrences:
            values = []
            for rank in range(ranks):
                key = (rank, step, rkstep, label)
                if key not in records:
                    raise ValueError(f"missing phase record {key}")
                values.append(records[key])
            samples[label].append(max(values))

    expected = ranks * steps * sum(PHASE_COUNTS.values())
    if len(records) != expected:
        raise ValueError(f"expected {expected} phase records, found {len(records)}")
    return samples


def summarize_logs(
    paths: list[Path], ranks: int, steps: int
) -> list[tuple[str, int, float, float, float]]:
    """Aggregate completed runs without merging their record namespaces."""
    _validate_dimensions(ranks, steps)
    if not paths:
        raise ValueError("at least one phase timing log is required")

    combined = {label: [] for label in PHASE_ORDER}
    for path in paths:
        samples = parse_phase_log(path.read_text(encoding="utf-8"), ranks, steps)
        for label in PHASE_ORDER:
            combined[label].extend(samples[label])

    return [
        (label, len(values), statistics.median(values), min(values), max(values))
        for label, values in combined.items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        rows = summarize_logs(args.log, args.ranks, args.steps)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as stream:
        stream.write("phase\tsamples\tmedian_seconds\tmin_seconds\tmax_seconds\n")
        for label, count, median, minimum, maximum in rows:
            stream.write(
                f"{label}\t{count}\t{median:.12e}\t{minimum:.12e}\t{maximum:.12e}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
