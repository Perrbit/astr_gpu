#!/usr/bin/env python3
"""Compare matched HBL wall and profile diagnostics stored in NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


QUANTITIES = (
    "cf_geometric",
    "qw_geometric",
    "profile_u",
    "profile_temperature",
)


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as handle:
        missing = [name for name in QUANTITIES if name not in handle]
        if missing:
            raise KeyError(f"{path}: missing diagnostics {missing}")
        values = {name: np.asarray(handle[name], dtype=np.float64) for name in QUANTITIES}
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise RuntimeError(f"{path}: non-finite HBL diagnostics")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    reference = load(args.reference)
    candidate = load(args.candidate)
    lines = [
        "status: pending",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        "",
        "quantity max_abs max_rel",
    ]
    passed = True
    for name in QUANTITIES:
        if reference[name].shape != candidate[name].shape:
            raise RuntimeError(
                f"{name}: shape mismatch {reference[name].shape} != {candidate[name].shape}"
            )
        absolute = np.abs(candidate[name] - reference[name])
        scale = np.maximum(np.abs(reference[name]), 1.0)
        relative = absolute / scale
        quantity_passed = bool(np.all((absolute <= args.atol) | (relative <= args.rtol)))
        passed = passed and quantity_passed
        lines.append(f"{name} {absolute.max():.16e} {relative.max():.16e}")
    lines[0] = f"status: {'pass' if passed else 'fail'}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
