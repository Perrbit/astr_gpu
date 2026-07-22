#!/usr/bin/env python3
"""Check that an ASTR flowfield remains at a prescribed uniform state."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


PRIMITIVE_NAMES = ("ro", "u1", "u2", "u3", "p", "t")


def read_fields(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "outdat" / "flowfield.h5"
    with h5py.File(path, "r") as handle:
        missing = [name for name in PRIMITIVE_NAMES if name not in handle]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        return {name: handle[name][...] for name in PRIMITIVE_NAMES}


def check_fields(
    fields: dict[str, np.ndarray],
    expected: dict[str, float],
    atol: float,
    rtol: float,
) -> tuple[list[str], bool]:
    lines: list[str] = []
    passed = True
    for name in PRIMITIVE_NAMES:
        field = fields[name]
        target = expected[name]
        abs_error = np.abs(field - target)
        linf = float(np.nanmax(abs_error))
        l2 = float(np.sqrt(np.nanmean((field - target) ** 2)))
        spread = float(np.nanmax(field) - np.nanmin(field))
        allowed = max(atol, rtol * abs(target))
        if not np.all(np.isfinite(field)) or linf > allowed:
            passed = False
        lines.append(
            f"{name} linf={linf:.16e} l2={l2:.16e} spread={spread:.16e} "
            f"expected={target:.16e} allowed={allowed:.16e}"
        )
    return lines, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--u1", type=float, default=0.7)
    parser.add_argument("--u2", type=float, default=-0.2)
    parser.add_argument("--u3", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--mach", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    args = parser.parse_args()

    if args.density <= 0.0 or args.temperature <= 0.0:
        raise ValueError("density and temperature must be positive")
    if args.gamma <= 1.0 or args.mach <= 0.0:
        raise ValueError("gamma must exceed one and Mach must be positive")

    pressure = args.density * args.temperature / (args.gamma * args.mach**2)
    expected = {
        "ro": args.density,
        "u1": args.u1,
        "u2": args.u2,
        "u3": args.u3,
        "p": pressure,
        "t": args.temperature,
    }
    lines, passed = check_fields(read_fields(args.input), expected, args.atol, args.rtol)
    report_lines = [f"status: {'pass' if passed else 'fail'}", *lines]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
