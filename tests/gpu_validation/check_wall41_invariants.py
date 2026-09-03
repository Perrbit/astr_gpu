#!/usr/bin/env python3
"""Check nonreacting isothermal no-slip wall invariants in ASTR HDF5 output."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FIELD_NAMES = ("ro", "u1", "u2", "u3", "p", "t")


def read_fields(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "outdat" / "flowfield.h5"
    with h5py.File(path, "r") as h5:
        missing = [name for name in FIELD_NAMES if name not in h5]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        return {name: np.asarray(h5[name]) for name in FIELD_NAMES}


def face(array: np.ndarray, axis: int, side: int) -> np.ndarray:
    index = 0 if side == 0 else array.shape[axis] - 1
    return np.take(array, index, axis=axis)


def max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--side", choices=("lower", "upper", "both"), default="both")
    parser.add_argument("--wall-temperature", required=True, type=float)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--mach", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    fields = read_fields(args.input)
    const2 = args.gamma * args.mach * args.mach
    const6 = 1.0 / (args.gamma - 1.0)
    lines = [
        "status: pass",
        f"axis: {args.axis}",
        f"atol: {args.atol:.16e}",
        f"wall_temperature: {args.wall_temperature:.16e}",
        "",
        "side invariant linf",
    ]
    status = 0

    sides = ((0, "lower"), (1, "upper"))
    if args.side != "both":
        sides = tuple(item for item in sides if item[1] == args.side)
    for side, side_name in sides:
        rho = face(fields["ro"], args.axis, side)
        u1 = face(fields["u1"], args.axis, side)
        u2 = face(fields["u2"], args.axis, side)
        u3 = face(fields["u3"], args.axis, side)
        prs = face(fields["p"], args.axis, side)
        tmp = face(fields["t"], args.axis, side)

        q1 = rho
        q2 = rho * u1
        q3 = rho * u2
        q4 = rho * u3
        q5 = prs * const6 + 0.5 * rho * (u1 * u1 + u2 * u2 + u3 * u3)
        checks = (
            ("velocity", max(max_abs(u1), max_abs(u2), max_abs(u3))),
            ("temperature", max_abs(tmp - args.wall_temperature)),
            ("eos", max_abs(rho * tmp - const2 * prs)),
            ("q1_density", max_abs(q1 - rho)),
            ("q2", max_abs(q2)),
            ("q3", max_abs(q3)),
            ("q4", max_abs(q4)),
            ("q5_internal_energy", max_abs(q5 - const6 * prs)),
        )
        for name, error in checks:
            lines.append(f"{side_name} {name} {error:.16e}")
            if not np.isfinite(error) or error > args.atol:
                status = 1

    if status:
        lines[0] = "status: fail"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
