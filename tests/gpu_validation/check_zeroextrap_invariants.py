#!/usr/bin/env python3
"""Check the CPU-defined ASTR zero-extrapolation semantics on both faces."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FIELD_NAMES = ("ro", "u1", "u2", "u3", "p", "t")
VELOCITY_NAMES = ("u1", "u2", "u3")


def read_fields(path: Path) -> dict[str, np.ndarray]:
    if path.is_dir():
        path = path / "outdat" / "flowfield.h5"
    with h5py.File(path, "r") as h5:
        missing = [name for name in FIELD_NAMES if name not in h5]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        return {name: np.asarray(h5[name]) for name in FIELD_NAMES}


def face(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis)


def extrapolation_error(array: np.ndarray, axis: int, side: str) -> float:
    if side == "lower":
        boundary = face(array, axis, 0)
        expected = (4.0 * face(array, axis, 1) - face(array, axis, 2)) / 3.0
    else:
        boundary = face(array, axis, -1)
        expected = (4.0 * face(array, axis, -2) - face(array, axis, -3)) / 3.0
    return float(np.max(np.abs(boundary - expected)))


def eos_error(
    fields: dict[str, np.ndarray], axis: int, side: str, const2: float
) -> float:
    index = 0 if side == "lower" else -1
    rho = face(fields["ro"], axis, index)
    pressure = face(fields["p"], axis, index)
    temperature = face(fields["t"], axis, index)
    return float(np.max(np.abs(rho * temperature / const2 - pressure)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--axis", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--mach", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    fields = read_fields(args.input)
    const2 = args.gamma * args.mach**2
    if args.axis == 2:
        # CPU x faces extrapolate p and T, then reconstruct rho from the EOS.
        extrapolated_names = (*VELOCITY_NAMES, "p", "t")
        reconstructed_name = "ro"
    else:
        # CPU y/z faces extrapolate p and rho, then reconstruct T from the EOS.
        extrapolated_names = (*VELOCITY_NAMES, "p", "ro")
        reconstructed_name = "t"

    lines = [
        "status: pass",
        f"axis: {args.axis}",
        f"gamma: {args.gamma:.16e}",
        f"mach: {args.mach:.16e}",
        f"const2: {const2:.16e}",
        f"directly_extrapolated: {','.join(extrapolated_names)}",
        f"eos_reconstructed: {reconstructed_name}",
        f"atol: {args.atol:.16e}",
        "",
        "side invariant linf",
    ]
    status = 0
    for side in ("lower", "upper"):
        for name in extrapolated_names:
            error = extrapolation_error(fields[name], args.axis, side)
            lines.append(f"{side} {name} {error:.16e}")
            if not np.isfinite(error) or error > args.atol:
                status = 1
        error = eos_error(fields, args.axis, side, const2)
        lines.append(f"{side} eos {error:.16e}")
        if not np.isfinite(error) or error > args.atol:
            status = 1

    if status:
        lines[0] = "status: fail"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
