#!/usr/bin/env python3
"""Check that bctype=11 replaces a stale x-min state with its inlet profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FIELD_NAMES = ("ro", "u1", "u2", "u3", "p", "t")


def flowfield_path(path: Path) -> Path:
    return path / "outdat" / "flowfield.h5" if path.is_dir() else path


def read_xmin(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(flowfield_path(path), "r") as handle:
        return {name: handle[name][..., 0] for name in FIELD_NAMES}


def read_profile(path: Path, const2: float) -> dict[str, np.ndarray]:
    with path.open("r", encoding="ascii") as handle:
        header = handle.readline().strip().lower()
    values = np.loadtxt(path, skiprows=4, ndmin=2)
    pressure_provided = "pressure=provided" in header
    density_provided = "density=provided" in header
    temperature = values[:, 3]
    pressure = values[:, 4] if pressure_provided else np.full_like(temperature, 1.0 / const2)
    density = values[:, 0] if density_provided else pressure / temperature * const2
    temperature_eos = pressure / density * const2
    return {
        "ro": density,
        "u1": values[:, 1],
        "u2": values[:, 2],
        "u3": np.zeros_like(density),
        "p": pressure,
        "t": temperature_eos,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--mach", required=True, type=float)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    const2 = args.gamma * args.mach**2
    fields = read_xmin(args.input)
    target = read_profile(args.profile, const2)
    lines = ["status: pass", f"atol: {args.atol:.16e}"]
    passed = True

    # y-endpoints are owned by the later y-wall/farfield boundary passes.
    for name in FIELD_NAMES:
        actual = fields[name][:, 1:-1]
        expected = target[name][None, 1:-1]
        error = float(np.max(np.abs(actual - expected)))
        lines.append(f"{name}_xmin_profile_linf: {error:.16e}")
        passed = passed and error <= args.atol

    eos_error = float(np.max(np.abs(fields["ro"][:, 1:-1] * fields["t"][:, 1:-1] - const2 * fields["p"][:, 1:-1])))
    lines.append(f"xmin_eos_linf: {eos_error:.16e}")
    passed = passed and eos_error <= args.atol

    if not passed:
        lines[0] = "status: fail"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
