#!/usr/bin/env python3
"""Check mixed subsonic/supersonic profile-inflow pressure treatment."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from check_profile_inflow11_invariants import FIELD_NAMES, flowfield_path, read_profile


def read_x_planes(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(flowfield_path(path), "r") as handle:
        return {name: handle[name][..., :3] for name in FIELD_NAMES}


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
    fields = read_x_planes(args.input)
    target = read_profile(args.profile, const2)
    normal_mach = target["u1"] / (np.sqrt(target["t"]) / args.mach)
    subsonic = normal_mach[1:-1] < 1.0
    supersonic = ~subsonic
    if not np.any(subsonic) or not np.any(supersonic):
        raise ValueError("profile must contain both subsonic and supersonic interior points")

    pressure_extrapolated = (4.0 * fields["p"][..., 1] - fields["p"][..., 2]) / 3.0
    expected_pressure = np.broadcast_to(target["p"][None, :], fields["p"][..., 0].shape).copy()
    expected_pressure[:, 1:-1][:, subsonic] = pressure_extrapolated[:, 1:-1][:, subsonic]
    expected_density = expected_pressure / target["t"][None, :] * const2

    lines = [
        "status: pass",
        f"atol: {args.atol:.16e}",
        f"subsonic_points: {int(np.count_nonzero(subsonic))}",
        f"supersonic_points: {int(np.count_nonzero(supersonic))}",
    ]
    passed = True
    expected = {
        "ro": expected_density,
        "u1": np.broadcast_to(target["u1"][None, :], expected_pressure.shape),
        "u2": np.broadcast_to(target["u2"][None, :], expected_pressure.shape),
        "u3": np.broadcast_to(target["u3"][None, :], expected_pressure.shape),
        "p": expected_pressure,
        "t": np.broadcast_to(target["t"][None, :], expected_pressure.shape),
    }
    for name in FIELD_NAMES:
        error = float(np.max(np.abs(fields[name][..., 0][:, 1:-1] - expected[name][:, 1:-1])))
        lines.append(f"{name}_xmin_expected_linf: {error:.16e}")
        passed = passed and error <= args.atol

    eos_error = float(
        np.max(
            np.abs(
                fields["ro"][..., 0][:, 1:-1] * fields["t"][..., 0][:, 1:-1]
                - const2 * fields["p"][..., 0][:, 1:-1]
            )
        )
    )
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
