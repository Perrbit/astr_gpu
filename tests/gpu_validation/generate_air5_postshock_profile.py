#!/usr/bin/env python3
"""Generate a deterministic fixed-air5 post-normal-shock reference profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from air5_postshock_reference import Air5PostShockReference
from air5_radau_reference import Air5RadauReference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument(
        "--source-mode", choices=("coupled", "chemical", "vt"), default="coupled"
    )
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise ValueError("profile requires at least two points")
    chemistry = Air5RadauReference(args.mechanism)
    reference = Air5PostShockReference(chemistry)
    result = reference.integrate(
        args.length, source_mode=args.source_mode, rtol=args.rtol
    )
    locations = np.linspace(0.0, args.length, args.points)
    profile = reference.sample(result, locations)
    rows = np.empty((args.points, 13))
    for index, point in enumerate(profile.points):
        conservative = reference.conservative_state(point)
        rows[index] = np.r_[
            locations[index],
            point.density,
            point.speed,
            point.pressure,
            point.temperature,
            point.tv,
            point.mass_fraction,
            conservative[10],
            conservative[4],
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        rows,
        fmt="%.17e",
        header=(
            "air5_postshock_profile_v1\n"
            f"source_mode={args.source_mode} length_m={args.length:.17e} "
            f"points={args.points} rtol={args.rtol:.17e}\n"
            "x rho u p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO Ev q5"
        ),
    )


if __name__ == "__main__":
    main()
