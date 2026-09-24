#!/usr/bin/env python3
"""Generate the frozen jump states for the reacting normal-shock gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from air5_postshock_reference import Air5PostShockReference
from air5_radau_reference import Air5RadauReference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outlet-reference-length", type=float)
    args = parser.parse_args()

    reference = Air5PostShockReference(Air5RadauReference(args.mechanism))
    outlet_header = ""
    if args.outlet_reference_length is not None:
        if not np.isfinite(args.outlet_reference_length) or args.outlet_reference_length <= 0.0:
            parser.error("--outlet-reference-length must be finite and positive")
        solution = reference.integrate(args.outlet_reference_length, rtol=1.0e-10)
        outlet = reference.sample(solution, np.array([args.outlet_reference_length])).points[0]
        outlet_header = f"outlet_pressure_pa={outlet.pressure:.17e}\n"
    rows = []
    for point in (reference.upstream, reference.postshock):
        conservative = reference.conservative_state(point)
        rows.append(
            np.r_[
                point.density,
                point.speed,
                point.pressure,
                point.temperature,
                point.tv,
                point.mass_fraction,
                conservative[10],
                conservative[4],
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        np.asarray(rows),
        fmt="%.17e",
        header=(
            "air5_normal_shock_states_v1\n"
            + outlet_header +
            "upstream row followed by frozen post-shock row\n"
            "rho u p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO Ev q5"
        ),
    )


if __name__ == "__main__":
    main()
