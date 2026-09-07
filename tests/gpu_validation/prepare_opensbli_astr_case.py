#!/usr/bin/env python3
"""Create a runnable ASTR case from the isolated OpenSBLI reference assets."""

import argparse
import json
from pathlib import Path
import shutil

from prepare_opensbli_reference_inputs import prepare as prepare_reference
from prepare_s1_flatplate_case import write_controller, write_input


def prepare_case(destination, use_gpu, nx, ny, nz, maxstep, feqchkpt, deltat, prandtl):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing ASTR case: {destination}")
    if min(nx, ny) < 12 or nz < 9:
        raise ValueError("OpenSBLI ASTR case requires nx/ny >= 12 and nz >= 9")
    assets = destination / "reference_assets"
    report = prepare_reference(assets, nx=nx, ny=ny, nz=nz, prandtl=prandtl)
    datin = destination / "datin"
    datin.mkdir()
    write_input(
        datin / "input.opensbli", nx - 1, ny - 1, nz - 1, use_gpu,
        "t", "f", "543e", "t", 1.0e-3, 950.0, 2.0, 288.0,
        1.676194, 52, 11, 3, 0,
    )
    write_controller(datin / "controller", maxstep, feqchkpt, deltat)
    shutil.copyfile(assets / "grid.h5", datin / "grid.flatplate.h5")
    shutil.copyfile(assets / "initial.h5", datin / "flowini3d.h5")
    shutil.copyfile(assets / "profile.dat", datin / "inlet.prof")
    shutil.copyfile(assets / "conservative_boundary.nml", datin / "conservative_boundary.nml")
    case_manifest = {
        "status": "RUNNABLE_UNVALIDATED",
        "reference_manifest": report,
        "input": "datin/input.opensbli",
        "boundary_config": "datin/conservative_boundary.nml",
        "maxstep": maxstep,
        "feqchkpt": feqchkpt,
        "deltat": deltat,
        "use_gpu": use_gpu,
    }
    (destination / "case_manifest.json").write_text(
        json.dumps(case_manifest, indent=2) + "\n", encoding="ascii"
    )
    return case_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--use-gpu", required=True, choices=("t", "f"))
    parser.add_argument("--nx", type=int, default=33)
    parser.add_argument("--ny", type=int, default=33)
    parser.add_argument("--nz", type=int, default=9)
    parser.add_argument("--maxstep", type=int, default=1)
    parser.add_argument("--feqchkpt", type=int, default=1)
    parser.add_argument("--deltat", type=float, default=1.0e-6)
    parser.add_argument("--prandtl", type=float, choices=(0.71, 0.72), default=0.72)
    args = parser.parse_args()
    if args.maxstep < 1 or args.feqchkpt < 1 or args.deltat <= 0.0:
        raise ValueError("Need positive maxstep, feqchkpt, and deltat")
    print(json.dumps(prepare_case(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
