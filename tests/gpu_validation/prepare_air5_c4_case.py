#!/usr/bin/env python3
"""Prepare a dimensional fixed-air5 validation case."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from air5_hbl_reference_case import (
    generate_similarity_initial_field,
    write_astr_air5_initial_field,
)

from air5_htr_profile import (
    export_astr_air5_profile,
    load_htr_similarity_profile,
    map_htr_profile_to_air5,
)


def next_data_line(lines: list[str], marker_index: int) -> int:
    for index in range(marker_index + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return index
    raise ValueError(f"no data line follows line {marker_index + 1}")


def replace_after_marker(lines: list[str], marker: str, value: str) -> None:
    for index, line in enumerate(lines):
        if marker in line:
            lines[next_data_line(lines, index)] = value
            return
    raise ValueError(f"marker not found: {marker}")


def replace_boundary_types(lines: list[str], values: tuple[int | str, ...]) -> None:
    if len(values) != 6:
        raise ValueError("six boundary types are required")
    for index, line in enumerate(lines):
        if "# bctype" not in line:
            continue
        cursor = index
        for value in values:
            cursor = next_data_line(lines, cursor)
            lines[cursor] = str(value)
        return
    raise ValueError("marker not found: # bctype")


def prepare_case(
    source: Path,
    destination: Path,
    grid: str,
    maxstep: int,
    deltat: str,
    diffterm: str,
    lfilter: str,
    use_gpu: str,
    initial_condition: str,
    list_frequency: int = 1,
    hbl_initial_field: str = "uniform",
) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    datin = destination / "datin"
    shutil.copytree(source, datin)

    input_file = datin / "input.air5_c4"
    source_input = datin / "input.dim"
    source_input.rename(input_file)
    lines = input_file.read_text(encoding="utf-8").splitlines()
    if initial_condition == "species-wave":
        replace_after_marker(lines, "flowtype", "air5wave")
    elif initial_condition == "reactor":
        replace_after_marker(lines, "flowtype", "air5reactor")
    elif initial_condition == "high-temperature-tgv":
        replace_after_marker(lines, "flowtype", "air5tgv")
    elif initial_condition == "shock-tube":
        replace_after_marker(lines, "flowtype", "air5shocktube")
    elif initial_condition == "advection-wave":
        replace_after_marker(lines, "flowtype", "air5advection")
    elif initial_condition == "diffusion-layer":
        replace_after_marker(lines, "flowtype", "air5difflayer")
    elif initial_condition == "ev-pulse":
        replace_after_marker(lines, "flowtype", "air5evpulse")
    elif initial_condition == "postshock":
        replace_after_marker(lines, "flowtype", "air5postshock")
    elif initial_condition == "normal-shock":
        replace_after_marker(lines, "flowtype", "air5normalshock")
    elif initial_condition == "high-enthalpy-boundary-layer":
        replace_after_marker(lines, "flowtype", "air5hbl")
    replace_after_marker(lines, "im,jm,km", grid)
    replace_after_marker(
        lines,
        "nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb",
        f"f,{diffterm},{lfilter},f,f,f,t,t,{use_gpu}",
    )
    replace_after_marker(
        lines,
        "ref_tem,ref_vel,ref_len,ref_den",
        "1000.d0,10.d0,1.d-2,1.d0",
    )
    if initial_condition in ("postshock", "normal-shock"):
        replace_after_marker(lines, "lihomo,ljhomo,lkhomo", "f,t,t")
        replace_boundary_types(lines, ("11,free", 21, 1, 1, 1, 1))
        replace_after_marker(
            lines,
            "ref_tem,ref_vel,ref_len,ref_den",
            "500.d0,10.d0,2.d-2,1.d0",
        )
    elif initial_condition == "high-enthalpy-boundary-layer":
        replace_after_marker(lines, "lihomo,ljhomo,lkhomo", "f,f,t")
        replace_boundary_types(lines, ("11,free", 50, "41,2925.d0", 51, 1, 1))
        replace_after_marker(
            lines,
            "ref_tem,ref_vel,ref_len,ref_den",
            "450.d0,2905.07d0,4.41262150017878821d-5,0.781314d0",
        )
        source_profile = (
            Path(__file__).resolve().parent
            / "data/htr_multispecies_tbl_mach6_similarity.dat"
        )
        htr_profile = load_htr_similarity_profile(source_profile)
        air5_profile, metadata = map_htr_profile_to_air5(
            htr_profile, pressure=101325.0
        )
        export_astr_air5_profile(
            datin / "air5_hbl_profile.dat", air5_profile, metadata
        )
        if hbl_initial_field == "matched":
            dimensions = tuple(int(value) for value in grid.split(","))
            if len(dimensions) != 3 or min(dimensions) < 1:
                raise ValueError("HBL grid must contain three positive dimensions")
            ref_len = 4.41262150017878821e-5
            x = np.linspace(0.0, 20.0 * ref_len, dimensions[0] + 1)
            y = np.linspace(0.0, 8.0 * ref_len, dimensions[1] + 1)
            evidence = generate_similarity_initial_field(
                source_profile,
                Path(__file__).resolve().parents[2]
                / "chemMech/air5_kimjo12.json",
                x=x,
                y=y,
                pressure=101325.0,
            )
            write_astr_air5_initial_field(
                datin / "air5_hbl_initial_field.dat", evidence
            )
        elif hbl_initial_field != "uniform":
            raise ValueError("HBL initial field must be uniform or matched")
    replace_after_marker(lines, "conschm,difschm,rkscheme", "643e,643e,rk3")
    reconstruction = (
        "3,f,0.3d0,0.05d0"
        if initial_condition in ("shock-tube", "normal-shock")
        else "5,f,0.3d0,0.05d0"
    )
    replace_after_marker(
        lines,
        "recon_schem, lchardecomp,bfacmpld,shkcrt",
        reconstruction,
    )
    replace_after_marker(
        lines,
        "num_species",
        "5,1.d0,1.d0,1.d0,1.d0,1.d0",
    )
    input_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    controller = datin / "controller"
    lines = controller.read_text(encoding="utf-8").splitlines()
    replace_after_marker(
        lines,
        "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg",
        f"{maxstep},{max(maxstep + 1, 2)},10,50,{list_frequency},50",
    )
    replace_after_marker(lines, "deltat", deltat)
    controller.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return input_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--grid", default="15,15,15")
    parser.add_argument("--maxstep", type=int, default=1)
    parser.add_argument("--deltat", default="1.d-7")
    parser.add_argument("--list-frequency", type=int, default=1)
    parser.add_argument("--diffterm", choices=("t", "f"), default="f")
    parser.add_argument("--lfilter", choices=("t", "f"), default="f")
    parser.add_argument("--use-gpu", choices=("t", "f"), default="t")
    parser.add_argument(
        "--hbl-initial-field", choices=("uniform", "matched"), default="uniform"
    )
    parser.add_argument(
        "--initial-condition",
        choices=(
            "tgv",
            "species-wave",
            "reactor",
            "high-temperature-tgv",
            "shock-tube",
            "advection-wave",
            "diffusion-layer",
            "ev-pulse",
            "postshock",
            "normal-shock",
            "high-enthalpy-boundary-layer",
        ),
        default="tgv",
    )
    args = parser.parse_args()
    if args.maxstep < 0:
        raise ValueError("maxstep must be non-negative")
    if args.list_frequency <= 0:
        raise ValueError("list frequency must be positive")

    source = Path(__file__).resolve().parents[2] / "examples" / "Taylor_Green_Vortex_SI" / "datin"
    input_file = prepare_case(
        source,
        args.destination,
        args.grid,
        args.maxstep,
        args.deltat,
        args.diffterm,
        args.lfilter,
        args.use_gpu,
        args.initial_condition,
        args.list_frequency,
        args.hbl_initial_field,
    )
    print(input_file)


if __name__ == "__main__":
    main()
