#!/usr/bin/env python3
"""Create the controlled S1 flat-plate input for explicit GPU validation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 4:
        raise argparse.ArgumentTypeError("grid dimensions must be at least 4")
    return parsed


def write_grid(
    path: Path,
    im: int,
    jm: int,
    km: int,
    x_min: float,
    x_max: float,
    y_stretch: float,
    z_length: float,
    warp_x: float,
    warp_y: float,
) -> np.ndarray:
    xline = np.linspace(x_min, x_max, im + 1)
    eta = np.linspace(0.0, 1.0, jm + 1)
    yline = np.expm1(y_stretch * eta) / np.expm1(y_stretch)
    zline = np.linspace(0.0, z_length, km + 1)
    x, y, z = np.meshgrid(xline, yline, zline, indexing="ij")
    streamwise = np.linspace(0.0, 1.0, im + 1)[:, None, None]
    wall_shape = y * (1.0 - y)
    x = x + warp_x * np.sin(np.pi * streamwise) * wall_shape
    y = y + warp_y * np.sin(2.0 * np.pi * streamwise) * wall_shape
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=np.asfortranarray(x).transpose(2, 1, 0))
        handle.create_dataset("y", data=np.asfortranarray(y).transpose(2, 1, 0))
        handle.create_dataset("z", data=np.asfortranarray(z).transpose(2, 1, 0))
    return yline


def write_profile(
    path: Path,
    yline: np.ndarray,
    delta: float,
    wall_temperature: float,
    isobaric: bool,
) -> None:
    velocity = 1.0 - np.exp(-((yline / delta) ** 2))
    temperature = 1.0 + (wall_temperature - 1.0) * np.exp(-((yline / delta) ** 2))
    density = 1.0 / temperature if isobaric else np.ones_like(temperature)
    with path.open("w", encoding="ascii") as handle:
        handle.write("S1-A0 flat-plate profile\n")
        handle.write("rho u v temperature\n")
        handle.write("0.08 0.026 0.018 0.01\n")
        handle.write("y rho u v temperature\n")
        for density_value, u, temperature_value in zip(density, velocity, temperature):
            handle.write(
                f"{density_value:.16e} {u:.16e} 0.0000000000000000e+00 {temperature_value:.16e}\n"
            )


def write_input(
    path: Path,
    im: int,
    jm: int,
    km: int,
    use_gpu: str,
    diffterm: str,
    lfilter: str,
    conschm: str,
    lchardecomp: str,
    shock_threshold: float,
    reynolds: float,
    mach: float,
    reference_temperature: float,
    wall_temperature: float,
    upper_bctype: int,
    x_min_bctype: int,
    ninit: int,
    sponge_im: int,
) -> None:
    path.write_text(
        f"""########################################################################
# Controlled S1 explicit three-dimensional flat-plate boundary layer
########################################################################

# flowtype
bl

# im,jm,km
{im},{jm},{km}

# lihomo,ljhomo,lkhomo
f,f,t

# nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb_input,use_gpu
t,{diffterm},{lfilter},t,f,f,f,f,{use_gpu}

# lrestar
f

# alfa_filter,kcutoff
0.49d0,48

# ref_t,reynolds,mach
{reference_temperature:.16e},{reynolds:.16e},{mach:.16e}

# conschm,difschm,rkscheme
{conschm},643e,rk3

# recon_schem,lchardecomp,bfacmpld,shkcrt
3,{lchardecomp},0.1d0,{shock_threshold:.16e}

# num_species
0

# turbmode,iomode
none,h

# bctype
{x_min_bctype},prof
21
41,{wall_temperature:.16e}
{upper_bctype}
1
1

# ninit
{ninit}

# spg_imin,spg_imax,spg_jmin,spg_jmax,spg_kmin,spg_kmax
0,{sponge_im},0,0,0,0

# gridfile
./datin/grid.flatplate.h5
""",
        encoding="ascii",
    )


def write_controller(path: Path, maxstep: int, feqchkpt: int) -> None:
    path.write_text(
        f"""############################################################
# Controlled S1-A0 controller
############################################################

# lwsequ,lwslic,lavg,lcracon
f,f,f,f

# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg
{maxstep},{feqchkpt},9999,9999,1,9999

# deltat
1.d-5
""",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst-case", required=True, type=Path)
    parser.add_argument("--use-gpu", required=True, choices=("t", "f"))
    parser.add_argument("--diffterm", choices=("t", "f"), default="t")
    parser.add_argument("--lfilter", choices=("t", "f"), default="f")
    parser.add_argument("--im", type=positive_int, default=64)
    parser.add_argument("--jm", type=positive_int, default=64)
    parser.add_argument("--km", type=positive_int, default=8)
    parser.add_argument("--conschm", choices=("643e", "543e"), default="643e")
    parser.add_argument("--lchardecomp", choices=("t", "f"), default="f")
    parser.add_argument("--shock-threshold", type=float, default=1.0e-3)
    parser.add_argument("--reynolds", type=float, default=1000.0)
    parser.add_argument("--mach", type=float, default=0.3)
    parser.add_argument("--reference-temperature", type=float, default=1.0)
    parser.add_argument("--wall-temperature", type=float, default=1.0)
    parser.add_argument("--x-min-bctype", choices=(11, 12), type=int, default=11)
    parser.add_argument("--upper-bctype", choices=(51, 52), type=int, default=51)
    parser.add_argument("--ninit", choices=(0, 3), type=int, default=0)
    parser.add_argument("--sponge-im", type=int, default=0)
    parser.add_argument("--profile-delta", type=float, default=0.08)
    parser.add_argument("--isobaric-profile", action="store_true")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-stretch", type=float, default=3.0)
    parser.add_argument("--z-length", type=float, default=0.25)
    parser.add_argument("--warp-x", type=float, default=0.0)
    parser.add_argument("--warp-y", type=float, default=0.0)
    parser.add_argument("--maxstep", type=int, default=2)
    parser.add_argument("--feqchkpt", type=int)
    args = parser.parse_args()
    if args.maxstep < 1:
        raise ValueError("--maxstep must be positive")
    if args.reynolds <= 0.0:
        raise ValueError("--reynolds must be positive")
    if args.mach <= 0.0:
        raise ValueError("--mach must be positive")
    if args.reference_temperature <= 0.0:
        raise ValueError("--reference-temperature must be positive")
    if args.wall_temperature <= 0.0:
        raise ValueError("--wall-temperature must be positive")
    if args.profile_delta <= 0.0:
        raise ValueError("--profile-delta must be positive")
    if args.x_max <= args.x_min:
        raise ValueError("--x-max must be greater than --x-min")
    if args.y_stretch <= 0.0:
        raise ValueError("--y-stretch must be positive")
    if args.z_length <= 0.0:
        raise ValueError("--z-length must be positive")
    if abs(args.warp_y) >= 1.0:
        raise ValueError("--warp-y magnitude must be below one to preserve wall-normal ordering")
    if args.sponge_im < 0 or args.sponge_im > args.im:
        raise ValueError("--sponge-im must lie in [0, im]")
    feqchkpt = args.maxstep if args.feqchkpt is None else args.feqchkpt
    if feqchkpt < 1:
        raise ValueError("--feqchkpt must be positive")

    if args.dst_case.exists():
        shutil.rmtree(args.dst_case)
    datin = args.dst_case / "datin"
    datin.mkdir(parents=True)
    yline = write_grid(
        datin / "grid.flatplate.h5",
        args.im,
        args.jm,
        args.km,
        args.x_min,
        args.x_max,
        args.y_stretch,
        args.z_length,
        args.warp_x,
        args.warp_y,
    )
    write_profile(
        datin / "inlet.prof",
        yline,
        args.profile_delta,
        args.wall_temperature,
        args.isobaric_profile,
    )
    write_input(
        datin / "input.flatplate",
        args.im,
        args.jm,
        args.km,
        args.use_gpu,
        args.diffterm,
        args.lfilter,
        args.conschm,
        args.lchardecomp,
        args.shock_threshold,
        args.reynolds,
        args.mach,
        args.reference_temperature,
        args.wall_temperature,
        args.upper_bctype,
        args.x_min_bctype,
        args.ninit,
        args.sponge_im,
    )
    write_controller(datin / "controller", args.maxstep, feqchkpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
