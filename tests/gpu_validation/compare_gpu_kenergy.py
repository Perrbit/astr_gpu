#!/usr/bin/env python3
"""Compare CPU flowstate kenergy with GPU scalar-reduction output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_flowstate(path: Path) -> tuple[list[str], np.ndarray]:
    fpath = path / "flowstate.dat" if path.is_dir() else path
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    with fpath.open() as fh:
        header = fh.readline().split()
    data = np.loadtxt(fpath, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def read_gpu_kenergy(path: Path) -> tuple[list[str], np.ndarray]:
    fpath = path / "gpu_kenergy.dat" if path.is_dir() else path
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    with fpath.open() as fh:
        header = fh.readline().split()
    data = np.loadtxt(fpath, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    args = parser.parse_args()

    cpu_header, cpu = read_flowstate(args.cpu)
    gpu_header, gpu = read_gpu_kenergy(args.gpu)
    if gpu_header != ["nstep", "time", "kenergy_gpu"]:
        raise ValueError(f"unexpected GPU kenergy header: {gpu_header}")

    nstep_col = cpu_header.index("nstep")
    time_col = cpu_header.index("time")
    kenergy_col = cpu_header.index("kenergy")

    status = 0
    lines = [
        "status: pass",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        "",
        "nstep time cpu_kenergy gpu_kenergy abs rel",
    ]

    cpu_by_step = {int(row[nstep_col]): row for row in cpu}
    max_abs = 0.0
    max_rel = 0.0
    for row in gpu:
        nstep = int(row[0])
        if nstep not in cpu_by_step:
            raise KeyError(f"GPU kenergy nstep {nstep} not present in CPU flowstate")
        cpu_row = cpu_by_step[nstep]
        cpu_value = float(cpu_row[kenergy_col])
        gpu_value = float(row[2])
        diff = abs(gpu_value - cpu_value)
        rel = diff / max(abs(cpu_value), 1.0)
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, rel)
        if diff > args.atol and rel > args.rtol:
            status = 1
        lines.append(
            f"{nstep:d} {float(cpu_row[time_col]):.16e} {cpu_value:.16e} "
            f"{gpu_value:.16e} {diff:.16e} {rel:.16e}"
        )

    lines.insert(4, f"max_abs: {max_abs:.16e}")
    lines.insert(5, f"max_rel: {max_rel:.16e}")
    if status != 0:
        lines[0] = "status: fail"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
