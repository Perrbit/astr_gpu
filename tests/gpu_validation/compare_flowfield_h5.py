#!/usr/bin/env python3
"""Compare ASTR HDF5 flowfield outputs and reconstructed q fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


PRIMITIVE_NAMES = ("ro", "u1", "u2", "u3", "p", "t")
Q_NAMES = ("q1", "q2", "q3", "q4", "q5")
DEFAULT_CONST6 = 1.0 / (1.4 - 1.0)


def flowfield_path(path: Path) -> Path:
    if path.is_dir():
        return path / "outdat" / "flowfield.h5"
    return path


def read_flowfield(path: Path) -> dict[str, np.ndarray]:
    fpath = flowfield_path(path)
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    with h5py.File(fpath, "r") as h5:
        missing = [name for name in PRIMITIVE_NAMES if name not in h5]
        if missing:
            raise KeyError(f"{fpath}: missing datasets {missing}")
        return {name: h5[name][...] for name in PRIMITIVE_NAMES}


def reconstruct_q(fields: dict[str, np.ndarray], const6: float) -> dict[str, np.ndarray]:
    rho = fields["ro"]
    u1 = fields["u1"]
    u2 = fields["u2"]
    u3 = fields["u3"]
    prs = fields["p"]
    return {
        "q1": rho,
        "q2": rho * u1,
        "q3": rho * u2,
        "q4": rho * u3,
        "q5": prs * const6 + 0.5 * rho * (u1 * u1 + u2 * u2 + u3 * u3),
    }


def norm_line(name: str, cpu: np.ndarray, gpu: np.ndarray) -> tuple[str, bool]:
    if cpu.shape != gpu.shape:
        return f"{name} shape_mismatch cpu={cpu.shape} gpu={gpu.shape}", False
    diff = gpu - cpu
    abs_diff = np.abs(diff)
    linf = float(np.nanmax(abs_diff))
    l2 = float(np.sqrt(np.nanmean(diff * diff)))
    ref_l2 = float(np.sqrt(np.nanmean(cpu * cpu)))
    idx = np.unravel_index(np.nanargmax(abs_diff), diff.shape)
    line = (
        f"{name} linf={linf:.16e} l2={l2:.16e} ref_l2={ref_l2:.16e} "
        f"idx={','.join(str(int(v)) for v in idx)} "
        f"cpu={float(cpu[idx]):.16e} gpu={float(gpu[idx]):.16e}"
    )
    return line, True


def extrema_line(name: str, cpu: np.ndarray, gpu: np.ndarray) -> str:
    return (
        f"{name} cpu_min={float(np.nanmin(cpu)):.16e} "
        f"gpu_min={float(np.nanmin(gpu)):.16e} "
        f"cpu_max={float(np.nanmax(cpu)):.16e} "
        f"gpu_max={float(np.nanmax(gpu)):.16e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--const6", type=float, default=DEFAULT_CONST6)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    args = parser.parse_args()

    cpu_prim = read_flowfield(args.cpu)
    gpu_prim = read_flowfield(args.gpu)
    cpu_q = reconstruct_q(cpu_prim, args.const6)
    gpu_q = reconstruct_q(gpu_prim, args.const6)

    lines: list[str] = [
        "status: pass",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        f"const6: {args.const6:.16e}",
        "",
        "dataset linf l2 ref_l2 idx cpu gpu",
    ]
    status = 0

    for group_name, cpu_fields, gpu_fields, names in (
        ("primitive", cpu_prim, gpu_prim, PRIMITIVE_NAMES),
        ("q", cpu_q, gpu_q, Q_NAMES),
    ):
        lines.append("")
        lines.append(f"[{group_name}]")
        for name in names:
            line, ok = norm_line(name, cpu_fields[name], gpu_fields[name])
            lines.append(line)
            if not ok:
                status = 1
                continue
            diff = gpu_fields[name] - cpu_fields[name]
            abs_diff = np.abs(diff)
            linf = float(np.nanmax(abs_diff))
            ref = float(np.nanmax(np.abs(cpu_fields[name])))
            allowed = max(args.atol, args.rtol * ref)
            if linf > allowed:
                status = 1

    lines.append("")
    lines.append("[primitive_extrema]")
    for name in ("ro", "p", "t"):
        lines.append(extrema_line(name, cpu_prim[name], gpu_prim[name]))

    if status != 0:
        lines[0] = "status: fail"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
