#!/usr/bin/env python3
"""Compare ASTR flowstate.dat statistical outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def flowstate_path(path: Path) -> Path:
    if path.is_dir():
        return path / "flowstate.dat"
    return path


def read_flowstate(path: Path) -> tuple[list[str], np.ndarray]:
    fpath = flowstate_path(path)
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    with fpath.open() as fh:
        header = fh.readline().split()
    data = np.loadtxt(fpath, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise ValueError(f"{fpath}: header has {len(header)} columns, data has {data.shape[1]}")
    return header, data


def format_float(value: float) -> str:
    return f"{value:.16e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    args = parser.parse_args()

    cpu_header, cpu = read_flowstate(args.cpu)
    gpu_header, gpu = read_flowstate(args.gpu)

    lines: list[str] = []
    status = 0

    if cpu_header != gpu_header:
        lines.append("status: fail")
        lines.append(f"reason: header mismatch cpu={cpu_header} gpu={gpu_header}")
        status = 1
    elif cpu.shape != gpu.shape:
        lines.append("status: fail")
        lines.append(f"reason: shape mismatch cpu={cpu.shape} gpu={gpu.shape}")
        status = 1
    else:
        diff = gpu - cpu
        scale = np.maximum(np.abs(cpu), 1.0)
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / scale
        pass_mask = (abs_diff <= args.atol) | (rel_diff <= args.rtol)
        status = 0 if bool(np.all(pass_mask)) else 1

        lines.append(f"status: {'pass' if status == 0 else 'fail'}")
        lines.append(f"columns: {' '.join(cpu_header)}")
        lines.append(f"atol: {format_float(args.atol)}")
        lines.append(f"rtol: {format_float(args.rtol)}")
        lines.append("")
        lines.append("metric max_abs max_rel final_cpu final_gpu final_abs final_rel")
        for col, name in enumerate(cpu_header):
            max_abs = float(abs_diff[:, col].max())
            max_rel = float(rel_diff[:, col].max())
            final_abs = float(abs_diff[-1, col])
            final_rel = float(rel_diff[-1, col])
            lines.append(
                " ".join(
                    [
                        name,
                        format_float(max_abs),
                        format_float(max_rel),
                        format_float(float(cpu[-1, col])),
                        format_float(float(gpu[-1, col])),
                        format_float(final_abs),
                        format_float(final_rel),
                    ]
                )
            )

        if status != 0:
            bad = np.argwhere(~pass_mask)[0]
            lines.append("")
            lines.append(
                "first_failure: "
                f"row={int(bad[0])} metric={cpu_header[int(bad[1])]} "
                f"cpu={format_float(float(cpu[bad[0], bad[1]]))} "
                f"gpu={format_float(float(gpu[bad[0], bad[1]]))} "
                f"abs={format_float(float(abs_diff[bad[0], bad[1]]))} "
                f"rel={format_float(float(rel_diff[bad[0], bad[1]]))}"
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
