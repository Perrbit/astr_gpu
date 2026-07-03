#!/usr/bin/env python3
"""Prepare an isolated Taylor-Green Vortex validation case."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def next_data_line(lines: list[str], start: int) -> int:
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped and not stripped.startswith("#"):
            return idx
    raise ValueError(f"no data line after line {start + 1}")


def set_runtime_flags(
    input_file: Path,
    use_gpu: str,
    lfilter: str | None,
    diffterm: str | None,
) -> None:
    lines = input_file.read_text().splitlines()
    marker = "nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            parts = [part.strip() for part in lines[data_idx].split(",")]
            if len(parts) not in (8, 9):
                raise ValueError(f"unexpected runtime flag line: {lines[data_idx]}")
            while len(parts) < 9:
                parts.append("f")
            if diffterm is not None:
                parts[1] = diffterm
            if lfilter is not None:
                parts[2] = lfilter
            parts[8] = use_gpu
            lines[data_idx] = ",".join(parts)
            input_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"runtime flag marker not found in {input_file}")


def set_controller_steps(controller_file: Path, maxstep: int, feqchkpt: int) -> None:
    lines = controller_file.read_text().splitlines()
    marker = "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            parts = [part.strip() for part in lines[data_idx].split(",")]
            if len(parts) != 6:
                raise ValueError(f"unexpected controller line: {lines[data_idx]}")
            parts[0] = str(maxstep)
            parts[1] = str(feqchkpt)
            lines[data_idx] = ",".join(parts)
            controller_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"controller marker not found in {controller_file}")


def set_scheme(input_file: Path, scheme: str) -> None:
    lines = input_file.read_text().splitlines()
    marker = "conschm,difschm,rkscheme"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            parts = [part.strip() for part in lines[data_idx].split(",")]
            if len(parts) < 3:
                raise ValueError(f"unexpected scheme line: {lines[data_idx]}")
            parts[0] = scheme
            parts[1] = scheme
            lines[data_idx] = ",".join(parts)
            input_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"scheme marker not found in {input_file}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-case", required=True, type=Path)
    parser.add_argument("--dst-case", required=True, type=Path)
    parser.add_argument("--use-gpu", required=True, choices=("t", "f"))
    parser.add_argument("--maxstep", required=True, type=int)
    parser.add_argument("--feqchkpt", type=int)
    parser.add_argument("--lfilter", choices=("t", "f"))
    parser.add_argument("--diffterm", choices=("t", "f"))
    parser.add_argument("--scheme", default="643e")
    args = parser.parse_args()

    if args.maxstep < 1:
        raise ValueError("--maxstep must be positive")
    if args.feqchkpt is not None and args.feqchkpt < 1:
        raise ValueError("--feqchkpt must be positive")

    if args.dst_case.exists():
        shutil.rmtree(args.dst_case)
    args.dst_case.mkdir(parents=True)
    shutil.copytree(args.src_case / "datin", args.dst_case / "datin")

    set_runtime_flags(
        args.dst_case / "datin" / "input.tgv",
        args.use_gpu,
        args.lfilter,
        args.diffterm,
    )
    set_scheme(args.dst_case / "datin" / "input.tgv", args.scheme)
    feqchkpt = args.maxstep if args.feqchkpt is None else args.feqchkpt
    set_controller_steps(args.dst_case / "datin" / "controller", args.maxstep, feqchkpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
