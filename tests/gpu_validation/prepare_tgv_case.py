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
    marker = "nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            parts = [part.strip() for part in lines[data_idx].split(",")]
            if len(parts) not in (7, 8, 9):
                raise ValueError(f"unexpected runtime flag line: {lines[data_idx]}")
            if len(parts) == 7:
                parts.append("f")
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


def set_flowtype(input_file: Path, flowtype: str) -> None:
    lines = input_file.read_text().splitlines()
    marker = "flowtype"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            lines[data_idx] = flowtype
            input_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"flowtype marker not found in {input_file}")


def set_homogeneous(input_file: Path, homogeneous: str) -> None:
    parts = [part.strip() for part in homogeneous.split(",")]
    if len(parts) != 3 or any(part not in ("t", "f") for part in parts):
        raise ValueError("--homogeneous must have the form t,t,t")
    lines = input_file.read_text().splitlines()
    marker = "lihomo,ljhomo,lkhomo"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            lines[data_idx] = ",".join(parts)
            input_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"homogeneous marker not found in {input_file}")


def set_bctype(input_file: Path, bctype: str) -> None:
    delimiter = ";" if ";" in bctype else ","
    parts = [part.strip() for part in bctype.split(delimiter)]
    if len(parts) != 6:
        raise ValueError("--bctype must have six boundary entries")
    lines = input_file.read_text().splitlines()
    marker = "bctype"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idxs: list[int] = []
            scan = idx
            while len(data_idxs) < 6:
                scan += 1
                if scan >= len(lines):
                    raise ValueError(f"not enough bctype lines in {input_file}")
                stripped = lines[scan].strip()
                if stripped and not stripped.startswith("#"):
                    data_idxs.append(scan)
            for data_idx, value in zip(data_idxs, parts, strict=True):
                lines[data_idx] = value
            input_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"bctype marker not found in {input_file}")


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


def set_controller_deltat(controller_file: Path, deltat: str) -> None:
    lines = controller_file.read_text().splitlines()
    marker = "deltat"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            lines[data_idx] = deltat
            controller_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"deltat marker not found in {controller_file}")


def set_grid(input_file: Path, grid: str) -> None:
    parts = [part.strip() for part in grid.split(",")]
    if len(parts) != 3:
        raise ValueError("--grid must have the form im,jm,km")
    dims = []
    for part in parts:
        value = int(part)
        if value < 1:
            raise ValueError("--grid dimensions must be positive")
        dims.append(str(value))

    lines = input_file.read_text().splitlines()
    marker = "im,jm,km"
    for idx, line in enumerate(lines):
        if marker in line:
            data_idx = next_data_line(lines, idx)
            lines[data_idx] = ",".join(dims)
            input_file.write_text("\n".join(lines) + "\n")
            return
    raise ValueError(f"grid marker not found in {input_file}")


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
    parser.add_argument("--input-name", default="input.tgv")
    parser.add_argument("--flowtype", help="optional replacement flowtype")
    parser.add_argument("--homogeneous", help="optional homogeneous flags as t,t,t")
    parser.add_argument(
        "--bctype",
        help="optional six boundary entries; use semicolons when entries contain commas",
    )
    parser.add_argument("--use-gpu", required=True, choices=("t", "f"))
    parser.add_argument("--maxstep", required=True, type=int)
    parser.add_argument("--feqchkpt", type=int)
    parser.add_argument("--lfilter", choices=("t", "f"))
    parser.add_argument("--diffterm", choices=("t", "f"))
    parser.add_argument("--scheme", default="643e")
    parser.add_argument("--grid", help="optional grid dimensions as im,jm,km")
    parser.add_argument("--deltat", help="optional controller deltat value, e.g. 5.d-4")
    args = parser.parse_args()

    if args.maxstep < 1:
        raise ValueError("--maxstep must be positive")
    if args.feqchkpt is not None and args.feqchkpt < 1:
        raise ValueError("--feqchkpt must be positive")

    if args.dst_case.exists():
        shutil.rmtree(args.dst_case)
    args.dst_case.mkdir(parents=True)
    shutil.copytree(args.src_case / "datin", args.dst_case / "datin")

    input_file = args.dst_case / "datin" / args.input_name
    if args.flowtype:
        set_flowtype(input_file, args.flowtype)
    if args.homogeneous:
        set_homogeneous(input_file, args.homogeneous)
    if args.bctype:
        set_bctype(input_file, args.bctype)
    set_runtime_flags(input_file, args.use_gpu, args.lfilter, args.diffterm)
    set_scheme(input_file, args.scheme)
    if args.grid:
        set_grid(input_file, args.grid)
    feqchkpt = args.maxstep if args.feqchkpt is None else args.feqchkpt
    set_controller_steps(args.dst_case / "datin" / "controller", args.maxstep, feqchkpt)
    if args.deltat:
        set_controller_deltat(args.dst_case / "datin" / "controller", args.deltat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
