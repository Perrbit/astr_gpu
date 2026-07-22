#!/usr/bin/env python3
"""Generate a uniform three-dimensional ASTR initial field."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_grid(value: str) -> tuple[int, int, int]:
    try:
        grid = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--grid must have the form im,jm,km") from exc
    if len(grid) != 3 or min(grid) < 1:
        raise argparse.ArgumentTypeError("--grid must contain three positive dimensions")
    return grid


def write_uniform_field(
    output: Path,
    grid: tuple[int, int, int],
    density: float,
    velocity: tuple[float, float, float],
    temperature: float,
) -> None:
    if density <= 0.0:
        raise ValueError("density must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    im, jm, km = grid
    shape = (km + 1, jm + 1, im + 1)
    values = {
        "ro": density,
        "u1": velocity[0],
        "u2": velocity[1],
        "u3": velocity[2],
        "t": temperature,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        for name, value in values.items():
            handle.create_dataset(name, data=np.full(shape, value, dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=parse_grid)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--u1", type=float, default=0.7)
    parser.add_argument("--u2", type=float, default=-0.2)
    parser.add_argument("--u3", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    write_uniform_field(
        args.output,
        args.grid,
        args.density,
        (args.u1, args.u2, args.u3),
        args.temperature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
