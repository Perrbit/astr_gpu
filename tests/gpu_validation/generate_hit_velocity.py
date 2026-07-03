#!/usr/bin/env python3
"""Generate a deterministic periodic HIT-style velocity.h5 for validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid", required=True, help="global grid as im,jm,km")
    parser.add_argument("--amplitude", type=float, default=0.05)
    args = parser.parse_args()

    dims = [int(part.strip()) for part in args.grid.split(",")]
    if len(dims) != 3 or any(dim < 1 for dim in dims):
        raise ValueError("--grid must be positive im,jm,km")
    im, jm, km = dims

    x = np.linspace(0.0, 2.0 * np.pi, im + 1, endpoint=True)
    y = np.linspace(0.0, 2.0 * np.pi, jm + 1, endpoint=True)
    z = np.linspace(0.0, 2.0 * np.pi, km + 1, endpoint=True)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    amp = args.amplitude
    u1 = amp * (np.sin(zz) + np.cos(yy))
    u2 = amp * (np.sin(xx) + np.cos(zz))
    u3 = amp * (np.sin(yy) + np.cos(xx))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as h5:
        h5.create_dataset("u1", data=np.asfortranarray(u1), dtype="f8")
        h5.create_dataset("u2", data=np.asfortranarray(u2), dtype="f8")
        h5.create_dataset("u3", data=np.asfortranarray(u3), dtype="f8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
