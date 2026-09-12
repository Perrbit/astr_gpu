#!/usr/bin/env python3
"""Generate deterministic CPU-compatible HDF5 planes for inlet tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


FIELD_AMPLITUDES = {
    "ro": 1.0e-3,
    "u1": 1.0e-2,
    "u2": 2.0e-3,
    "u3": 3.0e-3,
    "t": 5.0e-4,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def temporal_amplitude(index: int, time: float, delta_time: float, mode: str) -> float:
    if mode == "cubic":
        return (1.0 + time / delta_time) ** 3
    if mode == "nonpolynomial":
        return 1.0 + 0.35 * np.sin(0.73 * index) + 0.10 * np.cos(0.31 * index**2)
    raise ValueError(f"unsupported temporal mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jm", required=True, type=positive_int)
    parser.add_argument("--km", required=True, type=positive_int)
    parser.add_argument("--count", type=positive_int, default=8)
    parser.add_argument("--delta-time", required=True, type=float)
    parser.add_argument(
        "--temporal-mode", choices=("cubic", "nonpolynomial"), default="cubic"
    )
    args = parser.parse_args()
    if args.count < 4:
        raise ValueError("--count must be at least four")
    if not np.isfinite(args.delta_time) or args.delta_time <= 0.0:
        raise ValueError("--delta-time must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    eta = np.linspace(0.0, 1.0, args.jm + 1)[:, None]
    zeta = np.linspace(0.0, 1.0, args.km + 1)[None, :]
    spatial_mode = np.sin(np.pi * eta) * np.cos(2.0 * np.pi * zeta)
    for index in range(args.count):
        time = index * args.delta_time
        temporal_mode = temporal_amplitude(index, time, args.delta_time, args.temporal_mode)
        path = args.output / f"islice{index:05d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("time", data=np.float64(time))
            for name, amplitude in FIELD_AMPLITUDES.items():
                field = amplitude * temporal_mode * spatial_mode
                handle.create_dataset(name, data=np.asfortranarray(field).T, dtype="f8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
