#!/usr/bin/env python3
"""Load the pinned NASA WIND Mach-7 hypersonic-ramp comparison data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


SOURCE_URL = (
    "https://www.grc.nasa.gov/www/wind/valid/hypramp/hypramp01/"
    "hypramp01.html"
)
PINNED_SHA256 = {
    "run.E.dat": "6012b6d772afd49e92efe8cb6ab402eb0691c435027646140e131ce25dc7e7ba",
    "T.E.d": "41e010984aad293557a57206666ff709333e185220f7058e2f21521a154fdc4f",
    "p.E.d": "b08ef00ff3e378ebcbbb6e7e906bef636381104cd731ec9f219216a3022d7cad",
    "u.E.d": "f2176b55448a172e9f6d6bfcf6b60d6abe8696e947993c3a5c4859abbb3bdfc4",
}


@dataclass(frozen=True)
class NasaHyprampRun:
    mach: float
    pressure_psia: float
    temperature_rankine: float
    cfl: float
    surface_x_ft: np.ndarray
    surface_temperature_rankine: np.ndarray
    surface_pressure_psf: np.ndarray
    exit_velocity_fps: np.ndarray
    exit_y_ft: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_columns(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid two-column NASA reference file: {path}")
    return values


def load_nasa_hypramp_run(directory: Path) -> NasaHyprampRun:
    """Load NASA WIND Run E without treating it as experimental truth."""
    directory = Path(directory)
    for name, expected in PINNED_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"NASA reference checksum mismatch for {name}")

    input_text = (directory / "run.E.dat").read_text(encoding="ascii")
    match = re.search(
        r"freestream\s+static\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+"
        r"([0-9.eE+-]+)",
        input_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("NASA Run E freestream declaration is missing")
    cfl_match = re.search(r"^\s*cfl\s+([0-9.eE+-]+)", input_text, re.MULTILINE)
    if cfl_match is None:
        raise ValueError("NASA Run E CFL declaration is missing")
    required_tokens = (
        "finite rate",
        "file air-5sp-gen-30k.chm",
        "rhs roe second",
        "turbulence laminar",
    )
    normalized = input_text.lower()
    if any(token not in normalized for token in required_tokens):
        raise ValueError("NASA Run E is not the pinned laminar finite-rate case")

    temperature = _load_columns(directory / "T.E.d")
    pressure = _load_columns(directory / "p.E.d")
    velocity = _load_columns(directory / "u.E.d")
    if temperature.shape != pressure.shape or not np.array_equal(
        temperature[:, 0], pressure[:, 0]
    ):
        raise ValueError("NASA surface temperature and pressure stations differ")
    if np.any(np.diff(temperature[:, 0]) <= 0.0):
        raise ValueError("NASA surface stations must be strictly increasing")
    if np.any(np.diff(velocity[:, 1]) <= 0.0):
        raise ValueError("NASA exit-profile y coordinates must be increasing")
    if np.any(temperature[:, 1] <= 0.0) or np.any(pressure[:, 1] <= 0.0):
        raise ValueError("NASA surface reference contains a non-positive state")

    return NasaHyprampRun(
        mach=float(match.group(1)),
        pressure_psia=float(match.group(2)),
        temperature_rankine=float(match.group(3)),
        cfl=float(cfl_match.group(1)),
        surface_x_ft=temperature[:, 0],
        surface_temperature_rankine=temperature[:, 1],
        surface_pressure_psf=pressure[:, 1],
        exit_velocity_fps=velocity[:, 0],
        exit_y_ft=velocity[:, 1],
    )
