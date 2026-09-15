#!/usr/bin/env python3
"""Convert the pinned HTR Mach-6 multispecies similarity profile for ASTR."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence


HTR_COMMIT = "328f84270ce26b83e4c7a4e75c0681a129a48bda"
HTR_PROFILE_SHA256 = "5bca0f7dbb844d81444e1033059c9a9a18c516dc4b9f42b067c96965c33a468a"
AIR5_RU = 8.31446261815324
AIR5_MOLAR_MASS = (0.028, 0.032, 0.014, 0.016, 0.030)


@dataclass(frozen=True)
class HtrSimilarityRow:
    eta: float
    stream_function: float
    u: float
    temperature: float
    # Source order: N2, O2, NO, N, O.
    mole_fraction: tuple[float, float, float, float, float]
    viscosity: float
    density: float
    speed_of_sound: float
    re_x: float


@dataclass(frozen=True)
class Air5ProfileRow:
    y: float
    density: float
    u: float
    v: float
    w: float
    pressure: float
    temperature: float
    tv: float
    # ASTR order: N2, O2, N, O, NO.
    mass_fraction: tuple[float, float, float, float, float]

    def as_columns(self) -> tuple[float, ...]:
        return (
            self.y,
            self.density,
            self.u,
            self.v,
            self.w,
            self.pressure,
            self.temperature,
            self.tv,
            *self.mass_fraction,
        )


@dataclass(frozen=True)
class Air5ProfileMetadata:
    x_origin: float
    displacement_thickness: float
    re_delta_star: float
    max_source_density_relative_difference: float


def load_htr_similarity_profile(path: Path) -> list[HtrSimilarityRow]:
    with path.open(encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {
            "eta",
            "f",
            "u",
            "T",
            "X_N2",
            "X_O2",
            "X_NO",
            "X_N",
            "X_O",
            "mu",
            "rho",
            "SoS",
            "Rex",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError("unexpected HTR similarity-profile columns")
        rows = [
            HtrSimilarityRow(
                eta=float(row["eta"]),
                stream_function=float(row["f"]),
                u=float(row["u"]),
                temperature=float(row["T"]),
                mole_fraction=(
                    float(row["X_N2"]),
                    float(row["X_O2"]),
                    float(row["X_NO"]),
                    float(row["X_N"]),
                    float(row["X_O"]),
                ),
                viscosity=float(row["mu"]),
                density=float(row["rho"]),
                speed_of_sound=float(row["SoS"]),
                re_x=float(row["Rex"]),
            )
            for row in reader
        ]
    if len(rows) < 2:
        raise ValueError("HTR similarity profile requires at least two rows")
    if any(b.eta <= a.eta for a, b in zip(rows, rows[1:])):
        raise ValueError("HTR similarity coordinate must be strictly increasing")
    if any(row.temperature <= 0.0 or row.density <= 0.0 for row in rows):
        raise ValueError("HTR similarity profile contains a non-positive state")
    if any(min(row.mole_fraction) < 0.0 for row in rows):
        raise ValueError("HTR similarity profile contains a negative mole fraction")
    return rows


def _mole_to_astr_mass_fraction(
    source_mole_fraction: Sequence[float],
) -> tuple[float, float, float, float, float]:
    if len(source_mole_fraction) != 5:
        raise ValueError("five HTR mole fractions are required")
    # HTR N2/O2/NO/N/O -> ASTR N2/O2/N/O/NO.
    reordered = (
        source_mole_fraction[0],
        source_mole_fraction[1],
        source_mole_fraction[3],
        source_mole_fraction[4],
        source_mole_fraction[2],
    )
    weighted = tuple(x * molar_mass for x, molar_mass in zip(reordered, AIR5_MOLAR_MASS))
    total = sum(weighted)
    if total <= 0.0:
        raise ValueError("HTR composition has zero mixture molar mass")
    return tuple(value / total for value in weighted)  # type: ignore[return-value]


def map_htr_profile_to_air5(
    source: Sequence[HtrSimilarityRow], pressure: float
) -> tuple[list[Air5ProfileRow], Air5ProfileMetadata]:
    if len(source) < 2:
        raise ValueError("HTR similarity profile requires at least two rows")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("profile pressure must be finite and positive")

    edge = source[-1]
    re_x = source[0].re_x
    if re_x <= 0.0 or any(abs(row.re_x - re_x) > 1.0e-10 * re_x for row in source):
        raise ValueError("HTR profile must describe one positive Reynolds-number station")
    x_origin = re_x * edge.viscosity / (edge.u * edge.density)
    similarity_scale = x_origin * math.sqrt(2.0 / re_x)

    displacement_integral = 0.0
    y = [0.0] * len(source)
    for index in range(len(source) - 1):
        left = source[index]
        right = source[index + 1]
        left_integrand = (
            1.0 - left.u * left.density / (edge.u * edge.density)
        ) * edge.density / left.density
        right_integrand = (
            1.0 - right.u * right.density / (edge.u * edge.density)
        ) * edge.density / right.density
        deta = right.eta - left.eta
        displacement_integral += 0.5 * (left_integrand + right_integrand) * deta
        density_midpoint = 0.5 * (left.density + right.density)
        y[index + 1] = (
            y[index] + similarity_scale * edge.density / density_midpoint * deta
        )
    displacement_thickness = displacement_integral * similarity_scale
    re_delta_star = edge.u * edge.density * displacement_thickness / edge.viscosity

    mapped: list[Air5ProfileRow] = []
    density_differences: list[float] = []
    for y_value, row in zip(y, source):
        mass_fraction = _mole_to_astr_mass_fraction(row.mole_fraction)
        mixture_gas_constant = sum(
            value * AIR5_RU / molar_mass
            for value, molar_mass in zip(mass_fraction, AIR5_MOLAR_MASS)
        )
        density = pressure / (mixture_gas_constant * row.temperature)
        v = (
            0.5 * y_value / x_origin * row.u
            - edge.density
            / row.density
            / math.sqrt(2.0 * re_x)
            * row.stream_function
        )
        mapped.append(
            Air5ProfileRow(
                y=y_value,
                density=density,
                u=row.u,
                v=v,
                w=0.0,
                pressure=pressure,
                temperature=row.temperature,
                tv=row.temperature,
                mass_fraction=mass_fraction,
            )
        )
        density_differences.append(abs(density - row.density) / row.density)

    metadata = Air5ProfileMetadata(
        x_origin=x_origin,
        displacement_thickness=displacement_thickness,
        re_delta_star=re_delta_star,
        max_source_density_relative_difference=max(density_differences),
    )
    return mapped, metadata


def export_astr_air5_profile(
    path: Path,
    profile: Sequence[Air5ProfileRow],
    metadata: Air5ProfileMetadata,
) -> None:
    lines = [
        f"# Source: Stanford HTR-solver commit {HTR_COMMIT}",
        f"# source_sha256={HTR_PROFILE_SHA256} x_origin={metadata.x_origin:.17e} "
        f"delta_star={metadata.displacement_thickness:.17e} "
        f"Re_delta_star={metadata.re_delta_star:.17e}",
        "# columns: y rho u v w p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO",
    ]
    lines.extend(
        " ".join(f"{value:.17e}" for value in row.as_columns()) for row in profile
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pressure", type=float, default=101325.0)
    args = parser.parse_args()

    source = load_htr_similarity_profile(args.input)
    mapped, metadata = map_htr_profile_to_air5(source, args.pressure)
    export_astr_air5_profile(args.output, mapped, metadata)
    print(
        f"points={len(mapped)} x_origin={metadata.x_origin:.17e} "
        f"delta_star={metadata.displacement_thickness:.17e} "
        f"Re_delta_star={metadata.re_delta_star:.9f} "
        f"max_density_rel_diff={metadata.max_source_density_relative_difference:.9e}"
    )


if __name__ == "__main__":
    main()
