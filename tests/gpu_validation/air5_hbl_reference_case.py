#!/usr/bin/env python3
"""Matched independent-reference utilities for the fixed-air5 HBL gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tests.gpu_validation.air5_hbl_diagnostics import Air5HblDiagnostics
    from tests.gpu_validation.air5_hbl_marcher import (
        Air5HblBoundaryConditions,
        Air5HblMarcher,
    )
    from tests.gpu_validation.air5_hbl_reference import (
        Air5BoundaryLayerReference,
        Air5BoundaryLayerState,
    )
    from tests.gpu_validation.air5_htr_profile import (
        Air5ProfileMetadata,
        load_htr_similarity_profile,
        map_htr_profile_to_air5,
    )
except ModuleNotFoundError:
    from air5_hbl_diagnostics import Air5HblDiagnostics
    from air5_hbl_marcher import Air5HblBoundaryConditions, Air5HblMarcher
    from air5_hbl_reference import Air5BoundaryLayerReference, Air5BoundaryLayerState
    from air5_htr_profile import (
        Air5ProfileMetadata,
        load_htr_similarity_profile,
        map_htr_profile_to_air5,
    )


@dataclass(frozen=True)
class Air5HblReferenceComparison:
    passed: bool
    profile_u_l2_relative: float
    profile_temperature_l2_relative: float
    profile_tv_l2_relative: float
    major_species_l2_relative: float
    trace_species_max_absolute: float
    skin_friction_l2_relative: float
    total_heat_flux_l2_relative: float


@dataclass(frozen=True)
class Air5HblReferenceEvidence:
    diagnostics: Air5HblDiagnostics
    x_origin: float
    scaled_residual_max: float


@dataclass(frozen=True)
class Air5HblInitialFieldEvidence:
    x: np.ndarray
    y: np.ndarray
    conservative: np.ndarray
    x_origin: float
    inlet_scaled_error: float
    end_thickness_scale: float


def load_diagnostics(path: Path) -> Air5HblDiagnostics:
    """Load one complete diagnostic archive without accepting missing fields."""
    path = Path(path)
    with np.load(path) as archive:
        names = tuple(Air5HblDiagnostics.__dataclass_fields__)
        missing = tuple(name for name in names if name not in archive)
        if missing:
            raise ValueError(f"{path}: missing HBL diagnostics {missing}")
        values = {name: np.asarray(archive[name], dtype=np.float64) for name in names}
    return Air5HblDiagnostics(**values)


def htr_contract_on_grid(
    profile_path: Path, y: np.ndarray, *, pressure: float
) -> tuple[np.ndarray, Air5HblBoundaryConditions, Air5ProfileMetadata]:
    """Interpolate the frozen HTR inlet onto an explicitly matched wall-normal grid."""
    y = np.asarray(y, dtype=np.float64)
    if (
        y.ndim != 1
        or y.size < 3
        or not np.all(np.isfinite(y))
        or np.any(np.diff(y) <= 0.0)
    ):
        raise ValueError("HBL y coordinate must be finite and strictly increasing")
    mapped, metadata = map_htr_profile_to_air5(
        load_htr_similarity_profile(Path(profile_path)), pressure=pressure
    )
    source_y = np.asarray([row.y for row in mapped], dtype=np.float64)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(source_y[-1], 1.0)
    if abs(y[0] - source_y[0]) > tolerance or y[-1] < source_y[-1] - tolerance:
        raise ValueError("matched HBL grid must cover the HTR wall and edge coordinates")
    source_primitive = np.asarray(
        [
            [row.u, row.v, row.temperature, row.tv, *row.mass_fraction[1:5]]
            for row in mapped
        ],
        dtype=np.float64,
    )
    primitive = np.column_stack(
        [np.interp(y, source_y, source_primitive[:, column]) for column in range(8)]
    )
    edge = mapped[-1]
    wall = mapped[0]
    boundary = Air5HblBoundaryConditions(
        pressure=pressure,
        edge_velocity=np.array([edge.u, edge.v, edge.w], dtype=np.float64),
        edge_temperature=edge.temperature,
        edge_tv=edge.tv,
        edge_mass_fraction=np.asarray(edge.mass_fraction, dtype=np.float64),
        wall_velocity=wall.u,
        wall_temperature=wall.temperature,
        wall_tv=wall.tv,
    )
    return primitive, boundary, metadata


def _subdivided_coordinates(
    requested: np.ndarray, maximum_step: float
) -> np.ndarray:
    requested = np.unique(np.asarray(requested, dtype=np.float64))
    if (
        requested.ndim != 1
        or requested.size < 2
        or requested[0] != 0.0
        or np.any(np.diff(requested) <= 0.0)
        or not np.isfinite(maximum_step)
        or maximum_step <= 0.0
    ):
        raise ValueError("HBL streamwise coordinates or maximum step are invalid")
    pieces = [requested[:1]]
    for left, right in zip(requested[:-1], requested[1:]):
        intervals = max(1, int(np.ceil((right - left) / maximum_step)))
        pieces.append(np.linspace(left, right, intervals + 1)[1:])
    return np.concatenate(pieces)


def _interpolate_history(
    x: np.ndarray, values: np.ndarray, target: np.ndarray
) -> np.ndarray:
    flat = values.reshape((values.shape[0], -1))
    interpolated = np.column_stack(
        [np.interp(target, x, flat[:, column]) for column in range(flat.shape[1])]
    )
    return interpolated.reshape((target.size, *values.shape[1:]))


def _primitive_to_conservative(
    reference: Air5BoundaryLayerReference, primitive: np.ndarray, pressure: float
) -> np.ndarray:
    primitive = np.asarray(primitive, dtype=np.float64)
    if primitive.ndim != 3 or primitive.shape[2] != 8:
        raise ValueError("HBL primitive field must have shape (nx, ny, 8)")
    conservative = np.empty((*primitive.shape[:2], 11), dtype=np.float64)
    for streamwise in range(primitive.shape[0]):
        for wall_normal in range(primitive.shape[1]):
            point = primitive[streamwise, wall_normal]
            mass_fraction = np.r_[1.0 - np.sum(point[4:8]), point[4:8]]
            state = Air5BoundaryLayerState(
                pressure=pressure,
                velocity=np.array([point[0], point[1], 0.0], dtype=np.float64),
                temperature=point[2],
                tv=point[3],
                mass_fraction=mass_fraction,
            )
            conservative[streamwise, wall_normal] = (
                reference.conservative_state(state)
            )
    return conservative


def generate_similarity_initial_field(
    profile_path: Path,
    mechanism_path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    pressure: float,
) -> Air5HblInitialFieldEvidence:
    """Generate a boundary-compatible sqrt(x) expansion of the HTR profile."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 1
        or x.size < 2
        or x[0] != 0.0
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError("HBL x coordinate must start at zero and increase")
    mapped, metadata = map_htr_profile_to_air5(
        load_htr_similarity_profile(Path(profile_path)), pressure=pressure
    )
    source_y = np.asarray([row.y for row in mapped], dtype=np.float64)
    source_primitive = np.asarray(
        [
            [row.u, row.v, row.temperature, row.tv, *row.mass_fraction[1:5]]
            for row in mapped
        ],
        dtype=np.float64,
    )
    if y[0] != source_y[0] or y[-1] < source_y[-1]:
        raise ValueError("HBL target grid must cover the complete HTR profile")
    if metadata.x_origin <= 0.0:
        raise ValueError("HTR profile has a non-positive physical x origin")

    primitive = np.empty((x.size, y.size, 8), dtype=np.float64)
    normalized_y = y / y[-1]
    base_y = np.minimum(y, source_y[-1])
    for streamwise in range(x.size):
        thickness_scale = np.sqrt(
            (metadata.x_origin + x[streamwise]) / metadata.x_origin
        )
        denominator = thickness_scale - (thickness_scale - 1.0) * normalized_y
        mapped_y = base_y / denominator
        for column in range(8):
            primitive[streamwise, :, column] = np.interp(
                mapped_y, source_y, source_primitive[:, column]
            )
        primitive[streamwise, :, 1] /= thickness_scale

    inlet = primitive[0].copy()
    reference = Air5BoundaryLayerReference(Path(mechanism_path))
    conservative = _primitive_to_conservative(reference, primitive, pressure)
    inlet_conservative = _primitive_to_conservative(
        reference, inlet[np.newaxis, :, :], pressure
    )[0]
    inlet_scaled_error = float(
        np.max(
            np.abs(conservative[0] - inlet_conservative)
            / np.maximum(np.abs(inlet_conservative), 1.0)
        )
    )
    return Air5HblInitialFieldEvidence(
        x=x.copy(),
        y=y.copy(),
        conservative=conservative,
        x_origin=metadata.x_origin,
        inlet_scaled_error=inlet_scaled_error,
        end_thickness_scale=float(
            np.sqrt((metadata.x_origin + x[-1]) / metadata.x_origin)
        ),
    )


def write_astr_air5_initial_field(
    path: Path, evidence: Air5HblInitialFieldEvidence
) -> None:
    """Write the versioned conservative x-y field consumed by ASTR."""
    x = np.asarray(evidence.x, dtype=np.float64)
    y = np.asarray(evidence.y, dtype=np.float64)
    conservative = np.asarray(evidence.conservative, dtype=np.float64)
    if conservative.shape != (x.size, y.size, 11):
        raise ValueError("HBL conservative field shape does not match x-y coordinates")
    if not np.all(np.isfinite(conservative)):
        raise ValueError("HBL conservative field contains non-finite values")
    lines = [
        "# ASTR_AIR5_HBL_INITIAL_FIELD_V1",
        f"# x_origin={evidence.x_origin:.17e} "
        "mapping=htr_sqrt_x_boundary_compatible "
        f"end_thickness_scale={evidence.end_thickness_scale:.17e}",
        f"{x.size} {y.size}",
    ]
    for streamwise, x_value in enumerate(x):
        for wall_normal, y_value in enumerate(y):
            row = np.r_[x_value, y_value, conservative[streamwise, wall_normal]]
            lines.append(" ".join(f"{value:.17e}" for value in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def generate_reference_diagnostics(
    profile_path: Path,
    mechanism_path: Path,
    *,
    y: np.ndarray,
    wall_x: np.ndarray,
    profile_x: np.ndarray,
    pressure: float,
    maximum_streamwise_step: float,
    reference_points: int = 33,
    max_function_evaluations: int = 1200,
) -> Air5HblReferenceEvidence:
    """March the independent model at ``x_origin + x`` and match ASTR stations."""
    wall_x = np.asarray(wall_x, dtype=np.float64)
    profile_x = np.asarray(profile_x, dtype=np.float64)
    if (
        wall_x.ndim != 1
        or wall_x.size < 2
        or wall_x[0] != 0.0
        or np.any(np.diff(wall_x) <= 0.0)
        or profile_x.ndim != 1
        or profile_x.size == 0
        or np.any(profile_x < wall_x[0])
        or np.any(profile_x > wall_x[-1])
    ):
        raise ValueError("matched HBL wall and profile stations are invalid")
    diagnostic_y = np.asarray(y, dtype=np.float64)
    if reference_points < 9:
        raise ValueError("independent HBL reference requires at least nine y points")
    mapped, _ = map_htr_profile_to_air5(
        load_htr_similarity_profile(Path(profile_path)), pressure=pressure
    )
    reference_edge = float(mapped[-1].y)
    if diagnostic_y[0] != 0.0 or diagnostic_y[-1] < reference_edge:
        raise ValueError("diagnostic HBL grid must cover the complete HTR profile")
    reference_y = diagnostic_y[-1] * np.linspace(0.0, 1.0, reference_points) ** 2
    initial, boundary, metadata = htr_contract_on_grid(
        profile_path, reference_y, pressure=pressure
    )
    requested = np.unique(np.r_[wall_x, profile_x])
    local_x = _subdivided_coordinates(requested, maximum_streamwise_step)
    marcher = Air5HblMarcher(Path(mechanism_path))
    march = marcher.march_positive(
        initial,
        reference_y,
        metadata.x_origin + local_x,
        boundary,
        source_mode="coupled",
        similarity_farfield=True,
        max_function_evaluations=max_function_evaluations,
    )
    wall = marcher.wall_quantities(march, reference_y, boundary)
    primitive_on_reference_y = _interpolate_history(
        local_x, march.primitive, profile_x
    )
    primitive = np.empty((profile_x.size, diagnostic_y.size, 8), dtype=np.float64)
    for station in range(profile_x.size):
        for column in range(8):
            primitive[station, :, column] = np.interp(
                diagnostic_y,
                reference_y,
                primitive_on_reference_y[station, :, column],
            )
    mass_fraction = np.concatenate(
        (
            1.0 - np.sum(primitive[..., 4:8], axis=2, keepdims=True),
            primitive[..., 4:8],
        ),
        axis=2,
    )
    velocity = np.zeros((profile_x.size, diagnostic_y.size, 3), dtype=np.float64)
    velocity[..., 0:2] = primitive[..., 0:2]
    diagnostics = Air5HblDiagnostics(
        wall_x=wall_x.copy(),
        skin_friction=np.interp(wall_x, local_x, wall.skin_friction),
        heat_flux_total=np.interp(wall_x, local_x, wall.total_heat_flux),
        heat_flux_translational=np.interp(
            wall_x, local_x, wall.translational_heat_flux
        ),
        heat_flux_vibrational=np.interp(
            wall_x, local_x, wall.vibrational_heat_flux
        ),
        heat_flux_species=np.interp(wall_x, local_x, wall.species_enthalpy_flux),
        profile_x=profile_x.copy(),
        profile_y=diagnostic_y.copy(),
        profile_velocity=velocity,
        profile_temperature=primitive[..., 2],
        profile_tv=primitive[..., 3],
        profile_mass_fraction=mass_fraction,
    )
    return Air5HblReferenceEvidence(
        diagnostics=diagnostics,
        x_origin=metadata.x_origin,
        scaled_residual_max=march.scaled_linear_residual_max,
    )


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        return 0.0 if not np.any(difference) else np.inf
    return float(np.linalg.norm(difference.ravel()) / denominator)


def _validate_diagnostics_pair(
    reference: Air5HblDiagnostics, candidate: Air5HblDiagnostics
) -> None:
    for name in reference.__dataclass_fields__:
        reference_value = np.asarray(getattr(reference, name), dtype=np.float64)
        candidate_value = np.asarray(getattr(candidate, name), dtype=np.float64)
        if reference_value.shape != candidate_value.shape:
            raise ValueError(
                f"{name} shape mismatch: {reference_value.shape} != {candidate_value.shape}"
            )
        if not np.all(np.isfinite(reference_value)) or not np.all(
            np.isfinite(candidate_value)
        ):
            raise ValueError(f"{name} contains non-finite values")
    for name in ("wall_x", "profile_x", "profile_y"):
        if not np.allclose(
            getattr(reference, name), getattr(candidate, name), atol=2.0e-14, rtol=2.0e-14
        ):
            raise ValueError(f"{name} coordinates do not match")
    for result in (reference, candidate):
        if np.any(result.profile_mass_fraction < 0.0) or not np.allclose(
            np.sum(result.profile_mass_fraction, axis=2), 1.0, atol=2.0e-12, rtol=0.0
        ):
            raise ValueError("HBL profile mass fractions are not physical")


def compare_hbl_diagnostics(
    reference: Air5HblDiagnostics,
    candidate: Air5HblDiagnostics,
    *,
    profile_relative_tolerance: float,
    wall_relative_tolerance: float,
    trace_absolute_tolerance: float,
) -> Air5HblReferenceComparison:
    """Apply the frozen A1-R3 profile, wall, and trace-species acceptance gates."""
    if min(
        profile_relative_tolerance,
        wall_relative_tolerance,
        trace_absolute_tolerance,
    ) < 0.0:
        raise ValueError("HBL comparison tolerances must be non-negative")
    _validate_diagnostics_pair(reference, candidate)

    wall_slice = slice(1, -1) if reference.wall_x.size > 2 else slice(None)
    u_error = _relative_l2(
        candidate.profile_velocity[..., 0], reference.profile_velocity[..., 0]
    )
    temperature_error = _relative_l2(
        candidate.profile_temperature, reference.profile_temperature
    )
    tv_error = _relative_l2(candidate.profile_tv, reference.profile_tv)
    major_species_error = max(
        _relative_l2(
            candidate.profile_mass_fraction[..., species],
            reference.profile_mass_fraction[..., species],
        )
        for species in (0, 1)
    )
    trace_species_error = float(
        np.max(
            np.abs(
                candidate.profile_mass_fraction[..., 2:5]
                - reference.profile_mass_fraction[..., 2:5]
            )
        )
    )
    skin_friction_error = _relative_l2(
        candidate.skin_friction[wall_slice], reference.skin_friction[wall_slice]
    )
    heat_flux_error = _relative_l2(
        candidate.heat_flux_total[wall_slice], reference.heat_flux_total[wall_slice]
    )
    passed = (
        max(u_error, temperature_error, tv_error, major_species_error)
        <= profile_relative_tolerance
        and trace_species_error <= trace_absolute_tolerance
        and max(skin_friction_error, heat_flux_error) <= wall_relative_tolerance
    )
    return Air5HblReferenceComparison(
        passed=passed,
        profile_u_l2_relative=u_error,
        profile_temperature_l2_relative=temperature_error,
        profile_tv_l2_relative=tv_error,
        major_species_l2_relative=major_species_error,
        trace_species_max_absolute=trace_species_error,
        skin_friction_l2_relative=skin_friction_error,
        total_heat_flux_l2_relative=heat_flux_error,
    )


def format_comparison_report(
    result: Air5HblReferenceComparison,
    *,
    profile_relative_tolerance: float,
    wall_relative_tolerance: float,
    trace_absolute_tolerance: float,
) -> str:
    lines = [
        f"status: {'pass' if result.passed else 'fail'}",
        f"profile_relative_tolerance: {profile_relative_tolerance:.16e}",
        f"wall_relative_tolerance: {wall_relative_tolerance:.16e}",
        f"trace_absolute_tolerance: {trace_absolute_tolerance:.16e}",
    ]
    lines.extend(
        f"{name}: {value:.16e}"
        for name, value in result.__dict__.items()
        if name != "passed"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--profile-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--wall-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--trace-absolute-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()
    result = compare_hbl_diagnostics(
        load_diagnostics(args.reference),
        load_diagnostics(args.candidate),
        profile_relative_tolerance=args.profile_relative_tolerance,
        wall_relative_tolerance=args.wall_relative_tolerance,
        trace_absolute_tolerance=args.trace_absolute_tolerance,
    )
    report = format_comparison_report(
        result,
        profile_relative_tolerance=args.profile_relative_tolerance,
        wall_relative_tolerance=args.wall_relative_tolerance,
        trace_absolute_tolerance=args.trace_absolute_tolerance,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="ascii")
    print(report, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
