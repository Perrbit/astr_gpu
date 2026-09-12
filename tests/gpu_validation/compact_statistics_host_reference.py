#!/usr/bin/env python3
"""Independent host quadrature for compact production-statistics validation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np


VALIDATION_MAGIC = b"ASTRCRF1"
VALIDATION_VERSION = 1
ENDIAN_MARKER = 0x01020304


class ValidationHeader(NamedTuple):
    global_dims: tuple[int, int, int]
    topology: tuple[int, int, int]
    rank: int
    rank_coords: tuple[int, int, int]
    offsets: tuple[int, int, int]
    local_dims: tuple[int, int, int]
    halo_width: int
    step: int
    time: float
    has_wall: bool
    nondimensional: bool
    reynolds: float
    prandtl: float
    const5: float
    cp: float
    tempconst: float
    tempconst1: float


class ValidationSnapshot(NamedTuple):
    header: ValidationHeader
    coordinates: np.ndarray
    density: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    velocity_gradient: np.ndarray
    temperature_gradient: np.ndarray
    wall_coordinates: np.ndarray | None
    wall_normal: np.ndarray | None


class SolverComparison(NamedTuple):
    passed: bool
    rank_count: int
    sample_count: int
    max_abs: float
    details: tuple[str, ...]


def _read_array(stream, count: int, dtype: str, label: str) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    payload = stream.read(count * itemsize)
    if len(payload) != count * itemsize:
        raise ValueError(f"truncated compact validation {label}")
    return np.frombuffer(payload, dtype=dtype).copy()


def _reshape_fortran(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.asfortranarray(values.reshape(shape, order="F"))


def read_validation_snapshot(path: Path | str) -> ValidationSnapshot:
    """Read one CPU same-phase validation snapshot without production helpers."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(len(VALIDATION_MAGIC)) != VALIDATION_MAGIC:
            raise ValueError(f"{path}: compact validation magic mismatch")
        fixed = _read_array(stream, 22, "<i4", "fixed header")
        if int(fixed[0]) != VALIDATION_VERSION:
            raise ValueError(f"{path}: unsupported compact validation version")
        if int(fixed[1]) != ENDIAN_MARKER:
            raise ValueError(f"{path}: compact validation endianness mismatch")
        if int(fixed[2]) != 8:
            raise ValueError(f"{path}: compact validation real size mismatch")
        if int(fixed[3]) not in (0, 1) or int(fixed[4]) not in (0, 1):
            raise ValueError(f"{path}: invalid compact validation logical flag")
        global_dims = tuple(int(value) for value in fixed[5:8])
        topology = tuple(int(value) for value in fixed[8:11])
        rank = int(fixed[11])
        rank_coords = tuple(int(value) for value in fixed[12:15])
        offsets = tuple(int(value) for value in fixed[15:18])
        local_dims = tuple(int(value) for value in fixed[18:21])
        halo_width = int(fixed[21])
        if min(global_dims) < 1 or min(topology) < 1 or min(local_dims) < 1 or halo_width < 0:
            raise ValueError(f"{path}: invalid compact validation dimensions")
        step = int(_read_array(stream, 1, "<i8", "step")[0])
        time = float(_read_array(stream, 1, "<f8", "time")[0])
        constants = _read_array(stream, 6, "<f8", "physical constants")
        ni, nj, nk = local_dims
        point_count = ni * nj * nk
        coordinates = _reshape_fortran(
            _read_array(stream, point_count * 3, "<f8", "coordinates"),
            (ni, nj, nk, 3),
        )
        density = _reshape_fortran(
            _read_array(stream, point_count, "<f8", "density"), (ni, nj, nk)
        )
        velocity = _reshape_fortran(
            _read_array(stream, point_count * 3, "<f8", "velocity"),
            (ni, nj, nk, 3),
        )
        pressure = _reshape_fortran(
            _read_array(stream, point_count, "<f8", "pressure"), (ni, nj, nk)
        )
        temperature = _reshape_fortran(
            _read_array(stream, point_count, "<f8", "temperature"), (ni, nj, nk)
        )
        velocity_gradient = _reshape_fortran(
            _read_array(stream, point_count * 9, "<f8", "velocity gradient"),
            (ni, nj, nk, 3, 3),
        )
        temperature_gradient = _reshape_fortran(
            _read_array(stream, point_count * 3, "<f8", "temperature gradient"),
            (ni, nj, nk, 3),
        )
        wall_coordinates = None
        wall_normal = None
        if bool(fixed[3]):
            wall_coordinates = _reshape_fortran(
                _read_array(
                    stream,
                    (ni + 2 * halo_width) * nk * 3,
                    "<f8",
                    "wall coordinates",
                ),
                (ni + 2 * halo_width, nk, 3),
            )
            wall_normal = _reshape_fortran(
                _read_array(stream, ni * nk * 3, "<f8", "wall normal"),
                (ni, nk, 3),
            )
        if stream.read(1):
            raise ValueError(f"{path}: trailing compact validation payload")

    fields = (
        coordinates,
        density,
        velocity,
        pressure,
        temperature,
        velocity_gradient,
        temperature_gradient,
    )
    if wall_coordinates is not None:
        fields += (wall_coordinates, wall_normal)
    if not np.isfinite(time) or not np.all(np.isfinite(constants)):
        raise ValueError(f"{path}: non-finite compact validation metadata")
    if not all(np.all(np.isfinite(field)) for field in fields):
        raise ValueError(f"{path}: non-finite compact validation field")

    return ValidationSnapshot(
        header=ValidationHeader(
            global_dims=global_dims,
            topology=topology,
            rank=rank,
            rank_coords=rank_coords,
            offsets=offsets,
            local_dims=local_dims,
            halo_width=halo_width,
            step=step,
            time=time,
            has_wall=bool(fixed[3]),
            nondimensional=bool(fixed[4]),
            reynolds=float(constants[0]),
            prandtl=float(constants[1]),
            const5=float(constants[2]),
            cp=float(constants[3]),
            tempconst=float(constants[4]),
            tempconst1=float(constants[5]),
        ),
        coordinates=coordinates,
        density=density,
        velocity=velocity,
        pressure=pressure,
        temperature=temperature,
        velocity_gradient=velocity_gradient,
        temperature_gradient=temperature_gradient,
        wall_coordinates=wall_coordinates,
        wall_normal=wall_normal,
    )


def integrate_plane_moments(
    coordinates: np.ndarray,
    density: np.ndarray,
    velocity: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the approved 12 endpoint moments over physical z segments."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (ni,nj,nk,3)")
    point_shape = coordinates.shape[:3]
    if density.shape != point_shape or pressure.shape != point_shape or temperature.shape != point_shape:
        raise ValueError("scalar fields must match the coordinate point shape")
    if velocity.shape != point_shape + (3,):
        raise ValueError("velocity must have shape (ni,nj,nk,3)")
    if point_shape[2] < 2:
        raise ValueError("at least one physical z segment is required")
    if not all(np.all(np.isfinite(field)) for field in (coordinates, density, velocity, pressure, temperature)):
        raise ValueError("host-reference fields must be finite")

    u = velocity[:, :, :, 0]
    v = velocity[:, :, :, 1]
    w = velocity[:, :, :, 2]
    endpoint = np.stack(
        (
            density,
            density * u,
            density * v,
            density * w,
            density * temperature,
            pressure,
            density * u * u,
            density * v * v,
            density * w * w,
            density * u * v,
            density * u * w,
            density * v * w,
        ),
        axis=-1,
    )
    segment_length = np.linalg.norm(np.diff(coordinates, axis=2), axis=-1)
    if np.any(segment_length <= 0.0):
        raise ValueError("physical z segment lengths must be positive")
    measure = np.sum(segment_length, axis=2)
    moments = np.sum(
        0.5 * (endpoint[:, :, :-1, :] + endpoint[:, :, 1:, :]) * segment_length[:, :, :, None],
        axis=2,
    )
    return np.asfortranarray(measure), np.asfortranarray(moments)


def integrate_wall_moments(snapshot: ValidationSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """Independently project and integrate wall pressure, shear, and heat flux."""
    header = snapshot.header
    if not header.has_wall or snapshot.wall_coordinates is None or snapshot.wall_normal is None:
        raise ValueError("wall data are absent from this validation snapshot")
    ni, _, nk = header.local_dims
    hm = header.halo_width
    xwall = snapshot.wall_coordinates
    normal = snapshot.wall_normal
    normal_norm = np.linalg.norm(normal, axis=-1)
    if np.any(normal_norm <= 1.0e-300):
        raise ValueError("wall normal is degenerate")
    if not np.allclose(normal_norm, 1.0, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("wall normal must be unit length")

    point_values = np.empty((ni, nk, 3), dtype=np.float64)
    global_last_i = header.global_dims[0] - 1
    for i in range(ni):
        hi = hm + i
        global_i = header.offsets[0] + i
        if global_i == 0:
            di = xwall[hi + 1, :, :] - xwall[hi, :, :]
        elif global_i == global_last_i:
            di = xwall[hi, :, :] - xwall[hi - 1, :, :]
        else:
            di = xwall[hi + 1, :, :] - xwall[hi - 1, :, :]
        tangent = di - np.sum(di * normal[i, :, :], axis=-1)[:, None] * normal[i, :, :]
        tangent_norm = np.linalg.norm(tangent, axis=-1)
        if np.any(tangent_norm <= 1.0e-300):
            raise ValueError("wall streamwise tangent is degenerate")
        tangent /= tangent_norm[:, None]

        temperature = snapshot.temperature[i, 0, :]
        if header.nondimensional:
            mu = (
                temperature
                * np.sqrt(temperature)
                * header.tempconst1
                / (temperature + header.tempconst)
                / header.reynolds
            )
            conductivity = (mu / header.prandtl) / header.const5
        else:
            temperature_ratio = temperature / 273.15
            mu = (
                1.716e-5
                * temperature_ratio
                * np.sqrt(temperature_ratio)
                * (273.15 + 110.4)
                / (temperature + 110.4)
            )
            conductivity = header.cp * mu / header.prandtl

        gradient = snapshot.velocity_gradient[i, 0, :, :, :]
        divergence = np.trace(gradient, axis1=-2, axis2=-1)
        stress = mu[:, None, None] * (gradient + np.swapaxes(gradient, -1, -2))
        diagonal = np.arange(3)
        stress[:, diagonal, diagonal] -= (2.0 / 3.0) * mu[:, None] * divergence[:, None]
        traction = np.einsum("kij,kj->ki", stress, normal[i, :, :])
        wall_heat = conductivity * np.sum(
            snapshot.temperature_gradient[i, 0, :, :] * normal[i, :, :], axis=-1
        )
        point_values[i, :, 0] = snapshot.pressure[i, 0, :]
        point_values[i, :, 1] = np.sum(tangent * traction, axis=-1)
        point_values[i, :, 2] = wall_heat

    wall_nodes = xwall[hm : hm + ni, :, :]
    segment_length = np.linalg.norm(np.diff(wall_nodes, axis=1), axis=-1)
    if np.any(segment_length <= 0.0):
        raise ValueError("physical wall z segment lengths must be positive")
    measure = np.sum(segment_length, axis=1)
    moments = np.sum(
        0.5 * (point_values[:, :-1, :] + point_values[:, 1:, :])
        * segment_length[:, :, None],
        axis=1,
    )
    return np.asfortranarray(measure), np.asfortranarray(moments)


def _load_compact_statistics_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts/gpu_statistics/compact_statistics.py"
    spec = importlib.util.spec_from_file_location("compact_statistics_oracle_schema", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _independent_derived(
    plane_measure: np.ndarray,
    plane_sum: np.ndarray,
    sample_count: int,
    wall_measure: np.ndarray | None,
    wall_sum: np.ndarray | None,
) -> dict[str, np.ndarray]:
    rho_sum = plane_sum[:, :, 0]
    sample_measure = sample_count * plane_measure
    u = plane_sum[:, :, 1] / rho_sum
    v = plane_sum[:, :, 2] / rho_sum
    w = plane_sum[:, :, 3] / rho_sum
    result = {
        "rho_mean": rho_sum / sample_measure,
        "u_favre": u,
        "v_favre": v,
        "w_favre": w,
        "T_favre": plane_sum[:, :, 4] / rho_sum,
        "p_mean": plane_sum[:, :, 5] / sample_measure,
        "uu_favre": plane_sum[:, :, 6] / rho_sum - u * u,
        "vv_favre": plane_sum[:, :, 7] / rho_sum - v * v,
        "ww_favre": plane_sum[:, :, 8] / rho_sum - w * w,
        "uv_favre": plane_sum[:, :, 9] / rho_sum - u * v,
        "uw_favre": plane_sum[:, :, 10] / rho_sum - u * w,
        "vw_favre": plane_sum[:, :, 11] / rho_sum - v * w,
    }
    if wall_measure is not None and wall_sum is not None:
        denominator = sample_count * wall_measure
        result["wall_p_mean"] = wall_sum[:, 0] / denominator
        result["wall_tau_streamwise_mean"] = wall_sum[:, 1] / denominator
        result["wall_q_normal_mean"] = wall_sum[:, 2] / denominator
    return result


def _compare_array(
    label: str,
    candidate: np.ndarray,
    expected: np.ndarray,
    atol: float,
    rtol: float,
) -> tuple[bool, float, str]:
    if candidate.shape != expected.shape:
        return False, float("inf"), f"{label}: shape {candidate.shape} != {expected.shape}"
    difference = np.abs(candidate - expected)
    max_abs = float(np.max(difference)) if difference.size else 0.0
    passed = bool(np.all(difference <= atol + rtol * np.abs(expected)))
    return passed, max_abs, f"{label}: {'pass' if passed else 'fail'} max_abs={max_abs:.16e}"


def compare_solver_outputs(
    reference_paths: list[Path] | tuple[Path, ...],
    sidecar_dir: Path | str,
    *,
    atol: float,
    rtol: float,
) -> SolverComparison:
    """Compare GPU sidecars with independently integrated CPU sample snapshots."""
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("comparison tolerances must be non-negative")
    snapshots = [read_validation_snapshot(path) for path in reference_paths]
    if not snapshots:
        raise ValueError("no compact validation snapshots were supplied")
    by_rank: dict[int, list[ValidationSnapshot]] = {}
    for snapshot in snapshots:
        by_rank.setdefault(snapshot.header.rank, []).append(snapshot)
    for rank_snapshots in by_rank.values():
        rank_snapshots.sort(key=lambda item: item.header.step)

    compact = _load_compact_statistics_module()
    sidecar_paths = sorted(Path(sidecar_dir).glob("compact_stats.rank*.bin"))
    states = {compact.read_rank_file(path).header.rank: compact.read_rank_file(path) for path in sidecar_paths}
    if set(states) != set(by_rank):
        raise ValueError(
            f"rank set mismatch: reference={sorted(by_rank)} sidecar={sorted(states)}"
        )

    passed = True
    max_abs = 0.0
    details: list[str] = []
    sample_counts = {len(items) for items in by_rank.values()}
    if len(sample_counts) != 1:
        raise ValueError("reference ranks have unequal sample counts")
    sample_count = next(iter(sample_counts))

    for rank in sorted(by_rank):
        rank_snapshots = by_rank[rank]
        first = rank_snapshots[0]
        last = rank_snapshots[-1]
        state = states[rank]
        expected_identity = (
            first.header.global_dims,
            first.header.topology,
            first.header.rank_coords,
            first.header.offsets,
            first.header.local_dims,
            first.header.has_wall,
        )
        for snapshot in rank_snapshots[1:]:
            identity = (
                snapshot.header.global_dims,
                snapshot.header.topology,
                snapshot.header.rank_coords,
                snapshot.header.offsets,
                snapshot.header.local_dims,
                snapshot.header.has_wall,
            )
            if identity != expected_identity:
                raise ValueError(f"rank {rank}: reference decomposition changed between samples")
        candidate_identity = (
            state.header.global_dims,
            state.header.topology,
            state.header.rank_coords,
            state.header.offsets,
            state.header.local_dims,
            state.header.has_wall,
        )
        metadata_ok = (
            candidate_identity == expected_identity
            and state.header.sample_count == sample_count
            and state.header.sampling_start_step == first.header.step
            and state.header.checkpoint_step == last.header.step
            and abs(state.header.sampling_start_time - first.header.time)
            <= atol + rtol * abs(first.header.time)
            and abs(state.header.checkpoint_time - last.header.time)
            <= atol + rtol * abs(last.header.time)
        )
        passed = passed and metadata_ok
        details.append(f"rank{rank}.metadata: {'pass' if metadata_ok else 'fail'}")

        expected_plane_sum = np.zeros_like(state.plane_sum)
        expected_wall_sum = np.zeros_like(state.wall_sum) if state.header.has_wall else None
        expected_plane_measure = None
        expected_wall_measure = None
        for snapshot in rank_snapshots:
            plane_measure, plane_moments = integrate_plane_moments(
                snapshot.coordinates,
                snapshot.density,
                snapshot.velocity,
                snapshot.pressure,
                snapshot.temperature,
            )
            if expected_plane_measure is None:
                expected_plane_measure = plane_measure
            else:
                stable, error, line = _compare_array(
                    f"rank{rank}.reference_plane_measure",
                    plane_measure,
                    expected_plane_measure,
                    atol,
                    rtol,
                )
                if not stable:
                    raise ValueError(line)
                max_abs = max(max_abs, error)
            expected_plane_sum += plane_moments
            if snapshot.header.has_wall:
                wall_measure, wall_moments = integrate_wall_moments(snapshot)
                if expected_wall_measure is None:
                    expected_wall_measure = wall_measure
                else:
                    stable, _, line = _compare_array(
                        f"rank{rank}.reference_wall_measure",
                        wall_measure,
                        expected_wall_measure,
                        atol,
                        rtol,
                    )
                    if not stable:
                        raise ValueError(line)
                expected_wall_sum += wall_moments

        comparisons = (
            ("plane_measure", state.plane_measure, expected_plane_measure),
            ("plane_sum", state.plane_sum, expected_plane_sum),
        )
        for label, candidate, expected in comparisons:
            ok, error, line = _compare_array(
                f"rank{rank}.{label}", candidate, expected, atol, rtol
            )
            passed = passed and ok
            max_abs = max(max_abs, error)
            details.append(line)
        if state.header.has_wall:
            for label, candidate, expected in (
                ("wall_measure", state.wall_measure, expected_wall_measure),
                ("wall_sum", state.wall_sum, expected_wall_sum),
            ):
                ok, error, line = _compare_array(
                    f"rank{rank}.{label}", candidate, expected, atol, rtol
                )
                passed = passed and ok
                max_abs = max(max_abs, error)
                details.append(line)

        expected_derived = _independent_derived(
            expected_plane_measure,
            expected_plane_sum,
            sample_count,
            expected_wall_measure,
            expected_wall_sum,
        )
        candidate_derived = compact.derive_statistics(state)
        for name, expected in expected_derived.items():
            ok, error, line = _compare_array(
                f"rank{rank}.derived.{name}",
                candidate_derived[name],
                expected,
                atol,
                rtol,
            )
            passed = passed and ok
            max_abs = max(max_abs, error)
            details.append(line)

    return SolverComparison(passed, len(states), sample_count, max_abs, tuple(details))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", required=True, type=Path)
    parser.add_argument("--sidecar-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    args = parser.parse_args()
    reference_paths = sorted(
        args.reference_prefix.parent.glob(f"{args.reference_prefix.name}.step*.rank*.bin")
    )
    result = compare_solver_outputs(
        reference_paths, args.sidecar_dir, atol=args.atol, rtol=args.rtol
    )
    lines = [
        f"status: {'pass' if result.passed else 'fail'}",
        f"ranks: {result.rank_count}",
        f"samples_per_rank: {result.sample_count}",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        f"max_abs: {result.max_abs:.16e}",
        "",
        *result.details,
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
