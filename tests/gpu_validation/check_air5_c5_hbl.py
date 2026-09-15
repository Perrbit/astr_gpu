#!/usr/bin/env python3
"""Validate the fixed-air5 high-enthalpy boundary-layer A0 contract."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tests.gpu_validation.air5_radau_reference import Air5RadauReference
    from tests.gpu_validation.compare_q_validation_snapshots import read_q_snapshot
except ModuleNotFoundError:
    from air5_radau_reference import Air5RadauReference
    from compare_q_validation_snapshots import read_q_snapshot


NUMQ = 11
SPECIES = slice(5, 10)
RANK_PATTERN = re.compile(r"\.rank(\d{8})\.bin$")
CHEMISTRY_PHASES = (
    ("pre_chemistry", 1),
    ("post_chemistry", 1),
    ("post_transport", 1),
    ("post_chemistry", 2),
)
BOUNDARY_PHASES = (("pre_rhs", 1), ("pre_rhs", 2), ("pre_rhs", 3))


@dataclass(frozen=True)
class RankLayout:
    rank: int
    im: int
    jm: int
    km: int
    i0: int
    j0: int
    k0: int


@dataclass(frozen=True)
class HblMetrics:
    minimum_density: float
    minimum_species_density: float
    minimum_total_energy: float
    minimum_vibrational_energy: float
    minimum_temperature: float
    maximum_temperature: float
    minimum_vibrational_temperature: float
    maximum_vibrational_temperature: float
    max_species_mass_closure: float
    max_chemistry_constraint_change: float
    minimum_chemistry_change: float
    max_element_relative_change: float
    inlet_scaled_error: float
    farfield_scaled_error: float
    wall_scaled_error: float
    outflow_scaled_error: float
    z_extrusion_scaled_error: float


def _rank_from_path(path: Path) -> int:
    match = RANK_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse rank from {path}")
    return int(match.group(1))


def _load_parallel_layout(path: Path) -> dict[int, RankLayout]:
    layouts: dict[int, RankLayout] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 10 or not all(field.lstrip("+-").isdigit() for field in fields):
            continue
        rank, _, _, _, im, jm, km, i0, j0, k0 = map(int, fields)
        layouts[rank] = RankLayout(rank, im, jm, km, i0, j0, k0)
    if not layouts:
        raise ValueError(f"no rank records in {path}")
    return layouts


def _phase_files(prefix: Path, label: str, stage: int) -> dict[int, Path]:
    pattern = f"{prefix.name}.{label}.step00000000.rk{stage:02d}.rank*.bin"
    files = {
        _rank_from_path(path): path for path in sorted(prefix.parent.glob(pattern))
    }
    if not files:
        raise ValueError(f"no snapshots match {pattern}")
    return files


def _active_array(path: Path) -> np.ndarray:
    snapshot = read_q_snapshot(path)
    im, jm, km, hm, numq = snapshot.header
    if numq != NUMQ:
        raise ValueError(f"{path}: expected numq={NUMQ}, found {numq}")
    shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
    q = snapshot.values.reshape(shape, order="F")
    return q[hm : hm + im + 1, hm : hm + jm + 1, hm : hm + km + 1, :]


def _assemble_global(
    prefix: Path,
    label: str,
    stage: int,
    layouts: dict[int, RankLayout],
) -> np.ndarray:
    files = _phase_files(prefix, label, stage)
    if set(files) != set(layouts):
        raise ValueError(f"{label}/{stage}: snapshot and layout rank sets differ")
    ia = max(layout.i0 + layout.im for layout in layouts.values())
    ja = max(layout.j0 + layout.jm for layout in layouts.values())
    ka = max(layout.k0 + layout.km for layout in layouts.values())
    result = np.full((ia + 1, ja + 1, ka + 1, NUMQ), np.nan)
    for rank in sorted(files):
        layout = layouts[rank]
        values = _active_array(files[rank])
        expected_shape = (layout.im + 1, layout.jm + 1, layout.km + 1, NUMQ)
        if values.shape != expected_shape:
            raise ValueError(
                f"rank {rank}: snapshot shape {values.shape} != {expected_shape}"
            )
        target = result[
            layout.i0 : layout.i0 + layout.im + 1,
            layout.j0 : layout.j0 + layout.jm + 1,
            layout.k0 : layout.k0 + layout.km + 1,
            :,
        ]
        overlap = np.isfinite(target)
        if np.any(overlap) and not np.allclose(
            target[overlap], values[overlap], atol=1.0e-9, rtol=1.0e-10
        ):
            raise ValueError(f"rank {rank}: inconsistent shared-interface values")
        target[...] = values
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label}/{stage}: assembled field contains holes")
    return result


def _recover_primitives(
    q: np.ndarray, model: Air5RadauReference
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(q, dtype=np.float64)
    flat = values.reshape((-1, NUMQ))
    if not np.all(np.isfinite(flat)):
        raise ValueError("HBL snapshot contains non-finite values")
    rho = flat[:, 0]
    rho_species = flat[:, SPECIES]
    if np.any(rho <= 0.0):
        raise ValueError("HBL snapshot contains non-positive density")
    if np.any(rho_species < 0.0):
        raise ValueError("HBL snapshot contains negative species density")
    if np.any(flat[:, 4] <= 0.0):
        raise ValueError("HBL snapshot contains non-positive total energy")
    if np.any(flat[:, 10] <= 0.0):
        raise ValueError("HBL snapshot contains non-positive vibrational energy")
    momentum = flat[:, 1:4]
    velocity = momentum / rho[:, None]
    cv_density = rho_species @ model.cv_tr
    kinetic = np.sum(momentum * momentum, axis=1) / (2.0 * rho)
    temperature = (
        flat[:, 4]
        - kinetic
        - flat[:, 10]
        - rho_species @ model.formation_energy
    ) / cv_density
    tv = np.asarray(
        [model.tv_from_ev(rho_species[index], flat[index, 10]) for index in range(len(flat))]
    )
    pressure = np.sum(rho_species * model.gas_constant[None, :], axis=1) * temperature
    if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(pressure)):
        raise ValueError("HBL primitive reconstruction is non-finite")
    lower_t, upper_t = model.temperature_bounds
    lower_p, upper_p = model.pressure_bounds
    if np.any(temperature < lower_t) or np.any(temperature > upper_t):
        raise ValueError("HBL translational temperature is outside the mechanism domain")
    if np.any(pressure < lower_p) or np.any(pressure > upper_p):
        raise ValueError("HBL pressure is outside the mechanism domain")
    shape = values.shape[:-1]
    return (
        velocity.reshape((*shape, 3)),
        temperature.reshape(shape),
        tv.reshape(shape),
        pressure.reshape(shape),
        (rho_species / rho[:, None]).reshape((*shape, 5)),
    )


def _profile_q(
    path: Path, y_coordinates: np.ndarray, model: Air5RadauReference
) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] != 13:
        raise ValueError("HBL profile must contain at least two 13-column rows")
    if not np.all(np.isfinite(rows)) or np.any(np.diff(rows[:, 0]) <= 0.0):
        raise ValueError("HBL profile coordinates or values are invalid")

    interpolated = np.column_stack(
        [np.interp(y_coordinates, rows[:, 0], rows[:, column]) for column in range(1, 13)]
    )
    result = np.empty((len(y_coordinates), NUMQ))
    for index, row in enumerate(interpolated):
        velocity = row[1:4]
        pressure = float(row[4])
        temperature = float(row[5])
        tv = float(row[6])
        mass_fraction = row[7:12]
        if np.any(mass_fraction < 0.0) or abs(float(np.sum(mass_fraction)) - 1.0) > 2.0e-12:
            raise ValueError("HBL profile mass fractions are invalid")
        gas_constant = float(np.dot(mass_fraction, model.gas_constant))
        density = pressure / (gas_constant * temperature)
        rho_species = density * mass_fraction
        momentum = density * velocity
        ev = model.ev_from_tv(rho_species, tv)
        result[index, 0] = density
        result[index, 1:4] = momentum
        result[index, 4] = model.q5_from_state(
            density, momentum, rho_species, ev, temperature
        )
        result[index, SPECIES] = rho_species
        result[index, 10] = ev
    return result


def _scaled_error(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = np.maximum(np.abs(expected), 1.0)
    return float(np.max(np.abs(actual - expected) / scale))


def _wall_expected(q: np.ndarray, model: Air5RadauReference, wall_temperature: float) -> np.ndarray:
    ia, _, ka = (extent - 1 for extent in q.shape[:3])
    inner_one = q[1 : ia + 1, 1, :, :]
    inner_two = q[1 : ia + 1, 2, :, :]
    _, t_one, _, p_one, y_one = _recover_primitives(inner_one, model)
    _, _, _, p_two, _ = _recover_primitives(inner_two, model)
    pressure_wall = (4.0 * p_one - p_two) / 3.0
    expected = np.empty((ia, ka + 1, NUMQ))
    for i in range(ia):
        for k in range(ka + 1):
            mass_fraction = y_one[i, k]
            gas_constant = float(np.dot(mass_fraction, model.gas_constant))
            density = pressure_wall[i, k] / (gas_constant * wall_temperature)
            rho_species = density * mass_fraction
            momentum = np.zeros(3)
            ev = model.ev_from_tv(rho_species, wall_temperature)
            expected[i, k, 0] = density
            expected[i, k, 1:4] = momentum
            expected[i, k, 4] = model.q5_from_state(
                density, momentum, rho_species, ev, wall_temperature
            )
            expected[i, k, SPECIES] = rho_species
            expected[i, k, 10] = ev
    return expected


def analyze(
    prefix: Path,
    profile: Path,
    mechanism: Path,
    *,
    ref_len: float,
) -> HblMetrics:
    if ref_len <= 0.0:
        raise ValueError("HBL reference length must be positive")
    layouts = _load_parallel_layout(prefix.parent.parent / "datin" / "parallel.info")
    phases = {
        phase: _assemble_global(prefix, phase[0], phase[1], layouts)
        for phase in CHEMISTRY_PHASES + BOUNDARY_PHASES
    }
    model = Air5RadauReference(mechanism)
    all_values = tuple(phases.values())

    minimum_density = min(float(np.min(values[..., 0])) for values in all_values)
    minimum_species = min(float(np.min(values[..., SPECIES])) for values in all_values)
    minimum_q5 = min(float(np.min(values[..., 4])) for values in all_values)
    minimum_ev = min(float(np.min(values[..., 10])) for values in all_values)
    mass_closure = max(
        float(np.max(np.abs(values[..., 0] - np.sum(values[..., SPECIES], axis=-1))))
        for values in all_values
    )
    temperature_min = np.inf
    temperature_max = -np.inf
    tv_min = np.inf
    tv_max = -np.inf
    for values in all_values:
        _, temperature, tv, _, _ = _recover_primitives(values, model)
        temperature_min = min(temperature_min, float(np.min(temperature)))
        temperature_max = max(temperature_max, float(np.max(temperature)))
        tv_min = min(tv_min, float(np.min(tv)))
        tv_max = max(tv_max, float(np.max(tv)))

    pre = phases[("pre_chemistry", 1)]
    post_first = phases[("post_chemistry", 1)]
    post_transport = phases[("post_transport", 1)]
    post_second = phases[("post_chemistry", 2)]
    ia, ja, ka = (extent - 1 for extent in pre.shape[:3])
    if ia < 3 or ja < 3 or ka < 1:
        raise ValueError("HBL validation grid is too small for the physical-boundary contract")
    core = (slice(1, ia), slice(1, ja), slice(None))
    chemistry_pairs = ((pre[core], post_first[core]), (post_transport[core], post_second[core]))
    constraint_change = max(
        float(np.max(np.abs(after[..., :5] - before[..., :5])))
        for before, after in chemistry_pairs
    )
    chemistry_change = min(
        float(np.max(np.abs(after[..., 5:11] - before[..., 5:11])))
        for before, after in chemistry_pairs
    )
    element_change = 0.0
    atoms = model.atom_counts
    molar_mass = model.molar_mass
    for before, after in chemistry_pairs:
        before_element = (before[..., SPECIES] / molar_mass) @ atoms
        after_element = (after[..., SPECIES] / molar_mass) @ atoms
        scale = np.maximum(np.abs(before_element), 1.0)
        element_change = max(
            element_change,
            float(np.max(np.abs(after_element - before_element) / scale)),
        )

    y_coordinates = 2.0 * ref_len * np.arange(ja + 1, dtype=float) / ja
    profile_q = _profile_q(profile, y_coordinates, model)
    wall_temperature = float(np.loadtxt(profile, dtype=np.float64, ndmin=2)[0, 6])
    inlet_error = 0.0
    farfield_error = 0.0
    wall_error = 0.0
    outflow_error = 0.0
    extrusion_error = 0.0
    for phase in BOUNDARY_PHASES:
        values = phases[phase]
        inlet_expected = np.broadcast_to(profile_q[:, None, :], values[0].shape)
        inlet_error = max(inlet_error, _scaled_error(values[0], inlet_expected))
        farfield_expected = np.broadcast_to(profile_q[-1], values[:, ja, :, :].shape)
        farfield_error = max(
            farfield_error, _scaled_error(values[:, ja, :, :], farfield_expected)
        )
        wall_expected = _wall_expected(values, model, wall_temperature)
        wall_error = max(
            wall_error, _scaled_error(values[1 : ia + 1, 0, :, :], wall_expected)
        )
        outflow_expected = (
            4.0 * values[ia - 1, 1:ja, :, :] - values[ia - 2, 1:ja, :, :]
        ) / 3.0
        outflow_error = max(
            outflow_error,
            _scaled_error(values[ia, 1:ja, :, :], outflow_expected),
        )
        extrusion_expected = np.broadcast_to(values[:, :, :1, :], values.shape)
        extrusion_error = max(
            extrusion_error, _scaled_error(values, extrusion_expected)
        )

    return HblMetrics(
        minimum_density=minimum_density,
        minimum_species_density=minimum_species,
        minimum_total_energy=minimum_q5,
        minimum_vibrational_energy=minimum_ev,
        minimum_temperature=temperature_min,
        maximum_temperature=temperature_max,
        minimum_vibrational_temperature=tv_min,
        maximum_vibrational_temperature=tv_max,
        max_species_mass_closure=mass_closure,
        max_chemistry_constraint_change=constraint_change,
        minimum_chemistry_change=chemistry_change,
        max_element_relative_change=element_change,
        inlet_scaled_error=inlet_error,
        farfield_scaled_error=farfield_error,
        wall_scaled_error=wall_error,
        outflow_scaled_error=outflow_error,
        z_extrusion_scaled_error=extrusion_error,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--ref-len", required=True, type=float)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--mass-closure-atol", type=float, default=1.0e-10)
    parser.add_argument("--constraint-atol", type=float, default=1.0e-8)
    parser.add_argument("--element-rtol", type=float, default=2.0e-11)
    parser.add_argument("--minimum-chemistry-change", type=float, default=1.0e-14)
    parser.add_argument("--boundary-scaled-tol", type=float, default=2.0e-10)
    parser.add_argument("--extrusion-scaled-tol", type=float, default=2.0e-10)
    args = parser.parse_args()
    limits = (
        args.mass_closure_atol,
        args.constraint_atol,
        args.element_rtol,
        args.minimum_chemistry_change,
        args.boundary_scaled_tol,
        args.extrusion_scaled_tol,
    )
    if min(limits) < 0.0:
        raise ValueError("HBL validation limits must be non-negative")

    try:
        metrics = analyze(
            args.prefix, args.profile, args.mechanism, ref_len=args.ref_len
        )
        passed = (
            metrics.minimum_density > 0.0
            and metrics.minimum_species_density >= 0.0
            and metrics.minimum_total_energy > 0.0
            and metrics.minimum_vibrational_energy > 0.0
            and metrics.max_species_mass_closure <= args.mass_closure_atol
            and metrics.max_chemistry_constraint_change <= args.constraint_atol
            and metrics.minimum_chemistry_change >= args.minimum_chemistry_change
            and metrics.max_element_relative_change <= args.element_rtol
            and metrics.inlet_scaled_error <= args.boundary_scaled_tol
            and metrics.farfield_scaled_error <= args.boundary_scaled_tol
            and metrics.wall_scaled_error <= args.boundary_scaled_tol
            and metrics.outflow_scaled_error <= args.boundary_scaled_tol
            and metrics.z_extrusion_scaled_error <= args.extrusion_scaled_tol
        )
        lines = [
            f"status: {'pass' if passed else 'fail'}",
            f"minimum_density: {metrics.minimum_density:.16e}",
            f"minimum_species_density: {metrics.minimum_species_density:.16e}",
            f"minimum_total_energy: {metrics.minimum_total_energy:.16e}",
            f"minimum_vibrational_energy: {metrics.minimum_vibrational_energy:.16e}",
            f"temperature_range_k: {metrics.minimum_temperature:.16e} {metrics.maximum_temperature:.16e}",
            f"vibrational_temperature_range_k: {metrics.minimum_vibrational_temperature:.16e} {metrics.maximum_vibrational_temperature:.16e}",
            f"max_species_mass_closure: {metrics.max_species_mass_closure:.16e}",
            f"max_chemistry_constraint_change: {metrics.max_chemistry_constraint_change:.16e}",
            f"minimum_chemistry_change: {metrics.minimum_chemistry_change:.16e}",
            f"max_element_relative_change: {metrics.max_element_relative_change:.16e}",
            f"inlet_scaled_error: {metrics.inlet_scaled_error:.16e}",
            f"farfield_scaled_error: {metrics.farfield_scaled_error:.16e}",
            f"wall_scaled_error: {metrics.wall_scaled_error:.16e}",
            f"outflow_scaled_error: {metrics.outflow_scaled_error:.16e}",
            f"z_extrusion_scaled_error: {metrics.z_extrusion_scaled_error:.16e}",
        ]
    except (OSError, KeyError, TypeError, ValueError) as error:
        passed = False
        lines = ["status: fail", f"error: {error}"]
    output = "\n".join(lines) + "\n"
    print(output, end="")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(output, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
