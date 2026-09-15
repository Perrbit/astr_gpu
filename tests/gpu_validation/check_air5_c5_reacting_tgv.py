#!/usr/bin/env python3
"""Check one high-temperature reacting TGV Strang step from ASTR snapshots."""

from __future__ import annotations

import argparse
import json
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


PHASES = (
    ("pre_chemistry", 1),
    ("post_chemistry", 1),
    ("post_transport", 1),
    ("post_chemistry", 2),
)
RANK_PATTERN = re.compile(r"\.rank(\d{8})\.bin$")


@dataclass(frozen=True)
class ReactingTGVMetrics:
    max_chemistry_constraint_change: float
    minimum_chemistry_change: float
    max_species_mass_closure: float
    max_element_relative_change: float
    minimum_species_density: float
    minimum_temperature: float
    maximum_temperature: float
    minimum_vibrational_temperature: float
    maximum_vibrational_temperature: float
    transport_change: float
    x_velocity_range: float
    y_velocity_range: float
    z_modulation: float


@dataclass(frozen=True)
class RankLayout:
    rank: int
    im: int
    jm: int
    km: int
    i0: int
    j0: int
    k0: int


def phase_files(prefix: Path, label: str, stage: int) -> list[Path]:
    pattern = f"{prefix.name}.{label}.step00000000.rk{stage:02d}.rank*.bin"
    files = sorted(prefix.parent.glob(pattern))
    if not files:
        raise ValueError(f"no snapshots match {pattern}")
    return files


def rank_from_path(path: Path) -> int:
    match = RANK_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse rank from {path}")
    return int(match.group(1))


def active_array(path: Path) -> np.ndarray:
    snapshot = read_q_snapshot(path)
    im, jm, km, hm, numq = snapshot.header
    if numq != 11:
        raise ValueError(f"{path}: expected numq=11, found {numq}")
    shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
    q = snapshot.values.reshape(shape, order="F")
    return q[hm : hm + im + 1, hm : hm + jm + 1, hm : hm + km + 1, :]


def load_parallel_layout(path: Path) -> dict[int, RankLayout]:
    lines = path.read_text(encoding="ascii").splitlines()
    layouts: dict[int, RankLayout] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 10 or not all(field.lstrip("+-").isdigit() for field in fields):
            continue
        rank, _, _, _, im, jm, km, i0, j0, k0 = map(int, fields)
        layouts[rank] = RankLayout(rank, im, jm, km, i0, j0, k0)
    if not layouts:
        raise ValueError(f"no rank records in {path}")
    return layouts


def load_phases(prefix: Path) -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray]:
    phases: dict[tuple[str, int], np.ndarray] = {}
    expected_ranks: set[int] | None = None
    initial_by_rank: dict[int, np.ndarray] = {}
    for label, stage in PHASES:
        files = phase_files(prefix, label, stage)
        ranks = {rank_from_path(path) for path in files}
        if expected_ranks is None:
            expected_ranks = ranks
        elif ranks != expected_ranks:
            raise ValueError(f"phase {label}/{stage} rank set differs")
        arrays = []
        for path in files:
            active = active_array(path)
            arrays.append(active.reshape((-1, 11)))
            if label == "pre_chemistry" and stage == 1:
                initial_by_rank[rank_from_path(path)] = active
        phases[(label, stage)] = np.concatenate(arrays, axis=0)
    if len({values.shape for values in phases.values()}) != 1:
        raise ValueError("reacting TGV phase shapes differ")

    layout_path = prefix.parent.parent / "datin" / "parallel.info"
    layouts = load_parallel_layout(layout_path)
    if set(layouts) != set(initial_by_rank):
        raise ValueError("parallel layout and snapshot rank sets differ")
    ia = max(item.i0 + item.im for item in layouts.values())
    ja = max(item.j0 + item.jm for item in layouts.values())
    ka = max(item.k0 + item.km for item in layouts.values())
    global_initial = np.full((ia + 1, ja + 1, ka + 1, 11), np.nan)
    for rank, values in initial_by_rank.items():
        item = layouts[rank]
        if values.shape[:3] != (item.im + 1, item.jm + 1, item.km + 1):
            raise ValueError(f"rank {rank}: snapshot and parallel layout differ")
        target = global_initial[
            item.i0 : item.i0 + item.im + 1,
            item.j0 : item.j0 + item.jm + 1,
            item.k0 : item.k0 + item.km + 1,
            :,
        ]
        overlap = np.isfinite(target)
        if np.any(overlap) and not np.allclose(target[overlap], values[overlap], rtol=1e-12, atol=1e-10):
            raise ValueError(f"rank {rank}: inconsistent shared-interface values")
        target[...] = values
    if not np.all(np.isfinite(global_initial)):
        raise ValueError("assembled initial TGV field contains holes")
    return phases, global_initial


def load_species_data(mechanism: Path) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(mechanism.read_text(encoding="utf-8"))
    species = document["species"]
    molar_mass = np.asarray(
        [item["molecular_weight_kg_per_mol"] for item in species], dtype=np.float64
    )
    atoms = np.asarray(
        [[item["atoms"][element] for item in species] for element in ("N", "O")],
        dtype=np.float64,
    )
    if molar_mass.shape != (5,) or atoms.shape != (2, 5):
        raise ValueError("fixed air5 mechanism must contain five N/O species")
    return molar_mass, atoms


def element_density(q: np.ndarray, molar_mass: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    return (q[:, 5:10] / molar_mass) @ atoms.T


def reconstruct_temperature_bounds(
    phases: dict[tuple[str, int], np.ndarray], model: Air5RadauReference
) -> tuple[float, float, float, float]:
    temperature_min = np.inf
    temperature_max = -np.inf
    tv_min = np.inf
    tv_max = -np.inf
    for values in phases.values():
        for state in values:
            rho = float(state[0])
            momentum = state[1:4]
            rho_species = state[5:10]
            temperature = model.temperature_from_q5(
                rho, momentum, rho_species, float(state[10]), float(state[4])
            )
            tv = model.tv_from_ev(rho_species, float(state[10]))
            model.pressure(rho_species, temperature)
            temperature_min = min(temperature_min, temperature)
            temperature_max = max(temperature_max, temperature)
            tv_min = min(tv_min, tv)
            tv_max = max(tv_max, tv)
    return temperature_min, temperature_max, tv_min, tv_max


def analyze(prefix: Path, mechanism: Path) -> ReactingTGVMetrics:
    phases, initial = load_phases(prefix)
    model = Air5RadauReference(mechanism)
    molar_mass, atoms = load_species_data(mechanism)
    pre = phases[("pre_chemistry", 1)]
    post_first = phases[("post_chemistry", 1)]
    post_transport = phases[("post_transport", 1)]
    post_second = phases[("post_chemistry", 2)]

    constraint_change = max(
        float(np.max(np.abs(post_first[:, :5] - pre[:, :5]))),
        float(np.max(np.abs(post_second[:, :5] - post_transport[:, :5]))),
    )
    chemistry_change = min(
        float(np.max(np.abs(post_first[:, 5:11] - pre[:, 5:11]))),
        float(np.max(np.abs(post_second[:, 5:11] - post_transport[:, 5:11]))),
    )
    mass_closure = max(
        float(np.max(np.abs(values[:, 0] - np.sum(values[:, 5:10], axis=1))))
        for values in phases.values()
    )
    element_relative_change = 0.0
    for before, after in ((pre, post_first), (post_transport, post_second)):
        baseline = element_density(before, molar_mass, atoms)
        final = element_density(after, molar_mass, atoms)
        scale = np.maximum(np.abs(baseline), np.finfo(float).tiny)
        element_relative_change = max(
            element_relative_change, float(np.max(np.abs(final - baseline) / scale))
        )
    min_species = min(float(np.min(values[:, 5:10])) for values in phases.values())
    temperature_min, temperature_max, tv_min, tv_max = reconstruct_temperature_bounds(
        phases, model
    )
    velocity_x = initial[..., 1] / initial[..., 0]
    velocity_y = initial[..., 2] / initial[..., 0]
    z_modulation = max(
        float(np.max(np.ptp(velocity_x, axis=2))),
        float(np.max(np.ptp(velocity_y, axis=2))),
    )
    return ReactingTGVMetrics(
        max_chemistry_constraint_change=constraint_change,
        minimum_chemistry_change=chemistry_change,
        max_species_mass_closure=mass_closure,
        max_element_relative_change=element_relative_change,
        minimum_species_density=min_species,
        minimum_temperature=temperature_min,
        maximum_temperature=temperature_max,
        minimum_vibrational_temperature=tv_min,
        maximum_vibrational_temperature=tv_max,
        transport_change=float(np.max(np.abs(post_transport - post_first))),
        x_velocity_range=float(np.ptp(velocity_x)),
        y_velocity_range=float(np.ptp(velocity_y)),
        z_modulation=z_modulation,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--constraint-atol", type=float, default=1.0e-8)
    parser.add_argument("--mass-closure-atol", type=float, default=1.0e-10)
    parser.add_argument("--element-rtol", type=float, default=2.0e-11)
    parser.add_argument("--minimum-chemistry-change", type=float, default=1.0e-12)
    parser.add_argument("--minimum-transport-change", type=float, default=1.0e-12)
    parser.add_argument("--minimum-velocity-range", type=float, default=1.0)
    parser.add_argument("--minimum-temperature", type=float, default=5.0e3)
    args = parser.parse_args()
    thresholds = (
        args.constraint_atol,
        args.mass_closure_atol,
        args.element_rtol,
        args.minimum_chemistry_change,
        args.minimum_transport_change,
        args.minimum_velocity_range,
        args.minimum_temperature,
    )
    if min(thresholds) < 0.0:
        raise ValueError("reacting TGV thresholds must be non-negative")

    try:
        metrics = analyze(args.prefix, args.mechanism)
        passed = (
            metrics.max_chemistry_constraint_change <= args.constraint_atol
            and metrics.minimum_chemistry_change >= args.minimum_chemistry_change
            and metrics.max_species_mass_closure <= args.mass_closure_atol
            and metrics.max_element_relative_change <= args.element_rtol
            and metrics.minimum_species_density > 0.0
            and metrics.minimum_temperature >= args.minimum_temperature
            and metrics.transport_change >= args.minimum_transport_change
            and metrics.x_velocity_range >= args.minimum_velocity_range
            and metrics.y_velocity_range >= args.minimum_velocity_range
            and metrics.z_modulation >= args.minimum_velocity_range
        )
        lines = [
            f"status: {'pass' if passed else 'fail'}",
            f"max_chemistry_constraint_change: {metrics.max_chemistry_constraint_change:.16e}",
            f"minimum_chemistry_change: {metrics.minimum_chemistry_change:.16e}",
            f"max_species_mass_closure: {metrics.max_species_mass_closure:.16e}",
            f"max_element_relative_change: {metrics.max_element_relative_change:.16e}",
            f"minimum_species_density: {metrics.minimum_species_density:.16e}",
            f"temperature_range_k: {metrics.minimum_temperature:.16e} {metrics.maximum_temperature:.16e}",
            f"vibrational_temperature_range_k: {metrics.minimum_vibrational_temperature:.16e} {metrics.maximum_vibrational_temperature:.16e}",
            f"transport_change: {metrics.transport_change:.16e}",
            f"x_velocity_range: {metrics.x_velocity_range:.16e}",
            f"y_velocity_range: {metrics.y_velocity_range:.16e}",
            f"z_modulation: {metrics.z_modulation:.16e}",
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
