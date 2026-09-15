#!/usr/bin/env python3
"""Check one embedded uniform-reactor Strang step from ASTR q snapshots."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tests.gpu_validation.compare_q_validation_snapshots import read_q_snapshot
except ModuleNotFoundError:
    from compare_q_validation_snapshots import read_q_snapshot


PHASES = (
    ("pre_chemistry", 1),
    ("post_chemistry", 1),
    ("post_transport", 1),
    ("post_chemistry", 2),
)


@dataclass(frozen=True)
class ReactorMetrics:
    max_uniformity_error: float
    max_constraint_change: float
    max_transport_change: float
    minimum_chemistry_change: float
    max_species_mass_closure: float
    max_element_relative_change: float


def phase_files(prefix: Path, label: str, stage: int) -> list[Path]:
    pattern = f"{prefix.name}.{label}.step00000000.rk{stage:02d}.rank*.bin"
    files = sorted(prefix.parent.glob(pattern))
    if not files:
        raise ValueError(f"no snapshots match {pattern}")
    return files


def active_values(path: Path) -> np.ndarray:
    snapshot = read_q_snapshot(path)
    im, jm, km, hm, numq = snapshot.header
    if numq != 11:
        raise ValueError(f"{path}: expected numq=11, found {numq}")
    shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
    q = snapshot.values.reshape(shape, order="F")
    active = q[hm : hm + im + 1, hm : hm + jm + 1, hm : hm + km + 1, :]
    return active.reshape((-1, numq))


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


def load_phases(prefix: Path) -> dict[tuple[str, int], np.ndarray]:
    phases: dict[tuple[str, int], np.ndarray] = {}
    expected_ranks: int | None = None
    for label, stage in PHASES:
        files = phase_files(prefix, label, stage)
        if expected_ranks is None:
            expected_ranks = len(files)
        elif len(files) != expected_ranks:
            raise ValueError(
                f"phase {label}/{stage} has {len(files)} ranks, expected {expected_ranks}"
            )
        phases[(label, stage)] = np.concatenate(
            [active_values(path) for path in files], axis=0
        )
    return phases


def element_density(q: np.ndarray, molar_mass: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    return (q[:, 5:10] / molar_mass) @ atoms.T


def analyze(prefix: Path, mechanism: Path) -> ReactorMetrics:
    phases = load_phases(prefix)
    molar_mass, atoms = load_species_data(mechanism)
    pre = phases[("pre_chemistry", 1)]
    post_first = phases[("post_chemistry", 1)]
    post_transport = phases[("post_transport", 1)]
    post_second = phases[("post_chemistry", 2)]
    if len({values.shape for values in phases.values()}) != 1:
        raise ValueError("reactor phase shapes differ")

    uniformity = max(
        float(np.max(np.ptp(values, axis=0))) for values in phases.values()
    )
    constraint_change = max(
        float(np.max(np.abs(post_first[:, :5] - pre[:, :5]))),
        float(np.max(np.abs(post_second[:, :5] - post_transport[:, :5]))),
    )
    transport_change = float(np.max(np.abs(post_transport - post_first)))
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
        scale = np.maximum(np.abs(baseline), 1.0)
        element_relative_change = max(
            element_relative_change, float(np.max(np.abs(final - baseline) / scale))
        )
    return ReactorMetrics(
        max_uniformity_error=uniformity,
        max_constraint_change=constraint_change,
        max_transport_change=transport_change,
        minimum_chemistry_change=chemistry_change,
        max_species_mass_closure=mass_closure,
        max_element_relative_change=element_relative_change,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--element-rtol", type=float, default=5.0e-12)
    parser.add_argument("--minimum-chemistry-change", type=float, default=1.0e-12)
    args = parser.parse_args()
    if min(args.atol, args.element_rtol, args.minimum_chemistry_change) < 0.0:
        raise ValueError("reactor thresholds must be non-negative")

    try:
        metrics = analyze(args.prefix, args.mechanism)
        passed = (
            metrics.max_uniformity_error <= args.atol
            and metrics.max_constraint_change <= args.atol
            and metrics.max_transport_change <= args.atol
            and metrics.minimum_chemistry_change >= args.minimum_chemistry_change
            and metrics.max_species_mass_closure <= args.atol
            and metrics.max_element_relative_change <= args.element_rtol
        )
        lines = [
            f"status: {'pass' if passed else 'fail'}",
            f"max_uniformity_error: {metrics.max_uniformity_error:.16e}",
            f"max_constraint_change: {metrics.max_constraint_change:.16e}",
            f"max_transport_change: {metrics.max_transport_change:.16e}",
            f"minimum_chemistry_change: {metrics.minimum_chemistry_change:.16e}",
            f"max_species_mass_closure: {metrics.max_species_mass_closure:.16e}",
            f"max_element_relative_change: {metrics.max_element_relative_change:.16e}",
            f"absolute_limit: {args.atol:.16e}",
            f"element_relative_limit: {args.element_rtol:.16e}",
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
