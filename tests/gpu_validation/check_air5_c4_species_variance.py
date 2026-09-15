#!/usr/bin/env python3
"""Check that a periodic air5 composition wave is diffusively damped."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.gpu_validation.compare_q_validation_snapshots import read_q_snapshot


@dataclass(frozen=True)
class SpeciesVariance:
    mean: float
    variance: float
    mass: float
    species_mass: float


def snapshot_species_variance(paths: list[Path], component: int = 6) -> SpeciesVariance:
    if not paths:
        raise ValueError("no q snapshots were provided")
    density_parts: list[np.ndarray] = []
    species_parts: list[np.ndarray] = []
    for path in paths:
        snapshot = read_q_snapshot(path)
        im, jm, km, hm, numq = snapshot.header
        if not 1 <= component <= numq:
            raise ValueError(f"{path}: component {component} is outside q(1:{numq})")
        shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
        q = snapshot.values.reshape(shape, order="F")
        owned = np.s_[hm : hm + im, hm : hm + jm, hm : hm + km]
        density_parts.append(q[owned + (0,)].reshape(-1))
        species_parts.append(q[owned + (component - 1,)].reshape(-1))

    density = np.concatenate(density_parts)
    rho_species = np.concatenate(species_parts)
    if density.size == 0 or not np.all(density > 0.0):
        raise ValueError("owned q snapshots contain no positive-density points")
    mass = float(np.sum(density))
    species_mass = float(np.sum(rho_species))
    mean = species_mass / mass
    mass_fraction = rho_species / density
    variance = float(np.sum(density * (mass_fraction - mean) ** 2) / mass)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"species variance must be finite and positive, found {variance}")
    return SpeciesVariance(mean, variance, mass, species_mass)


def matching_snapshots(prefix: Path, label: str, stage: int) -> list[Path]:
    pattern = f"{prefix.name}.{label}.step00000000.rk{stage:02d}.rank*.bin"
    return sorted(prefix.parent.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--component", type=int, default=6)
    parser.add_argument("--minimum-relative-drop", type=float, default=1.0e-10)
    parser.add_argument("--mean-rtol", type=float, default=1.0e-11)
    args = parser.parse_args()

    initial_paths = matching_snapshots(args.prefix, "pre_rhs", 1)
    final_paths = matching_snapshots(args.prefix, "post_update", 3)
    if [path.name.split(".rank", 1)[1] for path in initial_paths] != [
        path.name.split(".rank", 1)[1] for path in final_paths
    ]:
        raise ValueError("initial and final rank snapshot sets do not match")

    initial = snapshot_species_variance(initial_paths, args.component)
    final = snapshot_species_variance(final_paths, args.component)
    relative_change = (final.variance - initial.variance) / initial.variance
    mean_relative_change = (final.mean - initial.mean) / max(abs(initial.mean), 1.0e-300)
    passed = (
        relative_change <= -args.minimum_relative_drop
        and abs(mean_relative_change) <= args.mean_rtol
    )
    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"component: {args.component}",
        f"rank_files: {len(initial_paths)}",
        f"initial_mean: {initial.mean:.16e}",
        f"final_mean: {final.mean:.16e}",
        f"mean_relative_change: {mean_relative_change:.16e}",
        f"initial_variance: {initial.variance:.16e}",
        f"final_variance: {final.variance:.16e}",
        f"variance_relative_change: {relative_change:.16e}",
        f"minimum_relative_drop: {args.minimum_relative_drop:.16e}",
        f"mean_rtol: {args.mean_rtol:.16e}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
