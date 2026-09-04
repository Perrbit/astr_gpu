#!/usr/bin/env python3
"""Summarize three-grid/two-time-step HBL diagnostics on common coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_case(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be LABEL=FILE.npz") from error
    if not label or not path:
        raise argparse.ArgumentTypeError("case must be LABEL=FILE.npz")
    return label, Path(path)


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(value - reference))
    return float(np.linalg.norm(value - reference) / denominator)


def interpolate_wall(source: dict[str, np.ndarray], target_x: np.ndarray, name: str) -> np.ndarray:
    mask = source["analysis_mask"].astype(bool)
    return np.interp(target_x, source["x"][mask], source[name][mask])


def common_wall_coordinates(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> np.ndarray:
    first_x = first["x"][first["analysis_mask"].astype(bool)]
    second_x = second["x"][second["analysis_mask"].astype(bool)]
    lower = max(float(first_x[0]), float(second_x[0]))
    upper = min(float(first_x[-1]), float(second_x[-1]))
    target = second_x[(second_x >= lower) & (second_x <= upper)]
    if target.size < 3:
        raise RuntimeError("wall curves have fewer than three common analysis coordinates")
    return target


def interpolate_profile(
    source: dict[str, np.ndarray], station_index: int, target_y: np.ndarray, name: str
) -> np.ndarray:
    return np.interp(target_y, source["profile_y"][station_index], source[name][station_index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-dt", action="append", required=True, type=parse_case)
    parser.add_argument("--fine-dt", action="append", required=True, type=parse_case)
    parser.add_argument(
        "--required-wall-quantity",
        action="append",
        choices=("cf_geometric", "qw_geometric"),
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if len(args.coarse_dt) != 3 or len(args.fine_dt) != 3:
        raise RuntimeError("exactly three coarse-dt and three fine-dt cases are required")
    if [label for label, _ in args.coarse_dt] != [label for label, _ in args.fine_dt]:
        raise RuntimeError("coarse/fine grid labels must match and remain ordered")

    def load(path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as handle:
            result = {name: np.asarray(handle[name]) for name in handle.files}
        if not all(np.all(np.isfinite(value)) for value in result.values()):
            raise RuntimeError(f"{path}: non-finite diagnostic values")
        return result

    coarse = [(label, load(path)) for label, path in args.coarse_dt]
    fine = [(label, load(path)) for label, path in args.fine_dt]
    required_wall = args.required_wall_quantity or ["cf_geometric", "qw_geometric"]
    required_keys = set(required_wall)
    required_keys.update(
        f"{name}_x{station:.8g}"
        for station in fine[0][1]["stations"]
        for name in ("profile_u", "profile_temperature")
    )
    time_deltas: dict[str, list[float]] = {}
    lines = [
        "status: pending",
        "scope: matched-time transient grid/time consistency; not a steady DNS validation",
        f"required_wall_quantities: {','.join(required_wall)}",
        "",
        "[time_step_sensitivity]",
    ]
    for (label, coarse_data), (_, fine_data) in zip(coarse, fine):
        target_x = common_wall_coordinates(coarse_data, fine_data)
        for name in ("cf_geometric", "qw_geometric"):
            coarse_interp = interpolate_wall(coarse_data, target_x, name)
            fine_interp = interpolate_wall(fine_data, target_x, name)
            delta = relative_l2(coarse_interp, fine_interp)
            time_deltas.setdefault(name, []).append(delta)
            lines.append(
                f"grid={label} quantity={name} coarse_vs_fine_rel_l2="
                f"{delta:.16e}"
            )
        for station_index, station in enumerate(fine_data["stations"]):
            target_y = fine_data["profile_y"][station_index]
            for name in ("profile_u", "profile_temperature"):
                coarse_interp = interpolate_profile(coarse_data, station_index, target_y, name)
                delta = relative_l2(coarse_interp, fine_data[name][station_index])
                key = f"{name}_x{station:.8g}"
                time_deltas.setdefault(key, []).append(delta)
                lines.append(
                    f"grid={label} station={station:.8g} quantity={name} coarse_vs_fine_rel_l2="
                    f"{delta:.16e}"
                )

    lines.extend(("", "[spatial_refinement_at_fine_dt]"))
    spatial_deltas: dict[str, list[float]] = {}
    for pair_index in range(2):
        coarse_label, coarse_data = fine[pair_index]
        fine_label, fine_data = fine[pair_index + 1]
        target_x = common_wall_coordinates(coarse_data, fine_data)
        for name in ("cf_geometric", "qw_geometric"):
            coarse_interp = interpolate_wall(coarse_data, target_x, name)
            fine_interp = interpolate_wall(fine_data, target_x, name)
            delta = relative_l2(coarse_interp, fine_interp)
            spatial_deltas.setdefault(name, []).append(delta)
            lines.append(
                f"pair={coarse_label}_to_{fine_label} quantity={name} rel_l2={delta:.16e}"
            )
        for station_index, station in enumerate(fine_data["stations"]):
            target_y = fine_data["profile_y"][station_index]
            for name in ("profile_u", "profile_temperature"):
                coarse_interp = interpolate_profile(coarse_data, station_index, target_y, name)
                delta = relative_l2(coarse_interp, fine_data[name][station_index])
                key = f"{name}_x{station:.8g}"
                spatial_deltas.setdefault(key, []).append(delta)
                lines.append(
                    f"pair={coarse_label}_to_{fine_label} station={station:.8g} "
                    f"quantity={name} rel_l2={delta:.16e}"
                )

    lines.extend(("", "[spatial_delta_trend]"))
    acceptance = True
    for name, deltas in spatial_deltas.items():
        decreasing = deltas[1] < deltas[0]
        ratio = deltas[1] / deltas[0] if deltas[0] != 0.0 else float("nan")
        required = name in required_keys
        if required and not decreasing:
            acceptance = False
        lines.append(
            f"quantity={name} coarse_medium={deltas[0]:.16e} medium_fine={deltas[1]:.16e} "
            f"ratio={ratio:.16e} decreasing={str(decreasing).lower()} required={str(required).lower()}"
        )

    lines.extend(("", "[time_vs_space_acceptance]"))
    for name in sorted(required_keys):
        temporal = time_deltas[name][-1]
        spatial = spatial_deltas[name][-1]
        dominated = temporal < spatial
        if not dominated:
            acceptance = False
        lines.append(
            f"quantity={name} fine_grid_time_delta={temporal:.16e} "
            f"medium_fine_space_delta={spatial:.16e} time_below_space={str(dominated).lower()}"
        )

    lines[0] = f"status: {'pass' if acceptance else 'fail'}"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\n".join(lines))
    return 0 if acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
