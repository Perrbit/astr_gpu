#!/usr/bin/env python3
"""Assemble rank-local compact statistics and derive analysis fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from compact_statistics import (
    MOMENT_NAMES,
    derive_statistics,
    merge_rank_states,
    read_rank_file,
)


DERIVED_NAMES = (
    "rho_mean",
    "u_favre",
    "v_favre",
    "w_favre",
    "T_favre",
    "p_mean",
)
STRESS_NAMES = ("uu_favre", "vv_favre", "ww_favre", "uv_favre", "uw_favre", "vw_favre")


def _rank_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("compact_stats.rank????????.bin"))
    if not files:
        raise ValueError(f"no compact statistics rank files in {input_dir}")
    return files


def _write_hdf5(path: Path, state, derived: dict[str, np.ndarray]) -> None:
    header = state.header
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.create_dataset("version", data=np.int32(1))
        metadata.create_dataset("checkpoint_step", data=np.int64(header.checkpoint_step))
        metadata.create_dataset("checkpoint_time", data=np.float64(header.checkpoint_time))
        metadata.create_dataset("sample_count", data=np.int64(header.sample_count))
        metadata.create_dataset("sampling_start_step", data=np.int64(header.sampling_start_step))
        metadata.create_dataset("sampling_start_time", data=np.float64(header.sampling_start_time))
        metadata.create_dataset("global_dims", data=np.asarray(header.global_dims, dtype=np.int32))
        metadata.create_dataset("topology", data=np.asarray(header.topology, dtype=np.int32))

        measure = handle.create_group("measure")
        measure.create_dataset("plane", data=state.plane_measure)
        if state.header.has_wall:
            measure.create_dataset("wall", data=state.wall_measure)

        raw = handle.create_group("raw")
        for index, name in enumerate(MOMENT_NAMES):
            raw.create_dataset(name, data=state.plane_sum[:, :, index])

        mean = handle.create_group("mean")
        for name in DERIVED_NAMES:
            output_name = name.removesuffix("_mean").removesuffix("_favre")
            mean.create_dataset(output_name, data=derived[name])

        stress = handle.create_group("stress")
        for name in STRESS_NAMES:
            stress.create_dataset(name.removesuffix("_favre"), data=derived[name])

        if state.header.has_wall:
            wall = handle.create_group("wall")
            wall.create_dataset("p", data=derived["wall_p_mean"])
            wall.create_dataset("tau_streamwise", data=derived["wall_tau_streamwise_mean"])
            wall.create_dataset("q_normal", data=derived["wall_q_normal_mean"])


def _write_metadata(path: Path, files: list[Path], state) -> None:
    header = state.header
    rho_sum = state.plane_sum[:, :, 0]
    lines = [
        "format: ASTR compact statistics v1",
        f"rank_files: {len(files)}",
        "global_dims: " + " ".join(str(value) for value in header.global_dims),
        "topology: " + " ".join(str(value) for value in header.topology),
        f"checkpoint_step: {header.checkpoint_step}",
        f"checkpoint_time: {header.checkpoint_time:.16e}",
        f"sample_count: {header.sample_count}",
        f"sampling_start_step: {header.sampling_start_step}",
        f"sampling_start_time: {header.sampling_start_time:.16e}",
        f"plane_measure_min: {float(np.min(state.plane_measure)):.16e}",
        f"plane_measure_max: {float(np.max(state.plane_measure)):.16e}",
        f"density_sum_min: {float(np.min(rho_sum)):.16e}",
        f"density_sum_max: {float(np.max(rho_sum)):.16e}",
    ]
    if header.has_wall:
        lines.extend(
            (
                f"wall_measure_min: {float(np.min(state.wall_measure)):.16e}",
                f"wall_measure_max: {float(np.max(state.wall_measure)):.16e}",
            )
        )
    lines.append("files:")
    lines.extend(f"  {file.name}" for file in files)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def assemble(input_dir: Path | str, output_h5: Path | str, metadata_path: Path | str) -> None:
    """Validate, merge, derive, and write one compact analysis dataset."""
    input_dir = Path(input_dir)
    files = _rank_files(input_dir)
    state = merge_rank_states(read_rank_file(path) for path in files)
    derived = derive_statistics(state)
    _write_hdf5(Path(output_h5), state, derived)
    _write_metadata(Path(metadata_path), files, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    assemble(args.input_dir, args.output_h5, args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
