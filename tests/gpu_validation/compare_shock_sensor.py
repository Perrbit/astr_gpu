#!/usr/bin/env python3
"""Compare CPU and GPU full-field shock-sensor validation dumps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SensorDump:
    sensor: np.ndarray
    mask: np.ndarray
    offset: tuple[int, int, int] = (0, 0, 0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.sensor.shape


@dataclass(frozen=True)
class SensorComparison:
    passed: bool
    max_abs: float
    max_rel: float
    max_index: tuple[int, int, int]
    mask_mismatches: int


def read_sensor_dump(path: Path) -> SensorDump:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty sensor dump")
    header = lines[0].split()
    if len(header) not in (5, 8) or header[:2] != ["#", "shock_sensor"]:
        raise ValueError(f"{path}: invalid sensor dump header")
    im, jm, km = (int(value) for value in header[2:5])
    offset = (0, 0, 0)
    if len(header) == 8:
        offset = tuple(int(value) for value in header[5:8])
    if min(im, jm, km) < 0:
        raise ValueError(f"{path}: negative dimensions")
    if min(offset) < 0:
        raise ValueError(f"{path}: negative global offset")

    shape = (im + 1, jm + 1, km + 1)
    sensor = np.empty(shape, dtype=np.float64)
    mask = np.empty(shape, dtype=np.int8)
    present = np.zeros(shape, dtype=bool)
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected five columns")
        i, j, k = (int(value) for value in parts[:3])
        if not (0 <= i <= im and 0 <= j <= jm and 0 <= k <= km):
            raise ValueError(f"{path}:{line_number}: index outside declared dimensions")
        if present[i, j, k]:
            raise ValueError(f"{path}:{line_number}: duplicate node {i},{j},{k}")
        sensor[i, j, k] = float(parts[3])
        mask_value = int(parts[4])
        if mask_value not in (0, 1):
            raise ValueError(f"{path}:{line_number}: mask must be zero or one")
        mask[i, j, k] = mask_value
        present[i, j, k] = True

    missing = int(np.count_nonzero(~present))
    if missing:
        raise ValueError(f"{path}: missing {missing} nodes")
    if not np.all(np.isfinite(sensor)):
        raise ValueError(f"{path}: non-finite raw sensor value")
    return SensorDump(sensor=sensor, mask=mask, offset=offset)


def read_sensor_dump_set(path: Path) -> SensorDump:
    path = Path(path)
    if path.exists():
        return read_sensor_dump(path)

    rank_paths = sorted(path.parent.glob(f"{path.name}.rank*"))
    if not rank_paths:
        raise FileNotFoundError(path)
    dumps = [read_sensor_dump(rank_path) for rank_path in rank_paths]
    global_shape = tuple(
        max(dump.offset[axis] + dump.shape[axis] for dump in dumps)
        for axis in range(3)
    )
    sensor = np.empty(global_shape, dtype=np.float64)
    mask = np.empty(global_shape, dtype=np.int8)
    present = np.zeros(global_shape, dtype=bool)

    for dump in dumps:
        slices = tuple(
            slice(dump.offset[axis], dump.offset[axis] + dump.shape[axis])
            for axis in range(3)
        )
        prior = present[slices]
        if np.any(prior):
            prior_sensor = sensor[slices][prior]
            prior_mask = mask[slices][prior]
            current_sensor = dump.sensor[prior]
            current_mask = dump.mask[prior]
            if not np.allclose(prior_sensor, current_sensor, atol=1.0e-12, rtol=1.0e-12):
                raise ValueError(f"{path}: inconsistent overlapping raw sensor")
            if not np.array_equal(prior_mask, current_mask):
                raise ValueError(f"{path}: inconsistent overlapping shock mask")
        sensor[slices] = dump.sensor
        mask[slices] = dump.mask
        present[slices] = True

    missing = int(np.count_nonzero(~present))
    if missing:
        raise ValueError(f"{path}: merged rank dumps leave {missing} global nodes uncovered")
    return SensorDump(sensor=sensor, mask=mask)


def read_sensor_dump_rank_set(path: Path) -> list[SensorDump]:
    path = Path(path)
    if path.exists():
        return [read_sensor_dump(path)]
    rank_paths = sorted(path.parent.glob(f"{path.name}.rank*"))
    if not rank_paths:
        raise FileNotFoundError(path)
    return [read_sensor_dump(rank_path) for rank_path in rank_paths]


def compare_sensor_dumps(
    cpu: SensorDump,
    gpu: SensorDump,
    atol: float,
    rtol: float,
) -> SensorComparison:
    if cpu.shape != gpu.shape:
        raise ValueError(f"sensor shape mismatch: cpu={cpu.shape}, gpu={gpu.shape}")
    difference = np.abs(gpu.sensor - cpu.sensor)
    max_flat_index = int(np.argmax(difference))
    max_index = tuple(int(value) for value in np.unravel_index(max_flat_index, difference.shape))
    max_abs = float(difference[max_index])
    scale = np.maximum(np.abs(cpu.sensor), max(atol, 1.0e-300))
    max_rel = float(np.max(difference / scale))
    mask_mismatches = int(np.count_nonzero(cpu.mask != gpu.mask))
    raw_passed = bool(np.all(difference <= atol + rtol * np.abs(cpu.sensor)))
    return SensorComparison(
        passed=raw_passed and mask_mismatches == 0,
        max_abs=max_abs,
        max_rel=max_rel,
        max_index=max_index,
        mask_mismatches=mask_mismatches,
    )


def compare_sensor_dump_rank_sets(
    cpu_dumps: list[SensorDump],
    gpu_dumps: list[SensorDump],
    atol: float,
    rtol: float,
) -> SensorComparison:
    if len(cpu_dumps) != len(gpu_dumps):
        raise ValueError(f"rank count mismatch: cpu={len(cpu_dumps)} gpu={len(gpu_dumps)}")
    if not cpu_dumps:
        raise ValueError("empty rank dump set")

    comparisons: list[SensorComparison] = []
    for rank, (cpu, gpu) in enumerate(zip(cpu_dumps, gpu_dumps, strict=True)):
        if cpu.offset != gpu.offset:
            raise ValueError(f"rank {rank}: global offset mismatch: cpu={cpu.offset} gpu={gpu.offset}")
        comparisons.append(compare_sensor_dumps(cpu, gpu, atol=atol, rtol=rtol))

    max_rank = max(range(len(comparisons)), key=lambda rank: comparisons[rank].max_abs)
    max_result = comparisons[max_rank]
    global_index = tuple(
        max_result.max_index[axis] + cpu_dumps[max_rank].offset[axis]
        for axis in range(3)
    )
    return SensorComparison(
        passed=all(result.passed for result in comparisons),
        max_abs=max_result.max_abs,
        max_rel=max(result.max_rel for result in comparisons),
        max_index=global_index,
        mask_mismatches=sum(result.mask_mismatches for result in comparisons),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--rankwise", action="store_true")
    args = parser.parse_args()

    if args.rankwise:
        cpu_dumps = read_sensor_dump_rank_set(args.cpu)
        gpu_dumps = read_sensor_dump_rank_set(args.gpu)
        result = compare_sensor_dump_rank_sets(cpu_dumps, gpu_dumps, args.atol, args.rtol)
        cpu_sensor = np.concatenate([dump.sensor.ravel() for dump in cpu_dumps])
        gpu_sensor = np.concatenate([dump.sensor.ravel() for dump in gpu_dumps])
        cpu_mask = np.concatenate([dump.mask.ravel() for dump in cpu_dumps])
        gpu_mask = np.concatenate([dump.mask.ravel() for dump in gpu_dumps])
        shape_text = "rankwise"
    else:
        cpu = read_sensor_dump_set(args.cpu)
        gpu = read_sensor_dump_set(args.gpu)
        result = compare_sensor_dumps(cpu, gpu, args.atol, args.rtol)
        cpu_sensor = cpu.sensor
        gpu_sensor = gpu.sensor
        cpu_mask = cpu.mask
        gpu_mask = gpu.mask
        shape_text = f"{cpu.shape[0]},{cpu.shape[1]},{cpu.shape[2]}"
    index_text = ",".join(str(value) for value in result.max_index)
    lines = [
        f"status: {'pass' if result.passed else 'fail'}",
        f"comparison_mode: {'rankwise' if args.rankwise else 'merged'}",
        f"shape: {shape_text}",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        f"max_abs: {result.max_abs:.16e}",
        f"max_rel: {result.max_rel:.16e}",
        f"max_index: {index_text}",
        f"mask_mismatches: {result.mask_mismatches}",
        f"cpu_min: {float(np.min(cpu_sensor)):.16e}",
        f"cpu_max: {float(np.max(cpu_sensor)):.16e}",
        f"cpu_avg: {float(np.mean(cpu_sensor)):.16e}",
        f"gpu_min: {float(np.min(gpu_sensor)):.16e}",
        f"gpu_max: {float(np.max(gpu_sensor)):.16e}",
        f"gpu_avg: {float(np.mean(gpu_sensor)):.16e}",
        f"cpu_shock_nodes: {int(np.count_nonzero(cpu_mask))}",
        f"gpu_shock_nodes: {int(np.count_nonzero(gpu_mask))}",
    ]
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
