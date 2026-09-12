#!/usr/bin/env python3
"""Compare CPU and GPU binary q snapshots at matching RK stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QSnapshot:
    header: tuple[int, int, int, int, int]
    values: np.ndarray


@dataclass(frozen=True)
class FileComparison:
    suffix: str
    passed: bool
    max_abs: float
    max_rel: float
    max_index: int


@dataclass(frozen=True)
class SnapshotSetComparison:
    passed: bool
    file_count: int
    max_abs: float
    max_rel: float
    comparisons: tuple[FileComparison, ...]


def read_q_snapshot(path: Path) -> QSnapshot:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        header_array = np.fromfile(stream, dtype=np.int32, count=5)
        values = np.fromfile(stream, dtype=np.float64)
    if header_array.size != 5:
        raise ValueError(f"{path}: incomplete q snapshot header")
    header = tuple(int(value) for value in header_array)
    im, jm, km, hm, numq = header
    if min(im, jm, km, hm) < 0 or numq < 1:
        raise ValueError(f"{path}: invalid q snapshot header {header}")
    expected = (im + 2 * hm + 1) * (jm + 2 * hm + 1) * (km + 2 * hm + 1) * numq
    if values.size != expected:
        raise ValueError(
            f"{path}: q payload size mismatch: expected {expected}, found {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite q snapshot value")
    return QSnapshot(header=header, values=values)


def snapshot_files(prefix: Path, labels: tuple[str, ...]) -> dict[str, Path]:
    prefix = Path(prefix)
    candidates = sorted(prefix.parent.glob(f"{prefix.name}.*.bin"))
    selected: dict[str, Path] = {}
    for path in candidates:
        suffix = path.name[len(prefix.name) + 1 :]
        if any(suffix.startswith(f"{label}.") for label in labels):
            selected[suffix] = path
    return selected


def compare_snapshot_sets(
    cpu_prefix: Path,
    gpu_prefix: Path,
    labels: tuple[str, ...],
    atol: float,
    rtol: float,
) -> SnapshotSetComparison:
    cpu_files = snapshot_files(cpu_prefix, labels)
    gpu_files = snapshot_files(gpu_prefix, labels)
    if not cpu_files and not gpu_files:
        raise ValueError("no matching q snapshot files found")
    if cpu_files.keys() != gpu_files.keys():
        missing_gpu = sorted(cpu_files.keys() - gpu_files.keys())
        missing_cpu = sorted(gpu_files.keys() - cpu_files.keys())
        raise ValueError(
            "snapshot file set mismatch: "
            f"missing_gpu={missing_gpu} missing_cpu={missing_cpu}"
        )

    comparisons: list[FileComparison] = []
    for suffix in sorted(cpu_files):
        cpu = read_q_snapshot(cpu_files[suffix])
        gpu = read_q_snapshot(gpu_files[suffix])
        if cpu.header != gpu.header:
            raise ValueError(
                f"{suffix}: header mismatch: cpu={cpu.header} gpu={gpu.header}"
            )
        difference = np.abs(gpu.values - cpu.values)
        max_index = int(np.argmax(difference))
        max_abs = float(difference[max_index])
        scale = np.maximum(np.abs(cpu.values), max(atol, 1.0e-300))
        max_rel = float(np.max(difference / scale))
        passed = bool(np.all(difference <= atol + rtol * np.abs(cpu.values)))
        comparisons.append(
            FileComparison(
                suffix=suffix,
                passed=passed,
                max_abs=max_abs,
                max_rel=max_rel,
                max_index=max_index,
            )
        )

    return SnapshotSetComparison(
        passed=all(item.passed for item in comparisons),
        file_count=len(comparisons),
        max_abs=max(item.max_abs for item in comparisons),
        max_rel=max(item.max_rel for item in comparisons),
        comparisons=tuple(comparisons),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-prefix", required=True, type=Path)
    parser.add_argument("--gpu-prefix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--labels", default="pre_rhs,post_update")
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    if not labels:
        raise ValueError("at least one snapshot label is required")
    result = compare_snapshot_sets(
        args.cpu_prefix,
        args.gpu_prefix,
        labels=labels,
        atol=args.atol,
        rtol=args.rtol,
    )
    lines = [
        f"status: {'pass' if result.passed else 'fail'}",
        f"files: {result.file_count}",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        f"max_abs: {result.max_abs:.16e}",
        f"max_rel: {result.max_rel:.16e}",
        "",
        "snapshot status max_abs max_rel flat_index",
    ]
    for item in result.comparisons:
        lines.append(
            f"{item.suffix} {'pass' if item.passed else 'fail'} "
            f"{item.max_abs:.16e} {item.max_rel:.16e} {item.max_index}"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
