#!/usr/bin/env python3
"""Summarize matched pinned-pipeline/device-aware A800 TGV timings."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingRow:
    label: str
    np: int
    topology: str
    grid: str
    local_grid: str
    halo_transport: str
    seconds: float
    wall_seconds: float
    spread: float
    max_memory_mib: int
    max_utilization_percent: int


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = [
            "label",
            "np",
            "topology",
            "grid",
            "local_grid",
            "halo_transport",
        ]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("empty scaling manifest")
    return rows


def read_timing(row: dict[str, str], run_root: Path) -> TimingRow | None:
    label = row["label"]
    path = run_root / label / f"{label}_timings.tsv"
    if not path.is_file():
        return None
    with path.open(encoding="ascii", newline="") as handle:
        samples = list(csv.DictReader(handle, delimiter="\t"))
    if len(samples) != 5:
        return None
    if any(sample["label"] != label for sample in samples):
        raise ValueError(f"timing label mismatch: {path}")
    values = [float(sample["median_rk_seconds"]) for sample in samples]
    median = statistics.median(values)
    if median <= 0.0:
        raise ValueError(f"non-positive timing: {path}")
    return TimingRow(
        label=label,
        np=int(row["np"]),
        topology=row["topology"],
        grid=row["grid"],
        local_grid=row["local_grid"],
        halo_transport=row["halo_transport"],
        seconds=median,
        wall_seconds=statistics.median(
            float(sample["wall_seconds"]) for sample in samples
        ),
        spread=(max(values) - min(values)) / median,
        max_memory_mib=max(int(sample["max_memory_mib"]) for sample in samples),
        max_utilization_percent=max(
            int(sample["max_utilization_percent"]) for sample in samples
        ),
    )


def load_rows(manifest: Path, run_root: Path) -> list[TimingRow]:
    return [
        timing
        for row in read_manifest(manifest)
        if (timing := read_timing(row, run_root)) is not None
    ]


def common_baseline(rows: list[TimingRow]) -> TimingRow:
    baselines = [row for row in rows if row.np == 1]
    if len(baselines) != 1:
        raise ValueError(f"expected one NP=1 baseline, found {len(baselines)}")
    return baselines[0]


def matched_pairs(rows: list[TimingRow]) -> list[tuple[TimingRow, TimingRow]]:
    grouped: dict[tuple[int, str, str], dict[str, TimingRow]] = {}
    for row in rows:
        if row.np == 1:
            continue
        key = (row.np, row.topology, row.grid)
        grouped.setdefault(key, {})[row.halo_transport] = row
    pairs = []
    for key in sorted(grouped):
        backends = grouped[key]
        if "pinned-pipeline" in backends and "device-aware" in backends:
            pairs.append((backends["pinned-pipeline"], backends["device-aware"]))
    return pairs


def write_rows(path: Path, rows: list[TimingRow], baseline: TimingRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "label",
                "np",
                "topology",
                "halo_transport",
                "median_rk_seconds",
                "speedup_from_np1",
                "parallel_efficiency",
                "relative_spread",
                "median_process_wall_seconds",
                "max_memory_mib",
                "max_utilization_percent",
            ]
        )
        for row in rows:
            speedup = baseline.seconds / row.seconds
            writer.writerow(
                [
                    row.label,
                    row.np,
                    row.topology,
                    row.halo_transport,
                    f"{row.seconds:.12f}",
                    f"{speedup:.12f}",
                    f"{speedup / row.np:.12f}",
                    f"{row.spread:.12f}",
                    f"{row.wall_seconds:.6f}",
                    row.max_memory_mib,
                    row.max_utilization_percent,
                ]
            )


def write_pairs(
    path: Path, pairs: list[tuple[TimingRow, TimingRow]]
) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "np",
                "topology",
                "pinned_pipeline_seconds",
                "device_aware_seconds",
                "transport_speedup",
                "device_aware_change_percent",
            ]
        )
        for pinned, device in pairs:
            speedup = pinned.seconds / device.seconds
            writer.writerow(
                [
                    pinned.np,
                    pinned.topology,
                    f"{pinned.seconds:.12f}",
                    f"{device.seconds:.12f}",
                    f"{speedup:.12f}",
                    f"{100.0 * (device.seconds / pinned.seconds - 1.0):.6f}",
                ]
            )


def write_summary(
    path: Path,
    rows: list[TimingRow],
    baseline: TimingRow,
    pairs: list[tuple[TimingRow, TimingRow]],
    expected_cases: int,
    expected_pairs: int,
) -> None:
    lines = [
        "# A800 CUDA-aware MPI scaling",
        "",
        "- grid: 512^3",
        "- precision: FP64",
        "- filter workspace: full five-component qwork_d",
        "- synchronization: explicit after every kernel",
        "- timing: five process launches, 20 retained complete RK steps each",
        "- field HDF5: disabled; compact statistics retained",
        f"- completed cases: {len(rows)}/{expected_cases}",
        f"- completed matched backend pairs: {len(pairs)}/{expected_pairs}",
        "",
        "## Strong scaling",
        "",
        "| NP | topology | backend | RK time (s) | speedup | parallel efficiency | spread |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        speedup = baseline.seconds / row.seconds
        lines.append(
            f"| {row.np} | `{row.topology}` | `{row.halo_transport}` | "
            f"{row.seconds:.9f} | {speedup:.5f} | "
            f"{100.0 * speedup / row.np:.2f}% | {100.0 * row.spread:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Backend comparison",
            "",
            "Transport speedup is pinned-pipeline time divided by device-aware time.",
            "",
            "| NP | topology | pinned-pipeline (s) | device-aware (s) | transport speedup | device-aware change |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for pinned, device in pairs:
        speedup = pinned.seconds / device.seconds
        change = 100.0 * (device.seconds / pinned.seconds - 1.0)
        lines.append(
            f"| {pinned.np} | `{pinned.topology}` | {pinned.seconds:.9f} | "
            f"{device.seconds:.9f} | {speedup:.5f} | {change:+.3f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--pair-tsv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=19)
    parser.add_argument("--expected-pairs", type=int, default=9)
    args = parser.parse_args()

    rows = load_rows(args.manifest, args.run_root)
    baseline = common_baseline(rows)
    pairs = matched_pairs(rows)
    write_rows(args.output_tsv, rows, baseline)
    write_pairs(args.pair_tsv, pairs)
    write_summary(
        args.summary,
        rows,
        baseline,
        pairs,
        args.expected_cases,
        args.expected_pairs,
    )
    print(f"completed_cases={len(rows)}")
    print(f"completed_pairs={len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
