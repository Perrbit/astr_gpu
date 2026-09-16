#!/usr/bin/env python3
"""Summarize topology-complete TGV strong and weak scaling measurements."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrixRow:
    label: str
    kind: str
    np: int
    topology: str
    grid: str
    local_grid: str
    halo_transport: str
    sync_mode: str
    seconds: float
    wall_seconds: float
    spread: float
    max_memory_mib: int
    max_utilization_percent: int


@dataclass(frozen=True)
class ScalingPoint:
    np: int
    topology: str
    halo_transport: str
    sync_mode: str
    seconds: float
    wall_seconds: float
    speedup: float
    efficiency: float
    throughput_ratio: float


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = [
            "label",
            "kind",
            "np",
            "topology",
            "grid",
            "local_grid",
            "halo_transport",
            "sync_mode",
        ]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected matrix manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("matrix manifest is empty")
    return rows


def _read_case(manifest_row: dict[str, str], run_root: Path) -> MatrixRow | None:
    label = manifest_row["label"]
    timing_path = run_root / label / f"{label}_timings.tsv"
    if not timing_path.is_file():
        return None
    with timing_path.open(encoding="ascii", newline="") as handle:
        timing_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(timing_rows) != 5:
        return None
    if any(row["label"] != label for row in timing_rows):
        raise ValueError(f"timing label mismatch in {timing_path}")
    medians = [float(row["median_rk_seconds"]) for row in timing_rows]
    wall_seconds = [float(row["wall_seconds"]) for row in timing_rows]
    median = statistics.median(medians)
    if median <= 0.0:
        raise ValueError(f"non-positive timing in {timing_path}")
    spread = (max(medians) - min(medians)) / median
    return MatrixRow(
        label=label,
        kind=manifest_row["kind"],
        np=int(manifest_row["np"]),
        topology=manifest_row["topology"],
        grid=manifest_row["grid"],
        local_grid=manifest_row["local_grid"],
        halo_transport=manifest_row["halo_transport"],
        sync_mode=manifest_row["sync_mode"],
        seconds=median,
        wall_seconds=statistics.median(wall_seconds),
        spread=spread,
        max_memory_mib=max(int(row["max_memory_mib"]) for row in timing_rows),
        max_utilization_percent=max(
            int(row["max_utilization_percent"]) for row in timing_rows
        ),
    )


def _best_scaling(rows: list[MatrixRow], kind: str) -> dict[int, ScalingPoint]:
    selected: dict[int, MatrixRow] = {}
    for row in rows:
        if row.kind != kind:
            continue
        previous = selected.get(row.np)
        if previous is None or row.seconds < previous.seconds:
            selected[row.np] = row
    baseline = selected.get(1)
    if baseline is None:
        return {}
    points = {}
    for np, row in sorted(selected.items()):
        speedup = baseline.seconds / row.seconds
        if kind == "strong":
            efficiency = speedup / np
            throughput_ratio = speedup
        else:
            efficiency = speedup
            throughput_ratio = np * speedup
        points[np] = ScalingPoint(
            np=np,
            topology=row.topology,
            halo_transport=row.halo_transport,
            sync_mode=row.sync_mode,
            seconds=row.seconds,
            wall_seconds=row.wall_seconds,
            speedup=speedup,
            efficiency=efficiency,
            throughput_ratio=throughput_ratio,
        )
    return points


def _best_gpu_envelope(rows: list[MatrixRow]) -> dict[int, MatrixRow]:
    baselines = [row for row in rows if row.kind == "strong" and row.np == 1]
    if not baselines:
        return {}
    baseline = min(baselines, key=lambda row: row.seconds)
    selected: dict[int, MatrixRow] = {}
    for row in rows:
        if row.kind not in ("strong", "control") or row.grid != baseline.grid:
            continue
        previous = selected.get(row.np)
        if previous is None or row.seconds < previous.seconds:
            selected[row.np] = row
    return dict(sorted(selected.items()))


def summarize_matrix(
    manifest_path: Path, run_root: Path
) -> tuple[list[MatrixRow], dict[int, ScalingPoint], dict[int, ScalingPoint]]:
    rows = []
    for manifest_row in _read_manifest(manifest_path):
        row = _read_case(manifest_row, run_root)
        if row is not None:
            rows.append(row)
    return rows, _best_scaling(rows, "strong"), _best_scaling(rows, "weak")


def write_tsv(path: Path, rows: list[MatrixRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "label",
                "kind",
                "np",
                "topology",
                "grid",
                "local_grid",
                "halo_transport",
                "sync_mode",
                "median_rk_seconds",
                "median_process_wall_seconds",
                "relative_spread",
                "max_memory_mib",
                "max_utilization_percent",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    row.kind,
                    row.np,
                    row.topology,
                    row.grid,
                    row.local_grid,
                    row.halo_transport,
                    row.sync_mode,
                    f"{row.seconds:.12f}",
                    f"{row.wall_seconds:.6f}",
                    f"{row.spread:.12f}",
                    row.max_memory_mib,
                    row.max_utilization_percent,
                ]
            )


def render_summary(
    rows: list[MatrixRow],
    strong: dict[int, ScalingPoint],
    weak: dict[int, ScalingPoint],
    expected_cases: int,
) -> str:
    lines = [
        "# A800 TGV FP64 Scaling Matrix",
        "",
        "- precision: FP64",
        "- filter workspace: full five-component qwork_d",
        "- filter: explicit tenth order",
        "- derivative: explicit sixth order central",
        "- primary synchronization: explicit after every kernel",
        "- primary multi-rank halo transport: pinned-pipeline",
        "- baseline: GPU NP=1; no CPU timing is included",
        "- timing: five independent runs, 20 retained complete RK steps per run",
        "- statistics remain enabled; process wall time includes initialization, statistics and finalization",
        f"- completed matrix entries: {len(rows)}/{expected_cases}",
        "",
        "## All completed topologies",
        "",
        "| kind | NP | topology | global grid | local grid | halo | sync | RK time (s) | process wall (s) | spread | peak memory (MiB) | peak utilization |",
        "|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.kind} | {row.np} | `{row.topology}` | `{row.grid}` | "
            f"`{row.local_grid}` | `{row.halo_transport}` | `{row.sync_mode}` | "
            f"{row.seconds:.9f} | {row.wall_seconds:.6f} | "
            f"{100.0 * row.spread:.3f}% | "
            f"{row.max_memory_mib} | {row.max_utilization_percent}% |"
        )
    lines.extend(
        [
            "",
            "## Topology-optimized strong scaling",
            "",
            "Strong speedup is GPU T1/Tp. Parallel efficiency is T1/(p Tp).",
            "",
            "| NP | selected topology | halo | sync | RK time (s) | process wall (s) | speedup | efficiency |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for point in strong.values():
        lines.append(
            f"| {point.np} | `{point.topology}` | `{point.halo_transport}` | "
            f"`{point.sync_mode}` | {point.seconds:.9f} | {point.wall_seconds:.6f} | "
            f"{point.speedup:.5f} | {100.0 * point.efficiency:.2f}% |"
        )
    envelope = _best_gpu_envelope(rows)
    if envelope:
        baseline_seconds = envelope[1].seconds
        lines.extend(
            [
                "",
                "## Best measured GPU configuration",
                "",
                "This envelope includes the communication and synchronization controls. "
                "All measured configurations remain listed above.",
                "",
                "| NP | case | topology | halo | sync | RK time (s) | speedup | efficiency |",
                "|---:|---|---|---|---|---:|---:|---:|",
            ]
        )
        for np, row in envelope.items():
            speedup = baseline_seconds / row.seconds
            efficiency = speedup / np
            lines.append(
                f"| {np} | `{row.label}` | `{row.topology}` | "
                f"`{row.halo_transport}` | `{row.sync_mode}` | {row.seconds:.9f} | "
                f"{speedup:.5f} | {100.0 * efficiency:.2f}% |"
            )
    lines.extend(
        [
            "",
            "## Weak scaling at fixed local size",
            "",
            "Each rank owns approximately 256x256x256 points. Weak efficiency is "
            "T1/Tp; throughput ratio is p T1/Tp. These anisotropic global grids are "
            "performance workloads, not TGV physical-validation cases.",
            "",
            "| NP | selected topology | halo | sync | RK time (s) | process wall (s) | weak efficiency | throughput ratio |",
            "|---:|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for point in weak.values():
        lines.append(
            f"| {point.np} | `{point.topology}` | `{point.halo_transport}` | "
            f"`{point.sync_mode}` | {point.seconds:.9f} | {point.wall_seconds:.6f} | "
            f"{100.0 * point.efficiency:.2f}% | {point.throughput_ratio:.5f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    manifest_rows = _read_manifest(args.manifest)
    rows, strong, weak = summarize_matrix(args.manifest, args.run_root)
    write_tsv(args.output_tsv, rows)
    summary = render_summary(rows, strong, weak, len(manifest_rows))
    args.summary.write_text(summary, encoding="ascii")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
