#!/usr/bin/env python3
"""Combine per-case TGV RK and GPU phase timing summaries."""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path


PHASE_ORDER = (
    "prepare",
    "filter",
    "solution_halo",
    "convection",
    "diffusion_flux",
    "diffusion_halo",
    "diffusion_rhs",
    "rk_update",
)


@dataclass(frozen=True)
class PhaseCase:
    label: str
    kind: str
    np: int
    topology: str
    grid: str
    local_grid: str
    halo_transport: str
    sync_mode: str
    rk_seconds: float
    rk_spread: float
    phases: dict[str, float]
    phase_samples: dict[str, int]


def _read_manifest(path: Path) -> list[dict[str, str]]:
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
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected matrix manifest columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("matrix manifest is empty")
    return rows


def _read_rk_timing(path: Path, label: str) -> tuple[float, float]:
    with path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 5:
        raise ValueError(f"expected five timing repeats in {path}, found {len(rows)}")
    if any(row["label"] != label for row in rows):
        raise ValueError(f"timing label mismatch in {path}")
    values = [float(row["median_rk_seconds"]) for row in rows]
    median = statistics.median(values)
    if median <= 0.0:
        raise ValueError(f"non-positive RK timing in {path}")
    return median, (max(values) - min(values)) / median


def _read_phases(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    expected = [
        "phase",
        "samples",
        "median_seconds",
        "min_seconds",
        "max_seconds",
    ]
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected phase summary columns in {path}")
        rows = list(reader)
    if [row["phase"] for row in rows] != list(PHASE_ORDER):
        raise ValueError(f"unexpected phase order in {path}")
    phases = {row["phase"]: float(row["median_seconds"]) for row in rows}
    samples = {row["phase"]: int(row["samples"]) for row in rows}
    if any(value < 0.0 for value in phases.values()):
        raise ValueError(f"negative phase timing in {path}")
    if any(value < 1 for value in samples.values()):
        raise ValueError(f"empty phase timing sample in {path}")
    return phases, samples


def read_cases(manifest_path: Path, run_root: Path) -> list[PhaseCase]:
    cases = []
    for row in _read_manifest(manifest_path):
        label = row["label"]
        case_root = run_root / label
        timing_path = case_root / f"{label}_timings.tsv"
        phase_path = case_root / f"{label}_phase_summary.tsv"
        if not timing_path.is_file() or not phase_path.is_file():
            continue
        rk_seconds, rk_spread = _read_rk_timing(timing_path, label)
        phases, phase_samples = _read_phases(phase_path)
        cases.append(
            PhaseCase(
                label=label,
                kind=row["kind"],
                np=int(row["np"]),
                topology=row["topology"],
                grid=row["grid"],
                local_grid=row["local_grid"],
                halo_transport=row["halo_transport"],
                sync_mode=row["sync_mode"],
                rk_seconds=rk_seconds,
                rk_spread=rk_spread,
                phases=phases,
                phase_samples=phase_samples,
            )
        )
    return cases


def write_tsv(path: Path, cases: list[PhaseCase]) -> None:
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
                "rk_relative_spread",
                "phase",
                "phase_samples",
                "phase_median_seconds",
            ]
        )
        for case in cases:
            for phase in PHASE_ORDER:
                writer.writerow(
                    [
                        case.label,
                        case.kind,
                        case.np,
                        case.topology,
                        case.grid,
                        case.local_grid,
                        case.halo_transport,
                        case.sync_mode,
                        f"{case.rk_seconds:.12e}",
                        f"{case.rk_spread:.12e}",
                        phase,
                        case.phase_samples[phase],
                        f"{case.phases[phase]:.12e}",
                    ]
                )


def _baseline(cases: list[PhaseCase], kind: str) -> PhaseCase | None:
    candidates = [case for case in cases if case.kind == kind and case.np == 1]
    return min(candidates, key=lambda case: case.rk_seconds) if candidates else None


def render_summary(cases: list[PhaseCase], expected_cases: int) -> str:
    lines = [
        "# A800 TGV internal phase diagnostics",
        "",
        "- purpose: attribute the observed multi-GPU scaling loss to solver phases",
        "- precision: FP64; filter workspace: full five-component qwork_d",
        "- synchronization: explicit after every kernel",
        "- phase timing is instrumented and must not replace the uninstrumented scaling result",
        "- phase values are medians per phase occurrence across all five runs and all steps",
        "- prepare is inclusive and must not be summed with the nested RK phases",
        "- field HDF5 output is disabled; compact statistics remain enabled",
        f"- completed matrix entries: {len(cases)}/{expected_cases}",
        "",
        "## Instrumented complete-RK timing",
        "",
        "| kind | NP | topology | halo | RK time (s) | repeat spread |",
        "|---|---:|---|---|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case.kind} | {case.np} | `{case.topology}` | "
            f"`{case.halo_transport}` | {case.rk_seconds:.9f} | "
            f"{100.0 * case.rk_spread:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Median phase occurrence",
            "",
            "All values are milliseconds. Halo columns include packing, MPI exchange and unpacking "
            "inside the corresponding instrumented phase.",
            "",
            "| kind | NP | topology | prepare | filter | solution halo | convection | diffusion flux | diffusion halo | diffusion RHS | RK update |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in cases:
        values = [1000.0 * case.phases[phase] for phase in PHASE_ORDER]
        lines.append(
            f"| {case.kind} | {case.np} | `{case.topology}` | "
            + " | ".join(f"{value:.3f}" for value in values)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Scaling relative to the matching NP=1 workload",
            "",
            "Ratios below are baseline phase time divided by measured phase time. "
            "For strong scaling, ideal ratios are NP. For weak scaling, ideal ratios are one.",
            "",
            "| kind | NP | topology | RK ratio | filter ratio | solution-halo ratio | convection ratio | diffusion-halo ratio | diffusion-RHS ratio |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for kind in ("strong", "weak"):
        baseline = _baseline(cases, kind)
        if baseline is None:
            continue
        for case in cases:
            if case.kind != kind:
                continue
            ratios = [
                baseline.rk_seconds / case.rk_seconds,
                baseline.phases["filter"] / case.phases["filter"],
                baseline.phases["solution_halo"] / case.phases["solution_halo"],
                baseline.phases["convection"] / case.phases["convection"],
                baseline.phases["diffusion_halo"] / case.phases["diffusion_halo"],
                baseline.phases["diffusion_rhs"] / case.phases["diffusion_rhs"],
            ]
            lines.append(
                f"| {kind} | {case.np} | `{case.topology}` | "
                + " | ".join(f"{ratio:.4f}" for ratio in ratios)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)
    cases = read_cases(args.manifest, args.run_root)
    write_tsv(args.output_tsv, cases)
    summary = render_summary(cases, len(manifest))
    args.summary.write_text(summary, encoding="ascii")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
