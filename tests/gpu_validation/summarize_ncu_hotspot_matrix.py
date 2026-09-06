#!/usr/bin/env python3

import argparse
import csv
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


STALL_KEYS = {
    "short_sb": "smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued",
    "long_sb": "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued",
    "math": "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued",
    "branch": "smsp__pcsamp_warps_issue_stalled_branch_resolving_not_issued",
    "tex": "smsp__pcsamp_warps_issue_stalled_tex_throttle_not_issued",
}


@dataclass(frozen=True)
class Profile:
    label: str
    kernel: str
    report: Path
    source: Path


def parse_number(value: str) -> float:
    if not value or value == "-":
        return math.nan
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return math.nan


def read_manifest(path: Path) -> list[Profile]:
    profiles = []
    with path.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            profiles.append(
                Profile(
                    label=row["label"],
                    kernel=row["kernel"],
                    report=Path(row["report"]),
                    source=Path(row["source"]),
                )
            )
    if not profiles:
        raise ValueError("NCU manifest contains no profiles")
    return profiles


def import_raw(report: Path) -> dict[str, float]:
    result = subprocess.run(
        [
            "ncu",
            "--import",
            str(report),
            "--page",
            "raw",
            "--csv",
            "--print-units",
            "base",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) < 3:
        raise ValueError(f"unexpected NCU raw output for {report}")
    header, values = rows[0], rows[2]
    if len(header) != len(values):
        raise ValueError(f"mismatched NCU columns for {report}")
    return {name: parse_number(value) for name, value in zip(header, values)}


def metric(metrics: dict[str, float], name: str) -> float:
    if name not in metrics or not math.isfinite(metrics[name]):
        raise ValueError(f"required NCU metric is missing: {name}")
    return metrics[name]


def stall_shares(metrics: dict[str, float]) -> dict[str, float]:
    prefix = "smsp__pcsamp_warps_issue_stalled_"
    suffix = "_not_issued"
    counts = {
        name: value
        for name, value in metrics.items()
        if name.startswith(prefix) and name.endswith(suffix) and math.isfinite(value)
    }
    total = sum(counts.values())
    if total <= 0.0:
        return {key: 0.0 for key in STALL_KEYS}
    return {
        key: 100.0 * metrics.get(metric_name, 0.0) / total
        for key, metric_name in STALL_KEYS.items()
    }


def bottleneck(sm_pct: float, dram_pct: float) -> str:
    if sm_pct >= dram_pct + 15.0:
        return "compute"
    if dram_pct >= sm_pct + 15.0:
        return "memory"
    return "mixed"


def source_hotspots(path: Path, count: int = 3) -> list[tuple[int, int, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        for header in reader:
            if "Line No" in header and "# Samples" in header:
                break
        else:
            raise ValueError(f"NCU source table header is missing in {path}")
        line_index = header.index("Line No")
        source_index = header.index("Source")
        samples_index = header.index("# Samples")
        for row in reader:
            line = row[line_index]
            source = row[source_index].strip()
            samples = row[samples_index]
            if not line.isdigit() or not source or not samples or samples == "-":
                continue
            rows.append((int(parse_number(samples)), int(line), source))
    rows.sort(reverse=True)
    return [(line, samples, source) for samples, line, source in rows[:count]]


def summarize(profiles: list[Profile]) -> tuple[list[dict[str, object]], list[str]]:
    rows = []
    source_sections = []
    for profile in profiles:
        metrics = import_raw(profile.report)
        stalls = stall_shares(metrics)
        sm_pct = metric(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
        dram_pct = metric(metrics, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
        row = {
            "label": profile.label,
            "kernel": profile.kernel,
            "duration_ms": metric(metrics, "gpu__time_duration.sum") / 1.0e6,
            "sm_pct": sm_pct,
            "dram_pct": dram_pct,
            "registers": int(metric(metrics, "launch__registers_per_thread")),
            "occupancy_pct": metric(
                metrics, "sm__warps_active.avg.pct_of_peak_sustained_active"
            ),
            "instructions_m": metric(metrics, "inst_executed") / 1.0e6,
            "excessive_l2_mib": metric(
                metrics, "derived__memory_l2_theoretical_sectors_global_excessive"
            )
            / (1024.0 * 1024.0),
            "branch_uniform_pct": metrics.get(
                "smsp__sass_average_branch_targets_threads_uniform.pct", math.nan
            ),
            "spill_instructions": int(metrics["sass__inst_executed_register_spilling"])
            if math.isfinite(metrics.get("sass__inst_executed_register_spilling", math.nan))
            else 0,
            "bottleneck": bottleneck(sm_pct, dram_pct),
            **stalls,
        }
        rows.append(row)
        source_sections.append(f"### `{profile.label}`")
        hotspots = source_hotspots(profile.source)
        if not hotspots:
            source_sections.append("No source-correlated samples were reported.")
        else:
            for line, samples, source in hotspots:
                source_sections.append(f"- line {line}, {samples} samples: `{source}`")
        source_sections.append("")
    return rows, source_sections


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, rows: list[dict[str, object]], source_sections: list[str]
) -> None:
    lines = [
        "# NCU Hotspot Matrix",
        "",
        "Stall columns are normalized shares of all not-issued warp samples. "
        "The classification compares SM and DRAM peak-throughput percentages.",
        "",
        "| Kernel | ms | SM % | DRAM % | Registers | Occupancy % | Instructions M | Excess L2 MiB | Uniform branch % | Short SB % | Long SB % | Math % | TEX % | Class |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        branch_uniform = row["branch_uniform_pct"]
        branch_text = (
            f"{branch_uniform:.2f}"
            if isinstance(branch_uniform, float) and math.isfinite(branch_uniform)
            else "n/a"
        )
        rendered = {**row, "branch_text": branch_text}
        lines.append(
            "| {label} | {duration_ms:.3f} | {sm_pct:.2f} | {dram_pct:.2f} | "
            "{registers} | {occupancy_pct:.2f} | {instructions_m:.2f} | "
            "{excessive_l2_mib:.2f} | {branch_text} | {short_sb:.2f} | "
            "{long_sb:.2f} | {math:.2f} | {tex:.2f} | {bottleneck} |".format(**rendered)
        )
    lines.extend(["", "## Source Hotspots", "", *source_sections])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    profiles = read_manifest(args.manifest)
    rows, source_sections = summarize(profiles)
    write_tsv(args.tsv, rows)
    write_markdown(args.markdown, rows, source_sections)


if __name__ == "__main__":
    main()
