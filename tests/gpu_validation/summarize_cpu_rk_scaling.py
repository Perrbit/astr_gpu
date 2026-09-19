#!/usr/bin/env python3
"""Summarize CPU complete-RK timings from one log per MPI rank count."""

import argparse
import re
import statistics
from pathlib import Path


TIMING = re.compile(
    r"^ASTR_CPU_RK_TIMING\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$",
    re.MULTILINE,
)
LOG_NAME = re.compile(r"np(\d+)\.log$")


def read_samples(path: Path) -> list[float]:
    records = [(int(step), float(total)) for step, _, _, total in TIMING.findall(
        path.read_text(encoding="ascii", errors="replace")
    )]
    if [step for step, _ in records] != list(range(6)):
        raise ValueError(f"{path}: expected CPU RK timing steps 0..5")
    return [total for step, total in records if step > 0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    samples_by_np: dict[int, list[float]] = {}
    for path in sorted(args.logs.glob("np*.log")):
        match = LOG_NAME.search(path.name)
        if match:
            samples_by_np[int(match.group(1))] = read_samples(path)
    if 1 not in samples_by_np:
        raise SystemExit("CPU NP=1 timing log is required")

    baseline = statistics.median(samples_by_np[1])
    rows = []
    for np, samples in sorted(samples_by_np.items()):
        median = statistics.median(samples)
        speedup = baseline / median
        rows.append((np, len(samples), median, speedup, speedup / np, min(samples), max(samples)))

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="ascii") as handle:
        handle.write("np\tsamples\tmedian_rk_seconds\tspeedup\tparallel_efficiency\tmin_rk_seconds\tmax_rk_seconds\n")
        for row in rows:
            handle.write(
                f"{row[0]}\t{row[1]}\t{row[2]:.9f}\t{row[3]:.6f}\t"
                f"{row[4]:.6f}\t{row[5]:.9f}\t{row[6]:.9f}\n"
            )

    lines = [
        "# A800 CPU 512^3 TGV pure-RK scaling",
        "",
        "Step 0 is discarded as warmup. Each reported value is the median of complete RK steps 1--5.",
        "Startup, initialization, CFL reporting, statistics output, and file I/O are outside the timer.",
        "",
        "| MPI ranks | Median RK time (s) | Speedup vs NP=1 | Parallel efficiency |",
        "|---:|---:|---:|---:|",
    ]
    for np, _, median, speedup, efficiency, _, _ in rows:
        lines.append(f"| {np} | {median:.6f} | {speedup:.3f} | {efficiency:.3%} |")
    args.summary.write_text("\n".join(lines) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
