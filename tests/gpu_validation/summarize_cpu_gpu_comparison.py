#!/usr/bin/env python3
"""Summarize matched CPU/GPU wall-time benchmarks."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    with args.timings.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["case"], row["backend"], int(row["np"]), row["topology"])
            groups[key].append(float(row["wall_seconds"]))

    if not groups or any(len(values) < 3 for values in groups.values()):
        raise SystemExit("each benchmark configuration requires at least three samples")

    rows = []
    for key, values in sorted(groups.items()):
        case, backend, np, topology = key
        rows.append(
            {
                "case": case,
                "backend": backend,
                "np": np,
                "topology": topology,
                "samples": len(values),
                "median_seconds": statistics.median(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="", encoding="ascii") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lookup = {(row["case"], row["backend"], row["np"]): row for row in rows}
    tgv_cpu1 = lookup[("tgv_256", "cpu", 1)]["median_seconds"]
    tgv_gpu1 = lookup[("tgv_256", "gpu", 1)]["median_seconds"]
    tgv_gpu4 = lookup[("tgv_256", "gpu", 4)]["median_seconds"]
    cpu_parallel = [row for row in rows if row["case"] == "tgv_256" and row["backend"] == "cpu"]
    cpu_best = min(cpu_parallel, key=lambda row: row["median_seconds"])
    shock_cpu = lookup[("shuosher_400x128x128", "cpu", 1)]["median_seconds"]
    shock_gpu = lookup[("shuosher_400x128x128", "gpu", 1)]["median_seconds"]

    text = [
        "# A800 CPU/GPU comparison",
        "",
        "All ratios use matched inputs, numerical schemes, step counts, and disabled field output.",
        "",
        f"- TGV CPU NP=1 / GPU NP=1 speedup: `{tgv_cpu1 / tgv_gpu1:.6f}x`",
        f"- TGV best CPU configuration: `NP={cpu_best['np']}` / `{cpu_best['topology']}`",
        f"- TGV best CPU / GPU NP=1 speedup: `{cpu_best['median_seconds'] / tgv_gpu1:.6f}x`",
        f"- TGV best CPU / GPU NP=4 speedup: `{cpu_best['median_seconds'] / tgv_gpu4:.6f}x`",
        f"- Shu-Osher CPU NP=1 / GPU NP=1 speedup: `{shock_cpu / shock_gpu:.6f}x`",
        "",
        "The CPU NP=1 ratio is a legacy-rank comparison. The best-CPU ratio is the node-resource comparison.",
        "",
    ]
    args.summary.write_text("\n".join(text), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
