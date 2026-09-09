#!/usr/bin/env python3
"""Compare ASTR TGV statistics with the DLR 512^3 spectral reference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    time: float
    energy: float
    dissipation: float
    enstrophy: float


@dataclass(frozen=True)
class Metrics:
    samples: int
    start_time: float
    end_time: float
    energy_relative_l2: float
    dissipation_relative_l2: float
    enstrophy_relative_l2: float
    reference_peak_time: float
    reference_peak_dissipation: float
    astr_peak_time: float
    astr_peak_dissipation: float
    peak_time_error: float
    peak_dissipation_relative_error: float
    aligned: tuple[tuple[Sample, Sample], ...]


def _strictly_increasing(samples: list[Sample], source: Path) -> list[Sample]:
    if len(samples) < 2:
        raise ValueError(f"{source} must contain at least two samples")
    for left, right in zip(samples, samples[1:]):
        if right.time <= left.time:
            raise ValueError(f"times in {source} must be strictly increasing")
    return samples


def read_reference(path: Path) -> list[Sample]:
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise ValueError(f"malformed reference row at {path}:{line_number}")
        time, energy, dissipation, enstrophy = map(float, fields)
        samples.append(Sample(time, energy, dissipation, enstrophy))
    return _strictly_increasing(samples, path)


def read_astr(path: Path, reynolds: float = 1600.0) -> list[Sample]:
    by_time: dict[float, Sample] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#") or fields[0] == "nstep":
            continue
        if len(fields) < 5:
            raise ValueError(f"malformed ASTR row at {path}:{line_number}")
        try:
            _, time, energy, enstrophy, dissipation = map(float, fields[:5])
        except ValueError as exc:
            raise ValueError(f"malformed ASTR row at {path}:{line_number}") from exc
        sample = Sample(time, energy, dissipation, enstrophy)
        previous = by_time.get(time)
        if previous is not None and previous != sample:
            raise ValueError(f"conflicting duplicate ASTR time {time}")
        by_time[time] = sample
    samples = [by_time[key] for key in sorted(by_time)]
    if samples and samples[0].time > 0.0:
        if reynolds <= 0.0:
            raise ValueError("Reynolds number must be positive")
        samples.insert(0, Sample(0.0, 0.125, 0.75 / reynolds, 0.375))
    return _strictly_increasing(samples, path)


def _interpolate(samples: list[Sample], target: float, index: int) -> tuple[Sample, int]:
    while index + 1 < len(samples) and samples[index + 1].time < target:
        index += 1
    left = samples[index]
    if math.isclose(left.time, target, rel_tol=0.0, abs_tol=1.0e-12):
        return Sample(target, left.energy, left.dissipation, left.enstrophy), index
    if index + 1 >= len(samples):
        raise ValueError("ASTR history does not cover the reference interval")
    right = samples[index + 1]
    if target < left.time - 1.0e-12 or target > right.time + 1.0e-12:
        raise ValueError("ASTR history does not cover the reference interval")
    weight = (target - left.time) / (right.time - left.time)
    return (
        Sample(
            target,
            left.energy + weight * (right.energy - left.energy),
            left.dissipation + weight * (right.dissipation - left.dissipation),
            left.enstrophy + weight * (right.enstrophy - left.enstrophy),
        ),
        index,
    )


def _relative_l2(reference: list[float], candidate: list[float]) -> float:
    denominator = sum(value * value for value in reference)
    if denominator == 0.0:
        raise ValueError("reference L2 norm is zero")
    return math.sqrt(
        sum((actual - expected) ** 2 for expected, actual in zip(reference, candidate))
        / denominator
    )


def compare_histories(reference: list[Sample], astr: list[Sample]) -> Metrics:
    if astr[0].time > reference[0].time + 1.0e-12 or astr[-1].time < reference[-1].time - 1.0e-12:
        raise ValueError("ASTR history does not cover the reference interval")
    aligned = []
    index = 0
    for expected in reference:
        actual, index = _interpolate(astr, expected.time, index)
        aligned.append((expected, actual))

    ref_peak = max(reference, key=lambda sample: sample.dissipation)
    astr_window = [sample for sample in astr if reference[0].time <= sample.time <= reference[-1].time]
    astr_peak = max(astr_window, key=lambda sample: sample.dissipation)
    return Metrics(
        samples=len(aligned),
        start_time=reference[0].time,
        end_time=reference[-1].time,
        energy_relative_l2=_relative_l2(
            [pair[0].energy for pair in aligned], [pair[1].energy for pair in aligned]
        ),
        dissipation_relative_l2=_relative_l2(
            [pair[0].dissipation for pair in aligned],
            [pair[1].dissipation for pair in aligned],
        ),
        enstrophy_relative_l2=_relative_l2(
            [pair[0].enstrophy for pair in aligned], [pair[1].enstrophy for pair in aligned]
        ),
        reference_peak_time=ref_peak.time,
        reference_peak_dissipation=ref_peak.dissipation,
        astr_peak_time=astr_peak.time,
        astr_peak_dissipation=astr_peak.dissipation,
        peak_time_error=astr_peak.time - ref_peak.time,
        peak_dissipation_relative_error=(astr_peak.dissipation - ref_peak.dissipation)
        / ref_peak.dissipation,
        aligned=tuple(aligned),
    )


def write_aligned(path: Path, metrics: Metrics) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "time",
                "reference_energy",
                "astr_energy",
                "reference_dissipation",
                "astr_dissipation",
                "reference_enstrophy",
                "astr_enstrophy",
            )
        )
        for expected, actual in metrics.aligned:
            writer.writerow(
                (
                    f"{expected.time:.12g}",
                    f"{expected.energy:.16e}",
                    f"{actual.energy:.16e}",
                    f"{expected.dissipation:.16e}",
                    f"{actual.dissipation:.16e}",
                    f"{expected.enstrophy:.16e}",
                    f"{actual.enstrophy:.16e}",
                )
            )


def write_plots(output_dir: Path, metrics: Metrics) -> None:
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee", "std-colors"])
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.grid"] = False
    plt.rcParams["grid.alpha"] = 0.0
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    rows = (
        ("energy", r"$E_k$"),
        ("dissipation", r"$-\mathrm{d}E_k/\mathrm{d}t$"),
        ("enstrophy", r"$\Omega$"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    times = [expected.time for expected, _ in metrics.aligned]
    for axis, (field, ylabel) in zip(axes, rows):
        axis.plot(times, [getattr(expected, field) for expected, _ in metrics.aligned], "r-", label="DLR spectral")
        axis.plot(times, [getattr(actual, field) for _, actual in metrics.aligned], "b--", label="ASTR GPU")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False)
    axes[-1].set_xlabel(r"$t$")
    fig.tight_layout()
    fig.savefig(output_dir / "tgv_dlr_comparison.eps", format="eps", bbox_inches="tight")
    fig.savefig(output_dir / "tgv_dlr_comparison.jpeg", format="jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--astr-flowstate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    metrics = compare_histories(read_reference(args.reference), read_astr(args.astr_flowstate))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_aligned(args.output_dir / "aligned_diagnostics.tsv", metrics)
    serializable = {key: value for key, value in metrics.__dict__.items() if key != "aligned"}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    summary = [
        "# TGV DLR Spectral Comparison",
        "",
        f"- aligned samples: `{metrics.samples}`",
        f"- reference interval: `{metrics.start_time:.6g}` to `{metrics.end_time:.6g}`",
        "- ASTR `t=0` sample: analytic TGV initial condition",
        f"- energy relative L2: `{metrics.energy_relative_l2:.9e}`",
        f"- dissipation relative L2: `{metrics.dissipation_relative_l2:.9e}`",
        f"- enstrophy relative L2: `{metrics.enstrophy_relative_l2:.9e}`",
        f"- reference dissipation peak: `{metrics.reference_peak_dissipation:.9e}` at `t={metrics.reference_peak_time:.6g}`",
        f"- ASTR dissipation peak: `{metrics.astr_peak_dissipation:.9e}` at `t={metrics.astr_peak_time:.6g}`",
        f"- peak-time error: `{metrics.peak_time_error:.9e}`",
        f"- peak-dissipation relative error: `{metrics.peak_dissipation_relative_error:.9e}`",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="ascii")
    if not args.no_plot:
        write_plots(args.output_dir, metrics)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
