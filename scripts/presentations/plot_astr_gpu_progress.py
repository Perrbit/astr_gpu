#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401


ROOT = Path("/home/dell/workspace/astr_gpu")
BENCHMARK_DIR = ROOT / "tests/gpu_validation/out/group_report_df1961_to_63fe7a8/channel_128_benchmark_20260903"
HISTORICAL_256_CPU = ROOT / "tests/gpu_validation/out/channel_256_cpu_np1_100steps_timed/summary.tsv"
HISTORICAL_256_GPU1 = ROOT / "tests/gpu_validation/out/channel_256_gpu_np1_100steps_timed/summary.tsv"
HISTORICAL_256_MATRIX = ROOT / "tests/gpu_validation/out/channel_256_gpu_mpirank_100steps/gpu_matrix_summary.tsv"
C10_256_TIMINGS = ROOT / "tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark_after_reduction_20260904/timings.tsv"
OUTPUT_DIR = ROOT / "documents/presentations/assets/astr_gpu_progress"

plt.style.use(["science", "ieee", "std-colors"])
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

BLUE = "#2F6FA3"
RED = "#C84747"
TEAL = "#2A8C82"
AMBER = "#B7832F"
CHARCOAL = "#25323A"
GRAY = "#AAB4BA"


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.eps", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.jpeg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_benchmark() -> tuple[float, dict[str, list[float]]]:
    samples: dict[str, list[float]] = {}
    cpu_baseline = None
    with (BENCHMARK_DIR / "benchmark_times.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["role"] == "cpu":
                cpu_baseline = float(row["seconds"])
            else:
                samples.setdefault(row["case"], []).append(float(row["seconds"]))
    with (BENCHMARK_DIR / "gpu_repeats.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            samples.setdefault(row["case"], []).append(float(row["seconds"]))
    if cpu_baseline is None:
        raise ValueError("NP=1 CPU baseline is missing")
    return cpu_baseline, samples


def read_single_timing(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    if row["status"] != "pass":
        raise ValueError(f"Historical timing did not pass: {path}")
    return float(row["seconds"])


def read_historical_256_matrix() -> list[dict[str, str]]:
    with HISTORICAL_256_MATRIX.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failed = [row["case"] for row in rows if row["status"] != "pass"]
    if failed:
        raise ValueError(f"Historical 256 matrix contains failed rows: {failed}")
    return rows


def plot_benchmark() -> None:
    cpu, samples = read_benchmark()
    case_order = [
        "channel_np1_1x1x1",
        "channel_np2_1x2x1",
        "channel_np2_2x1x1",
        "channel_np2_1x1x2",
    ]
    labels = ["GPU 1\n1x1x1", "GPU 2\n1x2x1", "GPU 2\n2x1x1", "GPU 2\n1x1x2"]
    medians = np.array([np.median(samples[name]) for name in case_order])
    lower = np.array([np.median(samples[name]) - min(samples[name]) for name in case_order])
    upper = np.array([max(samples[name]) - np.median(samples[name]) for name in case_order])
    speedups = cpu / medians

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.25), gridspec_kw={"wspace": 0.34})
    x = np.arange(len(labels))
    colors = [BLUE, TEAL, TEAL, TEAL]

    axes[0].bar(-1, cpu, color=RED, width=0.72, label="CPU NP=1")
    axes[0].bar(x, medians, color=colors, width=0.72, label="GPU median")
    axes[0].errorbar(x, medians, yerr=np.vstack([lower, upper]), fmt="none", ecolor=CHARCOAL, capsize=4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time (s), log scale")
    axes[0].set_xticks(np.r_[-1, x])
    axes[0].set_xticklabels(["CPU 1\n1x1x1", *labels])
    axes[0].set_ylim(8, 800)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].text(-1, cpu * 1.08, f"{cpu:.1f}", ha="center", va="bottom", color=RED, fontsize=12)
    for xpos, value in zip(x, medians):
        axes[0].text(xpos, value * 1.09, f"{value:.2f}", ha="center", va="bottom", color=CHARCOAL, fontsize=11)

    axes[1].bar(x, speedups, color=colors, width=0.72)
    axes[1].axhline(1.0, color=GRAY, linewidth=1.0)
    axes[1].set_ylabel("Speedup vs CPU NP=1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 42)
    for xpos, value in zip(x, speedups):
        axes[1].text(xpos, value + 0.8, f"{value:.2f}x", ha="center", va="bottom", color=CHARCOAL, fontsize=12)

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.95)
    save_figure(fig, "channel_128_benchmark")


def plot_historical_256_benchmark() -> None:
    cpu = read_single_timing(HISTORICAL_256_CPU)
    gpu1 = read_single_timing(HISTORICAL_256_GPU1)
    rows = read_historical_256_matrix()
    np2 = [row for row in rows if int(row["np"]) == 2]

    main_labels = ["CPU\nNP1", "GPU\nNP1", "GPU\nNP2-x", "GPU\nNP2-y", "GPU\nNP2-z"]
    main_times = np.array([cpu, gpu1, *[float(row["seconds"]) for row in np2]])
    main_colors = [RED, BLUE, TEAL, TEAL, TEAL]

    stress_rows = [row for row in rows if int(row["np"]) >= 4]
    stress_labels = [f"NP={row['np']}  {row['topology']}" for row in stress_rows]
    stress_times = np.array([float(row["seconds"]) for row in stress_rows])
    stress_colors = [AMBER if int(row["np"]) == 4 else RED for row in stress_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [1.0, 1.18], "wspace": 0.36})
    xpos = np.arange(len(main_labels))
    axes[0].bar(xpos, main_times, color=main_colors, width=0.70)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time (s), log scale")
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(main_labels)
    axes[0].set_ylim(45, 8000)
    for xvalue, value in zip(xpos, main_times):
        axes[0].text(xvalue, value * 1.10, f"{value:.0f}", ha="center", va="bottom", color=CHARCOAL, fontsize=10)

    ypos = np.arange(len(stress_rows))
    axes[1].barh(ypos, stress_times, color=stress_colors, height=0.66)
    axes[1].axvline(gpu1, color=BLUE, linewidth=1.2, linestyle="--", label="GPU NP=1")
    axes[1].set_xlabel("Wall time (s)")
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(stress_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 165)
    axes[1].legend(frameon=False, loc="lower right")
    for yvalue, value in zip(ypos, stress_times):
        axes[1].text(value + 2.0, yvalue, f"{value:.0f}", va="center", color=CHARCOAL, fontsize=9)

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.19, top=0.97)
    save_figure(fig, "channel_256_historical_benchmark")


def plot_problem_size_scaling_context() -> None:
    cpu128, samples128 = read_benchmark()
    del cpu128
    gpu1_128 = np.median(samples128["channel_np1_1x1x1"])
    gpu2_128 = np.median(samples128["channel_np2_1x2x1"])
    gpu1_256 = read_single_timing(HISTORICAL_256_GPU1)
    np2_256 = [row for row in read_historical_256_matrix() if int(row["np"]) == 2]
    gpu2_256 = min(float(row["seconds"]) for row in np2_256)
    gains = np.array([gpu1_128 / gpu2_128, gpu1_256 / gpu2_256])

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(2)
    bars = ax.bar(x, gains, color=[BLUE, TEAL], width=0.58)
    ax.axhline(1.0, color=RED, linewidth=1.2)
    ax.set_ylabel("Two-GPU speedup over one GPU")
    ax.set_xticks(x)
    ax.set_xticklabels(["128³\ncurrent, median", "256³\nhistorical, single run"])
    ax.set_ylim(0, 1.65)
    for bar, value in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}x", ha="center", va="bottom", color=CHARCOAL, fontsize=12)
    ax.text(0.98, 1.02, "break-even", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=RED, fontsize=10)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.23, top=0.96)
    save_figure(fig, "channel_problem_size_scaling_context")


def plot_correctness() -> None:
    labels = ["TGV", "Channel\nstats", "Sod MP7", "CURVE-C10", "CURVE-C12\nstats", "CURVE-C14"]
    errors = np.array([3.98e-13, 4.998e-13, 1.288e-14, 7.604e-13, 4.996e-11, 1.417e-13])
    limits = np.full(6, 1e-10)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.8, 4.15))
    ax.bar(x, errors, color=BLUE, width=0.62, label="Measured maximum error")
    ax.scatter(x, limits, marker="D", color=RED, s=42, zorder=4, label="Acceptance limit")
    ax.set_yscale("log")
    ax.set_ylabel("Absolute error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(3e-16, 3e-8)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    for xpos, value in zip(x, errors):
        ax.text(xpos, value * 1.35, f"{value:.1e}", ha="center", va="bottom", color=CHARCOAL, fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.92)
    save_figure(fig, "numerical_equivalence_errors")


def plot_curvilinear_mesh() -> None:
    im, jm = 24, 18
    xi = np.linspace(0.0, 1.0, im + 1)
    eta = np.linspace(0.0, 1.0, jm + 1)
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    shape = yy * (1.0 - yy)
    xmap = xx + 0.40 * np.sin(np.pi * xx) * shape
    ymap = yy + 0.20 * np.sin(2.0 * np.pi * xx) * shape

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i in range(im + 1):
        ax.plot(xmap[i, :], ymap[i, :], color=BLUE, linewidth=0.55, alpha=0.82)
    for j in range(jm + 1):
        ax.plot(xmap[:, j], ymap[:, j], color=RED, linewidth=0.55, alpha=0.72)
    ax.plot(xmap[:, 0], ymap[:, 0], color=CHARCOAL, linewidth=1.8)
    ax.plot(xmap[:, -1], ymap[:, -1], color=CHARCOAL, linewidth=1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.03, 1.15)
    ax.set_ylim(-0.03, 1.03)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.98)
    save_figure(fig, "curvilinear_mesh")


def read_c10_timings() -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    with C10_256_TIMINGS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            samples.setdefault(row["label"], []).append(float(row["seconds"]))
    if set(samples) != {"np1", "np2_x"}:
        raise ValueError(f"Unexpected C10 timing labels: {sorted(samples)}")
    if any(len(values) != 3 for values in samples.values()):
        raise ValueError("C10 benchmark must contain three repeats per configuration")
    return samples


def plot_latest_performance_summary() -> None:
    cpu128, samples128 = read_benchmark()
    gpu128 = float(np.median(samples128["channel_np1_1x1x1"]))
    c10 = read_c10_timings()
    c10_np1 = float(np.median(c10["np1"]))
    c10_np2 = float(np.median(c10["np2_x"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"wspace": 0.36})

    axes[0].bar([0, 1], [cpu128, gpu128], color=[RED, BLUE], width=0.62)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time (s), log scale")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["CPU NP=1", "GPU NP=1"])
    axes[0].set_ylim(8, 900)
    axes[0].text(0, cpu128 * 1.08, f"{cpu128:.1f} s", ha="center", va="bottom", color=RED, fontsize=12)
    axes[0].text(1, gpu128 * 1.12, f"{gpu128:.2f} s", ha="center", va="bottom", color=BLUE, fontsize=12)
    axes[0].text(0.5, 150, f"{cpu128 / gpu128:.2f}x faster", ha="center", va="center", color=CHARCOAL, fontsize=14)

    medians = [c10_np1, c10_np2]
    lower = [c10_np1 - min(c10["np1"]), c10_np2 - min(c10["np2_x"])]
    upper = [max(c10["np1"]) - c10_np1, max(c10["np2_x"]) - c10_np2]
    axes[1].bar([0, 1], medians, color=[BLUE, TEAL], width=0.62)
    axes[1].errorbar([0, 1], medians, yerr=np.array([lower, upper]), fmt="none", ecolor=CHARCOAL, capsize=5)
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["GPU NP=1", "GPU NP=2 x-slab"])
    axes[1].set_ylim(0, 68)
    axes[1].text(0, c10_np1 + 2.0, f"{c10_np1:.2f} s", ha="center", va="bottom", color=CHARCOAL, fontsize=12)
    axes[1].text(1, c10_np2 + 2.0, f"{c10_np2:.2f} s", ha="center", va="bottom", color=CHARCOAL, fontsize=12)
    axes[1].text(0.5, 63, f"{c10_np1 / c10_np2:.2f}x with 2 GPUs", ha="center", va="center", color=CHARCOAL, fontsize=14)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.95)
    save_figure(fig, "latest_performance_summary")


def plot_hbl_refinement_ratios() -> None:
    labels = ["Hot wall\n$C_f$", "Cold wall\n$C_f$", "Cold wall\n$q_w$", "Velocity\nprofiles", "Temperature\nprofiles"]
    values = np.array([0.4697, 0.4964, 0.5943, 0.4560, 0.4560])
    colors = [RED, BLUE, TEAL, BLUE, TEAL]

    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62)
    ax.axhline(1.0, color=CHARCOAL, linewidth=1.2, linestyle="--")
    ax.set_ylabel("Fine-grid change / coarse-grid change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    for xpos, value in zip(x, values):
        ax.text(xpos, value + 0.035, f"{value:.2f}", ha="center", va="bottom", color=CHARCOAL, fontsize=12)
    ax.text(4.45, 1.02, "smaller than 1: refinement trend passes", ha="right", va="bottom", color=CHARCOAL, fontsize=11)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.95)
    save_figure(fig, "hbl_refinement_ratios")


def main() -> None:
    plot_benchmark()
    plot_historical_256_benchmark()
    plot_problem_size_scaling_context()
    plot_correctness()
    plot_curvilinear_mesh()
    plot_latest_performance_summary()
    plot_hbl_refinement_ratios()


if __name__ == "__main__":
    main()
