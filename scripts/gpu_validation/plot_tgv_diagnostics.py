#!/usr/bin/env python3
"""Plot TGV CPU/GPU diagnostics from ASTR flowstate.dat outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scienceplots  # noqa: F401
import matplotlib.pyplot as plt


DEFAULT_CPU_FLOWSTATE = Path(
    "tests/gpu_validation/out/tgv_stats_compare_643e_diff_filter_10/cpu/flowstate.dat"
)
DEFAULT_GPU_FLOWSTATE = Path(
    "tests/gpu_validation/out/tgv_stats_compare_643e_diff_filter_10/gpu/flowstate.dat"
)
DEFAULT_OUTPUT_DIR = Path("tests/gpu_validation/out/tgv_diagnostics_l3")
DIAGNOSTICS = ("kenergy", "enstophy", "dissipation")


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


def read_flowstate(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        header = fh.readline().split()
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise ValueError(f"{path}: header has {len(header)} columns, data has {data.shape[1]}")
    return header, data


def column(header: list[str], data: np.ndarray, name: str) -> np.ndarray:
    try:
        index = header.index(name)
    except ValueError as exc:
        raise KeyError(f"missing flowstate column: {name}") from exc
    return data[:, index]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.eps", output_dir / f"{stem}.jpeg"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def plot_value(time: np.ndarray, cpu: np.ndarray, gpu: np.ndarray, name: str, output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(time, cpu, label="CPU", linewidth=1.4)
    ax.plot(time, gpu, "--", label="GPU", linewidth=1.2)
    ax.set_xlabel("$t$")
    ax.set_ylabel(name)
    ax.legend(frameon=False)
    return save_figure(fig, output_dir, f"tgv_{name}_cpu_gpu")


def plot_error(time: np.ndarray, cpu: np.ndarray, gpu: np.ndarray, name: str, output_dir: Path) -> list[Path]:
    diff = np.abs(gpu - cpu)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.semilogy(time, np.maximum(diff, np.finfo(float).tiny), linewidth=1.4)
    ax.set_xlabel("$t$")
    ax.set_ylabel(f"|{name}_GPU - {name}_CPU|")
    return save_figure(fig, output_dir, f"tgv_{name}_abs_error")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, default=DEFAULT_CPU_FLOWSTATE)
    parser.add_argument("--gpu", type=Path, default=DEFAULT_GPU_FLOWSTATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    cpu_header, cpu_data = read_flowstate(args.cpu)
    gpu_header, gpu_data = read_flowstate(args.gpu)
    if cpu_header != gpu_header:
        raise ValueError(f"flowstate headers differ: cpu={cpu_header}, gpu={gpu_header}")
    if cpu_data.shape != gpu_data.shape:
        raise ValueError(f"flowstate shapes differ: cpu={cpu_data.shape}, gpu={gpu_data.shape}")

    time = column(cpu_header, cpu_data, "time")
    gpu_time = column(gpu_header, gpu_data, "time")
    if not np.allclose(time, gpu_time, rtol=0.0, atol=0.0):
        raise ValueError("CPU/GPU flowstate time columns differ")

    summary_path = args.summary or args.output_dir / "tgv_diagnostics_summary.txt"
    lines = [
        "status: pass",
        f"cpu_flowstate: {args.cpu}",
        f"gpu_flowstate: {args.gpu}",
        f"output_dir: {args.output_dir}",
        "",
        "metric final_cpu final_gpu max_abs l2 value_eps value_jpeg error_eps error_jpeg",
    ]

    for name in DIAGNOSTICS:
        cpu = column(cpu_header, cpu_data, name)
        gpu = column(gpu_header, gpu_data, name)
        diff = gpu - cpu
        max_abs = float(np.max(np.abs(diff)))
        l2 = float(np.sqrt(np.mean(diff * diff)))
        value_paths = plot_value(time, cpu, gpu, name, args.output_dir)
        error_paths = plot_error(time, cpu, gpu, name, args.output_dir)
        lines.append(
            " ".join(
                [
                    name,
                    f"{float(cpu[-1]):.16e}",
                    f"{float(gpu[-1]):.16e}",
                    f"{max_abs:.16e}",
                    f"{l2:.16e}",
                    str(value_paths[0]),
                    str(value_paths[1]),
                    str(error_paths[0]),
                    str(error_paths[1]),
                ]
            )
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
