#!/usr/bin/env python3
"""Native-kind Ma12 interface gate; only ASTR advances the flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_m12_local_case import prepare
from run_compact_flatplate import configure
from compare_flowstate import read_flowstate
from tests.gpu_validation.compare_q_validation_snapshots import snapshot_files
from tests.gpu_validation.run_native_precision_tgv_gate import (
    LABELS, component_report, physical_state, primitive_state,cached_state_report,
)

GRID = (96, 48, 32)


def run_case(case: Path, executable: Path, source: Path, gpu: bool, topology: tuple,
             precision: str, filtering: bool, workspace: str, positivity: str = "off",
             sanitizer: str | None = None, diffusion: str = "643e",
             compact_transport: str = "pinned", buffer_output: bool = False,
             statistics_interval: int = 1) -> dict:
    prepare(source, case, "gpu" if gpu else "cpu", GRID)
    configure(case, "nscbc", diffusion, True, "t" if filtering else "f", True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ASTR_", "OMPI_MCA_", "UCX_"))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)),
        ASTR_WALL_BLOWING_MODE="legacy_random", ASTR_NSCBC_FARFIELD_MODE="incoming_only",
        ASTR_NSCBC_FARFIELD_RHO="1", ASTR_NSCBC_FARFIELD_U="1", ASTR_NSCBC_FARFIELD_V="0",
        ASTR_NSCBC_FARFIELD_W="0", ASTR_NSCBC_FARFIELD_T="1",
        ASTR_GPU_FILTER_WORKSPACE=workspace, ASTR_GPU_SYNC_MODE="explicit",
        ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_MP_POSITIVITY=positivity,
        ASTR_GEOMETRY_DUMP="diagnostics/geometry.h5",
        ASTR_VALIDATION_RHS_PREFIX="diagnostics/state", ASTR_VALIDATION_RHS_STEP="0",
        ASTR_VALIDATION_RHS_STEP_SECONDARY="9", OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
        OMPI_MCA_osc="pt2pt", OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
    env['ASTR_GPU_STATISTICS_INTERVAL'] = str(statistics_interval)
    if buffer_output:
        env.update(ASTR_BENCHMARK_BUFFER_OUTPUT='1', ASTR_GPU_RK_TIMING='1', ASTR_GPU_RANK_RK_TIMING='1',
                   ASTR_GPU_BENCHMARK_NO_FIELD_IO='1')
    if diffusion == "643c":
        env.update(ASTR_GPU_COMPACT_COMPARE="1", ASTR_GPU_COMPACT_TRANSPORT=compact_transport,
                   ASTR_GPU_COMPACT_SYNC="explicit")
        if gpu and compact_transport == "device-aware":
            env.update(OMPI_MCA_pml="ucx", UCX_MEMTYPE_CACHE="n", UCX_TLS="self,sm,cuda_copy")
            env.pop("OMPI_MCA_opal_cuda_support")
    command = ["mpirun", "--bind-to", "none", "-np", str(np.prod(topology))]
    if sanitizer:
        command += ["compute-sanitizer", "--tool", sanitizer, "--error-exitcode", "91"]
    with (case / "run.log").open("w") as log:
        subprocess.run([*command, str(executable.resolve()), "run", "datin/input.dat"], cwd=case, env=env, check=True,
            stdout=log, stderr=subprocess.STDOUT)
    output = (case / "run.log").read_text()
    steps = [int(n) for n in re.findall(r"ASTR_CFL complete_step=(\d+)", output)]
    if steps != list(range(10)) or "COMPUTATION CRASHED" in output or "ERROR STOP" in output:
        raise ValueError(f"not ten valid complete steps: {case}")
    cfl = [float(row.split()[-1]) for row in re.findall(
        r"ASTR_CFL directional/local_sum/upper_bound=([^\n]+)", output)]
    if not np.all(np.isfinite(cfl)) or max(cfl) > 0.5:
        raise ValueError(f"invalid or excessive fixture CFL: {case}")
    if gpu and f"ASTR_GPU_PRECISION_MODE={precision}" not in output:
        raise ValueError(f"unexpected native precision: {case}")
    if diffusion == "643c":
        if "COMPACT GLOBAL: physical-end plate" not in output:
            raise ValueError(f"compact plate not selected: {case}")
        if gpu and len(re.findall(r"COMPACT_SAME_STATE rank/kind/axis", output)) != 6 * int(np.prod(topology)):
            raise ValueError(f"compact plate same-state coverage differs: {case}")
        if gpu and filtering:
            expected = (15 if workspace == "scalar" else 3) * int(np.prod(topology))
            if len(re.findall(r"COMPACT_FILTER_SAME_STATE rank/axis/component", output)) != expected:
                raise ValueError(f"compact plate filter same-state coverage differs: {case}")
    if sanitizer and output.count("ERROR SUMMARY: 0 errors") != int(np.prod(topology)):
        raise ValueError(f"missing clean per-rank sanitizer summaries: {case}")
    dtype = np.dtype("float32" if precision == "fp32" else "float64")
    with h5py.File(case / "diagnostics/geometry.h5") as geometry:
        if any(geometry[key].dtype != dtype or not np.isfinite(geometry[key][...]).all()
               for key in geometry) or np.min(geometry["jacob"][...]) <= 0:
            raise ValueError(f"invalid or wrong-kind geometry: {case}")
    return {"max_cfl_upper": max(cfl), "floating_flags": re.findall(r"Warning: ieee_\w+ is signaling", output),
            "repair_reports": re.findall(r"[^\n]*(?:critical nodes|CRASHFIX|limited_rank_nodes)[^\n]*", output)}


def compare_cases(left: Path, right: Path, precision: str, ranks: int) -> dict:
    sources = [snapshot_files(case / "diagnostics/state", LABELS) for case in (left, right)]
    expected = {f"initialized.step00000000.rk01.rank{r:08d}.bin" for r in range(ranks)}
    expected.update(f"{label}.step{step:08d}.rk{stage:02d}.rank{r:08d}.bin"
        for label in ("pre_rhs", "post_update") for step in (0, 9)
        for stage in (1, 2, 3) for r in range(ranks))
    if any(set(files) != expected for files in sources):
        raise ValueError("missing first/last three-stage q snapshots")
    fields = {suffix: component_report(physical_state(sources[0][suffix], precision),
        physical_state(sources[1][suffix], precision)) for suffix in sorted(expected)}
    partitions = np.loadtxt(left / "datin/parallel.info", skiprows=3, dtype=int, ndmin=2)
    primitives, sensors, caches = {}, {}, {}
    dtype = np.dtype("float32" if precision == "fp32" else "float64")
    for row in partitions:
        rank, _, _, _, im, jm, km, i0, j0, _ = map(int, row)
        sides = ((i0 > 0, i0 + im < GRID[0]), (j0 > 0, j0 + jm < GRID[1]), (True, True))
        for step in (0, 9):
            for stage in (1, 2, 3):
                suffix = f"step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin"
                values = [primitive_state(case / ("diagnostics/state.pre_rhs_primitives." + suffix),
                    precision, halo_sides=sides) for case in (left, right)]
                primitives[suffix] = component_report(*values)
                caches[suffix] = {name: cached_state_report(physical_state(case / ("diagnostics/state.pre_rhs." + suffix),
                    precision), cached, precision, 12) for name, case, cached in
                    (("reference", left, values[0]), ("candidate", right, values[1]))}
                sensor_values, masks = [], []
                for case in (left, right):
                    with (case / ("diagnostics/state.sensor." + suffix)).open("rb") as stream:
                        header = np.fromfile(stream, np.int32, 3)
                        if tuple(header) != (im, jm, km):
                            raise ValueError("shock sensor header differs")
                        n = (im + 1) * (jm + 1) * (km + 1)
                        sensor_values.append(np.fromfile(stream, dtype, n))
                        masks.append(np.fromfile(stream, np.int8))
                    if sensor_values[-1].size != n or masks[-1].size != n or \
                            not np.isfinite(sensor_values[-1]).all() or not np.isin(masks[-1], [0, 1]).all():
                        raise ValueError("invalid shock sensor snapshot")
                sensors[suffix] = {"max_abs": float(np.max(np.abs(sensor_values[0].astype(np.float64) - sensor_values[1]))),
                    "mask_different_nodes": int(np.count_nonzero(masks[0] != masks[1]))}
    names, a = read_flowstate(left)
    other_names, b = read_flowstate(right)
    if names != other_names or a.shape != b.shape or a.shape[0] != 9 or \
            not np.isfinite(a).all() or not np.isfinite(b).all() or not np.array_equal(a[:, 0], b[:, 0]):
        raise ValueError("flowstate sample identity or finite values differ")
    statistics = component_report(a, b)
    if precision == "fp64" and any(row["max_scaled"] > 2e-10 for row in statistics[2:]):
        raise ValueError("FP64 Ma12 statistics regression")
    if precision == "fp64" and any(row["max_scaled"] > 2e-10
            for group in (fields, primitives) for rows in group.values() for row in rows):
        raise ValueError("FP64 Ma12 field regression")
    return {"fields": fields, "primitives": primitives, "sensors": sensors, "same_phase_cache": caches,
            "statistics": dict(zip(names, statistics)),
            "primitive_validity_domain": "physical nodes plus actual MPI/periodic axial halos; no physical-exterior ghosts"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--cpu-exe", type=Path, required=True)
    parser.add_argument("--gpu-exe", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/home/dell/workspace/M12_C13"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diffusion", choices=("643e", "643c"), default="643e")
    parser.add_argument("--compact-transport", choices=("pinned", "device-aware"), default="pinned")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "precision": args.precision, "grid_intervals": GRID,
        "complete_steps": 10, "dt_input": "5.d-3", "blowing": "original legacy_random, fixed topology",
        "diffusion": args.diffusion, "compact_transport": args.compact_transport,
        "fp32_differences": "report-only", "cases": {}}
    for topology in ((1, 1, 1), (2, 1, 1)):
        for filtering in (False, True):
            tag = "x".join(map(str, topology)) + f"_filter_{int(filtering)}"
            info = {}
            for backend, executable in (("cpu", args.cpu_exe), ("gpu", args.gpu_exe)):
                info[backend] = run_case(out / tag / backend, executable, args.source, backend == "gpu",
                    topology, args.precision, filtering, "scalar", diffusion=args.diffusion,
                    compact_transport=args.compact_transport)
                print(f"{args.precision} {tag} {backend}: ten complete steps", flush=True)
            info["comparison"] = compare_cases(out / tag / "cpu", out / tag / "gpu", args.precision, int(np.prod(topology)))
            result["cases"][tag] = info
            (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    tag = "2x1x1_filter_1"
    result["full_run"] = run_case(out / tag / "gpu_full", args.gpu_exe, args.source, True,
        (2, 1, 1), args.precision, True, "full", diffusion=args.diffusion,
        compact_transport=args.compact_transport)
    result["scalar_full"] = compare_cases(out / tag / "gpu", out / tag / "gpu_full", args.precision, 2)
    result["status"] = "passed"
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.precision}: Ma12 phase and scalar/full matrix passed", flush=True)


if __name__ == "__main__":
    main()
