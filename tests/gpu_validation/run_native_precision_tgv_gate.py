#!/usr/bin/env python3
"""Run the approved native-precision TGV matrix; precision differences are reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.gpu_validation.compare_q_validation_snapshots import (
    compare_snapshot_sets,
    read_q_snapshot,
    snapshot_files,
)
from tests.gpu_validation.compare_tgv_flowstate import read_flowstate

LABELS = ("initialized", "pre_rhs", "post_update")
TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))


def physical_state(path: Path, precision: str) -> np.ndarray:
    snapshot = read_q_snapshot(path, precision=precision)
    im, jm, km, halo, components = snapshot.header
    q = snapshot.values.reshape(
        (im + 2 * halo + 1, jm + 2 * halo + 1, km + 2 * halo + 1, components), order="F"
    )[halo:halo + im + 1, halo:halo + jm + 1, halo:halo + km + 1]
    internal = q[..., 4] - np.sum(q[..., 1:4] ** 2, axis=-1) / (q.dtype.type(2) * q[..., 0])
    if np.any(q[..., 0] <= 0) or not np.all(np.isfinite(internal)) or np.any(internal <= 0):
        raise ValueError(f"nonpositive density/internal energy: {path}")
    return q


def component_report(reference: np.ndarray, candidate: np.ndarray) -> list[dict]:
    rows = []
    for m in range(reference.shape[-1]):
        a = reference[..., m].astype(np.float64)
        difference = np.abs(candidate[..., m].astype(np.float64) - a)
        index = np.unravel_index(np.argmax(difference), difference.shape)
        scale = max(1.0, float(np.max(np.abs(a))))
        rows.append({"component": m + 1, "max_abs": float(difference[index]),
                     "max_scaled": float(difference[index]) / scale,
                     "local_ijk": [int(i) for i in index]})
    return rows


def cached_state_report(q: np.ndarray, cached: np.ndarray, precision: str, mach: float) -> dict:
    if not np.array_equal(q[..., 0], cached[..., 0]):
        raise ValueError("cached density differs from authoritative q at the same phase")
    dtype = np.float32 if precision == "fp32" else np.float64
    gamma = dtype(1.4)
    const6 = float(dtype(dtype(1) / (gamma - dtype(1))))
    const2 = float(dtype(gamma * dtype(mach) ** 2))
    rho, velocity, pressure = cached[..., 0].astype(np.float64), cached[..., 1:4].astype(np.float64), cached[..., 4].astype(np.float64)
    reconstructed = np.empty(q.shape, dtype=np.float64)
    reconstructed[..., 0] = rho
    reconstructed[..., 1:4] = rho[..., None] * velocity
    reconstructed[..., 4] = const6 * pressure + 0.5 * rho * np.sum(velocity * velocity, axis=-1)
    rows = component_report(q, reconstructed)
    expected_temperature = pressure / rho * const2
    temperature_error = float(np.max(np.abs(cached[..., 5].astype(np.float64) - expected_temperature)))
    if precision == "fp64" and (max(row["max_scaled"] for row in rows) > 2e-10 or
            temperature_error / max(1.0, float(np.max(np.abs(expected_temperature)))) > 2e-10):
        raise ValueError("FP64 same-phase primitive/state inconsistency")
    return {"conservative": rows, "temperature_max_abs": temperature_error,
            "arithmetic": "offline FP64 diagnostic, constants rounded to the source kind"}


def primitive_state(path: Path, precision: str, *, halo_sides=((True, True),) * 3) -> np.ndarray:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int32, count=4)
        values = np.fromfile(stream, dtype={"fp32": np.float32, "fp64": np.float64}[precision])
    if header.size != 4 or np.any(header < 0):
        raise ValueError(f"invalid primitive header: {path}")
    im, jm, km, halo = (int(i) for i in header)
    shape = (im + 2 * halo + 1, jm + 2 * halo + 1, km + 2 * halo + 1, 6)
    if values.size != np.prod(shape):
        raise ValueError(f"invalid primitive payload: {path}")
    storage = values.reshape(shape, order="F")
    physical = [slice(halo, halo + n + 1) for n in (im, jm, km)]
    # Axis-aligned operators do not read halo edges/corners outside two physical axes.
    for axis in range(3):
        stencil = physical.copy()
        lo, hi = halo_sides[axis]
        stencil[axis] = slice(0 if lo else halo, None if hi else halo + shape[axis] - 2 * halo)
        if not np.all(np.isfinite(storage[tuple(stencil)])):
            raise ValueError(f"nonfinite primitive stencil data: {path}")
    fields = storage[tuple(physical)]
    if np.any(fields[..., [0, 4, 5]] <= 0):
        raise ValueError(f"nonpositive density/pressure/temperature: {path}")
    return fields


def rhs_state(path: Path, precision: str) -> np.ndarray:
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=np.int32, count=4)
        values = np.fromfile(stream, dtype={"fp32": np.float32, "fp64": np.float64}[precision])
    if header.size != 4 or np.any(header[:3] < 0) or header[3] < 1:
        raise ValueError(f"invalid RHS header: {path}")
    shape = tuple(int(n) + 1 for n in header[:3]) + (int(header[3]),)
    if values.size != np.prod(shape) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid RHS payload: {path}")
    return values.reshape(shape, order="F")


def frozen_rhs_report(reference: Path, candidate: Path, precision: str, ranks: int) -> dict:
    reports = {}
    bits = np.uint32 if precision == "fp32" else np.uint64
    for rank in range(ranks):
        suffix = f"step00000000.rk01.rank{rank:08d}.bin"
        x = read_q_snapshot(reference / ("state.pre_rhs." + suffix), precision=precision)
        y = read_q_snapshot(candidate / ("state.pre_rhs." + suffix), precision=precision)
        if x.header != y.header:
            raise ValueError("frozen operator state layouts differ")
        im, jm, km, halo, components = x.header
        shape = (im + 2 * halo + 1, jm + 2 * halo + 1, km + 2 * halo + 1, components)
        a, b = x.values.reshape(shape, order="F"), y.values.reshape(shape, order="F")
        physical = [slice(halo, halo + n + 1) for n in (im, jm, km)]
        for axis in range(3):
            stencil = physical.copy()
            stencil[axis] = slice(None)
            if not np.array_equal(a[tuple(stencil)].view(bits), b[tuple(stencil)].view(bits)):
                raise ValueError(f"operator input is not frozen at {suffix}")
        conv = [rhs_state(root / ("state.conv." + suffix), precision) for root in (reference, candidate)]
        full = [rhs_state(root / ("state.full." + suffix), precision) for root in (reference, candidate)]
        diffusion = [f.astype(np.float64) - c for f, c in zip(full, conv)]
        rows = {"convection": component_report(*conv), "full": component_report(*full),
                "diffusion_increment": component_report(*diffusion)}
        if precision == "fp64" and any(row["max_abs"] > 1e-10 for values in rows.values() for row in values):
            raise ValueError(f"FP64 frozen RHS regression: {suffix}")
        reports[str(rank)] = {"q_and_axial_halos_bit_identical": True, **rows}
    return {"phase": "step0/RK1, filter off",
            "diffusion_definition": "stored full minus stored convective RHS, including accumulation rounding",
            "ranks": reports}


def compare_cases(reference: Path, candidate: Path, precision: str, ranks: int, *,
                  steps: int = 10, halo_sides_by_rank: dict | None = None) -> dict:
    comparison = compare_snapshot_sets(reference / "state", candidate / "state", LABELS,
        atol=1e-10, rtol=0.0, active_only=True, cpu_precision=precision, gpu_precision=precision,
        report_only=precision == "fp32")
    if not comparison.passed:
        raise ValueError(f"FP64 field regression: {candidate}")
    fields = {}
    left = snapshot_files(reference / "state", LABELS)
    right = snapshot_files(candidate / "state", LABELS)
    expected = {f"initialized.step00000000.rk01.rank{r:08d}.bin" for r in range(ranks)}
    expected.update(f"{label}.step{step:08d}.rk{stage:02d}.rank{r:08d}.bin"
        for label in ("pre_rhs", "post_update") for step in (0, steps - 1)
        for stage in (1, 2, 3) for r in range(ranks))
    if set(left) != expected or set(right) != expected:
        raise ValueError(f"initial/final RK snapshot coverage differs: {candidate}")
    for suffix in left:
        fields[suffix] = component_report(physical_state(left[suffix], precision),
                                         physical_state(right[suffix], precision))
    primitives, caches = {}, {}
    for step in (0, steps - 1):
        for stage in (1, 2, 3):
            for rank in range(ranks):
                suffix = f"pre_rhs_primitives.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin"
                sides = ((True, True),) * 3 if halo_sides_by_rank is None else halo_sides_by_rank[rank]
                a = primitive_state(reference / ("state." + suffix), precision, halo_sides=sides)
                b = primitive_state(candidate / ("state." + suffix), precision, halo_sides=sides)
                primitives[suffix] = component_report(a, b)
                q_suffix = suffix.replace("pre_rhs_primitives.", "pre_rhs.")
                caches[suffix] = {name: cached_state_report(physical_state(path / ("state." + q_suffix), precision),
                    values, precision, 0.1) for name, path, values in (("reference", reference, a), ("candidate", candidate, b))}
                if precision == "fp64" and any(row["max_abs"] > 1e-10 for row in primitives[suffix]):
                    raise ValueError(f"FP64 primitive regression: {candidate / suffix}")
    names, reference_rows = read_flowstate(reference)
    _, candidate_rows = read_flowstate(candidate)
    a = np.asarray(reference_rows)
    b = np.asarray(candidate_rows)
    if a.shape != b.shape or a.shape[0] != steps - 1 or not np.array_equal(a[:, 0], b[:, 0]) or \
            not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f"statistical sample identity differs: {candidate}")
    differences = np.max(np.abs(a - b), axis=0)
    if precision == "fp64" and np.any(differences[1:] > 1e-10):
        raise ValueError(f"FP64 statistics regression: {candidate}")
    return {"field_max_abs": comparison.max_abs, "field_max_scaled": comparison.max_scaled,
            "fields": fields, "primitive_order": ["rho", "u", "v", "w", "p", "T"],
            "primitives": primitives, "same_phase_cache": caches,
            "statistics_max_abs": dict(zip(names, differences.tolist()))}


def run_case(directory: Path, executable: Path, gpu: bool, topology: tuple[int, ...],
             filtering: bool, workspace: str, precision: str, scheme: str = "643e",
             compact_transport: str = "pinned") -> None:
    # The existing preparer owns the input syntax and zero-based maxstep convention.
    subprocess.run([sys.executable, str(ROOT / "tests/gpu_validation/prepare_tgv_case.py"),
        "--src-case", str(ROOT / "examples/Taylor_Green_Vortex"), "--dst-case", str(directory),
        "--use-gpu", "t" if gpu else "f", "--grid", "32,32,32", "--maxstep", "9",
        "--feqchkpt", "1", "--feqlist", "1", "--lfilter", "t" if filtering else "f",
        "--diffterm", "t", "--lreadgrid", "f", "--ninit", "0", "--restart", "f",
        "--scheme", scheme, "--deltat", "1.d-3"], check=True, cwd=ROOT)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ASTR_", "OMPI_MCA_", "UCX_"))}
    env.update(UCX_TLS="self,sm,cuda_copy" if gpu else "self,sm", OMPI_MCA_coll_hcoll_enable="0",
        ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)), ASTR_GPU_FILTER_WORKSPACE=workspace,
        ASTR_GPU_HALO_TRANSPORT="pageable", ASTR_GPU_SYNC_MODE="explicit",
        ASTR_VALIDATION_RHS_PREFIX=str(directory / "state"), ASTR_VALIDATION_RHS_STEP="0",
        ASTR_VALIDATION_RHS_STEP_SECONDARY="9")
    if scheme == "643c":
        env.update(ASTR_GPU_COMPACT_COMPARE="1", ASTR_GPU_COMPACT_TRANSPORT=compact_transport,
            ASTR_GPU_COMPACT_SYNC="explicit", ASTR_GPU_HALO_TRANSPORT="pinned",
            OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
            OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
        if gpu and compact_transport == "device-aware":
            env.update(OMPI_MCA_pml="ucx", UCX_MEMTYPE_CACHE="n", UCX_TLS="self,sm,cuda_copy")
            env.pop("OMPI_MCA_opal_cuda_support")
    with (directory / "run.log").open("w") as log:
        subprocess.run(["mpirun", "-np", str(np.prod(topology)), str(executable),
                        "run", "datin/input.tgv"], check=True, cwd=directory, env=env,
                       stdout=log, stderr=subprocess.STDOUT)
    output = (directory / "run.log").read_text()
    steps = [int(n) for n in re.findall(r"ASTR_CFL complete_step=(\d+)", output)]
    if steps != list(range(10)):
        raise ValueError(f"did not complete ten steps: {directory}; observed {steps}")
    if gpu and f"ASTR_GPU_PRECISION_MODE={precision}" not in output:
        raise ValueError(f"unexpected numerical precision: {directory}")
    if scheme == "643c":
        if "COMPACT GLOBAL: periodic unique-node" not in output:
            raise ValueError(f"global compact path not selected: {directory}")
        if gpu and len(re.findall(r"COMPACT_SAME_STATE rank/kind/axis", output)) != 9 * int(np.prod(topology)):
            raise ValueError(f"compact same-state coverage differs: {directory}")
        if gpu and filtering:
            expected = (15 if workspace == "scalar" else 3) * int(np.prod(topology))
            if len(re.findall(r"COMPACT_FILTER_SAME_STATE rank/axis/component", output)) != expected:
                raise ValueError(f"compact filter same-state coverage differs: {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--cpu-exe", type=Path, required=True)
    parser.add_argument("--gpu-exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scheme", choices=("643e", "643c"), default="643e")
    parser.add_argument("--compact-transport", choices=("pinned", "device-aware"), default="pinned")
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = {"precision": args.precision, "grid_intervals": [32, 32, 32], "complete_steps": 10,
              "deltat_input": "1.d-3", "acceptance": "algorithm-validity; fp32 differences report-only",
              "primitive_validity_domain": "physical nodes and axial stencil halos; no inactive halo corners",
              "scheme": args.scheme, "compact_transport": args.compact_transport, "cases": {}}
    for topology in TOPOLOGIES:
        tag = "x".join(map(str, topology))
        for filtering in (False, True):
            name = f"{tag}_filter_{int(filtering)}"
            cpu, gpu = root / name / "cpu", root / name / "gpu"
            run_case(cpu, args.cpu_exe.resolve(), False, topology, filtering, "scalar", args.precision,
                     args.scheme, args.compact_transport)
            run_case(gpu, args.gpu_exe.resolve(), True, topology, filtering, "scalar", args.precision,
                     args.scheme, args.compact_transport)
            result["cases"][name] = compare_cases(cpu, gpu, args.precision, int(np.prod(topology)))
            if not filtering and args.scheme == "643e":
                result["cases"][name]["frozen_rhs"] = frozen_rhs_report(cpu, gpu, args.precision, int(np.prod(topology)))
            (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{args.precision} {name}: ten-step validity passed; differences recorded", flush=True)
    name = "2x1x1_filter_1"
    full = root / name / "gpu_full"
    run_case(full, args.gpu_exe.resolve(), True, (2, 1, 1), True, "full", args.precision,
             args.scheme, args.compact_transport)
    result["scalar_full"] = compare_cases(root / name / "gpu", full, args.precision, 2)
    result["status"] = "passed"
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.precision}: matrix and scalar/full representative passed", flush=True)


if __name__ == "__main__":
    main()
