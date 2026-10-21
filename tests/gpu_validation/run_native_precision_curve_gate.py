#!/usr/bin/env python3
"""FP5 bounded CURVE checks; ASTR alone advances the flow and computes gradients."""

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
from check_curvilinear_metrics import analytic_metrics, metric_identity_residual, read_astr_metrics
from generate_curvilinear_tgv_grid import mapped_grid
from run_native_precision_tgv_gate import compare_cases, component_report, physical_state,primitive_state


def read_gradients(path: Path, precision: str) -> tuple[np.ndarray, dict]:
    dtype = np.dtype("float32" if precision == "fp32" else "float64")
    with path.open("rb") as stream:
        header = np.fromfile(stream, np.int32, 5)
        data = np.fromfile(stream, dtype)
    if header.size != 5 or np.any(header[:3] < 0) or not np.isin(header[3:], [0, 1]).all():
        raise ValueError(f"invalid gradient header: {path}")
    im, jm, km, lo, hi = map(int, header)
    cells = (im + 1) * (jm + 1) * (km + 1)
    plane = (im + 1) * (km + 1) * 3
    if data.size != cells * 12 + plane * (lo + hi) or not np.isfinite(data).all():
        raise ValueError(f"invalid gradient payload: {path}")
    values = data[:cells * 12].reshape((im + 1, jm + 1, km + 1, 12), order="F")
    normals, offset = {}, cells * 12
    for face, enabled in ((3, lo), (4, hi)):
        if enabled:
            normals[face] = data[offset:offset + plane].reshape((im + 1, km + 1, 3), order="F")
            offset += plane
    return values, normals


def geometry_reference(mapping: str) -> tuple[np.ndarray, np.ndarray]:
    if mapping == "periodic":
        return analytic_metrics((32, 32, 32), 0.15)
    xi, _, zeta = np.meshgrid(*([np.linspace(0, 2 * np.pi, 33)] * 3), indexing="ij")
    derivative = np.broadcast_to(np.eye(3), xi.shape + (3, 3)).copy()
    derivative[..., 1, 0] = 0.15 * np.cos(xi) * np.sin(zeta)
    derivative[..., 1, 2] = 0.15 * np.sin(xi) * np.cos(zeta)
    derivative *= 2 * np.pi / 32
    return np.linalg.det(derivative), np.linalg.inv(derivative)


def geometry_report(case: Path, precision: str, mapping: str) -> dict:
    dtype = np.dtype("float32" if precision == "fp32" else "float64")
    with h5py.File(case / "geometry.h5") as geometry:
        if any(geometry[name].dtype != dtype for name in geometry):
            raise ValueError("geometry is not stored in the native kind")
    jacobian, inverse = read_astr_metrics(case / "geometry.h5")
    if not np.isfinite(jacobian).all() or not np.isfinite(inverse).all() or np.min(jacobian) <= 0:
        raise ValueError("invalid CURVE geometry")
    exact_j, exact_inv = geometry_reference(mapping)
    identity = float(np.max(np.abs(metric_identity_residual(jacobian, inverse))))
    if precision == "fp64" and identity > 1e-10:
        raise ValueError(f"FP64 metric identity regression: {identity}")
    return {"minimum_jacobian": float(np.min(jacobian)), "identity_max_abs": identity,
        "analytic_jacobian_max_abs": float(np.max(np.abs(jacobian - exact_j))),
        "analytic_inverse_max_abs": float(np.max(np.abs(inverse - exact_inv))),
        "reference": "analytic mapping; geometric truncation errors are separate from CPU/GPU differences"}


def run_case(case: Path, executable: Path, gpu: bool, topology: tuple, precision: str,
             mapping: str, uniform: bool, gradients: bool = True, sanitizer: str | None = None) -> None:
    command = [sys.executable, str(ROOT / "tests/gpu_validation/prepare_tgv_case.py"),
        "--src-case", str(ROOT / "examples/Taylor_Green_Vortex"), "--dst-case", str(case),
        "--use-gpu", "t" if gpu else "f", "--grid", "32,32,32", "--maxstep", "3",
        "--feqchkpt", "1", "--feqlist", "1", "--lfilter", "t", "--diffterm", "t",
        "--lreadgrid", "t", "--gridfile", "datin/grid.curvilinear.h5", "--scheme", "643e",
        "--ninit", "3" if uniform else "0", "--restart", "f", "--deltat", "2.d-5"]
    if mapping == "y-wavy":
        command += ["--homogeneous", "t,f,t", "--bctype", "1;1;41,273.15d0;41,273.15d0;1;1"]
    subprocess.run(command, check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "tests/gpu_validation/generate_curvilinear_tgv_grid.py"),
        "--output", str(case / "datin/grid.curvilinear.h5"), "--report", str(case / "grid_quality.txt"),
        "--grid", "32,32,32", "--amplitude", "0.15", "--mapping", mapping], check=True)
    if uniform:
        wall = mapping == "y-wavy"
        subprocess.run([sys.executable, str(ROOT / "tests/gpu_validation/generate_uniform_flow_field.py"),
            "--output", str(case / "datin/flowini3d.h5"), "--grid", "32,32,32",
            "--density", str(1 / 273.15 if wall else 1), "--temperature", str(273.15 if wall else 1),
            "--u1", "0" if wall else "0.7", "--u2", "0" if wall else "-0.2",
            "--u3", "0" if wall else "0.1"], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ASTR_", "OMPI_MCA_", "UCX_"))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)), ASTR_GPU_SYNC_MODE="explicit",
        ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_FILTER_WORKSPACE="scalar",
        ASTR_GEOMETRY_DUMP="geometry.h5", ASTR_VALIDATION_RHS_PREFIX="state",
        ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_SECONDARY="3",
        OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
        OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
    if gradients:
        env["ASTR_VALIDATION_GRADIENTS"] = "1"
    command = ["mpirun", "--bind-to", "none", "-np", str(np.prod(topology))]
    if sanitizer:
        command += ["compute-sanitizer", "--tool", sanitizer, "--leak-check", "full", "--error-exitcode", "91"]
    with (case / "run.log").open("w") as log:
        subprocess.run([*command, str(executable.resolve()), "run", "datin/input.tgv"], cwd=case, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    output = (case / "run.log").read_text()
    if [int(n) for n in re.findall(r"ASTR_CFL complete_step=(\d+)", output)] != list(range(4)):
        raise ValueError("not four complete CURVE steps")
    cfl = [float(row.split()[-1]) for row in re.findall(r"ASTR_CFL directional/local_sum/upper_bound=([^\n]+)", output)]
    if len(cfl) != 4 or not np.isfinite(cfl).all() or max(cfl) > 0.5:
        raise ValueError(f"CURVE fixture CFL is invalid: {cfl}")
    if gpu and f"ASTR_GPU_PRECISION_MODE={precision}" not in output:
        raise ValueError("CURVE executable precision differs")
    if sanitizer and output.count("ERROR SUMMARY: 0 errors") != int(np.prod(topology)):
        raise ValueError("missing clean per-rank CURVE sanitizer summaries")


def compare(cpu: Path, gpu: Path, precision: str, mapping: str, uniform: bool) -> dict:
    partitions = np.loadtxt(cpu / "datin/parallel.info", skiprows=3, dtype=int, ndmin=2)
    sides = {}
    for row in partitions:
        rank, _, _, _, im, jm, km, i0, j0, k0 = map(int, row)
        sides[rank] = ((True, True), (mapping != "y-wavy" or j0 > 0,
                         mapping != "y-wavy" or j0 + jm < 32), (True, True))
    result = compare_cases(cpu, gpu, precision, len(partitions), steps=4, halo_sides_by_rank=sides)
    result["geometry"] = {name: geometry_report(case, precision, mapping) for name, case in (("cpu", cpu), ("gpu", gpu))}
    _, inverse = read_astr_metrics(gpu / "geometry.h5")
    gradients = {}
    bits = np.uint32 if precision == "fp32" else np.uint64
    for row in partitions:
        rank, _, _, _, im, jm, km, i0, j0, k0 = map(int, row)
        for step in (0, 3):
            for stage in (1, 2, 3):
                suffix = f"step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin"
                a, cpu_normals = read_gradients(cpu / ("state.gradients." + suffix), precision)
                b, gpu_normals = read_gradients(gpu / ("state.gradients." + suffix), precision)
                ref, ref_normals = read_gradients(gpu / ("state.gradient_reference." + suffix), precision)
                rows = component_report(a, b)
                frozen = component_report(ref, b)
                if precision == "fp64" and max(r["max_scaled"] for r in rows + frozen) > 2e-10:
                    raise ValueError("FP64 CURVE gradient regression")
                if cpu_normals.keys() != gpu_normals.keys() or ref_normals.keys() != gpu_normals.keys():
                    raise ValueError("CURVE normal ownership differs")
                normals = {}
                for face, normal in gpu_normals.items():
                    if not np.array_equal(normal.view(bits), ref_normals[face].view(bits)):
                        raise ValueError("device normal copy changed source bits")
                    j = 0 if face == 3 else 32
                    vector = inverse[i0:i0 + im + 1, j, k0:k0 + km + 1, 1, :]
                    expected = vector / np.linalg.norm(vector, axis=-1, keepdims=True) * (1 if face == 3 else -1)
                    projection = np.sum(normal.astype(np.float64) * expected, axis=-1)
                    if np.any(projection <= 0):
                        raise ValueError("CURVE wall normal is not inward")
                    normals[str(face)] = {"unit_error": float(np.max(np.abs(np.linalg.norm(normal.astype(np.float64), axis=-1) - 1))),
                        "metric_direction_error": float(np.max(np.abs(normal.astype(np.float64) - expected)))}
                    if precision == "fp64" and max(normals[str(face)].values()) > 1e-12:
                        raise ValueError("FP64 unit/metric wall-normal inconsistency")
                walls = {}
                for backend, case in (("cpu", cpu), ("gpu", gpu)):
                    primitive = primitive_state(case / ("state.pre_rhs_primitives." + suffix), precision,
                                                halo_sides=sides[rank])
                    for face in gpu_normals:
                        plane = primitive[:, 0 if face == 3 else -1, :, :]
                        velocity = float(np.max(np.abs(plane[..., 1:4])))
                        target = np.dtype(primitive.dtype).type(273.15)
                        temperature = float(np.max(np.abs(plane[..., 5].astype(np.float64) - float(target))))
                        if velocity != 0:
                            raise ValueError("stationary no-slip wall velocity is not zero")
                        if precision == "fp64" and temperature > 1e-12:
                            raise ValueError("FP64 prescribed wall temperature differs")
                        walls[f"{backend}_{face}"] = {"velocity_max_abs": velocity, "temperature_max_abs": temperature}
                gradients[suffix] = {"cpu_gpu": rows, "frozen_native_cpu_gpu": frozen, "normals": normals,
                                     "wall_state": walls}
    result["gradients"] = gradients
    if uniform:
        result["uniform_drift"] = {}
        for name, case in (("cpu", cpu), ("gpu", gpu)):
            errors = []
            for rank in range(len(partitions)):
                initial = physical_state(case / f"state.initialized.step00000000.rk01.rank{rank:08d}.bin", precision)
                final = physical_state(case / f"state.post_update.step00000003.rk03.rank{rank:08d}.bin", precision)
                errors += component_report(initial, final)
            if precision == "fp64" and max(r["max_scaled"] for r in errors) > 1e-10:
                raise ValueError("FP64 compatible constant-state drift")
            result["uniform_drift"][name] = errors
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "fp64"), required=True)
    parser.add_argument("--cpu-exe", type=Path, required=True)
    parser.add_argument("--gpu-exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"precision": args.precision, "steps": 4, "deltat_input": "2.d-5", "cases": {}}
    for mapping in ("periodic", "y-wavy"):
        split = (2, 1, 1) if mapping == "periodic" else (1, 2, 1)
        for uniform, topology in ((False, (1, 1, 1)), (False, split), (True, split)):
            tag = mapping + ("_constant_" if uniform else "_tgv_") + "x".join(map(str, topology))
            for backend, executable in (("cpu", args.cpu_exe), ("gpu", args.gpu_exe)):
                run_case(out / tag / backend, executable, backend == "gpu", topology,
                         args.precision, mapping, uniform)
            report["cases"][tag] = compare(out / tag / "cpu", out / tag / "gpu", args.precision, mapping, uniform)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(args.precision, tag, "four-step geometry/gradient/state checks passed", flush=True)
    report["status"] = "passed"
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
