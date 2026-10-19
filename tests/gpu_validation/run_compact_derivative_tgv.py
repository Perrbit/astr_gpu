#!/usr/bin/env python3
"""CP5 full-step CPU/GPU validation of the default globally coupled compact path."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from compare_q_validation_snapshots import read_q_snapshot, comparison_values
from compare_tgv_flowstate import read_flowstate

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 2e-10


def measure(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("field shape or finite-value check failed")
    axes = tuple(range(reference.ndim - 1))
    absolute = np.max(np.abs(candidate - reference), axis=axes)
    scaled = absolute / np.maximum(1.0, np.max(np.abs(reference), axis=axes))
    if np.max(scaled) > TOLERANCE:
        raise ValueError(f"per-component scaled field error: {scaled.tolist()}")
    return {"abs": absolute.tolist(), "scaled": scaled.tolist()}


def check_primitive_eos(qpath, primitive_path, mach=0.1):
    snapshot = read_q_snapshot(qpath)
    im, jm, km, hm, nq = snapshot.header
    shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1)
    with primitive_path.open("rb") as stream:
        header = tuple(np.fromfile(stream, dtype=np.int32, count=4))
        data = np.fromfile(stream, dtype=np.float64)
    if header != snapshot.header[:4] or data.size != 6 * np.prod(shape):
        raise ValueError("primitive snapshot header/payload mismatch")
    cached = data.reshape((*shape, 6), order="F")[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1, :]
    q = comparison_values(snapshot, True).reshape((im + 1, jm + 1, km + 1, nq), order="F")
    rho = q[..., 0]
    velocity = q[..., 1:4] / rho[..., None]
    gamma = 1.4
    pressure = (q[..., 4] - 0.5 * rho * np.sum(velocity * velocity, axis=-1)) / (1 / (gamma - 1))
    temperature = pressure / rho * (gamma * mach**2)
    if min(np.min(rho), np.min(pressure), np.min(temperature)) <= 0:
        raise ValueError("nonpositive density/pressure/temperature at RHS phase")
    expected = np.concatenate((rho[..., None], velocity, pressure[..., None], temperature[..., None]), axis=-1)
    return measure(expected, cached)


def assemble_final(case, grid, step):
    ownership = np.loadtxt(case / "datin/parallel.info", skiprows=3, dtype=int, ndmin=2)
    field = np.empty((*grid, 5))
    owners = np.zeros(grid, dtype=np.int32)
    for row in ownership:
        rank, _, _, _, im, jm, km, i0, j0, k0 = row
        suffix = f"state.post_update.step{step:08d}.rk03.rank{rank:08d}.bin"
        snapshot = read_q_snapshot(case / suffix)
        values = comparison_values(snapshot, True).reshape((im+1, jm+1, km+1, 5), order="F")
        region = (slice(i0, i0+im), slice(j0, j0+jm), slice(k0, k0+km))
        field[region] = values[:im, :jm, :km]
        owners[region] += 1
    if not np.all(owners == 1):
        raise ValueError("global periodic field does not have exactly one owner per node")
    return field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--grid", default="32,32,32")
    parser.add_argument("--topology", default="1,1,1")
    parser.add_argument("--scheme", default="643c", choices=("643c", "643e"))
    parser.add_argument("--check-only", action="store_true", help="reuse immutable completed runs")
    instrumentation = parser.add_mutually_exclusive_group()
    instrumentation.add_argument("--gpu-memcheck", dest="gpu_sanitizer", action="store_const", const="memcheck")
    instrumentation.add_argument("--gpu-sanitizer", choices=("memcheck", "initcheck", "racecheck", "synccheck"))
    parser.add_argument("--filter", choices=("f", "t"), default="f")
    parser.add_argument("--workspace", choices=("scalar", "full"), default="scalar")
    parser.add_argument("--compact-transport", choices=("pinned", "device-aware"), default="pinned")
    parser.add_argument("--compact-sync", choices=("explicit", "dependency"), default="explicit")
    parser.add_argument("--batch-rhs", type=int, default=65535)
    parser.add_argument("--baseline", type=Path, help="completed same-grid/filter run for global topology comparison")
    args = parser.parse_args()
    out = args.out.resolve()
    if not args.check_only:
        out.mkdir(parents=True, exist_ok=False)
    topology = tuple(map(int, args.topology.split(",")))
    ranks = int(np.prod(topology))
    steps = 10
    report = {"status": "running", "grid_intervals": args.grid, "topology": topology,
              "scheme": args.scheme, "filter": args.filter, "workspace": args.workspace,
              "full_steps": steps, "dt": 1e-3,
              "scope": "same globally coupled periodic compact operator" if args.scheme == "643c" else "explicit regression",
              "fields": {}, "cfl": {},
              "compact_transport": args.compact_transport}
    report.update(compact_sync=args.compact_sync,batch_rhs=args.batch_rhs)
    for backend, exe in (("cpu", args.cpu), ("gpu", args.gpu)):
        case = out / backend
        if not args.check_only:
            subprocess.run([sys.executable, str(ROOT / "tests/gpu_validation/prepare_tgv_case.py"),
                            "--src-case", str(ROOT / "examples/Taylor_Green_Vortex"), "--dst-case", str(case),
                            "--use-gpu", "t" if backend == "gpu" else "f", "--scheme", args.scheme,
                            "--grid", args.grid, "--maxstep", str(steps - 1), "--feqchkpt", "1",
                            "--feqlist", "1", "--lfilter", args.filter, "--diffterm", "t", "--deltat", "1.d-3"],
                           check=True)
            (case / "datin/input.output").write_text(
                "&output\n host_budget_bytes=67108864\n/\n&checkpoint\n/\n&volume\n/\n&slices\n/\n")
            env = dict(os.environ, ASTR_FORCE_MPI_TOPOLOGY=args.topology,
                       ASTR_GPU_COMPACT_COMPARE="1", ASTR_GPU_COMPACT_TRANSPORT=args.compact_transport,
                       ASTR_GPU_COMPACT_SYNC=args.compact_sync,ASTR_GPU_COMPACT_BATCH_RHS=str(args.batch_rhs),
                       ASTR_GPU_SYNC_MODE="explicit", ASTR_GPU_PRECISION_MODE="fp64",
                       ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_FILTER_WORKSPACE=args.workspace,
                       ASTR_VALIDATION_RHS_PREFIX=str(case / "state"),
                       ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_SECONDARY=str(steps - 1))
            env.pop("ASTR_GPU_COMPACT_DEVELOPMENT", None)
            env.update(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
                       OMPI_MCA_coll="^hcoll,ucc,cuda", OMPI_MCA_opal_cuda_support="0")
            if args.compact_transport == "device-aware" and backend == "gpu":
                env.update(OMPI_MCA_pml="ucx", UCX_MEMTYPE_CACHE="n", UCX_TLS="self,sm,cuda_copy",
                           UCX_CUDA_COPY_ENABLE_FABRIC="no", UCX_CUDA_COPY_DMABUF="no",
                           UCX_CUDA_IPC_ENABLE_MNNVL="no", OMPI_MCA_coll_hcoll_enable="0")
                env.pop("OMPI_MCA_opal_cuda_support", None)
            command = ["mpirun", "--oversubscribe", "--bind-to", "none", "-np", str(ranks),
                       str(exe.resolve()), "run", "datin/input.tgv"]
            if args.gpu_sanitizer and backend == "gpu":
                if args.compact_transport == 'pinned':
                    env.update(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
                               OMPI_MCA_coll="^hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
                report["sanitizer_mpi"] = {k: v for k, v in env.items() if k.startswith("OMPI_MCA_")}
                command = ["mpirun", "--oversubscribe", "--bind-to", "none", "-np", str(ranks),
                           "compute-sanitizer", "--tool", args.gpu_sanitizer,
                           "--error-exitcode", "99", "--log-file", str(case / (args.gpu_sanitizer + ".%p.log"))]
                if args.gpu_sanitizer == "memcheck":
                    command += ["--leak-check", "full"]
                if args.compact_transport == 'device-aware':
                    command += ['--suppressions',str(ROOT/'tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml')]
                command += [str(exe.resolve()), "run", "datin/input.tgv"]
            with (case / "run.log").open("w") as log:
                subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (case / "run.log").read_text(errors="replace")
        if "COMPUTATION CRASHED" in text or "ERROR STOP" in text:
            raise RuntimeError(f"{backend} failed; see {case / 'run.log'}")
        if args.scheme == "643c" and "COMPACT GLOBAL: periodic unique-node" not in text:
            raise ValueError(f"{backend} did not select the global compact operator")
        cfl = np.array([float(v) for v in re.findall(r"current CFL:\s*(\S+)", text)])
        if not len(cfl) or not np.isfinite(cfl).all() or cfl.max() > 1:
            raise ValueError(f"{backend} CFL check failed")
        report["cfl"][backend] = float(cfl.max())
        if backend == "gpu":
            times = [float(v) for v in re.findall(r"ASTR_CFL complete_step=\d+ state_time=\s*(\S+)", text)]
            if len(times) != steps or abs(times[-1] - steps * 1e-3) > 1e-15:
                raise ValueError("expected ten complete GPU RK steps to t=0.01")
            if args.gpu_sanitizer:
                logs = list(case.glob(args.gpu_sanitizer + "*.log"))
                clean = "ERROR SUMMARY: 0 errors"
                if args.gpu_sanitizer == "racecheck":
                    clean = "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                if len(logs) != ranks or any(clean not in log.read_text() for log in logs):
                    raise ValueError("missing clean per-rank full-step sanitizer summary")
                report["sanitizer"] = args.gpu_sanitizer
        if backend == "gpu" and args.scheme == "643c":
            checks = re.findall(r"COMPACT_SAME_STATE rank/kind/axis abs/scaled/residual:\s*"
                                r"(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)", text)
            if len(checks) != 9 * ranks:
                raise ValueError(f"expected {9*ranks} same-state checks, found {len(checks)}")
            report["same_state"] = [[int(v) for v in row[:3]] + [float(v) for v in row[3:]] for row in checks]
            if args.filter == "t":
                checks = re.findall(r"COMPACT_FILTER_SAME_STATE rank/axis/component abs/scaled/residual:\s*"
                                    r"(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)", text)
                count = (15 if args.workspace == "scalar" else 3) * ranks
                if len(checks) != count:
                    raise ValueError(f"expected {count} filter same-state checks, found {len(checks)}")
                report["filter_same_state"] = [[int(v) for v in row[:3]] + [float(v) for v in row[3:]] for row in checks]
        if args.scheme == "643c":
            eos = []
            for step in (0, steps - 1):
                for stage in (1, 2, 3):
                    for rank in range(ranks):
                        suffix = f".step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin"
                        eos.append(check_primitive_eos(case / ("state.pre_rhs" + suffix),
                                                       case / ("state.pre_rhs_primitives" + suffix)))
            report[backend + "_eos"] = eos
    report["phase_fields"] = {}
    for source in sorted((out / "cpu").glob("state.*.bin")):
        if not (source.name.startswith("state.pre_rhs.step") or source.name.startswith("state.post_update.step")):
            continue
        target = out / "gpu" / source.name
        reference, candidate = read_q_snapshot(source), read_q_snapshot(target)
        if reference.header != candidate.header:
            raise ValueError("phase snapshot headers differ")
        components = reference.header[-1]
        report["phase_fields"][source.name] = measure(
            comparison_values(reference, True).reshape((-1, components), order="F"),
            comparison_values(candidate, True).reshape((-1, components), order="F"))
    for rank in range(ranks):
        suffix = f"post_update.step{steps-1:08d}.rk03.rank{rank:08d}.bin"
        snapshots = [read_q_snapshot(out / b / ("state." + suffix)) for b in ("cpu", "gpu")]
        if snapshots[0].header != snapshots[1].header:
            raise ValueError("CPU/GPU snapshot headers differ")
        im, jm, km, _, nq = snapshots[0].header
        shape = (im + 1, jm + 1, km + 1, nq)
        fields = [comparison_values(s, True).reshape(shape, order="F") for s in snapshots]
        report["fields"][str(rank)] = measure(*fields)
        for q in fields:
            density = q[..., 0]
            internal = q[..., 4] - np.sum(q[..., 1:4] ** 2, axis=-1) / (2 * density)
            if np.min(density) <= 0 or np.min(internal) <= 0:
                raise ValueError("nonpositive density/internal energy")
    tables = []
    for backend in ("cpu", "gpu"):
        files = list((out / backend).rglob("flowstate.dat"))
        if len(files) != 1:
            raise ValueError(f"expected one {backend} flowstate table")
        _, rows = read_flowstate(files[0])
        tables.append(np.array(rows))
    if not np.array_equal(tables[0][:, 0], tables[1][:, 0]) or len(tables[0]) != steps - 1:
        raise ValueError("statistics sample steps/times differ")
    # CPU listwrite has fewer decimal digits than the GPU diagnostic writer.
    if not np.allclose(tables[0][:, 1], tables[1][:, 1], rtol=1e-12, atol=0.0):
        raise ValueError("statistics sample times differ beyond text precision")
    report["statistics_time_abs"] = float(np.max(np.abs(tables[0][:, 1] - tables[1][:, 1])))
    report["statistics"] = measure(tables[0][:, 2:], tables[1][:, 2:])
    if args.baseline:
        baseline = json.loads((args.baseline / "summary.json").read_text())
        for key in ("grid_intervals", "scheme", "filter", "full_steps", "dt", "scope"):
            if baseline[key] != report[key]:
                raise ValueError(f"topology baseline differs in {key}")
        grid = tuple(map(int, args.grid.split(",")))
        report["topology_fields"] = {}
        for backend in ("cpu", "gpu"):
            report["topology_fields"][backend] = measure(
                assemble_final(args.baseline / backend, grid, steps-1),
                assemble_final(out / backend, grid, steps-1))
    report["status"] = "pass"
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
