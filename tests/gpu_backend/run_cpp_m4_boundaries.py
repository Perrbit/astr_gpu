"""Four-step FP64 M4 checks on a profile plate and a static wavy mesh."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from run_cpp_multi_tgv_gate import layouts, physical, raw_field
from run_cpp_tgv_gate import compare, logged_steps, namelist


def set_field(text, name, value):
    updated, count = re.subn(rf"(?m)^(\s*{name}\s*=)[^\n]*", rf"\g<1> {value}", text)
    if count != 1:
        raise ValueError(f"expected one {name} in the supplied fixture")
    return updated


def run(binary, mpi, case, template, topology, device):
    shutil.copytree(template, case / "datin")
    (case / "outdat/new").mkdir(parents=True)
    config = case / "datin/input.case"
    text = config.read_text()
    for name, value in (("use_gpu", ".true." if device else ".false."), ("conschm", "'643e'"),
                        ("viscous_form", "'standard'"), ("lfilter", ".false.")):
        text = set_field(text, name, value)
    config.write_text(text)
    (case / "datin/input.run").write_text(namelist(dict(run_control=dict(max_steps=4),
        timestep=dict(mode="fixed", fixed_dt=1e-5), logging=dict(list_steps=1),
        repair=dict(crashfix=False), statistics=dict(enabled=False))))
    (case / "datin/input.output").write_text(namelist(dict(output=dict(directory="outdat/new",
        exact_restart=False, host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
        checkpoint=dict(enabled=False), volume=dict(enabled=False), slices=dict(enabled=False))))
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
        OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0", ASTR_GPU_HALO_TRANSPORT="pinned",
        ASTR_GPU_SYNC_MODE="dependency", ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)),
        ASTR_VALIDATION_RHS_PREFIX="outdat/rhs", ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_END="4")
    with (case / "run.log").open("w") as log:
        subprocess.run([str(mpi), "--bind-to", "none", "-n", "2", str(binary), "run"], cwd=case, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def rhs(path):
    with path.open("rb") as stream:
        nx, ny, nz, nc = np.fromfile(stream, np.int32, 4)
        return np.fromfile(stream, np.float64).reshape((nx + 1, ny + 1, nz + 1, nc), order="F")


def compare_case(candidate, reference, periodic):
    if "ASTR_GPU_CPP viscous_overlap_axis=xyz" not in (candidate / "run.log").read_text():
        raise AssertionError("not the M4 RK path")
    compare(logged_steps(candidate), logged_steps(reference))
    layout = layouts(candidate)
    errors = {}
    for label in ("pre_rhs", "pre_rhs_primitives", "post_source", "post_update"):
        files = sorted((candidate / "outdat").glob(f"rhs.{label}.*.bin"))
        if len(files) < 4 * 3 * len(layout):
            raise AssertionError(f"incomplete {label} phase evidence")
        maximum = 0.
        for path in files:
            oracle = reference / "outdat" / path.name
            rank = int(path.stem[-8:])
            if label == "post_source":
                actual, expected = rhs(path), rhs(oracle)
            else:
                primitive = label == "pre_rhs_primitives"
                actual = physical(raw_field(path, primitive=primitive)[1], layout[rank])
                expected = physical(raw_field(oracle, primitive=primitive)[1], layout[rank])
            maximum = max(maximum, max(compare(actual, expected)))
        errors[label] = maximum

    faces = 0
    for label in ("pre_rhs", "pre_rhs_primitives"):
        for path in sorted((candidate / "outdat").glob(f"rhs.{label}.step00000000.*.rank00000000.bin")):
            prefix = path.name.split(".rank")[0]
            fields = {r: raw_field(path.parent / f"{prefix}.rank{r:08d}.bin",
                                   primitive=label == "pre_rhs_primitives")[1] for r in layout}
            for rank, local in layout.items():
                h, = local["halo"]
                for axis in range(3):
                    transverse = tuple(slice(None) if i == axis else slice(h, h + n + 1)
                                       for i, n in enumerate(local["upper"])) + (slice(None),)
                    offsets = np.arange(1, h + 1)
                    for high in (False, True):
                        peer = local["higher" if high else "lower"][axis]
                        if local["topology"][axis] == 1:
                            if not periodic[axis]:
                                continue
                            peer = rank
                        if peer < 0:
                            continue
                        targets = h + local["upper"][axis] + offsets if high else h - offsets
                        sources = h + offsets if high else h + layout[peer]["upper"][axis] - offsets
                        actual = np.take(fields[rank][transverse], targets, axis=axis)
                        expected = np.take(fields[peer][transverse], sources, axis=axis)
                        if not np.array_equal(actual, expected):
                            raise AssertionError((prefix, rank, axis, high, "same-stage face mismatch"))
                        faces += 1
    return dict(phase_max_scaled=errors, exact_q_and_primitive_face_checks=faces)


def main():
    parser = argparse.ArgumentParser()
    for arg in ("binary", "mpiexec", "plate-datin", "curve-datin", "output"):
        parser.add_argument("--" + arg, required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, template, topology, periodic in (
        ("plate", args.plate_datin, (2, 1, 1), (False, False, True)),
        ("static_curve", args.curve_datin, (1, 2, 1), (True, False, True)),
    ):
        current = args.output / name
        for backend in ("cpu", "cpp"):
            run(args.binary, args.mpiexec, current / backend, template, topology, backend == "cpp")
        result = compare_case(current / "cpp", current / "cpu", periodic)
        (current / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, result, flush=True)


if __name__ == "__main__":
    main()
