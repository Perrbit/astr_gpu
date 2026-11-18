"""Existing compact32 inherited statistics: NP1 -> NP2 x -> NP2 z and exact tail."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import checkpoint, compare, logged_steps, namelist
from run_cpp_legacy_statistics_gate import compact_statistics
from run_cpp_insitu_gate import compare_hdf
from prepare_legacy_statistics_reference import sampling_isolation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "reference", "mpiexec", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--transport", choices=("pinned", "device"), default="pinned")
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary, reference = args.binary.resolve(), args.reference.resolve()
    shutil.copyfile(__file__, root / "runner.py")
    shutil.copyfile(binary, root / "tested_astr")
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
        OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0", ASTR_GPU_SYNC_MODE="dependency",
        ASTR_GPU_HALO_TRANSPORT=args.transport, ASTR_GPU_FILTER_WORKSPACE="scalar", ASTR_GPU_MP_POSITIVITY="off")
    if args.transport == "device":
        env.update(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    report = dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), precision=args.precision,cases={})
    with h5py.File(reference/'old_gpu_fixed/outdat/new/checkpoints/step000000000006/state.h5') as source:
        if source['q0001'].dtype.itemsize!=(4 if args.precision=='fp32' else 8):
            parser.error('actual repartition reference numerical kind mismatch')
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if args.precision=='fp32' else 2e-10)

    def run(name, topology, steps, restore=None, sampling=True):
        case = root / name
        case.mkdir()
        shutil.copytree(reference / "old_gpu_fixed/datin", case / "datin", copy_function=shutil.copyfile)
        (case / "outdat/new").mkdir(parents=True)
        (case / "datin/input.run").write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode="fixed", fixed_dt=1e-6), logging=dict(list_steps=1), repair=dict(crashfix=False),
            statistics=dict(enabled=sampling, sample_clock="steps", sample_steps=1, write_clock="steps", write_steps=0))))
        # Existing OR4 stores full batches. Across different topologies its
        # established reader restores physical data and rebuilds halos; only
        # the subsequent same-topology tail is asserted bitwise exact.
        (case / "datin/input.output").write_text(namelist(dict(output=dict(directory="outdat/new",
            restore_directory=str(restore or ""), exact_restart=True, checkpoint_statistics=True,
            host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
            checkpoint=dict(enabled=True, interval_steps=1, keep=2, initial_frame=False),
            volume=dict(enabled=False), slices=dict(enabled=False))))
        command = [str(args.mpiexec.resolve()), "--bind-to", "none", "-n", str(np.prod(topology)), str(binary), "run"]
        with (case / "run.log").open("w") as log:
            subprocess.run(command, cwd=case, env=env | dict(ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology))),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        q = checkpoint(batch(case, steps) / "state.h5")
        p = .4*(q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0])
        if not np.isfinite(q).all() or not (q[..., 0] > 0).all() or not (p > 0).all():
            raise AssertionError("invalid repartitioned physical field")
        return case

    def batch(case, step):
        return case / f"outdat/new/checkpoints/step{step:012d}"

    def compare_history(current, previous, step):
        cm, c = compact_statistics(batch(current, step) / "statistics.h5")
        pm, p = compact_statistics(batch(previous, step) / "statistics.h5")
        if not np.array_equal(cm[:4], pm[:4]) or not np.array_equal(cm[7:], pm[7:]):
            raise AssertionError("repartition reset or changed compact window identity")
        if cm[7] != step+1 or cm[8] != 0 or cm[9] != step or not np.any(c[:, :, 1:13]):
            raise AssertionError("missing nonzero accumulated sample history")
        return compare_data(c, p)

    continuous = run("continuous_z", (1, 1, 2), 6)
    report["cases"]["continuous"] = dict(references={})
    for backend in ("cpu", "old_gpu"):
        previous = reference / (backend + "_fixed")
        report["cases"]["continuous"]["references"][backend] = compare_data(checkpoint(batch(continuous, 6) / "state.h5"),
            checkpoint(batch(previous, 6) / "state.h5"))
    report["cases"]["continuous"]["statistics"] = compare_history(continuous, reference / "old_gpu_fixed", 6)
    seed = run("seed_np1", (1, 1, 1), 2)
    middle = run("middle_x", (2, 1, 1), 4, batch(seed, 2))
    target = run("target_z", (1, 1, 2), 6, batch(middle, 4))
    for label, case, step, previous in (("middle_x", middle, 4, reference / "old_gpu_fixed"),
                                      ("target_z", target, 6, continuous)):
        if "ASTR_OUTPUT_REPARTITION bl" not in (case / "run.log").read_text():
            raise AssertionError("repartition path not entered")
        with h5py.File(batch(case, step) / "statistics.h5") as statistics:
            inherited_mass = statistics["inherited/q0002"][...]
            if not np.isfinite(inherited_mass).all() or not np.any(inherited_mass):
                raise AssertionError("repartition did not inherit nonzero past mass integral")
        result = dict(state=compare_data(checkpoint(batch(case, step) / "state.h5"), checkpoint(batch(previous, step) / "state.h5")),
                      statistics=compare_history(case, previous, step))
        report["cases"][label] = result
    joined = np.concatenate((logged_steps(seed, 2), logged_steps(middle, 2, 3), logged_steps(target, 2, 5)))
    if not np.array_equal(joined, logged_steps(continuous, 6)):
        raise AssertionError("multi-hop repartition changed the completed-step clock")
    restored = run("same_z_exact", (1, 1, 2), 6, batch(target, 5))
    for file in ("state.h5", "statistics.h5"):
        compare_hdf(batch(target, 6) / file, batch(restored, 6) / file, exact=True)
    for file in ("control.bin", "archives.bin", "insitu_control.bin"):
        if (batch(target, 6) / file).read_bytes() != (batch(restored, 6) / file).read_bytes():
            raise AssertionError((file, "inherited-history same-topology exact controller mismatch"))
    quiet = run("sampling_off_z", (1, 1, 2), 6, sampling=False)
    report["sampling_isolation"] = sampling_isolation(batch(continuous, 6) / "state.h5", batch(quiet, 6) / "state.h5")
    report.update(same_z_after_inheritance_exact=True, multi_hop_clock_exact=True, transport=args.transport)
    (root / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
