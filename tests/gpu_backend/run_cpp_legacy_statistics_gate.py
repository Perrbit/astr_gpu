"""Actual C++ plate mean windows, compact-statistic output and exact history restore."""
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
from run_cpp_insitu_gate import compare_hdf
from run_cpp_physical_gate import stages
from prepare_legacy_statistics_reference import sampling_isolation
from run_cpp_native_tgv import stage_report as native_stages
from run_cpp_multi_tgv_gate import layouts


def compact_statistics(path):
    with h5py.File(path) as source:
        metadata = source["metadata"][...].reshape(-1)
        if len(metadata) != 14 or metadata[0] not in (3, 4) or metadata[11] != 32:
            raise AssertionError("compact statistics protocol")
        arrays = [source[f"q{n:04d}"][...].transpose(2, 1, 0) for n in range(1, 33)]
        if any(not np.isfinite(a).all() or np.any(a[:, :, -1] != 0) for a in arrays):
            raise AssertionError("compact payload nonfinite/terminal partition padding")
        canonical = np.stack([a[:, :, :-1].sum(axis=2) for a in arrays], axis=-1)
        if metadata[0] == 4:
            inherited = source["inherited"]
            if not np.array_equal(inherited["metadata"][...], source["metadata"][...]):
                raise AssertionError("compact inherited clock differs")
            base = np.stack([inherited[f"q{n:04d}"][...].transpose(2, 1, 0) for n in range(1, 33)], axis=-1)
            if not np.isfinite(base).all() or np.any(base[:, :, 1, :] != 0):
                raise AssertionError("nonfinite inherited history / terminal padding")
            # Measures and current endpoints are not cumulative historical sums.
            canonical[:, :, 1:13] += base[:, :, 0, 1:13]
            canonical[:, :, 14:17] += base[:, :, 0, 14:17]
        return metadata, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "reference", "mpiexec", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--mode", choices=("fixed", "automatic", "all"), default="all")
    parser.add_argument("--transport", choices=("pinned", "device"), default="pinned")
    parser.add_argument("--sync", choices=("explicit", "dependency"), default="dependency")
    parser.add_argument("--sanitizer", choices=("memcheck", "initcheck"))
    parser.add_argument("--old-gpu-migration", action="store_true")
    parser.add_argument("--sample-clock", choices=("steps", "time"), default="steps")
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args = parser.parse_args()
    if args.old_gpu_migration and args.ranks != 1:
        parser.error("the frozen migration source is NP1; use the same topology for backend migration")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reference = args.reference.resolve()
    for mode in ('fixed','automatic') if args.mode=='all' else (args.mode,):
        path=reference/f'old_gpu_{mode}'/'outdat/new/checkpoints/step000000000004/state.h5'
        with h5py.File(path) as source:
            if source['q0001'].dtype.itemsize!=(4 if args.precision=='fp32' else 8):
                parser.error('actual mean reference numerical kind mismatch')
    binary = args.binary.resolve()
    shutil.copyfile(__file__, root / "runner.py")
    shutil.copyfile(binary, root / "tested_astr")
    final = Path("outdat/new/checkpoints/step000000000004")
    topology = [1, 1, 1]
    topology["xyz".index(args.axis)] = args.ranks
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
        OMPI_MCA_osc="pt2pt", OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
        ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)), ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_COMPACT_TRANSPORT="device-aware" if args.transport == "device" else "pinned",
        ASTR_GPU_FILTER_WORKSPACE="scalar", ASTR_GPU_SYNC_MODE=args.sync, ASTR_GPU_MP_POSITIVITY="off",
        ASTR_VALIDATION_RHS_PREFIX="outdat/rhs", ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_END="4")
    if args.transport == "device":
        env.update(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    report = dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  topology=topology, precision=args.precision,transport=args.transport, sync=args.sync, sample_clock=args.sample_clock, cases={})
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if args.precision=='fp32' else 2e-10)

    def run(name, mode, steps=4, restore=None, sampling=True, intermediate=True, exact=True):
        case = root / name
        case.mkdir()
        source = reference / f"old_gpu_{mode}"
        shutil.copytree(source / "datin", case / "datin", copy_function=shutil.copyfile)
        (case / "outdat/new").mkdir(parents=True)
        timestep = (dict(mode="auto", cfl_target=.5, initial_dt=1e-6, max_dt=2e-6, growth_factor=1.1)
                    if mode == "automatic" else dict(mode="fixed", fixed_dt=1e-6))
        if restore and not exact and mode == "automatic":
            # Non-exact restart intentionally uses the current input controller.
            # Seed its new initial choice from the frozen source's next dt for
            # this matched-clock backend comparison, not as a production policy.
            with h5py.File(restore / "state.h5") as source_state:
                width=int(source_state['numerical_kind_bytes'][0])
                words=source_state["identity"][...][11:12]
                timestep["initial_dt"] = float(words.astype(np.int32).view(np.float32)[0] if width==4 else words.view(np.float64)[0])
        (case / "datin/input.run").write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=timestep, logging=dict(list_steps=1), repair=dict(crashfix=False),
            statistics=dict(enabled=sampling, sample_clock=args.sample_clock, sample_steps=1, sample_time=1.8e-6,
                            write_clock=args.sample_clock, write_steps=2 if intermediate else 0,
                            write_time=2e-6 if intermediate else 0))))
        (case / "datin/input.output").write_text(namelist(dict(output=dict(directory="outdat/new",
            restore_directory=str(restore or ""), exact_restart=exact, checkpoint_statistics=True,
            host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
            checkpoint=dict(enabled=True, mode="steps", interval_steps=2 if intermediate else steps,
                            keep=2, initial_frame=False), volume=dict(enabled=False), slices=dict(enabled=False))))
        command = [str(args.mpiexec.resolve()), "--bind-to", "none", "-n", str(args.ranks)]
        if args.sanitizer and name.endswith("continuous"):
            command += ["/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer",
                "--tool", args.sanitizer, "--error-exitcode", "99", "--log-file", str(case / "sanitizer.%p.log")]
            if args.sanitizer == "memcheck":
                command += ["--leak-check", "full"]
        with (case / "run.log").open("w") as log:
            subprocess.run(command + [str(binary), "run"], cwd=case, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if (case / "run.log").read_text().count("backend=cuda") != args.ranks:
            raise AssertionError("actual C++ main not selected")
        if args.sanitizer and name.endswith("continuous"):
            logs = list(case.glob("sanitizer.*.log"))
            if len(logs) != args.ranks or any("ERROR SUMMARY: 0 errors" not in p.read_text() for p in logs):
                raise AssertionError("missing clean per-rank main sanitizer record")
            if args.sanitizer == "memcheck" and any("LEAK SUMMARY: 0 bytes leaked" not in p.read_text() for p in logs):
                raise AssertionError("main compact statistic device allocation leaked")
        q = checkpoint(case / f"outdat/new/checkpoints/step{steps:012d}/state.h5")
        p = .4 * (q[..., 4] - .5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0])
        if not np.isfinite(q).all() or not (q[..., 0] > 0).all() or not (p > 0).all():
            raise AssertionError("nonfinite/nonpositive actual plate state")
        return case

    for mode in ("fixed", "automatic") if args.mode == "all" else (args.mode,):
        continuous = run(mode + "_continuous", mode)
        actual_metadata, actual = compact_statistics(continuous / final / "statistics.h5")
        original = reference / f"old_gpu_{mode}"
        expected_metadata, expected = compact_statistics(original / final / "statistics.h5")
        # compact_statistics already validates and canonicalizes both v3/v4.
        # Their topology-dependent inherited layout is not a sampling clock.
        if not np.array_equal(actual_metadata[1:4], expected_metadata[1:4]) or not np.array_equal(actual_metadata[7:], expected_metadata[7:]):
            raise AssertionError("old GPU / C++ compact window clocks differ")
        result = dict(compact_scaled=compare_data(actual, expected), references={},
            storage_versions=dict(actual=int(actual_metadata[0]),reference=int(expected_metadata[0])))
        for backend in ("cpu", "old_gpu"):
            previous = reference / f"{backend}_{mode}"
            phase=(native_stages(continuous,previous,layouts(continuous)) if args.precision=='fp32' else stages(continuous,previous))
            result["references"][backend] = dict(state=compare_data(checkpoint(continuous / final / "state.h5"),
                checkpoint(previous / final / "state.h5")), stages=phase,
                clocks=compare(logged_steps(continuous), logged_steps(previous)))
        quiet = run(mode + "_sampling_off", mode, sampling=False)
        result["sampling_isolation"] = sampling_isolation(continuous / final / "state.h5", quiet / final / "state.h5")
        if not np.array_equal(logged_steps(continuous), logged_steps(quiet)):
            raise AssertionError("sampling changed actual step clock")
        no_intermediate = run(mode + "_no_intermediate", mode, intermediate=False)
        for file in ("state.h5", "statistics.h5"):
            compare_hdf(continuous / final / file, no_intermediate / final / file, exact=True)
        prefix = run(mode + "_prefix", mode, steps=2)
        tail = run(mode + "_tail", mode, restore=prefix / "outdat/new/checkpoints/step000000000002")
        for file in ("state.h5", "statistics.h5"):
            compare_hdf(continuous / final / file, tail / final / file, exact=True)
        for control in ('control.bin','insitu_control.bin'):
            if (continuous / final / control).read_bytes() != (tail / final / control).read_bytes():
                raise AssertionError("exact actual C++ mean control restart "+control)
        full = np.loadtxt(continuous / "flowstate.dat", skiprows=1, ndmin=2)
        resumed = np.loadtxt(tail / "flowstate.dat", skiprows=1, ndmin=2)
        if not np.array_equal(resumed, full[np.isin(full[:, 0], resumed[:, 0])]):
            raise AssertionError("exact actual C++ mean scalar sequence")
        result.update(exact_state_statistics_control=True, exact_scalars=True, output_event_isolation=True)
        if args.old_gpu_migration:
            migrated = run(mode + "_old_gpu_migration", mode,
                restore=reference / f"old_gpu_{mode}_prefix/outdat/new/checkpoints/step000000000002", exact=False)
            restored_metadata, restored = compact_statistics(migrated / final / "statistics.h5")
            if not np.array_equal(restored_metadata, actual_metadata):
                raise AssertionError("old-GPU migration changed compact history clocks")
            result["old_gpu_nonexact_migration"] = dict(
                compact_scaled=compare_data(restored, actual),
                state=compare_data(checkpoint(migrated / final / "state.h5"), checkpoint(continuous / final / "state.h5")),
                resumed_clocks=compare(logged_steps(migrated, 2, 3), logged_steps(continuous)[2:]))
        report["cases"][mode] = result
        print(mode, json.dumps(result), flush=True)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
