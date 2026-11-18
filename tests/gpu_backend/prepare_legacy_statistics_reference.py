"""Freeze actual legacy mean-statistics lifecycle on the existing small plate fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from prepare_upwind_components import prepare_flat
from run_cpp_tgv_gate import checkpoint, compare, logged_steps, namelist
from run_cpp_insitu_gate import compare_hdf
from named_fixture import convert_case


def sampling_isolation(enabled, disabled):
    # Legacy sampling refreshes velocity/temperature periodic face halos. Those
    # derived caches need not have identical freshness with sampling disabled.
    # All physical fields and the complete conservative halo remain invariant.
    with h5py.File(enabled) as left, h5py.File(disabled) as right:
        for key in left:
            if key == "rank_extras":
                continue
            a, b = left[key][...], right[key][...]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise AssertionError((key, "sampling changed physical state/identity"))
        shape = np.asarray(left["identity"][...][3:6], int)
        extras_a, extras_b = left["rank_extras"][...], right["rank_extras"][...]
        offset = 0
        records = []
        for partition in left["partitions"][...].reshape(-1, 8):
            origin, cells, halo, count = partition[:3], partition[3:6], int(partition[6]), int(partition[7])
            a = extras_a[offset:offset+count].reshape(11, -1)
            b = extras_b[offset:offset+count].reshape(11, -1)
            word=np.uint32 if a.dtype.itemsize==4 else np.uint64
            changed = a.view(word) != b.view(word)
            if changed[:6].any() or changed[9].any():
                raise AssertionError("sampling changed conservative/density/pressure halo")
            owned = cells + (origin + cells == shape - 1)
            mask = np.ones(tuple(cells + 2*halo + 1), dtype=bool)
            mask[tuple(slice(halo, halo+n) for n in owned)] = False
            xyz = np.argwhere(mask.transpose(2, 1, 0))[:, ::-1] - halo
            outside = (xyz < 0) | (xyz > cells)
            global_xyz = xyz + origin
            permitted = outside.sum(axis=1) == 1
            # z is periodic; x/y physical exterior faces must stay untouched.
            permitted &= np.all((global_xyz[:, :2] >= 0) & (global_xyz[:, :2] < shape[:2]), axis=1)
            if changed[:, ~permitted].any():
                raise AssertionError("sampling changed physical duplicate/nonperiodic/corner cache")
            records.append(changed.sum(axis=1).tolist())
            offset += count
        if offset != len(extras_a):
            raise AssertionError("legacy statistic extras layout")
        return dict(physical_fields_and_full_conservative_halo_bits_equal=True,
                    refreshed_cache_halo_words=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "mpiexec", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--sample-clock", choices=("steps", "time"), default="steps")
    parser.add_argument("--case", choices=("compact", "repartition"), default="compact")
    parser.add_argument("--mode", choices=("fixed", "automatic", "all"), default="all")
    parser.add_argument("--joint-family", choices=("keep", "me6", "mp7ld_compact"))
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    parser.add_argument('--native-reference',type=Path)
    parser.add_argument('--steps',type=int,choices=(4,6))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "preparation.py")
    reference = args.reference.resolve()
    if args.precision=='fp32' and args.native_reference is None:
        parser.error('FP32 references require an explicit original native executable')
    if args.native_reference:shutil.copy2(args.native_reference,root/'original_astr')
    report = {}
    total_steps = args.steps or (6 if args.case == "repartition" else 4)
    final = Path(f"outdat/new/checkpoints/step{total_steps:012d}")
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
        OMPI_MCA_osc="pt2pt", OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
        ASTR_FORCE_MPI_TOPOLOGY="1,1,1", ASTR_GPU_HALO_TRANSPORT="pinned",
        ASTR_GPU_COMPACT_TRANSPORT="pinned", ASTR_GPU_FILTER_WORKSPACE="scalar", ASTR_GPU_SYNC_MODE="explicit",
        ASTR_GPU_MP_POSITIVITY="off", ASTR_VALIDATION_RHS_PREFIX="outdat/rhs",
        ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_END=str(total_steps))

    def run(name, backend, automatic, steps=None, restore=None, enabled=True):
        steps = total_steps if steps is None else steps
        gpu = backend == "old_gpu"
        binary = root/'original_astr' if args.native_reference else reference / "bin" / ("astr_cuda_fortran" if gpu else "astr_cpu")
        case = root / name
        prepare_flat(case, gpu)
        groups = convert_case((case / "datin/input.flatplate").read_text())
        groups["numerics"].update(convection_form="standard", viscous_form="standard", difschm="643c",
                                  lfilter=False, filter_scheme="explicit10")
        if args.case == "repartition":
            groups["numerics"].update(conschm="543e", difschm="643e", recon_schem=3, lchardecomp=False, lfilter=True)
            groups["boundary"]["bctype"][3] = 51
            groups["initialization"]["ninit"] = 0
        if args.joint_family:
            if args.joint_family=="mp7ld_compact":
                groups["numerics"].update(conschm="743e",difschm="643c",recon_schem=5,lchardecomp=True)
            else:
                groups["numerics"].update(conschm="643e",difschm="643e",recon_schem=3,lchardecomp=False,
                    convection_form="keep" if args.joint_family=="keep" else "standard",
                    viscous_form="me6_opti" if args.joint_family=="me6" else "standard")
            groups["boundary"]["bctype"]=[11,22,41,52,1,1]
            groups["boundary_options"]=dict(nscbc_farfield_mode="incoming_only" if args.joint_family=="mp7ld_compact" else "nonreflecting",
                profile_inflow_mode="complete_state")
        groups["precision_options"] = dict(gpu_precision_mode="fp64")
        (case / "datin/input.case").write_text(namelist(groups))
        timestep = (dict(mode="auto", cfl_target=.5, initial_dt=1e-6, max_dt=2e-6, growth_factor=1.1)
                    if automatic else dict(mode="fixed", fixed_dt=1e-6))
        (case / "datin/input.run").write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=timestep, logging=dict(list_steps=1), repair=dict(crashfix=False),
            statistics=dict(enabled=enabled, sample_clock=args.sample_clock, sample_steps=1, sample_time=1.8e-6,
                            write_clock=args.sample_clock, write_steps=2, write_time=2e-6))))
        (case / "datin/input.output").write_text(namelist(dict(output=dict(directory="outdat/new",
            restore_directory=str(restore or ""), exact_restart=True, checkpoint_statistics=True,
            host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
            checkpoint=dict(enabled=True, mode="steps", interval_steps=2, keep=2, initial_frame=False),
            volume=dict(enabled=False), slices=dict(enabled=False))))
        (case / "outdat/new").mkdir()
        command = [str(args.mpiexec.resolve()), "--bind-to", "none", "-n", "1", str(binary), "run"]
        with (case / "run.log").open("w") as log:
            subprocess.run(command, cwd=case, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        directories = sorted((case / "outdat/new/checkpoints").glob("step*"))
        expected_retained = list(range(4 if restore else 2, steps+1, 2))[-2:]
        if [int(p.name[4:]) for p in directories] != expected_retained:
            raise AssertionError("original mean checkpoint retention")
        for directory in directories:
            n = int(directory.name[4:])
            q = checkpoint(directory / "state.h5")
            p = .4 * (q[..., 4] - .5 * np.sum(q[..., 1:4]**2, axis=-1) / q[..., 0])
            if not np.isfinite(q).all() or not (q[..., 0] > 0).all() or not (p > 0).all():
                raise AssertionError("invalid original physical state")
            if enabled:
                with h5py.File(directory / "statistics.h5") as source:
                    def finite(key, value):
                        if isinstance(value, h5py.Dataset) and value.dtype.kind == "f" and not np.isfinite(value[...]).all():
                            raise AssertionError(f"nonfinite legacy mean reference: {name}/{key}")
                    source.visititems(finite)
                    metadata = source["metadata"][...].reshape(-1)
                    times = np.r_[0., np.cumsum(1e-6*1.1**np.arange(n) if automatic else np.full(n, 1e-6))]
                    expected_count = n+1 if args.sample_clock == "steps" else 1+np.count_nonzero(np.diff(np.floor(times/1.8e-6)))
                    version = 4 if args.case == "repartition" else 3
                    if gpu and (len(metadata) != 14 or metadata[0] != version or metadata[7] != expected_count or metadata[9] != n):
                        raise AssertionError(("original compact clock", name, metadata))
        return case

    for automatic in (False, True) if args.mode == "all" else (args.mode == "automatic",):
        mode = "automatic" if automatic else "fixed"
        pair = {}
        for backend in ("cpu", "old_gpu"):
            name = f"{backend}_{mode}"
            continuous = run(name, backend, automatic)
            quiet = run(name + "_off", backend, automatic, enabled=False)
            isolation = sampling_isolation(continuous / final / "state.h5", quiet / final / "state.h5")
            if not np.array_equal(logged_steps(continuous, total_steps), logged_steps(quiet, total_steps)):
                raise AssertionError("original legacy sampling changed the timestep")
            prefix = run(name + "_prefix", backend, automatic, steps=2)
            tail = run(name + "_tail", backend, automatic,
                       restore=prefix / "outdat/new/checkpoints/step000000000002")
            for file in ("state.h5", "statistics.h5"):
                compare_hdf(continuous / final / file, tail / final / file, exact=True)
            if (continuous / final / "control.bin").read_bytes() != (tail / final / "control.bin").read_bytes():
                raise AssertionError("original legacy exact control restart mismatch")
            report[name] = dict(state_and_statistics_exact_restart=True, sampling_clock_bits_unchanged=True,
                                sampling_isolation=isolation)
            pair[backend] = continuous
        report[mode + "_crosscheck"] = dict(state=compare(checkpoint(pair["cpu"] / final / "state.h5").astype(np.float64),
            checkpoint(pair["old_gpu"] / final / "state.h5").astype(np.float64),limit=None if args.precision=='fp32' else 2e-10),
            clocks=compare(logged_steps(pair["cpu"], total_steps), logged_steps(pair["old_gpu"], total_steps)))
        print(mode, json.dumps(report), flush=True)
    report["reference_binaries"] = {name: hashlib.sha256((root/'original_astr' if args.native_reference else reference / "bin" / name).read_bytes()).hexdigest()
                                   for name in ("astr_cpu", "astr_cuda_fortran")}
    report['precision']=args.precision
    report["scope"] = "existing admitted 16-cubed 743e/643c low-wall profile plate, fixed/automatic four steps; CPU44 and GPU12 statistics are distinct protocols, not compared as identical arrays"
    report["sample_clock"] = args.sample_clock
    report["case"] = args.case
    report["steps"] = total_steps
    if args.case == "repartition":
        report["scope"] = f"existing Cartesian 16-cubed 543e/643e filtered profile plate, bc11/21/41/51, ninit0; original role4 inherited-statistics contract, {total_steps} complete steps"
    if args.joint_family:
        report["scope"] = "existing Cartesian 16-cubed joint 11/22/41/52 plate; original CPU44/GPU12 mean protocols"
        report["joint_family"]=args.joint_family
    (root / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    for p in root.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)


if __name__ == "__main__":
    main()
