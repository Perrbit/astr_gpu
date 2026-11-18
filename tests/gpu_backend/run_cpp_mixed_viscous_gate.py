"""Ordinary C++ mixed storage versus frozen matching old-GPU phases; no checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import snapshot, compare, logged_steps, cfl_log
from prepare_mixed_viscous_reference import prepare_mixed_case
from run_cpp_multi_tgv_gate import layouts,raw_field
from run_cpp_physical_gate import stages
from run_cpp_high_wall_gate import rhs_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "reference", "template", "mpiexec", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--transport", choices=("pinned", "device"), default="pinned")
    parser.add_argument("--sync", choices=("explicit", "dependency"), default="dependency")
    parser.add_argument("--filter", choices=("none", "explicit10", "compact10"), default="none")
    parser.add_argument("--workspace", choices=("scalar", "full"), default="scalar")
    parser.add_argument("--automatic", action="store_true")
    parser.add_argument("--fixture", choices=("tgv", "plate"), default="tgv")
    parser.add_argument("--candidate", choices=("viscous_flux", "flux", "derivative"), default="viscous_flux")
    parser.add_argument("--sanitizer", choices=("memcheck", "initcheck"))
    parser.add_argument("--observation-isolation", action="store_true")
    parser.add_argument("--means", action="store_true")
    parser.add_argument("--archives", action="store_true")
    args = parser.parse_args()
    if (args.means or args.archives) and args.fixture!="plate":parser.error("these consumers use the existing plate fixture")
    root, reference, binary = args.output.resolve(), args.reference.resolve(), args.binary.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "runner.py");shutil.copyfile(binary, root / "tested_astr")
    topology = [1, 1, 1];topology["xyz".index(args.axis)] = args.ranks
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
        OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
        ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)), ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync, ASTR_GPU_FILTER_WORKSPACE=args.workspace, ASTR_VALIDATION_RHS_PREFIX="outdat/rhs",
        ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_END="4", ASTR_GPU_MP_POSITIVITY="off")
    if args.transport == "device":
        env.update(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    report = dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
                  transport=args.transport,sync=args.sync,filter=args.filter,workspace=args.workspace,
                  automatic=args.automatic,fixture=args.fixture,candidate=args.candidate,means=args.means,archives=args.archives,cases={})
    actual_fields, actual_scalars = {}, {}
    for mode in ("fp64", "mixed"):
        case = root / mode
        prepare_mixed_case(case,args.template.resolve(),True,mode=="mixed",args.automatic,args.filter,args.fixture,args.candidate,args.means)
        if args.archives:
            from check_plate_basic_archives import configure
            configure(case)
        command = [str(args.mpiexec.resolve()),"--bind-to","none","-n",str(args.ranks)]
        if args.sanitizer and mode == "mixed":
            command += ["/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer","--tool",args.sanitizer,
                "--error-exitcode","99","--log-file",str(case / "sanitizer.%p.log")]
            if args.sanitizer == "memcheck":command += ["--leak-check","full"]
        with (case / "run.log").open("w") as log:
            subprocess.run(command+[str(binary),"run"],cwd=case,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        text = (case / "run.log").read_text()
        if text.count("backend=cuda") != args.ranks:
            raise AssertionError("ordinary C++ main not selected")
        marker = "ASTR_GPU_CPP mixed_workspace="+args.candidate
        if (marker in text) != (mode == "mixed"):
            raise AssertionError("ordinary precision selection does not match the requested mode")
        if list((case / "outdat/new/checkpoints").glob("step*")):
            raise AssertionError("mixed gate unexpectedly wrote checkpoints")
        expected = snapshot(reference / ("old_gpu_mixed" if mode == "mixed" else "old_gpu_fp64") /
            "outdat/rhs.post_update.step00000003.rk03.rank00000000.bin")
        actual = np.empty_like(expected)
        errors = np.zeros(5)
        # Shared endpoints may differ by roundoff. Choose the same rank's copy
        # in both modes, independently of nondeterministic MPI log arrival.
        for rank, layout in sorted(layouts(case).items()):
            values = snapshot(case / f"outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin")
            body = tuple(slice(o,o+n+1) for o,n in zip(layout["origin"],layout["upper"]))
            errors = np.maximum(errors,compare(values,expected[body]))
            actual[body] = values
        pressure = .4*(actual[...,4]-.5*np.sum(actual[...,1:4]**2,axis=-1)/actual[...,0])
        if not np.isfinite(actual).all() or not (actual[...,0]>0).all() or not (pressure>0).all():
            raise AssertionError("mixed ordinary physical state")
        scalars = np.loadtxt(case / "flowstate.dat",skiprows=1,ndmin=2)
        previous = reference / ("old_gpu_mixed" if mode == "mixed" else "old_gpu_fp64")
        scalar_difference = np.max(abs(scalars-np.loadtxt(previous / "flowstate.dat",skiprows=1,ndmin=2)),axis=0)
        if not np.isfinite(scalars).all() or np.max(scalar_difference)>(1e-8 if args.candidate=="derivative" else 1e-10):
            raise AssertionError("same-mode old-GPU scalar comparison")
        if not np.array_equal(logged_steps(case),logged_steps(previous)):
            raise AssertionError("same-mode complete-step clocks differ")
        if args.sanitizer and mode == "mixed":
            logs = list(case.glob("sanitizer.*.log"))
            if len(logs)!=args.ranks or any("ERROR SUMMARY: 0 errors" not in p.read_text() for p in logs):
                raise AssertionError("mixed ordinary sanitizer")
            if args.sanitizer == "memcheck" and any("LEAK SUMMARY: 0 bytes leaked" not in p.read_text() for p in logs):
                raise AssertionError("mixed ordinary device leak")
        # LP2: identical-input operators retain their own 2e-10 gate. These
        # snapshots come from separately advanced MP1 trajectories instead.
        rhs_limit=None if mode=="mixed" and args.candidate=="flux" else 2e-10
        conv_errors=rhs_phase(case,previous,"conv",limit=rhs_limit,componentwise=True)
        full_errors=rhs_phase(case,previous,"full",limit=rhs_limit,componentwise=True)
        report["cases"][mode] = dict(old_gpu_scaled=errors.tolist(),old_gpu_scalar_max_abs=scalar_difference.tolist(),
            phases=stages(case,previous,rhs_limit=rhs_limit),convection=max(conv_errors),full_rhs=max(full_errors),
            rhs_scaled_by_variable=dict(convection=conv_errors,full=full_errors),
            cfl_scaled=compare(cfl_log(case),cfl_log(previous)),trajectory_rhs_scaled_limit=rhs_limit)
        if args.means:
            from run_cpp_legacy_statistics_gate import compact_statistics
            report["cases"][mode]["mean_statistics"]={}
            for step in (2,4):
                file=f"outdat/statistics.step{step:012d}.h5"
                meta,actual_mean=compact_statistics(case/file);old_meta,old_mean=compact_statistics(previous/file)
                if not np.array_equal(meta[:4],old_meta[:4]) or not np.array_equal(meta[7:],old_meta[7:]):
                    raise AssertionError("mixed mean sampling identity/coverage changed")
                report["cases"][mode]["mean_statistics"][str(step)]=compare(actual_mean,old_mean)
        if args.archives:
            from check_plate_basic_archives import check
            report["cases"][mode]["basic_archives"]=check(case)
        actual_fields[mode],actual_scalars[mode] = actual,scalars
    delta = np.max(abs(actual_fields["mixed"]-actual_fields["fp64"]),axis=(0,1,2))
    scalar_delta = np.max(abs(actual_scalars["mixed"]-actual_scalars["fp64"]),axis=0)
    field_atol,stat_atol=(2e-6,5e-11) if args.candidate=="flux" else ((0.,1e-8) if args.candidate=="derivative" else (1e-9,1e-10))
    if np.max(delta)>field_atol or np.max(scalar_delta)>stat_atol:
        raise AssertionError("ordinary mixed candidate exceeds its existing FP64 comparison gates")
    if args.candidate=="derivative":
        for rank in range(args.ranks):
            file=f"outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin"
            if raw_field(root/"fp64"/file)[1].tobytes()!=raw_field(root/"mixed"/file)[1].tobytes():
                raise AssertionError("diagnostic candidate changed authoritative q buffer words")
        report["diagnostic_only_full_q_bits_equal"]=True
    if args.observation_isolation:
        quiet=root/"mixed_no_intermediate_observation"
        prepare_mixed_case(quiet,args.template.resolve(),True,True,args.automatic,args.filter,args.fixture,args.candidate,args.means)
        with (quiet/"run.log").open("w") as log:
            subprocess.run([str(args.mpiexec.resolve()),"--bind-to","none","-n",str(args.ranks),str(binary),"run"],
                cwd=quiet,env=env|dict(ASTR_VALIDATION_RHS_STEP="4",ASTR_VALIDATION_RHS_STEP_END="4"),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        for rank in range(args.ranks):
            file=f"outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin"
            if raw_field(root/"mixed"/file)[1].tobytes()!=raw_field(quiet/file)[1].tobytes():
                raise AssertionError("intermediate observations changed the full authoritative q buffer")
        if not np.array_equal(actual_scalars["mixed"],np.loadtxt(quiet/"flowstate.dat",skiprows=1,ndmin=2)) or \
            not np.array_equal(logged_steps(root/"mixed"),logged_steps(quiet)):
            raise AssertionError("intermediate observations changed scalar values or clocks")
        report["intermediate_observation_q_scalar_clock_exact"]=True
        if args.means:
            from run_cpp_insitu_gate import compare_hdf
            for step in (2,4):
                file=f"outdat/statistics.step{step:012d}.h5"
                compare_hdf(root/"mixed"/file,quiet/file,exact=True)
            report["observation_archive_switch_mean_bits_equal"]=True
    report.update(mixed_vs_fp64_max_abs=delta.tolist(),mixed_vs_fp64_scalar_max_abs=scalar_delta.tolist(),
                  checkpoint_tested=False,native_fp32_tested=False,field_atol=field_atol,statistics_atol=stat_atol)
    (root / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":main()
