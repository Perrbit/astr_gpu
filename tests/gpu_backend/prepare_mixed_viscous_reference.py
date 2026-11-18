"""Freeze actual CPU/old-GPU phases for existing MP1/MP2 storage candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import prepare_case, namelist, snapshot, compare
from prepare_upwind_components import prepare_flat
from named_fixture import convert_case


def prepare_mixed_case(case,template,gpu,mixed,automatic,filter_scheme,fixture,candidate="viscous_flux",means=False,steps=4):
    precision=dict(gpu_precision_mode="mixed_workspace" if mixed else "fp64")
    if mixed:
        precision["gpu_mixed_candidate"]=candidate
    if fixture=="tgv":
        prepare_case(case,template,gpu,True,automatic,grid=16,steps=steps,dt=1e-4,max_dt=1e-3,
            numerics=dict(lfilter=filter_scheme!="none",
                filter_scheme="explicit10" if filter_scheme=="none" else filter_scheme,
                conschm="543e" if candidate=="flux" else "643e",difschm="643e",
                recon_schem=3 if candidate=="flux" else 0,lchardecomp=False),
            physics=dict(diffterm=candidate!="flux"),precision_options=precision)
    else:
        prepare_flat(case,gpu)
        groups=convert_case((case/"datin/input.flatplate").read_text())
        groups["numerics"].update(convection_form="standard",viscous_form="standard",
            conschm="543e",difschm="643e",recon_schem=3,lchardecomp=False,
            lfilter=filter_scheme!="none",filter_scheme="explicit10" if filter_scheme=="none" else filter_scheme)
        groups["boundary"]["bctype"][3]=51
        groups["precision_options"]=precision
        (case/"datin/input.case").write_text(namelist(groups))
        (case/"datin/input.run").write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode="auto",cfl_target=.5,initial_dt=1e-6,max_dt=1e-5,growth_factor=1.1)
                if automatic else dict(mode="fixed",fixed_dt=1e-6),
            logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=means,
                sample_clock="steps",sample_steps=1,write_clock="steps",write_steps=2))))
    (case/"datin/input.output").write_text(namelist(dict(output=dict(directory="outdat/new",
        exact_restart=True,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
        checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "template", "mpiexec", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--filter", choices=("none", "explicit10", "compact10"), default="none")
    parser.add_argument("--automatic", action="store_true")
    parser.add_argument("--fixture", choices=("tgv", "plate"), default="tgv")
    parser.add_argument("--candidate", choices=("viscous_flux", "flux", "derivative"), default="viscous_flux")
    parser.add_argument("--means", action="store_true")
    args = parser.parse_args()
    if args.means and args.fixture!="plate":parser.error("this mean-statistics fixture is a plate")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "preparation.py")
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
        OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0", ASTR_FORCE_MPI_TOPOLOGY="1,1,1",
        ASTR_GPU_SYNC_MODE="explicit", ASTR_VALIDATION_RHS_PREFIX="outdat/rhs", ASTR_VALIDATION_RHS_STEP="0",
        ASTR_VALIDATION_RHS_STEP_END="4", ASTR_VALIDATION_GRADIENTS="1", ASTR_GEOMETRY_DUMP="outdat/geometry.h5",
        ASTR_GPU_MP_POSITIVITY="off")
    report, fields, scalars = {}, {}, {}
    for name, executable, gpu, mixed in (("cpu", "astr_cpu", False, False),
        ("old_gpu_fp64", "astr_cuda_fortran", True, False), ("old_gpu_mixed", "astr_cuda_fortran", True, True)):
        case = root / name
        prepare_mixed_case(case,args.template.resolve(),gpu,mixed,args.automatic,args.filter,args.fixture,args.candidate,args.means)
        binary = (args.reference / "bin" / executable).resolve()
        with (case / "run.log").open("w") as log:
            subprocess.run([str(args.mpiexec.resolve()), "--bind-to", "none", "-n", "1", str(binary), "run"],
                cwd=case, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        fields[name] = snapshot(case / "outdat/rhs.post_update.step00000003.rk03.rank00000000.bin")
        q = fields[name]
        p = .4*(q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0])
        if not np.isfinite(q).all() or not (q[..., 0] > 0).all() or not (p > 0).all():
            raise AssertionError("original mixed reference physical state")
        scalars[name] = np.loadtxt(case / "flowstate.dat", skiprows=1, ndmin=2)
        if not np.isfinite(scalars[name]).all():
            raise AssertionError("original mixed scalar reference")
        active_workspace="flux_work" if args.candidate=="flux" else args.candidate
        if mixed and "ASTR_GPU_ACTIVE_MIXED_WORKSPACE="+active_workspace not in (case / "run.log").read_text():
            raise AssertionError("original mixed workspace not selected")
        report[name] = dict(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    report["cpu_old_gpu_fp64"] = compare(fields["cpu"], fields["old_gpu_fp64"])
    difference = np.max(np.abs(fields["old_gpu_mixed"]-fields["old_gpu_fp64"]), axis=(0, 1, 2))
    stat_difference = np.max(np.abs(scalars["old_gpu_mixed"]-scalars["old_gpu_fp64"]), axis=0)
    # Existing MP1 and MP2 TGV absolute gates, both with zero relative tolerance.
    field_atol,stat_atol=(2e-6,5e-11) if args.candidate=="flux" else ((0.,1e-8) if args.candidate=="derivative" else (1e-9,1e-10))
    if np.max(difference) > field_atol or np.max(stat_difference) > stat_atol:
        raise AssertionError("old mixed candidate exceeds its existing TGV absolute gates")
    report.update(mixed_vs_fp64_max_abs=difference.tolist(), scalar_max_abs=stat_difference.tolist(),
        filter=args.filter, automatic=args.automatic,fixture=args.fixture,initial_dt=1e-4 if args.fixture=="tgv" else 1e-6,
        candidate=args.candidate,field_atol=field_atol,statistics_atol=stat_atol,
        means=args.means,
        scope="16-cubed interval fixture, 4 steps, no checkpoint/archive; mixed storage candidate, not native FP32 or physical validation")
    if args.means:
        from run_cpp_legacy_statistics_gate import compact_statistics
        mean_deltas={}
        for step in (2,4):
            filename=f"outdat/statistics.step{step:012d}.h5"
            left,fp64=compact_statistics(root/"old_gpu_fp64"/filename)
            right,mixed=compact_statistics(root/"old_gpu_mixed"/filename)
            if not np.array_equal(left,right):raise AssertionError("original mixed mean clocks differ")
            mean_deltas[str(step)]=np.max(abs(mixed-fp64),axis=(0,1)).tolist()
        report["mean_mixed_fp64_abs_report_only"]=mean_deltas
    (root / "reference.json").write_text(json.dumps(report, indent=2)+"\n")
    for p in root.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
