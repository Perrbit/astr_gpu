"""Compare the C++ AIR5 filter stage against the original CUDA Fortran controller."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np

from run_air5_filter_fields import command, fixture, write_input


ROOT = Path(__file__).resolve().parents[2]


def build_reference(build, out, main_name="air5_legacy_filter_controller.cuf", public=None, main_flags=()):
    source = ROOT / "src_gpu/chemistry_solver_gpu.cuf"
    main = ROOT / "tests/gpu_backend" / main_name
    shutil.copy2(source, out)
    shutil.copy2(main, out)
    variant = out / "probe_chemistry_solver.cuf"
    if public is None:
        public = ["air5_flow_status_d, configure_air5_vibrational_floor_gpu, reset_air5_status",
                  "air5_save_filter_species_base_kernel, air5_limit_filtered_state_kernel"]
    variant.write_text(source.read_text().replace(
        "  public :: air5_prepare_spatial_state_gpu\n",
        "  public :: air5_prepare_spatial_state_gpu\n" + "".join(f"  public :: {line}\n" for line in public),
        1))
    work = build / "src"
    target = work / "CMakeFiles/astr.dir"
    flags = dict(l.split(" = ", 1) for l in (target / "flags.make").read_text().splitlines() if " = " in l)
    link = shlex.split((target / "link.txt").read_text())
    opts = [link[0], *shlex.split(flags["Fortran_FLAGS"]), *shlex.split(flags["Fortran_INCLUDES"]),
            *shlex.split(flags["Fortran_DEFINES"]), "-I" + str(out), "-I" + str(work),
            "-I" + str(ROOT / "src"), "-I" + str(ROOT / "src_gpu")]
    main_cuf = (out / main.name).with_suffix(".cuf")
    if main_cuf.name != main.name:
        shutil.copy2(main, main_cuf)
    solver_options = [part for key,value in flags.items() if key.endswith("/chemistry_solver_gpu.cuf.o_OPTIONS")
                      for part in value.split(";")]
    jobs = [opts + [*solver_options,"-c", variant, "-o", out / "solver.o"],
            opts + [*main_flags,"-c", main_cuf, "-o", out / "main.o"]]
    link = [t for t in link[:link.index("-o")] if not (t.startswith("-Wl,--dependency-file=") or
            t.endswith(("/astr.F90.o", "/chemistry_solver_gpu.cuf.o")))] + [
            out / "solver.o", out / "main.o", "-o", out / "reference"]
    # Preserve libraries after -o in the root target's original link order.
    original = shlex.split((target / "link.txt").read_text())
    link += original[original.index("-o")+2:]
    for i, cmd in enumerate(jobs + [link]):
        command(cmd, out if i<2 else work, out / f"reference_build{i}.log")
    tracked = [source, main, *[ROOT / "src_gpu" / n for n in
               ("solver_gpu.cuf", "halo_exchange_gpu.cuf", "chemistry_compensation_gpu.cuf")]]
    (out / "reference_provenance.json").write_text(json.dumps(
        dict(source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
             commands=[list(map(str, c)) for c in jobs + [link]]), indent=2) + "\n")
    return out / "reference"


def compare(old, new, ranks, expect_failure=False):
    for rank in range(ranks):
        suffix = f".rank{rank}.bin"
        a, b = [Path(str(p)+suffix).read_bytes() for p in (old, new)]
        header = np.frombuffer(a, "<i4", count=4)
        assert bool(header[3]) == expect_failure, (old, rank, header)
        assert a[:16] == b[:16], (old, new, rank, header, np.frombuffer(b, "<i4", count=4))
        assert a == b, (old, new, rank,
            int(np.count_nonzero(np.frombuffer(a[16:], "<u8")!=np.frombuffer(b[16:], "<u8"))))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-build", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memcheck", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference = build_reference(args.reference_build.resolve(), out)
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
                            OMPI_MCA_osc="pt2pt", OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
    device_env = env | dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    cases = [
        ("periodic", [1,1,1], -1, 0, 0, [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]),
        ("physical_y", [1,0,1], -1, 0, 1, [(1,1,1),(1,2,1)]),
        ("wall", [1,0,1], 1, 0, 1, [(1,1,1),(1,2,1)]),
        ("adiabatic", [1,0,1], 1, 1, 1, [(1,1,1),(1,2,1)]),
        ("failure", [0,1,1], -1, 0, 0, [(2,1,1)]),
    ]
    report = {}
    for name, periodic, wall, family, comp, topologies in cases:
        work = out / name
        work.mkdir()
        global_, q, carry, tr, v = fixture("smooth", periodic)
        if name == "failure":
            q[4,4,0,4] = -10.*abs(q[4,4,0,4])
        path = work / "input.bin"
        write_input(path, global_, periodic, q, carry, tr, v)
        for topo in topologies:
            for scalar in (0,1):
                ranks=math.prod(topo)
                launch=[args.mpiexec.resolve(), "--bind-to", "none", "-np", ranks]
                stem="".join(map(str, topo))+f"_scalar{scalar}"
                old=work / (stem + "_old")
                ref_env=env | dict(ASTR_GPU_FILTER_WORKSPACE="scalar" if scalar else "full")
                command([*launch, reference, path, old, *topo, comp, wall, 2, family], work, old.with_suffix(".log"), ref_env)
                for transport in ((0,1) if ranks>1 else (0,)):
                    tag=stem+f"_comm{transport}"
                    new=work / tag
                    cmd=[*launch, args.candidate.resolve(), path, new, *topo, scalar, comp, transport, 1, 2, wall, family]
                    run_env=device_env if transport else env
                    # The probe returns four on expected physical rejection.
                    if name == "failure":
                        with new.with_suffix(".log").open("w") as log:
                            rc=subprocess.run(list(map(str, cmd)), cwd=work, env=run_env,
                                              stdout=log, stderr=subprocess.STDOUT).returncode
                        assert rc==4, (tag, rc)
                        statuses=new.with_suffix(".log").read_text()
                        assert "local_status=0 global_status=3" in statuses
                        assert "local_status=3 global_status=3" in statuses
                    else:
                        command(cmd, work, new.with_suffix(".log"), run_env)
                    compare(old,new,ranks,name=="failure")
                    report[f"{name}/{tag}"]=dict(exact=True,topology=topo,scalar=scalar,compensated=comp,
                        transport=transport,command=list(map(str,cmd)))
        print(name, "matches original controller", flush=True)
        (out / "controller_report.json").write_text(json.dumps(report, indent=2)+"\n")
    if args.memcheck:
        work=out/"wall"
        new=work/"memcheck"
        cmd=[args.mpiexec.resolve(),"--bind-to","none","-np",2,args.memcheck.resolve(),"--tool","memcheck",
             "--leak-check","full","--error-exitcode","99",args.candidate.resolve(),work/"input.bin",new,
             1,2,1,1,1,0,0,2,1,0]
        command(cmd,work,"memcheck.log",env)
        compare(work/"121_scalar1_old",new,2)


if __name__=="__main__":
    main()
