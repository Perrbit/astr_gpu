"""Frozen original AIR5 diffusion budgets and RHS from prescribed viscous fluxes."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from run_air5_filter_fields import command, fixture
from run_air5_legacy_filter_controller import build_reference


def read(prefix, rank):
    path=Path(f"{prefix}.rank{rank}.bin")
    with path.open("rb") as stream:
        header=np.fromfile(stream,"<i4",6)
        minimum=np.fromfile(stream,"<f8",1)[0]
        size=math.prod(header[:3]+11)
        ratios=np.fromfile(stream,"<f8",2*size).reshape(2,size)
        rhs=np.fromfile(stream,"<f8").reshape(11,-1)
    assert np.isfinite(ratios).all() and np.isfinite(rhs).all()
    assert np.min(ratios)>=0. and np.max(ratios)<=1.
    return header,minimum,ratios,rhs


def compare(old, candidate, ranks):
    max_ratio=0.;max_rhs=0.;n_limited=0;min_energy=1.
    for rank in range(ranks):
        a,x,r,ref=read(old,rank)
        b,y,s,got=read(candidate,rank)
        assert np.array_equal(a,b), (old,candidate,rank,a,b)
        # Reuse the frozen face/RHS component absolute+relative scales.
        ratio_error=float(np.max(abs(r-s)/(1e-13+5e-12*abs(r))))
        scaled=np.maximum(1.,np.max(abs(ref),axis=0))
        rhs_error=float(np.max(abs(got-ref)/(1e-13*scaled+5e-12*abs(ref))))
        assert ratio_error<=1. and rhs_error<=1., (old,candidate,rank,ratio_error,rhs_error)
        assert abs(x-y)<=1e-13+5e-12*abs(x)
        assert np.array_equal(got[0],ref[0]), (candidate,rank,"diffusion changed mass RHS")
        max_ratio=max(max_ratio,ratio_error);max_rhs=max(max_rhs,rhs_error)
        n_limited+=int(b[5]);min_energy=min(min_energy,float(s[1].min()))
    return dict(ratio_budget=max_ratio,rhs_budget=max_rhs,species_points_limited=n_limited,
                minimum_energy_ratio=min_energy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("candidate","mpiexec","output"):
        parser.add_argument(f"--{key}",type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-build",type=Path)
    source.add_argument("--frozen-reference",type=Path)
    parser.add_argument("--memcheck",type=Path)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True)
    public=["configure_air5_vibrational_floor_gpu,air5_flow_status_d,reset_air5_status",
            "reset_air5_diffusion_failure,air5_diffusion_layered_d,air5_convection_consistent_d",
            "air5_limited_points_d,air5_minimum_ratio_bits_d,air5_diffusion_ratio_kernel",
            "air5_reset_diffusion_ratio_kernel,air5_diffusion_energy_ratio_kernel",
            "air5_diffusion_rhs_x_kernel,air5_diffusion_rhs_y_kernel,air5_diffusion_rhs_z_kernel"]
    if args.frozen_reference:
        reference=args.frozen_reference.resolve(strict=True)
    else:
        reference=build_reference(args.reference_build.resolve(),out,"air5_diffusion_probe.F90",public,["-DASTR_OLD_DIFFUSION"])
    (out/"reference_identity.txt").write_text(str(reference)+"\n")
    candidate=args.candidate.resolve()
    env=os.environ|dict(OMPI_MCA_pml="ob1",OMPI_MCA_btl="self,vader,tcp",OMPI_MCA_osc="pt2pt",
                        OMPI_MCA_coll="^cuda,hcoll,ucc",OMPI_MCA_opal_cuda_support="0")
    inp=out/"state.bin"
    _,q,_,_,floor=fixture("smooth",[1,1,1])
    np.r_[q[2,2,2,:],floor].astype("<f8").tofile(inp)
    report={}
    for test,topology in [(0,(1,1,1)),(0,(2,1,1)),(1,(1,2,1)),(2,(2,1,1)),(2,(1,1,2)),(3,(2,1,1))]:
        ranks=math.prod(topology)
        launch=[args.mpiexec.resolve(),"--bind-to","none","-np",ranks]
        for layered in (0,1):
            label=f"p{test}_t{''.join(map(str,topology))}_l{layered}"
            work=out/label;work.mkdir()
            old=work/"old"
            command([*launch,reference,inp,old,*topology,layered,test,0],work,"old.log",env)
            for mode in ((0,1) if ranks>1 else (0,)):
                new=work/f"new{mode}"
                cmd=[*launch,candidate,inp,new,*topology,layered,test,mode]
                runtime=env if not mode else env|dict(OMPI_MCA_pml="ucx",OMPI_MCA_opal_cuda_support="1")
                command(cmd,work,f"new{mode}.log",runtime)
                measured=compare(old,new,ranks)
                statuses=[list(map(int,read(new,i)[0][3:5])) for i in range(ranks)]
                assert statuses==([[3,3],[0,3]] if test==3 else [[0,0]]*ranks), (label,statuses)
                if test in (1,2):
                    assert measured["species_points_limited"]>0, (label,measured)
                if test==2 and layered:
                    assert measured["minimum_energy_ratio"]<1., (label,measured)
                report[f"{label}/{mode}"]=dict(**measured,status=statuses,command=list(map(str,cmd)))
                if args.memcheck and test==2 and layered and topology==(2,1,1) and mode==0:
                    mem=work/"mem"
                    cmd[cmd.index(new)]=mem
                    sanitizer=[args.memcheck.resolve(),"--tool","memcheck","--leak-check","full","--error-exitcode","99"]
                    command([*launch,*sanitizer,*cmd[len(launch):]],work,"memcheck.log",env)
                    compare(old,mem,ranks)
            (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(label, "passed",flush=True)
    for mode in (0,1):
        work=out/f"uniform_m{mode}";work.mkdir()
        runtime=env if not mode else env|dict(OMPI_MCA_pml="ucx",OMPI_MCA_opal_cuda_support="1")
        cmd=[args.mpiexec.resolve(),"--bind-to","none","-np",2,candidate,inp,work/"out",2,1,1,1,4,mode]
        command(cmd,work,"run.log",runtime)
        report[f"uniform/{mode}"]=dict(rhs_unchanged=True,command=list(map(str,cmd)))
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
