"""Verify the AIR5 device RK preparation against the original GPU controller."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from run_air5_legacy_filter_controller import build_reference
from run_air5_filter_fields import command, fixture, write_input


def read(prefix, rank):
    path = Path(f"{prefix}.rank{rank}.bin")
    shape = np.fromfile(path, "<i4", count=4)
    ijk = tuple(int(n)+1 for n in shape[2::-1])
    hl = tuple(n+10 for n in ijk)
    qsize = 22*math.prod(ijk)+11*math.prod(hl)
    values = np.fromfile(path, "<f8", offset=16)
    assert values.size == qsize+12*math.prod(hl)
    return shape, values[:qsize], values[qsize:].reshape((12,*hl))


def compare(old, new, ranks, failed):
    worst = 0.
    for rank in range(ranks):
        a, q1, f1 = read(old, rank)
        b, q2, f2 = read(new, rank)
        assert np.array_equal(a,b) and int(b[3])==(3 if failed else 0), (old,new,rank,a,b)
        assert np.array_equal(q1.view("u8"),q2.view("u8")), (old,new,rank,
            np.count_nonzero(q1.view("u8")!=q2.view("u8")),float(np.max(abs(q1-q2))))
        valid = f1[0]!=-999.
        assert np.all(f2[:,~valid]==-999.) and bool(valid.any())!=failed
        if failed:
            continue
        for fields in (slice(0,1),slice(1,4),slice(4,5),slice(5,6),slice(6,7),slice(7,12)):
            ref, got = f1[fields][:,valid], f2[fields][:,valid]
            assert np.isfinite(got).all()
            scale = np.maximum(1., np.max(abs(ref),axis=0))
            ratio=float(np.max(abs(got-ref)/(1e-13*scale+5e-12*abs(ref))))
            worst=max(worst,ratio)
    assert worst<=1., (new,worst)
    return worst


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("reference-build","candidate","mpiexec","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--memcheck",type=Path)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    reference=build_reference(args.reference_build.resolve(),out)
    candidate=args.candidate.resolve()
    env=os.environ|dict(OMPI_MCA_pml="ob1",OMPI_MCA_btl="self,vader,tcp",OMPI_MCA_osc="pt2pt",
                        OMPI_MCA_coll="^cuda,hcoll,ucc",OMPI_MCA_opal_cuda_support="0")
    cases=[
        ("periodic", [1,1,1], (1,1,1), 1, 1, 1),
        ("periodic_x", [1,1,1], (2,1,1), 1, 1, 1),
        ("periodic_z", [1,1,1], (1,1,2), 0, 0, 1),
        ("postshock_y", [0,1,1], (1,2,1), 1, 0, 1),
        ("postshock_x", [0,1,1], (2,1,1), 0, 1, 1),
        ("unfiltered_later", [0,1,1], (2,1,1), 0, 1, 4),
        ("legacy_failure", [0,1,1], (2,1,1), 0, 1, 1),
        ("unfiltered_reuse", [1,1,1], (2,1,1), 1, 1, 3),
        ("conservative", [1,1,1], (2,1,1), 0, 1, 2),
    ]
    report={}
    for name,periodic,topology,comp,scalar,stage in cases:
        work=out/name;work.mkdir()
        ranks=math.prod(topology)
        global_,q,carry,tr,v=fixture("smooth",periodic)
        if stage in (1,4):
            q[:,-1,:,:]*=1.+2.**-20
        failed=name=="legacy_failure"
        if failed:
            q[4,4,0,4]=-10.*abs(q[4,4,0,4])
        path=work/"input.bin";write_input(path,global_,periodic,q,carry,tr,v)
        launch=[args.mpiexec.resolve(),"--bind-to","none","-np",ranks]
        old=work/"old"
        if stage in (1,4):
            command([*launch,reference,path,old,*topology,comp,-1,2,0,"unfiltered" if stage==4 else "spatial"],
                    work,"old.log",env|dict(ASTR_GPU_FILTER_WORKSPACE="scalar" if scalar else "full"))
        for mode in (0,1) if ranks>1 else (0,):
            new=work/f"mode{mode}"
            cmd=[*launch,candidate,path,new,*topology,scalar,comp,mode,1,2,-1,0,stage]
            runenv=env if not mode else env|dict(OMPI_MCA_pml="ucx",OMPI_MCA_opal_cuda_support="1")
            with new.with_suffix(".log").open("w") as log:
                result=subprocess.run(list(map(str,cmd)),cwd=work,env=runenv,stdout=log,stderr=subprocess.STDOUT).returncode
            assert result==(4 if failed else 0), (new,result)
            if stage in (1,4):
                ratio=compare(old,new,ranks,failed)
            else:
                ratio=None
                for rank in range(ranks):
                    head,data,caches=read(new,rank)
                    assert head[3]==0
                    if stage==3:
                        assert np.all(caches==-999.)
                        first=rank*(global_[0]//2);last=first+global_[0]//2+1
                        expect=np.stack([q[:,:,first:last,:],carry[:,:,first:last,:]])
                        assert np.array_equal(data[:expect.size].view("u8"),expect.transpose(0,4,1,2,3).ravel().view("u8"))
                    else:
                        assert np.isfinite(caches).all() and np.all(caches[:,5:-5,5:-5,5:-5]!=-999.)
                        if mode:
                            compare(work/"mode0",new,ranks,False)
            report[f"{name}/{mode}"]=dict(legacy_budget_ratio=ratio,command=list(map(str,cmd)))
            if args.memcheck and name=="postshock_y" and mode==0:
                mem=work/"memcheck"
                cmd[cmd.index(new)]=mem
                check=[args.memcheck.resolve(),"--tool","memcheck","--leak-check","full","--error-exitcode","99"]
                command([*launch,*check,*cmd[len(launch):]],work,"memcheck.log",env)
                compare(old,mem,ranks,False)
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name, "passed",flush=True)


if __name__=="__main__":
    main()
