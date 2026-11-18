"""Original/new full conservative filter on resident fields and real MPI."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from run_air5_flux_components import extract

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests/gpu_validation"))
from test_air5_filter_multidim_probe import fixture3
from test_air5_filter_flux_probe import fixture as fixture1

BASE=["src/chemistry_air5_data.F90","src/chemistry_core.F90","src/chemistry_properties.F90"]
MATH=["src/chemistry_filter_math.F90","src/chemistry_filter_operator.F90"]
FILES=BASE+MATH+["src/chemistry_compensation.F90","src/chemistry_solver.F90","src/chemistry_pure_vt_body.inc",
    "src/chemistry_vibrational_difference_body.inc","src_gpu/chemistry_core_gpu.cuf","src_gpu/chemistry_flow_state_gpu.cuf",
    "src_gpu/chemistry_solver_gpu.cuf","src_gpu/chemistry_filter.cu","src_gpu/chemistry_filter_math.cuh",
    "src_gpu/chemistry_flux_math.cuh","src_gpu/chemistry_compensation_math.cuh","src_gpu/chemistry_properties_math.cuh",
    "src_gpu/halo_transport.cu","src_gpu/chemistry_transport.F90","src_gpu/chemistry_transport.h",
    "src_gpu/device_state.cu","src_gpu/device_state_internal.cuh","src_gpu/CMakeLists.txt",
    "tests/gpu_backend/CMakeLists.txt","tests/gpu_backend/air5_filter_field_probe.cu",
    "tests/gpu_backend/air5_filter_field_reference.F90","tests/gpu_backend/run_air5_filter_fields.py",
    "tests/gpu_validation/test_air5_filter_multidim_probe.py","tests/gpu_validation/test_air5_filter_flux_probe.py",
    "tests/gpu_validation/test_air5_filter_predicate_probe.py","tests/gpu_validation/air5_radau_reference.py","chemMech/air5_kimjo12.json"]

def command(command,directory,log,env=None):
    with (directory/log).open("w") as stream:subprocess.run(list(map(str,command)),cwd=directory,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def references(out,compiler):
    source=out/"source";commands={}
    cpu=(source/"src/chemistry_solver.F90").read_text();gpu=(source/"src_gpu/chemistry_solver_gpu.cuf").read_text()
    for name in ["cpu","old_gpu"]:
        work=out/name;work.mkdir();on_gpu=name=="old_gpu";objects=[];commands[name]=[]
        (work/"frozen_filter_cpu.inc").write_text("\n".join(extract(cpu,f) for f in ["check_air5_filter_status","exchange_air5_filter_strips","exchange_air5_filter_faces","air5_filter_conservative"]))
        (work/"frozen_compensated_add.inc").write_text(extract((source/"src/chemistry_compensation.F90").read_text(),"compensated_add"))
        if on_gpu:
            funcs=["filter_gpu_index","filter_pack_gpu","filter_get_gpu","filter_faces_gpu","filter_update_gpu","filter_alias_gpu",
                "exchange_air5_filter_strips_gpu","exchange_air5_filter_plane_gpu","filter_check_state_gpu","reset_air5_status","air5_filter_conservative_gpu"]
            (work/"frozen_filter_gpu.inc").write_text("\n".join(extract(gpu,f) for f in funcs))
        flags=[str(compiler),"-O2","-Mpreprocess","-Mextend","-I"+str(work),"-I"+str(source/"src")]
        if on_gpu:flags += ["-cuda","-gpu=rdc,maxregcount:128","-D_CUDA","-DASTR_AIR5_REFERENCE_GPU"]
        files=BASE+(["src_gpu/chemistry_core_gpu.cuf","src_gpu/chemistry_flow_state_gpu.cuf"] if on_gpu else [])+MATH+["tests/gpu_backend/air5_filter_field_reference.F90"]
        for f in files:
            path=source/f;opts=flags.copy()
            if f in MATH:
                opts += ["-Kieee","-Mnofma"]
                if on_gpu:opts += ["-gpu=nofma,noflushz"]
            if on_gpu and f.endswith("air5_filter_field_reference.F90"):
                path=work/"reference.cuf";shutil.copy2(source/f,path)
            obj=work/(Path(f).stem+".o");objects.append(obj);cmd=opts+["-c",path,"-o",obj]
            commands[name].append(list(map(str,cmd)));command(cmd,work,obj.stem+".log")
        cmd=flags+objects+["-o",work/"probe"];commands[name].append(list(map(str,cmd)));command(cmd,work,"link.log")
    return commands

def fixture(name,periodic):
    global_=np.array([16,12,10]);k,j,i=np.mgrid[:11,:13,:17]
    mapped=[(a*32//n)%32 if p else np.minimum(a*32//n,31) for a,n,p in zip([i,j,k],global_,periodic)]
    if name=="pressure-floor":
        model,one=fixture1(name);q=np.broadcast_to(one[0],i.shape+(11,)).copy();carry=.25*abs(np.spacing(q));carry[...,1:4]=0.
    else:
        model,original,low=fixture3(name);q=original[mapped[2],mapped[1],mapped[0]];carry=low[mapped[2],mapped[1],mapped[0]]
        if name!="carry-only":carry=.25*abs(np.spacing(q));carry[...,1:4]=0.
    tr=model.formation_energy+300.*model.cv_tr;v=np.array([model.species_vibrational_energy(s,300.) for s in range(5)])
    return global_,q.copy(),carry.copy(),tr,v

def write_input(path,global_,periodic,q,carry,tr,v):
    with path.open("wb") as stream:
        np.r_[global_,periodic].astype("<i4").tofile(stream);np.r_[tr,v].astype("<f8").tofile(stream)
        np.concatenate([q,carry],axis=-1).astype("<f8").tofile(stream)

def read(path):
    with path.open("rb") as stream:
        head=np.fromfile(stream,"<i4",4);nx,ny,nz=head[:3]+1;data=np.fromfile(stream,"<f8").reshape(2,11,nz,ny,nx)
    assert head[3]==0,(str(path),int(head[3]));return data.transpose(0,2,3,4,1)

def gather(prefix,topology,global_,periodic):
    shape=tuple((global_+1)[::-1]);out=np.full((2,)+shape+(11,),np.nan);seen=np.zeros(shape,bool)
    for rank in range(math.prod(topology)):
        coordinate=np.array(np.unravel_index(rank,topology));upper=global_//topology;first=coordinate*upper
        value=read(Path(str(prefix)+f".rank{rank}.bin"));slices=tuple(slice(int(a),int(a+n+1)) for a,n in zip(first[::-1],upper[::-1]))
        present=out[(slice(None),)+slices];old=seen[slices]
        if old.any():assert np.array_equal(present[:,old].view('u8'),value[:,old].view('u8')),"shared alias words differ"
        out[(slice(None),)+slices]=value;seen[slices]=True
    assert seen.all() and np.isfinite(out).all()
    for axis,p in enumerate(periodic):
        if p:assert np.array_equal(np.take(out,0,axis=3-axis).view('u8'),np.take(out,-1,axis=3-axis).view('u8')),"periodic alias words differ"
    return out

def conservation(before,after,periodic):
    shape=before.shape[1:4];weight=np.ones(shape)
    for a,p in enumerate(periodic):
        w=np.ones(shape[2-a]);w[-1]=0 if p else .5
        if not p:w[0]=.5
        reshape=[1,1,1];reshape[2-a]=len(w);weight*=w.reshape(reshape)
    state=after[0]-after[1];assert (state[...,0]>0).all() and (state[...,5:11]>=0).all()
    closure=float(np.max(abs(np.sum(state[...,5:10]/state[...,0,None],axis=-1)-1)));assert closure<=128*np.finfo(float).eps
    maximum=0.
    for c in range(11):
        old=math.fsum((before[0,...,c]*weight).ravel())-math.fsum((before[1,...,c]*weight).ravel())
        new=math.fsum((after[0,...,c]*weight).ravel())-math.fsum((after[1,...,c]*weight).ravel())
        if old!=0:
            error=abs(new-old)/abs(old);assert error<=1e-14,(c,error);maximum=max(maximum,error)
        else:
            scale=math.fsum(((abs(before[0,...,c])+abs(before[1,...,c]))*weight).ravel())
            u=np.finfo(float).eps/2;gamma=(2*np.prod(shape)+12)*u/(1-(2*np.prod(shape)+12)*u)
            assert abs(new)<=gamma*scale,(c,new,scale)
    return dict(maximum_relative_drift=maximum,maximum_closure_error=closure)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--candidate",required=True,type=Path)
    parser.add_argument("--compiler",required=True,type=Path);parser.add_argument("--mpiexec",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path);parser.add_argument("--memcheck",type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in FILES:
        dst=out/"source"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst);hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    candidate=out/"candidate";shutil.copy2(args.candidate.resolve(),candidate)
    report=dict(source_sha256=hashes,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),commands=references(out,args.compiler),cases={})
    env=os.environ|dict(OMPI_MCA_pml="ob1",OMPI_MCA_btl="self,vader,tcp",OMPI_MCA_osc="pt2pt",OMPI_MCA_coll="^cuda,hcoll,ucc",OMPI_MCA_opal_cuda_support="0")
    cases=[("smooth",[1,1,1],0),("trace-step",[1,0,1],1),("vibration",[0,0,0],1),("pressure-floor",[1,0,1],1),("carry-only",[1,1,1],1)]
    for name,periodic,comp in cases:
        work=out/name;work.mkdir();global_,q,carry,tr,v=fixture(name,periodic)
        if not comp:carry*=0
        path=work/"input.bin";write_input(path,global_,periodic,q,carry,tr,v);initial=np.array([q,carry]);products={}
        for backend in ["cpu","old_gpu"]:
            prefix=work/backend;cmd=[args.mpiexec,"--bind-to","none","-np","1",out/backend/"probe",path,prefix,1,1,1,1,comp,2]
            command(cmd,work,backend+".log",env);products[backend]=gather(prefix,[1,1,1],global_,periodic)
        entry=dict(reference_difference=float(np.max(abs(products["cpu"]-products["old_gpu"]))),runs={})
        assert entry["reference_difference"]<=2e-10,(name,"reference discrepancy",entry["reference_difference"])
        configurations=[([1,1,1],1,0,0),([1,1,1],0,0,1)]
        if name=="smooth":configurations += [(t,1,m,0) for t in [[2,1,1],[1,2,1],[1,1,2],[2,2,1]] for m in [0,1]]
        elif name=="trace-step":configurations += [(t,1,m,1) for t in [[1,2,1],[2,2,1]] for m in [0,1]]
        else:configurations += [([1,1,2],1,m,0) for m in [0,1]]
        for topo,scalar,mode,debug in configurations:
            label=f'np{math.prod(topo)}_{"".join(map(str,topo))}_s{scalar}_m{mode}_d{debug}'
            prefix=work/label;run_env=env if mode==0 else env|dict(OMPI_MCA_pml="ucx",OMPI_MCA_opal_cuda_support="1")
            cmd=[args.mpiexec,"--bind-to","none","-np",math.prod(topo),candidate,path,prefix,*topo,scalar,comp,mode,debug,2]
            command(cmd,work,label+".log",run_env);value=gather(prefix,topo,global_,periodic)
            difference={b:float(np.max(abs((value[0]-p[0])-(value[1]-p[1])))) for b,p in products.items()}
            assert max(difference.values())<=2e-10,(name,label,difference)
            if name=="carry-only":
                tiny=512*np.finfo(float).eps*np.max(abs(carry));assert max(difference.values())<=tiny,(label,difference,tiny)
            entry["runs"][label]=dict(difference=difference,conservation=conservation(initial,value,periodic),command=list(map(str,cmd)))
        report["cases"][name]=entry;(out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(name,len(entry["runs"]),"field runs passed",flush=True)
    if args.memcheck:
        work=out/"trace-step";prefix=work/"memcheck"
        cmd=[args.mpiexec,"--bind-to","none","-np","2",args.memcheck,"--tool","memcheck","--leak-check","full","--error-exitcode","99",candidate,work/"input.bin",prefix,1,2,1,1,1,0,0,2]
        command(cmd,work,"memcheck.log",env);text=(work/"memcheck.log").read_text()
        assert text.count("ERROR SUMMARY: 0 errors")==2 and text.count("LEAK SUMMARY: 0 bytes leaked")==2
        report["memcheck"]=dict(command=list(map(str,cmd)),clean_ranks=2)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
