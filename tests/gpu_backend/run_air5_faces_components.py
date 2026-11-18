"""Frozen CPU/old-GPU comparison of resident AIR5 directional face candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_flux_components import extract
from run_air5_properties_components import read as read_properties

ROOT=Path(__file__).resolve().parents[2]
BASE=["src/astr_precision.F90","src/constdef.F90","src/chemistry_air5_data.F90","src/chemistry_core.F90","src/chemistry_properties.F90"]
GPU=["src_gpu/chemistry_core_gpu.cuf"]
FILES=BASE+GPU+["src/chemistry_solver.F90","src_gpu/chemistry_solver_gpu.cuf","src/chemistry_pure_vt_body.inc",
    "src/chemistry_vibrational_difference_body.inc","src_gpu/chemistry_faces.cu","src_gpu/chemistry_face_math.cuh",
    "src_gpu/chemistry_transport.h","src_gpu/chemistry_transport.F90","src_gpu/chemistry_properties_math.cuh",
    "src_gpu/chemistry_air5_data.h","src_gpu/device_state.h","src_gpu/device_state_internal.cuh","src_gpu/shock_sensor_internal.cuh",
    "src_gpu/CMakeLists.txt","tests/gpu_backend/air5_faces_probe.cu","tests/gpu_backend/air5_faces_storage.F90",
    "tests/gpu_backend/air5_faces_reference.F90"]

def run(command,out,log):
    with (out/log).open("w") as stream:
        subprocess.run(list(map(str,command)),cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)

def references(out,compiler):
    common=["air5_minmod2","air5_minmod4","air5_suw3","air5_mp5","air5_mp7"]
    cpu=common+["air5_frozen_spectral_radius","air5_llf_split_flux","air5_shock_interface_active","air5_projected_convective_flux",
        "air5_projected_convective_line_flux","air5_centered_convective_face_flux","air5_shock_convective_face_flux",
        "air5_high_order_convective_face_flux","air5_selective_convective_face_flux","air5_low_order_convective_face_flux",
        "projected_air5_diffusive_flux","projected_air5_line_flux","projected_air5_centered_face_flux","projected_air5_face_flux"]
    gpu=[s+"_gpu" for s in common]+["record_air5_status","air5_convective_flux_at","air5_frozen_spectral_radius_gpu",
        "air5_llf_split_flux_gpu","air5_shock_interface_active_gpu","air5_convective_line_flux","air5_centered_convective_face_flux",
        "air5_shock_convective_face_flux_gpu","air5_high_order_convective_face_flux","air5_selective_convective_face_flux_gpu",
        "air5_low_order_convective_face_flux","air5_diffusive_flux_at","air5_diffusive_line_flux_at",
        "air5_centered_diffusive_face_flux","air5_diffusive_face_flux"]
    report={}
    for backend,names in [("cpu",cpu),("old_gpu",gpu)]:
        work=out/backend;work.mkdir();source=out/"source";is_gpu=backend=="old_gpu"
        original=source/("src_gpu/chemistry_solver_gpu.cuf" if is_gpu else "src/chemistry_solver.F90")
        (work/"frozen_faces_functions.inc").write_text("\n".join(extract(original.read_text(),name) for name in names))
        flags=[str(compiler),"-O2","-Mpreprocess","-Mextend","-I"+str(source/"src"),"-I"+str(work)]
        files=BASE.copy()
        if is_gpu:
            flags += ["-cuda","-gpu=rdc,maxregcount:128","-D_CUDA","-DASTR_AIR5_REFERENCE_GPU"];files += GPU
        files += ["tests/gpu_backend/air5_faces_storage.F90","tests/gpu_backend/air5_faces_reference.F90"]
        commands=[];objects=[]
        for name in files:
            obj=work/(Path(name).stem+".o");objects.append(str(obj));command=flags+[str(source/name),"-c","-o",str(obj)]
            # Match the original device source mode. NVHPC 26.1 ICEs on this
            # module-array operator when the identical wrapper is named .F90.
            if is_gpu and name.endswith("air5_faces_reference.F90"):
                cuf=work/"frozen_faces_reference.cuf";shutil.copy2(source/name,cuf)
                command=flags+[str(cuf),"-c","-o",str(obj)]
            commands.append(command);run(command,work,obj.stem+".log")
        command=flags+objects+["-o",str(work/"probe")];commands.append(command);run(command,work,"link.log");report[backend]=commands
    return report

def jobs():
    result=[]
    for axis in range(1,4):
        for ntype in range(1,5):
            for symmetric in [0,1]:
                for capturing in [0,1]:result.append([axis,ntype,0,capturing,symmetric])
                result.append([axis,ntype,1,0,symmetric])
            for kind in [2,3]:result.append([axis,ntype,kind,0,1])
    return np.array(result,dtype="<i4")

def make_input(path,properties,mode,configurations=None):
    im,jm,km=8,7,6;h=5
    k,j,i=np.mgrid[-h:km+h+1,-h:jm+h+1,-h:im+h+1];xyz=np.stack([i,j,k],axis=-1).reshape(-1,3)
    face=np.sum((xyz<0)|(xyz>[im,jm,km]),axis=1)<=1
    index=(17*xyz[:,0]+23*xyz[:,1]+31*xyz[:,2])%431 if mode=="nonuniform" else np.full(len(xyz),175)
    p=properties[index];table=np.full((len(xyz),64),np.nan,dtype="<f8")
    table[face,:11]=p[face,:11];table[face,11]=p[face,11];table[face,12:15]=p[face,12:15]
    table[face,15]=p[face,22];table[face,16:21]=p[face,16:21]
    metric=np.broadcast_to(np.eye(3),(len(xyz),3,3)).copy();jac=np.ones(len(xyz))
    if mode=="nonuniform":
        metric[:,0,0]+=.01*xyz[:,0];metric[:,1,1]+=.015*xyz[:,1];metric[:,2,2]+=.02*xyz[:,2]
        metric[:,0,1]=.1;metric[:,0,2]=-.07;metric[:,1,0]=.08;metric[:,1,2]=.04;metric[:,2,0]=-.03;metric[:,2,1]=.09
        jac=.7+.02*xyz[:,0]+.015*xyz[:,1]+.01*xyz[:,2]
    table[face,21:30]=metric.transpose(0,2,1).reshape(-1,9)[face];table[face,30]=jac[face]
    polynomial=np.ones(len(xyz)) if mode=="constant" else 1.+.03*xyz[:,0]-.02*xyz[:,1]+.01*xyz[:,2]
    if mode=="nonuniform":polynomial+=.3*np.sin(.7*xyz[:,0]+.2*xyz[:,1]-.4*xyz[:,2])
    table[face,31:37]=polynomial[face,None]*np.array([1.,-.5,.25,2.,.4,3.])*1e-4
    table[face,37:40]=polynomial[face,None]*np.array([2.,-1.,.3])*100.
    coefficients=np.array([.1,-.1,.05,-.07,.02]);coefficients[0]=-sum(coefficients[1:])
    js=polynomial[:,None,None]*np.array([1.,-.4,.2])[None,:,None]*coefficients[None,None,:]*1e-5
    table[face,40:55]=js.reshape(-1,15)[face];table[face,55:58]=polynomial[face,None]*np.array([1.,3.,-2.])*.1
    table[face,58:64]=polynomial[face,None]*np.array([.2,-.1,.3,.4,-.2,.1])*1000.
    kk,jj,ii=np.mgrid[:km+1,:jm+1,:im+1];mask=((ii+2*jj+3*kk)%7<2).astype("<i4")
    configurations=jobs() if configurations is None else configurations
    with path.open("wb") as stream:
        np.array([im,jm,km,len(configurations)],dtype="<i4").tofile(stream);configurations.tofile(stream);table.tofile(stream);mask.tofile(stream)
    return table.reshape(km+11,jm+11,im+11,64)

def read(path):
    records={}
    with path.open("rb") as stream:
        n=int(np.fromfile(stream,"<i4",1)[0])
        for _ in range(n):
            packet=np.fromfile(stream,"<i4",6);assert len(packet)==6
            records[tuple(map(int,packet[:5]))]=np.fromfile(stream,"<f8",11*int(packet[5])).reshape(-1,11)
        assert stream.read()==b"" and len(records)==n
    return records

def compare(a,b):
    assert a.keys()==b.keys();worst=0;key=None
    for job,ref in b.items():
        value=a[job];assert np.isfinite(value).all() and value.shape==ref.shape
        scale=np.maximum(1.,abs(ref).max(axis=1,keepdims=True));ratio=float(np.max(abs(value-ref)/(1e-13*scale+5e-12*abs(ref))))
        if ratio>worst:worst=ratio;key=job
        assert ratio<=1.,(job,ratio)
    return dict(maximum_budget_ratio=worst,case=key)

def independent(record,table,mode):
    h=5;im,jm,km=8,7,6;worst=0
    for job,value in record.items():
        axis,ntype,kind,capturing,symmetric=job;axis-=1
        if kind<=1 and mode!="constant":continue
        first=[0,0,0];first[axis]=-1
        kk,jj,ii=np.mgrid[first[2]:km+1,first[1]:jm+1,first[0]:im+1]
        point=np.stack([ii,jj,kk],axis=-1).reshape(-1,3);other=point.copy();other[:,axis]+=1
        left=table[point[:,2]+h,point[:,1]+h,point[:,0]+h];right=table[other[:,2]+h,other[:,1]+h,other[:,0]+h]
        if kind<=1:
            expected=left[:,:11]*left[:,12+axis,None];expected[:,1+axis]+=left[:,15]
            expected[:,4]+=left[:,15]*left[:,12+axis]
            if not symmetric:expected[:,5]=expected[:,0]-expected[:,6:10].sum(axis=1)
        else:
            def project(a):
                out=np.zeros((len(a),11));tensor=np.array([[0,1,2],[1,3,4],[2,4,5]])
                if kind==2:
                    out[:,1:4]=a[:,31+tensor[:,axis]];out[:,4]=a[:,37+axis];out[:,10]=a[:,55+axis]
                else:out[:,4]=a[:,58+axis];out[:,10]=a[:,61+axis]
                out[:,5:10]=a[:,40+5*axis:45+5*axis];return out
            expected=.5*(project(left)+project(right))
        scale=np.maximum(1.,abs(expected).max(axis=1,keepdims=True));error=float(np.max(abs(value-expected)/(1e-13*scale+5e-12*abs(expected))))
        assert error<=1.,(mode,job,error);worst=max(worst,error)
    return worst

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path);p.add_argument("--compiler",required=True,type=Path)
    p.add_argument("--properties-reference",required=True,type=Path);p.add_argument("--memcheck",action="store_true")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in FILES:
        target=out/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(__file__,out/"driver.py");shutil.copy2(args.candidate,out/"candidate")
    report=dict(scope="face reconstruction only; no positivity, RHS, MPI or CFD acceptance",commands=references(out,args.compiler.resolve()),cases={})
    _,props=read_properties(args.properties_reference.resolve()/"cpu_all.bin")
    for mode in ["constant","affine","nonuniform","inviscid"]:
        configurations=jobs()
        if mode=="inviscid":configurations=configurations[(configurations[:,2]<=1)&(configurations[:,3]==0)]
        source_mode="constant" if mode=="inviscid" else mode
        table=make_input(out/f"{mode}.bin",props,source_mode,configurations);ref={}
        for backend in ["cpu","old_gpu"]:
            run([out/backend/"probe",out/f"{mode}.bin",out/f"{mode}_{backend}.bin"],out,f"{mode}_{backend}.log")
            ref[backend]=read(out/f"{mode}_{backend}.bin")
        result={"old_cpu":compare(ref["old_gpu"],ref["cpu"])}
        for debug in [0,1]:
            run([out/"candidate",out/f"{mode}.bin",out/f"{mode}_cpp{debug}.bin",debug],out,f"{mode}_cpp{debug}.log")
            current=read(out/f"{mode}_cpp{debug}.bin")
            result[f"cpp{debug}_cpu"]=compare(current,ref["cpu"]);result[f"cpp{debug}_old"]=compare(current,ref["old_gpu"])
            if mode!="nonuniform":result[f"independent{debug}"]=independent(current,table,source_mode)
            for job,value in current.items():
                if job[2]<=1 and job[4]==0:
                    other=job[:4]+(1,);assert np.array_equal(value[:,[0,1,2,3,4,6,7,8,9,10]],current[other][:,[0,1,2,3,4,6,7,8,9,10]])
            if debug==0:quiet=current
            else:assert all(np.array_equal(current[k],quiet[k]) for k in current)
        if mode=="nonuniform":
            changed=sum(not np.array_equal(value,quiet[job[:3]+(1,)+job[4:]]) for job,value in quiet.items() if job[2]==0 and job[3]==0)
            assert changed>0;result["selective_capturing_changed_cases"]=changed
        report["cases"][mode]=result;print(mode,json.dumps(result))
    if args.memcheck:
        tool=shutil.which("compute-sanitizer")
        if tool is None:raise RuntimeError("requested memory tool unavailable")
        for mode in ["nonuniform","inviscid"]:
            name="memcheck" if mode=="nonuniform" else "memcheck_inviscid"
            run([tool,"--tool","memcheck","--leak-check","full","--error-exitcode","77","--log-file",out/f"{name}.log",
                out/"candidate",out/f"{mode}.bin",out/f"{name}.bin",0],out,f"{name}_stdout.log")
    report.update(status="face-components-passed",candidate_sha256=hashlib.sha256((out/"candidate").read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((out/"source"/name).read_bytes()).hexdigest() for name in FILES})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
