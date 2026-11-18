"""Compare resident symmetric AIR5 RK-budget/limited-face/RHS stages.

All state advancement here is in compiled original/new GPU routines. Python
only supplies frozen states and checks snapshots/discrete flux identities.
"""
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
BASE=["src/astr_precision.F90","src/constdef.F90","src/chemistry_air5_data.F90","src/chemistry_core.F90",
    "src/chemistry_properties.F90","src_gpu/chemistry_core_gpu.cuf","src_gpu/chemistry_flow_state_gpu.cuf"]
FILES=BASE+["src_gpu/chemistry_solver_gpu.cuf","src_gpu/chemistry_compensation_gpu.cuf","src_gpu/halo_exchange_gpu.cuf","src/chemistry_pure_vt_body.inc",
    "src/chemistry_vibrational_difference_body.inc","src_gpu/chemistry_convection.cu","src_gpu/chemistry_faces.cu",
    "src_gpu/chemistry_face_math.cuh","src_gpu/chemistry_flux_math.cuh","src_gpu/chemistry_properties_math.cuh",
    "src_gpu/chemistry_transport.h","src_gpu/chemistry_transport.F90","src_gpu/device_state.h","src_gpu/device_state.F90",
    "src_gpu/device_state.cu","src_gpu/device_state_internal.cuh","src_gpu/halo_transport.cu","src_gpu/chemistry_air5_data.h",
    "src_gpu/CMakeLists.txt","tests/gpu_backend/air5_faces_storage.F90","tests/gpu_backend/air5_convection_reference.cuf",
    "tests/gpu_backend/air5_convection_probe.cu","tests/gpu_backend/air5_convection_compensation.cuf"]

def run(command,out,log):
    with (out/log).open("w") as stream:
        subprocess.run(list(map(str,command)),cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)

def build_reference(out,compiler):
    work=out/"old_gpu";work.mkdir();source=out/"source"
    scalar=["record_air5_status","air5_minmod2_gpu","air5_minmod4_gpu","air5_suw3_gpu","air5_mp5_gpu","air5_mp7_gpu",
        "air5_convective_flux_at","air5_frozen_spectral_radius_gpu","air5_llf_split_flux_gpu","air5_shock_interface_active_gpu",
        "air5_convective_line_flux","air5_centered_convective_face_flux","air5_shock_convective_face_flux_gpu",
        "air5_high_order_convective_face_flux","air5_selective_convective_face_flux_gpu","air5_low_order_convective_face_flux",
        "air5_vibrational_excess_gpu","air5_translational_margin_gpu","air5_state_is_admissible_gpu","air5_symmetric_face_trial_gpu"]
    stages=["air5_symmetric_active","air5_symmetric_low_face","air5_top_normal_limit_kernel","air5_symmetric_base_kernel",
        "air5_symmetric_face_kernel","record_air5_face_failure","air5_symmetric_apply_kernel","air5_symmetric_final_kernel",
        "air5_symmetric_plane_kernel","air5_rk3_first_update_kernel","air5_rk3_update_kernel"]
    body=(source/"src_gpu/chemistry_solver_gpu.cuf").read_text()
    halo=(source/"src_gpu/halo_exchange_gpu.cuf").read_text()
    (work/"frozen_convection_functions.inc").write_text("\n".join(extract(body,name) for name in scalar+stages)+
        "\n"+"\n".join(extract(halo,"field_qswap_"+axis+"_kernel") for axis in ["x","y","z"]))
    compensation=(source/"src_gpu/chemistry_compensation_gpu.cuf").read_text()
    (work/"frozen_rk_compensation.inc").write_text("\n".join(extract(compensation,name) for name in ["compensated_add_gpu","compensated_rk_gpu"]))
    flags=[str(compiler),"-O2","-Mpreprocess","-Mextend","-cuda","-gpu=rdc,maxregcount:128","-D_CUDA",
        "-DASTR_AIR5_REFERENCE_GPU","-I"+str(work),"-I"+str(source/"src")]
    files=BASE+["tests/gpu_backend/air5_faces_storage.F90","tests/gpu_backend/air5_convection_compensation.cuf",
        "tests/gpu_backend/air5_convection_reference.cuf"]
    commands=[];objects=[]
    for name in files:
        obj=work/(Path(name).stem+".o");objects.append(str(obj));options=flags.copy()
        if name.endswith("air5_convection_compensation.cuf"):options += ["-Kieee","-Mnofma","-gpu=nofma,noflushz"]
        command=options+[str(source/name),"-c","-o",str(obj)]
        commands.append(command);run(command,work,obj.stem+".log")
    command=flags+objects+["-o",str(work/"probe")];commands.append(command);run(command,work,"link.log");return commands

def make_input(path,properties,mode):
    im,jm,km=8,7,6;h=5;types=[3,3,3] if mode.endswith("periodic") else ([4,3,3] if mode.endswith("open_x") else [4,4,3])
    jobs=[]
    for top in ([0,1] if mode=="contact_hbl" else [0]):
        for stage in [1,2,3]:
            for capturing in [0,1]:
                for preserve in [0,1]:jobs.append(types+[stage,capturing,preserve,top,0])
    if mode=="contact_hbl":jobs.append(types+[1,1,1,0,1])
    jobs=np.array(jobs,dtype="<i4");floor=properties[0,109:114];dt=1e-7
    kk,jj,ii=np.mgrid[-h:km+h+1,-h:jm+h+1,-h:im+h+1];xyz=np.stack([ii,jj,kk],axis=-1).reshape(-1,3)
    face=np.sum((xyz<0)|(xyz>[im,jm,km]),axis=1)<=1;mapped=xyz.copy()
    for a,upper in enumerate([im,jm,km]):mapped[:,a]=mapped[:,a]%upper if types[a]==3 else np.clip(mapped[:,a],0,upper)
    if mode.startswith("constant"):index=np.full(len(xyz),120)
    else:index=119+(mapped[:,0]//2+2*(mapped[:,1]//2)+3*(mapped[:,2]//2))%7
    prop=properties[index];table=np.full((len(xyz),31),np.nan,dtype="<f8")
    table[face,:11]=prop[face,:11];table[face,11]=prop[face,11];table[face,12:15]=prop[face,12:15]
    table[face,15]=prop[face,22];table[face,16:21]=prop[face,16:21]
    metric=np.broadcast_to(np.eye(3),(len(xyz),3,3)).copy();jac=np.ones(len(xyz))
    if not mode.startswith("constant"):
        metric[:,0,1]=.07;metric[:,1,0]=.03;metric[:,1,2]=-.04;metric[:,2,1]=.02
        jac=1.+.1*np.sin(2*np.pi*mapped[:,0]/im)*np.cos(2*np.pi*mapped[:,2]/km)
    table[face,21:30]=metric.transpose(0,2,1).reshape(-1,9)[face];table[face,30]=jac[face]
    physical=np.all((xyz>=0)&(xyz<=[im,jm,km]),axis=1)
    q=table[physical,:11].reshape(km+1,jm+1,im+1,11);j=jac[physical].reshape(km+1,jm+1,im+1,1)
    saved=q*j*.99999;rhs=q*j*.001;initial=np.concatenate([saved,rhs],axis=-1).astype("<f8")
    top=(2.*rhs[:,jm,:,:]).transpose(2,0,1).copy().astype("<f8")
    k,y,x=np.mgrid[:km+1,:jm+1,:im+1];mask=((x+2*y+3*k)%5<2).astype("<i4")
    with path.open("wb") as stream:
        np.array([im,jm,km,len(jobs)],dtype="<i4").tofile(stream);jobs.tofile(stream)
        np.asarray(floor,dtype="<f8").tofile(stream);np.array([dt],dtype="<f8").tofile(stream)
        table.tofile(stream);initial.tofile(stream);top.tofile(stream);mask.tofile(stream)
    return dict(upper=[im,jm,km],jobs=jobs,dt=dt,initial=initial)

def read(path,upper):
    im,jm,km=upper;total=(im+11)*(jm+11)*(km+11);physical=(im+1)*(jm+1)*(km+1);faces=(im+2)*(jm+2)*(km+2)
    result=[]
    with path.open("rb") as stream:
        n=int(np.fromfile(stream,"<i4",1)[0])
        for _ in range(n):
            option=np.fromfile(stream,"<i4",9)
            data={"option":option[:8],"status":int(option[8]),
                "base":np.fromfile(stream,"<f8",11*total).reshape(11,km+11,jm+11,im+11),
                "start":np.fromfile(stream,"<f8",11*physical).reshape(11,km+1,jm+1,im+1),
                "faces":np.fromfile(stream,"<f8",36*faces).reshape(3,12,km+2,jm+2,im+2),
                "rhs":np.fromfile(stream,"<f8",11*physical).reshape(11,km+1,jm+1,im+1),
                "rk":np.fromfile(stream,"<f8",11*physical).reshape(11,km+1,jm+1,im+1),
                "failure":np.fromfile(stream,"<i4",7),"failure_values":np.fromfile(stream,"<f8",66).reshape(6,11),
                "failure_dt":float(np.fromfile(stream,"<f8",1)[0])}
            result.append(data)
        assert stream.read()==b""
    return result

def ratio(a,b):
    assert np.isfinite(a).all() and np.isfinite(b).all() and a.shape==b.shape
    # Same original local chemistry component absolute/relative budget.
    scale=max(1.,float(np.max(abs(b))));value=float(np.max(abs(a-b)/(1e-13*scale+5e-12*abs(b))))
    assert value<=1.,value
    return value

def verify(records,metadata,mode):
    im,jm,km=metadata["upper"];k,j,i=np.mgrid[:km+1,:jm+1,:im+1]
    rhs_ratio=0.;conservation_ratio=0.;limited=0
    for record in records:
        option=record["option"];top=option[6]!=0;corrupt=option[7]!=0
        assert record["status"]==(202 if corrupt else 0)
        active=np.ones(i.shape,dtype=bool)
        if top:active=(j==jm)&(i!=0)
        else:
            for a,p in enumerate([i,j,k]):
                if option[a] in [1,4]:active &= p!=0
                if option[a] in [2,4]:active &= p!=metadata["upper"][a]
        expected=record["start"].astype(np.longdouble).copy();magnitude=np.zeros_like(expected)
        for axis in range(3):
            if top and axis==1:continue
            if corrupt and axis>0:break
            p=[k+1,j+1,i+1];left=[a.copy() for a in p];left[2-axis]-=1
            face=record["faces"][axis]
            ll=face[:11,left[0],left[1],left[2]]
            rr=face[:11,p[0],p[1],p[2]]
            expected[:,active]+=ll[:,active].astype(np.longdouble)-rr[:,active].astype(np.longdouble)
            magnitude[:,active]+=abs(ll[:,active])+abs(rr[:,active])
            beta=face[11];limited+=int(np.sum((beta>0)&(beta<1)))
        gamma=64*np.finfo(float).eps/(1-64*np.finfo(float).eps)
        difference=abs(record["rhs"].astype(np.longdouble)-expected)
        rounding_bound=gamma*(abs(record["start"])+magnitude)
        assert np.all(difference<=rounding_bound)
        rhs_ratio=max(rhs_ratio,float(np.max(np.divide(difference,rounding_bound,out=np.zeros_like(difference),where=rounding_bound>0))))
        original=metadata["initial"][...,11:].transpose(3,0,1,2)
        assert np.array_equal(record["rhs"][:,~active],original[:,~active])
        if mode.startswith("constant"):
            # Mixed centered/MP arithmetic need only preserve the constant field
            # to the existing operation budget, not make cancellation exact.
            assert np.all(abs(record["rhs"])<=gamma*magnitude)
        if np.all(option[:3]==3) and not corrupt and not top:
            residual=abs(record["rhs"][:,:km,:jm,:im].astype(np.longdouble).sum(axis=(1,2,3)))
            scale=magnitude[:,:km,:jm,:im].sum(axis=(1,2,3))
            bound=gamma*scale;assert np.all(residual<=bound)
            conservation_ratio=max(conservation_ratio,float(np.max(np.divide(residual,bound,out=np.zeros_like(residual),where=bound>0))))
        if corrupt:
            assert record["failure"].tolist()==[1,1,1,1,1,202,0]
            assert record["failure_values"][0,0]<0
    return dict(discrete_rhs_budget_ratio=rhs_ratio,periodic_flux_balance_ratio=conservation_ratio,limited_face_coefficients=limited)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path);p.add_argument("--compiler",required=True,type=Path)
    p.add_argument("--mpiexec",required=True,type=Path)
    p.add_argument("--properties-reference",required=True,type=Path);p.add_argument("--memcheck",action="store_true")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in FILES:
        target=out/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(args.candidate,out/"candidate");shutil.copy2(__file__,out/"driver.py")
    report=dict(scope="frozen field convection RK-budget stages, not complete CFD/MPI",commands=build_reference(out,args.compiler.resolve()),cases={})
    _,properties=read_properties(args.properties_reference.resolve()/"cpu_all.bin")
    launcher=[args.mpiexec.resolve(),"-np","1"];report["launcher"]=list(map(str,launcher))
    for mode in ["constant_periodic","contact_periodic","contact_open_x","contact_hbl"]:
        metadata=make_input(out/f"{mode}.bin",properties,mode)
        run([out/"old_gpu/probe",out/f"{mode}.bin",out/f"{mode}_old.bin"],out,f"{mode}_old.log")
        reference=read(out/f"{mode}_old.bin",metadata["upper"]);info={}
        for debug in [0,1]:
            run(launcher+[out/"candidate",out/f"{mode}.bin",out/f"{mode}_cpp{debug}.bin",debug],out,f"{mode}_cpp{debug}.log")
            current=read(out/f"{mode}_cpp{debug}.bin",metadata["upper"]);assert len(current)==len(reference)
            errors={name:0. for name in ["base","start","faces","beta","rk","failure_values"]};raw_rhs=0.;rhs_bound_ratio=0.
            for row,old in zip(current,reference):
                assert np.array_equal(row["option"],old["option"]) and row["status"]==old["status"]
                assert np.array_equal(row["failure"],old["failure"]) and row["failure_dt"]==old["failure_dt"]
                for name in errors:
                    if name=="beta":value=ratio(row["faces"][:,11],old["faces"][:,11])
                    elif name=="faces":value=ratio(row["faces"][:,:11],old["faces"][:,:11])
                    else:value=ratio(row[name],old[name])
                    errors[name]=max(errors[name],value)
                raw_rhs=max(raw_rhs,float(np.max(abs(row["rhs"]-old["rhs"]))))
                # Cross-backend RHS differences are conditioned by the actual
                # face operands. Keep their raw values; do not normalize a
                # cancellation residual with the property-vector forward gate.
                scale=abs(row["start"]).astype(np.longdouble)+abs(old["start"])
                delta=abs(row["start"].astype(np.longdouble)-old["start"])
                im,jm,km=metadata["upper"];k,j,i=np.mgrid[:km+1,:jm+1,:im+1];pidx=[k+1,j+1,i+1]
                for axis in range(3):
                    if row["option"][6] and axis==1:continue
                    if row["option"][7] and axis>0:break
                    left=[x.copy() for x in pidx];left[2-axis]-=1
                    new_face=row["faces"][axis];old_face=old["faces"][axis]
                    for index in [left,pidx]:
                        f=new_face[:11,index[0],index[1],index[2]]
                        g=old_face[:11,index[0],index[1],index[2]]
                        scale+=abs(f)+abs(g);delta+=abs(f.astype(np.longdouble)-g)
                gamma=64*np.finfo(float).eps/(1-64*np.finfo(float).eps);bound=delta+gamma*scale
                actual=abs(row["rhs"].astype(np.longdouble)-old["rhs"])
                assert np.all(actual<=bound)
                rhs_bound_ratio=max(rhs_bound_ratio,float(np.max(np.divide(actual,bound,out=np.zeros_like(actual),where=bound>0))))
            info[str(debug)]=dict(reference_budget_ratios=errors,raw_rhs_max_abs=raw_rhs,rhs_operand_bound_ratio=rhs_bound_ratio,
                independent=verify(current,metadata,mode))
            if debug==0:quiet=current
            else:
                for row,old in zip(current,quiet):
                    for name in errors:
                        if name!="beta":assert np.array_equal(row[name],old[name])
                    assert np.array_equal(row["rhs"],old["rhs"])
        report["cases"][mode]=info;print(mode,json.dumps(info))
    if args.memcheck:
        tool=shutil.which("compute-sanitizer")
        if tool is None:raise RuntimeError("requested memory tool unavailable")
        run(launcher+[tool,"--tool","memcheck","--leak-check","full","--error-exitcode","77","--log-file",out/"memcheck.log",
            out/"candidate",out/"contact_hbl.bin",out/"memcheck.bin",0],out,"memcheck_stdout.log")
    report.update(status="symmetric-field-stages-passed",source_sha256={name:hashlib.sha256((out/"source"/name).read_bytes()).hexdigest() for name in FILES})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
