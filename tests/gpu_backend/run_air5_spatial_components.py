"""Manufactured AIR5 field gradients and frozen CPU/old-GPU local flux oracles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_properties_components import read as read_properties

ROOT=Path(__file__).resolve().parents[2]

def run(command,out,log):
    with (out/log).open("w") as stream:
        subprocess.run(list(map(str,command)),cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)

def scaled_error(a,b):
    scale=np.maximum(1.,abs(b).max(axis=0))
    error=float(np.max(abs(a-b)/scale))
    assert np.isfinite(a).all() and error<=2e-10,error
    return error

def component_ratio(a,b):
    scale=np.maximum(1.,abs(b).max(axis=1,keepdims=True))
    error=float(np.max(abs(a-b)/(1e-13*scale+5e-12*abs(b))))
    assert np.isfinite(a).all() and error<=1.,error
    return error

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--memcheck",action="store_true")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    reference=args.reference.resolve();shutil.copy2(args.candidate,out/"candidate")
    files=["src_gpu/chemistry_transport.cu","src_gpu/chemistry_transport.h","src_gpu/chemistry_transport.F90",
        "src_gpu/chemistry_properties_math.cuh","src_gpu/device_state.h","src_gpu/device_state.F90",
        "src_gpu/fluid_explicit.cu","src_gpu/device_state_internal.cuh","tests/gpu_backend/air5_spatial_probe.cu"]
    for name in files:
        target=out/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(__file__,out/"driver.py")
    im,jm,km=8,7,6;halo=5
    z,y,x=np.mgrid[-halo:km+halo+1,-halo:jm+halo+1,-halo:im+halo+1]
    xyz=np.stack([x,y,z],axis=-1).reshape(-1,3)
    physical=np.all((xyz>=0)&(xyz<=[im,jm,km]),axis=1)
    face=np.sum((xyz<0)|(xyz>[im,jm,km]),axis=1)<=1
    # Quadratic fields are exact even under the original second-order physical
    # closure; the non-diagonal metrics exercise the full tensor projection.
    bases=np.array([.02,250.,12.,-7.,4000.,3500.,.72,.19,.04,.03])
    magnitude=np.array([2e-5,.5,.1,.08,3.,2.,1e-5,2e-5,1e-5,1e-5])
    xx,yy,zz=xyz.T
    poly=xx-.7*yy+.3*zz+.02*xx**2-.03*yy**2+.04*zz**2+.01*xx*yy
    values=bases+poly[:,None]*magnitude
    fractions=np.column_stack([values[:,6:10],1.-values[:,6:10].sum(axis=1)])
    # Oracle pressure is obtained below; supplied field pressure uses that same
    # CPU result, isolating spatial layout/derivative errors from EOS rounding.
    metric=np.array([[1.,.11,-.07],[.08,1.3,.04],[-.03,.09,.9]])
    derivatives=np.stack([1.+.04*xx+.01*yy,-.7-.06*yy+.01*xx,.3+.08*zz],axis=1)
    gradients=derivatives@metric
    derivatives_all=magnitude[None,:,None]*gradients[:,None,:]
    dy=np.concatenate([derivatives_all[:,6:10,:],-derivatives_all[:,6:10,:].sum(axis=1,keepdims=True)],axis=1)
    configs=[([0,0,0],[0,0,0],0),([1,1,1],[1,1,1],1),([1,0,1],[0,1,0],0),([0,1,0],[1,0,1],1)]
    report={"scope":"field gradients/local transport fluxes, not divergence/positivity/MPI/full CFD", "cases":[]}
    for closure,(low,high,zero) in enumerate(configs):
        local_dy=dy.copy()
        if zero:local_dy[yy==0,:,1]=0.
        oracle=np.zeros((int(face.sum()),41),dtype="<f8")
        oracle[:,0]=values[face,0];oracle[:,1:4]=values[face,1:4];oracle[:,4]=values[face,4]
        oracle[:,5:10]=fractions[face];oracle[:,10]=values[face,5]
        oracle[:,11:20]=derivatives_all[face,1:4,:].transpose(0,2,1).reshape(-1,9)
        oracle[:,20:23]=derivatives_all[face,4,:];oracle[:,23:26]=derivatives_all[face,5,:]
        oracle[:,26:41]=local_dy[face].transpose(0,2,1).reshape(-1,15)
        name=f"closure{closure}"
        with (out/f"{name}_oracle.bin").open("wb") as stream:
            np.array([len(oracle)],dtype="<i4").tofile(stream);oracle.tofile(stream)
        refs={}
        for backend in ["cpu","old_gpu"]:
            run([reference/backend/"probe",out/f"{name}_oracle.bin",out/f"{name}_{backend}.bin","all"],out,f"{name}_{backend}.log")
            status,data=read_properties(out/f"{name}_{backend}.bin");assert np.all(status==0)
            refs[backend]=data
        field=np.full((len(xyz),21),np.nan,dtype="<f8")
        field[face,:4]=values[face,:4];field[face,4:6]=values[face,4:6]
        field[face,6]=refs["cpu"][:,23];field[face,7:12]=fractions[face]
        field[face,12:]=metric.ravel(order="F")
        analytic=np.concatenate([derivatives_all[physical,1:4,:].transpose(0,2,1).reshape(-1,9),
            derivatives_all[physical,4,:],derivatives_all[physical,5,:],local_dy[physical].transpose(0,2,1).reshape(-1,15)],axis=1)
        physical_in_face=physical[face]
        for layered in [0,1]:
            key=f"{name}_layer{layered}"
            with (out/f"{key}.bin").open("wb") as stream:
                np.array([im,jm,km]+low+high+[layered,zero],dtype="<i4").tofile(stream);field.tofile(stream)
            run([out/"candidate",out/f"{key}.bin",out/f"{key}_cpp.bin"],out,f"{key}.log")
            result=np.fromfile(out/f"{key}_cpp.bin","<f8").reshape(-1,63)
            errors={"gradients_analytic_scaled":scaled_error(result[:,:30],analytic)}
            for backend,ref in refs.items():
                ref=ref[physical_in_face]
                expected=np.column_stack([ref[:,[74,77,80,78,81,82]],ref[:,98:101],ref[:,83:98],ref[:,101:104]])
                errors[backend+"_flux_budget_ratio"]=component_ratio(result[:,30:57],expected)
                if layered:
                    js=ref[:,83:98].reshape(-1,3,5)
                    energies=np.concatenate([-np.sum(js*ref[:,None,104:109],axis=2),-np.sum(js*ref[:,None,109:114],axis=2)],axis=1)
                    errors[backend+"_layered_budget_ratio"]=component_ratio(result[:,57:63],energies)
                else:assert np.all(result[:,57:]==0)
            report["cases"].append(dict(closure=closure,layered=layered,nodes=int(physical.sum()),errors=errors))
    convergence=[]
    mechanism=json.loads((ROOT/"chemMech/air5_kimjo12.json").read_text())
    gas=np.array([8.31446261815324/s["molecular_weight_kg_per_mol"] for s in mechanism["species"]])
    for intervals in [16,32,64]:
        zz,yy,xx=np.mgrid[-halo:5+halo,-halo:5+halo,-halo:intervals+halo+1]
        xyz=np.stack([xx,yy,zz],axis=-1).reshape(-1,3)
        physical=np.all((xyz>=0)&(xyz<=[intervals,4,4]),axis=1)
        face=np.sum((xyz<0)|(xyz>[intervals,4,4]),axis=1)<=1
        x=2.*np.pi*xyz[:,0]/intervals
        field=np.full((len(xyz),21),np.nan,dtype="<f8")
        field[face,0]=.02;field[face,1:4]=[250.,12.,-7.]
        field[face,4]=4000.+200.*np.sin(x[face]);field[face,5]=3500.+100.*np.sin(x[face])
        field[face,7:12]=[.72,.19,.04,.03,.02]
        field[face,6]=field[face,0]*field[face,4]*(field[face,7:12]@gas)
        field[face,12:]=np.diag([intervals/(2.*np.pi),1.,1.]).ravel(order="F")
        name=f"periodic{intervals}"
        with (out/f"{name}.bin").open("wb") as stream:
            np.array([intervals,4,4]+[0]*8,dtype="<i4").tofile(stream);field.tofile(stream)
        run([out/"candidate",out/f"{name}.bin",out/f"{name}_cpp.bin"],out,f"{name}.log")
        result=np.fromfile(out/f"{name}_cpp.bin","<f8").reshape(-1,63)
        error=float(np.max(abs(result[:,12]-100.*np.cos(x[physical])))/100.)
        convergence.append(dict(intervals=intervals,tv_gradient_relative_max=error))
    orders=np.log2([convergence[i]["tv_gradient_relative_max"]/convergence[i+1]["tv_gradient_relative_max"] for i in range(2)])
    assert np.all(orders>5.7),orders
    report["periodic_gradient_convergence"]=dict(cases=convergence,observed_orders=orders.tolist())
    if args.memcheck:
        tool=shutil.which("compute-sanitizer")
        if tool is None:raise RuntimeError("requested memory tool unavailable")
        run([tool,"--tool","memcheck","--leak-check","full","--error-exitcode","77","--log-file",out/"memcheck.log",
            out/"candidate",out/"closure1_layer1.bin",out/"memcheck.bin"],out,"memcheck_stdout.log")
    report.update(status="field-components-passed",reference=str(reference),
        reference_sha256={b:hashlib.sha256((reference/b/"probe").read_bytes()).hexdigest() for b in ["cpu","old_gpu"]},
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(cases=report["cases"],convergence=report["periodic_gradient_convergence"]),indent=2))

if __name__=="__main__":main()
