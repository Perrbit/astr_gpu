"""AIR5 frozen-acoustic CFL and three-stage Jacobian-weighted RK components."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_properties_components import read as read_properties

ROOT=Path(__file__).resolve().parents[2]
SOURCES=["src/astr_precision.F90","src/chemistry_air5_data.F90","src/chemistry_core.F90",
         "src/chemistry_properties.F90","src/chemistry_compensation.F90","src/cfl_spectrum.F90"]
DEPENDENCIES=["src/chemistry_pure_vt_body.inc","src/chemistry_vibrational_difference_body.inc",
              "src_gpu/chemistry_core_gpu.cuf","src_gpu/chemistry_compensation_gpu.cuf",
              "src_gpu/chemistry_solver_gpu.cuf","src_gpu/cfl_gpu.cuf",
              "tests/gpu_backend/air5_transport_reference.F90"]

def run(command,out,log):
    with (out/log).open("w") as stream: subprocess.run(list(map(str,command)),cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)

def reference(path):
    with path.open("rb") as stream:
        n=int(np.fromfile(stream,"<i4",1)[0]);status=np.fromfile(stream,"<i4",2*n).reshape(n,2)
        data=np.fromfile(stream,"<f8",104*n).reshape(n,104);assert stream.read()==b""
    assert np.all(status[:,0]==0) and np.all(status[:,1]==1)
    return data

def ratio(actual,ref):
    assert actual.shape==ref.shape and np.isfinite(actual).all() and np.isfinite(ref).all()
    scale=np.maximum(1.,abs(ref).max(axis=1,keepdims=True))
    value=float(np.max(abs(actual-ref)/(1e-13*scale+5e-12*abs(ref))))
    assert value<=1.,value
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--build",required=True,type=Path)
    p.add_argument("--compiler",required=True,type=Path);p.add_argument("--properties-reference",required=True,type=Path)
    p.add_argument("--memcheck",action="store_true");args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);build=args.build.resolve()
    files=SOURCES+DEPENDENCIES+["src_gpu/chemistry_transport.cu","src_gpu/chemistry_transport.h","src_gpu/chemistry_transport.F90",
        "src_gpu/chemistry_properties_math.cuh","src_gpu/chemistry_compensation_math.cuh","src_gpu/chemistry_coupling.cu",
        "src_gpu/chemistry_properties.cu","src_gpu/device_state.h","src_gpu/device_state_internal.cuh",
        "src_gpu/fluid_explicit.cu","src_gpu/fluid_diagnostics.cu","tests/gpu_backend/air5_transport_probe.cu","src_gpu/CMakeLists.txt"]
    for name in files:
        target=out/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(__file__,out/"driver.py");shutil.copy2(build/"bin/astr_backend_air5_transport",out/"candidate")
    source=out/"source"
    body=(source/"src_gpu/chemistry_compensation_gpu.cuf").read_text()
    functions=[]
    for name in ["compensated_add_gpu","compensated_rk_gpu"]:
        first=body.index("  attributes(device) subroutine "+name+"(")
        last=body.index("\n  end subroutine",first);last=body.index("\n",last+1)
        functions.append(body[first:last])
    # Direct routine extraction: preserve original GPU scalar arithmetic, not a rewritten solver.
    (out/"frozen_compensation_gpu.F90").write_text("module frozen_compensation_gpu\nuse iso_fortran_env,only:real64\ncontains\n"+"\n".join(functions)+"\nend module\n")
    report=dict(scope="CFL and RK component equations; not CFD transport or physical CURVE acceptance",commands={},comparisons={})
    for backend in ["cpu","old_gpu"]:
        target=out/backend;target.mkdir()
        flags=[str(args.compiler.resolve()),"-O2","-Mpreprocess","-Mextend","-I"+str(source/"src")]
        sources=[source/name for name in SOURCES]
        if backend=="old_gpu":
            flags += ["-cuda","-gpu=rdc,maxregcount:128","-D_CUDA","-DASTR_AIR5_REFERENCE_GPU"]
            sources += [source/"src_gpu/chemistry_core_gpu.cuf",out/"frozen_compensation_gpu.F90"]
        sources += [source/"tests/gpu_backend/air5_transport_reference.F90"]
        commands=[];objects=[]
        for i,path in enumerate(sources):
            obj=target/(path.stem+".o");objects.append(str(obj));command=flags+["-c",str(path),"-o",str(obj)]
            commands.append(command);run(command,target,f"compile{i}.log")
        command=flags+objects+["-o",str(target/"probe")];commands.append(command);run(command,target,"link.log")
        report["commands"][backend]=commands
    _,old=read_properties(args.properties_reference.resolve()/"cpu_all.bin")
    n=60;dt=.003;x=np.zeros((n,65),dtype="<f8")
    q=old[np.arange(n)*5,:11].copy()
    for i in range(n):
        x[i,:11]=q[i];jac=.25+(i%17)/13.;x[i,44]=jac
        m=np.array([[1.+.01*i,.1,-.05],[.1,1.2+.005*i,.07],[-.05,.07,.9+.009*i]])
        x[i,45:54]=m.ravel(order="F")
        x[i,54:]=np.finfo(float).eps*q[i]*.125
        x[i,55]*=-1
        for stage,t in enumerate([0.,dt,.5*dt]):
            forcing=q[i]*(.125+.05*t-.01*t*t)
            x[i,11+11*stage:22+11*stage]=jac*forcing
    x[-2,:11]=x[-1,:11];x[-2,45:54]=x[-1,45:54]*10.;x[-1,45:54]=x[-2,45:54]
    # A deliberate exact maximum tie for all four wave-rate components.
    for i in [n-2,n-1]:
        for stage,t in enumerate([0.,dt,.5*dt]): x[i,11+11*stage:22+11*stage]=x[i,44]*x[i,:11]*(.125+.05*t-.01*t*t)
    with (out/"input.bin").open("wb") as stream:
        np.array([n],dtype="<i4").tofile(stream);np.array([dt],dtype="<f8").tofile(stream);x.tofile(stream)
    ids=np.arange(n);coord=np.stack((ids%5,(ids//5)%4,ids//20),axis=1)
    active=(coord[:,0]>=1)&(coord[:,0]<=3)&(coord[:,1]>=1)&(coord[:,2]<=1)
    key=(coord[:,2]+3)*19*21+(coord[:,1]+4)*21+coord[:,0]+5
    for comp in [0,1]:
        for backend in ["cpu","old_gpu"]:
            run([out/backend/"probe",out/"input.bin",out/f"{backend}{comp}.bin",str(comp)],out,f"{backend}{comp}.log")
        cpu=reference(out/f"cpu{comp}.bin");gpu=reference(out/f"old_gpu{comp}.bin")
        run([out/"candidate",out/"input.bin",out/f"cpp{comp}.bin",str(comp)],out,f"cpp{comp}.log")
        with (out/f"cpp{comp}.bin").open("rb") as stream:
            assert int(np.fromfile(stream,"<i4",1)[0])==n
            current=np.fromfile(stream,"<f8",104*n).reshape(n,104)
            rates=np.fromfile(stream,"<f8",4);keys=np.fromfile(stream,"<i8",4);profile=np.fromfile(stream,"<f8",4*19).reshape(19,4)
            assert stream.read()==b""
        comparisons={"old_gpu_cpu":ratio(gpu,cpu),"rk_cpp_cpu":ratio(current[active,:99],cpu[active,:99]),
                     "rk_cpp_old_gpu":ratio(current[active,:99],gpu[active,:99])}
        expected=cpu[:,99:103].max(axis=0)
        comparisons["cfl_maxima"]=ratio(rates[None,:],expected[None,:])
        expected_keys=[int(key[cpu[:,99+a]==expected[a]].min()) for a in range(4)]
        assert keys.tolist()==expected_keys
        for j in range(4):
            expected=cpu[coord[:,1]==j,99:103].max(axis=0)
            ratio(profile[4+j:5+j],expected[None,:])
        assert np.all(profile[:4]==0) and np.all(profile[8:]==0)
        true=x[:,:11]*(1.+.125*dt+.05*dt*dt/2.-.01*dt**3/3.)
        analytical=float(np.max(abs(current[active,22:33]-true[active])/np.maximum(1.,abs(true[active]))))
        assert analytical<=1e-12
        comparisons["quadratic_forcing_analytic_scaled_error"]=analytical
        report["comparisons"][comp]=comparisons
    if args.memcheck:
        sanitizer=shutil.which("compute-sanitizer")
        if sanitizer is None: raise RuntimeError("requested memory tool unavailable")
        run([sanitizer,"--tool","memcheck","--leak-check","full","--error-exitcode","77","--log-file",out/"memcheck.log",
             out/"candidate",out/"input.bin",out/"memcheck.bin","1"],out,"memcheck_stdout.log")
    report.update(status="component-equations-passed",active_nodes=int(active.sum()),nodes=n,
                  source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report["comparisons"],indent=2))

if __name__=="__main__":main()
