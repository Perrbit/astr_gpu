"""Freeze AIR5 flux constraints and compare the C++ device implementation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from run_air5_properties_components import read as read_properties

ROOT=Path(__file__).resolve().parents[2]
SOURCES=["src/chemistry_air5_data.F90","src/chemistry_core.F90","src/chemistry_properties.F90"]
GPU_SOURCES=["src_gpu/chemistry_core_gpu.cuf","src_gpu/chemistry_flow_state_gpu.cuf"]
DEPENDENCIES=["src/chemistry_pure_vt_body.inc","src/chemistry_vibrational_difference_body.inc",
    "src/chemistry_solver.F90","src_gpu/chemistry_solver_gpu.cuf","tests/gpu_backend/air5_flux_reference.F90",
    "src_gpu/chemistry_flux_math.cuh","src_gpu/chemistry_properties_math.cuh","src_gpu/chemistry_air5_data.h",
    "tests/gpu_backend/air5_flux_math_probe.cu","tests/gpu_backend/CMakeLists.txt"]

def run(command,out,log):
    with (out/log).open("w") as stream:
        subprocess.run(list(map(str,command)),cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)

def extract(body,name):
    start=re.search(r"^  (?!end )[^!\n]*\b(?:function|subroutine) "+name+r"\(",body,re.M)
    assert start is not None,name
    end=re.search(r"^  end (?:function|subroutine)(?: "+name+r")?[ \t]*$",body[start.start():],re.M)
    assert end is not None,name
    return body[start.start():start.start()+end.end()]+"\n"

def build_references(out,compiler):
    commands={}
    base=["air5_vibrational_excess","air5_translational_margin","air5_state_is_admissible",
        "air5_state_constraint_mask","air5_face_state_is_admissible","air5_admissible_face_ratio","air5_symmetric_face_trial"]
    for backend in ["cpu","old_gpu"]:
        work=out/backend;work.mkdir();gpu=backend=="old_gpu"
        source=out/"source";original="src_gpu/chemistry_solver_gpu.cuf" if gpu else "src/chemistry_solver.F90"
        names=[name+"_gpu" if gpu else name for name in base]
        if gpu:names=["air5_interior_ratio_gpu","air5_transport_species_ratio_gpu"]+names
        (work/"frozen_flux_functions.inc").write_text("\n".join(extract((source/original).read_text(),name) for name in names))
        flags=[str(compiler),"-O2","-Mpreprocess","-Mextend","-I"+str(source/"src"),"-I"+str(work)]
        files=SOURCES.copy()
        if gpu:
            flags += ["-cuda","-gpu=rdc,maxregcount:128","-D_CUDA","-DASTR_AIR5_REFERENCE_GPU"]
            files += GPU_SOURCES
        files += ["tests/gpu_backend/air5_flux_reference.F90"]
        commands[backend]=[];objects=[]
        for name in files:
            obj=work/(Path(name).stem+".o");objects.append(obj)
            path=source/name
            # Preserve the original CUDA Fortran source mode. On NVHPC26.1,
            # -cuda alone on a .F90 wrapper miscompiles this predicate chain.
            if gpu and name.endswith("air5_flux_reference.F90"):
                path=work/"frozen_flux_reference.cuf";shutil.copy2(source/name,path)
            command=flags+["-c",path,"-o",obj];commands[backend].append(list(map(str,command)));run(command,work,obj.stem+".log")
        command=flags+objects+["-o",work/"probe"];commands[backend].append(list(map(str,command)));run(command,work,"link.log")
    return commands

def inputs(reference):
    _,properties=read_properties(reference/"cpu_all.bin")
    floor=properties[0,109:114].copy();gas=8.31446261815324/np.array([.028,.032,.014,.016,.030])
    cases=[];labels=[];known=[]
    def append(x,label,witness=None):
        x[70:75]=floor;cases.append(x);labels.append(label);known.append(witness)
    def projection(c,lo,hi,mass,weights=None,moment=0.,label="projection",witness=None):
        x=np.zeros(80);x[:5]=c;x[5:10]=lo;x[10:15]=hi;x[15]=mass
        if weights is not None:x[16:21]=weights;x[21]=moment;x[79]=1
        append(x,label,witness)
    high=np.array([.7,.3,0.,0.,-1e-12]);lo=np.zeros(5);hi=np.array([1.,1.,0.,0.,1e-10])
    for mode in [0,1]:
        for permutation in itertools.permutations(range(5)):
            p=np.array(permutation)
            projection(high[p],lo[p],hi[p],sum(high),gas[p] if mode else None,float(gas@high),"trace-permutation")
        for power in [-100,0,100]:
            for sign in [-1,1]:
                factor=2.**power
                projection(high*factor*sign,(lo if sign==1 else -hi)*factor,(hi if sign==1 else -lo)*factor,
                    sum(high)*factor*sign,gas if mode else None,float(gas@high)*factor*sign,"trace-scale-orientation")
    # Independent convex optimum: its KKT normal is a combination of the two
    # affine rows plus outward normals of explicitly active box constraints.
    rng=np.random.default_rng(851118)
    for mode in [0,1]:
        for i in range(40):
            witness=rng.uniform(-.3,.5,5);width=rng.uniform(.1,.6,5)
            lower=witness-width;upper=witness+width
            weights=gas/gas.max() if mode else np.zeros(5)
            candidate=witness+.13+.07*weights
            for s in range(5):
                if (s+i)%3==0:lower[s]=witness[s];candidate[s]-=.4
                elif (s+i)%3==1:upper[s]=witness[s];candidate[s]+=.4
            projection(candidate,lower,upper,float(sum(witness)),weights if mode else None,float(weights@witness),"kkt",witness)
    for mode in [0,1]:
        weights=gas if mode else None
        for values,bounds,target,moment,label in [
            (np.zeros(5),(np.zeros(5),np.zeros(5)),0.,0.,"zero"),
            (high,(lo,hi),3.,float(gas@high),"infeasible-mass"),
            (high,(lo,hi),float(sum(high)),1e3,"infeasible-moment"),
            (np.array([np.nan,0,0,0,0]),(lo,hi),1.,0.,"nonfinite")]:
            if mode==0 and label=="infeasible-moment":continue
            projection(values,*bounds,target,weights,moment,label)
    for weights,moment in [(np.zeros(5),0.),(np.zeros(5),1.),(np.ones(5)*2,2.*sum(high))]:
        projection(high,lo,hi,float(sum(high)),weights,float(moment),"degenerate")
    # Captured transverse and RK2 redistribution inputs from the existing
    # air5_transport_roundoff_probe, retained without replacing trace values.
    high=np.array([-6.6174449004242214e-24,-9.9261673506363321e-24,0.,0.,0.])
    witness=np.array([8.45181987420972325e-15,4.15262142530430207e-15,0.,0.,1.9831122966049362e-27])
    projection(high,high-witness/3e-8,high+witness/3e-8,-3.97046694025453284e-23,gas,float(gas@high),"captured-transverse")
    high=np.array([-4.1943372176048245e-23,-1.1888063246313867e-23,-6.078716994208774e-27,0.,0.])
    witness=np.array([1.5744425821605036e-10,1.5729792623244496e-11,-1.7317405083933405e-10,0.,0.])
    lo=witness-np.array([9.679070480478613e-15,2.74589500326727e-15,2.12002968377418e-18,0.,0.])/7.5e-9
    hi=witness+np.array([9.692948310003935e-15,2.7339443446462405e-15,2.1646756354916756e-19,0.,0.])/7.5e-9
    projection(high,lo,hi,-5.383751413935631e-23,gas,float(gas@high),"captured-rk2")
    for power in [-100,0,100]:
        for value in [0.,np.nextafter(0.,1.),.3,1.,2.]:
            x=np.zeros(80);x[79]=2;x[0]=value;x[1]=1e-38*2.**power;x[2]=-x[1];x[3]=16*x[1];append(x,"roundoff-ratio")
    # Use original valid, interior-temperature states, rather than synthesizing
    # an independent thermodynamic model for the coupled face trials.
    qrows=properties[175:191,:11]
    for index,q in enumerate(qrows):
        for component in [0,4,5,9,10]:
            for species_budget in [0,1]:
                x=np.zeros(80);x[:11]=q;x[11+component]=-abs(q[component])
                if q[component]==0:x[11+component]=-1e-10
                x[22:24]=[1,species_budget];x[79]=3;append(x,"face-ratio")
        for shape in range(6):
            low=q*180.;low[1]+=1e5;low[4]+=1e5*180.
            high=low.copy()
            if shape==1:high[9]-=q[0]*1e6;high[5]+=q[0]*1e6
            elif shape==2:high[4]+=abs(q[4])*1e6
            elif shape==3:high[10]+=abs(q[10])*1e6
            elif shape==4:high[0]-=q[0]*1e6
            elif shape==5:high[1]+=abs(q[1])*1e6
            for left,right in [(1,1),(0,1),(1,0)]:
                x=np.zeros(80);x[:11]=high;x[11:22]=low;x[22:33]=q*left;x[33:44]=q*right
                x[44:55]=q;x[55:66]=q;x[66:69]=[1e-6,left,right];x[79]=4
                append(x,f"symmetric-{shape}-{left}-{right}")
    q=qrows[0]
    for kind in ["density","nonfinite","species-base","energy-base"]:
        x=np.zeros(80);x[:11]=q*180.;x[1]+=1e5;x[4]+=1e5*180.;x[11:22]=x[:11]
        x[22:33]=q;x[33:44]=q;x[44:55]=q;x[55:66]=q;x[66:69]=[1e-6,1,1];x[79]=4
        if kind=="density":x[44]=-1.
        elif kind=="nonfinite":x[48]=np.nan
        elif kind=="species-base":x[27]=-abs(q[5])
        else:x[26]=-1e12
        append(x,"face-failure-"+kind)
    return np.array(cases,dtype="<f8"),labels,known

def read(path):
    with path.open("rb") as stream:
        n=int(np.fromfile(stream,"<i4",1)[0]);status=np.fromfile(stream,"<i4",n)
        data=np.fromfile(stream,"<f8",17*n).reshape(n,17);assert stream.read()==b""
    return status,data

def compare(a,b,status,x,labels,operations=range(5)):
    report={}
    for op in operations:
        active=(x[:,79]==op)&(status==0);aa=a[active];bb=b[active]
        if op==3:assert np.array_equal(aa[:,1:3],bb[:,1:3])
        scale=np.maximum(1.,abs(bb).max(axis=1,keepdims=True))
        ratio=abs(aa-bb)/(1e-13*scale+5e-12*abs(bb))
        maximum=float(ratio.max());report[str(op)]=maximum
        if maximum>1:
            row,col=np.unravel_index(np.argmax(ratio),ratio.shape);index=np.flatnonzero(active)[row]
            raise AssertionError((op,index,labels[index],col,aa[row,col],bb[row,col],maximum))
    return report

def check_flux_constraints(x,status,data,labels,known):
    eps=np.finfo(float).eps;maximum=0.
    for i in np.flatnonzero((x[:,79]<=1)&(status==0)):
        f=data[i,:5];candidate=x[i,:5];lo=x[i,5:10];hi=x[i,10:15]
        assert np.all(f>=lo)&np.all(f<=hi),(i,labels[i])
        magnitude=max(abs(x[i,15]),max(abs(candidate)),max(abs(lo)),max(abs(hi)))
        if magnitude==0:
            assert np.all(f==0)
            continue
        # Higher-precision residual evaluation is independent of the device's
        # summation order; retain the original backward-error budgets.
        scale=np.longdouble(magnitude);cf=candidate.astype(np.longdouble)/scale;ff=f.astype(np.longdouble)/scale
        mass=np.longdouble(x[i,15])/scale;weights=x[i,16:21];sw=max(abs(weights))
        moment_mode=x[i,79]==1 and sw!=0
        if not moment_mode:
            limit=32*eps*max(abs(mass),np.sum(abs(cf)))
            residual=abs(np.sum(ff)-mass)
            assert residual<=limit,(i,labels[i],float(residual),float(limit))
            maximum=max(maximum,float(residual/limit) if limit else 0.)
        else:
            g=weights.astype(np.longdouble)/np.longdouble(sw)
            moment=(np.longdouble(x[i,21])/np.longdouble(sw))/scale
            tolerance=64*eps*max(abs(mass),abs(moment),np.sum(abs(cf)))
            mass_limit=tolerance+64*eps*np.sum(abs(ff));moment_limit=tolerance+64*eps*np.sum(abs(g*ff))
            residual=abs(np.sum(ff)-mass);moment_residual=abs(np.sum(g*ff)-moment)
            assert residual<=mass_limit and moment_residual<=moment_limit,(i,labels[i],float(residual),float(moment_residual))
            maximum=max(maximum,float(residual/mass_limit) if mass_limit else 0.,float(moment_residual/moment_limit) if moment_limit else 0.)
    for i,witness in enumerate(known):
        if witness is not None:
            assert status[i]==0 and np.max(abs(data[i,:5]-witness))<=64*eps*max(1.,np.max(abs(witness)))
    failed=(x[:,79]<=1)&(status!=0)
    assert np.array_equal(data[failed,:5].copy().view(np.uint64),x[failed,:5].copy().view(np.uint64))
    failed=(x[:,79]==4)&(status!=0)
    assert np.array_equal(data[failed,:11].copy().view(np.uint64),x[failed,:11].copy().view(np.uint64))
    assert np.all(data[failed,11]==-1)
    for i,label in enumerate(labels):
        if label.startswith("face-failure-"):assert status[i]!=0,(i,label)
    return dict(maximum_affine_residual_budget_ratio=maximum,independent_kkt_cases=sum(w is not None for w in known),
        preserved_failed_faces=int(failed.sum()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True);p.add_argument("--compiler",type=Path,required=True)
    p.add_argument("--properties-reference",type=Path,required=True);p.add_argument("--memcheck",action="store_true")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in SOURCES+GPU_SOURCES+DEPENDENCIES:
        target=out/"source"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    shutil.copy2(__file__,out/"driver.py");shutil.copy2(args.candidate,out/"candidate")
    report=dict(scope="local flux projection, energy budgets and face trials; not MPI or CFD",commands=build_references(out,args.compiler.resolve()))
    x,labels,known=inputs(args.properties_reference.resolve());report["cases"]=len(x)
    with (out/"input.bin").open("wb") as stream:
        np.array([len(x)],dtype="<i4").tofile(stream);x.tofile(stream)
    (out/"labels.json").write_text(json.dumps(labels)+"\n")
    report["comparisons"]={}
    for cache in [0,1]:
        results={}
        for backend in ["cpu","old_gpu","cpp"]:
            binary=out/"candidate" if backend=="cpp" else out/backend/"probe"
            path=out/f"{backend}{cache}.bin"
            run([binary,out/"input.bin",path,cache],out,f"{backend}{cache}.log");results[backend]=read(path)
        cpu,old,cpp=(results[b] for b in ["cpu","old_gpu","cpp"])
        assert np.array_equal(cpu[0],old[0]),[(i,labels[i],int(cpu[0][i]),int(old[0][i])) for i in np.flatnonzero(cpu[0]!=old[0])[:8]]
        assert np.array_equal(cpu[0],cpp[0]),[(i,labels[i],int(cpu[0][i]),int(cpp[0][i])) for i in np.flatnonzero(cpu[0]!=cpp[0])[:8]]
        report["comparisons"][cache]={"old_cpu":compare(old[1],cpu[1],cpu[0],x,labels),
            "cpp_cpu":compare(cpp[1],cpu[1],cpu[0],x,labels),"cpp_old":compare(cpp[1],old[1],old[0],x,labels)}
        report["comparisons"][cache]["independent_constraints"]=check_flux_constraints(x,cpp[0],cpp[1],labels,known)
        for op in range(5):
            ids=x[:,79]==op
            print(cache,op,dict(zip(*np.unique(cpp[0][ids],return_counts=True))))
    if args.memcheck:
        tool=shutil.which("compute-sanitizer")
        if tool is None:raise RuntimeError("requested memory tool unavailable")
        run([tool,"--tool","memcheck","--leak-check","full","--error-exitcode","77","--log-file",out/"memcheck.log",
            out/"candidate",out/"input.bin",out/"memcheck.bin",1],out,"memcheck_stdout.log")
    report.update(status="local-flux-components-passed",candidate_sha256=hashlib.sha256((out/"candidate").read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((out/"source"/name).read_bytes()).hexdigest() for name in SOURCES+GPU_SOURCES+DEPENDENCIES})
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report["comparisons"],indent=2))

if __name__=="__main__":main()
