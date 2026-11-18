"""Freeze existing AIR5 conservative-filter math; no CFD advancement in Python."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tests/gpu_validation"))
from test_air5_filter_predicate_probe import predicate_states,exact_margins,exact_pressure_margin
from test_air5_filter_flux_probe import fixture as periodic_fixture
from test_air5_filter_boundary_probe import fixture as boundary_fixture

BASE=["src/chemistry_air5_data.F90","src/chemistry_core.F90","src/chemistry_properties.F90"]
MATH=["src/chemistry_filter_math.F90","src/chemistry_filter_operator.F90"]
FILES=BASE+MATH+["src/chemistry_pure_vt_body.inc","src/chemistry_vibrational_difference_body.inc",
    "src_gpu/chemistry_core_gpu.cuf","src_gpu/chemistry_filter_math.cuh","src_gpu/chemistry_flux_math.cuh",
    "src_gpu/chemistry_properties_math.cuh","src_gpu/chemistry_air5_data.h","src/CMakeLists.txt",
    "tests/gpu_backend/air5_filter_math_probe.cu","tests/gpu_backend/air5_filter_reference.F90",
    "tests/gpu_backend/run_air5_filter_components.py","tests/gpu_backend/CMakeLists.txt",
    "tests/gpu_validation/test_air5_filter_predicate_probe.py","tests/gpu_validation/test_air5_filter_flux_probe.py",
    "tests/gpu_validation/test_air5_filter_boundary_probe.py","tests/gpu_validation/air5_radau_reference.py",
    "chemMech/air5_kimjo12.json"]

def run(command,out,log):
    with (out/log).open("w") as stream:
        subprocess.run(list(map(str,command)),cwd=out,stdout=stream,stderr=subprocess.STDOUT,check=True)

def build_references(out,compiler):
    commands={};source=out/"source"
    for backend in ["cpu","old_gpu"]:
        gpu=backend=="old_gpu";work=out/backend;work.mkdir();commands[backend]=[];objects=[]
        flags=[str(compiler),"-O2","-Mpreprocess","-Mextend","-I"+str(source/"src"),"-I"+str(work)]
        if gpu:flags += ["-cuda","-gpu=rdc,maxregcount:128","-D_CUDA","-DASTR_AIR5_REFERENCE_GPU"]
        files=BASE+(["src_gpu/chemistry_core_gpu.cuf"] if gpu else [])+MATH+["tests/gpu_backend/air5_filter_reference.F90"]
        for name in files:
            path=source/name;options=flags.copy()
            if name in MATH:
                options += ["-Kieee","-Mnofma"]
                if gpu:options += ["-gpu=nofma,noflushz"]
            if gpu and name.endswith("air5_filter_reference.F90"):
                path=work/"frozen_filter_reference.cuf";shutil.copy2(source/name,path)
            obj=work/(Path(name).stem+".o");objects.append(obj);command=options+["-c",path,"-o",obj]
            commands[backend].append(list(map(str,command)));run(command,work,obj.stem+".log")
        command=flags+objects+["-o",work/"probe"];commands[backend].append(list(map(str,command)));run(command,work,"link.log")
    return commands

def read(path):
    with path.open("rb") as stream:
        n=int(np.fromfile(stream,"<i4",1)[0]);status=np.fromfile(stream,"<i4",n)
        output=np.fromfile(stream,"<f8").reshape(n,48)
    return status,output

def write(path,rows):
    with path.open("wb") as stream:np.array([len(rows)],"<i4").tofile(stream);np.asarray(rows,"<f8").tofile(stream)

def compare(rows,labels,ref,candidate):
    rs,rv=ref;cs,cv=candidate
    if not np.array_equal(rs,cs):
        i=int(np.flatnonzero(rs!=cs)[0]);raise AssertionError((labels[i],"status",int(rs[i]),int(cs[i])))
    assert np.isfinite(rv).all() and np.isfinite(cv).all()
    difference=abs(cv-rv);per_row=np.max(difference,axis=1)
    if np.any(per_row>2e-10):
        i=int(np.argmax(per_row));raise AssertionError((labels[i],"original T3 absolute budget",float(per_row[i])))
    for i,x in enumerate(rows):
        columns={0:[0,1,2],1:[0],2:[0],6:[12]}.get(int(x[255]),[])
        assert np.array_equal(cv[i,columns],rv[i,columns]),(labels[i],"predicate")
    return dict(maximum_absolute_difference=float(np.max(difference)),statuses_exact=True,predicates_exact=True)

def basic_inputs():
    rows=[];labels=[];expected={}
    def add(x,label,truth=None):
        if truth is not None:expected[len(rows)]=truth
        rows.append(x);labels.append(label)
    model,q=periodic_fixture("constant");formation=model.formation_energy;pressure_floor=2500.
    names,predicates=predicate_states()
    for label,row in zip(names,predicates):
        x=np.zeros(256);x[:21]=row
        v,t=exact_margins(row);p=exact_pressure_margin(row[:11],formation,pressure_floor)
        add(x,label,{0:int(v>=0 and t>=0),1:int(p>=0)})
    base=np.array([1.,1.,0.,0.,-288135.5,.5,.5,0.,0.,0.,1.])
    rng=np.random.default_rng(2217)
    for i in range(67):
        high=base.copy()
        if i<3:
            if i:high[4]=np.nextafter(high[4],-np.inf if i==1 else np.inf)
        else:
            high[0]=rng.uniform(.001,.1);y=rng.random(5);y/=y.sum();high[5:10]=high[0]*y
            high[1:4]=high[0]*rng.uniform(-4000,4000,3);high[10]=rng.uniform(1,10000)
            high[4]=high[10]+high[5:10]@formation+high[1:4]@high[1:4]/(2*high[0])+pressure_floor
            high[4]=np.nextafter(high[4],np.inf if i%2 else -np.inf)
        x=np.zeros(256);x[:11]=high
        p=exact_pressure_margin(high,formation,pressure_floor);v,t=exact_margins(x[:21])
        add(x,f"pressure-{i}",{0:int(v>=0 and t>=0),1:int(p>=0)})
        carry=rng.uniform(-.5,.5,11)*abs(np.spacing(high))
        x=np.zeros(256);x[:11]=high;x[11:22]=carry;x[255]=1
        p=exact_pressure_margin(high,formation,pressure_floor,carry)
        add(x,f"represented-{i}",{0:int(p>=0)})
    for c in [0,1,4,5,10]:
        for sign in [-1,0,1]:
            x=np.zeros(256);x[:11]=base;x[11+c]=sign*abs(np.spacing(base[c]))/4.;x[255]=1
            p=exact_pressure_margin(base,formation,pressure_floor,x[11:22]);add(x,f"quarter-ulp-{c}-{sign}",{0:int(p>=0)})
    for i in range(80):
        lower=-rng.random(5);upper=rng.random(5);candidate=rng.uniform(-1,1,5)
        lower[i%5]=upper[i%5]=0.
        raw=[sum(lower),sum(upper),sum(candidate),np.nextafter(sum(upper),np.inf)][i%4]
        factor=2.**(-40 if i%2 else 0)
        x=np.zeros(256);x[:5]=candidate*factor;x[5:10]=lower*factor;x[10:15]=upper*factor;x[15]=raw*factor;x[255]=3
        add(x,f"mass-bound-{i}")
    return np.array(rows),labels,expected,model

def high_inputs():
    rows=[];labels=[];groups=[];fixtures=[]
    for periodic in [True,False]:
        n=32;nodes=32 if periodic else 33
        for degree in [0,4,5,9,10]:
            q=np.array([float(Fraction(i,32)**degree) for i in range(nodes)])
            for compensated in [False,True]:
                carry=.25*abs(np.spacing(q)) if compensated else np.zeros_like(q)
                indices=[]
                for g in range(nodes):
                    stencil=np.zeros((10,11));low=np.zeros_like(stencil)
                    for layer in range(-4,6):
                        i=g+layer
                        if periodic:i%=nodes
                        elif i<1 or i>n-1:continue
                        stencil[layer+4,1]=q[i];low[layer+4,1]=carry[i]
                    x=np.zeros(256);x[:110]=stencil.reshape(-1);x[110:220]=low.reshape(-1);x[220:224]=[g,n,periodic,compensated];x[255]=4
                    indices.append(len(rows));rows.append(x);labels.append(f"polynomial-{periodic}-{degree}-{compensated}-{g}")
                groups.append((indices,q,carry,periodic,degree))
        names=["constant","smooth","zero","trace","trace-step","cancellation","energy","vibration","density"]
        if periodic:names.append("pressure-floor")
        for name in names:
            model,q=(periodic_fixture if periodic else boundary_fixture)(name)
            tr=model.formation_energy+300.*model.cv_tr;v=model.gas_constant*model.theta_v/np.expm1(np.where(model.theta_v>0,model.theta_v/300.,1.))
            for compensated in [False,True]:
                carry=.25*abs(np.spacing(q)) if compensated else np.zeros_like(q)
                for g in range(0 if periodic else 1,nodes if periodic else n-1):
                    stencil=np.zeros((10,11));low=np.zeros_like(stencil)
                    for layer in range(-4,6):
                        i=g+layer
                        if periodic:i%=nodes
                        elif i<1 or i>n-1:continue
                        stencil[layer+4]=q[i];low[layer+4]=carry[i]
                    x=np.zeros(256);x[:110]=stencil.reshape(-1);x[110:220]=low.reshape(-1);x[220:224]=[g,n,periodic,compensated];x[255]=4
                    fixtures.append((len(rows),q[g],q[(g+1)%nodes],carry[g],carry[(g+1)%nodes],tr,v,compensated))
                    rows.append(x);labels.append(f"fixture-{periodic}-{name}-{compensated}-{g}")
    return np.array(rows),labels,groups,fixtures

def verify_polynomials(output,groups):
    maximum=0.;null_cases=0
    for indices,q,carry,periodic,degree in groups:
        flux=output[indices,1];state=[Fraction(float(a))-Fraction(float(b)) for a,b in zip(q,carry)]
        if periodic:
            delta=np.array([float(sum(((-1)**k*math.comb(10,k))*state[(i+k-5)%len(q)] for k in range(11))/1024) for i in range(len(q))])
            actual=flux-np.roll(flux,1)
        else:
            d5=np.zeros((len(q)-7,len(q)-2),dtype=np.int64)
            for row in range(len(d5)):d5[row,row:row+6]=[-1,5,-10,10,-5,1]
            matrix=-(d5.T@d5);delta=np.zeros(len(q))
            for i in range(len(q)-2):delta[i+1]=float(sum(int(matrix[i,j])*state[j+1] for j in np.flatnonzero(matrix[i]))/1024)
            actual=np.zeros(len(q));actual[1:-1]=flux[1:-1]-flux[:-2]
        error=float(max(abs(actual-delta)));assert error<=2e-10,(periodic,degree,error);maximum=max(maximum,error)
        if not periodic and degree<=4 and not np.any(carry):assert np.all(actual==0);null_cases+=1
    return dict(maximum_independent_delta_error=maximum,exact_physical_null_polynomials=null_cases)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",required=True,type=Path);parser.add_argument("--compiler",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path);parser.add_argument("--memcheck",type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in FILES:
        dst=out/"source"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst);hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    candidate=out/"candidate";shutil.copy2(args.candidate.resolve(),candidate)
    report=dict(source_sha256=hashes,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                original_filter_absolute_budget=2e-10,commands=build_references(out,args.compiler),stages={})
    def stage(name,rows,labels,extra=None):
        path=out/(name+".bin");write(path,rows);products={}
        for backend,binary in [("cpu",out/"cpu/probe"),("old_gpu",out/"old_gpu/probe"),("cpp",candidate)]:
            result=out/(name+"_"+backend+".bin");run([binary,path,result],out,name+"_"+backend+".log");products[backend]=read(result)
        (out/(name+"_labels.json")).write_text(json.dumps(labels,indent=2)+"\n")
        value=dict(inputs=len(rows),comparisons={b:compare(rows,labels,products[b],products["cpp"]) for b in ["cpu","old_gpu"]})
        if extra:value["independent"]=extra(rows,products["cpp"])
        report["stages"][name]=value;(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,len(rows),value,flush=True);return products
    rows,labels,expected,model=basic_inputs()
    def exact(rows,product):
        status,values=product;assert np.all(status==0)
        for i,flags in expected.items():
            for c,truth in flags.items():assert values[i,c]==truth,(labels[i],c,values[i,c],truth)
        mass_cases=0
        for i,x in enumerate(rows):
            if int(x[255])!=3:continue
            mass=Fraction(float(values[i,0]));assert sum(map(lambda v:Fraction(float(v)),x[5:10]))<=mass<=sum(map(lambda v:Fraction(float(v)),x[10:15]));mass_cases+=1
        return dict(exact_rational_predicate_rows=len(expected),exact_mass_bound_rows=mass_cases)
    stage("basic",rows,labels,exact)
    rows,labels,groups,fixtures=high_inputs()
    high=stage("high",rows,labels,lambda x,p:verify_polynomials(p[1],groups))
    projection=[];projection_labels=[]
    for index,left,right,lc,rc,tr,v,compensated in fixtures:
        x=np.zeros(256);x[:11]=left-lc;x[11:22]=right-rc;x[22:33]=high["cpu"][1][index,:11];x[255]=5
        projection.append(x);projection_labels.append(labels[index])
    projected=stage("bounds",np.array(projection),projection_labels)
    assert np.all(projected["cpu"][0]==0),"original mass projection failed"
    thermal=[];thermal_labels=[];pairs=[]
    for row,(index,left,right,lc,rc,tr,v,compensated) in enumerate(fixtures):
        x=np.zeros(256);x[:11]=left-lc;x[11:22]=right-rc;x[22:33]=high["cpu"][1][index,:11]
        x[22]=projected["cpu"][1][row,10];x[27:32]=projected["cpu"][1][row,11:16]
        x[33:38]=tr;x[38:43]=v
        x[43:54]=np.nextafter(abs(left)+abs(lc),np.inf) if compensated else abs(left)
        x[54:65]=np.nextafter(abs(right)+abs(rc),np.inf) if compensated else abs(right)
        if compensated:
            x[43:54][(abs(left)+abs(lc))==0]=0.;x[54:65][(abs(right)+abs(rc))==0]=0.
        x[65]=compensated;x[255]=6;thermal.append(x);thermal_labels.append(labels[index])
        y=np.zeros(256);y[:11]=x[:11];y[11:22]=x[43:54];y[22:33]=x[22:33];y[33]=compensated;y[255]=2;pairs.append(y)
    stage("pairs",np.array(pairs),thermal_labels)
    limited=stage("thermal",np.array(thermal),thermal_labels)
    assert np.all(limited["cpu"][0]==0),"original thermal endpoint failed"
    frozen=limited["cpp"][1][:,11]==0
    assert np.all(limited["cpp"][1][frozen,:11]==0),"frozen face has a nonzero exchange"
    # The original update skips arithmetic when both incident fluxes are zero.
    # Its near-cone freeze need not satisfy the box for a hypothetical commit.
    assert np.all(limited["cpp"][1][~frozen,12]==1),"nonzero returned pair is not admissible"
    checked=0
    for x,answer in zip(thermal,limited["cpp"][1]):
        for state in [x[:11]+2.*answer[:11],x[11:22]-2.*answer[:11]]:
            assert state[0]>0 and np.all(state[5:10]>=0)
            vibration,thermal_margin=exact_margins(np.r_[state,x[33:43]])
            assert vibration>=0 and thermal_margin>=0
            assert exact_pressure_margin(state,model.formation_energy,2500.)>=0
            checked+=1
    report["exact_rational_returned_pair_states"]=checked
    ratios=limited["cpp"][1][:,11];report["thermal_counts"]={"zero":int(np.sum(ratios==0)),"limited":int(np.sum((ratios>0)&(ratios<1))),"one":int(np.sum(ratios==1))}
    failures=np.array([thermal[0].copy(),thermal[0].copy()]);failures[0,0]=-1.;failures[1,1]=np.nan
    rejected=stage("failures",failures,["negative-density-base","nonfinite-base"])
    for status,values in rejected.values():
        assert np.all(status==3) and np.array_equal(values[:,:11],failures[:,22:33]) and np.all(values[:,11]==1)
    if args.memcheck:
        combined=np.array(thermal)
        write(out/"memcheck_input.bin",combined)
        run([args.memcheck,"--tool","memcheck","--leak-check","full","--error-exitcode","99",candidate,out/"memcheck_input.bin",out/"memcheck_output.bin"],out,"memcheck.log")
        text=(out/"memcheck.log").read_text();assert "ERROR SUMMARY: 0 errors" in text and "LEAK SUMMARY: 0 bytes leaked" in text
        report["memcheck"]="thermal pair components: zero errors/leaks"
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":main()
