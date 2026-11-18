"""Compare resident AIR5 convection limits to the original CUDA Fortran kernels."""
import argparse
import json
from math import prod
import os
from pathlib import Path

import numpy as np

from run_air5_filter_fields import command, fixture3
from run_air5_legacy_filter_controller import build_reference


def fixture(path, name, periodic):
    sensor_axis={"sensor-x":0,"sensor-z":2}.get(name)
    model, state, _=fixture3("smooth" if sensor_axis is not None else ("trace-step" if name=="mp-trace" else name))
    sizes=tuple(32 if d==sensor_axis else 12 for d in range(3)) if sensor_axis is not None else (16,12,10)
    index=[(np.arange(n+1)*32//n)%32 if p else np.minimum(np.arange(n+1)*32//n,31) for n,p in zip(sizes,periodic)]
    q=state[np.ix_(*index[::-1])]
    rho=q[...,0]
    u=q[...,1:4]/rho[...,None]
    if sensor_axis is not None:
        q[:]=state[0,0,0]
        u=q[...,1:4]/rho[...,None]
        old_kinetic=.5*rho*np.sum(u*u,axis=-1)
        s=np.indices(rho.shape)[2-sensor_axis]
        pulse=np.maximum(0.,1.-abs(s-16)/3.)
        u[...,sensor_axis]=450.-90.*pulse
        q[...,1:4]=rho[...,None]*u
        q[...,4]+=.5*rho*np.sum(u*u,axis=-1)-old_kinetic+(q[...,5:10]@model.cv_tr)*(500.*pulse)
    if name in ("trace-step","mp-trace"):
        # Carry the existing zero/trace jump across all three axes.
        u[...,0]=450.;u[...,1]=-70.;u[...,2]=90.
        q[...,1:4]=rho[...,None]*u
        q[...,4]+=.5*rho*np.sum(u*u,axis=-1)
    if name=="mp-trace":
        # Preserve density, T and Tv while prescribing a mixed zero/trace
        # stencil of NO, compensated by the nitrogen mass.
        species=q[...,5:10]
        temperature=(q[...,4]-q[...,10]-np.sum(q[...,1:4]**2,axis=-1)/(2*rho)
                     -species@model.formation_energy)/(species@model.cv_tr)
        z,y,x=np.indices(rho.shape)
        z,y,x=(grid%n if p else grid for grid,n,p in zip((z,y,x),sizes[::-1],periodic[::-1]))
        level=(17*x+13*y+5*z+3*x*y+7*y*z+11*x*z) % 11
        abundance=np.where(level<3,0.,np.where(level<7,1.e-19,1.e-8))
        change=rho*abundance-q[...,8]
        vibrational=model.species_vibrational_energy(3,2000.)-model.species_vibrational_energy(0,2000.)
        q[...,5]-=change;q[...,8]+=change
        q[...,10]+=change*vibrational
        q[...,4]+=change*(model.formation_energy[3]-model.formation_energy[0]
                         +(model.cv_tr[3]-model.cv_tr[0])*temperature+vibrational)
    species=q[...,5:10]
    temperature=(q[...,4]-q[...,10]-np.sum(q[...,1:4]**2,axis=-1)/(2*rho)-species@model.formation_energy)/(species@model.cv_tr)
    pressure=temperature*(species@model.gas_constant)
    floor=[model.species_vibrational_energy(s,300.) for s in range(5)]
    fields=np.concatenate([q,u,pressure[...,None],species/rho[...,None]],axis=-1).transpose(3,0,1,2)
    with path.open("wb") as f:
        np.r_[sizes,periodic].astype("<i4").tofile(f)
        np.asarray(floor,dtype="<f8").tofile(f)
        fields.astype("<f8").tofile(f)


def read(prefix, rank):
    file=Path(f"{prefix}.rank{rank}.bin")
    with file.open("rb") as f:
        header=np.fromfile(f,"<i4",19)
        nx,ny,nz=header[:3]
        minima=np.fromfile(f,"<f8",10)
        ratios=np.fromfile(f,"<f8",2*prod(header[:3]+11))
        rhs=np.fromfile(f,"<f8").reshape(2,11,(nx+1)*(ny+1)*(nz+1))
    assert np.isfinite(rhs).all() and np.isfinite(ratios).all()
    assert np.all((ratios>=0.) & (ratios<=1.))
    return header,minima,ratios,rhs


def error(got, reference, scale):
    return float(np.max(abs(got-reference)/(1.e-13*scale+5.e-12*abs(reference))))


def compare(old, new, np_, test, require_limit=False):
    maxima=np.zeros(4)
    statuses=[]
    minimum_ratio=1.
    for rank in range(np_):
        a,amin,ar,arhs=read(old,rank)
        b,bmin,br,brhs=read(new,rank)
        assert np.array_equal(a,b),(old,new,rank,a,b)
        statuses.append(list(map(int,b[3:5])))
        minimum_ratio=min(minimum_ratio,float(br.min()))
        for rhs in (arhs,brhs):
            if test==3:
                assert np.array_equal(rhs[0].view("u8"),rhs[1].view("u8")),(new,rank,"failure wrote limited RHS")
        values=[error(br,ar,1.),error(bmin,amin,1.),*[
            error(y,x,np.maximum(1.,np.max(abs(x),axis=0))) for x,y in zip(arhs,brhs)]]
        assert max(values)<=1.,(old,new,rank,values)
        maxima=np.maximum(maxima,values)
    assert statuses==([[3,3],[0,3]] if test==3 else [[0,0]]*np_),statuses
    if test==2 or require_limit:
        assert minimum_ratio<1.,(new,"trace-step fixture does not activate the limiter")
    return dict(zip(("ratios","minimum","raw_rhs","limited_rhs"),map(float,maxima))) | {
        "status":statuses,"minimum_ratio":minimum_ratio}


def compare_sensor(old, new, ranks):
    def load(prefix, rank):
        with Path(f"{prefix}.sensor.rank{rank}.bin").open("rb") as f:
            extent=np.fromfile(f,"<i4",3)
            nodes=prod(extent+1)
            gradient=np.fromfile(f,"<f8",9*nodes).reshape(9,nodes)
            sensor=np.fromfile(f,"<f8",nodes)
            mask=np.fromfile(f,"i1",nodes)
            assert f.read(1)==b""
        assert np.isfinite(gradient).all() and np.isfinite(sensor).all()
        assert np.all((mask==0)|(mask==1))
        return gradient,sensor,mask

    maxima=np.zeros(2)
    counts=[]
    for rank in range(ranks):
        a,s,m=load(old,rank)
        b,t,n=load(new,rank)
        assert np.array_equal(m,n),(rank,int(np.count_nonzero(m!=n)))
        counts.append([int(m.size-m.sum()),int(m.sum())])
        e=[error(b,a,np.maximum(1.,abs(a).max(axis=1))[:,None]),error(t,s,1.)]
        assert max(e)<=1.,(rank,e)
        maxima=np.maximum(maxima,e)
    assert all(sum(c)>0 for c in zip(*counts)),counts
    return dict(gradient_error=float(maxima[0]),sensor_error=float(maxima[1]),sensor_mask_counts=counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("candidate","mpiexec","output"):
        parser.add_argument(f"--{arg}",required=True,type=Path)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-build",type=Path)
    source.add_argument("--frozen-reference",type=Path)
    parser.add_argument("--memcheck",type=Path)
    parser.add_argument("--sensor",action="store_true",help="use actual Ducros/pressure sensor instead of preset masks")
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True)
    public=[
        "configure_air5_vibrational_floor_gpu,air5_flow_status_d,reset_air5_status",
        "reset_air5_diffusion_failure,air5_diffusion_layered_d,air5_convection_consistent_d,air5_convection_symmetric_d",
        "air5_limited_points_d,air5_minimum_ratio_bits_d,air5_constraint_active_points_d,air5_constraint_minimum_ratio_bits_d",
        "air5_species_constraint_active_points_d,air5_species_constraint_minimum_ratio_bits_d,air5_diffusion_failure_location_d",
        "air5_zero_rhs_kernel,air5_full_state_convection_ratio_kernel,air5_limit_full_state_convection_kernel",
        "air5_reset_diffusion_ratio_kernel,air5_convective_rhs_x_kernel,air5_convective_rhs_y_kernel,air5_convective_rhs_z_kernel",
        "air5_selective_convective_rhs_x_kernel,air5_selective_convective_rhs_y_kernel,air5_selective_convective_rhs_z_kernel",
    ]
    reference=args.frozen_reference.resolve(strict=True) if args.frozen_reference else build_reference(
        args.reference_build.resolve(),out,"air5_convection_budget_probe.F90",public,["-DASTR_OLD_CONVECTION"])
    (out/"reference_identity.txt").write_text(str(reference)+"\n")
    exe=args.candidate.resolve()
    env=os.environ|dict(OMPI_MCA_pml="ob1",OMPI_MCA_btl="self,vader,tcp",OMPI_MCA_osc="pt2pt",
                        OMPI_MCA_coll="^cuda,hcoll,ucc",OMPI_MCA_opal_cuda_support="0")
    report={}
    cases=[(0,"smooth",(1,1,1),(1,1,1)),(1,"smooth",(0,0,1),(2,1,1)),
           (2,"trace-step",(1,1,1),(1,2,1)),(2,"trace-step",(0,0,1),(1,1,2)),
           (1,"mp-trace",(0,0,1),(1,1,2)),
           (3,"smooth",(0,0,1),(2,1,1))]
    if args.sensor:
        cases=[(4,"sensor-x",(0,0,1),(1,1,1)),(4,"sensor-x",(0,0,1),(2,1,1)),
               (4,"sensor-z",(0,0,1),(1,1,2))]
    for test,name,periodic,topology in cases:
        path=out/f"case{test}_{''.join(map(str,topology))}.bin";fixture(path,name,periodic)
        ranks=prod(topology)
        launch=[args.mpiexec.resolve(),"--bind-to","none","-np",ranks]
        for limiter in (0,1,2):
            key=f"p{test}_t{''.join(map(str,topology))}_l{limiter}"
            work=out/key;work.mkdir()
            old=work/"old"
            command([*launch,reference,path,old,*topology,limiter,test,0],work,"old.log",env)
            for mode in ((0,1) if ranks>1 else (0,)):
                new=work/f"new{mode}"
                runenv=env if not mode else env|dict(OMPI_MCA_pml="ucx",OMPI_MCA_opal_cuda_support="1")
                cmd=[*launch,exe,path,new,*topology,limiter,test,mode]
                command(cmd,work,f"new{mode}.log",runenv)
                sensor_report=compare_sensor(old,new,ranks) if args.sensor else {}
                report[f"{key}/{mode}"]=compare(old,new,ranks,test,name=="mp-trace")|sensor_report|dict(command=list(map(str,cmd)))
                if args.memcheck and test in (1,4) and topology==(2,1,1) and limiter==2 and mode==0:
                    observed=work/"memcheck";cmd[cmd.index(new)]=observed
                    sanitizer=[args.memcheck.resolve(),"--tool","memcheck","--leak-check","full","--error-exitcode","99"]
                    command([*launch,*sanitizer,*cmd[len(launch):]],work,"memcheck.log",env)
                    if args.sensor:
                        compare_sensor(old,observed,ranks)
                    compare(old,observed,ranks,test)
            (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
            print(key,"passed",flush=True)


if __name__=="__main__":
    main()
