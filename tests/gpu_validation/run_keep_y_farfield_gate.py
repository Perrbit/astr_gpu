"""Native existing geometric y-NSCBC/source coupling with KEEP and ME6."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from generate_curvilinear_tgv_grid import mapped_grid
from run_keep_simple_boundary_gate import prepare,frozen_rhs
from run_keep_periodic_gate import compare_q
from run_output_restart_validation import compare_fields

FINAL=Path('outdat/new/checkpoints/step000000000004')


def fields(case,uniform):
    x,y,z,_=mapped_grid(16,16,16,.15,'y-wavy')
    with h5py.File(case/'datin/grid.simple.h5','w') as grid:
        for name,values in zip(('x','y','z'),(x,y,z)):
            grid[name]=values.transpose(2,1,0)
    eta=np.linspace(0,1,17)[None,:,None]
    envelope=np.broadcast_to(eta**2,x.shape)
    temperature=np.ones_like(x) if uniform else 1+.01*envelope*np.sin(x)*np.cos(z)
    values=dict(ro=1/temperature,t=temperature)
    values['u1']=np.zeros_like(x) if uniform else .02*envelope*(1+.1*np.cos(x)*np.cos(z))
    values['u2']=np.zeros_like(x) if uniform else .003*envelope*np.sin(x)*np.cos(z)
    values['u3']=np.zeros_like(x) if uniform else .005*envelope*np.cos(x)*np.sin(z)
    with h5py.File(case/'datin/flowini3d.h5','w') as initial:
        for name,value in values.items():
            initial[name]=value.transpose(2,1,0)


def phase(case,name):
    path=case/f'outdat/rhs.{name}.step00000000.rk01.rank00000000.bin'
    with path.open('rb') as stream:
        im,jm,km,nq=np.fromfile(stream,np.int32,4)
        return np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F')


def incoming_projector(state,normal):
    rho=state[0];velocity=state[1:4]/rho
    pressure=.4*(state[4]-.5*np.dot(state[1:4],velocity));sound=np.sqrt(1.4*pressure/rho)
    un=np.dot(velocity,normal);tol=64*np.finfo(float).eps*max(sound,abs(un))
    dp=np.r_[.2*np.dot(velocity,velocity),-.4*velocity,.4]
    momentum=np.r_[-un,normal,0.]
    projectors=[]
    for sign in (-1,1):
        right=np.r_[1.,velocity+sign*sound*normal,(state[4]+pressure)/rho+sign*sound*un]
        projectors.append(np.outer(right,(dp+sign*sound*momentum)/(2*sound**2)))
    minus,plus=projectors
    incoming=minus if un-sound<-tol else np.zeros((5,5))
    if un+sound<-tol:
        incoming=incoming+plus
    if un<-tol:
        incoming=incoming+np.eye(5)-minus-plus
    return incoming,int(un-sound<-tol)+int(un+sound<-tol)+3*int(un<-tol)


def source_projection(case,uniform):
    snap=read_q_snapshot(case/'outdat/rhs.pre_rhs.step00000000.rk01.rank00000000.bin')
    im,jm,km,hm,nq=snap.header
    q=snap.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    q=q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
    source=phase(case,'full')-phase(case,'conv')
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        normal=np.stack([geometry[f'q{6+3*c:04d}'][:].transpose(2,1,0) for c in range(3)],axis=-1)[:,jm]
    normal/=np.linalg.norm(normal,axis=-1)[...,None]
    largest=0.;idempotency=0.;incoming_counts=set()
    for i in range(im+1):
        for k in range(km+1):
            incoming,count=incoming_projector(q[i,jm,k],normal[i,k])
            incoming_counts.add(count)
            idempotency=max(idempotency,float(np.max(abs(incoming@incoming-incoming))))
            largest=max(largest,float(np.max(abs(incoming@source[i,jm,k]))))
    maximum=float(np.max(abs(source[:,jm])))
    if largest>2e-10 or idempotency>2e-10:
        raise AssertionError(f'Geometric incoming viscous-source projection differs: {largest} {idempotency}')
    if not uniform and maximum<=1e-14:
        raise AssertionError('The nonuniform fixture did not exercise a nonzero retained viscous source')
    return dict(incoming_source_max_abs=largest,retained_source_max_abs=maximum,
                projector_idempotency_max_abs=idempotency,incoming_wave_counts=sorted(incoming_counts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--uniform',action='store_true')
    parser.add_argument('--axes',nargs='*',choices=['x','y','z'],default=['x','y','z'])
    parser.add_argument('--multi-axis',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_VALIDATION_RK_STATE='1')
    topologies=[(1,1,1)]
    for axis in args.axes:
        topology=[1,1,1];topology['xyz'.index(axis)]=2;topologies.append(tuple(topology))
    if args.multi_axis:
        topologies.append((2,2,1))
    report=dict(status='running',convection=args.convection,viscosity=args.viscosity,
                uniform=args.uniform,boundaries=[1,1,41,52,1,1],cases={})
    singles={}
    for topology in topologies:
        ranks=int(np.prod(topology));label=''.join(map(str,topology));runs={}
        for backend in ('cpu','gpu'):
            case=out/f'{backend}_{label}'
            prepare(case,'41y',backend,'643e','643e','off',convection=args.convection,
                    viscosity=args.viscosity,upper_nonreflecting=True)
            fields(case,args.uniform)
            local=dict(env,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)))
            if ranks==1:
                local.update(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0')
            executable=[str(getattr(args,backend).resolve())]
            if args.memcheck and backend=='gpu' and ranks==1:
                executable=[shutil.which('compute-sanitizer'),'--tool','memcheck','--error-exitcode','99',*executable]
            command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks),*executable,'run','datin/input.case']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
            if args.memcheck and backend=='gpu' and ranks==1 and 'ERROR SUMMARY: 0 errors' not in (case/'run.log').read_text():
                raise AssertionError('Missing successful GPU memcheck summary')
            if b'y52-nr-geom-source-v1' not in (case/FINAL/'control.bin').read_bytes():
                raise AssertionError('Geometric complete-source boundary identity missing')
            if ranks==1:
                singles[backend]=case
                report.setdefault('source_projection',{})[backend]=source_projection(case,args.uniform)
            else:
                report['cases'][f'{backend}_{label}_decomposition']=compare_q(singles[backend]/FINAL/'state.h5',case/FINAL/'state.h5')
                seed=case/'outdat/new/checkpoints/step000000000002'
                resumed=out/f'{backend}_{label}_exact'
                prepare(resumed,'41y',backend,'643e','643e','off',convection=args.convection,
                    viscosity=args.viscosity,upper_nonreflecting=True,restore=seed)
                with (resumed/'run.log').open('w') as log:
                    subprocess.run(command,cwd=resumed,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
                compare_fields(case/FINAL/'state.h5',resumed/FINAL/'state.h5')
                for name in ('control.bin','archives.bin','insitu_control.bin'):
                    if (case/FINAL/name).read_bytes()!=(resumed/FINAL/name).read_bytes():
                        raise AssertionError(f'Exact curved y-NSCBC continuation changed {name}')
                report['cases'][f'{backend}_{label}_exact']='all state datasets and control files identical'
            runs[backend]=case
        report['cases'][f'cpu_gpu_{label}']=compare_q(runs['cpu']/FINAL/'state.h5',runs['gpu']/FINAL/'state.h5')
        if ranks==1:
            report['stages']=frozen_rhs(runs['cpu'],runs['gpu'],[1])
        if args.uniform:
            for backend,case in runs.items():
                with h5py.File(case/FINAL/'state.h5') as state:
                    expected=(1.,0.,0.,0.,1/(1.4*.3**2*.4))
                    error=max(float(np.max(abs(state[f'q{c+1:04d}'][:]-value))) for c,value in enumerate(expected))
                if error>2e-10:
                    raise AssertionError(f'Stagnant physical state changed: {backend} {error}')
                report.setdefault('uniform_error',{})[backend+'_'+label]=error
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(args.convection,args.viscosity,topology,'compared',flush=True)
    report['status']='passed_four_step_explicit_y_nonreflecting_coupling'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
