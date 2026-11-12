"""Existing mapped bc41 walls with explicit new forms; ASTR advances all states."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from generate_curvilinear_tgv_grid import mapped_grid
from run_keep_simple_boundary_gate import prepare,frozen_rhs,scaled
from run_keep_periodic_gate import compare_q
from run_output_restart_validation import compare_fields
from run_native_precision_curve_gate import read_gradients
from compare_q_validation_snapshots import read_q_snapshot
from run_keep_open_gate import wall_forcing_response

FINAL=Path('outdat/new/checkpoints/step000000000004')


def setup(case,backend,axis,convection,viscosity,uniform,filtering,restore=None,blowing='off'):
    prepare(case,'41'+axis,backend,'643e','643e',filtering,restore=restore,
            convection=convection,viscosity=viscosity,
            wall_blowing_mode=None if blowing=='off' else ('legacy_random' if blowing=='legacy_random' else 'current'))
    if restore is not None:
        return
    xyz=mapped_grid(16,16,16,.15,axis+'-wavy')[:3]
    logical=np.meshgrid(*([np.linspace(0,2*np.pi,17)]*3),indexing='ij')
    direction='xyz'.index(axis);tangent=[d for d in range(3) if d!=direction]
    profile=np.sin(logical[direction]/2)**2
    wave=np.cos(logical[tangent[0]])*np.cos(logical[tangent[1]])
    temperature=np.ones_like(profile) if uniform else 1+.001*profile*wave
    velocity=[np.zeros_like(profile) if uniform else .002*(d+1)*profile*(1+.1*wave)
              for d in range(3)]
    fields=dict(ro=1/temperature,t=temperature,**{f'u{d+1}':v for d,v in enumerate(velocity)})
    with h5py.File(case/'datin/grid.simple.h5','w') as grid:
        for name,value in zip('xyz',xyz):
            grid[name]=value.transpose(2,1,0)
    with h5py.File(case/'datin/flowini3d.h5','w') as initial:
        for name,value in fields.items():
            initial[name]=value.transpose(2,1,0)
    if blowing!='off':
        (case/'datin/wallbs.dat').write_text('Periodic mapped-wall fixture\namplitude beta xa xb xc\n'
            f'1.e-3 2.0 1.0 2.0 3.0\nnmod_t nmod_z\n{2 if blowing=="modal" else 0} 2\n')
        if blowing=='modal':
            (case/'datin/wallbs_phase.dat').write_text(' '.join(f'{v:.17e}' for v in np.linspace(.05,.75,15))+'\n')


def wall_targets(case,axis,blowing):
    direction='xyz'.index(axis);report={}
    for stage in range(1,4):
        snapshot=read_q_snapshot(case/f'outdat/rhs.pre_rhs.step00000000.rk{stage:02d}.rank00000000.bin')
        im,jm,km,hm,nq=snapshot.header
        q=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
        q=q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
        velocity=q[...,1:4]/q[...,0,None]
        pressure=.4*(q[...,4]-.5*np.sum(q[...,1:4]*velocity,axis=-1))
        temperature=1.4*.3**2*pressure/q[...,0]
        errors=[]
        for face,inside,second in ((0,1,2),(q.shape[direction]-1,q.shape[direction]-2,q.shape[direction]-3)):
            indices=[slice(None)]*3;indices[direction]=face;indices=tuple(indices)
            inner=[slice(None)]*3;inner[direction]=inside;inner=tuple(inner)
            next_inner=[slice(None)]*3;next_inner[direction]=second;next_inner=tuple(next_inner)
            if blowing!='off' and direction==1 and face==0:
                _,normals=read_gradients(case/f'outdat/rhs.gradients.step00000000.rk{stage:02d}.rank00000000.bin','fp64')
                normal=normals[3]
                speed=np.sum(velocity[indices]*normal,axis=-1)
                errors.append(scaled([velocity[indices],speed[...,None]*normal]))
                if stage==1:
                    errors.append(wall_forcing_response(case,speed,blowing,envelope=(1.,2.,3.)))
                    if np.max(abs(speed))<=1e-14:
                        raise AssertionError('Mapped-wall forcing did not produce a nonzero normal velocity')
            else:
                errors.append(scaled([velocity[indices],np.zeros_like(velocity[indices])]))
            errors.extend([
                           scaled([temperature[indices],np.ones_like(temperature[indices])]),
                           scaled([pressure[indices],(4*pressure[inner]-pressure[next_inner])/3])])
        report[str(stage)]=max(errors)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wall-axis',choices=['x','y','z'],default='y')
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--uniform',action='store_true')
    parser.add_argument('--filter',choices=['off','explicit10','compact10'],default='off')
    parser.add_argument('--workspace',choices=['scalar','full'],default='scalar')
    parser.add_argument('--blowing',choices=['off','counter','modal','legacy_random'],default='off')
    parser.add_argument('--axes',nargs='*',choices=['x','y','z'],default=['x','y','z'])
    parser.add_argument('--multi-axis',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args()
    if args.convection=='standard' and args.viscosity!='me6_opti':
        parser.error('The original-convection gate requires ME6 viscosity')
    if args.blowing!='off' and args.wall_axis!='y':
        parser.error('Existing generic wall forcing is a lower-y boundary feature')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
               ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_FILTER_WORKSPACE=args.workspace)
    topologies=[(1,1,1)]
    for axis in args.axes:
        topology=[1,1,1];topology['xyz'.index(axis)]=2;topologies.append(tuple(topology))
    if args.multi_axis:
        topologies.append((2,2,1))
    report=dict(status='running',wall_axis=args.wall_axis,convection=args.convection,viscosity=args.viscosity,
                intervals=[16]*3,steps=4,dt=1e-5,amplitude=.15,uniform=args.uniform,
                filter=args.filter,workspace=args.workspace,blowing=args.blowing,cases={})
    singles={}
    for topology in topologies:
        label=''.join(map(str,topology));ranks=int(np.prod(topology));runs={}
        for backend in ('cpu','gpu'):
            case=out/(backend+'_'+label)
            setup(case,backend,args.wall_axis,args.convection,args.viscosity,args.uniform,args.filter,blowing=args.blowing)
            local=dict(env,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)))
            if ranks==1:
                local.update(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
                             ASTR_VALIDATION_GRADIENTS='1')
            executable=[str(getattr(args,backend).resolve())]
            if args.memcheck and backend=='gpu' and ranks==1:
                executable=[shutil.which('compute-sanitizer'),'--tool','memcheck','--error-exitcode','99',*executable]
            command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks),*executable,'run','datin/input.case']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
            if args.memcheck and backend=='gpu' and ranks==1 and 'ERROR SUMMARY: 0 errors' not in (case/'run.log').read_text():
                raise AssertionError('Mapped-wall memcheck did not report zero errors')
            if args.blowing!='off' and not (case/'outdat/new/resources/wallbs.dat').is_file():
                raise AssertionError('Mapped-wall checkpoint did not freeze its forcing parameters')
            if args.blowing=='modal' and not (case/'outdat/new/resources/wallbs_phase.dat').is_file():
                raise AssertionError('Mapped-wall checkpoint did not freeze its forcing phases')
            if ranks==1:
                singles[backend]=case
                report.setdefault('wall_targets',{})[backend]=wall_targets(case,args.wall_axis,args.blowing)
            else:
                if args.blowing=='legacy_random':
                    report['cases'][backend+'_'+label+'_decomposition']='not a same-input comparison: rank-local forcing differs'
                else:
                    report['cases'][backend+'_'+label+'_decomposition']=compare_q(singles[backend]/FINAL/'state.h5',case/FINAL/'state.h5')
                resumed=out/(backend+'_'+label+'_exact')
                setup(resumed,backend,args.wall_axis,args.convection,args.viscosity,args.uniform,args.filter,
                      restore=case/'outdat/new/checkpoints/step000000000002',blowing=args.blowing)
                with (resumed/'run.log').open('w') as log:
                    subprocess.run(command,cwd=resumed,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
                compare_fields(case/FINAL/'state.h5',resumed/FINAL/'state.h5')
                for name in ('control.bin','archives.bin','insitu_control.bin'):
                    if (case/FINAL/name).read_bytes()!=(resumed/FINAL/name).read_bytes():
                        raise AssertionError(f'Mapped-wall exact continuation changed {name}')
                report['cases'][backend+'_'+label+'_exact']='all state/control values identical'
                if any((resumed/'datin'/name).exists() for name in ('wallbs.dat','wallbs_phase.dat','grid.simple.h5','flowini3d.h5')):
                    raise AssertionError('Mapped-wall restart retained an external forcing/grid/initial-state input')
            if args.uniform:
                with h5py.File(case/FINAL/'state.h5') as state:
                    expected=(1.,0.,0.,0.,1/(1.4*.3**2*.4))
                    error=max(float(np.max(abs(state[f'q{c+1:04d}'][:]-value))) for c,value in enumerate(expected))
                if error>2e-10:
                    raise AssertionError(f'Mapped-wall stationary gas changed: {error}')
                report.setdefault('uniform_error',{})[backend+'_'+label]=error
            runs[backend]=case
        report['cases']['cpu_gpu_'+label]=compare_q(runs['cpu']/FINAL/'state.h5',runs['gpu']/FINAL/'state.h5')
        if ranks==1:
            report['stages']=frozen_rhs(runs['cpu'],runs['gpu'],['xyz'.index(args.wall_axis)])
            report['gradients']={}
            for stage in range(1,4):
                values=[read_gradients(case/f'outdat/rhs.gradients.step00000000.rk{stage:02d}.rank00000000.bin','fp64')[0]
                        for case in (runs['cpu'],runs['gpu'])]
                report['gradients'][str(stage)]=scaled(values)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('mapped bc41',args.wall_axis,args.convection,args.viscosity,topology,'compared',flush=True)
    report['status']='passed_four_step_mapped_isothermal_wall'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
