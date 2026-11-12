"""Real-solver low-order KEEP bc41 wall, same-phase RHS and restart checks."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_output_restart_validation import run_case, compare_fields
from run_keep_periodic_gate import compare_q

ROOT = Path(__file__).resolve().parents[2]
FINAL = Path('outdat/new/checkpoints/step000000000004')


def write_wall_initial(path,intervals=16):
    x,y,z=np.meshgrid(np.linspace(0,2*np.pi,intervals+1),np.linspace(0,np.pi,intervals+1),
                      np.linspace(0,2*np.pi,intervals+1),indexing='ij')
    profile=np.sin(y)
    fields={'ro':np.ones_like(x),'t':np.ones_like(x),
            'u1':0.05*profile*(1+0.1*np.cos(x)*np.cos(z)),
            'u2':0.002*np.sin(x)*profile**2,'u3':0.001*np.cos(z)*profile**2}
    for key in ('u1','u2','u3'):
        fields[key][:,0,:]=0;fields[key][:,-1,:]=0
    with h5py.File(path,'x') as state:
        for key,value in fields.items():
            state[key]=value.transpose(2,1,0)


def wall_rhs(first,second):
    suffix='.step00000000.rk01.rank00000000.bin'
    snapshots=[read_q_snapshot(path/'outdat'/('rhs.pre_rhs'+suffix)) for path in (first,second)]
    if snapshots[0].header!=snapshots[1].header:
        raise AssertionError('KEEP wall frozen layouts differ')
    im,jm,km,hm,nq=snapshots[0].header
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    indices=np.indices(shape)-hm
    sizes=np.array([im,jm,km])[:,None,None,None]
    outside=(indices<0)|(indices>sizes)
    mask=(outside.sum(axis=0)<=1)&((indices>=-3)&(indices<=sizes+3)).all(axis=0)
    mask&=(indices[1]>=0)&(indices[1]<=jm)
    q=[snapshot.values.reshape((*shape,nq),order='F')[mask] for snapshot in snapshots]
    if any(not np.isfinite(value).all() for value in q):
        raise AssertionError('nonfinite KEEP wall operator input')
    difference=float(np.max(abs(q[0]-q[1])/np.maximum(1,np.maximum(abs(q[0]),abs(q[1])))))
    if difference>2e-10:
        raise AssertionError(f'KEEP wall input states differ: {difference}')
    result={'q_bit_identical_on_valid_faces':np.array_equal(q[0],q[1]),'q_max_scaled':difference}
    primitives=[]
    for path in (first,second):
        with (path/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
            header=np.fromfile(stream,np.int32,4)
            data=np.fromfile(stream,np.float64)
        if tuple(header)!=(im,jm,km,hm):
            raise AssertionError('KEEP wall primitive layout differs')
        primitives.append(data.reshape((*shape,6),order='F')[mask])
    if any(not np.isfinite(value).all() for value in primitives):
        raise AssertionError('nonfinite KEEP wall primitive input')
    result['primitives_bit_identical']=np.array_equal(primitives[0],primitives[1])
    result['primitives_max_scaled']=float(np.max(abs(primitives[0]-primitives[1])/np.maximum(1,
        np.maximum(abs(primitives[0]),abs(primitives[1])))))
    if result['primitives_max_scaled']>2e-10:
        raise AssertionError('KEEP wall primitive inputs differ')
    consistency=[]
    for state,cache in zip(q,primitives):
        density=state[:,0]
        velocity=state[:,1:4]/density[:,None]
        pressure=0.4*(state[:,4]-0.5*np.sum(state[:,1:4]*velocity,axis=1))
        expected=np.column_stack((density,velocity,pressure,(1.4*0.3**2)*pressure/density))
        consistency.append(float(np.max(abs(cache-expected)/np.maximum(1,np.maximum(abs(cache),abs(expected))))))
    result['primitive_q_consistency']=consistency
    if max(consistency)>2e-10:
        raise AssertionError('KEEP wall cache does not correspond to its current q')
    compare_fields(first/'outdat/new/resources/geometry.h5',second/'outdat/new/resources/geometry.h5')
    for phase in ('conv','full'):
        arrays=[]
        for path in (first,second):
            with (path/'outdat'/('rhs.'+phase+suffix)).open('rb') as stream:
                header=np.fromfile(stream,np.int32,4)
                data=np.fromfile(stream,np.float64)
            if tuple(header)!=(im,jm,km,nq):
                raise AssertionError('KEEP wall RHS layout differs')
            arrays.append(data.reshape((im+1,jm+1,km+1,nq),order='F'))
        if any(not np.isfinite(value).all() for value in arrays):
            raise AssertionError('nonfinite KEEP wall RHS')
        a,b=arrays
        errors=np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b))),axis=(0,1,2))
        result[phase]=errors.tolist()
        if max(errors)>2e-10:
            raise AssertionError(f'KEEP wall {phase} RHS differs: {errors}')
    return result


def wall_state_budget(case,step,initial_snapshot):
    """J-weighted stored-state changes, not a boundary energy-stability proof."""
    snapshot=read_q_snapshot(initial_snapshot)
    im,jm,km,hm,nq=snapshot.header
    q0=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    q0=q0[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
    with h5py.File(case/'outdat/new/resources/geometry.h5') as state:
        jac=state['q0004'][:].transpose(2,1,0)
    checkpoint=case/f'outdat/new/checkpoints/step{step:012d}/state.h5'
    with h5py.File(checkpoint) as state:
        q1=np.stack([state[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,6)],axis=-1)
    if jac.shape!=q0.shape[:3] or q0.shape!=q1.shape:
        raise AssertionError('KEEP wall budget requires the same global five-variable mesh')
    if not np.isfinite(q0).all() or not np.isfinite(q1).all() or not (jac>0).all():
        raise AssertionError('invalid KEEP wall budget state or measure')
    weights=jac[:-1,:,:-1].copy()
    weights[:,0,:]*=0.5;weights[:,-1,:]*=0.5
    measure=float(weights.sum())
    means=[np.sum(weights[...,None]*q[:-1,:,:-1,:],axis=(0,1,2))/measure for q in (q0,q1)]
    internal=q1[...,4]-np.sum(q1[...,1:4]**2,axis=-1)/(2*q1[...,0])
    return dict(source=str(checkpoint),initial_snapshot=str(initial_snapshot),
        quadrature='unique periodic x/z nodes; trapezoidal y endpoints; stored J weights',
        interpretation='stored-state means include wall projection and forcing/heat effects; not zero-energy or exact-balance gates',
        measure=measure,initial_mean=means[0].tolist(),final_mean=means[1].tolist(),
        mean_change=(means[1]-means[0]).tolist(),rho_min=float(q1[...,0].min()),
        rho_max=float(q1[...,0].max()),internal_min=float(internal.min()),
        wall_normal_momentum_max=float(np.max(abs(q1[:,[0,-1],:,2]))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--axes',nargs='*',choices=['x','y','z'],default=['x','y','z'])
    parser.add_argument('--multi-axis',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--scheme',choices=['643e','643c'],default='643e')
    parser.add_argument('--diffusion-scheme',choices=['643e','643c'],default='643e')
    parser.add_argument('--filter',choices=['off','explicit10','compact10'],default='off')
    parser.add_argument('--timestep',choices=['fixed','auto'],default='fixed')
    parser.add_argument('--crashfix',action='store_true')
    parser.add_argument('--viscous-form',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--transport',choices=['pinned','device-aware'],default='pinned')
    opts=parser.parse_args()
    opts.output=opts.output.resolve();opts.output.mkdir(parents=True,exist_ok=False)
    initial=opts.output/'wall_initial.h5';write_wall_initial(initial)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    numerics={'convection_form':'keep','viscous_form':opts.viscous_form,'difschm':opts.diffusion_scheme}
    filtered=opts.filter!='off'
    if filtered:
        numerics['filter_scheme']=opts.filter
    wall_id='w24' if opts.scheme=='643e' else 'w34'
    timestep="mode='fixed',fixed_dt=1.d-4" if opts.timestep=='fixed' else "mode='auto',cfl_target=0.5d0,growth_factor=1.1d0"
    runtime=f"""&run_control max_steps=4 /
&timestep {timestep} /
&logging list_steps=1 /
&repair crashfix={'t' if opts.crashfix else 'f'} /
&statistics enabled=f /
"""
    report={'status':'running','cases':{},'rhs':{},'steps':4,'timestep':opts.timestep,
            'dt':1e-4 if opts.timestep=='fixed' else None,'crashfix':opts.crashfix,
            'initial_field':str(initial),'filter':opts.filter,'wall':'bc41, low-order '+wall_id,
            'scheme':opts.scheme,'diffusion_scheme':opts.diffusion_scheme,'viscous_form':opts.viscous_form,
            'transport':opts.transport,'budgets':{}}
    singles={}
    matrix=[(1,'x',None)]+[(2,a,None) for a in opts.axes]
    if opts.multi_axis:
        matrix.append((4,'xy',(2,2,1)))
    for ranks,axis,topology in matrix:
        runs={};finals={}
        for backend in ('cpu','gpu'):
            args=SimpleNamespace(output=opts.output,executable=getattr(opts,backend).resolve(),
                mpiexec=opts.mpiexec.resolve(),case='channel',mode='steps',restart_step=2,
                initial_dimension=3,initial_restart=False,legacy_statistics=False,statistics=False,
                filter_workspace='scalar',force='fixed',axis=axis[0],no_samples=True,wall_samples=False,unrestricted=True)
            case,_=run_case(args,ROOT,backend,ranks,'keep_wall_'+axis,4,lfilter=filtered,scheme=opts.scheme,
                numerics=numerics,initial_resource=initial,run_config_text=runtime,topology=topology,
                rhs_snapshot_step=0 if ranks==1 else None,memcheck=opts.memcheck and backend=='gpu',
                solver_halo_transport=opts.transport)
            if ('/'+wall_id).encode() not in (case/FINAL/'control.bin').read_bytes():
                raise AssertionError('KEEP low-order wall identity missing')
            if opts.viscous_form=='me6_opti' and b'/me6opti-g2w4-v1' not in (case/FINAL/'control.bin').read_bytes():
                raise AssertionError('ME6 low-order wall identity missing')
            if ('KEEP WALL '+wall_id+':') not in (case/'run.log').read_text():
                raise AssertionError('KEEP wall geometry path missing')
            runs[backend]=case;finals[backend]=case/FINAL/'state.h5'
            if ranks==1:
                singles[backend]=finals[backend]
            else:
                report['cases'][f'{backend}_np{ranks}_{axis}_vs_np1']=compare_q(singles[backend],finals[backend])
            if ranks==2 and axis=='x':
                seed=case/'outdat/new/checkpoints/step000000000002'
                resumed,_=run_case(args,ROOT,backend,ranks,'keep_wall_'+axis+'_resumed',4,restore=seed,
                    lfilter=filtered,scheme=opts.scheme,numerics=numerics,initial_resource=initial,run_config_text=runtime,
                    solver_halo_transport=opts.transport)
                compare_fields(case/FINAL/'state.h5',resumed/FINAL/'state.h5')
                for filename in ('control.bin','archives.bin','insitu_control.bin'):
                    if (case/FINAL/filename).read_bytes()!=(resumed/FINAL/filename).read_bytes():
                        raise AssertionError(f'KEEP wall exact restore differs: {backend} {filename}')
                report['cases'][f'{backend}_np2_x_exact']='state datasets and control files identical'
                if backend=='cpu':
                    changed={'convection_form':'standard','viscous_form':'standard','difschm':'643e'}
                    run_case(args,ROOT,backend,ranks,'wall_changed_exact',4,restore=seed,lfilter=False,scheme='643e',
                        numerics=changed,initial_resource=initial,run_config_text=runtime,reject='compact operator identity mismatch')
                    migrated,_=run_case(args,ROOT,backend,ranks,'wall_changed_nonexact',4,restore=seed,lfilter=False,
                        scheme='643e',numerics=changed,initial_resource=initial,run_config_text=runtime,exact_restart=False)
                    compare_q(migrated/FINAL/'state.h5',migrated/FINAL/'state.h5')
                    report['cases']['changed_to_standard_unfiltered']='exact rejected; explicit non-exact migration finite and positive'
                    if filtered:
                        alternate=dict(numerics,filter_scheme='explicit10' if opts.filter=='compact10' else 'compact10')
                        run_case(args,ROOT,backend,ranks,'wall_changed_filter_exact',4,restore=seed,lfilter=True,
                            scheme=opts.scheme,numerics=alternate,initial_resource=initial,run_config_text=runtime,
                            reject='compact operator identity mismatch')
                        report['cases']['changed_filter']='exact restore rejected different filter'
        report['cases'][f'cpu_gpu_np{ranks}_{axis}']=compare_q(finals['cpu'],finals['gpu'])
        if ranks==1:
            report['rhs']=wall_rhs(runs['cpu'],runs['gpu'])
            for backend in ('cpu','gpu'):
                source=runs[backend]/'outdat/rhs.initialized.step00000000.rk01.rank00000000.bin'
                report['budgets'][backend]=wall_state_budget(runs[backend],4,source)
        (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'KEEP wall NP={ranks} {axis}: fields compared',flush=True)
    report['status']='passed'
    (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
