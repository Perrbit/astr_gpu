"""Native ASTR fixed-time mesh self-convergence for the existing low-order wall."""
import argparse
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import h5py
import numpy as np

from insitu_cfl_gate import CflGate
from run_keep_periodic_gate import compare_q
from run_keep_wall_gate import write_wall_initial
from run_output_restart_validation import run_case

ROOT=Path(__file__).resolve().parents[2]
FAMILIES={'explicit_standard':('643e','standard'),
          'compact_standard':('643c','standard'),
          'explicit_me6':('643e','me6_opti')}


def state_at(case,steps):
    checkpoint=case/f'outdat/new/checkpoints/step{steps:012d}'
    control=(checkpoint/'control.bin').read_bytes()
    if control[:8]!=b'ASTROC07':
        raise AssertionError('Fixed-step wall convergence requires the FP64 ASTROC07 control layout')
    step,time,dt_used,dt_next=struct.unpack_from('<qddd',control,72)
    if step!=steps or abs(time-.01)>64*np.finfo(float).eps:
        raise AssertionError(f'Different complete-step physical time: {step}, {time}')
    with h5py.File(checkpoint/'state.h5') as state:
        q=np.stack([state[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,6)],axis=-1)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        xyz=np.stack([geometry[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,4)],axis=-1)
        jac=geometry['q0004'][:].transpose(2,1,0)
    if q.dtype!=np.float64 or not np.isfinite(q).all() or not np.isfinite(xyz).all() or not np.isfinite(jac).all() or not (jac>0).all():
        raise AssertionError('Nonfinite wall state or invalid measure')
    internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
    if not (q[...,0]>0).all() or not (internal>0).all():
        raise AssertionError('Nonpositive density or internal energy')
    return q,xyz,jac,dict(step=step,time=time,dt_used=dt_used,dt_next=dt_next,
                         rho_min=float(q[...,0].min()),internal_min=float(internal.min()))


def difference(coarse,fine):
    a,xyz,jac,_=coarse
    b,xyz_f,_,_=fine
    ratio=(b.shape[0]-1)//(a.shape[0]-1)
    b=b[::ratio,::ratio,::ratio,:]
    xyz_f=xyz_f[::ratio,::ratio,::ratio,:]
    if a.shape!=b.shape or np.max(abs(xyz-xyz_f))>2e-10:
        raise AssertionError('Mesh comparison does not use the same physical nodes')
    delta=abs(a-b)
    scaled=delta/np.maximum(1,abs(b))
    weights=jac[:-1,:,:-1].copy()
    weights[:,0,:]*=.5;weights[:,-1,:]*=.5
    measure=weights.sum()
    result={'coordinate_max_abs':float(np.max(abs(xyz-xyz_f))),
            'measure':float(measure),'absolute':{},'scaled':{}}
    for label,error in (('absolute',delta),('scaled',scaled)):
        physical=error[:-1,:,:-1,:]
        result[label]={'L1':(np.sum(weights[...,None]*physical,axis=(0,1,2))/measure).tolist(),
                       'L2':np.sqrt(np.sum(weights[...,None]*physical**2,axis=(0,1,2))/measure).tolist(),
                       'Linf':np.max(error,axis=(0,1,2)).tolist()}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    for n in (16,32,64,128):
        write_wall_initial(out/f'initial_n{n}.h5',n)
    report={'status':'running','scope':'fixed-time mesh self-convergence, not analytic error or a global sixth-order claim',
            'time':.01,'dt':1e-4,'steps':100,'filter':False,'crashfix':False,'ranks':2,'topology':[2,1,1],
            'quadrature':'stored J; unique periodic x/z nodes, half-weight physical y endpoints',
            'initial':'existing wall field in normalized computational coordinates',
            'families':{}}

    def run(backend,family,n,dt=1e-4):
        scheme,viscous=FAMILIES[family]
        steps=round(.01/dt)
        opts=SimpleNamespace(output=out,executable=getattr(args,backend).resolve(),mpiexec=args.mpiexec.resolve(),
            case='channel',mode='steps',restart_step=steps,initial_dimension=3,initial_restart=False,
            legacy_statistics=False,statistics=False,filter_workspace='scalar',force='fixed',axis='x',
            no_samples=True,wall_samples=False,unrestricted=True)
        runtime=f'''&run_control max_steps={steps} /
&timestep mode='fixed',fixed_dt={dt:.17e} /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
'''
        case,_=run_case(opts,ROOT,backend,2,f'{family}_n{n}_steps{steps}',steps,
            lfilter=False,scheme=scheme,numerics={'convection_form':'keep','conschm':scheme,
              'difschm':'643e','viscous_form':viscous},grid=f'{n},{n},{n}',
            initial_resource=out/f'initial_n{n}.h5',run_config_text=runtime,
            checkpoint_interval=steps,checkpoint_keep=2)
        cfl=CflGate(case/'run.log',.5,dt,0,steps);cfl.check(final=True)
        data=state_at(case,steps)
        if data[0].shape!=(n+1,n+1,n+1,5):
            raise AssertionError('Native checkpoint differs from the declared refinement mesh')
        if data[3]['dt_used']!=dt or data[3]['dt_next']!=dt:
            raise AssertionError('Wall convergence changed the prescribed timestep')
        print(backend,family,n,steps,'C_upper',cfl.maximum,flush=True)
        return case,data,dict(path=str(case),cfl_max=cfl.maximum,**data[3])

    for family in args.families:
        entry={'runs':{},'successive_differences':{},'finest_reference_differences':{},'apparent_orders':{}}
        report['families'][family]=entry
        data={};paths={}
        for n in (16,32,64,128):
            paths[n],data[n],entry['runs'][f'gpu_n{n}']=run('gpu',family,n)
            if n==32:
                cpu,_,entry['runs']['cpu_n32']=run('cpu',family,n)
                final=f'outdat/new/checkpoints/step{100:012d}/state.h5'
                entry['cpu_gpu_n32']=compare_q(cpu/final,paths[n]/final)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for n in (16,32,64):
            entry['successive_differences'][str(n)]=difference(data[n],data[2*n])
            entry['finest_reference_differences'][str(n)]=difference(data[n],data[128])
        for n in (16,32):
            entry['apparent_orders'][str(n)]={}
            for norm in ('L1','L2','Linf'):
                a=np.array(entry['successive_differences'][str(n)]['absolute'][norm])
                b=np.array(entry['successive_differences'][str(2*n)]['absolute'][norm])
                entry['apparent_orders'][str(n)][norm]=[float(np.log2(x/y)) if x>0 and y>0 else None for x,y in zip(a,b)]
        _,half,entry['runs']['gpu_n32_half_dt']=run('gpu',family,32,5e-5)
        entry['time_step_difference_n32']=difference(data[32],half)
        entry['status']='measured; apparent orders and temporal sensitivity must be interpreted, not assumed'
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='measured'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
