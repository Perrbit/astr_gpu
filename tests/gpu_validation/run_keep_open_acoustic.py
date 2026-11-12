"""Execute the approved OPEN-P1 native KEEP acoustic window, stopping at its first failed gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter, sleep

import h5py
import numpy as np

from analyze_keep_open_acoustic import analyze
from compare_q_validation_snapshots import read_q_snapshot
from insitu_cfl_gate import CflGate
from named_fixture import prepare_named_run
from prepare_s1_flatplate_case import write_profile

ROOT=Path(__file__).resolve().parents[2]


def prepare(case,backend,mode,extended,pulse,fine,steps,snapshots):
    if case.exists():
        raise FileExistsError(case)
    ny=(192 if extended else 128)*(2 if fine else 1)
    height=1.5 if extended else 1.
    dt=5e-5 if fine else 1e-4
    subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/prepare_s1_flatplate_case.py'),
        '--dst-case',str(case),'--use-gpu','t' if backend=='gpu' else 'f',
        '--im','64','--jm',str(ny),'--km','16','--ninit','3',
        '--x-max-bctype','22','--upper-bctype','52','--reference-temperature','273.15',
        '--maxstep',str(max(1,steps-1)),'--deltat',str(dt)],check=True,capture_output=True,text=True)
    datin=case/'datin'
    yline=np.arange(ny+1)*(height/ny)
    x,y,z=np.meshgrid(np.linspace(0,10,65),yline,np.linspace(0,.25,17),indexing='ij')
    with h5py.File(datin/'grid.flatplate.h5','w') as grid:
        for name,value in zip(('x','y','z'),(x,y,z)):
            grid[name]=value.transpose(2,1,0)
    write_profile(datin/'inlet.prof',yline,.08,1.,False,False)
    smooth=lambda s:s**3*(10+s*(-15+6*s))
    envelope=smooth(np.clip(x-2,0,1))*smooth(np.clip(8-x,0,1))
    dy=y-.45
    packet=np.where(abs(dy)<.06,np.cos(np.pi*dy/(2*.06))**4,0.)
    gamma=1.4;a0=1/.3;p0=1/(gamma*.3**2)
    logpi=np.log1p((1e-4 if pulse else 0.)*envelope*packet/p0)
    fields=dict(ro=np.exp(logpi/gamma),t=np.exp((gamma-1)/gamma*logpi),
        u1=1-np.exp(-(y/.08)**2),u2=2*a0/(gamma-1)*np.expm1((gamma-1)/(2*gamma)*logpi),
        u3=np.zeros_like(y))
    with h5py.File(datin/'flowini3d.h5','x') as initial:
        for name,value in fields.items():
            initial[name]=value.transpose(2,1,0)
    boundary=dict(nscbc_farfield_mode=mode)
    if mode=='incoming_only':
        boundary.update(nscbc_farfield_rho=1.,nscbc_farfield_u=1.,nscbc_farfield_v=0.,
                        nscbc_farfield_w=0.,nscbc_farfield_t=1.)
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_')}
    primary=prepare_named_run(case,'datin/input.flatplate',env,
        numerics=dict(convection_form='keep',viscous_form='standard',conschm='643e',difschm='643e'),
        boundary_options=boundary)
    (datin/'input.run').write_text(f"""&run_control max_steps={steps} /
&timestep mode='fixed',fixed_dt={dt:.17e} /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
""")
    (case/'outdat/new').mkdir(parents=True)
    (datin/'input.output').write_text("""&output
 directory='outdat/new',exact_restart=t,checkpoint_statistics=f,
 host_budget_bytes=-1,device_budget_bytes=-1,buffer_bytes=67108864,device_reserve_bytes=0
/
&checkpoint
 enabled=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    if not snapshots:
        product=dict(id='acoustic-probe',kind='slice',enable_request=1,source='instantaneous',
            fields=['density','velocity_x','velocity_y','velocity_z','pressure','temperature'],
            sampling=dict(mode='grid_index',axis='j',index=224 if fine else 112,
                          index_ranges=dict(i=[29,35],k=[0,16])),
            output=dict(channel='data',format='hdf5-xdmf'),schedule=dict(mode='steps',interval=5))
        (case/'input.products.jsonl').write_text('\n'.join(json.dumps(record) for record in
            [dict(catalog_schema='astr-insitu-products-v1'),product])+'\n')
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_FORCE_MPI_TOPOLOGY='1,2,1',ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_VALIDATION_RK_STATE='1',ASTR_VALIDATION_PROFILE_STAGE_STATE='1')
    if snapshots:
        env.update(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
                   ASTR_VALIDATION_RHS_STEP_SECONDARY='3')
    return primary,env,dt


def run(case,backend,mode,extended,pulse,fine,steps,snapshots,args):
    primary,env,dt=prepare(case,backend,mode,extended,pulse,fine,steps,snapshots)
    command=[str(args.mpiexec),'--mca','coll_hcoll_enable','0','-np','2',
             str(getattr(args,backend)),'run',primary]
    recorded={key:value for key,value in env.items() if key.startswith(('ASTR_','OMPI_','OMP_'))}
    (case/'launch.json').write_text(json.dumps(dict(command=command,environment=recorded),indent=2)+'\n')
    cfl=CflGate(case/'run.log',.5,dt,0,steps)
    start=perf_counter()
    with (case/'run.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                cfl.check()
                sleep(.1)
            if process.returncode:
                raise RuntimeError(f'Native ASTR exited {process.returncode}: {case}/run.log')
            cfl.check(final=True)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
            process.wait()
            raise
    text=(case/'run.log').read_text(errors='replace')
    if 'KEEP OPEN:' not in text:
        raise AssertionError('Native KEEP open path was not selected')
    if backend=='gpu':
        seen={rank:[] for rank in range(2)}
        for line in text.splitlines():
            if line.startswith('ASTR_VALIDATION_RK_STATE_PASS '):
                _,rank,step=line.split();seen[int(rank)].append(int(step))
        if any(values!=list(range(steps)) for values in seen.values()):
            raise AssertionError('Missing complete-RK per-rank positive-state checks')
    result=dict(case=str(case),backend=backend,mode=mode,extended=extended,pulse=pulse,
                fine=fine,steps=steps,dt=dt,maximum_c_upper=cfl.maximum,elapsed_seconds=perf_counter()-start)
    if not snapshots:
        segments=list((case/'outdat/new/slice_products/acoustic-probe').glob('*/segment*'))
        if len(segments)!=1:
            raise AssertionError(f'Expected one numerical probe segment: {segments}')
        recorded_steps=sorted(int(path.stem[4:]) for path in segments[0].glob('step*.h5'))
        if recorded_steps!=list(range(0,steps+1,5)):
            raise AssertionError('Numerical probe lacks the approved complete-step sampling sequence')
        result['segment']=str(segments[0])
    print(json.dumps(result),flush=True)
    return result


def compare_short(cpu,gpu,output):
    report={}
    for step in (0,3):
        for stage in (1,2,3):
            for rank in (0,1):
                suffix=f'.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                for phase in ('pre_rhs','post_update','boundary','conv','full'):
                    arrays=[]
                    for case in (cpu,gpu):
                        path=case/'outdat'/('rhs.'+phase+suffix)
                        if phase in ('pre_rhs','post_update'):
                            snapshot=read_q_snapshot(path)
                            im,jm,km,hm,nq=snapshot.header
                            array=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
                            arrays.append(array[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1,:])
                        else:
                            with path.open('rb') as stream:
                                im,jm,km,nq=np.fromfile(stream,np.int32,4)
                                arrays.append(np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F'))
                    if phase=='boundary':
                        arrays[0]=-arrays[0]
                    a,b=arrays
                    if not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise AssertionError('Nonfinite four-step state or RHS')
                    error=float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
                    report[f'{step}/{stage}/{rank}/{phase}']=error
                    (output/'short_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
                    if error>2e-10:
                        raise AssertionError(f'Four-step CPU/GPU difference exceeds 2e-10: {step}/{stage}/{rank}/{phase}: {error}')
    return dict(max_scaled=max(report.values()),comparisons=len(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.cpu=args.cpu.resolve();args.gpu=args.gpu.resolve();args.mpiexec=args.mpiexec.absolute()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='approved OPEN-P1; explicit KEEP and standard viscosity; NP=2 y',runs=[],metrics={})
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def launch(label,backend='gpu',mode='nonreflecting',extended=False,pulse=False,fine=False,short=False):
        result=run(out/label,backend,mode,extended,pulse,fine,4 if short else (4600 if fine else 2300),short,args)
        report['runs'].append(result);save();return result
    save()
    try:
        cpu=launch('short_cpu',backend='cpu',pulse=True,short=True)
        gpu=launch('short_gpu',pulse=True,short=True)
        report['short']=compare_short(Path(cpu['case']),Path(gpu['case']),out);save()
        coarse=None
        for fine in (False,True):
            level='fine' if fine else 'coarse'
            reference=[launch(level+'_reference_'+kind,extended=True,pulse=pulse,fine=fine)
                       for kind,pulse in (('baseline',False),('pulse',True))]
            for mode in (('nonreflecting',) if fine else ('nonreflecting','compatibility','incoming_only')):
                pair=[launch(level+'_'+mode+'_'+kind,mode=mode,pulse=pulse,fine=fine)
                      for kind,pulse in (('baseline',False),('pulse',True))]
                factor=2 if fine else 1
                metrics=analyze([value['segment'] for value in pair+reference],1.,1/.3,
                    [1050*factor,1500*factor],[1800*factor,2250*factor],out/(level+'_'+mode+'_analysis'))
                report['metrics'][level+'_'+mode]=metrics;save()
                if metrics['incident_relative_difference']>.01:
                    raise AssertionError(f'{level} {mode}: incident integral differs by more than 1%')
                if mode=='nonreflecting':
                    if metrics['return_ratio']>.05:
                        raise AssertionError(f'{level} nonreflecting: return ratio exceeds 5%')
                    if fine and abs(metrics['return_ratio']-coarse)>.01:
                        raise AssertionError('Nonreflecting coarse/fine return ratio differs by more than 0.01')
                    coarse=metrics['return_ratio']
        report['status']='passed';save()
    except BaseException as error:
        report.update(status='stopped',failure=str(error));save();raise


if __name__=='__main__':
    main()
