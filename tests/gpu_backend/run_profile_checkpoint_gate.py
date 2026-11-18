"""Real static-profile archives, same-build continuation and MP revision identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

from run_cpp_tgv_gate import namelist,logged_steps,checkpoint,compare
from run_cpp_restart_gate import exact_state

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from run_native_precision_restart_gate import check_state
from run_output_restart_validation import archive_schedule_payload


def replace_value(text,key,value):
    text,count=re.subn(r'(?m)^(\s*'+re.escape(key)+r'\s*=)[^\n]*',lambda m:m[1]+' '+value,text)
    if count!=1:raise AssertionError('expected one fixture assignment: '+key)
    return text


def flags(path):
    with path.open('rb') as f:
        f.read(8);return np.fromfile(f,np.int64,8)


def frame(case,step):
    found=list((case/'outdat/new/fields').glob(f'*/step{step:012d}/data.h5'))
    if len(found)!=1:raise AssertionError('one complete-step volume required')
    return found[0]


def archive_phase(case,precision):
    clocks=logged_steps(case);width=4 if precision=='fp32' else 8
    dtype=np.dtype('float32' if width==4 else 'float64')
    word=np.int32 if width==4 else np.int64
    names=('density','velocity_x','velocity_y','velocity_z','pressure','temperature')
    for step in (2,4):
        saved=case/f'outdat/new/checkpoints/step{step:012d}/state.h5'
        with h5py.File(saved) as authority,h5py.File(frame(case,step)) as data:
            if not np.array_equal(data['metadata'][2:5],clocks[step-1,2:5].astype(dtype).view(word).astype(np.int64)):
                raise AssertionError('volume complete-step clock words')
            for index,name in enumerate(names,6):
                x,y=data[name][...],authority[f'q{index:04d}'][...]
                if x.dtype!=dtype or x.tobytes()!=y.tobytes():raise AssertionError('volume/cache phase: '+name)
            vector=np.stack([data['velocity_'+d][...] for d in 'xyz'],axis=-1)
            if data['velocity'].dtype!=dtype or data['velocity'][...].tobytes()!=vector.tobytes():
                raise AssertionError('native volume vector/component words')
        for axis,index in (('i',16),('j',0),('k',8)):
            files=list((case/'outdat/new/slice_products').glob(f'plane-{axis}/*/segment*/step{step:012d}.h5'))
            if len(files)!=1:raise AssertionError('slice frame identity')
            selection=[slice(None)]*3;selection[2-'ijk'.index(axis)]=index
            with h5py.File(files[0]) as data,h5py.File(saved) as authority:
                if data['clock'][0]!=clocks[step-1,2]:raise AssertionError('slice complete-step clock')
                for c,name in enumerate(names,6):
                    x,y=data[name][...],authority[f'q{c:04d}'][tuple(selection)]
                    if x.dtype!=dtype or x.tobytes()!=y.tobytes():raise AssertionError('slice/cache phase: '+name)
    for axis in 'ijk':
        found=sorted(p.name for p in (case/'outdat/new/slice_products').glob(f'plane-{axis}/*/segment*/step*.h5'))
        if found!=[f'step{step:012d}.h5' for step in (0,2,4)]:raise AssertionError('initial/complete slice inventory')
    return dict(volume_steps=[2,4],slice_steps=[0,2,4],fields=list(names),real_bytes=width)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','template','mpiexec','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    p.add_argument('--implementation',choices=('cpp','cuda-fortran','cpu'),default='cpp')
    p.add_argument('--mp',choices=('off','flux'),default='off')
    p.add_argument('--filter',action='store_true')
    p.add_argument('--explicit-top',action='store_true',help='existing543e/643e, bc11/21/41/52 static-profile subset')
    p.add_argument('--automatic',action='store_true')
    p.add_argument('--ranks',type=int,choices=(1,2),default=1)
    p.add_argument('--axis',choices=('x','y','z'),default='x')
    p.add_argument('--transport',choices=('pinned','device'),default='pinned')
    p.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    p.add_argument('--legacy-binary',type=Path)
    p.add_argument('--nonexact-probe',action='store_true')
    p.add_argument('--reuse-main',action='store_true',help='reuse completed main cases; execute explicitly requested unrun continuation probes')
    a=p.parse_args();root=a.output.resolve()
    if not a.reuse_main:
        root.mkdir(parents=True,exist_ok=False)
        shutil.copy2(a.binary,root/'tested_astr');shutil.copy2(__file__,root/'runner.py')
    if a.legacy_binary:
        if a.mp!='flux' or a.precision!='fp64' or a.ranks!=1:raise AssertionError('legacy test uses the actual FP64 NP1 flux reference')
        if not a.reuse_main:shutil.copy2(a.legacy_binary,root/'legacy_astr')
    topology=[1,1,1];topology['xyz'.index(a.axis)]=a.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ucx' if a.transport=='device' else 'ob1',OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='1' if a.transport=='device' else '0',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_HALO_TRANSPORT=a.transport,
        ASTR_GPU_SYNC_MODE=a.sync,ASTR_GPU_COMPACT_SYNC=a.sync,ASTR_GPU_COMPACT_TRANSPORT='device-aware' if a.transport=='device' else 'pinned',
        ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    definitions=[]
    for axis,index in (('i',16),('j',0),('k',8)):
        definitions.append(dict(id='plane-'+axis,kind='slice',enable_request=1,source='instantaneous',
            fields=['density','velocity_x','velocity_y','velocity_z','pressure','temperature'],
            sampling=dict(mode='grid_index',axis=axis,index=index,
                index_ranges={other:[0,16] for other in 'ijk' if other!=axis}),
            output=dict(channel='data',format='hdf5-xdmf'),schedule=dict(mode='steps',interval=2)))

    def run(name,steps=4,restore=None,exact=True,writing=True,binary=None,reject=False,mp=None):
        case=root/name
        if a.reuse_main and name in ('continuous','no-output','prefix','resumed'):return case
        shutil.copytree(a.template,case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        text=(case/'datin/input.case').read_text()
        text=replace_value(text,'use_gpu','.false.' if a.implementation=='cpu' else '.true.')
        text=replace_value(text,'gpu_mp_positivity',repr(a.mp if mp is None else mp))
        text=replace_value(text,'lfilter','.true.' if a.filter else '.false.')
        if a.explicit_top:
            text=replace_value(text,'conschm',"'543e'");text=replace_value(text,'difschm',"'643e'")
            text=replace_value(text,'recon_schem','3');text=replace_value(text,'lchardecomp','.false.')
            text=replace_value(text,'bctype','11,21,41,52,1,1')
        (case/'datin/input.case').write_text(text)
        timestep=(dict(mode='auto',cfl_target=.5,initial_dt=1e-6,max_dt=2e-6,growth_factor=1.1)
                  if a.automatic else dict(mode='fixed',fixed_dt=1e-6))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=timestep,logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.insitu').write_text('&insitu_run enabled=f /\n')
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=exact,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=writing,mode='steps',interval_steps=2,initial_frame=False),slices=dict(enabled=False))))
        records=definitions if writing else []
        (case/'input.products.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in
            [dict(catalog_schema='astr-insitu-products-v1'),*records]))
        with (case/'run.log').open('w') as log:
            result=subprocess.run([str(a.mpiexec.resolve()),'--bind-to','none','-n',str(a.ranks),
                str(binary or root/'tested_astr'),'run'],cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if reject:
            log=(case/'run.log').read_text()
            if result.returncode==0 or 'numerical/controller contract mismatch' not in log or 'ASTR_TIMESTEP complete_step=' in log:
                raise AssertionError('expected numerical identity rejection before advancement')
        elif result.returncode:raise AssertionError('ASTR case failed: '+str(case))
        return case

    full=run('continuous');quiet=run('no-output',writing=False)
    snapshots=sorted((full/'outdat').glob('rhs.post_update.*.bin'))
    if len(snapshots)!=12*a.ranks:raise AssertionError('all RK state observations required')
    for file in snapshots:
        if file.read_bytes()!=(quiet/'outdat'/file.name).read_bytes():raise AssertionError('output changed full RK q/halo words: '+file.name)
    prefix=run('prefix',steps=2);source=prefix/'outdat/new/checkpoints/step000000000002'
    tail=run('resumed',restore=source)
    last=Path('outdat/new/checkpoints/step000000000004')
    count=exact_state(full/last/'state.h5',tail/last/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        if (full/last/name).read_bytes()!=(tail/last/name).read_bytes():raise AssertionError('exact profile control: '+name)
    if archive_schedule_payload(full/last/'archives.bin')!=archive_schedule_payload(tail/last/'archives.bin'):
        raise AssertionError('exact profile archive schedule')
    if not np.array_equal(logged_steps(tail,2,3),logged_steps(full)[2:]):raise AssertionError('exact profile clock')
    actual_flags=flags(full/last/'control.bin')
    if bool(actual_flags[6]&32)!=(a.mp=='flux') or bool(actual_flags[6]&2)!=(a.mp=='flux'):
        raise AssertionError('MP algorithm identity flags')
    report=dict(precision=a.precision,implementation=a.implementation,mp=a.mp,filter=a.filter,
        explicit_top=a.explicit_top,automatic=a.automatic,topology=topology,
        binary_sha256=hashlib.sha256((root/'tested_astr').read_bytes()).hexdigest(),
        exact_state_datasets=count,exact_controls=True,complete_RK_output_isolation=True,
        exact_control_files=['control.bin','insitu_control.bin'],archive_schedule_equal=True,
        physical=check_state(full/last/'state.h5',a.precision),control_flag7=int(actual_flags[6]),
        archive_phase=archive_phase(full,a.precision))
    exact_state(frame(full,4),frame(tail,4))
    for axis in 'ijk':
        left=list((full/'outdat/new/slice_products').glob(f'plane-{axis}/*/segment*/step000000000004.h5'))
        right=list((tail/'outdat/new/slice_products').glob(f'plane-{axis}/*/segment*/step000000000004.h5'))
        if len(left)!=1 or len(right)!=1:raise AssertionError('restored numerical slice identity')
        exact_state(left[0],right[0])
    if a.legacy_binary:
        previous=run('legacy-prefix',steps=2,binary=root/'legacy_astr')
        old_source=previous/'outdat/new/checkpoints/step000000000002';old=flags(old_source/'control.bin')
        if not old[6]&2 or old[6]&32:raise AssertionError('actual legacy MP checkpoint revision')
        run('legacy-exact-rejected',restore=old_source,reject=True)
        migrated=run('legacy-nonexact',restore=old_source,exact=False)
        migrated_flags=flags(migrated/last/'control.bin')
        if not migrated_flags[6]&32 or not migrated_flags[6]&2:raise AssertionError('migrated MP revision was not saved')
        error=compare(checkpoint(migrated/last/'state.h5'),checkpoint(full/last/'state.h5'))
        run('legacy-off-rejected',restore=old_source,exact=False,mp='off',reject=True)
        report['legacy']=dict(binary_sha256=hashlib.sha256((root/'legacy_astr').read_bytes()).hexdigest(),
            source_flag7=int(old[6]),exact_rejected=True,explicit_nonexact_scaled=error,MP_mode_toggle_still_rejected=True)
    if a.nonexact_probe:
        migrated=run('nonexact-from-full',restore=source,exact=False)
        light_prefix=run('light-prefix',steps=2,exact=False)
        light_source=light_prefix/'outdat/new/checkpoints/step000000000002'
        light_tail=run('light-resumed',restore=light_source,exact=False)
        probes={}
        for label,case in (('from_full',migrated),('from_light',light_tail)):
            physical=check_state(case/last/'state.h5',a.precision)
            with h5py.File(case/last/'state.h5') as f:
                if len([k for k in f if k.startswith('q')])!=5 or f['rank_extras'].size!=0:
                    raise AssertionError('light checkpoint retains caches/halo/shared padding')
            if not np.array_equal(logged_steps(case,2,3),logged_steps(full)[2:]):
                raise AssertionError('fixed-step nonexact profile clock')
            delta=compare(checkpoint(case/last/'state.h5').astype(np.float64),checkpoint(full/last/'state.h5').astype(np.float64),
                limit=None if a.precision=='fp32' else 2e-10)
            probes[label]=dict(physical=physical,conservative_scaled=delta,five_fields=True,no_rank_extras=True)
        report['nonexact_probes']=probes
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
