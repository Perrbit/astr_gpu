"""Actual C++ physical-midspan wall products against frozen original GPU/CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import h5py
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from named_fixture import namelist
from test_runtime_catalog_sbli_walls import definition,frames,reference
from test_runtime_catalog_walls import rows
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_cpp_restart_gate import exact_state
from run_output_restart_validation import archive_schedule_payload
from run_native_precision_restart_gate import check_state

FIELDS=('p_pinf','skin_friction','wall_heat_into_wall_normalized')
FINAL=Path('outdat/new/checkpoints/step000000000004')


def curve_values(case,tail=False):
    result={}
    for field in FIELDS:
        paths=frames(case,field)
        if [int(p.stem.split('step')[1]) for p in paths]!=([4] if tail else [0,2,4]):raise AssertionError('midspan curve clock')
        for path in paths:
            data=np.asarray([[float(row[k]) for k in ('step','time','global_i','x','y','z','value')] for row in rows(path)])
            if data.shape!=(65,7) or not np.isfinite(data).all():raise AssertionError('midspan curve shape/finiteness')
            if not np.array_equal(data[:,2],np.arange(65)):raise AssertionError('midspan unique x coverage')
            if 'physical_midspan_interpolation; no spanwise average' not in path.read_text():raise AssertionError('midspan quantity definition')
            result[field+'/'+path.name]=data
    return result


def pictures(case,tail=False):
    result={}
    for field in FIELDS:
        paths=frames(case,field+'-picture',True)
        if any(list(p.parent.glob('*.eps')) for p in paths):raise AssertionError('only JPEG was requested')
        if [int(p.stem.split('step')[1]) for p in paths]!=([4] if tail else [0,2,4]):raise AssertionError('midspan image clock')
        for path in paths:
            data=np.asarray(Image.open(path).convert('RGB'))
            if data.shape!=(600,1000,3) or data.std()<1:raise AssertionError('midspan chart raster')
            receipt=json.loads(path.with_name(path.stem+'.rank00000000.json').read_text())
            numeric=next(p for p in frames(case,field) if p.name==path.with_suffix('.csv').name)
            values=np.asarray([float(row['value']) for row in rows(numeric)])
            if not np.array_equal(receipt['product']['actual_range'],[values.min(),values.max()]):raise AssertionError('midspan chart/CSV range')
            if receipt['product']['points']!=65:raise AssertionError('midspan chart selected x rows')
            result[field+'/'+path.name]=data
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','implementation','output','mpiexec'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ranks',type=int,choices=(1,2),default=1)
    p.add_argument('--transport',choices=('pinned','device-aware'),default='pinned',help='private postprocessing transport')
    p.add_argument('--pipeline',choices=('standard-device','direct-device'),default='standard-device')
    p.add_argument('--images',action='store_true');p.add_argument('--memcheck',action='store_true')
    p.add_argument('--reuse-completed-full',action='store_true',help='explicitly reuse a completed full run after an analysis-only correction')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=a.reuse_completed_full);ref=a.reference.resolve()
    binary=root/'cpp_astr'
    if a.reuse_completed_full:
        if hashlib.sha256(binary.read_bytes()).digest()!=hashlib.sha256(a.binary.read_bytes()).digest():raise AssertionError('reuse binary identity')
    else:
        shutil.copy2(a.binary,binary);binary.chmod(0o555)
    frozen=json.loads((ref/'reference.json').read_text());dt=frozen['clock'][0][3]
    with h5py.File(ref/'gpu/datin/grid.flatplate.h5') as geometry:
        zline=geometry['z'][:,0,0]
    mid=.5*(zline[0]+zline[-1]);first=int(np.searchsorted(zline,mid,side='left'))-1
    expected_fraction=float((mid-zline[first])/(zline[first+1]-zline[first]))
    records=[definition(f,f) for f in FIELDS]
    if a.images:records+=[definition(f+'-picture',f,True) for f in FIELDS]
    for record in records:record['schedule']['interval']=2
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_osc='pt2pt',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY=f'{a.ranks},1,1',
        ASTR_INSITU_WRITE_EPS='0')
    if a.transport=='device-aware':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',UCX_MEMTYPE_CACHE='n')
    def run(name,steps=4,restore=None,observe=True,instrument=False):
        case=root/name;shutil.copytree(ref/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        shutil.copy2(ref/'gpu/case_manifest.json',case/'case_manifest.json')
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=dt),logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if observe:
            (case/'outdat/render').mkdir()
            config=(ref/'gpu/datin/input.insitu').read_text()
            old_impl=re.search(r"implementation_path='([^']+)'",config)[1]
            config=config.replace(old_impl,str(a.implementation.resolve())).replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{a.pipeline}'")
            config=config.replace("postprocess_transport='pinned'",f"postprocess_transport='{a.transport}'")
            (case/'datin/input.insitu').write_text(config)
            (case/'input.products.jsonl').write_text('\n'.join(json.dumps(r) for r in
                [{'catalog_schema':'astr-insitu-products-v1'},*records])+'\n')
            execution['ASTR_INSITU_CONFIG']='datin/input.insitu'
        else:(case/'datin/input.insitu').unlink()
        command=[str(a.mpiexec.resolve()),'-n',str(a.ranks)]
        if instrument:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
            '--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command += [str(binary),'run','datin/input.case']
        with (case/'run.log').open('x') as log:subprocess.run(command,cwd=case,env=execution,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if text.count('backend=cuda')!=a.ranks or 'The job is done!' not in text:raise AssertionError('actual C++ run identity')
        if instrument:
            logs=list(case.glob('memcheck.*.log'))
            if len(logs)!=a.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):raise AssertionError('main memory check')
        return case
    if a.reuse_completed_full:
        full=root/'full'
        if 'The job is done!' not in (full/'run.log').read_text():raise AssertionError('the explicitly reused full run did not complete')
        shutil.copy2(ref/'gpu/case_manifest.json',full/'case_manifest.json')
    else:full=run('full',instrument=a.memcheck)
    actual=curve_values(full);old=curve_values(ref/'gpu')
    report=dict(ranks=a.ranks,transport=a.transport,pipeline=a.pipeline,images=a.images,
        physical=check_state(full/FINAL/'state.h5','fp64'),flow={},curves={},binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    for backend in ('cpu','gpu'):report['flow'][backend]=compare(checkpoint(full/FINAL/'state.h5'),checkpoint(ref/backend/FINAL/'state.h5'))
    for key,data in actual.items():
        if not np.array_equal(data[:,:3],old[key][:,:3]):raise AssertionError('midspan reference phase/index')
        report['curves'][key]=compare(data[:,3:],old[key][:,3:])
    x,z,expected=reference(full,4)
    report['independent']={field:compare(actual[field+'/frame.step000000000004.csv'][:,3:],
        np.column_stack([x,np.zeros_like(x),np.full_like(x,z),expected[:,c]])) for c,field in enumerate(FIELDS)}
    status=json.loads((full/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=len(records) or any(p['state']!='active' for p in status['products']):raise AssertionError(status)
    images=pictures(full) if a.images else {}
    quiet=run('quiet',observe=False);report['observation_isolation']=exact_state(full/FINAL/'state.h5',quiet/FINAL/'state.h5')
    prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    report['exact_restart']=exact_state(full/FINAL/'state.h5',tail/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        if (full/FINAL/name).read_bytes()!=(tail/FINAL/name).read_bytes():raise AssertionError('midspan exact control: '+name)
    if archive_schedule_payload(full/FINAL/'archives.bin')!=archive_schedule_payload(tail/FINAL/'archives.bin'):raise AssertionError('midspan schedule continuation')
    for key,value in curve_values(tail,True).items():
        if value.tobytes()!=actual[key].tobytes():raise AssertionError('midspan resumed curve words')
    if a.images:
        again=pictures(tail,True)
        if any(value.tobytes()!=images[key].tobytes() for key,value in again.items()):raise AssertionError('midspan resumed pixels')
        report.update(images_main=len(images),images_tail=len(again),pixels_exact=True)
    receipts=re.findall(r'ASTR_RUNTIME_CPP_SBLI_WALL_SOURCE rank=(\d+) step=(\d+) curve_d2h_bytes=(\d+) '
        r'private_d2h_bytes=(\d+) private_h2d_bytes=(\d+) mpi_sent_bytes=(\d+) fraction=\s*(\S+)',(full/'run.log').read_text())
    if len(receipts)!=3*a.ranks or any(int(row[2])!=4*(64//a.ranks+1)*8 or float(row[6])!=expected_fraction for row in receipts):
        raise AssertionError('one shared two-plane capture per event')
    if a.transport=='device-aware' and any(int(row[3]) or int(row[4]) for row in receipts):raise AssertionError('device path staged on host')
    report['transfer_receipts']=receipts;report['clock']=logged_steps(full).tolist()
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('actual C++ SBLI physical-midspan products and continuation complete')


if __name__=='__main__':main()
