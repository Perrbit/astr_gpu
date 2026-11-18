"""Actual runtime-catalog M12 images through the original M12 session preset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from named_fixture import namelist
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_cpp_restart_gate import exact_state
from run_output_restart_validation import archive_schedule_payload
from run_native_precision_restart_gate import check_state

FINAL=Path('outdat/new/checkpoints/step000000000004')
UNLIMITED=(2**63-1)//4096  # Existing Fortran unlimited_budget_bytes sentinel.


def definitions():
    lines=(ROOT/'scripts/output/input.products.m12.short.example.jsonl').read_text().splitlines()
    records=[json.loads(line) for line in lines][1:]
    for record in records:
        record['schedule']['interval']=2
        if record['output']['channel']!='image':continue
        if record['kind']=='q_surface':
            record['display'].update(camera=[150.,100.,200.],focal_point=[40.,2.,45.],parallel_scale=65.)
        elif record['kind']=='streamlines':
            record['display'].update(camera=[1300.,350.,1000.],focal_point=[550.,10.,45.],parallel_scale=450.)
        else:record['display']['parallel_scale']=450.
    return records


def images(case,ranks=2,tail=False):
    pictures={};receipts={}
    for record in definitions():
        if record['output']['channel']!='image':continue
        frames=sorted((case/'outdat/new/render_products'/record['id']).glob('*/segment*/frame.step*.jpeg'))
        expected=[4] if tail else [0,2,4]
        if [int(p.stem.split('step')[1]) for p in frames]!=expected:raise AssertionError(('image clock',record['id'],frames))
        for frame in frames:
            if list(frame.parent.glob('*.eps')):raise AssertionError('JPEG only')
            pixels=np.asarray(Image.open(frame).convert('RGB'))
            if pixels.shape!=(600,800,3) or pixels.std()<1:raise AssertionError('M12 raster shape/blank')
            rows=[json.loads(p.read_text()) for p in sorted(frame.parent.glob(frame.stem+'.rank*.json'))]
            if len(rows)!=ranks or any(row['field_d2h_bytes'] or row['geometry_host_bytes'] for row in rows):
                raise AssertionError('M12 image producer host field/geometry staging')
            cells=sum(row['product']['local_cells'] for row in rows)
            # The early laminar state need not contain the prescribed Q=.001 surface.
            if record['kind']!='q_surface' and cells<=0:raise AssertionError(('empty non-Q product',record['id']))
            key=record['id']+'/'+frame.name;pictures[key]=pixels;receipts[key]=rows
    return pictures,receipts


def numeric(case,step=4):
    frames=list((case/'outdat/new/slice_products/numerical').glob(f'*/segment*/step{step:012d}.h5'))
    if len(frames)!=1:raise AssertionError(('numerical M12 slice',frames))
    result={}
    with h5py.File(frames[0]) as h:
        def collect(name,obj):
            if isinstance(obj,h5py.Dataset):result[name]=obj[...]
        h.visititems(collect)
    with h5py.File(frames[0].parent/'geometry.h5') as h:
        result['geometry/coordinates']=h['coordinates'][...]
    return result


def compare_numeric(case,reference,exact=False):
    actual,old=numeric(case),numeric(reference)
    if actual.keys()!=old.keys():raise AssertionError('M12 numerical slice identities')
    errors={}
    for key,value in actual.items():
        expected=old[key]
        if value.shape!=expected.shape or value.dtype!=expected.dtype:raise AssertionError(('slice layout',key))
        if exact:
            if value.tobytes()!=expected.tobytes():raise AssertionError(('restored slice words',key))
        elif value.dtype.kind=='f':
            errors[key]=max(compare(value.reshape(-1,1),expected.reshape(-1,1)))
        elif not np.array_equal(value,expected):raise AssertionError(('slice metadata',key))
    return errors


def independent_plane(case):
    observed=numeric(case)
    with h5py.File(case/FINAL/'state.h5') as h:
        q=np.stack([h[f'q{c:04d}'][12,:,:] for c in range(1,6)],axis=-1)
    rho=q[...,0];v=q[...,1:4]/rho[...,None]
    pressure=(q[...,4]-.5*rho*np.sum(v*v,axis=-1))/(1/(1.4-1))
    expected=dict(density=rho,velocity_x=v[...,0],velocity_y=v[...,1],velocity_z=v[...,2],pressure=pressure)
    errors={key:max(compare(observed[key].reshape(-1,1),value.reshape(-1,1))) for key,value in expected.items()}
    with h5py.File(case/'outdat/new/resources/geometry.h5') as h:
        coordinates=np.stack([h[f'q{c:04d}'][12,:,:] for c in range(1,4)],axis=-1)
    if coordinates.shape!=observed['geometry/coordinates'].shape or coordinates.tobytes()!=observed['geometry/coordinates'].tobytes():
        raise AssertionError('grid-index plane coordinate copies differ from static resources')
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','fixture','implementation','mpiexec','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--reference',type=Path)
    p.add_argument('--pipeline',choices=('standard-device','direct-device'),default='standard-device')
    p.add_argument('--postprocess-transport',choices=('pinned','device-aware'),default='pinned')
    p.add_argument('--solver-transport',choices=('pinned','device'),default='pinned')
    p.add_argument('--profile',choices=('m12_diagnostics','boundary_layer'),default='m12_diagnostics')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binary=root/'astr';shutil.copy2(a.binary,binary);binary.chmod(0o555)
    shutil.copy2(__file__,root/'runner.py')
    fixture=a.fixture.resolve();records=definitions()
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY='2,1,1',
        ASTR_INSITU_WRITE_EPS='0')
    env['ASTR_GPU_HALO_TRANSPORT']=a.solver_transport
    if a.postprocess_transport=='device-aware' or a.solver_transport=='device':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',UCX_MEMTYPE_CACHE='n')
    for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):env.pop(key,None)
    def run(name,steps=4,restore=None,observe=True):
        case=root/name;shutil.copytree(fixture/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True);(case/'outdat/render').mkdir()
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=.02),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if observe:
            (case/'datin/input.insitu').write_text(namelist(dict(insitu_run=dict(enabled=True,
                statistics=False,render=True,products=a.profile,streamline_seeds='bl-layered256' if a.profile=='m12_diagnostics' else 'bl-layered64',
                processing_backend='device',derivative_backend='gpu',postprocess_transport=a.postprocess_transport,
                rendering_pipeline=a.pipeline,host_budget_bytes=UNLIMITED,device_budget_bytes=UNLIMITED,device_reserve_bytes=0,
                implementation_path=str(a.implementation.resolve()),pipeline_file=str(ROOT/'scripts/insitu/device_render_pipeline.py'),
                output_directory='outdat/render',initial_frame=False,final_frame=False,
                product_ids=['q_surface.image','velocity_slice.image','instantaneous_streamlines.image'],
                product_modes=['steps','steps','steps'],product_steps=[2,2,2]))))
            execution['ASTR_INSITU_CONFIG']='datin/input.insitu'
        else:(case/'datin/input.insitu').unlink()
        (case/'input.products.jsonl').write_text('\n'.join(json.dumps(r) for r in
            [dict(catalog_schema='astr-insitu-products-v1'),*(records if observe and restore is None else [])])+'\n')
        with (case/'run.log').open('x') as log:
            subprocess.run([str(a.mpiexec.resolve()),'--bind-to','none','-n','2',str(binary),'run','datin/input.case'],
                cwd=case,env=execution,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'The job is done!' not in text:raise AssertionError('unfinished M12 render case')
        if observe:
            status=json.loads((case/'outdat/new/products.status.json').read_text())
            if len(status['products'])!=len(records) or any(r['state']!='active' for r in status['products']):raise AssertionError(status)
        return case
    full=run('full');pixels,receipts=images(full)
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),pipeline=a.pipeline,profile=a.profile,
        postprocess_transport=a.postprocess_transport,solver_transport=a.solver_transport,
        physical=check_state(full/FINAL/'state.h5','fp64'),flow={},image_count=len(pixels),receipts=receipts)
    report['numeric_layout']={key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in numeric(full).items()}
    report['independent_primary_plane']=independent_plane(full)
    for backend in ('cpu','gpu'):
        report['flow'][backend]=compare(checkpoint(full/FINAL/'state.h5'),checkpoint(fixture/backend/FINAL/'state.h5'))
        if not np.array_equal(logged_steps(full),logged_steps(fixture/backend)):raise AssertionError('frozen numerical clock')
    if a.freeze_only:
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555);print('actual original M12 catalog images frozen');return
    if a.reference is None:p.error('--reference required after freezing')
    if (full/'run.log').read_text().count('backend=cuda')!=2:raise AssertionError('actual new C++ backend')
    old,old_receipts=images(a.reference/'full')
    report['image_mean_absolute_difference']={key:float(np.mean(abs(value.astype(float)-old[key]))) for key,value in pixels.items()}
    report['numeric']=compare_numeric(full,a.reference/'full')
    quiet=run('quiet',observe=False)
    report['observation_isolation']=exact_state(full/FINAL/'state.h5',quiet/FINAL/'state.h5')
    prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    report['exact_restart']=exact_state(full/FINAL/'state.h5',tail/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        if (full/FINAL/name).read_bytes()!=(tail/FINAL/name).read_bytes():raise AssertionError('exact '+name)
    if archive_schedule_payload(full/FINAL/'archives.bin')!=archive_schedule_payload(tail/FINAL/'archives.bin'):
        raise AssertionError('exact product scheduling')
    resumed,_=images(tail,tail=True)
    if any(value.tobytes()!=pixels[key].tobytes() for key,value in resumed.items()):raise AssertionError('M12 restored image pixels')
    compare_numeric(tail,full,exact=True)
    report.update(restored_images=len(resumed),restored_pixels_exact=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('C++ M12 actual catalog images and restart complete')


if __name__=='__main__':main()
