"""Existing wall material, sparse histories and profiles on ordinary C++ ASTR."""
import argparse
from copy import deepcopy
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from named_fixture import namelist
from test_insitu_wall_transport_runtime import wall_mean_definitions,check_wall_means
from test_index_slice_runtime import segments,frames
from test_insitu_station_profiles import read_profile
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_cpp_insitu_gate import compare_hdf
from test_runtime_catalog_images import image_product
from test_output_insitu_restart import BACKEND


def definitions():
    items=wall_mean_definitions()
    for item in items:
        item['sampling']['index_ranges']['i']=[0,16]
        span=item['sampling']['index_ranges']['k']
        item['sampling']['index_ranges']['k']=[n*2//3 for n in span]
    mean=deepcopy(items[0]);mean.update(id='mean-wall-plane',source='reynolds',statistics_source='selected-time')
    return items+[mean]


def inspect(case,items):
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=len(items) or any(p['state']!='active' for p in status['products']):
        raise AssertionError(('wall consumers not active',status))
    report={}
    wall_records=[items[0],*[p for p in items if p['kind']=='wall_profile']]
    check_wall_means(case,wall_records,lambda key,value: report.__setitem__(key,value),
        steps=4,dt=1e-5,window=(.5e-5,3.5e-5),ni=17,nk=16)
    return report


def image_definitions():
    result=[]
    for name,source in [('instant-wall-image','instantaneous'),('mean-wall-image','reynolds')]:
        item=image_product(name,index=0,field='wall_shear_x')
        item['source']=source
        if source=='reynolds':item['statistics_source']='selected-time'
        item['sampling']=dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,16],k=[0,16]))
        item['display'].update(camera=[np.pi,-13.,np.pi],focal_point=[np.pi,0.,np.pi],view_up=[0.,0.,1.],
            parallel_scale=4.,color_range=[0.,3e-9])
        result.append(item)
    return result


def inspect_images(case,ranks,tail=False):
    directories={p.parent.parent.name:p for p in (case/'outdat/new/render_products').glob('*/*/segment*')}
    if set(directories)!={'instant-wall-image','mean-wall-image'}:
        raise AssertionError(('wall image products',directories))
    result={}
    for name,directory in directories.items():
        steps=([4] if tail else [0,2,4]) if name=='instant-wall-image' else ([3] if tail else [1,3])
        files=sorted(directory.glob('frame.step*.jpeg'))
        if [int(p.stem.split('step')[1]) for p in files]!=steps or list(directory.glob('*.eps')):
            raise AssertionError(('wall image clock/EPS',name))
        receipts=[json.loads(p.read_text()) for p in directory.glob('frame.*.rank*.json')]
        if len(receipts)!=len(steps)*ranks or any(p['field_d2h_bytes'] or p['geometry_host_bytes'] for p in receipts):
            raise AssertionError('wall images read back device fields/geometry')
        for step in steps:
            if sum(p['product']['local_cells'] for p in receipts if p['step']==step)<=0:
                raise AssertionError('empty wall image geometry')
            path=directory/f'frame.step{step:012d}.jpeg'
            pixels=np.asarray(Image.open(path).convert('RGB'))
            if pixels.shape!=(600,800,3) or pixels.std()<1:
                raise AssertionError('wall image raster')
            result[name+f'/step{step}']=pixels
    return result


def numerical(case):
    return {p.parent.parent.name:p for p in segments(case)}


def profiles(case):
    return {p.parent.parent.name:p for p in (case/'outdat/new/wall_profile_products').glob('*/*/segment*')}


def transfer_usage(case,ranks,pml):
    text=(case/'run.log').read_text()
    counters=re.findall(r'ASTR_INSITU_SELECTED_SAMPLER rank=(\d+) kernel_calls=(\d+) status_d2h_bytes=(\d+) '
        r'index_h2d_bytes=(\d+) flow_d2h_bytes=(\d+)',text)
    if len(counters)!=ranks or sorted(int(row[0]) for row in counters)!=list(range(ranks)):
        raise AssertionError('missing selected sampler transfer records')
    if any(int(row[2])!=4*int(row[1]) or int(row[4])!=0 for row in counters):
        raise AssertionError('wall moment updates downloaded field values or unexpected status payloads')
    traffic=sum(int(value) for value in re.findall(r'ASTR_WALL_SAMPLE .*private_exchange_bytes=(\d+)',text))
    if pml=='ob1' and traffic!=0:
        raise AssertionError('the explicit host-MPI memory check must not exchange device packets')
    return dict(sampler_counters=[list(map(int,row)) for row in counters],private_exchange_bytes=traffic,
        flow_d2h_bytes=0,mpi_pml=pml)


def compare_products(actual,reference,exact=False):
    a,b=numerical(actual),numerical(reference)
    if set(a)!=set(b):
        raise AssertionError('wall numerical plane identities')
    report={}
    for name in a:
        selected=frames(a[name])
        expected=[p.name for p in frames(b[name]) if not exact or int(p.stem[4:])>2]
        if [p.name for p in selected]!=expected:
            raise AssertionError(('wall plane clock',name,selected,expected))
        report[name]=max(compare_hdf(p,b[name]/p.name,exact) for p in selected)
    a,b=profiles(actual),profiles(reference)
    if set(a)!=set(b):
        raise AssertionError('wall profile identities')
    for name in a:
        selected=sorted(a[name].glob('frame.step*.csv'))
        expected=[p.name for p in sorted(b[name].glob('frame.step*.csv')) if not exact or int(p.stem.split('step')[1])>2]
        if [p.name for p in selected]!=expected:
            raise AssertionError(('wall profile clock',name))
        error=0.
        for path in selected:
            if exact:
                if path.read_bytes()!=(b[name]/path.name).read_bytes():
                    raise AssertionError(('restored wall profile bytes',name))
            else:
                x,_=read_profile(path);y,_=read_profile(b[name]/path.name)
                if x.dtype.names!=y.dtype.names:raise AssertionError('wall profile columns')
                for field in x.dtype.names:
                    error=max(error,max(compare(x[field].reshape(-1,1),y[field].reshape(-1,1))))
        report[name]=error
    return report


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','reference-binaries','fixture','output','mpiexec'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--ranks',type=int,default=2)
    parser.add_argument('--axis',choices=['x','y','z'],default='y')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--mpi-pml',choices=['ucx','ob1'],default='ucx')
    parser.add_argument('--images',action='store_true')
    parser.add_argument('--renderer-reference',type=Path)
    opt=parser.parse_args()
    if opt.images and opt.renderer_reference is None:
        parser.error('--images requires the immutable Catalyst-enabled --renderer-reference')
    output=opt.output.resolve();output.mkdir(parents=True,exist_ok=False)
    topology=[1,1,1];topology['xyz'.index(opt.axis)]=opt.ranks
    items=definitions()
    if opt.images:items+=image_definitions()
    env=os.environ|dict(OMPI_MCA_pml=opt.mpi_pml,OMPI_MCA_opal_cuda_support='1' if opt.mpi_pml=='ucx' else '0',
        OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_INSITU_WRITE_EPS='0')
    def run(name,binary,backend,ranks,steps=4,restore=None,enabled=True):
        case=output/name;case.mkdir()
        shutil.copytree(opt.fixture.resolve()/(backend+'_y')/'datin',case/'datin');(case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(
            output=dict(directory='outdat/new',restore_directory=str(restore or ''),exact_restart=True,
                checkpoint_statistics=True,host_budget_bytes=-1,device_budget_bytes=-1,buffer_bytes=67108864,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        budget=np.iinfo(np.int64).max//4096
        options=dict(enabled=enabled,statistics=enabled,render=enabled and opt.images,
            products='catalog_stats' if enabled else 'all',statistics_window=[.5e-5,3.5e-5],output_directory='outdat',
            host_budget_bytes=int(budget),device_budget_bytes=int(budget),device_reserve_bytes=0)
        if opt.images and enabled:
            options.update(processing_backend='device',derivative_backend='gpu',postprocess_transport='pinned',
                rendering_pipeline='direct-device',implementation_path=str(BACKEND),
                pipeline_file=str(ROOT/'scripts/insitu/device_render_pipeline.py'))
        (case/'datin/input.insitu').write_text(namelist(dict(insitu_run=options)))
        records=[] if restore is not None or not enabled else items
        (case/'input.products.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in
            [dict(catalog_schema='astr-insitu-products-v1'),*records]))
        selected=[1,1,1] if ranks==1 else topology
        config=env|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,selected)),ASTR_INSITU_CONFIG='datin/input.insitu')
        if opt.images:
            for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):config.pop(key,None)
        command=[str(opt.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if opt.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                '--error-exitcode','99','--log-file','memcheck.%p.log']
        command +=[str(binary.resolve()),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=config,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case
    if opt.freeze_only:
        refs={}
        for name,backend,binary in [('cpu','cpu','astr_cpu'),('old_gpu','gpu','astr_cuda_fortran')]:
            case=run(name,opt.reference_binaries/'bin'/binary,backend,1)
            refs[name]=inspect(case,items)
        refs['pair_fields']=compare_products(output/'cpu',output/'old_gpu')
        (output/'reference.json').write_text(json.dumps(refs,indent=2)+'\n');print(json.dumps(refs,indent=2));return
    if opt.reference_results is None:
        parser.error('--reference-results selects the frozen wall consumer evidence')
    actual=run('cpp',opt.binary,'gpu',opt.ranks);report=inspect(actual,items)
    if 'backend=cuda' not in (actual/'run.log').read_text():raise AssertionError('C++ main not selected')
    final='outdat/new/checkpoints/step000000000004'
    report.update(topology=topology,transfers=transfer_usage(actual,opt.ranks,opt.mpi_pml),references={})
    for name in ('cpu','old_gpu'):
        reference=opt.reference_results/name
        report['references'][name]=dict(products=compare_products(actual,reference),
            terminal=max(compare(checkpoint(actual/final/'state.h5'),checkpoint(reference/final/'state.h5'))))
    quiet=run('cpp-off',opt.binary,'gpu',opt.ranks,enabled=False)
    compare_hdf(actual/final/'state.h5',quiet/final/'state.h5',exact=True)
    if not np.array_equal(logged_steps(actual),logged_steps(quiet)):raise AssertionError('wall observation changed step control')
    prefix=run('cpp-prefix',opt.binary,'gpu',opt.ranks,steps=2)
    tail=run('cpp-tail',opt.binary,'gpu',opt.ranks,restore=prefix/'outdat/new/checkpoints/step000000000002')
    for name in ('state.h5','statistics.h5'):compare_hdf(actual/final/name,tail/final/name,exact=True)
    if (actual/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('restored wall controls')
    report.update(observation_state_and_clock_bits=True,exact_restart=compare_products(tail,actual,exact=True))
    if opt.images:
        reference=run('old-gpu-images',opt.renderer_reference,'gpu',opt.ranks)
        inspect(reference,items)
        current=inspect_images(actual,opt.ranks);previous=inspect_images(reference,opt.ranks)
        report['old_new_pixel_mean_absolute_difference']={key:float(np.mean(abs(value.astype(float)-previous[key])))
            for key,value in current.items()}
        restored=inspect_images(tail,opt.ranks,tail=True)
        if any(not np.array_equal(value,current[key]) for key,value in restored.items()):
            raise AssertionError('wall image pixels changed on exact continuation')
        report['image_restart_pixels_equal']=True
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
