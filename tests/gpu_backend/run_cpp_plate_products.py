"""Real static-plate products, selected histories and GPU-resident images."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil

import numpy as np
from PIL import Image

from run_cpp_tgv_gate import namelist,logged_steps,compare
from run_cpp_restart_gate import datasets
from run_cpp_insitu_gate import compare_hdf
from run_cpp_wall_consumers import compare_products
from test_insitu_wall_transport_runtime import wall_mean_definitions,check_wall_means
from test_runtime_catalog_images import image_product,q_product,line_product
from test_output_insitu_restart import BACKEND

ROOT=Path(__file__).resolve().parents[2]


def numeric_definitions():
    items=wall_mean_definitions()
    for item in items:
        item['sampling']['index_ranges']['i']=[0,16]
        span=item['sampling']['index_ranges']['k'];item['sampling']['index_ranges']['k']=[n*2//3 for n in span]
    mean=deepcopy(items[0]);mean.update(id='mean-wall-plane',source='reynolds',statistics_source='selected-time')
    return items+[mean]


def image_definitions(preset='plate'):
    plane=image_product('plate-plane',8,'velocity_x')
    plane['sampling']['index_ranges']=dict(i=[0,16],j=[0,16])
    plane['display'].update(camera=[5.,.5,12.],focal_point=[5.,.5,.125],view_up=[0.,1.,0.],
        parallel_scale=4.,color_range=[0.,1.2])
    mean=deepcopy(plane);mean.update(id='mean-plate-plane',source='reynolds',statistics_source='selected-time')
    wall=image_product('plate-wall',0,'wall_shear_x')
    wall['sampling']=dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,16],k=[0,16]))
    wall['display'].update(camera=[5.,-5.,.125],focal_point=[5.,0.,.125],view_up=[0.,0.,1.],
        parallel_scale=4.,color_range=[0.,.02])
    iso=q_product('plate-vortices',1e-4,'velocity_magnitude')
    iso['sampling']['bounds']=[0.,10.,0.,1.,0.,.25]
    iso['display'].update(camera=[12.,-5.,4.],focal_point=[5.,.5,.125],view_up=[0.,1.,0.],
        parallel_scale=5.,color_range=[0.,1.2])
    lines=line_product('plate-lines')
    lines['sampling']['bounds']=[0.,10.,0.,1.,0.,.25];lines['display']=deepcopy(iso['display'])
    lines['extraction'].update(seed_layout='sbli-layered16',initial_step=.002,minimum_step=1e-5,maximum_step=.02,
        maximum_error=1e-8,maximum_length=2.,maximum_steps=10000,direction='forward')
    if preset=='high-open':
        for item in (plane,mean):
            item['display'].update(camera=[.5,.5,2.],focal_point=[.5,.5,.0625],parallel_scale=.65)
        wall['display'].update(camera=[.5,-2.,.0625],focal_point=[.5,0.,.0625],parallel_scale=.65,
            color_range=[-.001,.001])
        for item in (iso,lines):
            item['sampling']['bounds']=[0.,1.,0.,1.,0.,.125]
            item['display'].update(camera=[1.4,-1.,1.],focal_point=[.5,.5,.0625],parallel_scale=.8)
        lines['display'].update(camera=[.9,.7,1.5],focal_point=[.38,.5,.0625],parallel_scale=.65)
        lines['extraction']['maximum_length']=.75
    return [plane,mean,wall,iso,lines]


def inspect(case,records):
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=len(records) or any(p['state']!='active' for p in status['products']):
        raise AssertionError(('inactive plate products',status))
    report={}
    check_wall_means(case,[records[0],*[p for p in records if p['kind']=='wall_profile']],
        lambda key,value:report.__setitem__(key,value),steps=4,dt=1e-6,window=(.5e-6,3.5e-6),ni=17,nk=16)
    return report


def renderer_state(actual,reference):
    current,old=datasets(actual),datasets(reference)
    if current.keys()!=old.keys():raise AssertionError('renderer state dataset identities')
    fields={}
    for key,expected in old.items():
        if key=='rank_extras':continue
        observed=current[key]
        if observed.shape!=expected.shape or observed.dtype!=expected.dtype:
            raise AssertionError(('renderer state layout',key))
        if expected.dtype.kind=='f':
            fields[key]=max(compare(observed.reshape(-1,1),expected.reshape(-1,1)))
        elif not np.array_equal(observed,expected):
            raise AssertionError(('renderer state phase/metadata',key))
    # Legacy H1 physical-exterior primitive caches are not defined flow values.
    # Exact same-backend continuation still compares the entire state above.
    return dict(physical_fields_scaled=fields,
        legacy_extra_nonfinite_words=int(np.count_nonzero(~np.isfinite(old['rank_extras']))),
        cpp_extra_nonfinite_words=int(np.count_nonzero(~np.isfinite(current['rank_extras']))))


def images(case,ranks,tail=False,complete_lines=False,records=None):
    result={}
    for record in image_definitions() if records is None else records:
        paths=list((case/'outdat/new/render_products'/record['id']).glob('*/segment*'))
        if len(paths)!=1:raise AssertionError(('image segment',record['id'],paths))
        directory=paths[0]
        steps=([3] if tail else [1,3]) if record['source']=='reynolds' else ([4] if tail else [0,2,4])
        files=sorted(directory.glob('frame.step*.jpeg'))
        if [int(p.stem.split('step')[1]) for p in files]!=steps or list(directory.glob('*.eps')):
            raise AssertionError(('image clock/EPS',record['id']))
        receipts=[json.loads(p.read_text()) for p in directory.glob('frame.*.rank*.json')]
        if len(receipts)!=len(steps)*ranks or any(p['field_d2h_bytes'] or p['geometry_host_bytes'] for p in receipts):
            raise AssertionError('image field/geometry download or missing per-rank receipt')
        for step in steps:
            at_step=[p for p in receipts if p['step']==step]
            if sum(p['product']['local_cells'] for p in at_step)<=0:raise AssertionError('empty plate image geometry')
            if complete_lines and record['kind']=='streamlines':
                for row in at_step:
                    trace=row['streamline']
                    if (trace['termination_step_budget'] or trace['termination_domain_exit'] or
                        trace['termination_length']+trace['termination_subminimum']!=trace['trajectory_particles']):
                        raise AssertionError(('plate streamline did not complete the requested arc',step,trace))
            pixels=np.asarray(Image.open(directory/f'frame.step{step:012d}.jpeg').convert('RGB'))
            if pixels.shape!=(600,800,3) or pixels.std()<1:raise AssertionError('plate raster layout or blank image')
            result[record['id']+f'/step{step}']=pixels
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','reference-binaries','fixture','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=2)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--images',action='store_true')
    parser.add_argument('--image-preset',choices=['plate','high-open'],default='plate')
    parser.add_argument('--renderer-reference',type=Path)
    parser.add_argument('--renderer-results',type=Path,help='reuse immutable old-renderer output with unchanged inputs')
    parser.add_argument('--completed-run',type=Path,help='reuse an unchanged completed full case; run only remaining comparisons')
    parser.add_argument('--pipeline',choices=['standard-device','direct-device'],default='direct-device')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if args.images and args.renderer_reference is None and args.renderer_results is None:
        parser.error('--images needs a frozen Catalyst GPU reference or its immutable results')
    records=numeric_definitions();image_records=image_definitions(args.image_preset)
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env=os.environ|dict(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_osc='pt2pt',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_COMPACT_TRANSPORT='pinned',
        ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_INSITU_WRITE_EPS='0')
    for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):env.pop(key,None)
    def run(name,binary,backend,ranks,steps=4,restore=None,enabled=True,render=False):
        case=root/name;case.mkdir()
        source=args.fixture.resolve()/('cpu' if backend=='cpu' else 'old_gpu')/'datin'
        shutil.copytree(source,case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        options=dict(enabled=enabled,statistics=enabled,render=render,products='catalog_stats' if enabled else 'all',
            statistics_window=[.5e-6,3.5e-6],output_directory='outdat',
            host_budget_bytes=-1 if enabled else 0,device_budget_bytes=-1 if enabled else 0,device_reserve_bytes=0)
        if render:
            options.update(processing_backend='device',derivative_backend='gpu',postprocess_transport='pinned',
                rendering_pipeline=args.pipeline,implementation_path=str(BACKEND),
                pipeline_file=str(ROOT/'scripts/insitu/device_render_pipeline.py'))
        (case/'datin/input.insitu').write_text(namelist(dict(insitu_run=options)))
        items=records+(image_records if render else [])
        payload=[] if not enabled or restore is not None else items
        (case/'input.products.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in
            [dict(catalog_schema='astr-insitu-products-v1'),*payload]))
        config=env|dict(ASTR_INSITU_CONFIG='datin/input.insitu',
            ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology if ranks>1 else [1,1,1])))
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks),str(binary.resolve()),'run'],
                cwd=case,env=config,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case,items
    if args.freeze_only:
        report={}
        for name,backend,binary in [('cpu','cpu','astr_cpu'),('old_gpu','gpu','astr_cuda_fortran')]:
            case,items=run(name,args.reference_binaries/'bin'/binary,backend,1)
            report[name]=inspect(case,items)
        report['pair_products']=compare_products(root/'cpu',root/'old_gpu')
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        root.chmod(0o555);print(json.dumps(report,indent=2));return
    if args.reference_results is None:parser.error('--reference-results is required after freezing')
    if args.images:
        if args.renderer_results:
            old=args.renderer_results.resolve();old_items=records+image_records
        else:
            old,old_items=run('old-gpu-images',args.renderer_reference,'gpu',args.ranks,render=True)
        inspect(old,old_items)
    if args.completed_run:
        current=args.completed_run.resolve();items=records+(image_records if args.images else [])
    else:
        current,items=run('cpp',args.binary,'gpu',args.ranks,render=args.images)
    if 'backend=cuda' not in (current/'run.log').read_text():raise AssertionError('native C++ main not selected')
    report=inspect(current,items);report.update(ranks=args.ranks,axis=args.axis,references={},
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),image_preset=args.image_preset,
        reused_full_case=str(current) if args.completed_run else None)
    for backend in ('cpu','old_gpu'):
        report['references'][backend]=compare_products(current,args.reference_results/backend)
    final=Path('outdat/new/checkpoints/step000000000004')
    quiet,_=run('cpp-off',args.binary,'gpu',args.ranks,enabled=False)
    compare_hdf(current/final/'state.h5',quiet/final/'state.h5',exact=True)
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('products changed time control')
    prefix,_=run('cpp-prefix',args.binary,'gpu',args.ranks,steps=2,render=args.images)
    tail,_=run('cpp-tail',args.binary,'gpu',args.ranks,restore=prefix/'outdat/new/checkpoints/step000000000002',render=args.images)
    for name in ('state.h5','statistics.h5'):compare_hdf(current/final/name,tail/final/name,exact=True)
    if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('restored controls differ')
    report.update(observation_state_clock_bits=True,exact_restore=compare_products(tail,current,exact=True))
    if args.images:
        report['renderer_state']=renderer_state(current/final/'state.h5',old/final/'state.h5')
        if args.renderer_reference:
            report['renderer_reference_sha256']=hashlib.sha256(args.renderer_reference.read_bytes()).hexdigest()
        now=images(current,args.ranks,complete_lines=True,records=image_records)
        previous=images(old,args.ranks,records=image_records)
        resumed=images(tail,args.ranks,tail=True,complete_lines=True,records=image_records)
        report['pixel_mean_absolute_error']={k:float(np.mean(abs(v.astype(float)-previous[k]))) for k,v in now.items()}
        if any(not np.array_equal(v,now[k]) for k,v in resumed.items()):raise AssertionError('image restore pixels changed')
        report['image_restore_pixels_equal']=True
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
