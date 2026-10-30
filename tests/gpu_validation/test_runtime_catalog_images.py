"""Actual completed-step images from explicit dynamic product definitions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import errno
import re
from copy import deepcopy

import h5py
import numpy as np
import pytest
from PIL import Image

from test_index_slice_runtime import ROOT, MPIEXEC, arguments as data_arguments, product
from test_insitu_catalog_products import visual_product
from test_insitu_device_products import resident_configuration
from test_output_insitu_restart import configuration
from run_output_restart_validation import compare_fields, run_case


def arguments(output):
    args=data_arguments(output)
    args.executable=Path(os.environ.get('ASTR_OUTPUT_INSITU_EXE',ROOT/'build_insitu_device_render/bin/astr')).resolve()
    return args


def image_product(name='lower', index=4, field='velocity_x'):
    item = visual_product(name=name)
    item['field'] = field
    item['sampling'] = dict(mode='grid_index', axis='k', index=index,
                            index_ranges=dict(i=[0, 32], j=[0, 32]))
    item['display'].update(camera=[np.pi, np.pi, 13.],
                           focal_point=[np.pi, np.pi, index * 2. * np.pi / 32.],
                           view_up=[0., 1., 0.], parallel_scale=4., color_range=[-1., 1.])
    if field == 'velocity_magnitude':
        item['display'].update(color_range=[0., 1.5], color_map='grayscale')
    item['schedule']['interval'] = 2
    return item


def q_product(name='lower', threshold=.25, field='velocity_magnitude'):
    item=image_product(name=name,field=field)
    item['kind']='q_surface'
    item['sampling']=dict(mode='volume',bounds=[0.,2*np.pi,0.,2*np.pi,0.,2*np.pi])
    item['extraction']=dict(isovalue=threshold)
    item['display'].update(camera=[11.,-10.,9.],focal_point=[np.pi]*3,view_up=[0.,0.,1.],
        parallel_scale=6.4,color_map='diverging_blue_red')
    return item


def dynamic_image(kind,name='lower',index=4,field='velocity_x'):
    if kind=='streamlines':
        return line_product(name=name,field=field)
    if kind=='q_surface':
        return q_product(name=name,field=field)
    item=image_product(name=name,index=index,field=field)
    if kind=='physical_plane':
        item['sampling']=dict(mode='physical_plane',origin=[0.,0.,index*2*np.pi/32.],normal=[0.,0.,7.],
            bounds=[0.,2*np.pi,0.,2*np.pi,0.,2*np.pi])
    return item


def line_product(name='lines',field='velocity_magnitude',source='instantaneous'):
    item=visual_product('streamlines',name=name)
    item['field']=field;item['source']=source
    if source!='instantaneous': item['statistics_source']='velocity-time'
    item['sampling']['bounds']=[0.,2*np.pi,0.,2*np.pi,0.,2*np.pi]
    item['display'].update(camera=[7.,-7.,7.],focal_point=[np.pi/2,np.pi/2,np.pi/4],view_up=[0.,0.,1.],
        parallel_scale=2.2,color_range=[0.,.75] if field=='velocity_magnitude' else [-.75,.75])
    h=2*np.pi/32.
    item['extraction'].update(seed_layout='line16',initial_step=.1*h,minimum_step=.01*h,maximum_step=.5*h,
        maximum_error=1e-8,maximum_length=2*np.pi,maximum_steps=100000,direction='both')
    item['schedule']['interval']=2
    return item


@pytest.mark.parametrize('ranks,pipeline',[(1,'standard-device'),(2,'standard-device'),(2,'direct-device')])
def test_dynamic_streamline_instances_reuse_trajectories_not_display_settings(tmp_path,ranks,pipeline):
    args=arguments(tmp_path)
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='velocity_slice')
    a=line_product();b=deepcopy(a);b['id']='other-camera'
    b['display']['camera'][0]+=2.;b['display']['line_width']=3.
    c=line_product('component','velocity_x')
    d=line_product('shorter');d['extraction']['maximum_length']=np.pi
    definitions=[a,b,c,d]
    baseline,_=run_case(args,ROOT,'gpu',ranks,'without-lines',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',ranks,'with-lines',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=definitions,postprocess_transport='pinned')
    compare_fields(baseline/'outdat/new/checkpoints/step000000000002/state.h5',
                   case/'outdat/new/checkpoints/step000000000002/state.h5')
    log=(case/'run.log').read_text()
    assert log.count('ASTR_RUNTIME_INPUT ')==2*ranks
    assert log.count('integrations=2 clipped_geometries=2 scalar_builds=3 owner_reuse=1 retained_after=0')==2*ranks
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(x['state']=='active' and x['last_output_step']==2 for x in status['products'])
    actual=image_segments(case);assert len(actual)==4
    for segment in actual:
        assert len(list(segment.glob('*.jpeg')))==2
        records=[json.loads(path.read_text()) for path in segment.glob('frame.*.rank*.json')]
        assert len(records)==2*ranks
        assert all(not record['geometry_empty'] for record in records)
        for step in (0,2):
            assert sum(record['product']['local_cells'] for record in records if record['step']==step)>0
        for record in records:
            assert record['field_d2h_bytes']==record['geometry_host_bytes']==0
            trace=record['streamline'];assert trace['trajectory_seed_count']==16 and trace['trajectory_particles']==32
            assert sum(trace['termination_'+x] for x in ('length','step_budget','domain_exit','subminimum'))==32
            assert trace['trajectory_integration_performed']==int(record['id'] in ('lines','shorter'))


def line_oracle(path):
    data=path.read_bytes();assert data[:8]==b'ASTRLI01'
    width,points,cells=map(int,np.frombuffer(data,dtype='<u8',count=3,offset=8))
    assert width in (4,8)
    dtype=np.dtype('<f4' if width==4 else '<f8');offset=32
    xyz=np.frombuffer(data,dtype=dtype,count=3*points,offset=offset).reshape(points,3);offset+=3*points*width
    values=np.frombuffer(data,dtype=dtype,count=points,offset=offset);offset+=points*width
    indices=np.frombuffer(data,dtype='<i8',count=2*cells,offset=offset).reshape(cells,2);offset+=cells*16
    assert offset==len(data)
    assert np.isfinite(xyz).all() and np.isfinite(values).all()
    if cells:
        assert indices.min()>=0 and indices.max()<points
        assert not np.any(np.all(xyz[indices[:,0]]==xyz[indices[:,1]],axis=1))
    return xyz,values,indices


def source_velocity(checkpoint,source):
    def periodic(values): return np.pad(values[:32,:32,:32],((0,1),)*3,mode='wrap')
    if source=='instantaneous':
        with h5py.File(checkpoint/'state.h5') as state:
            density=state['q0001'][...]
            return np.stack([periodic(state[f'q{c:04d}'][...]/density) for c in (2,3,4)],axis=-1)
    start=10 if source=='reynolds' else 13
    with h5py.File(checkpoint/'statistics.h5') as state:
        return np.stack([periodic(state[f'q{c:04d}'][...]) for c in range(start,start+3)],axis=-1)


def sample_velocity(values,xyz):
    scaled=np.clip(xyz/(2*np.pi/32),0,32)
    base=np.minimum(np.floor(scaled).astype(int),31);fraction=scaled-base
    result=np.zeros((len(xyz),3))
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                weight=np.prod(np.where(np.array([i,j,k]),fraction,1-fraction),axis=1)
                result+=weight[:,None]*values[base[:,2]+k,base[:,1]+j,base[:,0]+i]
    return result


@pytest.mark.parametrize('pipeline,mode',[('standard-device','pinned'),('direct-device','pinned'),('direct-device','device-aware')])
def test_streamline_colors_match_the_selected_instantaneous_or_mean_source(tmp_path,pipeline,mode):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,mode,statistics=True,interval=1,profile='velocity_slice')
    definitions=[line_product(source,field='velocity_magnitude',source=source)
                 for source in ('instantaneous','reynolds','favre')]
    for item in definitions: item['schedule']['interval']=1
    cpu,_=run_case(args,ROOT,'cpu',2,'cpu-source',2,grid='32,32,32',insitu_config=configuration(render=False,interval=1))
    baseline,_=run_case(args,ROOT,'gpu',2,'gpu-source',2,grid='32,32,32',
        insitu_config=config,product_catalog_records=[],postprocess_transport=mode)
    case,_=run_case(args,ROOT,'gpu',2,'line-sources',2,grid='32,32,32',insitu_config=config,
        postprocess_transport=mode,product_catalog_records=definitions,runtime_line_oracle=True)
    ending='outdat/new/checkpoints/step000000000002'
    for name in ('state.h5','statistics.h5'): compare_fields(baseline/ending/name,case/ending/name)
    for definition in definitions:
        source=definition['source'];cpu_velocity=source_velocity(cpu/ending,source);gpu_velocity=source_velocity(case/ending,source)
        paths=list((case/'outdat').glob(f'runtime_line_oracle.{source}.step2.rank*.bin'));assert len(paths)==2
        total=0
        for path in paths:
            xyz,actual,indices=line_oracle(path);total+=len(indices)
            for reference in (cpu_velocity,gpu_velocity):
                velocity=sample_velocity(reference,xyz)
                expected=np.sqrt(np.sum(velocity*velocity,axis=1))
                if len(xyz): np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-10)
        assert total>0
        segment=next(p for p in image_segments(case) if p.parent.parent.name==source)
        steps=[int(p.name.split('.')[1][4:]) for p in sorted(segment.glob('*.jpeg'))]
        assert steps==([0,1,2] if source=='instantaneous' else [1,2])
        for path in segment.glob('frame.step000000000002.rank*.json'):
            record=json.loads(path.read_text());assert record['validation_geometry_readback_requested'] is True
            if source!='instantaneous':
                assert record['statistics']['provider']=='velocity-time' and record['statistics']['covered_duration']>0
                assert record['statistics']['weight']==('density_time_integral' if source=='favre' else 'time_integral')


@pytest.mark.parametrize('pipeline,mapping,seeds',[('standard-device',None,'tgv-stratified'),
    ('standard-device','y-wavy','line16'),('direct-device','y-wavy','line16')])
def test_runtime_streamline_existing_dense_seeds_and_curve_mapping(tmp_path,pipeline,mapping,seeds):
    args=arguments(tmp_path)
    if mapping:
        args.scale_timestep=2e-5;args.maximum_cfl=.5
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='curve_demo' if mapping else 'velocity_slice')
    item=line_product();item['extraction']['seed_layout']=seeds
    definitions=[item]
    if seeds=='tgv-stratified':
        item['display'].update(camera=[11.,-10.,9.],focal_point=[np.pi]*3,parallel_scale=6.4)
        sparse=line_product('sparse')
        other=deepcopy(item);other['id']='dense-other-camera';other['display']['camera'][0]+=1.
        definitions=[sparse,item,other]
    baseline,_=run_case(args,ROOT,'gpu',2,'coverage-empty',2,grid='32,32,32',tgv_mapping=mapping,
        insitu_config=config,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'coverage-lines',2,grid='32,32,32',tgv_mapping=mapping,
        insitu_config=config,product_catalog_records=definitions,postprocess_transport='pinned')
    compare_fields(baseline/'outdat/new/checkpoints/step000000000002/state.h5',
                   case/'outdat/new/checkpoints/step000000000002/state.h5')
    for segment in image_segments(case):
        records=[json.loads(p.read_text()) for p in segment.glob('frame.*.rank*.json')]
        assert len(records)==4 and all(not record['geometry_empty'] for record in records)
        expected=256 if seeds=='tgv-stratified' and segment.parent.parent.name!='sparse' else 16
        assert all(record['streamline']['trajectory_seed_count']==expected and
            record['streamline']['trajectory_particles']==2*expected for record in records)
        assert all(record['field_d2h_bytes']==record['geometry_host_bytes']==0 for record in records)
        assert len(list(segment.glob('*.jpeg')))==2
    if seeds=='tgv-stratified':
        log=(case/'run.log').read_text()
        assert log.count('integrations=2 clipped_geometries=2 scalar_builds=2 owner_reuse=1 retained_after=0')==4
    if mapping:
        assert json.loads((case/'cfl_gate.json').read_text())['maximum']<=.5


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','compatible'])
def test_mean_streamline_exact_continuation_preserves_coverage_and_pixels(tmp_path,pipeline):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    definitions=[line_product(source,source=source) for source in ('reynolds','favre')]
    for item in definitions: item['schedule']['interval']=4
    common=dict(grid='32,32,32',insitu_config=config,postprocess_transport='pinned',
        checkpoint_interval=5,product_catalog_records=definitions)
    continuous,_=run_case(args,ROOT,'gpu',2,'mean-lines-continuous',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'mean-lines-first',5,**common)
    resumed,_=run_case(args,ROOT,'gpu',2,'mean-lines-resumed',12,grid='32,32,32',insitu_config=config,
        postprocess_transport='pinned',checkpoint_interval=5,product_catalog_records=[],
        restore=first/'outdat/new/checkpoints/step000000000005')
    ending='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): compare_fields(continuous/ending/name,resumed/ending/name)
    references={p.parent.parent.name:p for p in image_segments(continuous)}
    for segment in image_segments(resumed):
        expected=references[segment.parent.parent.name]
        assert [p.name for p in segment.glob('*.jpeg')]==['frame.step000000000009.jpeg']
        for path in segment.glob('*.jpeg'):
            with Image.open(path) as a,Image.open(expected/path.name) as b:
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        for path in segment.glob('frame.*.rank*.json'):
            a=json.loads(path.read_text());b=json.loads((expected/path.name).read_text())
            assert a['statistics']==b['statistics'] and a['definition']==b['definition']


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_agreed_streamline_failure_releases_private_mpi_and_keeps_other_products(tmp_path,pipeline):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    failed=line_product('failed');failed['extraction']['maximum_steps']=1
    definitions=[failed,line_product('healthy'),image_product()]
    baseline,_=run_case(args,ROOT,'gpu',2,'shared-failure-baseline',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'shared-trace-failure',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=definitions,postprocess_transport='pinned',trace_collective_fault=True)
    ending='outdat/new/checkpoints/step000000000002'
    for name in ('state.h5','statistics.h5'): compare_fields(baseline/ending/name,case/ending/name)
    status={x['id']:x for x in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['failed']['state']=='resource-disabled' and status['failed']['last_output_step']<0
    assert 'injected collectively agreed trajectory failure' in status['failed']['reason']
    for segment in image_segments(case):
        name=segment.parent.parent.name
        assert len(list(segment.glob('*.jpeg')))==(0 if name=='failed' else 2)
    assert status['healthy']['state']==status['lower']['state']=='active'


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('kind',['slice','physical_plane','q_surface','streamlines'])
def test_rank_local_allocation_refusal_keeps_cfd_and_needs_explicit_request(
        tmp_path,pipeline,kind,catalog_replay_library):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    failed=dynamic_image(kind,'failed')
    healthy=image_product('healthy',20,'velocity_magnitude')
    retry=deepcopy(failed);retry['enable_request']=2
    baseline,_=run_case(args,ROOT,'gpu',2,'allocation-baseline',8,grid='32,32,32',insitu_config=config,
        product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'allocation-recovery',8,grid='32,32,32',insitu_config=config,
        product_catalog_records=[failed,healthy],product_poll_steps=1,
        runtime_allocation_refusal=('failed',2,1),
        memcheck=os.environ.get('ASTR_RUNTIME_ALLOCATION_MEMCHECK')=='1',
        catalog_replay=(catalog_replay_library,{6:[retry,healthy]}))
    ending='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'): compare_fields(baseline/ending/name,case/ending/name)
    status={x['id']:x for x in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['failed']['state']==status['healthy']['state']=='active'
    assert status['failed']['version']==1 and status['failed']['consumed_enable_request']==2
    segments=image_segments(case)
    failed_frames=sorted([int(p.name.split('.')[1][4:]) for p in sorted(segment.glob('*.jpeg'))]
        for segment in segments if segment.parent.parent.name=='failed')
    assert failed_frames==[[0],[5,7]]
    good=next(segment for segment in segments if segment.parent.parent.name=='healthy')
    assert [int(p.name.split('.')[1][4:]) for p in sorted(good.glob('*.jpeg'))]==[0,2,4,6,8]
    log=(case/'run.log').read_text()
    assert 'ALLOCATION REFUSED source=runtime allocation test' in log
    assert 'device allocation would exceed budget/reserve' in log
    assert 'id=failed step=4 ' not in log


@pytest.mark.parametrize('pipeline,mapping,stage',[
    ('standard-device',None,'prepare'),('direct-device',None,'round'),
    ('compatible',None,'concatenate'),('standard-device','y-wavy','owner-query')])
def test_distributed_trace_allocation_failure_has_a_matched_exit(
        tmp_path,pipeline,mapping,stage,catalog_replay_library):
    args=arguments(tmp_path)
    if mapping:
        args.scale_timestep=2e-5;args.maximum_cfl=.5
    config=resident_configuration(pipeline,statistics=False,interval=1,
        profile='curve_demo' if mapping else 'velocity_slice')
    failed=line_product('failed')
    # The second trajectory must remain usable after the first one's MPI release.
    healthy=line_product('healthy');healthy['extraction']['maximum_length']=np.pi
    retry=deepcopy(failed);retry['enable_request']=2
    common=dict(grid='32,32,32',tgv_mapping=mapping,insitu_config=config)
    baseline,_=run_case(args,ROOT,'gpu',2,'off',8,**common,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'trace-refusal',8,**common,
        product_catalog_records=[failed,healthy],product_poll_steps=1,
        trace_allocation_refusal=('failed',2,1,stage),
        catalog_replay=(catalog_replay_library,{6:[retry,healthy]}),
        memcheck=os.environ.get('ASTR_RUNTIME_TRACE_FAILURE_MEMCHECK')=='1')
    final='outdat/new/checkpoints/step000000000008/state.h5'
    compare_fields(baseline/final,case/final)
    actual={name:[] for name in ('failed','healthy')}
    for segment in image_segments(case):
        actual[segment.parents[1].name].extend(int(path.name[10:22]) for path in segment.glob('*.jpeg'))
    assert sorted(actual['failed'])==[0,5,7] and sorted(actual['healthy'])==[0,2,4,6,8]
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['failed']['state']==status['healthy']['state']=='active',status
    assert status['failed']['version']==1 and status['failed']['consumed_enable_request']==2
    log=(case/'run.log').read_text()
    assert 'ALLOCATION REFUSED source=runtime distributed trace test' in log
    assert 'id=failed step=4 ' not in log


def raw_product(image):
    fields = (['velocity_x', 'velocity_y', 'velocity_z'] if image['field'] == 'velocity_magnitude'
              else [image['field']])
    item = product(image['id'] + '-data', fields=fields)
    item['source']=image['source']
    if 'statistics_source' in image:
        item['statistics_source']=image['statistics_source']
    item['sampling'] = image['sampling']
    item['schedule'] = image['schedule']
    return item


def image_segments(case):
    return sorted((case / 'outdat/new/render_products').glob('*/*/segment*'))


@pytest.fixture(scope='module')
def catalog_replay_library(tmp_path_factory):
    library=tmp_path_factory.mktemp('catalog_replay')/'replay.so'
    result=subprocess.run(['cc','-std=c99','-Wall','-Wextra','-Werror','-shared','-fPIC',
        str(ROOT/'tests/gpu_validation/runtime_catalog_replay.c'),'-ldl','-o',str(library)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return library


def test_image_failure_preload_is_path_scoped(tmp_path,catalog_replay_library):
    source=tmp_path/'source.partial';source.write_bytes(b'image-payload')
    env=dict(os.environ,LD_PRELOAD=str(catalog_replay_library),
             ASTR_TEST_IMAGE_FAIL_ID='lower',ASTR_TEST_IMAGE_FAIL_STEP='2')
    script='import os,sys\ntry: os.link(sys.argv[1],sys.argv[2])\nexcept OSError as exc: raise SystemExit(exc.errno)'
    for name,expected in [('lower',errno.EIO),('upper',0)]:
        target=tmp_path/f'render_products/{name}/version1/segment/frame.step000000000002.jpeg'
        target.parent.mkdir(parents=True)
        result=subprocess.run([sys.executable,'-c',script,str(source),str(target)],env=env,capture_output=True,text=True)
        assert result.returncode==expected,result.stdout+result.stderr
        assert target.exists()==(expected==0)


@pytest.mark.skipif(not os.environ.get('ASTR_RUNTIME_PLANE_PROBE'),reason='Select the root-built runtime image component probe')
@pytest.mark.parametrize('sanitize',[False,True])
def test_runtime_image_component_and_memory(sanitize):
    wrapper=['compute-sanitizer','--tool','memcheck','--error-exitcode','99'] if sanitize else []
    command=[str(MPIEXEC),'--oversubscribe','--mca','pml','ob1','--mca','btl','self,vader,tcp',
        '--mca','osc','pt2pt','--mca','coll','^cuda,hcoll,ucc','--mca','opal_cuda_support','0',
        '-n','2',*wrapper,os.environ['ASTR_RUNTIME_PLANE_PROBE']]
    result=subprocess.run(command,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'RUNTIME_IMAGE_COMPONENT_PASS source_bytes=8' in result.stdout
    assert float(result.stdout.split('maxabs=')[1].split()[0])<=2e-10
    assert result.stdout.count('triangulations=1 scalar_builds=2 owner_reuse=1 retained_after=0')==2
    assert result.stdout.count('extractions=1 scalar_builds=3 owner_reuse=1 retained_after=0')==2
    assert result.stdout.count('extractions=2 scalar_builds=3 owner_reuse=2 retained_after=0')==2
    if sanitize:
        assert (result.stdout+result.stderr).count('ERROR SUMMARY: 0 errors')==2


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_same_step_image_instances_share_fields_and_geometry(tmp_path,pipeline):
    args=arguments(tmp_path)
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='velocity_slice')
    a=image_product();b=deepcopy(a);b['id']='other-camera'
    b['display']['camera'][0]+=1.;b['display']['color_range']=[-.5,.5]
    c=image_product('other-field',field='velocity_y')
    d=image_product('gradient',field='vorticity_z');d['display']['color_range']=[-2.,2.]
    definitions=[a,b,c,d]
    case,_=run_case(args,ROOT,'gpu',2,'grouped-images',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=definitions+[raw_product(x) for x in (a,c,d)])
    log=(case/'run.log').read_text()
    assert log.count('ASTR_RUNTIME_INPUT ')==4
    assert log.count(' fields=3')==4
    assert log.count('triangulations=1 scalar_builds=3 owner_reuse=1 retained_after=0')==4
    assert len(image_segments(case))==4
    for segment in image_segments(case):
        name=segment.parent.parent.name
        definition=next(x for x in definitions if x['id']==name)
        source_name='lower' if name=='other-camera' else name
        raw=next((case/f'outdat/new/slice_products/{source_name}-data').glob('*/segment*'))
        for step in (0,2):
            records=[json.loads(x.read_text()) for x in segment.glob(f'frame.step{step:012d}.rank*.json')]
            assert len(records)==2
            ranges=[x['product']['actual_range'] for x in records]
            with h5py.File(raw/f'step{step:012d}.h5') as fields:
                values=fields[definition['field']][...]
            np.testing.assert_allclose([min(x[0] for x in ranges),max(x[1] for x in ranges)],
                [values.min(),values.max()],rtol=0,atol=2e-10)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('axis',['x','z'])
def test_physical_plane_instances_use_native_fields_and_one_cut(tmp_path,pipeline,axis):
    args=arguments(tmp_path);args.axis=axis
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='velocity_slice')
    a=image_product(field='Q_rs');a['display']['color_range']=[-1.,1.]
    b=image_product('speed',field='velocity_magnitude')
    c=deepcopy(a);c['id']='other-camera';c['display']['camera'][0]+=1.
    originals=[deepcopy(a),deepcopy(b)]
    for item in (a,b,c):
        item['sampling']=dict(mode='physical_plane',origin=[0.,0.,np.pi/4],normal=[0.,0.,7.],
            bounds=[0.,2*np.pi,0.,2*np.pi,0.,2*np.pi])
    case,_=run_case(args,ROOT,'gpu',2,'physical-images',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=[a,b,c]+[raw_product(item) for item in originals])
    log=(case/'run.log').read_text()
    assert log.count('ASTR_RUNTIME_INPUT ')==4
    assert log.count(' fields=4')==4
    assert log.count('ASTR_RUNTIME_PHYSICAL_GROUP ')==4
    assert log.count('extractions=1 scalar_builds=2 owner_reuse=1 retained_after=0')==(4 if axis=='x' else 2)
    for segment in image_segments(case):
        name=segment.parent.parent.name
        raw=next((case/f'outdat/new/slice_products/{"lower" if name=="other-camera" else name}-data').glob('*/segment*'))
        for step in (0,2):
            records=[json.loads(p.read_text()) for p in segment.glob(f'frame.step{step:012d}.rank*.json')]
            assert len(records)==2
            ranges=[x['product']['actual_range'] for x in records if x['product']['local_points']>0]
            with h5py.File(raw/f'step{step:012d}.h5') as data:
                values=(np.sqrt(sum(data[x][...]**2 for x in ('velocity_x','velocity_y','velocity_z')))
                    if name=='speed' else data['Q_rs'][...])
            np.testing.assert_allclose([min(x[0] for x in ranges),max(x[1] for x in ranges)],
                [values.min(),values.max()],rtol=0,atol=2e-10)
            for record in records:
                assert record['geometry_host_bytes']==record['field_d2h_bytes']==0
                np.testing.assert_allclose(record['physical_bounds'],[0,2*np.pi,0,2*np.pi,np.pi/4,np.pi/4],
                    rtol=0,atol=2e-10)
            with Image.open(segment/f'frame.step{step:012d}.jpeg') as picture:
                assert np.count_nonzero(np.min(np.asarray(picture.convert('RGB')),axis=2)<200)>5000


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_q_thresholds_share_one_native_volume_and_frozen_geometry(tmp_path,pipeline):
    args=arguments(tmp_path)
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='q_surface')
    a=q_product();b=deepcopy(a);b['id']='other-camera';b['display']['camera'][0]+=2.
    c=q_product('colored-Q',field='Q_rs');c['display']['color_range']=[-.5,.5]
    d=q_product('Q-zero',threshold=0.)
    definitions=[a,b,c,d]
    baseline,_=run_case(args,ROOT,'gpu',2,'q-empty',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=[],insitu_timing=True)
    case,_=run_case(args,ROOT,'gpu',2,'q-instances',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=definitions,insitu_timing=True)
    compare_fields(baseline/'outdat/new/checkpoints/step000000000002/state.h5',
                   case/'outdat/new/checkpoints/step000000000002/state.h5')
    log=(case/'run.log').read_text()
    assert log.count('ASTR_RUNTIME_INPUT ')==4 and log.count(' fields=4')==4
    assert log.count('extractions=2 scalar_builds=3 owner_reuse=1 retained_after=0')==4
    timing=[]
    for line in log.splitlines():
        if not line.startswith('ASTR_RUNTIME_STAGE_TIMING '): continue
        row=dict(re.findall(r'(\w+)=([^\s]+)',line))
        assert row['status']=='0'
        for key in ('native_local_seconds','consensus_seconds','consumer_inclusive_seconds'):
            row[key]=float(row[key]);assert np.isfinite(row[key]) and row[key]>=0
        timing.append(row)
    assert len(timing)==16
    (case/'runtime_cost.json').write_text(json.dumps(dict(
        scope='32-cubed two-step interface diagnostic, includes cold initialization; not a production benchmark',
        pipeline=pipeline,rank_product_stages=timing),indent=2))
    assert len(image_segments(case))==4
    for segment in image_segments(case):
        assert len(list(segment.glob('*.jpeg')))==2
        for path in segment.glob('frame.*.rank*.json'):
            receipt=json.loads(path.read_text())
            assert receipt['product']['local_points']>0 and receipt['product']['local_cells']>0
            assert receipt['geometry_host_bytes']==receipt['field_d2h_bytes']==0
            if receipt['id']=='colored-Q':
                np.testing.assert_allclose(receipt['product']['actual_range'],[.25,.25],rtol=0,atol=2e-10)
        with Image.open(segment/'frame.step000000000002.jpeg') as image:
            assert np.count_nonzero(np.min(np.asarray(image.convert('RGB')),axis=2)<200)>5000


@pytest.mark.parametrize('kind,pipeline',[('slice','standard-device'),('physical_plane','standard-device'),
    ('q_surface','standard-device'),('q_surface','direct-device'),('q_surface','compatible')])
def test_private_image_transport_preserves_fields_and_pixels(tmp_path,kind,pipeline):
    args=arguments(tmp_path)
    definitions=[dynamic_image(kind,field='Q_rs'),dynamic_image(kind,'speed',field='velocity_magnitude')]
    cases=[]
    for mode in ('pinned','device-aware'):
        config=resident_configuration(pipeline,mode,statistics=False,interval=1,profile='velocity_slice')
        case,_=run_case(args,ROOT,'gpu',2,mode,2,grid='32,32,32',insitu_config=config,
            postprocess_transport=mode,product_catalog_records=definitions)
        cases.append(case)
        log=(case/'run.log').read_text()
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines()
              if line.startswith('ASTR_RUNTIME_TRANSFER ')]
        assert len(rows)==4
        payload=sum(int(x['private_d2h_bytes'])+int(x['private_h2d_bytes']) for x in rows)
        assert payload>0 if mode=='pinned' else payload==0
        assert all(x['transport']==mode for x in rows)
        assert 'ASTR_RUNTIME_IMAGE_DISABLED' not in log
    compare_fields(cases[0]/'outdat/new/checkpoints/step000000000002/state.h5',
                   cases[1]/'outdat/new/checkpoints/step000000000002/state.h5')
    reference={p.parent.parent.name:p for p in image_segments(cases[0])}
    for segment in image_segments(cases[1]):
        expected=reference[segment.parent.parent.name]
        for path in segment.glob('*.jpeg'):
            with Image.open(path) as actual,Image.open(expected/path.name) as prior:
                np.testing.assert_array_equal(np.asarray(actual),np.asarray(prior))
        for path in segment.glob('frame.*.rank*.json'):
            a=json.loads(path.read_text());b=json.loads((expected/path.name).read_text())
            assert a['product']['actual_range']==b['product']['actual_range']


def test_unavailable_aware_image_transport_is_local_not_a_host_fallback(tmp_path):
    args=arguments(tmp_path)
    config=resident_configuration('standard-device','device-aware',statistics=False,interval=1,profile='velocity_slice')
    # Deliberately launch with CUDA-aware MPI disabled. The requested product mode stays device-aware.
    case,_=run_case(args,ROOT,'gpu',2,'aware-unavailable',2,grid='32,32,32',insitu_config=config,
        postprocess_transport='pinned',product_catalog_records=[image_product(field='Q_rs'),raw_product(image_product())])
    log=(case/'run.log').read_text()
    assert 'CUDA-aware MPI is unavailable for the requested private image transport' in log
    status={x['id']:x for x in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['lower']['state']=='resource-disabled'
    assert status['lower-data']['state']=='active'
    assert not image_segments(case)
    assert list((case/'outdat/new/slice_products/lower-data').rglob('step000000000002.h5'))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_valid_empty_surfaces_publish_without_disabling_the_product(tmp_path,pipeline):
    args=arguments(tmp_path)
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='velocity_slice')
    surface=q_product('empty-Q',threshold=10.)
    plane=dynamic_image('physical_plane','outside-plane')
    plane['sampling']['origin'][2]=10.
    case,_=run_case(args,ROOT,'gpu',2,'empty-surfaces',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=[surface,plane,image_product()])
    status={x['id']:x for x in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert all(x['state']=='active' and x['last_output_step']==2 for x in status.values())
    assert len(image_segments(case))==3
    for segment in image_segments(case):
        assert len(list(segment.glob('*.jpeg')))==2
        for path in segment.glob('frame.*.rank*.json'):
            record=json.loads(path.read_text())
            empty=record['id']!='lower'
            assert record['geometry_empty']==empty
            if empty:
                assert record['physical_bounds'] is None
                assert record['product']['local_cells']==record['product']['local_points']==0
                assert 'actual_range' not in record['product']


@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'z')])
def test_explicit_index_images_and_no_implicit_profile_products(tmp_path, pipeline, ranks, axis):
    args = arguments(tmp_path)
    args.axis = axis
    # The per-product JPEG-only choice overrides this legacy process-wide option.
    args.insitu_write_eps = '1'
    config = resident_configuration(pipeline, statistics=False, interval=1, profile='velocity_slice')
    baseline, _ = run_case(args, ROOT, 'gpu', ranks, 'catalog-empty', 2, grid='32,32,32',
                           insitu_config=config, product_catalog_records=[])
    assert not list((baseline / 'outdat').rglob('*.jpeg'))
    definitions = [image_product(), image_product('upper', 20, 'velocity_magnitude')]
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'catalog-images', 2, grid='32,32,32',
                       insitu_config=config,
                       product_catalog_records=definitions + [raw_product(item) for item in definitions])
    compare_fields(baseline / 'outdat/new/checkpoints/step000000000002/state.h5',
                   case / 'outdat/new/checkpoints/step000000000002/state.h5')
    actual = image_segments(case)
    assert len(actual) == 2
    for segment in actual:
        name = segment.parent.parent.name
        expected = next(item for item in definitions if item['id'] == name)
        pictures = sorted(segment.glob('*.jpeg'))
        assert [path.name for path in pictures] == ['frame.step000000000000.jpeg', 'frame.step000000000002.jpeg']
        assert not list(segment.glob('*.eps'))
        for path in pictures:
            with Image.open(path) as image:
                assert image.size == tuple(expected['output']['image_size'])
                assert tuple(round(x) for x in image.info['dpi']) == (300, 300)
                pixels = np.asarray(image.convert('RGB'))
                assert np.count_nonzero(np.min(pixels, axis=2) < 200) > 5000
        records = [json.loads(path.read_text()) for path in sorted(segment.glob('frame.*.rank*.json'))]
        assert len(records) == 2 * ranks
        raw = next((case / f'outdat/new/slice_products/{name}-data').glob('*/segment*'))
        with h5py.File(raw / 'geometry.h5') as grid:
            coordinates = grid['coordinates'][...]
        bounds = np.array([[coordinates[...,d].min(), coordinates[...,d].max()] for d in range(3)]).ravel()
        for record in records:
            assert record['id'] == name and record['definition_version'] == 1
            assert record['rendering_pipeline'] == pipeline
            assert record['geometry_host_bytes'] == record['field_d2h_bytes'] == 0
            np.testing.assert_allclose(record['physical_bounds'], bounds, rtol=0, atol=2e-10)
            np.testing.assert_array_equal(record['camera']['position'], expected['display']['camera'])
            assert record['product']['image'] == 'published'
            assert record['product']['image_formats'] == ['jpeg']
            assert record['product']['lighting_enabled'] is False
        for step in (0,2):
            selected=[record['product']['actual_range'] for record in records
                      if record['step']==step and record['product']['local_points']>0]
            observed=[min(x[0] for x in selected),max(x[1] for x in selected)]
            with h5py.File(raw/f'step{step:012d}.h5') as data:
                if expected['field']=='velocity_magnitude':
                    values=np.sqrt(sum(data[field][...]**2 for field in ('velocity_x','velocity_y','velocity_z')))
                else:
                    values=data[expected['field']][...]
            np.testing.assert_allclose(observed,[values.min(),values.max()],rtol=0,atol=2e-10)
    assert len(list((case / 'outdat').rglob('*.jpeg'))) == 4
    status = json.loads((case / 'outdat/new/products.status.json').read_text())
    assert all(item['state'] == 'active' and item['last_output_step'] == 2 for item in status['products'])


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('kind',['slice','physical_plane','q_surface','streamlines'])
def test_actual_hot_updates_retire_only_changed_views(tmp_path,pipeline,catalog_replay_library,kind):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    a=dynamic_image(kind)
    b=dynamic_image(kind,'upper',20,'velocity_magnitude')
    a['schedule']['interval']=b['schedule']['interval']=4
    moved=deepcopy(a);moved['display']['camera'][0]+=1
    frequent=deepcopy(moved);frequent['schedule']['interval']=1
    changed=deepcopy(b);changed['field']='velocity_y'
    changed['display'].update(color_map='diverging_blue_red',color_range=[-1.,1.])
    changed['schedule']['interval']=3
    invalid=deepcopy(frequent);invalid['field']='not_a_quantity'
    publications={2:[moved,b],3:[frequent,b],4:[changed],5:[],
                  6:[invalid,changed],7:[frequent,changed]}
    baseline,_=run_case(args,ROOT,'gpu',2,'hot-baseline',12,grid='32,32,32',insitu_config=config,
        product_catalog_records=[],product_poll_steps=1)
    case,_=run_case(args,ROOT,'gpu',2,'hot-images',12,grid='32,32,32',insitu_config=config,
        product_catalog_records=[a,b],product_poll_steps=1,
        catalog_replay=(catalog_replay_library,publications),insitu_memory_trace=True)
    histories={}
    receipts=[]
    for segment in image_segments(case):
        key=(segment.parent.parent.name,int(segment.parent.name.removeprefix('version')))
        histories[key]=[int(p.name.split('.')[1][4:]) for p in sorted(segment.glob('frame.step*.jpeg'))]
        receipts.extend(json.loads(path.read_text()) for path in segment.glob('frame.*.rank00000000.json'))
    assert histories=={('lower',1):[0],('lower',2):[1,2,6,7,8,9,10,11,12],
                       ('upper',1):[0],('upper',2):[3,5,8,11]}
    for record in receipts:
        assert record['definition']['id']==record['id']
        if record['id']=='lower' and record['step']==2:
            assert record['definition_version']==2 and record['applied_step']==2
            assert record['definition']['schedule']['interval']==1
        if record['id']=='upper' and record['step']>=5:
            assert record['definition_version']==2 and record['applied_step']==5
    log=(case/'run.log').read_text()
    assert log.count('ASTR_TEST_CATALOG_PUBLISH poll=')==6
    assert 'ASTR_RUNTIME_VIEWS step=4 count=0' in log
    assert 'ASTR_RUNTIME_CATALOG_CLEANUP step=4 actor_keys=0' in log
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(item['state']=='active' and item['version']==2 for item in status['products'])
    for name in ('state.h5','statistics.h5'):
        compare_fields(baseline/'outdat/new/checkpoints/step000000000012'/name,
                       case/'outdat/new/checkpoints/step000000000012'/name)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('kind',['slice','physical_plane','q_surface','streamlines'])
def test_dynamic_image_exact_continuation_keeps_effective_catalog(tmp_path,pipeline,kind):
    args=arguments(tmp_path)
    args.statistics=kind!='slice'
    config=resident_configuration(pipeline,statistics=args.statistics,interval=1,profile='velocity_slice')
    definition=dynamic_image(kind,field='velocity_magnitude' if kind in ('q_surface','streamlines') else 'velocity_x')
    definition['schedule']['interval']=4
    common=dict(grid='32,32,32',insitu_config=config,checkpoint_interval=5,
                product_catalog_records=[definition])
    continuous,_=run_case(args,ROOT,'gpu',2,'continuous-images',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'first-images',5,**common)
    conflict=q_product(threshold=0.) if kind=='q_surface' else dynamic_image(kind,index=20)
    if kind=='streamlines': conflict['extraction']['maximum_length']=np.pi/3
    resumed,_=run_case(args,ROOT,'gpu',2,'resumed-images',12,grid='32,32,32',insitu_config=config,
        checkpoint_interval=5,restore=first/'outdat/new/checkpoints/step000000000005',
        product_catalog_records=[conflict])
    reference=image_segments(continuous)[0];actual=image_segments(resumed)[0]
    assert [p.name for p in sorted(actual.glob('*.jpeg'))]==['frame.step000000000008.jpeg','frame.step000000000012.jpeg']
    for picture in actual.glob('*.jpeg'):
        with Image.open(picture) as a,Image.open(reference/picture.name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    compare_fields(continuous/'outdat/new/checkpoints/step000000000012/state.h5',
                   resumed/'outdat/new/checkpoints/step000000000012/state.h5')
    if args.statistics:
        compare_fields(continuous/'outdat/new/checkpoints/step000000000012/statistics.h5',
                       resumed/'outdat/new/checkpoints/step000000000012/statistics.h5')


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('kind',['slice','physical_plane','q_surface','streamlines'])
def test_image_publication_failure_is_local_and_requires_new_request(tmp_path,pipeline,catalog_replay_library,kind):
    args=arguments(tmp_path)
    config=resident_configuration(pipeline,statistics=False,interval=1,profile='velocity_slice')
    a=dynamic_image(kind)
    b=dynamic_image(kind,'upper',20,'velocity_magnitude')
    cadence=deepcopy(a);cadence['schedule']['interval']=1
    retry=deepcopy(cadence);retry['enable_request']=2
    case,_=run_case(args,ROOT,'gpu',2,'image-io-failure',8,grid='32,32,32',insitu_config=config,
        product_catalog_records=[a,b],product_poll_steps=1,
        catalog_replay=(catalog_replay_library,{4:[cadence,b],6:[retry,b]}),
        image_publication_failure=('lower',2))
    lower=[segment for segment in image_segments(case) if segment.parent.parent.name=='lower']
    histories=sorted([int(p.name.split('.')[1][4:]) for p in sorted(segment.glob('*.jpeg'))] for segment in lower)
    assert histories==[[0],[5,6,7,8]]
    upper=next(segment for segment in image_segments(case) if segment.parent.parent.name=='upper')
    assert [int(p.name.split('.')[1][4:]) for p in sorted(upper.glob('*.jpeg'))]==[0,2,4,6,8]
    assert not list((case/'outdat/new/render_products').rglob('*.partial'))
    log=(case/'run.log').read_text()
    assert 'ASTR_RUNTIME_IMAGE_DISABLED lower:' in log
    assert 'id=lower step=3 ' not in log and 'id=lower step=4 ' not in log
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    records={item['id']:item for item in status['products']}
    assert records['lower']['state']=='active' and records['lower']['version']==1
    assert records['lower']['consumed_enable_request']==2


def test_new_segment_does_not_reuse_the_cached_view_output_directory(tmp_path,catalog_replay_library):
    args=arguments(tmp_path)
    config=resident_configuration('standard-device',statistics=False,interval=1,profile='velocity_slice')
    initial=image_product();initial['schedule']['interval']=1
    request=deepcopy(initial);request['enable_request']=2
    case,_=run_case(args,ROOT,'gpu',1,'new-segment',4,grid='32,32,32',insitu_config=config,
        product_catalog_records=[initial],product_poll_steps=1,catalog_replay=(catalog_replay_library,{4:[request]}))
    histories=sorted([int(p.name.split('.')[1][4:]) for p in sorted(segment.glob('*.jpeg'))]
                     for segment in image_segments(case))
    assert histories==[[0,1,2],[3,4]]


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','compatible'])
def test_existing_mean_images_keep_the_real_statistics_measure(tmp_path,pipeline):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    means=[]
    for source in ('reynolds','favre'):
        item=image_product(source);item['source']=source;item['statistics_source']='velocity-time'
        item['schedule']['interval']=4
        means.append(item)
    case,_=run_case(args,ROOT,'gpu',2,'mean-images',12,grid='32,32,32',insitu_config=config,
        product_catalog_records=means+[raw_product(item) for item in means])
    for segment in image_segments(case):
        source=segment.parent.parent.name
        assert [p.name for p in sorted(segment.glob('*.jpeg'))]==[
            f'frame.step{step:012d}.jpeg' for step in (1,5,9)]
        records=[json.loads(p.read_text()) for p in segment.glob('frame.*.rank*.json')]
        assert len(records)==6
        raw=next((case/f'outdat/new/slice_products/{source}-data').glob('*/segment*'))
        for record in records:
            statistics=record['statistics']
            assert statistics['provider']=='velocity-time'
            assert statistics['weight']==('density_time_integral' if source=='favre' else 'time_integral')
            assert statistics['requested_window']==[.0005,.0115]
            assert statistics['covered_duration']>0
            assert 'sample_count' not in statistics
        for step in (1,5,9):
            ranges=[r['product']['actual_range'] for r in records if r['step']==step and r['product']['local_points']]
            with h5py.File(raw/f'step{step:012d}.h5') as data:
                values=data['velocity_x'][...]
            np.testing.assert_allclose([min(x[0] for x in ranges),max(x[1] for x in ranges)],
                                       [values.min(),values.max()],rtol=0,atol=2e-10)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','compatible'])
def test_mean_image_exact_continuation_keeps_statistics_and_sampling(tmp_path,pipeline):
    args=arguments(tmp_path);args.statistics=True
    config=resident_configuration(pipeline,statistics=True,interval=1,profile='velocity_slice')
    definitions=[]
    for source in ('reynolds','favre'):
        item=image_product(source);item.update(source=source,statistics_source='velocity-time')
        item['schedule']['interval']=4
        definitions.append(item)
    common=dict(grid='32,32,32',insitu_config=config,checkpoint_interval=5,product_catalog_records=definitions)
    continuous,_=run_case(args,ROOT,'gpu',2,'mean-continuous',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'mean-prefix',5,**common)
    resumed,_=run_case(args,ROOT,'gpu',2,'mean-resumed',12,
        restore=first/'outdat/new/checkpoints/step000000000005',**common)
    for segment in image_segments(resumed):
        original=next(path for path in image_segments(continuous) if path.parent.parent.name==segment.parent.parent.name)
        assert [p.name for p in segment.glob('*.jpeg')]==['frame.step000000000009.jpeg']
        with Image.open(segment/'frame.step000000000009.jpeg') as a,Image.open(original/'frame.step000000000009.jpeg') as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        for rank in (0,1):
            filename=f'frame.step000000000009.rank{rank:08d}.json'
            a=json.loads((segment/filename).read_text());b=json.loads((original/filename).read_text())
            assert a['statistics']==b['statistics'] and a['definition']==b['definition']
    for name in ('state.h5','statistics.h5'):
        compare_fields(continuous/'outdat/new/checkpoints/step000000000012'/name,
                       resumed/'outdat/new/checkpoints/step000000000012'/name)


def test_static_curved_index_surface_uses_physical_coordinates(tmp_path):
    args=arguments(tmp_path);args.scale_timestep=2e-5;args.maximum_cfl=.5
    config=resident_configuration('standard-device',statistics=False,interval=1,profile='curve_demo')
    definition=image_product()
    definition['sampling']=dict(mode='grid_index',axis='j',index=4,
        index_ranges=dict(i=[0,32],k=[0,32]))
    definition['display'].update(camera=[13.,11.,12.],focal_point=[np.pi,np.pi/4.,np.pi],
        view_up=[0.,1.,0.],parallel_scale=5.)
    case,_=run_case(args,ROOT,'gpu',2,'curved-index',2,grid='32,32,32',insitu_config=config,
        tgv_mapping='y-wavy',product_catalog_records=[definition,raw_product(definition)])
    segment=image_segments(case)[0]
    assert len(list(segment.glob('*.jpeg')))==2
    raw=next((case/'outdat/new/slice_products/lower-data').glob('*/segment*'))
    with h5py.File(raw/'geometry.h5') as data:
        xyz=data['coordinates'][...]
    assert np.ptp(xyz[...,1])>0.01
    expected=np.array([[xyz[...,d].min(),xyz[...,d].max()] for d in range(3)]).ravel()
    records=[json.loads(path.read_text()) for path in segment.glob('frame.*.rank*.json')]
    for record in records:
        np.testing.assert_allclose(record['physical_bounds'],expected,rtol=0,atol=2e-10)
    assert json.loads((case/'cfl_gate.json').read_text())['maximum']<=.5
