"""Dynamic wall curves consume the existing completed-step wall diagnostics."""
import csv
import json
import os
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
import pytest
from PIL import Image

from run_output_restart_validation import run_case,compare_fields
from test_insitu_channel_walls import arguments as channel_arguments,global_component
from test_insitu_channel_statistics import config
from test_output_insitu_restart import ROOT
from test_runtime_catalog_images import catalog_replay_library,image_product
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import product as numerical_product,segments as numerical_segments,frames as numerical_frames


def wall_product(name='pressure',field='wall_pressure',side='lower',index=4):
    return dict(id=name,kind='wall_curve',enable_request=1,source='instantaneous',field=field,
        sampling=dict(mode='wall',axis='j',side=side,spanwise_index=index,
                      bounds=[0.,2*np.pi,-2.,2.,0.,np.pi]),
        output=dict(channel='data',format='csv'),schedule=dict(mode='steps',interval=4))


def wall_chart(name='chart',field='wall_pressure',side='lower',index=4):
    product=wall_product(name,field,side,index)
    limits={'wall_pressure':[0.,12.],'wall_shear_x':[-.01,.01],'wall_heat_into_gas':[-.01,.01],
            'skin_friction':[-.02,.02],'p_pinf':[0.,1.5]}
    product['display']=dict(y_range=limits[field],line_width=2.)
    product['output']=dict(channel='image',formats=['jpeg'],image_size=[800,600],jpeg_quality=95,dpi=300)
    product['schedule']['interval']=2
    return product


def chart_segments(case,name):
    return sorted((case/'outdat/new/render_products'/name).glob('version*/segment*'))


def wall_options(pipeline,**kwargs):
    options=resident_configuration('compatible' if pipeline=='host' else pipeline,**kwargs)
    if pipeline=='host':
        options=options.replace("processing_backend='device'","processing_backend='host'").replace(
            "derivative_backend='gpu'","derivative_backend='cpu'").replace("postprocess_transport='pinned',",'')
    return options


def arguments(output,backend):
    args=channel_arguments(output,backend,samples=False)
    args.executable=Path(os.environ.get('ASTR_RUNTIME_WALL_EXE',ROOT/'build_insitu_device_render/bin/astr'))
    return args


def wall_frames(case,name):
    return sorted((case/'outdat/new/wall_products'/name).glob('version*/segment*/frame.*.csv'))


def rows(path):
    return list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')))


@pytest.mark.parametrize('pipeline,mean',[('standard-device',False),('standard-device',True),('host',False),('host',True)])
def test_optional_wall_source_allocation_refusal_keeps_images_and_cfd(tmp_path,catalog_replay_library,pipeline,mean):
    args=arguments(tmp_path,'gpu')
    options=wall_options(pipeline,statistics=mean,interval=1,profile='channel_walls')
    wall=wall_chart('wall');raw=wall_product('raw');raw['schedule']['interval']=2
    if mean:
        for item in (wall,raw): item.update(source='reynolds',statistics_source='wall-time')
    plane=image_product('plane');plane['sampling']['index_ranges']=dict(i=[0,16],j=[0,16])
    again=deepcopy(wall);again['enable_request']=2
    again_raw=deepcopy(raw);again_raw['enable_request']=2
    off,_=run_case(args,ROOT,'gpu',2,'off',8,insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'source-refusal',8,insitu_config=options,
        product_catalog_records=[wall,raw,plane],product_poll_steps=1,
        wall_capture_refusal=(3 if mean else 2,1,'mean' if mean else 'capture'),
        catalog_replay=(catalog_replay_library,{6:[again,again_raw,plane]}),
        memcheck=os.environ.get('ASTR_RUNTIME_WALL_SOURCE_MEMCHECK')=='1')
    final='outdat/new/checkpoints/step000000000008/state.h5'
    compare_fields(off/final,case/final)
    actual=[]
    for segment in chart_segments(case,'wall'):
        actual.extend(int(path.name[10:22]) for path in segment.glob('*.jpeg'))
    expected=[1,5,7] if mean else [0,5,7]
    assert sorted(actual)==expected
    assert sorted(int(path.name[10:22]) for path in wall_frames(case,'raw'))==expected
    planes=sorted((case/'outdat/new/render_products/plane').glob('*/*/*.jpeg'))
    assert [int(path.name[10:22]) for path in planes]==[0,2,4,6,8]
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert all(row['state']=='active' for row in status.values()),status
    assert all(status[name]['consumed_enable_request']==2 for name in ('wall','raw'))
    assert 'optional ' in (case/'run.log').read_text() and 'allocation' in (case/'run.log').read_text()
    if mean: compare_fields(off/final.replace('state.h5','statistics.h5'),case/final.replace('state.h5','statistics.h5'))


@pytest.mark.parametrize('ranks,pipeline',[(1,'standard-device'),(2,'standard-device'),(2,'direct-device')])
def test_wall_charts_share_the_csv_line_and_preserve_the_solver(tmp_path,ranks,pipeline):
    args=arguments(tmp_path,'gpu')
    options=resident_configuration(pipeline,statistics=True,interval=1,profile='channel_walls')
    if installed:=os.environ.get('ASTR_RUNTIME_WALL_INSTALLED_PIPELINE'):
        original=str(ROOT/'scripts/insitu/device_render_pipeline.py')
        assert original in options and Path(installed).is_file()
        options=options.replace(original,installed)
    a=wall_chart();b=deepcopy(a);b['id']='wide';b['display'].update(y_range=[0.,20.],line_width=3.)
    raw=wall_product();raw['schedule']['interval']=2
    plane=image_product('plane');plane['sampling']['index_ranges']=dict(i=[0,16],j=[0,16])
    baseline,_=run_case(args,ROOT,'gpu',ranks,'baseline',2,insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',ranks,'wall-charts',2,insitu_config=options,
        product_catalog_records=[a,b,raw,plane],memcheck=os.environ.get('ASTR_RUNTIME_WALL_CHART_MEMCHECK')=='1')
    end='outdat/new/checkpoints/step000000000002'
    for name in ('state.h5','statistics.h5'): compare_fields(baseline/end/name,case/end/name)
    for name in ('chart','wide'):
        segment=chart_segments(case,name)[0]
        assert len(list(segment.glob('*.jpeg')))==2
        assert not list(segment.glob('*.csv')) and not list(segment.glob('*.eps'))
        for path in segment.glob('*.jpeg'):
            with Image.open(path) as image: assert image.size==(800,600)
        for path in segment.glob('frame.*.json'):
            record=json.loads(path.read_text());assert record['chart_backend']=='existing-host-matplotlib'
            assert record['host_curve_bytes']==17*4*8
            step=record['step'];table=rows(wall_frames(case,'pressure')[step//2])
            value=np.array([float(row['value']) for row in table])
            np.testing.assert_array_equal(record['product']['actual_range'],[value.min(),value.max()])
            assert record['product']['y_range']==([0.,12.] if name=='chart' else [0.,20.])
            assert record['product']['lighting_enabled'] is False
    assert len(list((case/'outdat').rglob('*.jpeg')))==6
    assert all('selected_lines=1' in line for line in (case/'run.log').read_text().splitlines()
               if line.startswith('ASTR_RUNTIME_WALL_GROUP'))
    assert all(item['state']=='active' for item in json.loads((case/'outdat/new/products.status.json').read_text())['products'])


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','compatible','host'])
@pytest.mark.parametrize('mean',[False,True])
def test_wall_chart_exact_continuation_preserves_pixels_and_existing_mean(tmp_path,pipeline,mean):
    args=arguments(tmp_path,'gpu')
    options=wall_options(pipeline,statistics=True,interval=1,profile='channel_walls')
    definition=wall_chart();definition['schedule']['interval']=4
    if mean: definition.update(source='reynolds',statistics_source='wall-time')
    common=dict(insitu_config=options,topology=(1,2,1),checkpoint_interval=5,product_catalog_records=[definition])
    continuous,_=run_case(args,ROOT,'gpu',2,'continuous',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'first',5,**common)
    resumed,_=run_case(args,ROOT,'gpu',2,'resumed',12,**dict(common,product_catalog_records=[]),
        restore=first/'outdat/new/checkpoints/step000000000005')
    expected=chart_segments(continuous,'chart')[0]
    actual=chart_segments(resumed,'chart')[0]
    frames=sorted(actual.glob('*.jpeg'))
    assert [int(p.name[10:22]) for p in frames]==([9] if mean else [8,12])
    for path in frames:
        with Image.open(path) as a,Image.open(expected/path.name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        a=json.loads(path.with_suffix('.rank00000000.json').read_text())
        b=json.loads((expected/path.name).with_suffix('.rank00000000.json').read_text())
        assert a['definition']==b['definition'] and a['statistics']==b['statistics']
        if mean:
            assert a['statistics']['provider']=='wall-time'
            assert a['statistics']['weight']=='time_integral'
    end='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): compare_fields(continuous/end/name,resumed/end/name)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','compatible'])
def test_wall_chart_display_updates_and_bad_record_are_instance_local(tmp_path,pipeline,catalog_replay_library):
    args=arguments(tmp_path,'gpu');options=resident_configuration(pipeline,statistics=True,interval=1,profile='channel_walls')
    a=wall_chart();a['schedule']['interval']=4
    b=wall_chart('healthy');b['schedule']['interval']=4
    changed=deepcopy(a);changed['display'].update(y_range=[0.,20.],line_width=3.)
    frequent=deepcopy(changed);frequent['schedule']['interval']=1
    invalid=deepcopy(frequent);invalid['display']['y_range']=[1.,1.]
    baseline,_=run_case(args,ROOT,'gpu',2,'baseline',8,insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'hot-charts',8,insitu_config=options,product_poll_steps=1,
        product_catalog_records=[a,b],catalog_replay=(catalog_replay_library,
        {2:[changed,b],3:[frequent,b],4:[invalid,b],5:[frequent,b]}))
    histories={int(segment.parent.name.removeprefix('version')):
        [int(p.name[10:22]) for p in sorted(segment.glob('*.jpeg'))] for segment in chart_segments(case,'chart')}
    assert histories=={1:[0],2:[1,2,4,5,6,7,8]}
    assert [int(p.name[10:22]) for p in sorted(chart_segments(case,'healthy')[0].glob('*.jpeg'))]==[0,4,8]
    for path in chart_segments(case,'chart')[-1].glob('frame.*.json'):
        record=json.loads(path.read_text())
        assert record['product']['y_range']==[0.,20.] and record['product']['line_width']==3.
    end='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'): compare_fields(baseline/end/name,case/end/name)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','compatible'])
def test_wall_chart_image_io_failure_requires_a_new_request(tmp_path,pipeline,catalog_replay_library):
    args=arguments(tmp_path,'gpu');options=resident_configuration(pipeline,statistics=True,interval=1,profile='channel_walls')
    a=wall_chart();a['schedule']['interval']=4
    b=wall_chart('healthy');b['schedule']['interval']=4
    frequent=deepcopy(a);frequent['schedule']['interval']=1
    retry=deepcopy(frequent);retry['enable_request']=2
    baseline,_=run_case(args,ROOT,'gpu',2,'baseline',8,insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'failed-charts',8,insitu_config=options,product_poll_steps=1,
        product_catalog_records=[a,b],catalog_replay=(catalog_replay_library,{6:[frequent,b],8:[retry,b]}),
        image_publication_failure=('chart',4))
    assert sorted([int(p.name[10:22]) for p in sorted(segment.glob('*.jpeg'))]
                  for segment in chart_segments(case,'chart'))==[[0],[7,8]]
    assert [int(p.name[10:22]) for p in sorted(chart_segments(case,'healthy')[0].glob('*.jpeg'))]==[0,4,8]
    assert not list((case/'outdat/new/render_products').rglob('*.partial'))
    status={x['id']:x for x in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['chart']['state']==status['healthy']['state']=='active'
    assert status['chart']['consumed_enable_request']==2 and status['chart']['version']==1
    end='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'): compare_fields(baseline/end/name,case/end/name)


def test_compatible_wall_chart_and_explicit_geometry_consumer_share_the_session(tmp_path):
    args=arguments(tmp_path,'gpu')
    options=resident_configuration('compatible',statistics=False,interval=1,profile='channel_walls')
    picture=wall_chart();data=wall_product()
    plane=image_product('plane');plane['sampling']['index_ranges']=dict(i=[0,16],j=[0,16])
    case,_=run_case(args,ROOT,'gpu',2,'compatible-wall-only',2,insitu_config=options,
        product_catalog_records=[picture,data,plane])
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['chart']['state']==status['pressure']['state']=='active'
    assert status['plane']['state']=='active'
    assert len(list((case/'outdat').rglob('*.jpeg')))==4
    assert wall_frames(case,'pressure')
    for path in chart_segments(case,'chart')[0].glob('frame.*.json'):
        record=json.loads(path.read_text())
        assert record['rendering_pipeline']=='compatible'
        assert record['chart_backend']=='existing-host-matplotlib'


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_empty_wall_chart_keeps_its_own_physical_time_clock(tmp_path,pipeline):
    args=arguments(tmp_path,'gpu');options=resident_configuration(pipeline,statistics=False,interval=1,profile='channel_walls')
    empty=wall_chart('empty');empty['sampling']['bounds'][:2]=[10.,11.]
    empty['schedule']=dict(mode='time',interval=.001)
    healthy=wall_chart('healthy')
    case,_=run_case(args,ROOT,'gpu',2,'empty-chart',2,insitu_config=options,product_catalog_records=[empty,healthy])
    segment=chart_segments(case,'empty')[0]
    assert [int(p.name[10:22]) for p in sorted(segment.glob('*.jpeg'))]==[0,1,2]
    assert len(list(chart_segments(case,'healthy')[0].glob('*.jpeg')))==2
    for path in segment.glob('frame.*.json'):
        record=json.loads(path.read_text())
        assert record['host_curve_bytes']==0 and record['product']['points']==0
        assert record['product']['actual_range'] is None
    assert all(x['state']=='active' for x in json.loads((case/'outdat/new/products.status.json').read_text())['products'])


def test_wall_chart_formats_are_per_instance_not_a_process_default(tmp_path):
    args=arguments(tmp_path,'gpu');args.insitu_write_eps='1'
    options=resident_configuration('direct-device',statistics=False,interval=1,profile='channel_walls')
    paired=wall_chart('paired');paired['output']['formats']=['jpeg','eps']
    jpeg=wall_chart('jpeg')
    case,_=run_case(args,ROOT,'gpu',2,'chart-formats',2,insitu_config=options,product_catalog_records=[paired,jpeg])
    for name,eps_count in (('paired',2),('jpeg',0)):
        segment=chart_segments(case,name)[0]
        assert len(list(segment.glob('*.jpeg')))==2 and len(list(segment.glob('*.eps')))==eps_count
        assert not list(segment.glob('*.csv'))


def test_complete_wall_chart_example_uses_only_its_declared_products(tmp_path):
    args=arguments(tmp_path,'gpu');options=resident_configuration('standard-device',statistics=True,interval=1,profile='channel_walls')
    path=ROOT/'scripts/output/input.products.channel.wall-charts.example.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()][1:]
    case,_=run_case(args,ROOT,'gpu',2,'chart-example',2,insitu_config=options,product_catalog_records=records,
        insitu_timing=True,monitor_resources=True)
    assert len(list((case/'outdat').rglob('*.jpeg')))==4
    assert not list((case/'outdat/new/render_products').rglob('*.csv'))
    for definition in records:
        segment=chart_segments(case,definition['id'])[0]
        assert len(list(segment.glob('*.jpeg')))==1
        receipt=json.loads(next(segment.glob('frame.*.json')).read_text())
        assert receipt['definition']==definition and receipt['product']['actual_range'] is not None
        if definition['source']=='reynolds': assert receipt['statistics']['provider']=='wall-time'
    log=(case/'run.log').read_text()
    assert 'ASTR_RUNTIME_STAGE_TIMING' in log
    stages=[]
    for line in log.splitlines():
        if not line.startswith('ASTR_INSITU_STAGE_TIMING runtime_wall_'): continue
        _,stage,step,rank,seconds=line.split()
        seconds=float(seconds);assert np.isfinite(seconds) and seconds>=0
        stages.append(dict(stage=stage,step=int(step),rank=int(rank),seconds=seconds))
    assert {row['stage'] for row in stages}=={
        'runtime_wall_capture','runtime_wall_select_reduce','runtime_wall_publication_inclusive'}
    (case/'runtime_wall_cost.json').write_text(json.dumps(dict(
        scope='rank-local capture, shared line selection/reduction and publication including consumer; do not sum ranks',
        records=stages),indent=2))
    assert (case/'resources.sampled.json').is_file()


@pytest.mark.parametrize('pipeline',['standard-device','direct-device','host'])
def test_wall_chart_existing_curved_wall_geometry(tmp_path,pipeline):
    args=arguments(tmp_path,'gpu');args.case='tgv';args.initial_dimension=0
    args.scale_timestep=2e-5;args.maximum_cfl=.5
    options=wall_options(pipeline,statistics=False,interval=1,profile='channel_walls')
    picture=wall_chart();picture['display']['y_range']=[0.,5000.]
    raw=wall_product();raw['schedule']['interval']=2
    oracle=numerical_product('pressure-oracle',fields=['pressure'])
    oracle['sampling']=dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,32],k=[4,4]))
    oracle['schedule']['interval']=2
    masked=deepcopy(picture);masked['id']='masked';masked['sampling']['bounds'][2:4]=[-.03,.03]
    masked_data=deepcopy(raw);masked_data['id']='masked-data';masked_data['sampling']=deepcopy(masked['sampling'])
    common=dict(insitu_config=options,grid='32,32,32',tgv_mapping='y-wavy')
    baseline,_=run_case(args,ROOT,'gpu',2,'baseline',2,**common,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'curved-wall',2,**common,product_catalog_records=[picture,raw,oracle,masked,masked_data])
    end='outdat/new/checkpoints/step000000000002/state.h5';compare_fields(baseline/end,case/end)
    assert json.loads((case/'cfl_gate.json').read_text())['maximum']<=.5
    for path in chart_segments(case,'chart')[0].glob('frame.*.json'):
        record=json.loads(path.read_text());step=record['step']
        values=np.array([float(row['value']) for row in rows(wall_frames(case,'pressure')[step//2])])
        assert len(values)==33
        np.testing.assert_array_equal(record['product']['actual_range'],[values.min(),values.max()])
        with h5py.File(numerical_frames(numerical_segments(case)[0])[step//2]) as state:
            np.testing.assert_allclose(values,state['pressure'][...].ravel(),rtol=0,atol=2e-10)
        if step==2:
            with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
                q=np.stack([state[f'q{c:04d}'][...].transpose(2,1,0)[:,0,4] for c in range(1,6)],axis=-1)
            expected=(q[:,4]-.5*np.sum(q[:,1:4]**2,axis=1)/q[:,0])/2.5
            np.testing.assert_allclose(values,expected,rtol=0,atol=2e-10)
    for path in chart_segments(case,'masked')[0].glob('frame.*.json'):
        record=json.loads(path.read_text());step=record['step']
        table=rows(wall_frames(case,'masked-data')[step//2])
        indices=np.array([int(row['global_i']) for row in table])
        assert len(indices)==7 and record['product']['points']==7
        assert record['connectivity_breaks']==int(np.count_nonzero(np.diff(indices)>1))==2


@pytest.mark.skipif(not os.environ.get('ASTR_RUNTIME_WALL_AIR5_EXE'),reason='Select the current root-built AIR5 executable')
@pytest.mark.parametrize('pipeline',['standard-device','direct-device','host'])
def test_wall_chart_existing_air5_pressure_and_empty_wall_rank(tmp_path,pipeline):
    from test_insitu_air5_walls import arguments as air5_arguments
    from run_output_air5_restart_validation import launch
    args=air5_arguments(tmp_path,(1,2,1));args.executable=Path(os.environ['ASTR_RUNTIME_WALL_AIR5_EXE'])
    options=wall_options(pipeline,statistics=False,interval=1,profile='air5_walls')
    picture=wall_chart();picture['sampling']['bounds']=[0.,.08,0.,.01,0.,.002]
    picture['display']['y_range']=[0.,1e6]
    raw=wall_product();raw['sampling']=deepcopy(picture['sampling']);raw['schedule']['interval']=2
    oracle=numerical_product('pressure-oracle',fields=['pressure'])
    oracle['sampling']=dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,16],k=[4,4]))
    oracle['schedule']['interval']=2
    baseline,_=launch(args,'gpu',2,'baseline',2,interval=2,insitu_config=options,product_catalog_records=[],
        directory_budget_bytes=256*1024**2)
    case,_=launch(args,'gpu',2,'air5-wall-chart',2,interval=2,insitu_config=options,
        product_catalog_records=[picture,raw,oracle],directory_budget_bytes=256*1024**2)
    end='outdat/new/checkpoints/step000000000002/state.h5';compare_fields(baseline/end,case/end)
    segment=chart_segments(case,'chart')[0];assert len(list(segment.glob('*.jpeg')))==2
    for path in segment.glob('frame.*.json'):
        record=json.loads(path.read_text());step=record['step']
        values=np.array([float(row['value']) for row in rows(wall_frames(case,'pressure')[step//2])])
        assert len(values)==17 and record['nondimensional'] is False
        np.testing.assert_array_equal(record['product']['actual_range'],[values.min(),values.max()])
        with h5py.File(numerical_frames(numerical_segments(case)[0])[step//2]) as state:
            np.testing.assert_allclose(values,state['pressure'][...].ravel(),rtol=0,atol=2e-10)
        if step==2:
            with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
                expected=state['q0027'][...].transpose(2,1,0)[:,0,4]
            np.testing.assert_allclose(values,expected,rtol=0,atol=2e-10)


def reference(case,step,side,index):
    path=case/f'outdat/new/checkpoints/step{step:012d}/state.h5'
    q=np.stack([global_component(path,f'q{c:04d}') for c in range(1,6)],axis=-1)
    geometry=case/'outdat/new/resources/geometry.h5'
    xyz=np.stack([global_component(geometry,f'q{c:04d}') for c in range(1,4)],axis=-1)
    metric=global_component(geometry,'q0009')
    j=0 if side=='lower' else 16;sign=1 if side=='lower' else -1
    points=q[:,[j+sign*n for n in range(3)],index]
    rho=points[...,0];velocity=points[...,1:4]/rho[...,None]
    p=(points[...,4]-.5*rho*np.sum(velocity*velocity,axis=-1))/2.5
    t=p/rho*(1.4*.3**2)
    s=110.4/273.15
    mu=t[:,0]*np.sqrt(t[:,0])*(1+s)/(t[:,0]+s)/3000.
    du=(-.5*velocity[:,2,0]+2*velocity[:,1,0]-1.5*velocity[:,0,0])*metric[:,j,index]
    dt=(-.5*t[:,2]+2*t[:,1]-1.5*t[:,0])*metric[:,j,index]
    return xyz[:,j,index],np.stack([p[:,0],mu*du,-(mu/.72)/((1.4-1)*.3**2)*dt],axis=-1)


@pytest.mark.parametrize('backend,topology',[('cpu',(1,1,1)),('gpu',(1,1,1)),('cpu',(2,1,1)),
    ('gpu',(2,1,1)),('gpu',(1,2,1)),('gpu',(1,1,2))])
def test_wall_csv_matches_existing_diagnostic_and_does_not_change_state(tmp_path,backend,topology):
    args=arguments(tmp_path,backend);ranks=int(np.prod(topology))
    definitions=[wall_product(),wall_product('shear','wall_shear_x'),
                 wall_product('heat','wall_heat_into_gas'),wall_product('upper','wall_pressure','upper'),
                 wall_product('Cf','skin_friction'),wall_product('pressure_ratio','p_pinf')]
    baseline,_=run_case(args,ROOT,backend,ranks,'baseline',4,topology=topology,insitu_config=config(False),
        product_catalog_records=[])
    case,_=run_case(args,ROOT,backend,ranks,'wall-curves',4,topology=topology,insitu_config=config(False),
        product_catalog_records=definitions,memcheck=os.environ.get('ASTR_RUNTIME_WALL_MEMCHECK')=='1')
    end='outdat/new/checkpoints/step000000000004/state.h5'
    compare_fields(baseline/end,case/end)
    compare_fields(baseline/end.replace('state.h5','statistics.h5'),case/end.replace('state.h5','statistics.h5'))
    for definition in definitions:
        paths=wall_frames(case,definition['id']);assert len(paths)==2
        frame=paths[-1];table=rows(frame);assert len(table)==17
        xyz,value=reference(case,4,definition['sampling']['side'],4)
        np.testing.assert_array_equal([int(x['global_i']) for x in table],np.arange(17))
        np.testing.assert_array_equal([[float(x[a]) for a in 'xyz'] for x in table],xyz)
        column={'wall_pressure':0,'wall_shear_x':1,'wall_heat_into_gas':2,'skin_friction':1,'p_pinf':0}[definition['field']]
        factor={'skin_friction':2.,'p_pinf':1.4*.3**2}.get(definition['field'],1.)
        np.testing.assert_allclose([float(x['value']) for x in table],value[:,column]*factor,rtol=0,atol=2e-10)
        assert 'sampling=global_k_index; no spanwise average' in frame.read_text()
    log=(case/'run.log').read_text()
    groups=[line for line in log.splitlines() if line.startswith('ASTR_RUNTIME_WALL_GROUP')]
    assert len(groups)==2*ranks and all('selected_lines=2' in line for line in groups)
    status=json.loads((case/'outdat/new/products.status.json').read_text())['products']
    assert all(x['state']=='active' for x in status)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_instantaneous_wall_csv_does_not_require_or_enable_statistics(tmp_path,backend):
    args=arguments(tmp_path,backend)
    options=config(False).replace('statistics=.true.','statistics=.false.')
    case,_=run_case(args,ROOT,backend,2,'no-statistics',4,topology=(2,1,1),insitu_config=options,
        product_catalog_records=[wall_product()])
    assert len(wall_frames(case,'pressure'))==2
    path=case/'outdat/new/checkpoints/step000000000004/statistics.h5'
    if path.exists():
        with h5py.File(path) as state: assert 'wall_statistics' not in state
    assert not list((case/'outdat').rglob('*.jpeg'))


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('mean',[False,True])
def test_wall_exact_continuation_keeps_definition_clock_and_existing_means(tmp_path,backend,mean):
    args=arguments(tmp_path,backend);definition=wall_product()
    if mean: definition.update(source='reynolds',statistics_source='wall-time')
    common=dict(topology=(1,2,1),insitu_config=config(False),checkpoint_interval=5,
                product_catalog_records=[definition])
    continuous,_=run_case(args,ROOT,backend,2,'continuous',12,**common,
        memcheck=os.environ.get('ASTR_RUNTIME_WALL_MEMCHECK')=='1')
    first,_=run_case(args,ROOT,backend,2,'first',5,**common)
    conflicting=deepcopy(definition);conflicting['sampling']['spanwise_index']=13
    resumed,_=run_case(args,ROOT,backend,2,'resumed',12,**dict(common,product_catalog_records=[conflicting]),
        restore=first/'outdat/new/checkpoints/step000000000005')
    previous={p.name:p for p in wall_frames(continuous,'pressure')}
    actual=wall_frames(resumed,'pressure')
    assert [int(p.name[10:22]) for p in actual]==[8,12]
    for path in actual: assert path.read_bytes()==previous[path.name].read_bytes()
    end='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): compare_fields(continuous/end/name,resumed/end/name)
    if mean:
        path=actual[-1]
        with h5py.File(resumed/end/'statistics.h5') as state:
            expected=state['wall_statistics/q0010'][...].transpose(2,1,0)[np.arange(17)%16,0,4]
        np.testing.assert_allclose([float(x['value']) for x in rows(path)],expected,rtol=0,atol=2e-10)
        assert 'source=reynolds' in path.read_text()
        clock=next(line for line in path.read_text().splitlines() if line.startswith('# requested_window'))
        coverage=np.fromstring(clock.split('=',1)[1].split(' samples=')[0],sep=' ')
        np.testing.assert_allclose(coverage,[.003,.0115,.0085],rtol=0,atol=1e-15)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_hot_updates_reuse_the_effective_control_registry(tmp_path,backend,catalog_replay_library):
    args=arguments(tmp_path,backend);a=wall_product();b=wall_product('heat','wall_heat_into_gas')
    moved=deepcopy(a);moved['sampling']['spanwise_index']=8
    frequent=deepcopy(moved);frequent['schedule']['interval']=1
    invalid=deepcopy(frequent);invalid['field']='not_a_quantity'
    publications={2:[moved,b],3:[frequent,b],4:[invalid,b],5:[frequent,b]}
    plain,_=run_case(args,ROOT,backend,2,'baseline',8,topology=(2,1,1),insitu_config=config(False),
        product_catalog_records=[])
    case,_=run_case(args,ROOT,backend,2,'hot-wall',8,topology=(2,1,1),insitu_config=config(False),
        product_catalog_records=[a,b],product_poll_steps=1,catalog_replay=(catalog_replay_library,publications))
    histories={int(p.parent.parent.name.removeprefix('version')):[] for p in wall_frames(case,'pressure')}
    for path in wall_frames(case,'pressure'):
        histories[int(path.parent.parent.name.removeprefix('version'))].append(int(path.name[10:22]))
    assert histories=={1:[0],2:[1,2,4,5,6,7,8]}
    assert [int(p.name[10:22]) for p in wall_frames(case,'heat')]==[0,4,8]
    end='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'): compare_fields(plain/end/name,case/end/name)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_publication_failure_only_stops_its_instance_until_new_request(tmp_path,backend,catalog_replay_library):
    args=arguments(tmp_path,backend);a=wall_product();b=wall_product('heat','wall_heat_into_gas')
    cadence=deepcopy(a);cadence['schedule']['interval']=1
    retry=deepcopy(cadence);retry['enable_request']=2
    plain,_=run_case(args,ROOT,backend,2,'baseline',8,topology=(2,1,1),insitu_config=config(False),
        product_catalog_records=[])
    case,_=run_case(args,ROOT,backend,2,'failed-wall',8,topology=(2,1,1),insitu_config=config(False),
        product_catalog_records=[a,b],product_poll_steps=1,
        catalog_replay=(catalog_replay_library,{6:[cadence,b],8:[retry,b]}),wall_publication_failure=('pressure',4))
    segments=(case/'outdat/new/wall_products/pressure').glob('version*/segment*')
    histories=sorted([int(p.name[10:22]) for p in sorted(segment.glob('frame.*.csv'))] for segment in segments)
    assert histories==[[0],[7,8]]
    assert [int(p.name[10:22]) for p in wall_frames(case,'heat')]==[0,4,8]
    assert not list((case/'outdat/new/wall_products').rglob('*.partial'))
    status={x['id']:x for x in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['pressure']['state']==status['heat']['state']=='active'
    assert status['pressure']['version']==1 and status['pressure']['consumed_enable_request']==2
    end='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'): compare_fields(plain/end/name,case/end/name)
