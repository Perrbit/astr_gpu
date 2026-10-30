"""Explicit host processing reuses the existing CPU diagnostics and VTK filters."""
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from run_output_restart_validation import run_case,compare_fields
from test_runtime_catalog_images import arguments,dynamic_image,image_segments,source_velocity,sample_velocity,catalog_replay_library
from test_runtime_catalog_geometry import definition,segments,read,assert_native,compare_geometry
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import ROOT

EXE=os.environ.get('ASTR_RUNTIME_HOST_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root host-processing/Catalyst executable')


def params(path):
    args=arguments(path);args.executable=Path(EXE).resolve();return args


def options(statistics=False):
    result=resident_configuration('compatible',statistics=statistics,interval=1,profile='all').replace(
        "processing_backend='device'","processing_backend='host'").replace(
        "derivative_backend='gpu'","derivative_backend='cpu'").replace("postprocess_transport='pinned',",'')
    if installed:=os.environ.get('ASTR_RUNTIME_HOST_INSTALLED_PIPELINE'):
        assert Path(installed).is_file()
        result=result.replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),installed)
        if template:=os.environ.get('ASTR_RUNTIME_HOST_INSTALLED_CONFIG'):
            from test_output_insitu_restart import BACKEND
            assert not statistics
            result=Path(template).read_text().replace('@CATALYST_IMPLEMENTATION_PATH@',str(BACKEND)).replace(
                '@ASTR_INSITU_TOOLS@',str(Path(installed).parent))
    return result


@pytest.mark.parametrize('ranks',[1,2])
def test_all_host_kinds_share_extraction_and_keep_native_fields(tmp_path,ranks):
    args=params(tmp_path);config=options()
    items=[]
    for kind in ('slice','physical_plane','q_surface','streamlines'):
        items.extend([dynamic_image(kind,'picture-'+kind),definition(kind,'archive-'+kind)])
    off,_=run_case(args,ROOT,'gpu',ranks,'off',2,grid='32,32,32',insitu_config=config,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',ranks,'host',2,grid='32,32,32',insitu_config=config,product_catalog_records=items,
        memcheck=ranks==2 and os.environ.get('ASTR_RUNTIME_HOST_MEMCHECK')=='1')
    final='outdat/new/checkpoints/step000000000002'
    compare_fields(off/final/'state.h5',case/final/'state.h5')
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in state['products']),state
    assert len(list((case/'outdat').rglob('*.jpeg')))==8
    for kind in ('slice','physical_plane','q_surface'): assert_native(case,kind,'archive-'+kind)
    segment=next(path for path in segments(case) if path.parents[1].name=='archive-streamlines')
    poly,xyz,data,cell=read(segment/'frame.step000000000002.pvtp')
    assert poly.GetFieldData().GetAbstractArray('processing_backend').GetValue(0)=='host'
    expected=sample_velocity(source_velocity(case/final,'instantaneous'),xyz)
    for component,field in enumerate(('velocity_x','velocity_y','velocity_z')):
        np.testing.assert_allclose(data[field],expected[:,component],rtol=0,atol=2e-10)
    assert xyz.dtype==np.float64 and set(cell)=={'SeedIds','IntegrationDirection','ReasonForTermination'}
    assert np.isfinite(data['IntegrationTime']).all()
    assert set(np.unique(cell['SeedIds']))==set(range(16))
    assert set(np.unique(cell['IntegrationDirection']))=={-1,1}
    source_downloads=[]
    for path in image_segments(case):
        for file in path.glob('*.jpeg'):
            with Image.open(file) as image:
                assert image.size==(800,600) and (np.asarray(image.convert('RGB'))<240).any()
        for file in path.glob('frame.*.rank*.json'):
            record=json.loads(file.read_text());assert record['processing_backend']=='host'
            assert record['geometry_d2h_bytes']==0
            source_downloads.append(record['field_d2h_bytes'])
    assert sum(source_downloads)>0
    transfers=re.findall(r'transport=host private_d2h_bytes=(\d+) private_h2d_bytes=(\d+)',(case/'run.log').read_text())
    assert transfers
    assert (sum(int(d)+int(u) for d,u in transfers)>0)==(ranks==2)
    groups=[line for line in (case/'run.log').read_text().splitlines() if line.startswith('ASTR_RUNTIME_HOST_GROUP ')]
    assert len(groups)==8*ranks and all('extractions=1 ' in line for line in groups),groups


def test_host_views_reuse_the_same_native_trajectory(tmp_path):
    args=params(tmp_path);config=options()
    a=dynamic_image('streamlines','first')
    b=deepcopy(a);b['id']='second';b['display']['camera'][0]+=1
    c=deepcopy(a);c['id']='wide';c['display']['line_width']=3.
    raw=definition('streamlines','archive')
    case,_=run_case(args,ROOT,'gpu',2,'views',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=[a,b,c,raw])
    groups=[line for line in (case/'run.log').read_text().splitlines() if line.startswith('ASTR_RUNTIME_HOST_GROUP ')]
    assert len(groups)==4 and all('extractions=1 integrations=1 owner_reuse=2 ' in line for line in groups),groups
    assert len(list((case/'outdat').rglob('*.jpeg')))==6


def test_host_exact_continuation_keeps_existing_means_fields_and_pixels(tmp_path):
    args=params(tmp_path);args.statistics=True;config=options(statistics=True)
    items=[]
    for kind in ('slice','physical_plane','q_surface','streamlines'):
        items.extend([dynamic_image(kind,'picture-'+kind),definition(kind,'archive-'+kind)])
    for item in items: item['schedule']['interval']=4
    for source in ('reynolds','favre'):
        item=definition('streamlines',source)
        item.update(source=source,statistics_source='velocity-time');item['schedule']['interval']=4
        items.append(item)
    common=dict(grid='32,32,32',insitu_config=config,product_catalog_records=items,checkpoint_interval=5)
    full,_=run_case(args,ROOT,'gpu',2,'full',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'prefix',5,**common)
    tail,_=run_case(args,ROOT,'gpu',2,'tail',12,**dict(common,product_catalog_records=[]),
        restore=first/'outdat/new/checkpoints/step000000000005')
    final='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): compare_fields(full/final/name,tail/final/name)
    expected={path.parents[1].name:path for path in segments(full)}
    for path in segments(tail):
        files=sorted(path.glob('*.pvtp'));name=path.parents[1].name
        assert [int(file.name[10:22]) for file in files]==([9] if name in ('reynolds','favre') else [8,12])
        for file in files: compare_geometry(expected[name]/file.name,file)
    pictures={path.parents[1].name:path for path in image_segments(full)}
    for path in image_segments(tail):
        for file in path.glob('*.jpeg'):
            with Image.open(file) as a,Image.open(pictures[path.parents[1].name]/file.name) as b:
                np.testing.assert_array_equal(a,b)


def test_host_existing_mean_velocities_are_sampled_at_the_same_complete_step(tmp_path):
    args=params(tmp_path);args.statistics=True;config=options(statistics=True)
    items=[]
    for source in ('reynolds','favre'):
        item=definition('streamlines',source)
        item.update(source=source,statistics_source='velocity-time');item['schedule']['interval']=1
        items.append(item)
    case,_=run_case(args,ROOT,'gpu',2,'mean-values',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=items)
    ending=case/'outdat/new/checkpoints/step000000000002'
    for path in segments(case):
        source=path.parents[1].name
        _,xyz,data,_=read(path/'frame.step000000000002.pvtp')
        expected=sample_velocity(source_velocity(ending,source),xyz)
        for component,field in enumerate(('velocity_x','velocity_y','velocity_z')):
            np.testing.assert_allclose(data[field],expected[:,component],rtol=0,atol=2e-10)


def test_host_geometry_publication_failure_is_local_and_needs_a_new_request(tmp_path,catalog_replay_library):
    args=params(tmp_path);config=options()
    failed=definition('physical_plane','failed');failed['schedule']['interval']=4
    healthy=dynamic_image('q_surface','healthy')
    frequent=deepcopy(failed);frequent['schedule']['interval']=1
    again=deepcopy(frequent);again['enable_request']=2
    off,_=run_case(args,ROOT,'gpu',2,'off',8,grid='32,32,32',insitu_config=config,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'publish',8,grid='32,32,32',insitu_config=config,
        product_catalog_records=[failed,healthy],product_poll_steps=1,
        geometry_publication_failure=('failed',4,'piece',1),
        catalog_replay=(catalog_replay_library,{6:[frequent,healthy],8:[again,healthy]}))
    final='outdat/new/checkpoints/step000000000008/state.h5';compare_fields(off/final,case/final)
    files=[file for path in segments(case) for file in path.glob('*.pvtp')]
    assert sorted(int(file.name[10:22]) for file in files)==[0,7,8]
    assert not list((case/'outdat/new/geometry_products').rglob('*step000000000004*'))
    assert not list((case/'outdat/new/geometry_products').rglob('*.partial'))
    assert len(list((case/'outdat').rglob('*.jpeg')))==5


def test_host_source_budget_refusal_is_local_and_requires_explicit_reenable(tmp_path,catalog_replay_library):
    args=params(tmp_path);config=options()
    failed=dynamic_image('q_surface','failed');healthy=dynamic_image('slice','healthy')
    again=deepcopy(failed);again['enable_request']=2
    off,_=run_case(args,ROOT,'gpu',2,'off',8,grid='32,32,32',insitu_config=config,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'budget',8,grid='32,32,32',insitu_config=config,
        product_catalog_records=[failed,healthy],product_poll_steps=1,
        runtime_allocation_refusal=('failed',2,1,'host'),
        catalog_replay=(catalog_replay_library,{6:[again,healthy]}))
    final='outdat/new/checkpoints/step000000000008/state.h5';compare_fields(off/final,case/final)
    actual={name:[] for name in ('failed','healthy')}
    for path in image_segments(case):
        actual[path.parents[1].name].extend(int(file.name[10:22]) for file in path.glob('*.jpeg'))
    assert sorted(actual['failed'])==[0,5,7] and sorted(actual['healthy'])==[0,2,4,6,8]
    assert 'Host source exceeds its explicit buffer budget' in (case/'run.log').read_text()


def test_host_wall_charts_and_csv_use_existing_diagnostics(tmp_path):
    from test_runtime_catalog_walls import arguments as wall_arguments,wall_chart,wall_product,wall_frames,rows,chart_segments
    config=resident_configuration('compatible',statistics=True,interval=1,profile='channel_walls').replace(
        "processing_backend='device'","processing_backend='host'").replace(
        "derivative_backend='gpu'","derivative_backend='cpu'").replace("postprocess_transport='pinned',",'')
    items=[];raw_items=[]
    for source in ('instantaneous','reynolds'):
        for field in ('wall_pressure','skin_friction','wall_heat_into_gas'):
            name=source+'-'+field
            chart=wall_chart(name,field);raw=wall_product('raw-'+name,field);raw['schedule']['interval']=2
            for item in (chart,raw):
                item['source']=source
                if source=='reynolds': item['statistics_source']='wall-time'
            items.extend([chart,raw]);raw_items.append(raw)
    args=wall_arguments(tmp_path,'gpu');args.executable=Path(EXE).resolve()
    off,_=run_case(args,ROOT,'gpu',2,'off',4,insitu_config=config,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'host-walls',4,insitu_config=config,product_catalog_records=items)
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'): compare_fields(off/final/name,case/final/name)
    cpu_config=config.replace('render=.true.','render=.false.')
    cpu,_=run_case(args,ROOT,'cpu',2,'cpu-reference',4,insitu_config=cpu_config,product_catalog_records=raw_items)
    for definition in raw_items:
        name=definition['id'];actual=wall_frames(case,name);expected=wall_frames(cpu,name)
        assert [path.name for path in actual]==[path.name for path in expected]
        for a,b in zip(actual,expected):
            va=np.array([[float(row[key]) for key in ('x','y','z','value')] for row in rows(a)])
            vb=np.array([[float(row[key]) for key in ('x','y','z','value')] for row in rows(b)])
            np.testing.assert_allclose(va,vb,rtol=0,atol=2e-10)
        chart_name=name.removeprefix('raw-')
        receipts=list(chart_segments(case,chart_name)[0].glob('frame.*.json'))
        assert len(receipts)==len(actual)
        for file in receipts:
            record=json.loads(file.read_text());assert record['processing_backend']=='host'
