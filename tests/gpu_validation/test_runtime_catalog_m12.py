"""Dynamic products on the existing read-only M12 short fixture, not production."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import h5py
import numpy as np
from PIL import Image
import pytest

from run_m12_local_compare import run
from run_output_restart_validation import compare_fields
from test_runtime_catalog_images import image_product,q_product,line_product,image_segments
from test_index_slice_runtime import product,segments,frames,MPIEXEC,ROOT

EXE=os.environ.get('ASTR_RUNTIME_M12_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root M12-UDF executable')
LIBRARY=Path('/home/dell/workspace/astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst')


def settings(pipeline):
    return f"""&insitu_run
 enabled=t,statistics=f,render=t,initial_frame=f,final_frame=f,
 products='m12_diagnostics',streamline_seeds='bl-layered256',
 processing_backend='device',derivative_backend='gpu',rendering_pipeline='{pipeline}',
 postprocess_transport='pinned',output_directory='outdat/render',
 host_budget_bytes=4294967296,device_budget_bytes=2147483648,device_reserve_bytes=1073741824,
 schedule_mode='steps',step_interval=1,
 implementation_path='{LIBRARY}',pipeline_file='{ROOT}/scripts/insitu/device_render_pipeline.py'
/
"""


def arguments(output,memcheck=False):
    return SimpleNamespace(output=output,source=ROOT.parent/'M12_C13',executable=Path(EXE).resolve(),
        mpiexec=MPIEXEC,diagnostics=False,capture_rhs=False,memcheck=memcheck,sanitizer='compute-sanitizer',
        halo_transport='pinned',insitu_write_eps='0',directory_budget_bytes=256*1024**2,
        runtime_timeout_seconds=300)


def definitions():
    cut=image_product('section',12,field='u_Uinf')
    cut['sampling']['index_ranges']=dict(i=[0,64],j=[0,32])
    cut['display'].update(camera=[550.,75.,1400.],focal_point=[550.,75.,45.],view_up=[0.,1.,0.],
        parallel_scale=200.,color_range=[0.,1.1])
    physical=deepcopy(cut);physical['id']='physical-section'
    physical['sampling']=dict(mode='physical_plane',origin=[0.,0.,45.],normal=[0.,0.,1.],
        bounds=[0.,1100.,0.,150.,0.,90.])
    iso=q_product('vortices',.001,field='u_Uinf')
    iso['sampling']['bounds']=[0.,1100.,0.,150.,0.,90.]
    iso['display']=deepcopy(cut['display']);iso['display'].update(camera=[550.,350.,1000.])
    line=line_product('trajectories',field='u_Uinf')
    line['sampling']['bounds']=[0.,1100.,0.,150.,0.,90.]
    line['display']=deepcopy(iso['display'])
    line['extraction'].update(seed_layout='bl-layered64',initial_step=.5,minimum_step=.01,maximum_step=5.,
        maximum_length=1200.,maximum_steps=4096)
    data=product('numerical',fields=['density','velocity_x','velocity_y','velocity_z','pressure','Q_rs'])
    data['sampling']=deepcopy(cut['sampling'])
    probe=dict(id='pressure-monitor',kind='probe',enable_request=1,source='instantaneous',fields=['pressure'],
        sampling=dict(mode='grid_node',indices=[4,2,12]),output=dict(channel='data',format='csv'))
    items=[cut,physical,iso,line,data,probe]
    for item in items: item['schedule']=dict(mode='steps',interval=5)
    return items


def checkpoint(case,step=10):
    return case/f'outdat/new/checkpoints/step{step:012d}/state.h5'


def test_public_m12_preparation_and_native_inspection(tmp_path):
    from run_m12_insitu_acceptance import configuration,m12_short_products,check_images
    args=arguments(tmp_path);args.capture_rhs=False
    config=configuration(LIBRARY)
    off=run(args,'gpu','2,1,1',case_label='off',steps=10,checkpoint_interval=5,exact_restart=True)
    case=run(args,'gpu','2,1,1',insitu_config=config,case_label='native-entry',steps=10,
        checkpoint_interval=5,exact_restart=True,product_catalog_records=m12_short_products())
    compare_fields(checkpoint(off),checkpoint(case))
    result=check_images(case,2)
    assert len(result['products'])==4 and len(result['frames'])==24 and result['publication_complete'],result
    assert_only_requested_products(case)


def assert_only_requested_products(case):
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    assert len(image_segments(case))==4
    for segment in image_segments(case):
        assert len(list(segment.glob('*.jpeg')))==3
        for path in segment.glob('*.jpeg'):
            with Image.open(path) as image: assert image.size==(800,600)
        for path in segment.glob('frame.*.rank*.json'):
            record=json.loads(path.read_text())
            assert record['field_d2h_bytes']==record['geometry_host_bytes']==0
    assert not list((case/'outdat/render').glob('*.jpeg'))
    assert not (case/'outdat/render/probes.csv').exists()
    assert not list((case/'outdat/render').glob('wall.*.csv'))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_m12_dynamic_families_preserve_original_blowing_and_state(tmp_path,pipeline):
    args=arguments(tmp_path);config=settings(pipeline);items=definitions()
    off=run(args,'gpu','2,1,1',insitu_config=config,case_label='off',steps=10,
        checkpoint_interval=5,exact_restart=True,product_catalog_records=[])
    case=run(args,'gpu','2,1,1',insitu_config=config,case_label='on',steps=10,
        checkpoint_interval=5,exact_restart=True,product_catalog_records=items)
    compare_fields(checkpoint(off),checkpoint(case));assert_only_requested_products(case)
    text=(case/'run.log').read_text();assert 'legacy_random' in text
    assert text.count('ASTR_RUNTIME_LINE_HALO ')==6
    assert 'selected_field_d2h_bytes=0' in text


def test_m12_data_diagnostics_match_the_same_topology_cpu_source(tmp_path):
    args=arguments(tmp_path);data=[item for item in definitions() if item['output']['channel']=='data']
    cpu=run(args,'cpu','2,1,1',case_label='cpu',steps=10,checkpoint_interval=5,
        exact_restart=True,product_catalog_records=data)
    gpu=run(args,'gpu','2,1,1',case_label='gpu',steps=10,checkpoint_interval=5,
        exact_restart=True,product_catalog_records=data)
    maxima={}
    for left,right in zip(frames(segments(cpu)[0]),frames(segments(gpu)[0]),strict=True):
        with h5py.File(left) as a,h5py.File(right) as b:
            for field in data[0]['fields']:
                difference=float(np.max(np.abs(a[field][...]-b[field][...])))
                maxima[field]=max(maxima.get(field,0.),difference)
                assert difference<=2e-10,(field,difference)
    with h5py.File(checkpoint(cpu)) as a,h5py.File(checkpoint(gpu)) as b:
        # Global q datasets cover the authoritative physical fields/caches.
        # rank_extras also preserves unused outer halo storage for exact same-backend restoration.
        fields=[key for key in a if key.startswith('q') and len(key)==5]
        assert len(fields)==11
        for key in fields:
            assert np.isfinite(a[key][...]).all() and np.isfinite(b[key][...]).all(),key
            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=2e-10,err_msg=key)
    (tmp_path/'source_errors.json').write_text(json.dumps(maxima,indent=2))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_m12_exact_resume_retains_dynamic_catalog_and_original_rng(tmp_path,pipeline):
    args=arguments(tmp_path);config=settings(pipeline);items=definitions()
    full=run(args,'gpu','2,1,1',insitu_config=config,case_label='full',steps=10,
        checkpoint_interval=5,exact_restart=True,product_catalog_records=items)
    prefix=run(args,'gpu','2,1,1',insitu_config=config,case_label='prefix',steps=5,
        checkpoint_interval=5,exact_restart=True,product_catalog_records=items)
    source=prefix/'outdat/new/checkpoints/step000000000005'
    tail=run(args,'gpu','2,1,1',insitu_config=config,case_label='tail',steps=10,
        checkpoint_interval=5,exact_restart=True,restore=source,product_catalog_records=[])
    compare_fields(checkpoint(full),checkpoint(tail))
    for segment in image_segments(tail):
        pictures=list(segment.glob('*.jpeg'));assert len(pictures)==1
        picture=pictures[0];assert 'step000000000010' in picture.name
        original=next(path for path in (full/'outdat/new/render_products').rglob(picture.name)
            if path.parents[2].name==picture.parents[2].name)
        with Image.open(picture) as a,Image.open(original) as b:
            np.testing.assert_array_equal(a,b)
    for path in (tail/'outdat/new').rglob('*.eps'): pytest.fail(str(path))


def test_m12_new_capture_and_trace_lifetimes_under_memcheck(tmp_path):
    args=arguments(tmp_path,memcheck=True)
    case=run(args,'gpu','2,1,1',insitu_config=settings('standard-device'),case_label='memcheck',steps=5,
        checkpoint_interval=5,exact_restart=True,product_catalog_records=definitions())
    assert len(list(case.glob('memcheck.*.log')))==2
