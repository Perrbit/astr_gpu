"""Explicit compatible rendering of the same native device extractions."""
from copy import deepcopy
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case,compare_fields
from test_runtime_catalog_images import arguments,dynamic_image,image_segments,source_velocity,sample_velocity,catalog_replay_library
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import ROOT
from test_insitu_catalog_products import geometry_product

EXE=os.environ.get('ASTR_RUNTIME_COMPATIBLE_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the current root-built compatible executable')


def params(path):
    result=arguments(path);result.executable=Path(EXE).resolve();return result


def records():
    products=[]
    for kind in ('slice','physical_plane','q_surface','streamlines'):
        item=dynamic_image(kind,kind,field='velocity_x')
        products.append(item)
        changed=deepcopy(item);changed['id']=kind+'-other-camera';changed['display']['camera'][0]+=1.
        products.append(changed)
    return products


def test_all_compatible_kinds_share_native_packets_not_implicit_products(tmp_path):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    if installed:=os.environ.get('ASTR_RUNTIME_COMPATIBLE_INSTALLED_PIPELINE'):
        assert Path(installed).is_file()
        options=options.replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),installed)
    off,_=run_case(args,ROOT,'gpu',2,'off',2,grid='32,32,32',insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'images',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=records(),runtime_compatible_oracle=True)
    final='outdat/new/checkpoints/step000000000002'
    compare_fields(off/final/'state.h5',case/final/'state.h5')
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    assert len(image_segments(case))==8 and len(list((case/'outdat').rglob('*.jpeg')))==16
    reference=source_velocity(case/final,'instantaneous')
    expected_packets=0
    for segment in image_segments(case):
        name=segment.parents[1].name
        for path in segment.glob('frame.*.rank*.json'):
            receipt=json.loads(path.read_text())
            expected_packets+=receipt['product']['local_points']>0
            assert receipt['rendering_pipeline']=='compatible' and receipt['field_d2h_bytes']==0
            assert (receipt['geometry_host_bytes']>0)==(receipt['product']['local_points']>0)
            assert receipt['geometry_d2h_bytes']==(0 if name.endswith('-other-camera') else receipt['geometry_host_bytes'])
        for path in segment.glob('*.jpeg'):
            with Image.open(path) as image:
                assert image.size==(800,600) and np.count_nonzero(np.min(np.asarray(image),axis=2)<200)>1000
        for path in (case/'outdat').glob(f'runtime_compatible_oracle.{name}.step2.rank*.npz'):
            with np.load(path) as packet:
                xyz,quantity,indices=packet['xyz'],packet['quantity'],packet['connectivity']
                assert xyz.dtype==quantity.dtype==np.float64 and indices.dtype==np.int64
                assert np.isfinite(xyz).all() and np.isfinite(quantity).all()
                if len(xyz)==0:
                    assert len(indices)==len(quantity)==0
                    continue
                assert len(indices)%int(packet['arity'])==0 and indices.min()>=0 and indices.max()<len(xyz)
                np.testing.assert_allclose(quantity,sample_velocity(reference,xyz)[:,0],rtol=0,atol=2e-10)
                if name.startswith('slice') or name.startswith('physical_plane'):
                    np.testing.assert_allclose(xyz[:,2],np.pi/4,rtol=0,atol=2e-10)
    assert len(list((case/'outdat').glob('runtime_compatible_oracle.*.npz')))==expected_packets


@pytest.mark.parametrize('kind',['slice','physical_plane','q_surface','streamlines'])
def test_compatible_exact_resume_preserves_effective_definitions_and_pixels(tmp_path,kind):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    item=dynamic_image(kind,kind);item['schedule']['interval']=4
    common=dict(grid='32,32,32',insitu_config=options,checkpoint_interval=5,product_catalog_records=[item])
    full,_=run_case(args,ROOT,'gpu',2,'full',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'first',5,**common)
    tail,_=run_case(args,ROOT,'gpu',2,'tail',12,**dict(common,product_catalog_records=[]),
        restore=first/'outdat/new/checkpoints/step000000000005')
    final='outdat/new/checkpoints/step000000000012/state.h5';compare_fields(full/final,tail/final)
    expected,actual=image_segments(full)[0],image_segments(tail)[0]
    frames=sorted(actual.glob('*.jpeg'));assert [p.name for p in frames]==[
        'frame.step000000000008.jpeg','frame.step000000000012.jpeg']
    for path in frames:
        with Image.open(path) as a,Image.open(expected/path.name) as b:
            np.testing.assert_array_equal(a,b)


def test_compatible_packet_and_retirement_lifetime_under_memcheck(tmp_path):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    run_case(args,ROOT,'gpu',2,'memcheck',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=records(),memcheck=True)


def test_compatible_host_refusal_is_local_and_requires_a_new_request(tmp_path,catalog_replay_library):
    args=params(tmp_path);args.statistics=True
    options=resident_configuration('compatible',statistics=True,interval=1,profile='velocity_slice')
    failed=dynamic_image('q_surface','failed');healthy=dynamic_image('slice','healthy')
    retry=deepcopy(failed);retry['enable_request']=2
    off,_=run_case(args,ROOT,'gpu',2,'off',8,grid='32,32,32',insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'host-refusal',8,grid='32,32,32',insitu_config=options,
        product_catalog_records=[failed,healthy],product_poll_steps=1,
        runtime_allocation_refusal=('failed',2,1,'compatible'),
        catalog_replay=(catalog_replay_library,{6:[retry,healthy]}),memcheck=True)
    final='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'): compare_fields(off/final/name,case/final/name)
    actual={name:[] for name in ('failed','healthy')}
    for segment in image_segments(case):
        actual[segment.parents[1].name].extend(int(path.name[10:22]) for path in segment.glob('*.jpeg'))
    assert sorted(actual['failed'])==[0,5,7] and sorted(actual['healthy'])==[0,2,4,6,8]
    assert 'Explicit compatible geometry exceeds the controlled host-buffer budget' in (case/'run.log').read_text()


def test_compatible_image_publication_failure_and_cadence_do_not_retry(tmp_path,catalog_replay_library):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    failed=dynamic_image('q_surface','failed');failed['schedule']['interval']=4
    healthy=dynamic_image('slice','healthy')
    frequent=deepcopy(failed);frequent['schedule']['interval']=1
    retry=deepcopy(frequent);retry['enable_request']=2
    off,_=run_case(args,ROOT,'gpu',2,'off',8,grid='32,32,32',insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'publish-failed',8,grid='32,32,32',insitu_config=options,
        product_catalog_records=[failed,healthy],product_poll_steps=1,image_publication_failure=('failed',4),
        catalog_replay=(catalog_replay_library,{6:[frequent,healthy],8:[retry,healthy]}))
    final='outdat/new/checkpoints/step000000000008/state.h5';compare_fields(off/final,case/final)
    actual={name:[] for name in ('failed','healthy')}
    for segment in image_segments(case):
        actual[segment.parents[1].name].extend(int(path.name[10:22]) for path in segment.glob('*.jpeg'))
    assert sorted(actual['failed'])==[0,7,8] and sorted(actual['healthy'])==[0,2,4,6,8]
    assert not list((case/'outdat/new/render_products').rglob('*.partial'))


def test_compatible_valid_empty_products_do_not_suspend(tmp_path):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    iso=dynamic_image('q_surface','empty-Q');iso['extraction']['isovalue']=100.
    plane=dynamic_image('physical_plane','outside-plane');plane['sampling']['origin'][2]=20.
    case,_=run_case(args,ROOT,'gpu',2,'empty',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=[iso,plane])
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    for segment in image_segments(case):
        assert len(list(segment.glob('*.jpeg')))==2
        for path in segment.glob('frame.*.json'):
            row=json.loads(path.read_text())
            assert row['geometry_empty'] and row['physical_bounds'] is None
            assert row['geometry_d2h_bytes']==row['geometry_host_bytes']==0


def test_resident_geometry_request_is_reported_without_a_readback_fallback(tmp_path):
    args=params(tmp_path);options=resident_configuration('standard-device',statistics=False,interval=1,profile='velocity_slice')
    geometry=geometry_product();geometry['id']='future-geometry'
    picture=dynamic_image('slice','healthy')
    case,_=run_case(args,ROOT,'gpu',2,'writer-unfinished',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=[geometry,picture])
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['future-geometry']['state']=='invalid-configuration'
    assert 'requires explicit compatible' in json.dumps(status['future-geometry'])
    assert status['healthy']['state']=='active'
    assert len(list((case/'outdat').rglob('*.jpeg')))==2
    assert not list((case/'outdat/new/geometry_products').rglob('*'))
