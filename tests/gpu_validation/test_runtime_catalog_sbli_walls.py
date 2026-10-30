"""Original physical-midspan SBLI wall quantities, with dynamic instances."""
import json
import os
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
from PIL import Image
import pytest

from test_hcw_insitu import launch,configuration_2d,logical_derivative
from run_output_restart_validation import compare_fields
from test_runtime_catalog_walls import rows
from test_runtime_catalog_images import catalog_replay_library

pytestmark=pytest.mark.skipif(not os.environ.get('ASTR_HCW_INSITU_EXE'),reason='Select the current root HCW executable')


def definition(name,field,image=False):
    record=dict(id=name,kind='wall_curve',enable_request=1,source='instantaneous',field=field,
        sampling=dict(mode='wall_midspan',axis='j',side='lower',bounds=[0.,2.,0.,.03,0.,.002]),
        output=dict(channel='data',format='csv'),schedule=dict(mode='steps',interval=1))
    if image:
        record['output']=dict(channel='image',formats=['jpeg'],image_size=[1000,600],jpeg_quality=95,dpi=300)
        record['sampling']['bounds'][:2]=[.67,1.01]
        record['display']=dict(y_range={'p_pinf':[0.,3.],'skin_friction':[-.01,.01],
            'wall_heat_into_wall_normalized':[-.01,.01]}[field],line_width=2.)
    return record


def frames(case,name,image=False):
    family='render_products' if image else 'wall_products'
    suffix='jpeg' if image else 'csv'
    return sorted((case/'outdat/new'/family/name).glob(f'version*/segment*/frame.*.{suffix}'),key=lambda path:path.name)


def reference(case,step):
    with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
        q=np.stack([state[f'q{i:04d}'][...] for i in range(1,6)],axis=-1)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as grid:
        xyz=np.stack([grid[f'q{i:04d}'][...] for i in range(1,4)],axis=-1)
        metric=grid['q0009'][...]
    flow=json.loads((case/'case_manifest.json').read_text())['flow']
    mach=flow['mach'];rho=q[...,0];velocity=q[...,1:4]/rho[...,None]
    p=(q[...,4]-.5*rho*np.sum(velocity*velocity,axis=-1))/2.5
    temperature=p/rho*(1.4*mach*mach)
    du=logical_derivative(velocity[...,0],1)*metric
    dt=logical_derivative(temperature,1)*metric
    s=flow['sutherland_temperature_K']/flow['temperature_K']
    mu=temperature*np.sqrt(temperature)*(1+s)/((temperature+s)*flow['reynolds_reference'])
    values=np.stack([p[:,0,:]*(1.4*mach*mach),2*mu[:,0,:]*du[:,0,:],
        (mu[:,0,:]/flow['prandtl'])/(.4*mach*mach)*dt[:,0,:]],axis=-1)
    z=.5*(xyz[0,0,0,2]+xyz[-1,0,0,2])
    first=np.searchsorted(xyz[:,0,0,2],z,side='right')-1
    fraction=(z-xyz[first,0,0,2])/(xyz[first+1,0,0,2]-xyz[first,0,0,2])
    assert 0.<fraction<1.  # Five spanwise intervals exercise an actual two-layer interpolation.
    return xyz[0,0,:,0],z,(1-fraction)*values[first]+fraction*values[first+1]


@pytest.mark.parametrize('ranks,pipeline',[(1,'standard-device'),(2,'standard-device'),(2,'direct-device')])
def test_original_sbli_midspan_values_share_one_capture_and_do_not_change_flow(tmp_path,ranks,pipeline):
    fields=('p_pinf','skin_friction','wall_heat_into_wall_normalized')
    records=[definition(field,field) for field in fields]+[definition(field+'-picture',field,True) for field in fields]
    config=configuration_2d().replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    baseline=launch(tmp_path/'baseline',ranks,True,steps=2,insitu_config=config,product_catalog_records=[])
    case=launch(tmp_path/'observed',ranks,True,steps=2,insitu_config=config,product_catalog_records=records)
    final='outdat/new/checkpoints/step000000000002/state.h5';compare_fields(baseline/final,case/final)
    x,z,expected=reference(case,2)
    for column,field in enumerate(fields):
        paths=frames(case,field);assert len(paths)==3
        data=np.array([[float(row[name]) for name in ('step','time','global_i','x','y','z','value')]
                       for row in rows(paths[-1])])
        np.testing.assert_array_equal(data[:,2],np.arange(len(x)))
        np.testing.assert_allclose(data[:,3],x,rtol=0,atol=2e-10)
        np.testing.assert_allclose(data[:,5],z,rtol=0,atol=2e-10)
        np.testing.assert_allclose(data[:,6],expected[:,column],rtol=0,atol=2e-10)
        assert 'physical_midspan_interpolation; no spanwise average' in paths[-1].read_text()
    for field in fields:
        pictures=frames(case,field+'-picture',True);assert len(pictures)==3
        for path in pictures:
            with Image.open(path) as image: assert image.size==(1000,600)
    assert len(list((case/'outdat').rglob('*.jpeg')))==9
    log=(case/'run.log').read_text()
    assert log.count('ASTR_RUNTIME_SBLI_WALL_SOURCE rank=')==3*ranks


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_midspan_exact_checkpoint_continuation_preserves_fields_csv_and_pixels(tmp_path,pipeline):
    records=[definition('pressure','p_pinf'),definition('picture','wall_heat_into_wall_normalized',True)]
    for record in records: record['schedule']['interval']=4
    config=configuration_2d().replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    common=dict(insitu_config=config,checkpoint_interval=5,product_catalog_records=records)
    full=launch(tmp_path/'full',2,True,steps=12,**common)
    part=launch(tmp_path/'part',2,True,steps=5,**common)
    tail=launch(tmp_path/'tail',2,True,steps=12,**dict(common,product_catalog_records=[]),
        restore=part/'outdat/new/checkpoints/step000000000005')
    final='outdat/new/checkpoints/step000000000012/state.h5';compare_fields(full/final,tail/final)
    assert [int(p.name[10:22]) for p in frames(tail,'pressure')]==[8,12]
    for path in frames(tail,'pressure'):
        expected=next(p for p in frames(full,'pressure') if p.name==path.name)
        assert path.read_bytes()==expected.read_bytes()
    for path in frames(tail,'picture',True):
        expected=next(p for p in frames(full,'picture',True) if p.name==path.name)
        with Image.open(path) as a,Image.open(expected) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def test_midspan_csv_and_image_failures_are_local_and_need_new_requests(tmp_path,catalog_replay_library):
    a=definition('pressure','p_pinf');b=definition('picture','skin_friction',True)
    healthy=definition('healthy','skin_friction',True)
    for record in (a,b,healthy): record['schedule']['interval']=4
    frequent=deepcopy(a);frequent['schedule']['interval']=1
    frequent_image=deepcopy(b);frequent_image['schedule']['interval']=1
    retry=deepcopy(frequent);retry['enable_request']=2
    retry_image=deepcopy(frequent_image);retry_image['enable_request']=2
    config=configuration_2d()
    full=launch(tmp_path/'baseline',2,True,steps=8,insitu_config=config,product_catalog_records=[])
    case=launch(tmp_path/'failed',2,True,steps=8,insitu_config=config,product_catalog_records=[a,b,healthy],
        product_poll_steps=1,catalog_replay=(catalog_replay_library,
        {6:[frequent,frequent_image,healthy],8:[retry,retry_image,healthy]}),
        wall_publication_failure=('pressure',4),image_publication_failure=('picture',4))
    final='outdat/new/checkpoints/step000000000008/state.h5';compare_fields(full/final,case/final)
    for name,image in (('pressure',False),('picture',True)):
        assert [int(p.name[10:22]) for p in frames(case,name,image)]==[0,7,8]
    assert [int(p.name[10:22]) for p in frames(case,'healthy',True)]==[0,4,8]
    assert not list((case/'outdat/new').rglob('*.partial'))
    status={r['id']:r for r in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert all(r['state']=='active' for r in status.values())
    assert status['pressure']['consumed_enable_request']==status['picture']['consumed_enable_request']==2


def test_midspan_capture_and_publication_lifetime_under_two_rank_memcheck(tmp_path):
    records=[definition('pressure','p_pinf'),definition('picture','skin_friction',True)]
    launch(tmp_path/'memcheck',2,True,steps=2,insitu_config=configuration_2d(),
        product_catalog_records=records,memcheck=True)


def test_midspan_source_refusal_preserves_monitors_and_requires_new_requests(tmp_path,catalog_replay_library):
    item=definition('midspan','p_pinf');item['schedule']['interval']=2
    retry=deepcopy(item);retry['enable_request']=2
    probe=dict(id='healthy',kind='probe',enable_request=1,source='instantaneous',fields=['pressure'],
        sampling=dict(mode='grid_node',indices=[4,2,2]),output=dict(channel='data',format='csv'),
        schedule=dict(mode='steps',interval=2))
    config=configuration_2d()
    baseline=launch(tmp_path/'baseline',2,True,steps=8,insitu_config=config,product_catalog_records=[])
    case=launch(tmp_path/'refusal',2,True,steps=8,insitu_config=config,product_catalog_records=[item,probe],
        product_poll_steps=1,wall_capture_refusal=(2,1,'midspan'),
        catalog_replay=(catalog_replay_library,{6:[retry,probe]}),memcheck=True)
    final='outdat/new/checkpoints/step000000000008/state.h5';compare_fields(baseline/final,case/final)
    assert sorted(int(p.name[10:22]) for p in frames(case,'midspan'))==[0,5,7]
    monitors=list((case/'outdat/new/monitor_products/healthy').rglob('*.csv'))
    assert len(monitors)==1
    assert [int(row['step']) for row in rows(monitors[0])]==[0,2,4,6,8]
    state={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert state['midspan']['state']=='active' and state['midspan']['consumed_enable_request']==2
    assert 'optional SBLI wall source allocation' in (case/'run.log').read_text()


def test_midspan_does_not_misrepresent_a_blowing_wall_and_other_products_continue(tmp_path):
    probe=dict(id='pressure-probe',kind='probe',enable_request=1,source='instantaneous',fields=['pressure'],
        sampling=dict(mode='grid_node',indices=[4,2,2]),output=dict(channel='data',format='csv'),
        schedule=dict(mode='steps',interval=1))
    config=configuration_2d()
    baseline=launch(tmp_path/'baseline',2,True,steps=2,insitu_config=config,wall_blowing=True,product_catalog_records=[])
    case=launch(tmp_path/'observed',2,True,steps=2,insitu_config=config,wall_blowing=True,
        product_catalog_records=[definition('midspan','skin_friction'),probe])
    final='outdat/new/checkpoints/step000000000002/state.h5';compare_fields(baseline/final,case/final)
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['midspan']['state']=='invalid-configuration'
    assert 'stationary impermeable wall' in json.dumps(status['midspan'])
    assert status['pressure-probe']['state']=='active'
    assert list((case/'outdat/new/monitor_products/pressure-probe').rglob('*.csv'))
    assert not frames(case,'midspan')
