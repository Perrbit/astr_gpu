"""TB2 two-dimensional means of instantaneous complete wall transport, never span-reduced."""
import json
import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from test_index_slice_runtime import product, segments, frames
from test_insitu_sbli_statistics import launch as launch_sbli, DT, WINDOW, TOPOLOGIES
from test_insitu_station_profiles import endpoint_weights
from run_output_restart_validation import compare_fields

EXE = os.environ.get('ASTR_SELECTED_RUNTIME_EXE')
pytestmark = pytest.mark.skipif(not EXE,reason='Select a root-CMake FP64 executable')
FIELDS = ['density','temperature','wall_shear_x','wall_shear_z','wall_heat_into_wall','dynamic_viscosity']


def definitions(nx=64,nz=16):
    raw = product('wall-current',FIELDS)
    raw['sampling'] = dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,nx],k=[0,nz]))
    raw['schedule']['interval'] = 1
    mean = deepcopy(raw)
    mean.update(id='wall-time-mean',source='reynolds',statistics_source='selected-time')
    return [raw,mean]


def check(case,steps=12,dt=DT,window=WINDOW,nx=64,nz=16,canonical_periodic=False):
    available = {p.parent.parent.name:p for p in segments(case)}
    assert set(available)=={'wall-current','wall-time-mean'},(case/'run.log').read_text()
    samples = {}
    for path in frames(available['wall-current']):
        with h5py.File(path) as data:
            step = int(np.asarray(data['step']).item())
            samples[step] = {field:data[field][...] for field in FIELDS}
            if canonical_periodic:
                # The selected history owns unique periodic nodes; the image endpoint is node zero.
                for field in FIELDS:
                    samples[step][field][-1] = samples[step][field][0]
    assert set(samples)==set(range(steps+1))
    mean_frames = frames(available['wall-time-mean'])
    assert len(mean_frames)==steps
    worst = 0.
    for path in mean_frames:
        with h5py.File(path) as actual:
            step = int(np.asarray(actual['step']).item())
            weights = endpoint_weights(step,dt,window)
            assert 'statistics_source=selected-time' in str(actual.attrs['source_measure'])
            for field in FIELDS:
                observed = actual[field][...]
                assert observed.shape==(nz+1,nx+1) and np.isfinite(observed).all()
                reference = np.tensordot(weights,np.stack([samples[s][field] for s in range(step+1)]),axes=1)/weights.sum()
                error = np.abs(observed-reference)/np.maximum(1.,np.maximum(np.abs(observed),np.abs(reference)))
                worst = max(worst,float(error.max()))
                assert error.max()<=2e-10,(path,field,error.max())
    return worst


@pytest.fixture(scope='session')
def continuous(tmp_path_factory):
    cases = {}
    def get(backend):
        if backend not in cases:
            root = tmp_path_factory.mktemp('wall_footprint_np2z_'+backend)
            cases[backend] = launch_sbli(root/backend,backend,(1,1,2),records=definitions(),checkpoint=True)
        return cases[backend]
    return get


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_wall_footprint_nodewise_means(tmp_path,backend,topology,record_property,request):
    case = (request.getfixturevalue('continuous')(backend) if topology==(1,1,2) else
            launch_sbli(tmp_path/backend,backend,topology,records=definitions()))
    record_property('max_scaled_formula_difference',check(case))
    record_property('case_directory',str(case))
    record_property('backend',backend)
    record_property('topology',','.join(map(str,topology)))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_footprint_exact_restart_and_switch(tmp_path,backend,continuous):
    full = continuous(backend)
    prefix = launch_sbli(tmp_path/'prefix',backend,(1,1,2),steps=5,records=definitions(),checkpoint=True)
    resumed = launch_sbli(tmp_path/'tail',backend,(1,1,2),
                          restore=prefix/'outdat/new/checkpoints/step000000000005',records=definitions(),checkpoint=True)
    without = launch_sbli(tmp_path/'off',backend,(1,1,2),records=[],statistics=False,checkpoint=True)
    final = Path('outdat/new/checkpoints/step000000000012')
    compare_fields(full/final/'state.h5',without/final/'state.h5')
    for name in ('state.h5','statistics.h5'):
        compare_fields(full/final/name,resumed/final/name)
    reference = {p.parent.parent.name:p for p in segments(full)}
    for segment in segments(resumed):
        for path in frames(segment):
            with h5py.File(path) as actual,h5py.File(reference[segment.parent.parent.name]/path.name) as expected:
                for field in actual:
                    np.testing.assert_array_equal(actual[field],expected[field],err_msg=f'{path}/{field}')


@pytest.mark.skipif(not os.environ.get('ASTR_SELECTED_RUNTIME_M12_SOURCE'),reason='Select the existing read-only M12 resources')
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_footprint_preserves_random_blowing_spanwise_structure(tmp_path,backend,monkeypatch,record_property):
    from test_insitu_wall_transport_runtime import run_wall_fixture

    for name,value in dict(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='true',
        OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_coll_hcoll_enable='0',OMPI_MCA_osc='sm,pt2pt',
        UCX_MEMTYPE_CACHE='n',UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',
        UCX_CUDA_IPC_ENABLE_MNNVL='no',UCX_TLS='self,sm,cuda_copy,cuda_ipc').items():
        monkeypatch.setenv(name,value)
    case = tmp_path/backend
    run_wall_fixture(case,backend,(2,1,1),definitions(nz=24),statistics=True,checkpoint=False)
    record_property('max_scaled_formula_difference',check(case,steps=10,dt=.02,window=(.01,.19),nz=24,canonical_periodic=True))
    mean = next(p for p in segments(case) if p.parent.parent.name=='wall-time-mean')
    with h5py.File(frames(mean)[-1]) as data:
        values = data['wall_shear_x'][...]
        assert np.any(np.ptp(values[:-1],axis=0)>0),'a wall footprint must preserve its actual spanwise structure'
        np.testing.assert_array_equal(values[-1],values[0])


def test_wall_footprint_mean_gpu_memcheck(tmp_path):
    case=launch_sbli(tmp_path/'gpu','gpu',(1,1,2),steps=2,records=definitions(),memcheck=True)
    text=(case/'run.log').read_text()
    assert text.count('ASTR_GPU_PRE_MPI_DEVICE=')==2
    logs=list(case.glob('memcheck.*.log'))
    assert len(logs)==2
    assert all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device executable')
@pytest.mark.parametrize('topology,pipeline',[((1,1,1),'standard-device'),((2,1,1),'standard-device'),
    ((1,2,1),'standard-device'),((1,1,2),'standard-device'),((2,2,1),'standard-device'),
    ((2,1,2),'standard-device'),((1,2,2),'standard-device'),((2,1,1),'direct-device')])
def test_wall_footprint_device_images(tmp_path,topology,pipeline):
    from PIL import Image
    from test_insitu_device_products import resident_configuration
    from test_runtime_catalog_images import image_segments

    config = resident_configuration(pipeline,profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config = config.replace("output_directory='outdat/render'","output_directory='outdat'")
    records = wall_image_definitions()
    case = launch_sbli(tmp_path/'gpu','gpu',topology,steps=2,records=records,insitu_config=config,
                       executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'])
    status = json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    paths = image_segments(case)
    assert len(paths)==4
    ranks = int(np.prod(topology))
    for segment in paths:
        count = 3 if segment.parent.parent.name.startswith('instantaneous-') else 2
        assert len(list(segment.glob('*.jpeg')))==count
        for path in segment.glob('*.jpeg'):
            with Image.open(path) as image:
                assert image.size==(800,600) and np.any(np.asarray(image.convert('RGB'))<230)
        receipts = [json.loads(path.read_text()) for path in segment.glob('frame.*.rank*.json')]
        assert len(receipts)==count*ranks
        for receipt in receipts:
            assert receipt['field_d2h_bytes']==receipt['geometry_d2h_bytes']==receipt['geometry_host_bytes']==0
            assert receipt['coordinates']['display_scale']==[1.,1.,20.]
            if receipt['source']=='reynolds':
                assert receipt['statistics']['provider']=='selected-time'
                assert receipt['statistics']['quantity']=='local_accumulated_mean'


def wall_image_definitions(sources=('instantaneous','reynolds')):
    from test_insitu_catalog_products import visual_product

    records = []
    for source in sources:
        for field,limit in [('wall_shear_x',.0007),('wall_heat_into_wall',.0005)]:
            item = visual_product(name=source+'-'+field)
            item.update(source=source,field=field)
            if source=='reynolds':item['statistics_source']='selected-time'
            item['sampling'] = dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,64],k=[0,16]))
            item['schedule']['interval'] = 1
            item['display'].update(camera=[4.5,-20.,.125],focal_point=[4.5,0.,.125],view_up=[0.,0.,1.],
                parallel_scale=6.5,color_range=[-limit,limit],coordinates=dict(display_scale=[1.,1.,20.]))
            records.append(item)
    return records


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device executable')
def test_wall_footprint_image_cost_without_oracles(tmp_path):
    from test_insitu_device_products import resident_configuration
    import re

    config=resident_configuration('standard-device',profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    common=dict(steps=2,executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'],monitor=True,insitu_config=config)
    off=launch_sbli(tmp_path/'off','gpu',(2,1,1),records=[],**common)
    baseline=json.loads((off/'resources.sampled.json').read_text())
    on=launch_sbli(tmp_path/'on','gpu',(2,1,1),records=wall_image_definitions(('reynolds',)),resource_baseline=baseline,**common)
    result=dict(scope='one local 64x48x16 GPU NP2 x two-step window, two mean-wall images per step; not a stable performance benchmark',cases={})
    for name,case in [('off',off),('on',on)]:
        assert not list((case/'outdat').rglob('*.h5')),'input HDF5 remains in datin; no output reference or checkpoint HDF5'
        values={}
        for line in (case/'run.log').read_text().splitlines():
            match=re.fullmatch(r'ASTR_INSITU_STAGE_TIMING completed_step_inclusive (\d+) (\d+)\s+(\S+)',line)
            if match:values[tuple(map(int,match.group(1,2)))]=float(match[3])
        assert set(values)=={(s,r) for s in (1,2) for r in (0,1)}
        result['cases'][name]=dict(first_step_max_rank_seconds=max(values[1,r] for r in (0,1)),
            second_step_max_rank_seconds=max(values[2,r] for r in (0,1)),
            two_steps_max_rank_sum_seconds=max(sum(values[s,r] for s in (1,2)) for r in (0,1)),
            memory=json.loads((case/'resources.sampled.json').read_text()))
    assert all(row['state']=='active' for row in json.loads((on/'outdat/new/products.status.json').read_text())['products'])
    (tmp_path/'cost.json').write_text(json.dumps(result,indent=2)+'\n')
