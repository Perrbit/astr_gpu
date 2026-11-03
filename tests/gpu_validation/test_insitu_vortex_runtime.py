"""TB3 real complete-step gradients and instantaneous vortex diagnostics."""
import json
import os
import re

import h5py
import numpy as np
import pytest

from test_index_slice_runtime import product,segments,frames
from test_insitu_sbli_statistics import launch,TOPOLOGIES
from test_insitu_selected_runtime import configuration
from test_insitu_vortex_math import evaluate
from test_runtime_catalog_images import q_product,image_segments,arguments as image_arguments
from test_insitu_device_products import resident_configuration
from run_output_restart_validation import run_case,compare_fields
from test_index_slice_runtime import ROOT

EXE=os.environ.get('ASTR_SELECTED_RUNTIME_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root FP64 CPU/GPU executable')
GRADIENTS=['velocity_gradient_'+suffix for suffix in ('xx','yx','zx','xy','yy','zy','xz','yz','zz')]
FIELDS=[*GRADIENTS,'lambda2','lambda_ci']


def unrestricted_config(config):
    config=re.sub(r'(host_budget_bytes|device_budget_bytes)\s*=\s*\d+',r'\1=-1',config)
    return re.sub(r'device_reserve_bytes\s*=\s*\d+','device_reserve_bytes=0',config)


def definitions():
    row=product('vortex-diagnostics',FIELDS)
    row['sampling']=dict(mode='grid_index',axis='k',index=8,index_ranges=dict(i=[16,48],j=[0,48]))
    row['schedule']['interval']=1
    return [row]


@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_real_vortex_same_state(tmp_path,topology,record_property):
    # The existing catalog_stats session requires statistics=t; instantaneous
    # requests themselves create no selected-time accumulator groups.
    config=unrestricted_config(configuration())
    cases={backend:launch(tmp_path/backend,backend,topology,steps=2,records=definitions(),
                          insitu_config=config,unrestricted=True) for backend in ('cpu','gpu')}
    datasets={}
    same_state_error=0.
    for backend,case in cases.items():
        state=json.loads((case/'outdat/new/products.status.json').read_text())
        assert state['products'][0]['state']=='active',state
        available=segments(case)
        assert len(available)==1,(case/'run.log').read_text()
        paths=frames(available[0]);assert len(paths)==3
        datasets[backend]=[]
        for path in paths:
            with h5py.File(path) as data:
                packet=np.stack([data[field][...] for field in FIELDS],axis=-1)
            assert np.isfinite(packet).all()
            a=packet[...,:9].reshape(*packet.shape[:-1],3,3).swapaxes(-1,-2)
            square=a@a
            reference=np.stack([np.linalg.eigvalsh(.5*(square+square.swapaxes(-1,-2)))[...,1],
                                np.abs(np.linalg.eigvals(a).imag).max(axis=-1)],axis=-1)
            error=np.abs(packet[...,9:]-reference)/np.maximum(1.,np.maximum(np.abs(packet[...,9:]),np.abs(reference)))
            assert error.max()<=2e-10,(backend,path,error.max(),np.unravel_index(error.argmax(),error.shape))
            # Both mathematical backends see the exact same FP64 gradient,
            # rather than distinct, roundoff-perturbed CFD trajectories.
            flat=a.reshape(-1,3,3)
            for oracle_backend in ('cpu','gpu'):
                oracle,status,_=evaluate(flat,oracle_backend)
                assert not np.any(status)
                oracle=oracle.reshape(reference.shape)
                same=np.abs(packet[...,9:]-oracle)/np.maximum(1.,np.maximum(np.abs(packet[...,9:]),np.abs(oracle)))
                same_state_error=max(same_state_error,float(same.max()))
                assert same.max()<=2e-10,(backend,oracle_backend,path,same.max())
            datasets[backend].append(packet)
    cpu=np.array(datasets['cpu']);gpu=np.array(datasets['gpu'])
    difference=np.abs(cpu-gpu)/np.maximum(1.,np.maximum(np.abs(cpu),np.abs(gpu)))
    record_property('maximum_scaled_cpu_gpu_difference',float(difference.max()))
    record_property('maximum_scaled_same_state_difference',same_state_error)
    (tmp_path/'comparison.json').write_text(json.dumps(dict(
        per_field_trajectory_scaled_difference=dict(zip(FIELDS,difference.max(axis=(0,1,2)).tolist())),
        maximum_scaled_same_state_difference=same_state_error),indent=2)+'\n')
    assert difference[...,:9].max()<=2e-10


def surface_definitions(sbli=False):
    rows=[]
    for kind,threshold in [('lambda2_surface',-1e-7 if sbli else -.1),
                           ('lambda_ci_surface',.01 if sbli else .4)]:
        row=q_product(name=kind,threshold=threshold)
        row['kind']=kind
        row['schedule']['interval']=1
        row['sampling']['mode']='volume_index'
        row['sampling']['index_ranges']=dict(i=[16,48],j=[0,48],k=[4,12]) if sbli else dict(i=[4,28],j=[4,28],k=[4,28])
        if sbli:
            row['sampling']['bounds']=[-1,10,0,1,0,.25]
            row['display'].update(camera=[8,-5,4],focal_point=[4,.25,.125],parallel_scale=4,color_range=[0,1.1])
        rows.append(row)
    return rows


def render_configuration(pipeline):
    config=resident_configuration(pipeline,profile='catalog_stats')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    return unrestricted_config(config)


def check_images(case,steps,ranks):
    from PIL import Image
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in state['products']),state
    available=image_segments(case);assert len(available)==2
    for segment in available:
        paths=list(segment.glob('*.jpeg'));assert len(paths)==steps+1
        for path in paths:
            with Image.open(path) as image:
                assert image.size==(800,600) and np.ptp(np.asarray(image.convert('RGB')))>100
        receipts=[json.loads(path.read_text()) for path in segment.glob('frame.*.rank*.json')]
        assert len(receipts)==(steps+1)*ranks
        assert any(not row['geometry_empty'] for row in receipts)
        assert all(row['field_d2h_bytes']==row['geometry_d2h_bytes']==row['geometry_host_bytes']==0 for row in receipts)
    return available


@pytest.mark.parametrize('topology,pipeline',[(t,'standard-device') for t in TOPOLOGIES]+[((2,1,1),'direct-device')])
def test_sbli_resident_vortex_surfaces(tmp_path,topology,pipeline):
    config=render_configuration(pipeline)
    case=launch(tmp_path/'gpu','gpu',topology,steps=2,records=surface_definitions(True),
        insitu_config=config,unrestricted=True)
    check_images(case,2,int(np.prod(topology)))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_tgv_resident_vortex_surfaces(tmp_path,pipeline):
    args=image_arguments(tmp_path);args.executable=EXE;args.unrestricted=True;args.insitu_write_eps='0'
    config=render_configuration(pipeline)
    case,_=run_case(args,ROOT,'gpu',2,'vortex',2,grid='32,32,32',insitu_config=config,
        product_catalog_records=surface_definitions(),postprocess_transport='device-aware')
    check_images(case,2,2)


def test_vortex_surfaces_exact_restart_and_output_switch(tmp_path):
    config=render_configuration('standard-device');rows=surface_definitions(True)
    full=launch(tmp_path/'full','gpu',(2,1,1),records=rows,insitu_config=config,checkpoint=True,unrestricted=True)
    prefix=launch(tmp_path/'prefix','gpu',(2,1,1),steps=5,records=rows,insitu_config=config,checkpoint=True,unrestricted=True)
    resumed=launch(tmp_path/'tail','gpu',(2,1,1),restore=prefix/'outdat/new/checkpoints/step000000000005',
        records=rows,insitu_config=config,checkpoint=True,unrestricted=True)
    off=launch(tmp_path/'off','gpu',(2,1,1),records=[],insitu_config=config.replace('render=.true.','render=.false.'),
        checkpoint=True,unrestricted=True)
    final='outdat/new/checkpoints/step000000000012'
    compare_fields(full/final/'state.h5',off/final/'state.h5')
    compare_fields(full/final/'state.h5',resumed/final/'state.h5')
    reference={p.parent.parent.name:p for p in image_segments(full)}
    for segment in image_segments(resumed):
        for path in segment.glob('*.jpeg'):
            assert path.read_bytes()==(reference[segment.parent.parent.name]/path.name).read_bytes()


def test_vortex_device_memcheck(tmp_path):
    case=launch(tmp_path/'gpu','gpu',(2,1,2),steps=2,records=surface_definitions(True),
        insitu_config=render_configuration('standard-device'),unrestricted=True,memcheck=True)
    logs=list(case.glob('memcheck.*.log'));assert len(logs)==4
    assert all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_vortex_roi_preserves_external_stencil(tmp_path,backend):
    from copy import deepcopy
    roi=definitions()[0];full=deepcopy(roi);full['id']='full-plane'
    full['sampling']['index_ranges']['i']=[0,64]
    case=launch(tmp_path/backend,backend,(2,1,2),steps=2,records=[full,roi],
        insitu_config=unrestricted_config(configuration()),unrestricted=True)
    selected={p.parent.parent.name:p for p in segments(case)}
    for a,b in zip(frames(selected['full-plane']),frames(selected['vortex-diagnostics'])):
        with h5py.File(a) as global_field,h5py.File(b) as roi_field:
            for field in FIELDS:
                np.testing.assert_array_equal(global_field[field][:,16:49],roi_field[field][...],err_msg=field)


def test_vortex_surface_cost_without_field_oracles(tmp_path):
    common=dict(steps=2,insitu_config=render_configuration('standard-device'),unrestricted=True,monitor=True)
    off=launch(tmp_path/'off','gpu',(2,1,1),records=[],**common)
    baseline=json.loads((off/'resources.sampled.json').read_text())
    on=launch(tmp_path/'on','gpu',(2,1,1),records=surface_definitions(True),resource_baseline=baseline,**common)
    result=dict(scope='NP2 x, two GRCBC steps including both vortex extractions, rendering and JPEG; excludes initial frame; not a stable benchmark',cases={})
    for name,case in [('off',off),('on',on)]:
        assert not list((case/'outdat').rglob('*.h5'))
        times={}
        text=(case/'run.log').read_text()
        for line in text.splitlines():
            match=re.fullmatch(r'ASTR_INSITU_STAGE_TIMING completed_step_inclusive (\d+) (\d+)\s+(\S+)',line)
            if match:times[tuple(map(int,match.group(1,2)))]=float(match[3])
        assert set(times)=={(s,r) for s in (1,2) for r in (0,1)}
        result['cases'][name]=dict(step_max_rank_seconds=[max(times[s,r] for r in (0,1)) for s in (1,2)],
            two_steps_max_rank_sum_seconds=max(sum(times[s,r] for s in (1,2)) for r in (0,1)),
            resources=json.loads((case/'resources.sampled.json').read_text()))
        if name=='on':
            packets=re.findall(r'ASTR_VORTEX_DEVICE nodes=(\d+) scratch_bytes=(\d+) result_components=(\d+)',text)
            assert len(packets)==6 and all(int(c)==2 and int(b)==int(n)*72 for n,b,c in packets),packets
            result['cases'][name]['scratch_packets']=[dict(nodes=int(n),bytes=int(b),components=int(c)) for n,b,c in packets]
    check_images(on,2,2)
    (tmp_path/'cost.json').write_text(json.dumps(result,indent=2)+'\n')
