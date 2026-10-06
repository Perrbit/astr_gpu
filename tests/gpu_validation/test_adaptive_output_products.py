"""AP3 actual products; native decision records, independent geometry/field readback."""
import csv
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from test_insitu_product_dispatch import params,config,receipts
from test_insitu_device_products import check_geometry_fields,field_difference
from test_adaptive_output_runtime import EVENT_CONFIG,GROUPS
from run_output_restart_validation import run_case
from test_output_insitu_restart import ROOT

pytestmark=pytest.mark.skipif(not os.environ.get('ASTR_OUTPUT_INSITU_EXE') or not os.environ.get('ASTR_OUTPUT_INSITU_BACKEND'),
                           reason='Select resident executable and Catalyst explicitly')
RENDER_CONFIG=EVENT_CONFIG.replace('r_off=0.723,hold_time=0.005','r_off=0.73,hold_time=0.0045')
NATIVE_GROUPS=GROUPS.replace('dense_interval_steps=1','dense_interval_steps=2')


def settings(pipeline,transport):
    c=config(pipeline,transport,statistics=False)
    c=c.replace('product_times=0,0.008', '''product_times=0,0.008,
 product_adaptive(1:2)=t,t,product_dense_steps=2,0,product_dense_times=0,0.002,
 product_events(1,1)='decay',product_events(1,2)='decay' ''')
    if pipeline=='compatible':
        c=c.replace("product_steps(3)=6", "product_steps(3)=6,product_adaptive(3)=t,product_dense_steps(3)=2,product_windows(1,3)='burst'")
    return c


def expected(pipeline):
    result={1:['q_surface','instantaneous_streamlines'],3:['q_surface','instantaneous_streamlines'],
            5:['q_surface','instantaneous_streamlines'],11:['q_surface']}
    if pipeline=='compatible':
        for step in (3,5,11):
            result[step].append('velocity_slice')
    return result


def check_adaptive_receipts(case,ranks,pipeline):
    output=receipts(case,ranks,expected(pipeline))
    for rank in range(ranks):
        for step in expected(pipeline):
            record=json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            states=record['adaptive_outputs']
            q=states['q_surface.image']
            assert q['success_step']==q['attempt_step']==step
            assert q['dense']==(step<6)
            assert q['next_step']==step+(2 if step<6 else 6)
            assert q['reason']==(2 if step==1 else 1)
            with (output/f'resources.rank{rank:08d}.csv').open() as stream:
                next(stream)
                resource=list(csv.DictReader(stream))
            assert resource and max(int(row['host_increment_bytes']) for row in resource)<=4*1024**3
            assert max(int(row['device_increment_bytes']) for row in resource)<=2*1024**3
            assert min(int(row['device_free_bytes']) for row in resource)>=1024**3
    return output


@pytest.mark.parametrize('pipeline',['compatible','standard-device','direct-device'])
@pytest.mark.parametrize('transport',['pinned','device-aware'])
@pytest.mark.parametrize('ranks',[1,2])
def test_actual_adaptive_products_and_native_archives(tmp_path,pipeline,transport,ranks,record_property):
    args=params(tmp_path); args.statistics=False
    case,size=run_case(args,ROOT,'gpu',ranks,'adaptive_products',12,grid='32,32,32',
        insitu_config=settings(pipeline,transport),postprocess_transport=transport,adaptive_config=RENDER_CONFIG,
        archive_groups=NATIVE_GROUPS,tgv_reynolds=1,checkpoint_interval=11,checkpoint_keep=2,
        buffer_bytes=1048576,insitu_timing=True)
    output=check_adaptive_receipts(case,ranks,pipeline)
    assert len(list(output.glob('*.jpeg')))==7
    assert len(list(output.glob('*.pvtp')))==(3 if pipeline=='compatible' else 0)
    if pipeline=='compatible':
        record_property('geometry_absolute_error',check_geometry_fields(case,11,statistics=False,products=['velocity_slice']))
    else:
        assert all(json.loads(p.read_text())['geometry_host_bytes']==0 for p in output.glob('mesh_step*json'))
    for label,identities in [('fields',[3,5,10]),('slices',[3,5,11])]:
        frames=sorted((case/'outdat/new'/label).glob('segment*/step*'))
        assert [int(p.name[4:]) for p in frames]==identities
        for frame in frames:
            with h5py.File(frame/'data.h5') as f:
                datasets=[]; f.visititems(lambda n,o:datasets.append(n) if isinstance(o,h5py.Dataset) else None)
                assert datasets and all(np.isfinite(f[n][...]).all() for n in datasets)
    frame=next((case/'outdat/new/slices').glob('segment*/step000000000011/data.h5'))
    with h5py.File(frame) as saved,h5py.File(case/'outdat/new/checkpoints/step000000000011/state.h5') as state:
        errors={}
        for offset,name in enumerate(('density','velocity_x','velocity_y','velocity_z','pressure','temperature'),6):
            errors[name]=float(np.max(np.abs(saved['k000000000004/'+name][:]-state[f'q{offset:04d}'][4,:,:])))
            assert errors[name]<=2e-10,errors
        record_property('slice_absolute_errors',errors)
    record_property('directory_bytes',size)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_adaptive_missing_wait_exact_restart(tmp_path,pipeline,record_property):
    args=params(tmp_path); args.statistics=False
    c=settings(pipeline,'pinned').replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),
                                        str(ROOT/'tests/gpu_validation/insitu_resident_failure_pipeline.py'))
    opts=dict(grid='32,32,32',insitu_config=c,postprocess_transport='pinned',adaptive_config=RENDER_CONFIG,
              tgv_reynolds=1,checkpoint_interval=100,checkpoint_keep=1,buffer_bytes=1048576)
    full,a=run_case(args,ROOT,'gpu',2,'continuous_fault_ap_image',12,**opts)
    part,b=run_case(args,ROOT,'gpu',2,'partial_fault_ap_image',5,**opts)
    tail,c=run_case(args,ROOT,'gpu',2,'resumed_fault_ap_image',12,
                    restore=part/'outdat/new/checkpoints/step000000000005',**opts)
    final='outdat/new/checkpoints/step000000000012'
    for name in ('control.bin','insitu_control.bin'):
        assert (full/final/name).read_bytes()==(tail/final/name).read_bytes()
    field_difference(full/final/'state.h5',tail/final/'state.h5')
    for root in (full,part):
        missing=list((root/'outdat/render').glob('missing.q_surface.step00000005.rank*.json'))
        assert len(missing)==2
        assert not (root/'outdat/render/q_surface.step00000005.jpeg').exists()
    assert sorted(int(p.name.split('step')[1][:8]) for p in (tail/'outdat/render').glob('q_surface*.jpeg'))==[11]
    assert sorted(int(p.name.split('step')[1][:8]) for p in (tail/'outdat/render').glob('instantaneous_streamlines*.jpeg'))==[]
    record_property('directory_bytes',[a,b,c])


@pytest.mark.parametrize('change',['threshold','period','transport'])
def test_adaptive_render_override_preserves_unaffected_time_clock(tmp_path,change):
    args=params(tmp_path); args.statistics=True
    old=config('standard-device','pinned',statistics=True).replace('product_times=0,0.008',
        "product_times=0,0.008,product_adaptive(1)=t,product_dense_steps(1)=1,product_events(1,1)='decay'")
    opts=dict(grid='32,32,32',tgv_reynolds=1,buffer_bytes=1048576,checkpoint_interval=100,checkpoint_keep=1)
    full,_=run_case(args,ROOT,'gpu',2,'full',12,insitu_config=old,postprocess_transport='pinned',
                    adaptive_config=RENDER_CONFIG,**opts)
    part,_=run_case(args,ROOT,'gpu',2,'part',5,insitu_config=old,postprocess_transport='pinned',
                    adaptive_config=RENDER_CONFIG,**opts)
    source=part/'outdat/new/checkpoints/step000000000005'
    new,cfg,transport=old,RENDER_CONFIG,'pinned'
    if change=='threshold': cfg=cfg.replace('r_on=0.74','r_on=1')
    if change=='period': new=new.replace('product_steps=6,0','product_steps=4,0')
    if change=='transport':
        transport='device-aware'; new=new.replace("postprocess_transport='pinned'","postprocess_transport='device-aware'")
    reject='adaptive monitor/control metadata tail' if change=='threshold' else 'native render configuration differs; select explicit override'
    run_case(args,ROOT,'gpu',2,'reject',12,restore=source,insitu_config=new,adaptive_config=cfg,
             postprocess_transport=transport,reject=reject,**opts)
    tail,_=run_case(args,ROOT,'gpu',2,'tail',12,restore=source,insitu_config=new,adaptive_config=cfg,
                    postprocess_transport=transport,override=True,**opts)
    receipts(tail,2,{8:['instantaneous_streamlines'],9 if change=='period' else 11:['q_surface']})
    final='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): field_difference(full/final/name,tail/final/name)
    if change=='transport':
        assert (full/final/'control.bin').read_bytes()==(tail/final/'control.bin').read_bytes()
        assert (full/final/'insitu_control.bin').read_bytes()[192:]==(tail/final/'insitu_control.bin').read_bytes()[192:]
