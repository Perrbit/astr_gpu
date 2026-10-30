"""Bounded X2 control/extraction cost accounting, not a production speed claim."""
import csv
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import pytest

from run_output_restart_validation import run_case,compare_fields
from test_runtime_catalog_images import arguments,dynamic_image,catalog_replay_library
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import ROOT

EXE=os.environ.get('ASTR_RUNTIME_COST_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Explicitly select the bounded cost campaign executable')


def costs(case):
    log=(case/'run.log').read_text()
    stages={}
    for name,step,rank,seconds in re.findall(r'ASTR_INSITU_STAGE_TIMING (\w+) (-?\d+) (\d+)\s+(\S+)',log):
        entry=stages.setdefault(name,{})
        entry.setdefault(rank,[]).append(float(seconds))
    result=dict(launch=json.loads((case/'timing.launch.json').read_text()),stages={
        name:dict(max_rank_total_seconds=max(map(sum,ranks.values())),
            per_rank_seconds={rank:sum(values) for rank,values in ranks.items()},
            per_rank_events={rank:len(values) for rank,values in ranks.items()})
        for name,ranks in stages.items()})
    preparation=[]
    for rank,name,step,local,agreement,consumer,status in re.findall(
            r'ASTR_RUNTIME_STAGE_TIMING rank=(\d+) id=(\S+) step=(\d+) native_local_seconds=(\S+) '
            r'consensus_seconds=(\S+) consumer_inclusive_seconds=(\S+) status=(\d+)',log):
        preparation.append(dict(rank=int(rank),id=name,step=int(step),local_seconds=float(local),
            agreement_seconds=float(agreement),consumer_inclusive_seconds=float(consumer),status=int(status)))
    result['products']=preparation
    result['host_products']=[dict(rank=int(rank),id=name,step=int(step),prepare_inclusive_seconds=float(prepare),
        consumer_inclusive_seconds=float(consumer),status=int(status))
        for rank,name,step,prepare,consumer,status in re.findall(
            r'ASTR_RUNTIME_HOST_TIMING rank=(\d+) id=(\S+) step=(\d+) prepare_inclusive_seconds=(\S+) '
            r'consumer_inclusive_seconds=(\S+) status=(\d+)',log)]
    receipts=[json.loads(path.read_text()) for path in (case/'outdat/new/render_products').glob(
        '*/*/segment*/frame.*.rank*.json')]
    result['receipts']=dict(count=len(receipts),
        geometry_download_bytes=sum(row.get('geometry_d2h_bytes',0) for row in receipts),
        field_download_bytes=sum(row.get('field_d2h_bytes',0) for row in receipts))
    result['private_transfers']=[dict(rank=int(rank),step=int(step),transport=transport,
        download_bytes=int(download),upload_bytes=int(upload))
        for rank,step,transport,download,upload in re.findall(
            r'ASTR_RUNTIME_TRANSFER rank=(\d+) step=(\d+) transport=(\S+) private_d2h_bytes=(\d+) private_h2d_bytes=(\d+)',log)]
    result['resources']=[]
    for path in sorted((case/'outdat').rglob('resources.rank*.csv')):
        with path.open() as stream:
            identity=stream.readline().strip();rows=list(csv.DictReader(stream))
        result['resources'].append(dict(file=str(path.relative_to(case)),identity=identity,
            host_increment_peak_bytes=max(int(row['host_increment_bytes']) for row in rows),
            device_increment_peak_bytes=max(int(row['device_increment_bytes']) for row in rows),
            device_free_min_bytes=min(int(row['device_free_bytes']) for row in rows)))
    return result


def test_control_and_product_costs_keep_authoritative_state(tmp_path,catalog_replay_library):
    args=arguments(tmp_path);args.executable=Path(EXE).resolve()
    options=resident_configuration('standard-device',statistics=False,interval=1,profile='velocity_slice')
    quiet=dynamic_image('slice','quiet');quiet['schedule']['interval']=1000
    changed=deepcopy(quiet);changed['display']['camera'][0]+=1
    pictures=[dynamic_image(kind,kind) for kind in ('slice','physical_plane','q_surface','streamlines')]
    for item in pictures: item['schedule']['interval']=4
    cases={}
    host=options.replace("products='velocity_slice'","products='all'").replace(
        "processing_backend='device'","processing_backend='host'").replace(
        "derivative_backend='gpu'","derivative_backend='cpu'").replace("postprocess_transport='pinned',",'').replace(
        "rendering_pipeline='standard-device'","rendering_pipeline='compatible'")
    for name,config,items,replay in (
        ('no-insitu',None,[],None),('empty',options,[],None),
        ('unchanged',options,[quiet],None),
        ('changed',options,[quiet],(catalog_replay_library,{6:[changed]})),
        ('products',options,pictures,None),('host-products',host,pictures,None)):
        case,_=run_case(args,ROOT,'gpu',2,name,12,grid='32,32,32',insitu_config=config,
            product_catalog_records=items,product_poll_steps=1,catalog_replay=replay,
            insitu_timing=True,insitu_memory_trace=True)
        cases[name]=case
    final='outdat/new/checkpoints/step000000000012/state.h5'
    for case in cases.values(): compare_fields(cases['no-insitu']/final,case/final)
    report=dict(scope='32^3 FP64 TGV, NP=2 x, 12 steps; one run per condition with internal timing/resource counters, '
        'without an external sanitizer/profiler; '
        'same final diagnostic checkpoint included; no production speedup claim',
        measurements={name:costs(case) for name,case in cases.items()})
    assert not list((cases['empty']/'outdat').rglob('*.jpeg'))
    assert report['measurements']['products']['receipts']['geometry_download_bytes']==0
    assert report['measurements']['products']['receipts']['field_download_bytes']==0
    assert len(list((cases['products']/'outdat').rglob('*.jpeg')))==16
    assert len(list((cases['host-products']/'outdat').rglob('*.jpeg')))==16
    assert report['measurements']['host-products']['receipts']['geometry_download_bytes']==0
    assert report['measurements']['host-products']['receipts']['field_download_bytes']>0
    (tmp_path/'costs.json').write_text(json.dumps(report,indent=2)+'\n')
