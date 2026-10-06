"""IS8-R8 real resident failure and immutable checkpoint recovery gates."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from run_output_restart_validation import run_case
from test_output_insitu_restart import ROOT,arguments
from test_insitu_device_products import resident_configuration,field_difference
from test_insitu_is4_failures import fault_library


def config(pipeline):
    return resident_configuration(pipeline,interval=2,profile='q_surface').replace(
        str(ROOT/'scripts/insitu/device_render_pipeline.py'),
        str(ROOT/'tests/gpu_validation/insitu_resident_failure_pipeline.py'))


def params(path):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict resident candidate')
    args=arguments(path)
    args.directory_budget_bytes=256*1024**2
    args.runtime_timeout_seconds=90
    return args


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('phase,message',[
    ('initialize','ASTR_RESIDENT_INITIALIZATION_FAILURE_INJECTED'),
    ('draw','ASTR_RESIDENT_DRAW_FAILURE_INJECTED')])
def test_resident_failure_is_fatal(tmp_path,pipeline,phase,message):
    case,_=run_case(params(tmp_path),ROOT,'gpu',2,'fault_'+phase,4,grid='32,32,32',
        insitu_config=config(pipeline),checkpoint_interval=2,reject=message,failure_after_start=True)
    assert not list((case/'outdat/new/checkpoints').rglob('COMPLETE'))
    assert not list((case/'outdat/render').glob('*.jpeg'))
    assert not (case/'outdat/render/lifecycle_rank0.json').exists()


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_resident_image_publication_is_recorded(tmp_path,pipeline):
    args=params(tmp_path)
    healthy,_=run_case(args,ROOT,'gpu',2,'healthy',4,grid='32,32,32',
        insitu_config=config(pipeline),checkpoint_interval=2)
    failed,_=run_case(args,ROOT,'gpu',2,'fault_image',4,grid='32,32,32',
        insitu_config=config(pipeline),checkpoint_interval=2)
    journals=[]
    output=failed/'outdat/render'
    for rank in (0,1):
        receipt=json.loads((output/f'missing.q_surface.step00000002.rank{rank:08d}.json').read_text())
        assert receipt['status']=='missing' and receipt['phase']=='stage_eps' and receipt['errno']==5
        assert json.loads((output/f'mesh_step00000002_rank{rank}.json').read_text())['products']['q_surface']['image']==receipt
        journals.append(receipt)
        assert json.loads((output/f'lifecycle_rank{rank}.json').read_text())['finalized']
    assert journals[0]==journals[1]
    assert not list(output.glob('*step00000002.jpeg')) and not list(output.glob('*.partial'))
    for suffix in ('jpeg','eps'):
        name=f'q_surface.step00000004.{suffix}'
        assert (output/name).read_bytes()==(healthy/'outdat/render'/name).read_bytes()
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        field_difference(failed/final/name,healthy/final/name)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('phase',['before_statistics','before_render_control','before_complete'])
def test_resident_interrupted_save_recovers(tmp_path,pipeline,phase,fault_library):
    args=params(tmp_path)
    healthy,_=run_case(args,ROOT,'gpu',2,'healthy',6,grid='32,32,32',
        insitu_config=config(pipeline),checkpoint_interval=2)
    failed,_=run_case(args,ROOT,'gpu',2,'fault_save',6,grid='32,32,32',
        insitu_config=config(pipeline),checkpoint_interval=2,
        test_fault=(fault_library,phase,4),reject='ASTR_IS4_INTERRUPTION',failure_after_start=True)
    folder=failed/'outdat/new/checkpoints'
    source=folder/'step000000000002'
    candidate=folder/'step000000000004.tmp'
    assert (source/'COMPLETE').is_file() and candidate.is_dir()
    assert not (candidate/'COMPLETE').exists()
    assert (folder/'LATEST').read_text()==source.name+'\n'
    before={str(p.relative_to(failed/'outdat/new')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (failed/'outdat/new').rglob('*') if p.is_file()}
    run_case(args,ROOT,'gpu',2,'reject_partial',6,grid='32,32,32',restore=candidate,
        insitu_config=config(pipeline),checkpoint_interval=2,reject='invalid new checkpoint bundle')
    changed=config('direct-device' if pipeline=='standard-device' else 'standard-device')
    run_case(args,ROOT,'gpu',2,'reject_identity',6,grid='32,32,32',restore=source,
        insitu_config=changed,checkpoint_interval=2,
        reject='rendering pipeline restart mismatch; cross-pipeline restore is unsupported')
    resumed,_=run_case(args,ROOT,'gpu',2,'resumed',6,grid='32,32,32',restore=source,
        insitu_config=config(pipeline),checkpoint_interval=2)
    final='outdat/new/checkpoints/step000000000006'
    for name in ('state.h5','statistics.h5'):
        field_difference(resumed/final/name,healthy/final/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (resumed/final/name).read_bytes()==(healthy/final/name).read_bytes()
    for step in (4,6):
        for suffix in ('jpeg','eps'):
            name=f'q_surface.step{step:08d}.{suffix}'
            assert (resumed/'outdat/render'/name).read_bytes()==(healthy/'outdat/render'/name).read_bytes()
    after={str(p.relative_to(failed/'outdat/new')):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (failed/'outdat/new').rglob('*') if p.is_file()}
    assert before==after
