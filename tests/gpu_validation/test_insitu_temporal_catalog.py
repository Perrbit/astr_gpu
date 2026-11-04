"""TB4 explicit sampling cadence, line support and temporal checkpoint definitions."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_insitu_catalog import HEADER,PROBE,run

pytestmark=pytest.mark.skipif(not PROBE,reason='Select the root catalog probe')


def definition(kind='time_space',image=False):
    row=dict(id=kind,kind=kind,source='instantaneous',enable_request=1,
        fields=['velocity_x','pressure'],sampling=dict(mode='grid_index',axis='j',index=2,
            index_ranges=dict(i=[0,64],k=[4,4])),
        temporal=dict(block_samples=16),output=dict(channel='data',format='csv'),
        schedule=dict(mode='steps',interval=10))
    if kind=='probe_spectrum':
        row['sampling']=dict(mode='grid_node',indices=[24,2,4])
        row['temporal']=dict(segment_length=30)
    if image:
        row.pop('fields');row['field']='pressure'
        row['output']=dict(channel='image',formats=['jpeg'],image_size=[800,600],jpeg_quality=95,dpi=300)
        row['display']=dict(color_range=[0,1],color_map='diverging_blue_red') if kind=='time_space' else dict(y_range=[0,1],line_width=2)
    return row


@pytest.mark.parametrize('name',['input.products.tgv.temporal.example.jsonl','input.products.tgv.temporal.images.example.jsonl',
    'input.products.tgv.temporal.mean.example.jsonl','input.products.flatplate.temporal.example.jsonl'])
def test_shipped_temporal_definitions(tmp_path,name):
    path=Path(__file__).resolve().parents[2]/'scripts/output'/name
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0 and result.stdout.count('valid=T')==2 and 'valid=F' not in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize('kind',['time_space','probe_spectrum'])
@pytest.mark.parametrize('image',[False,True])
def test_temporal_definition_and_control_roundtrip(tmp_path,kind,image):
    row=definition(kind,image)
    result=run(tmp_path,[HEADER,row])
    assert result.returncode==0 and 'valid=T' in result.stdout,result.stdout+result.stderr
    expected='temporal=1 0 16 F' if kind=='time_space' else 'temporal=1 30 0 F'
    assert expected in result.stdout,result.stdout
    probe=os.environ.get('ASTR_CATALOG_STATE_PROBE')
    assert probe,'Select the explicit control roundtrip probe'
    path=tmp_path/'roundtrip.jsonl';path.write_text('\n'.join(json.dumps(r) for r in [HEADER,row])+'\n')
    restored=subprocess.run([probe,str(path),'visual-checkpoint'],capture_output=True,text=True)
    assert restored.returncode==0,restored.stdout+restored.stderr


def test_probe_raw_history_is_explicit_and_sampling_is_not_image_cadence(tmp_path):
    row=definition('probe_spectrum')
    row['temporal'].update(sample_steps=3,archive_samples=True,block_samples=17)
    result=run(tmp_path,[HEADER,row])
    assert 'valid=T' in result.stdout and 'temporal=3 30 17 T' in result.stdout,result.stdout
    assert 'steps=10' in result.stdout


@pytest.mark.parametrize('kind',['time_space','probe_spectrum'])
def test_temporal_collection_checkpoint_and_display_independence(tmp_path,kind):
    row=definition(kind)
    if kind=='time_space': row['sampling']['index_ranges']['i']=[0,8]
    path=tmp_path/'history.jsonl';path.write_text('\n'.join(json.dumps(r) for r in [HEADER,row])+'\n')
    result=subprocess.run([os.environ['ASTR_CATALOG_STATE_PROBE'],str(path),'temporal-history'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS: runtime catalog temporal-history' in result.stdout


@pytest.mark.parametrize('ranks',[1,2,4])
def test_temporal_mpi_configuration_restore_and_unchanged_republication(tmp_path,ranks):
    probe=os.environ.get('ASTR_CATALOG_MPI_PROBE');launcher=os.environ.get('ASTR_CATALOG_MPIEXEC')
    if not probe or not launcher: pytest.skip('Select root catalog MPI probe and launcher')
    formal=tmp_path/'products.jsonl';candidate=tmp_path/'candidate.jsonl';repeat=tmp_path/'repeat.jsonl'
    row=definition('probe_spectrum');row['temporal']=dict(sample_steps=3,segment_length=4,archive_samples=True,block_samples=4)
    formal.write_text('\n'.join(json.dumps(r) for r in [HEADER,row])+'\n')
    row['temporal']=dict(sample_steps=5,segment_length=6)
    for path in (candidate,repeat):path.write_text('\n'.join(json.dumps(r) for r in [HEADER,row])+'\n')
    result=subprocess.run([launcher,'--bind-to','none','--oversubscribe','-np',str(ranks),probe,
        str(formal),str(candidate),'temporal',str(repeat)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    for rank in range(ranks):assert f'RANK {rank} TEMPORAL_OK' in result.stdout


@pytest.mark.parametrize('change',[
    dict(temporal={}),dict(temporal=dict(segment_length=31)),
    dict(temporal=dict(segment_length=0)),dict(temporal=dict(segment_length=32,sample_steps=0)),
    dict(temporal=dict(segment_length=32,archive_samples=1)),
    dict(temporal=dict(segment_length=32,archive_samples=True)),
    dict(temporal=dict(segment_length=32,block_samples=16)),
    dict(temporal=dict(segment_length=32,unknown=1)),
])
def test_bad_spectrum_only_disables_its_record(tmp_path,change):
    bad=definition('probe_spectrum');bad.update(deepcopy(change))
    good=definition()
    result=run(tmp_path,[HEADER,bad,good])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T' in result.stdout,result.stdout


@pytest.mark.parametrize('ranges',[dict(i=[0,64],k=[0,16]),dict(i=[4,4],k=[4,4])])
def test_time_space_is_a_line_not_a_plane_or_a_point(tmp_path,ranges):
    row=definition();row['sampling']['index_ranges']=ranges
    result=run(tmp_path,[HEADER,row])
    assert 'valid=F' in result.stdout and 'exactly one varying' in result.stdout,result.stdout
