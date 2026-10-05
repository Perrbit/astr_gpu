"""Bounded image-only failures and collective native continuation."""
import errno
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
import image_publication as publication
from run_output_restart_validation import run_case,compare_fields
from test_output_insitu_restart import arguments,configuration
from test_insitu_products import compare_geometry

FINAL='outdat/new/checkpoints/step000000000004'


@pytest.mark.parametrize('code',(errno.EACCES,errno.ENOSPC,errno.EIO))
@pytest.mark.parametrize('phase',('stage_jpeg','stage_eps','publish_jpeg','publish_eps'))
def test_publication_cleans_partial_pair(tmp_path,monkeypatch,code,phase):
    picture=tmp_path/'frame.jpeg'
    if phase.startswith('stage_'):
        original=publication.write_payload

        def write(stream,payload):
            if '.'+phase[6:]+'.partial' in stream.name:
                stream.write(payload[:2])
                raise OSError(code,'injected')
            original(stream,payload)

        monkeypatch.setattr(publication,'write_payload',write)
    else:
        original=publication.os.link

        def link(source,destination):
            if destination.suffix=='.'+phase[8:]:
                raise OSError(code,'injected')
            original(source,destination)

        monkeypatch.setattr(publication.os,'link',link)
    result=publication.publish_pair(picture,b'jpeg',b'eps')
    assert result['errno']==code and result['phase']==phase
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('exception',(OSError(errno.EROFS,'not approved'),
                                    RuntimeError('not an I/O error')))
def test_unknown_failure_is_fatal(tmp_path,monkeypatch,exception):
    def write(stream,payload):
        raise exception

    monkeypatch.setattr(publication,'write_payload',write)
    with pytest.raises(type(exception)):
        publication.publish_pair(tmp_path/'frame.jpeg',b'jpeg',b'eps')
    assert not list(tmp_path.iterdir())


def test_pair_and_no_overwrite(tmp_path):
    picture=tmp_path/'frame.jpeg'
    assert publication.publish_pair(picture,b'jpeg',b'eps') is None
    assert picture.read_bytes()==b'jpeg' and picture.with_suffix('.eps').read_bytes()==b'eps'
    with pytest.raises(FileExistsError):
        publication.publish_pair(picture,b'other',b'other')
    assert picture.read_bytes()==b'jpeg'


def config():
    return configuration(interval=2).replace('statistics_window=0.0005,0.0115',
        'statistics_window=0.0005,0.0035').replace('render=.true.,',
        "render=.true., derivative_backend='gpu',products='q_surface',").replace(
        str(ROOT/'scripts/insitu/tgv_pipeline.py'),
        str(ROOT/'tests/gpu_validation/insitu_image_fault_pipeline.py'))


def params(path):
    args=arguments(path); args.statistics=False
    args.directory_budget_bytes=256*1024**2
    return args


def resource_and_frames(case,ranks,steps):
    import csv
    output=case/'outdat/render'
    for rank in range(ranks):
        lifecycle=json.loads((output/f'lifecycle_rank{rank}.json').read_text())
        assert lifecycle=={'frames':[[step,step*.001] for step in steps],'finalized':True}
        with (output/f'resources.rank{rank:08d}.csv').open() as stream:
            stream.readline()
            for row in csv.DictReader(stream):
                assert int(row['host_increment_bytes'])<=4*1024**3
                assert int(row['device_increment_bytes'])<=2*1024**3
                assert int(row['device_free_bytes'])>=1024**3
    assert not list(output.rglob('*.partial'))


@pytest.fixture(scope='module',params=((2,1,1),(2,2,1)),ids=('np2','np4'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology))
    args=params(tmp_path_factory.mktemp('insitu_image_reference'))
    case,_=run_case(args,ROOT,'gpu',ranks,'healthy',4,grid='32,32,32',topology=topology,
                    checkpoint_interval=1,insitu_config=config())
    resource_and_frames(case,ranks,[2,4])
    return case,topology,ranks


@pytest.mark.parametrize('fault,code',(('eacces',errno.EACCES),('enospc',errno.ENOSPC),('eio',errno.EIO)))
def test_collective_missing_image_and_exact_restart(reference,tmp_path,fault,code,record_property):
    healthy,topology,ranks=reference
    args=params(tmp_path)
    failed,size=run_case(args,ROOT,'gpu',ranks,'fault_'+fault,4,grid='32,32,32',topology=topology,
                         checkpoint_interval=1,insitu_config=config())
    output=failed/'outdat/render'
    resource_and_frames(failed,ranks,[2,4])
    journals=[]
    for rank in range(ranks):
        journal=json.loads((output/f'missing.q_surface.step00000002.rank{rank:08d}.json').read_text())
        journals.append(journal)
        assert journal['status']=='missing' and journal['errno']==code
        assert journal['step']==2 and journal['time']==.002 and journal['phase']=='stage_eps'
        receipt=json.loads((output/f'mesh_step00000002_rank{rank}.json').read_text())
        assert receipt['products']['q_surface']['image']==journal
        name=f'sample.statistics.step00000004.rank{rank:08d}.bin'
        assert (output/name).read_bytes()==(healthy/'outdat/render'/name).read_bytes()
    assert all(journal==journals[0] for journal in journals)
    assert not (output/'q_surface.step00000002.jpeg').exists()
    assert not (output/'q_surface.step00000002.eps').exists()
    for step in (2,4):
        name=f'q_surface.step{step:08d}.pvtp'
        compare_geometry(healthy/'outdat/render'/name,output/name,['u','v','w','Q_rs'])
    with Image.open(output/'q_surface.step00000004.jpeg') as picture:
        assert np.asarray(picture).shape==(600,800,3)
        assert np.any(np.asarray(picture)<245)
    for extension in ('jpeg','eps'):
        name='q_surface.step00000004.'+extension
        assert (output/name).read_bytes()==(healthy/'outdat/render'/name).read_bytes()
    assert 'ASTR_INSITU_IMAGE_MISSING' in (failed/'run.log').read_text()
    for name in ('state.h5','statistics.h5'):
        compare_fields(healthy/FINAL/name,failed/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (healthy/FINAL/name).read_bytes()==(failed/FINAL/name).read_bytes()
    resumed,_=run_case(args,ROOT,'gpu',ranks,'resumed',4,grid='32,32,32',topology=topology,
                      restore=failed/'outdat/new/checkpoints/step000000000003',
                      checkpoint_interval=1,insitu_config=config())
    resource_and_frames(resumed,ranks,[4])
    assert not list((resumed/'outdat/render').glob('missing.*'))
    for name in ('state.h5','statistics.h5'):
        compare_fields(failed/FINAL/name,resumed/FINAL/name)
    for path in (resumed/'outdat/render').rglob('*'):
        if path.is_file() and (path.suffix in ('.jpeg','.eps','.vtp') or path.name.startswith('sample.statistics')):
            assert path.read_bytes()==(output/path.relative_to(resumed/'outdat/render')).read_bytes()
    record_property('directory_bytes',size)
    record_property('missing_image',json.dumps(journals[0]))


@pytest.mark.parametrize('fault,message',(('unknown','injected unknown image failure'),
    ('journal','Cannot record missing in-situ image'),('geometry','injected geometry failure')))
def test_nonrecoverable_collective_stop(tmp_path,fault,message):
    case,_=run_case(params(tmp_path),ROOT,'gpu',2,'fault_'+fault,4,grid='32,32,32',
                    checkpoint_interval=99,insitu_config=config(),reject=message)
    assert not list((case/'outdat/render').glob('mesh_step00000004*'))
    assert not (case/'outdat/render'/'q_surface.step00000004.jpeg').exists()
