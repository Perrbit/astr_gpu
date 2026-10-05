"""Actual native statistics/render/checkpoint interruption and fatal gates."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from run_output_restart_validation import run_case,compare_fields
from test_output_insitu_restart import arguments,configuration
from test_insitu_image_publication import resource_and_frames
from test_output_series_repair import seal_file_records,repair
from test_checkpoint_bundle import fault_library as protection_library

ROOT=Path(__file__).resolve().parents[2]
FINAL='outdat/new/checkpoints/step000000000006'


def fingerprints(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def config():
    return configuration(interval=2).replace('statistics_window=0.0005,0.0115',
        'statistics_window=0.0005,0.0055').replace('render=.true.,',
        "render=.true., derivative_backend='gpu',products='q_surface',").replace(
        str(ROOT/'scripts/insitu/tgv_pipeline.py'),str(ROOT/'tests/gpu_validation/insitu_is4_failure_pipeline.py'))


def params(path):
    args=arguments(path); args.statistics=False; args.directory_budget_bytes=256*1024**2
    args.runtime_timeout_seconds=45
    return args


@pytest.fixture(scope='module')
def fault_library(tmp_path_factory):
    target=tmp_path_factory.mktemp('is4_preload')/'fault.so'
    compiler=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpicc')
    subprocess.run([str(compiler),'-shared','-fPIC','-Wall','-Wextra','-Werror',
        str(ROOT/'tests/gpu_validation/insitu_is4_fault_preload.c'),'-ldl','-o',str(target)],check=True)
    return target


@pytest.fixture(scope='module',params=((2,1,1),(2,2,1)),ids=('np2','np4'))
def reference(request,tmp_path_factory,protection_library):
    topology=request.param; ranks=topology[0]*topology[1]*topology[2]
    args=params(tmp_path_factory.mktemp('is4_failure_reference'))
    case,_=run_case(args,ROOT,'gpu',ranks,'reference',6,grid='32,32,32',topology=topology,
                    checkpoint_interval=2,insitu_config=config(),
                    publication_fault=(protection_library,'protect_batch',2,0))
    resource_and_frames(case,ranks,[2,4,6])
    return case,topology,ranks


@pytest.mark.parametrize('phase,message',(
    ('nonfinite_sample','invalid selected velocity state'),
    ('missing_field','Missing required product fields'),
    ('statistics_failure','checkpoint state rejected:'),
    ('mpi_failure','Catalyst status consensus MPI failure')))
def test_required_failure_is_fatal(reference,tmp_path,fault_library,phase,message):
    _,topology,ranks=reference
    args=params(tmp_path)
    case,_=run_case(args,ROOT,'gpu',ranks,'fault_'+phase,6,grid='32,32,32',topology=topology,
        checkpoint_interval=2,insitu_config=config(),reject=message,
        test_fault=(fault_library,phase,4),failure_after_start=True)
    log=(case/'run.log').read_text()
    if phase!='missing_field':
        assert 'ASTR_IS4_'+{'nonfinite_sample':'NONFINITE_SAMPLE','statistics_failure':'STATISTICS_ERROR',
                          'mpi_failure':'MPI_ERROR'}[phase]+'_INJECTED' in log
    assert not (case/FINAL).exists()
    assert not (case/'outdat/new/checkpoints/step000000000004/COMPLETE').exists()
    assert not list((case/'outdat/render').glob('missing.*'))
    assert not (case/'outdat/render/lifecycle_rank0.json').exists()
    assert not list((case/'outdat/render').glob('*step00000006*'))


@pytest.mark.parametrize('phase',('before_statistics','before_render_control','before_complete'))
def test_interrupted_candidate_and_old_complete_recovery(reference,tmp_path,fault_library,phase,record_property):
    healthy,topology,ranks=reference
    args=params(tmp_path)
    failed,size=run_case(args,ROOT,'gpu',ranks,'fault_'+phase,6,grid='32,32,32',topology=topology,
        checkpoint_interval=2,insitu_config=config(),reject='ASTR_IS4_INTERRUPTION',
        test_fault=(fault_library,phase,4),failure_after_start=True)
    root=failed/'outdat/new'; candidate=root/'checkpoints/step000000000004.tmp'
    source=root/'checkpoints/step000000000002'
    assert candidate.is_dir() and (candidate/'state.h5').is_file()
    assert not (candidate/'COMPLETE').exists()
    assert not (root/'checkpoints/step000000000004').exists()
    assert (root/'checkpoints/LATEST').read_text()==source.name+'\n'
    before=json.loads((failed/'outdat/render/is4_backup_before.json').read_text())
    assert before==fingerprints(source)
    expected=healthy/'outdat/new/checkpoints'/source.name
    for name in ('state.h5','statistics.h5'):
        compare_fields(source/name,expected/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (source/name).read_bytes()==(expected/name).read_bytes()
    original=fingerprints(root)
    run_case(args,ROOT,'gpu',ranks,'rejected_partial',6,grid='32,32,32',topology=topology,
        restore=candidate,checkpoint_interval=2,insitu_config=config(),
        reject='invalid new checkpoint bundle')
    resumed,_=run_case(args,ROOT,'gpu',ranks,'recovered',6,grid='32,32,32',topology=topology,
        restore=source,checkpoint_interval=2,insitu_config=config())
    resource_and_frames(resumed,ranks,[4,6])
    for name in ('state.h5','statistics.h5'):
        compare_fields(healthy/FINAL/name,resumed/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (healthy/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    for path in (resumed/'outdat/render').rglob('*'):
        if path.is_file() and (path.suffix in ('.jpeg','.eps','.vtp') or path.name.startswith('sample.statistics')):
            assert path.read_bytes()==(healthy/'outdat/render'/path.relative_to(resumed/'outdat/render')).read_bytes()
    assert original==fingerprints(root)
    record_property('interruption_phase',phase); record_property('directory_bytes',size)


@pytest.mark.parametrize('mismatch',('statistics','render_control'))
def test_resealed_cross_step_pair_is_rejected(reference,tmp_path,mismatch):
    healthy,topology,ranks=reference
    root=tmp_path/'mixed'
    shutil.copytree(healthy/'outdat/new/resources',root/'resources')
    batch=root/'checkpoints/step000000000004'
    shutil.copytree(healthy/'outdat/new/checkpoints/step000000000004',batch)
    member='statistics.h5' if mismatch=='statistics' else 'insitu_control.bin'
    shutil.copyfile(healthy/'outdat/new/checkpoints/step000000000002'/member,batch/member)
    names=[row.split()[0] for row in (batch/'MANIFEST').read_text().splitlines()[2:]]
    (batch/'MANIFEST').write_text(seal_file_records(batch,names,'ASTR_CHECKPOINT_BUNDLE 1'))
    size,crc=repair.fingerprint(batch/'MANIFEST')
    (batch/'COMPLETE').write_text(f'ASTR_COMPLETE_1 {size} {crc:016X}\n')
    before=fingerprints(root)
    message='statistics file clock mismatch' if mismatch=='statistics' else 'native render clock/version mismatch'
    run_case(params(tmp_path),ROOT,'gpu',ranks,'reject_mixed',6,grid='32,32,32',topology=topology,
             restore=batch,checkpoint_interval=2,insitu_config=config(),reject=message)
    assert before==fingerprints(root)
