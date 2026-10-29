"""M12 plane-only supply against the immutable full-volume reference."""
import json
import os
from pathlib import Path
import re
import struct
import subprocess

import numpy as np
from PIL import Image
import pytest

from run_m12_hd_coverage import prepare_case,MPIEXEC
from run_m12_boundary_compare import environment
from run_output_restart_validation import compare_fields
from insitu_resource_monitor import run_monitored
from insitu_cfl_gate import CflGate

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/'tests/gpu_validation/out/insitu_memory_20261018'
FROZEN=CAMPAIGN/'frozen/astr'
CURRENT=Path(os.environ.get('ASTR_OUTPUT_INSITU_EXE',ROOT/'build_m12_device_render/bin/astr'))
FINAL='outdat/new/checkpoints/step000000000002/state.h5'
DIRECTORY_BUDGET=512*1024**2


def launch(directory,executable,profile,baseline=None,steps=2,memcheck=False,pipeline_file=None,restore=None,
           extra_device_budget=2*1024**3,override=False):
    prepare_case(ROOT.parent/'M12_C13',directory,(64,32,24),steps,True,checkpoint=True)
    config=directory/'datin/input.insitu'
    text=config.read_text().replace('host_budget_bytes=17179869184','host_budget_bytes=4294967296').replace(
        'device_budget_bytes=6442450944',f'device_budget_bytes={extra_device_budget}').replace(
        'device_reserve_bytes=2147483648','device_reserve_bytes=1073741824')
    if profile=='off':
        text='&insitu_run\n enabled=f\n/\n'
    else:
        periods='1000000000,2,1000000000' if profile=='slice_only' else '2,2,1000000000'
        text=text.replace('product_steps=250,100,250','product_steps='+periods)
    config.write_text(text)
    if pipeline_file is not None:
        config.write_text(config.read_text().replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),str(pipeline_file)))
    if restore is not None:
        output=directory/'datin/input.output'
        output.write_text(output.read_text().replace("restore_directory=''",f"restore_directory='{restore}'").replace(
            "restart_output='saved'","restart_output='override'" if override else "restart_output='saved'"))
    env=environment('2,1,1')
    env.update(ASTR_GPU_MP_POSITIVITY='flux',ASTR_INSITU_CONFIG='datin/input.insitu',
               ASTR_INSITU_TEST_ORACLE_IO='0',ASTR_INSITU_MEMORY_TRACE='1',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
        env.pop(name,None)
    command=[str(MPIEXEC),'--mca','pml','ob1','--mca','btl','self,vader,tcp','--bind-to','none',
        '-np','2']
    if memcheck:
        command+=['compute-sanitizer','--tool','memcheck','--error-exitcode','99',
                  '--log-file',str(directory/'memcheck.%p.log')]
    command+=[str(executable),'run','datin/input.dat']
    first=0 if restore is None else int(restore.name.removeprefix('step'))
    cfl=CflGate(directory/'run.log',1.,.02,first,steps)
    with (directory/'run.log').open('w') as log:
        result=run_monitored(command,directory,env,log,directory/'resources.sampled.json',baseline=baseline,
            device_extra_budget_bytes=extra_device_budget,timeout_seconds=300,check_progress=cfl.check)
    assert 'The job is done!' in (directory/'run.log').read_text()
    assert sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())<=DIRECTORY_BUDGET
    return result


def assert_render_control(actual,reference):
    from test_checkpoint_export import export
    normalized=[]
    for raw in (actual,reference):
        if raw[:8]==b'ASTRIR07':
            from run_output_restart_validation import native_render_payload
            raw=native_render_payload(raw)
        assert raw[:8] in (b'ASTRIR04',b'ASTRIR06')
        if raw[:8]==b'ASTRIR06':
            assert raw[-64:-56]==b'ASTRMC01'
            mask,*ranges=struct.unpack('<q6d',raw[-56:])
            assert 0<=mask<=7 and np.isfinite(ranges).all()
            for i in range(3):
                assert (ranges[2*i]<ranges[2*i+1]) if mask&(1<<i) else ranges[2*i:2*i+2]==[0.,0.]
            raw=raw[:-64]
        normalized.append(raw)
    a,b=normalized
    assert len(a)==len(b) and a[8:112]==b[8:112] and a[128:]==b[128:]
    decoration=0x4750554445524956^0x44455650524F4453^0x5354414E44415244
    for raw,script in [(actual,ROOT/'scripts/insitu/device_render_pipeline.py'),
                       (reference,CAMPAIGN/'frozen/insitu_im5/device_render_pipeline.py')]:
        size,crc=export.fingerprint(script)
        assert struct.unpack_from('<2Q',raw,112)==(size,crc^decoration)


def test_capture_frozen_m12_windows(tmp_path):
    completed=os.environ.get('ASTR_INSITU_MEMORY_M12_CAPTURE_PREFIX')
    if completed:
        off=Path(completed)/'off'
        assert 'The job is done!' in (off/'run.log').read_text()
        baseline=json.loads((off/'resources.sampled.json').read_text())
    else:
        off=tmp_path/'off';baseline=launch(off,FROZEN,'off')
    references=dict(off=str(off))
    for profile in ('slice_only','mixed'):
        if completed and profile=='slice_only':
            case=Path(completed)/profile
            assert 'The job is done!' in (case/'run.log').read_text()
            assert sum(p.stat().st_size for p in case.rglob('*') if p.is_file())<=DIRECTORY_BUDGET
        else:
            case=tmp_path/profile
            launch(case,FROZEN,profile,baseline,pipeline_file=CAMPAIGN/'frozen/insitu_im5/device_render_pipeline.py',
                extra_device_budget=(3 if profile=='mixed' else 2)*1024**3)
        compare_fields(case/FINAL,off/FINAL)
        assert list((case/'outdat/render').glob('*.jpeg'))
        references[profile]=str(case)
    (tmp_path/'reference.json').write_text(json.dumps(references,indent=2)+'\n')


@pytest.mark.parametrize('profile',['slice_only','mixed'])
def test_local_plane_supply_matches_reference(tmp_path,profile,record_property):
    references=json.loads(Path(os.environ['ASTR_INSITU_MEMORY_M12_REFERENCE']).read_text())
    old=Path(references[profile]);off=Path(references['off'])
    launch(tmp_path/'current',CURRENT,profile,json.loads((off/'resources.sampled.json').read_text()),
        extra_device_budget=(3 if profile=='mixed' else 2)*1024**3)
    case=tmp_path/'current'
    compare_fields(case/FINAL,old/FINAL)
    for name in ('control.bin','insitu_control.bin'):
        a,b=(p/Path(FINAL).parent/name for p in (case,old))
        if name=='insitu_control.bin':
            assert_render_control(a.read_bytes(),b.read_bytes())
        else:
            assert a.read_bytes()==b.read_bytes(),name
    images=sorted(p.name for p in (old/'outdat/render').glob('*.jpeg'))
    assert images==sorted(p.name for p in (case/'outdat/render').glob('*.jpeg'))
    for name in images:
        with Image.open(old/'outdat/render'/name) as a,Image.open(case/'outdat/render'/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    if profile=='slice_only':
        text=(case/'run.log').read_text()
        assert 'stage=frame-inputs ' not in text
        assert 'stage=m12-plane-inputs ' in text
        assert 'stage=canonical-state-and-velocity ' not in text
        payloads=re.findall(r'stage=m12-plane-inputs domain=sample rank=\d+ payload_bytes=(\d+)',text)
        assert len(payloads)==2 and all(int(value)==48*2*33*(33+25) for value in payloads),payloads
    report=dict(reference=str(old),candidate=str(case),
        reference_resources=json.loads((old/'resources.sampled.json').read_text()),
        candidate_resources=json.loads((case/'resources.sampled.json').read_text()),
        reference_eps_count=len(list((old/'outdat/render').glob('*.eps'))),
        candidate_eps_count=len(list((case/'outdat/render').glob('*.eps'))),
        jpeg_pixels_identical=True,authoritative_state_and_rng_identical=True)
    (tmp_path/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    record_property('comparison',report)


def test_local_plane_two_frames_memcheck(tmp_path):
    references=json.loads(Path(os.environ['ASTR_INSITU_MEMORY_M12_REFERENCE']).read_text())
    baseline=json.loads((Path(references['off'])/'resources.sampled.json').read_text())
    before=CAMPAIGN/'frozen/astr_im1_im3_im6'
    launch(tmp_path/'full',before,'slice_only',baseline,steps=4,
        pipeline_file=CAMPAIGN/'frozen/insitu_im5/device_render_pipeline.py')
    launch(tmp_path/'local',CURRENT,'slice_only',baseline,steps=4,memcheck=True)
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','control.bin','insitu_control.bin'):
        first,second=(tmp_path/d/final/name for d in ('full','local'))
        if name.endswith('.h5'):
            compare_fields(first,second)
        elif name=='insitu_control.bin':
            assert_render_control(second.read_bytes(),first.read_bytes())
        else:
            assert first.read_bytes()==second.read_bytes(),name
    images=sorted((tmp_path/'full/outdat/render').glob('*.jpeg'))
    assert {int(p.stem.split('step')[-1]) for p in images}=={2,4}
    for image in images:
        with Image.open(image) as a,Image.open(tmp_path/'local/outdat/render'/image.name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=image.name)
    logs=list((tmp_path/'local').glob('memcheck.*.log'))
    assert len(logs)==2
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)


def test_captured_plane_control_identity_is_not_a_clock_change():
    refs=json.loads((CAMPAIGN/'m12_same_format_reference.json').read_text())
    old=Path(refs['slice_only'])
    case=CAMPAIGN/'im7_group_remaining/test_local_plane_supply_matche0/current'
    assert 'The job is done!' in (case/'run.log').read_text()
    compare_fields(case/FINAL,old/FINAL)
    a,b=((p/Path(FINAL).parent/'insitu_control.bin').read_bytes() for p in (case,old))
    assert_render_control(a,b)
    changed=bytearray(a);changed[224]^=1
    with pytest.raises(AssertionError): assert_render_control(bytes(changed),b)
    changed=bytearray(a);changed[120]^=1
    with pytest.raises(AssertionError): assert_render_control(bytes(changed),b)
    for p in (old/'outdat/render').glob('*.jpeg'):
        with Image.open(p) as x,Image.open(case/'outdat/render'/p.name) as y:
            np.testing.assert_array_equal(np.asarray(x),np.asarray(y),err_msg=p.name)


def test_local_plane_same_version_exact_continuation(tmp_path):
    first=tmp_path/'first'
    launch(first,CURRENT,'slice_only',steps=2)
    source=first/'outdat/new/checkpoints/step000000000002'
    from run_output_restart_validation import native_render_payload
    raw=native_render_payload((source/'insitu_control.bin').read_bytes())
    assert raw[:8]==b'ASTRIR06' and raw[-64:-56]==b'ASTRMC01'
    mask,*ranges=struct.unpack('<q6d',raw[-56:])
    record=json.loads((first/'outdat/render/mesh_step00000002_rank0.json').read_text())
    assert mask==7
    expected=[v for name in ('velocity_slice','temperature_slice','spanwise_slice')
              for v in record['products'][name]['color_range']]
    np.testing.assert_array_equal(ranges,expected)
    before={p:p.read_bytes() for p in source.iterdir() if p.is_file()}
    launch(tmp_path/'continuous',CURRENT,'slice_only',steps=4)
    launch(tmp_path/'resumed',CURRENT,'slice_only',steps=4,restore=source)
    final='outdat/new/checkpoints/step000000000004'
    compare_fields(tmp_path/'continuous'/final/'state.h5',tmp_path/'resumed'/final/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        assert (tmp_path/'continuous'/final/name).read_bytes()==(tmp_path/'resumed'/final/name).read_bytes(),name
    for p in (tmp_path/'resumed/outdat/render').glob('*.jpeg'):
        with Image.open(p) as x,Image.open(tmp_path/'continuous/outdat/render'/p.name) as y:
            np.testing.assert_array_equal(np.asarray(x),np.asarray(y),err_msg=p.name)
    assert before=={p:p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_pre_catalog_plane_checkpoint_is_not_restart_compatible(tmp_path):
    source=CAMPAIGN/'im7_group_remaining/test_local_plane_supply_matche0/current/outdat/new/checkpoints/step000000000002'
    before={p:p.read_bytes() for p in source.iterdir() if p.is_file()}
    with pytest.raises(subprocess.CalledProcessError):
        launch(tmp_path/'rejected',CURRENT,'slice_only',steps=4,restore=source)
    log=(tmp_path/'rejected/run.log').read_text()
    assert 'ASTRIR07' in log
    assert not list((tmp_path/'rejected/outdat/new').rglob('COMPLETE'))
    with pytest.raises(subprocess.CalledProcessError):
        launch(tmp_path/'override',CURRENT,'slice_only',steps=4,restore=source,override=True)
    assert 'ASTRIR07' in (tmp_path/'override/run.log').read_text()
    assert before=={p:p.read_bytes() for p in source.iterdir() if p.is_file()}
