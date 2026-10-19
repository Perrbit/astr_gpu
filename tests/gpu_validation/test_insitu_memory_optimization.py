"""Bounded same-state image and memory windows for IM0 onward."""
import json
import os
from pathlib import Path
import re
import subprocess
import sqlite3

import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case,compare_fields
from test_output_insitu_restart import arguments,configuration
from test_insitu_device_products import resident_configuration

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/'tests/gpu_validation/out/insitu_memory_20261018'
FROZEN=CAMPAIGN/'frozen/astr'
CURRENT=Path(os.environ.get('ASTR_OUTPUT_INSITU_EXE',ROOT/'build_m12_device_render/bin/astr'))
LIBRARY=ROOT.parent/'astr_dependencies/build/paraview-6.1.1-astr-device-gcc13/lib/catalyst'
FINAL='outdat/new/checkpoints/step000000000008/state.h5'


def settings(profile=None):
    if profile is None:
        result=configuration(render=False,statistics=False,interval=1).replace('enabled=t','enabled=f')
    else:
        result=resident_configuration('standard-device',statistics=False,interval=1,profile=profile)
    return re.sub(r"implementation_path='[^']*'",f"implementation_path='{LIBRARY}'",result)


def test_capture_frozen_memory_windows(tmp_path):
    baseline=None
    references={}
    for profile in (None,'q_surface','velocity_slice','streamlines'):
        name=profile or 'off'
        args=arguments(tmp_path/name)
        args.statistics=False;args.executable=FROZEN
        args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        case,_=run_case(args,ROOT,'gpu',2,name,8,grid='32,32,32',checkpoint_interval=8,
            insitu_config=settings(profile),monitor_resources=True,resource_baseline=baseline)
        references[name]=str(case)
        if profile is None:
            baseline=json.loads((case/'resources.sampled.json').read_text())
    (tmp_path/'reference.json').write_text(json.dumps(references,indent=2)+'\n')


@pytest.fixture(scope='module')
def off_reference(tmp_path_factory):
    recorded=os.environ.get('ASTR_INSITU_MEMORY_REFERENCE')
    if recorded:
        references=json.loads(Path(recorded).read_text())
        case=Path(references['off'])
        return case,json.loads((case/'resources.sampled.json').read_text())
    args=arguments(tmp_path_factory.mktemp('off'))
    args.statistics=False
    args.insitu_write_eps='0'
    args.executable=FROZEN
    args.directory_budget_bytes=256*1024**2
    case,_=run_case(args,ROOT,'gpu',2,'off',8,grid='32,32,32',
        checkpoint_interval=8,insitu_config=settings(),monitor_resources=True)
    return case,json.loads((case/'resources.sampled.json').read_text())


@pytest.mark.parametrize('profile',['q_surface','velocity_slice','streamlines'])
def test_same_state_memory_window(tmp_path,profile,off_reference,record_property):
    off,baseline=off_reference
    cases={}
    for name,executable in [('frozen',FROZEN),('current',CURRENT)]:
        if name=='frozen' and os.environ.get('ASTR_INSITU_MEMORY_REFERENCE'):
            cases[name]=Path(json.loads(Path(os.environ['ASTR_INSITU_MEMORY_REFERENCE']).read_text())[profile])
            continue
        args=arguments(tmp_path/name)
        args.statistics=False;args.executable=executable
        args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        case,_=run_case(args,ROOT,'gpu',2,name,8,grid='32,32,32',checkpoint_interval=8,
            insitu_config=settings(profile),monitor_resources=True,resource_baseline=baseline,
            insitu_memory_trace=name=='current')
        compare_fields(case/FINAL,off/FINAL)
        cases[name]=case
    first,second=(cases[name]/'outdat/render' for name in ('frozen','current'))
    assert not list(second.glob('*.eps'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert images and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    text=(cases['current']/'run.log').read_text()
    rows=re.findall(r'ASTR_INSITU_MEMORY stage=(\S+) domain=(\S+) (?:rank|device)=(\d+) '
                    r'payload_bytes=(\d+) device_used_bytes=(\d+)',text)
    assert rows and any(row[0]=='frame-inputs' for row in rows),text
    if profile=='q_surface':
        assert any(row[0]=='display-metadata' for row in rows)
    metadata=[int(row[3]) for row in rows if row[0]=='display-metadata']
    assert metadata and not any(metadata),metadata
    measurements={name:json.loads((case/'resources.sampled.json').read_text()) for name,case in cases.items()}
    (tmp_path/'memory.json').write_text(json.dumps(dict(profile=profile,allocations=rows,
        measurements=measurements,images=images),indent=2)+'\n')
    record_property('memory',measurements)


def test_q_only_payload_is_one_scalar(tmp_path,off_reference):
    _,baseline=off_reference
    args=arguments(tmp_path);args.statistics=False;args.executable=CURRENT;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2
    case,_=run_case(args,ROOT,'gpu',2,'q_only',8,grid='32,32,32',checkpoint_interval=8,
        insitu_config=settings('q_surface'),monitor_resources=True,resource_baseline=baseline,
        insitu_memory_trace=True)
    rows=re.findall(r'ASTR_INSITU_MEMORY stage=diagnostic-supply domain=sample rank=\d+ payload_bytes=(\d+)',
                    (case/'run.log').read_text())
    assert len(rows)==16
    assert all(int(value)==8*17*33*33 for value in rows),rows


def test_combined_products_memory_window(tmp_path,off_reference):
    off,baseline=off_reference
    cases={}
    for name,executable in [('frozen',FROZEN),('current',CURRENT)]:
        args=arguments(tmp_path/name);args.executable=executable;args.statistics=False;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        cases[name],_=run_case(args,ROOT,'gpu',2,name,8,grid='32,32,32',checkpoint_interval=8,
            insitu_config=settings('all'),monitor_resources=True,resource_baseline=baseline,
            insitu_memory_trace=name=='current')
        compare_fields(cases[name]/FINAL,off/FINAL)
    first,second=(cases[name]/'outdat/render' for name in ('frozen','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==32 and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    report={name:json.loads((case/'resources.sampled.json').read_text()) for name,case in cases.items()}
    (tmp_path/'memory.json').write_text(json.dumps(report,indent=2)+'\n')


def test_capture_frozen_statistics_window(tmp_path):
    args=arguments(tmp_path);args.executable=FROZEN
    args.directory_budget_bytes=256*1024**2
    case,_=run_case(args,ROOT,'gpu',2,'statistics',8,grid='32,32,32',checkpoint_interval=4,
        insitu_config=configuration(render=False,statistics=True),monitor_resources=True)
    (tmp_path/'reference.json').write_text(json.dumps(dict(case=str(case)),indent=2)+'\n')


def test_statistics_results_are_not_resident(tmp_path):
    reference=Path(json.loads(Path(os.environ['ASTR_INSITU_MEMORY_STATISTICS_REFERENCE']).read_text())['case'])
    args=arguments(tmp_path);args.executable=CURRENT
    args.directory_budget_bytes=256*1024**2
    config=configuration(render=False,statistics=True)
    whole,_=run_case(args,ROOT,'gpu',2,'statistics',8,grid='32,32,32',checkpoint_interval=4,
        insitu_config=config,monitor_resources=True,insitu_memory_trace=True)
    final='outdat/new/checkpoints/step000000000008'
    for name in ('state.h5','statistics.h5'):
        compare_fields(whole/final/name,reference/final/name)
    outputs=list((reference/'outdat/render').glob('*.statistics.*.bin'))
    assert len(outputs)==2
    for path in outputs:
        assert path.read_bytes()==(whole/'outdat/render'/path.name).read_bytes(),path.name
    rows=re.findall(r'ASTR_INSITU_MEMORY stage=statistics-persistent domain=sample rank=\d+ payload_bytes=(\d+)',
                    (whole/'run.log').read_text())
    samples=re.findall(r'ASTR_INSITU_GPU_STATS rank=\d+ samples=(\d+)',(whole/'run.log').read_text())
    assert len(samples)==2 and len(rows)==sum(map(int,samples))
    assert all(int(value)<31*8*16*32*32 for value in rows),rows
    resumed,_=run_case(args,ROOT,'gpu',2,'statistics_resumed',8,grid='32,32,32',checkpoint_interval=4,
        insitu_config=config,restore=whole/'outdat/new/checkpoints/step000000000004',insitu_memory_trace=True)
    for name in ('state.h5','statistics.h5'):
        compare_fields(whole/final/name,resumed/final/name)
    for path in (whole/'outdat/render').glob('*.statistics.*.bin'):
        assert path.read_bytes()==(resumed/'outdat/render'/path.name).read_bytes(),path.name


def test_curved_statistics_remain_exact(tmp_path):
    from test_insitu_curve_statistics import check_independent,configuration as curve_configuration
    cases={}
    for name,executable in [('frozen',FROZEN),('current',CURRENT)]:
        args=arguments(tmp_path/name,'y');args.executable=executable
        args.directory_budget_bytes=256*1024**2
        cases[name],_=run_case(args,ROOT,'gpu',2,name,4,grid='32,32,32',tgv_mapping='periodic',
            checkpoint_interval=1,insitu_config=curve_configuration())
    check_independent(cases['current'],2)
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases['current']/final/name,cases['frozen']/final/name)
    args=arguments(tmp_path/'resumed','y');args.executable=CURRENT
    args.directory_budget_bytes=256*1024**2
    resumed,_=run_case(args,ROOT,'gpu',2,'curved_restart',4,grid='32,32,32',tgv_mapping='periodic',
        checkpoint_interval=1,insitu_config=curve_configuration(),
        restore=cases['current']/'outdat/new/checkpoints/step000000000003')
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases['current']/final/name,resumed/final/name)
    for path in (cases['current']/'outdat/render').glob('*.statistics.*.bin'):
        assert path.read_bytes()==(resumed/'outdat/render'/path.name).read_bytes()
        assert path.read_bytes()==(cases['frozen']/'outdat/render'/path.name).read_bytes()


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_wall_scalar_reduction_preserves_images(tmp_path,pipeline):
    from test_insitu_channel_walls import arguments as wall_arguments
    cases={}
    config=settings('channel_walls').replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    for name,executable in [('frozen',FROZEN),('current',CURRENT)]:
        recorded=os.environ.get('ASTR_INSITU_MEMORY_REFERENCE')
        if name=='frozen' and recorded:
            cases[name]=Path(json.loads(Path(recorded).read_text())['wall_'+pipeline])
            assert 'The job is done!' in (cases[name]/'run.log').read_text()
            assert (cases[name]/'datin/input.insitu').read_text()==config
            continue
        args=wall_arguments(tmp_path/name,'gpu',samples=False)
        args.executable=executable;args.insitu_write_eps='0'
        cases[name],_=run_case(args,ROOT,'gpu',2,name,4,topology=(2,1,1),checkpoint_interval=1,
            insitu_config=config,insitu_memory_trace=name=='current')
    final='outdat/new/checkpoints/step000000000004/state.h5'
    compare_fields(cases['current']/final,cases['frozen']/final)
    first,second=(cases[name]/'outdat/render' for name in ('frozen','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==12 and images==sorted(p.name for p in second.glob('*.jpeg'))
    assert not list(second.glob('*.eps'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    rows=re.findall(r'stage=display-metadata domain=geometry device=\d+ payload_bytes=(\d+)',
                    (cases['current']/'run.log').read_text())
    assert len(rows)==24 and not any(map(int,rows)),rows


def test_mean_products_and_readonly_preview(tmp_path,monkeypatch):
    from test_insitu_preview import prepare_native_preview,source_snapshot
    config=settings('all').replace('statistics=.false.','statistics=.true.').replace(
        'statistics_window=0.0005,0.0115','statistics_window=0.003,0.004')
    cases={}
    for name,executable in [('frozen',CAMPAIGN/'frozen/astr_im2'),('current',CURRENT)]:
        recorded=os.environ.get('ASTR_INSITU_MEMORY_REFERENCE')
        if name=='frozen' and recorded:
            cases[name]=Path(json.loads(Path(recorded).read_text())['mean_products'])
            assert 'The job is done!' in (cases[name]/'run.log').read_text()
            assert (cases[name]/'datin/input.insitu').read_text()==config
            continue
        args=arguments(tmp_path/name);args.executable=executable;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        cases[name],_=run_case(args,ROOT,'gpu',2,'timing_pinned',4,grid='32,32,32',
            checkpoint_interval=1,insitu_config=config,insitu_memory_trace=name=='current')
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases['current']/final/name,cases['frozen']/final/name)
    first,second=(cases[name]/'outdat/render' for name in ('frozen','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==18 and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    monkeypatch.setenv('ASTR_INSITU_PREVIEW_SOURCE_ROOT',str(cases['current'].parent))
    monkeypatch.setenv('ASTR_OUTPUT_INSITU_EXE',str(CURRENT))
    monkeypatch.setenv('ASTR_OUTPUT_INSITU_BACKEND',str(LIBRARY))
    preview=tmp_path/'preview';preview.mkdir()
    source,_,executable,launcher,env,output=prepare_native_preview(preview,2,True)
    env['ASTR_INSITU_WRITE_EPS']='0'
    before=source_snapshot(source)
    with (preview/'run.log').open('w') as log:
        subprocess.run([str(launcher),'-np','2',str(executable),'insitu-preview','datin/input.tgv'],
            cwd=preview,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    assert source_snapshot(source)==before
    log=(preview/'run.log').read_text()
    assert log.count('flow_mismatches=0 statistics_mismatches=0')==2,log
    assert 'stage=image-only-owner' not in log
    assert len(list(output.glob('*.jpeg')))==4 and not list(output.glob('*.eps'))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_image_only_actual_memcheck(tmp_path,pipeline):
    args=arguments(tmp_path);args.executable=CURRENT;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2;args.runtime_timeout_seconds=300
    config=settings('all').replace('statistics=.false.','statistics=.true.').replace('initial_frame=f','initial_frame=t')
    config=config.replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    case,_=run_case(args,ROOT,'gpu',2,'image_memcheck',2,enabled=False,grid='32,32,32',
        insitu_config=config,no_field_io=True,memcheck=True,insitu_memory_trace=True)
    reports=list(case.glob('memcheck.*.log'))
    assert len(reports)==2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)
    assert 'stage=image-only-owner ' in (case/'run.log').read_text()
    for rank in range(2):
        lifecycle=json.loads((case/f'outdat/render/lifecycle_rank{rank}.json').read_text())
        assert lifecycle==dict(frames=[[0,0.],[1,.001],[2,.002]],finalized=True)
    assert not list((case/'outdat/render').glob('*.eps'))


def test_image_only_actual_transfer_ledger(tmp_path):
    from test_insitu_device_native_trace import copies
    args=arguments(tmp_path);args.executable=CURRENT;args.statistics=False;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2;args.runtime_timeout_seconds=300
    args.nsys_trace_domains='cuda,nvtx,mpi'
    case,_=run_case(args,ROOT,'gpu',2,'image_trace',2,enabled=False,grid='32,32,32',
        insitu_config=settings('all'),no_field_io=True,nsys_trace=True,insitu_timing=True)
    ledger={}
    for rank in range(2):
        path=case/f'trace.rank{rank}.sqlite'
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
            sections={}
            for name,allowed in (('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION',{4,8,72}),
                    ('ASTR_IS8_RESIDENT_TRACE_PACK',{4,8}),('ASTR_IS8_TRACE_CONTROL_READ',{1024,2048})):
                values=[n for k,n in copies(db,name) if k=='CUDA_MEMCPY_KIND_DTOH']
                assert values and set(values)<=allowed,(name,values)
                sections[name]=sum(values)
            for name in ('ASTR_IS8_RESIDENT_TRACE_CONCATENATE','ASTR_IS8_RESIDENT_RENDER'):
                values=copies(db,name)
                assert all(k=='CUDA_MEMCPY_KIND_DTOD' for k,_ in values),(name,values)
                sections[name]=sum(n for _,n in values)
            ledger[rank]=sections
    (tmp_path/'transfers.json').write_text(json.dumps(ledger,indent=2)+'\n')


@pytest.mark.parametrize('profile',['q_surface','streamlines'])
@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_curve_image_only_products(tmp_path,profile,pipeline):
    cases={}
    for name,executable in [('frozen',CAMPAIGN/'frozen/astr_im2'),('current',CURRENT)]:
        args=arguments(tmp_path/name);args.executable=executable;args.statistics=False;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        config=settings(profile).replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
        cases[name],_=run_case(args,ROOT,'gpu',2,name,4,grid='32,32,32',tgv_mapping='periodic',
            checkpoint_interval=4,insitu_config=config,insitu_memory_trace=name=='current')
    final='outdat/new/checkpoints/step000000000004/state.h5'
    compare_fields(cases['current']/final,cases['frozen']/final)
    first,second=(cases[name]/'outdat/render' for name in ('frozen','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert images and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    allocations=re.findall(r'stage=static-coordinate-allocation domain=sample rank=\d+ payload_bytes=(\d+)',
                           (cases['current']/'run.log').read_text())
    assert len(allocations)==2 and set(map(int,allocations))=={24*17*33*33},allocations


def test_explicit_full_diagnostics_remain_available(tmp_path):
    args=arguments(tmp_path);args.executable=CURRENT;args.statistics=False;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2
    case,_=run_case(args,ROOT,'gpu',2,'rich_audit',2,enabled=False,grid='32,32,32',
        insitu_config=settings('q_surface'),no_field_io=True,resident_audit=True,insitu_memory_trace=True)
    text=(case/'run.log').read_text()
    rows=re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=\d+ step=\d+ product=q_surface '
                    r'field_maxabs=(\S+) display_maxabs=(\S+)',text)
    assert len(rows)==4 and all(float(a)<=2e-10 and float(b)==0. for a,b in rows),text
    assert 'stage=image-only-owner ' not in text
    payloads=re.findall(r'stage=diagnostic-supply domain=sample rank=\d+ payload_bytes=(\d+)',text)
    assert len(payloads)==4 and set(map(int,payloads))=={14*8*17*33*33},payloads


def test_captured_im4_device_allocation_reduction(tmp_path):
    references=CAMPAIGN/'im4_transfer_reference/test_image_only_actual_transfe0/gpu_np2_image_trace'
    candidate=CAMPAIGN/'im4_transfer/test_image_only_actual_transfe0/gpu_np2_image_trace'
    report={}
    for label,case in [('reference',references),('candidate',candidate)]:
        report[label]={}
        for rank in range(2):
            path=case/f'trace.rank{rank}.sqlite'
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                stages={}
                for stage in ('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION','ASTR_IS8_RESIDENT_TRACE_PACK',
                              'ASTR_IS8_RESIDENT_TRACE_CONCATENATE'):
                    intervals=db.execute('select count(*) from NVTX_EVENTS where text=?',(stage,)).fetchone()[0]
                    allocations,bytes_=db.execute('select count(*),coalesce(sum(m.bytes),0) '
                        'from CUDA_GPU_MEMORY_USAGE_EVENTS m join ENUM_CUDA_MEM_KIND k on k.id=m.memKind '
                        'join ENUM_CUDA_DEV_MEM_EVENT_OPER o on o.id=m.memoryOperationType '
                        "where k.name='CUDA_MEMOPR_MEMORY_KIND_DEVICE' and o.name='CUDA_DEV_MEM_EVENT_OPR_ALLOCATION' "
                        'and m.name is null and exists(select 1 from NVTX_EVENTS n where n.text=? '
                        'and m.start>=n.start and m.start<=n.end)',(stage,)).fetchone()
                    assert intervals>0 and allocations>0,(stage,intervals,allocations)
                    stages[stage]=dict(intervals=intervals,allocations=allocations,allocated_bytes=bytes_)
                report[label][rank]=stages
    for rank in range(2):
        for stage,old in report['reference'][rank].items():
            new=report['candidate'][rank][stage]
            assert old['intervals']==new['intervals']
            assert new['allocations']<old['allocations'] and new['allocated_bytes']<old['allocated_bytes']
    (tmp_path/'allocations.json').write_text(json.dumps(report,indent=2)+'\n')


def test_curve_cached_geometry_statistics_restart(tmp_path):
    config=settings('streamlines').replace('statistics=.false.','statistics=.true.,mean_streamline_render=t').replace(
        'statistics_window=0.0005,0.0115','statistics_window=0.003,0.004')
    cases={}
    for name,executable in [('reference',CAMPAIGN/'frozen/astr_im4'),('current',CURRENT)]:
        recorded=os.environ.get('ASTR_INSITU_MEMORY_REFERENCE')
        if name=='reference' and recorded:
            cases[name]=Path(json.loads(Path(recorded).read_text())['curve_statistics_reference'])
            assert 'The job is done!' in (cases[name]/'run.log').read_text()
            assert (cases[name]/'datin/input.insitu').read_text()==config
            continue
        args=arguments(tmp_path/name);args.executable=executable;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        cases[name],_=run_case(args,ROOT,'gpu',2,name,4,grid='32,32,32',tgv_mapping='periodic',
            checkpoint_interval=1,insitu_config=config,insitu_memory_trace=name=='current',memcheck=name=='current')
    args=arguments(tmp_path/'restart');args.executable=CURRENT;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2
    cases['restart'],_=run_case(args,ROOT,'gpu',2,'restart',4,grid='32,32,32',tgv_mapping='periodic',
        checkpoint_interval=1,insitu_config=config,insitu_memory_trace=True,
        restore=cases['current']/'outdat/new/checkpoints/step000000000003')
    final='outdat/new/checkpoints/step000000000004'
    for name in ('reference','restart'):
        for field in ('state.h5','statistics.h5'):
            compare_fields(cases['current']/final/field,cases[name]/final/field)
        for p in (cases[name]/'outdat/render').glob('*.jpeg'):
            with Image.open(p) as a,Image.open(cases['current']/'outdat/render'/p.name) as b:
                np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=p.name)
    for name in ('control.bin','insitu_control.bin'):
        assert (cases['current']/final/name).read_bytes()==(cases['restart']/final/name).read_bytes()
    for name in ('current','restart'):
        log=(cases[name]/'run.log').read_text()
        assert log.count('stage=static-coordinate-allocation ')==2,log
        assert log.count('stage=static-coordinate-release ')==2,log
    reports=list(cases['current'].glob('memcheck.*.log'))
    assert len(reports)==2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)


@pytest.mark.parametrize('axis',['x','y','z'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
def test_halo_component_views_match_reference(tmp_path,axis,mode):
    config=settings('streamlines').replace("postprocess_transport='pinned'",f"postprocess_transport='{mode}'")
    cases={}
    for name,executable in [('reference',CAMPAIGN/'frozen/astr_im4'),('current',CURRENT)]:
        args=arguments(tmp_path/name,axis);args.executable=executable;args.statistics=False;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        cases[name],_=run_case(args,ROOT,'gpu',2,name,4,grid='32,32,32',checkpoint_interval=4,
            insitu_config=config,postprocess_transport=mode,insitu_memory_trace=name=='current')
    final='outdat/new/checkpoints/step000000000004/state.h5'
    compare_fields(cases['current']/final,cases['reference']/final)
    first,second=(cases[name]/'outdat/render' for name in ('reference','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==8 and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    rows=re.findall(r'stage=halo-component-view domain=geometry device=\d+ payload_bytes=(\d+)',
                    (cases['current']/'run.log').read_text())
    assert rows and not any(map(int,rows)),rows


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_resident_frame_references_retire_after_images(tmp_path,pipeline):
    config=settings('all').replace('initial_frame=f','initial_frame=t').replace(
        "rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    cases={}
    for name,executable in [('reference',CAMPAIGN/'frozen/astr_im5'),('current',CURRENT)]:
        args=arguments(tmp_path/name);args.executable=executable;args.statistics=False;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2;args.runtime_timeout_seconds=300
        run_config=config if name=='current' else config.replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),
            str(CAMPAIGN/'frozen/insitu_im5/device_render_pipeline.py'))
        cases[name],_=run_case(args,ROOT,'gpu',2,name,2,grid='32,32,32',checkpoint_interval=2,
            insitu_config=run_config,insitu_memory_trace=name=='current',memcheck=name=='current')
    final='outdat/new/checkpoints/step000000000002/state.h5'
    compare_fields(cases['current']/final,cases['reference']/final)
    first,second=(cases[name]/'outdat/render' for name in ('reference','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==12 and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    log=(cases['current']/'run.log').read_text()
    retired=re.findall(r'ASTR_INSITU_RESIDENT_RETIRE rank=\d+ step=(\d+) arrays_detached=1 owners_retained=0',log)
    assert sorted(map(int,retired))==[0,0,1,1,2,2],log
    for rank in range(2):
        groups=re.findall(rf'ASTR_INSITU_RESIDENT_GROUP rank={rank} step=(\d+) group=(\d+) complete=(\d+)',log)
        assert list(map(lambda row:tuple(map(int,row)),groups))==[
            (step,group,int(group==2)) for step in range(3) for group in (1,2)],groups
    reports=list(cases['current'].glob('memcheck.*.log'))
    assert len(reports)==2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)
    for rank in range(2):
        assert json.loads((second/f'lifecycle_rank{rank}.json').read_text())==dict(
            frames=[[0,0.],[1,.001],[2,.002]],finalized=True)


def test_m12_display_checkpoint_change_leaves_tgv_format_unchanged(tmp_path):
    reference=CAMPAIGN/'im7_groups/test_resident_frame_references0/current/gpu_np2_current'
    config=(reference/'datin/input.insitu').read_text()
    args=arguments(tmp_path);args.executable=CURRENT;args.statistics=False;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2;args.runtime_timeout_seconds=300
    case,_=run_case(args,ROOT,'gpu',2,'current',2,grid='32,32,32',checkpoint_interval=2,
        insitu_config=config)
    final='outdat/new/checkpoints/step000000000002'
    compare_fields(case/final/'state.h5',reference/final/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        actual=(case/final/name).read_bytes()
        assert actual==(reference/final/name).read_bytes(),name
        if name=='insitu_control.bin':
            assert actual[:8]==b'ASTRIR03'


def test_generated_mapper_hook_keeps_production_direct_path_unchanged(tmp_path):
    reference=CAMPAIGN/'im7_groups/test_resident_frame_references1/current/gpu_np2_current'
    config=(reference/'datin/input.insitu').read_text()
    args=arguments(tmp_path);args.executable=CURRENT;args.statistics=False;args.insitu_write_eps='0'
    args.directory_budget_bytes=256*1024**2;args.runtime_timeout_seconds=300
    case,_=run_case(args,ROOT,'gpu',2,'current',2,grid='32,32,32',checkpoint_interval=2,
        insitu_config=config,memcheck=True)
    final='outdat/new/checkpoints/step000000000002'
    compare_fields(case/final/'state.h5',reference/final/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        assert (case/final/name).read_bytes()==(reference/final/name).read_bytes(),name
    images=sorted(p.name for p in (reference/'outdat/render').glob('*.jpeg'))
    assert len(images)==12 and images==sorted(p.name for p in (case/'outdat/render').glob('*.jpeg'))
    for name in images:
        with Image.open(reference/'outdat/render'/name) as a,Image.open(case/'outdat/render'/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    checks=list(case.glob('memcheck.*.log'))
    assert len(checks)==2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in checks)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_resident_retirement_twenty_frames(tmp_path,pipeline):
    config=settings('all').replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    report={};cases={}
    for name,executable in [('reference',CAMPAIGN/'frozen/astr_im5'),('current',CURRENT)]:
        args=arguments(tmp_path/name);args.executable=executable;args.statistics=False;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2;args.runtime_timeout_seconds=300
        run_config=config if name=='current' else config.replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),
            str(CAMPAIGN/'frozen/insitu_im5/device_render_pipeline.py'))
        recorded=os.environ.get('ASTR_INSITU_MEMORY_REFERENCE')
        if name=='reference' and recorded:
            cases[name]=Path(json.loads(Path(recorded).read_text())['twenty_'+pipeline])
            assert 'The job is done!' in (cases[name]/'run.log').read_text()
            assert (cases[name]/'datin/input.insitu').read_text()==run_config
        else:
            cases[name],_=run_case(args,ROOT,'gpu',2,name,20,grid='32,32,32',checkpoint_interval=20,
                insitu_config=run_config,monitor_resources=True,insitu_memory_trace=name=='current',insitu_timing=True)
        report[name]=json.loads((cases[name]/'resources.sampled.json').read_text())
        report[name]['timing']=json.loads((cases[name]/'timing.launch.json').read_text())
    final='outdat/new/checkpoints/step000000000020/state.h5'
    compare_fields(cases['current']/final,cases['reference']/final)
    first,second=(cases[name]/'outdat/render' for name in ('reference','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==80 and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)
    for rank in range(2):
        lifecycle=json.loads((second/f'lifecycle_rank{rank}.json').read_text())
        assert len(lifecycle['frames'])==20 and lifecycle['finalized']
    (tmp_path/'memory.json').write_text(json.dumps(report,indent=2)+'\n')


def test_captured_retirement_frees_device_payload_and_preserves_clocks():
    curve=CAMPAIGN/'im7_release_fixed/test_curve_cached_geometry_sta0'
    final='outdat/new/checkpoints/step000000000004'
    for name in ('control.bin','insitu_control.bin'):
        assert (curve/'current/gpu_np2_current'/final/name).read_bytes()==(
            curve/'restart/gpu_np2_restart'/final/name).read_bytes()
    trace=CAMPAIGN/'im7_windows/test_image_only_actual_transfe0/gpu_np2_image_trace'
    for rank,expected in enumerate((322320,233760)):
        with sqlite3.connect((trace/f'trace.rank{rank}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            count,bytes_=db.execute('select count(*),coalesce(sum(m.bytes),0) '
                'from CUDA_GPU_MEMORY_USAGE_EVENTS m join ENUM_CUDA_DEV_MEM_EVENT_OPER o '
                'on o.id=m.memoryOperationType where o.name=? and exists(select 1 from NVTX_EVENTS n '
                'where n.text=? and m.start>=n.start and m.start<=n.end)',
                ('CUDA_DEV_MEM_EVENT_OPR_DEALLOCATION','ASTR_IS8_RESIDENT_RETIRE')).fetchone()
            assert count>0 and bytes_==expected,(rank,count,bytes_)


def test_retired_inputs_preserve_readonly_preview(tmp_path,monkeypatch):
    from test_insitu_preview import prepare_native_preview,source_snapshot
    monkeypatch.setenv('ASTR_INSITU_PREVIEW_SOURCE_ROOT',str(
        CAMPAIGN/'im5_actual/test_mean_products_and_readonl0/current'))
    monkeypatch.setenv('ASTR_OUTPUT_INSITU_EXE',str(CURRENT))
    monkeypatch.setenv('ASTR_OUTPUT_INSITU_BACKEND',str(LIBRARY))
    source,_,executable,launcher,env,output=prepare_native_preview(tmp_path,2,True)
    env['ASTR_INSITU_WRITE_EPS']='0'
    before=source_snapshot(source)
    with (tmp_path/'run.log').open('w') as log:
        subprocess.run([str(launcher),'-np','2',str(executable),'insitu-preview','datin/input.tgv'],
            cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    assert source_snapshot(source)==before
    log=(tmp_path/'run.log').read_text()
    assert log.count('flow_mismatches=0 statistics_mismatches=0')==2,log
    assert len(list(output.glob('*.jpeg')))==4 and not list(output.glob('*.eps'))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_product_groups_keep_missing_image_and_restart_clocks(tmp_path,pipeline):
    config=settings('all').replace('statistics=.false.','statistics=.true.').replace(
        'step_interval=1',"product_ids='q_surface.image','instantaneous_streamlines.image',"
        "product_modes='steps','steps',product_steps=2,1").replace('final_frame=t','final_frame=f').replace(
        "rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'").replace(
        str(ROOT/'scripts/insitu/device_render_pipeline.py'),
        str(ROOT/'tests/gpu_validation/insitu_resident_failure_pipeline.py'))
    cases={}
    for name,steps in [('continuous',4),('first',2),('resumed',4)]:
        args=arguments(tmp_path/name);args.executable=CURRENT
        args.insitu_write_eps='1'  # Exercise the existing recoverable EPS write fault explicitly.
        args.directory_budget_bytes=256*1024**2
        restore=None if name!='resumed' else cases['first']/'outdat/new/checkpoints/step000000000002'
        cases[name],_=run_case(args,ROOT,'gpu',2,'groups_'+name+'_fault_image',steps,grid='32,32,32',
            checkpoint_interval=2,insitu_config=config,restore=restore,insitu_memory_trace=True)
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases['continuous']/final/name,cases['resumed']/final/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (cases['continuous']/final/name).read_bytes()==(cases['resumed']/final/name).read_bytes()
    for name,case in cases.items():
        output=case/'outdat/render'
        steps=(3,4) if name=='resumed' else range(1,5 if name=='continuous' else 3)
        for rank in range(2):
            lifecycle=json.loads((output/f'lifecycle_rank{rank}.json').read_text())
            assert lifecycle['frames']==[[step,step*.001] for step in steps] and lifecycle['finalized']
            for step in steps:
                record=json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                assert set(record['products'])==({'instantaneous_streamlines','q_surface'}
                    if step%2==0 else {'instantaneous_streamlines'})
            if name!='resumed':
                receipt=json.loads((output/f'missing.q_surface.step00000002.rank{rank:08d}.json').read_text())
                assert (receipt['status'],receipt['errno'],receipt['phase'])==('missing',5,'stage_eps')
        assert not list(output.glob('*.partial'))
    for p in (cases['resumed']/'outdat/render').glob('*.jpeg'):
        with Image.open(p) as a,Image.open(cases['continuous']/'outdat/render'/p.name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=p.name)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_curve_surface_retires_before_physical_streamlines(tmp_path,pipeline):
    config=settings('curve_demo').replace("rendering_pipeline='standard-device'",f"rendering_pipeline='{pipeline}'")
    cases={}
    for name,executable in [('reference',CAMPAIGN/'frozen/astr_im7_retirement'),('current',CURRENT)]:
        args=arguments(tmp_path/name);args.executable=executable;args.statistics=False;args.insitu_write_eps='0'
        args.directory_budget_bytes=256*1024**2
        run_config=config if name=='current' else config.replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),
            str(CAMPAIGN/'frozen/insitu_im7_retirement/device_render_pipeline.py'))
        cases[name],_=run_case(args,ROOT,'gpu',2,name,2,grid='32,32,32',tgv_mapping='periodic',
            checkpoint_interval=2,insitu_config=run_config,insitu_memory_trace=name=='current')
    final='outdat/new/checkpoints/step000000000002/state.h5'
    compare_fields(cases['current']/final,cases['reference']/final)
    first,second=(cases[name]/'outdat/render' for name in ('reference','current'))
    images=sorted(p.name for p in first.glob('*.jpeg'))
    assert len(images)==4 and images==sorted(p.name for p in second.glob('*.jpeg'))
    for name in images:
        with Image.open(first/name) as a,Image.open(second/name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b),err_msg=name)


def test_captured_groups_reduce_overlapping_cuda_allocations(tmp_path):
    report={}
    for label,folder in [('reference','im7_windows'),('grouped','im7_group_windows')]:
        peaks=[]
        for rank in range(2):
            path=CAMPAIGN/folder/'test_image_only_actual_transfe0/gpu_np2_image_trace'/f'trace.rank{rank}.sqlite'
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                rows=db.execute('select m.address,m.bytes,m.memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS m '
                    'join ENUM_CUDA_MEM_KIND k on k.id=m.memKind where k.name=? and m.name is null order by m.start',
                    ('CUDA_MEMOPR_MEMORY_KIND_DEVICE',)).fetchall()
                live={};current=peak=0
                for address,bytes_,operation in rows:
                    if operation==0:
                        assert address not in live
                        live[address]=bytes_;current+=bytes_;peak=max(peak,current)
                    else:
                        assert operation==1 and address in live
                        current-=live.pop(address)
                assert not live and current==0
                peaks.append(peak)
                if label=='grouped':
                    ranges={name:db.execute('select start,end from NVTX_EVENTS where text=? order by start',
                        (name,)).fetchall() for name in ('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION',
                        'ASTR_IS8_RESIDENT_RETIRE','ASTR_IS8_RESIDENT_STREAMLINES')}
                    assert len(ranges['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION'])==2
                    for _,end in ranges['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION']:
                        retired=next(interval for interval in ranges['ASTR_IS8_RESIDENT_RETIRE'] if interval[0]>=end)
                        traced=next(interval for interval in ranges['ASTR_IS8_RESIDENT_STREAMLINES'] if interval[0]>=end)
                        assert retired[1]<=traced[0]
        report[label]=peaks
    assert all(a-b==110480 for a,b in zip(report['reference'],report['grouped'])),report
    (tmp_path/'cuda_allocation_peaks.json').write_text(json.dumps(report,indent=2)+'\n')
