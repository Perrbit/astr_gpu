"""PF real-product clocks, bounded transfers, and exact native continuation."""
import json
import os
from pathlib import Path
import re
import struct
from types import SimpleNamespace

from PIL import Image
import numpy as np
import pytest

from test_output_insitu_restart import ROOT, arguments, configuration
from test_insitu_device_products import device_configuration, resident_configuration, field_difference, cross_backend_statistics_difference
from run_output_restart_validation import run_case

pytestmark = pytest.mark.skipif(
    not os.environ.get('ASTR_OUTPUT_INSITU_EXE') or not os.environ.get('ASTR_OUTPUT_INSITU_BACKEND'),
    reason='Select the resident-rendering executable and Catalyst implementation explicitly')


def config(pipeline='compatible', transport='pinned', statistics=False, host=False):
    base = configuration(statistics=statistics) if host else (
        device_configuration(transport, statistics=statistics) if pipeline == 'compatible' else
        resident_configuration(pipeline, transport, statistics=statistics))
    clocks = """product_ids='q_surface.image','instantaneous_streamlines.image',
 product_modes='steps','time',product_steps=6,0,product_times=0,0.008"""
    if pipeline == 'compatible':
        clocks += ",product_ids(3)='velocity_slice.geometry',product_modes(3)='steps',product_steps(3)=6"
    return re.sub(r'step_interval=\d+', clocks, base).replace('final_frame=t', 'final_frame=f')


def params(path):
    args = arguments(path)
    args.directory_budget_bytes = 64 * 1024**2
    args.runtime_timeout_seconds = 300
    return args


def receipts(case, ranks, expected):
    output = case / 'outdat/render'
    for rank in range(ranks):
        records = {int(p.stem.split('_')[1][4:]): json.loads(p.read_text())
                   for p in output.glob(f'mesh_step*_rank{rank}.json')}
        assert set(records) == set(expected)
        for step, names in expected.items():
            assert set(records[step]['products']) == set(names)
    for path in output.glob('*.jpeg'):
        assert path.with_suffix('.eps').is_file()
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        assert pixels.shape == (600, 800, 3) and np.any(pixels < 245)
    return output


@pytest.mark.parametrize('pipeline', ['compatible', 'standard-device', 'direct-device'])
@pytest.mark.parametrize('transport', ['pinned', 'device-aware'])
@pytest.mark.parametrize('ranks', [1, 2])
def test_fixed_product_routing(tmp_path, pipeline, transport, ranks, record_property):
    args = params(tmp_path)
    case, size = run_case(args, ROOT, 'gpu', ranks, 'pf_routing', 12,
        grid='32,32,32', insitu_config=config(pipeline, transport),
        checkpoint_keep=1, checkpoint_interval=100, postprocess_transport=transport, insitu_timing=True)
    expected = {6: ['q_surface'], 8: ['instantaneous_streamlines'], 12: ['q_surface']}
    if pipeline == 'compatible':
        for step in (6, 12):
            expected[step].append('velocity_slice')
    output = receipts(case, ranks, expected)
    assert len(list(output.glob('*.jpeg'))) == 3
    assert not list(output.glob('velocity_slice*.jpeg'))
    assert not list(output.glob('q_surface*.pvtp'))
    assert len(list(output.glob('*.pvtp'))) == (2 if pipeline == 'compatible' else 0)
    log = (case / 'run.log').read_text()
    assert log.count('device_gradient_q_inclusive') == ranks * 2
    assert 'device_reynolds_lines_inclusive' not in log and 'device_favre_lines_inclusive' not in log
    assert 'device_crossing_lines_inclusive' not in log
    control = case / 'outdat/new/checkpoints/step000000000012/insitu_control.bin'
    assert control.read_bytes()[:8] == b'ASTRIR04'
    record_property('directory_bytes', size)


@pytest.mark.parametrize('pipeline', ['compatible', 'standard-device', 'direct-device'])
@pytest.mark.parametrize('ranks', [1, 2])
def test_exact_product_restart_and_output_isolation(tmp_path, pipeline, ranks, record_property):
    args = params(tmp_path)
    settings = config(pipeline, statistics=True)
    continuous, a = run_case(args, ROOT, 'gpu', ranks, 'pf_continuous', 12,
        grid='32,32,32', insitu_config=settings, checkpoint_keep=1, checkpoint_interval=100,
        postprocess_transport='pinned')
    first, b = run_case(args, ROOT, 'gpu', ranks, 'pf_first', 5,
        grid='32,32,32', insitu_config=settings, checkpoint_keep=1, checkpoint_interval=100,
        postprocess_transport='pinned')
    source = first / 'outdat/new/checkpoints/step000000000005'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, c = run_case(args, ROOT, 'gpu', ranks, 'pf_resumed', 12,
        grid='32,32,32', insitu_config=settings, restore=source, checkpoint_keep=1,
        checkpoint_interval=100, postprocess_transport='pinned')
    plain, d = run_case(args, ROOT, 'gpu', ranks, 'pf_disabled', 12,
        grid='32,32,32', insitu_config=configuration(render=False, statistics=True),
        checkpoint_keep=1, checkpoint_interval=100)
    final = 'outdat/new/checkpoints/step000000000012'
    for name in ('state.h5', 'statistics.h5'):
        field_difference(continuous / final / name, resumed / final / name)
        field_difference(continuous / final / name, plain / final / name)
    assert (continuous / final / 'insitu_control.bin').read_bytes() == (resumed / final / 'insitu_control.bin').read_bytes()
    for path in (continuous / 'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg', '.eps', '.vtp', '.pvtp'):
            relative=path.relative_to(continuous / 'outdat/render')
            assert path.read_bytes() == (resumed / 'outdat/render' / relative).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    record_property('directory_bytes', [a, b, c, d])


@pytest.mark.parametrize('ranks',[1,2])
def test_native_16_flow_cache_statistics_exact_continuation(tmp_path,ranks,record_property):
    import h5py
    args=params(tmp_path)
    final='outdat/new/checkpoints/step000000000012'
    cases={}
    sizes=[]
    for backend in ('cpu','gpu'):
        continuous,a=run_case(args,ROOT,backend,ranks,'pf16_continuous',12,checkpoint_keep=1,
            checkpoint_interval=100,insitu_config=configuration(render=False,statistics=True))
        first,b=run_case(args,ROOT,backend,ranks,'pf16_first',5,checkpoint_keep=1,
            checkpoint_interval=100,insitu_config=configuration(render=False,statistics=True))
        resumed,c=run_case(args,ROOT,backend,ranks,'pf16_resumed',12,checkpoint_keep=1,
            checkpoint_interval=100,insitu_config=configuration(render=False,statistics=True),
            restore=first/'outdat/new/checkpoints/step000000000005')
        plain_args=SimpleNamespace(**vars(args)); plain_args.statistics=False
        plain,d=run_case(plain_args,ROOT,backend,ranks,'pf16_no_statistics',12,checkpoint_keep=1,
            checkpoint_interval=100,insitu_config=None)
        for name in ('state.h5','statistics.h5'):
            field_difference(continuous/final/name,resumed/final/name)
        field_difference(continuous/final/'state.h5',plain/final/'state.h5')
        cases[backend]=continuous/final
        sizes.extend((a,b,c,d))
    differences={}
    with h5py.File(cases['cpu']/'state.h5') as cpu,h5py.File(cases['gpu']/'state.h5') as gpu:
        for name in [f'q{c:04d}' for c in range(1,12)]+['rank_extras']:
            a,b=cpu[name][...],gpu[name][...]
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            differences[name]=float(np.max(abs(a-b))) if a.size else 0.
            assert differences[name]<=2e-10,(name,differences[name])
    differences['statistics']=cross_backend_statistics_difference(cases['gpu']/'statistics.h5',
        cases['cpu']/'statistics.h5',cells=16)
    record_property('maxabs',differences)
    record_property('directory_bytes',sizes)


@pytest.mark.parametrize('transport',['pinned','device-aware'])
def test_transport_override_preserves_all_product_clocks(tmp_path,transport):
    args=params(tmp_path)
    initial='device-aware' if transport=='pinned' else 'pinned'
    settings=config('standard-device',initial,statistics=True)
    continuous,_=run_case(args,ROOT,'gpu',2,'pf_transport_continuous',12,grid='32,32,32',
        insitu_config=settings,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport=initial)
    first,_=run_case(args,ROOT,'gpu',2,'pf_transport_first',5,grid='32,32,32',
        insitu_config=settings,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport=initial)
    source=first/'outdat/new/checkpoints/step000000000005'
    changed=config('standard-device',transport,statistics=True)
    run_case(args,ROOT,'gpu',2,'pf_reject_transport',12,grid='32,32,32',restore=source,
        insitu_config=changed,postprocess_transport=transport,checkpoint_interval=100,
        reject='native render configuration differs; select explicit override')
    resumed,_=run_case(args,ROOT,'gpu',2,'pf_override_transport',12,grid='32,32,32',restore=source,
        insitu_config=changed,postprocess_transport=transport,checkpoint_interval=100,checkpoint_keep=1,override=True)
    receipts(resumed,2,{6:['q_surface'],8:['instantaneous_streamlines'],12:['q_surface']})
    final='outdat/new/checkpoints/step000000000012'
    assert (continuous/final/'insitu_control.bin').read_bytes()[192:]==(resumed/final/'insitu_control.bin').read_bytes()[192:]
    field_difference(continuous/final/'state.h5',resumed/final/'state.h5')
    field_difference(continuous/final/'statistics.h5',resumed/final/'statistics.h5')


def test_clock_change_requires_override_and_reanchors_explicitly(tmp_path):
    args=params(tmp_path)
    original=config('standard-device',statistics=True)
    first,_=run_case(args,ROOT,'gpu',2,'pf_change_first',5,grid='32,32,32',
        insitu_config=original,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport='pinned')
    source=first/'outdat/new/checkpoints/step000000000005'
    changed=original.replace('product_steps=6,0','product_steps=4,0')
    run_case(args,ROOT,'gpu',2,'pf_reject_clock',12,grid='32,32,32',restore=source,
        insitu_config=changed,checkpoint_interval=100,postprocess_transport='pinned',
        reject='native render configuration differs; select explicit override')
    resumed,_=run_case(args,ROOT,'gpu',2,'pf_override_clock',12,grid='32,32,32',restore=source,
        insitu_config=changed,checkpoint_interval=100,checkpoint_keep=1,postprocess_transport='pinned',override=True)
    receipts(resumed,2,{9:['q_surface']})


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_missing_image_pair_history_survives_restart(tmp_path,pipeline):
    args=params(tmp_path)
    settings=resident_configuration(pipeline,statistics=True,profile='q_surface').replace(
        'step_interval=1',"product_ids='q_surface.image',product_modes='steps',product_steps=2").replace(
        'final_frame=t','final_frame=f').replace(
        str(ROOT/'scripts/insitu/device_render_pipeline.py'),
        str(ROOT/'tests/gpu_validation/insitu_resident_failure_pipeline.py'))
    continuous,_=run_case(args,ROOT,'gpu',2,'pf_continuous_fault_image',12,grid='32,32,32',
        insitu_config=settings,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport='pinned')
    first,_=run_case(args,ROOT,'gpu',2,'pf_first_fault_image',5,grid='32,32,32',
        insitu_config=settings,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport='pinned')
    source=first/'outdat/new/checkpoints/step000000000005'
    resumed,_=run_case(args,ROOT,'gpu',2,'pf_resumed_fault_image',12,grid='32,32,32',restore=source,
        insitu_config=settings,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport='pinned')
    final='outdat/new/checkpoints/step000000000012/insitu_control.bin'
    payload=(continuous/final).read_bytes()
    assert payload==(resumed/final).read_bytes()
    assert struct.unpack_from('<7q',payload,224+96)==(12,12,2,0,0,5,2)
    for case in (continuous,first):
        output=case/'outdat/render'
        assert not list(output.glob('*step00000002.jpeg')) and not list(output.glob('*step00000002.eps'))
        assert not list(output.glob('*.partial'))
        for rank in range(2):
            receipt=json.loads((output/f'missing.q_surface.step00000002.rank{rank:08d}.json').read_text())
            assert receipt['errno']==5 and receipt['phase']=='stage_eps'


@pytest.mark.parametrize('pipeline',['compatible','standard-device','direct-device'])
def test_independent_reynolds_and_favre_supply_without_q(tmp_path,pipeline):
    args=params(tmp_path)
    base=device_configuration('pinned',statistics=True) if pipeline=='compatible' else resident_configuration(pipeline)
    clocks="product_ids='mean_reynolds_streamlines.image','mean_favre_streamlines.image'," \
        "product_modes='steps','time',product_steps=6,0,product_times=0,0.008"
    settings=base.replace('step_interval=1',clocks).replace('final_frame=t','final_frame=f').replace(
        'initial_frame=f','initial_frame=t')
    case,_=run_case(args,ROOT,'gpu',2,'pf_means',8,grid='32,32,32',insitu_config=settings,
        checkpoint_keep=1,checkpoint_interval=100,postprocess_transport='pinned',insitu_timing=True)
    output=receipts(case,2,{0:[],6:['mean_reynolds_streamlines'],8:['mean_favre_streamlines']})
    assert len(list(output.glob('*.jpeg')))==2 and not list(output.glob('*.pvtp'))
    log=(case/'run.log').read_text()
    assert 'device_gradient_q_inclusive' not in log and 'device_instant_lines_inclusive' not in log
    assert 'device_crossing_lines_inclusive' not in log


def test_compatible_geometry_is_readable_and_cpu_gpu_fields_agree(tmp_path,record_property):
    from test_insitu_device_products import check_geometry_fields
    args=params(tmp_path)
    clocks="product_ids='q_surface.geometry','velocity_slice.geometry','crossing_streamlines.geometry'," \
        "product_modes='steps','steps','steps',product_steps=4,4,4"
    host=configuration(statistics=False).replace('step_interval=8',clocks).replace('final_frame=t','final_frame=f')
    device=device_configuration('pinned',statistics=False,interval=8).replace('step_interval=8',clocks).replace(
        'final_frame=t','final_frame=f')
    cases=[]; errors={}
    for name,settings in [('cpu_diagnostics',host),('device',device)]:
        case,_=run_case(args,ROOT,'gpu',2,'pf_geometry_'+name,4,grid='32,32,32',
            insitu_config=settings,checkpoint_keep=1,checkpoint_interval=100,
            postprocess_transport='pinned' if name=='device' else None)
        cases.append(case/'outdat/render')
        assert not list(cases[-1].glob('*.jpeg')) and not list(cases[-1].glob('*.eps'))
        receipts(case,2,{4:['q_surface','velocity_slice','crossing_streamlines']})
        assert json.loads((cases[-1]/'mesh_step00000004_rank0.json').read_text())['crossing_maxabs']<=2e-10
        errors[name]=check_geometry_fields(case,4,statistics=False,
            products=['q_surface','velocity_slice','crossing_streamlines'],device_products=name=='device')
    record_property('geometry_field_atol',2e-10)
    record_property('field_interpolation_maxabs',errors)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('transport',['pinned','device-aware'])
def test_independent_products_memcheck(tmp_path,pipeline,transport,record_property):
    args=params(tmp_path)
    settings=resident_configuration(pipeline,transport,statistics=True).replace('step_interval=1',
        "product_ids='q_surface.image','mean_reynolds_streamlines.image','mean_favre_streamlines.image',"
        "product_modes='steps','steps','steps',product_steps=6,2,3").replace('final_frame=t','final_frame=f')
    case,size=run_case(args,ROOT,'gpu',2,'pf_memcheck',6,grid='32,32,32',insitu_config=settings,
        checkpoint_keep=1,checkpoint_interval=100,postprocess_transport=transport,memcheck=True)
    reports=list(case.glob('memcheck.*.log'))
    assert len(reports)==2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)
    record_property('directory_bytes',size)


def test_explicit_disable_restore_preserves_flow_and_statistics(tmp_path):
    args=params(tmp_path)
    original=config('standard-device',statistics=True)
    first,_=run_case(args,ROOT,'gpu',2,'pf_disable_first',5,grid='32,32,32',
        insitu_config=original,checkpoint_keep=1,checkpoint_interval=100,postprocess_transport='pinned')
    source=first/'outdat/new/checkpoints/step000000000005'
    changed=configuration(render=False,statistics=True)
    plain,_=run_case(args,ROOT,'gpu',2,'pf_disable_plain',12,grid='32,32,32',
        insitu_config=changed,checkpoint_keep=1,checkpoint_interval=100)
    resumed,_=run_case(args,ROOT,'gpu',2,'pf_disable_resumed',12,grid='32,32,32',restore=source,
        insitu_config=changed,checkpoint_interval=100,checkpoint_keep=1,override=True)
    final='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'):
        field_difference(plain/final/name,resumed/final/name)
    assert not list((resumed/'outdat/render').glob('*.jpeg'))


@pytest.mark.parametrize('pipeline,transport', [
    ('standard-device', 'pinned'), ('direct-device', 'device-aware')])
def test_due_only_resident_transfer_attribution(tmp_path, pipeline, transport, record_property):
    import sqlite3
    from test_insitu_device_native_trace import copies

    args = params(tmp_path)
    settings = resident_configuration(pipeline, transport, statistics=True).replace('step_interval=1',
        "product_ids='q_surface.image','mean_reynolds_streamlines.image','mean_favre_streamlines.image',"
        "product_modes='steps','steps','steps',product_steps=2,1,3").replace('final_frame=t', 'final_frame=f')
    case, size = run_case(args, ROOT, 'gpu', 2, 'pf_trace', 2, enabled=False,
        grid='32,32,32', insitu_config=settings, postprocess_transport=transport,
        no_field_io=True, insitu_timing=True, nsys_trace=True, pixel_audit=True)
    receipts(case, 2, {1: ['mean_reynolds_streamlines'], 2: ['q_surface', 'mean_reynolds_streamlines']})
    assert not list((case/'outdat/render').rglob('*.vtp'))
    ledger = []
    for path in sorted(case.glob('trace.rank*.sqlite')):
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            counts = dict(db.execute("select text,count(*) from NVTX_EVENTS "
                "where text like 'ASTR_IS8_%' group by text"))
            assert counts['ASTR_IS8_DEVICE_SAMPLE'] == counts['ASTR_IS8_DEVICE_MEAN_SUPPLY'] == 2
            assert counts['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION'] == 1
            assert counts['ASTR_IS8_RESIDENT_STREAMLINES'] == counts['ASTR_IS8_RESIDENT_RENDER'] == 2
            assert 'ASTR_IS8_COMPACT_GEOMETRY_READ' not in counts and 'ASTR_IS8_COMPACT_TRACE_READ' not in counts
            assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
            render = copies(db, 'ASTR_IS8_RESIDENT_RENDER')
            assert render and all(kind == 'CUDA_MEMCPY_KIND_DTOD' for kind, _ in render)
            faces = copies(db, 'ASTR_IS8_DEVICE_SAMPLE') + copies(db, 'ASTR_IS8_DEVICE_MEAN_SUPPLY')
            # A Reynolds-only clock communicates three mean components, not both mean fields.
            sizes = sorted([34848]*2 + [26136]*2 + [109512]*8)
            if transport == 'pinned':
                for kind in ('CUDA_MEMCPY_KIND_DTOH', 'CUDA_MEMCPY_KIND_HTOD'):
                    assert sorted(n for k, n in faces if k == kind and n > 144) == sizes
                assert not any(kind == 'CUDA_MEMCPY_KIND_PTOP' for kind, _ in faces)
            else:
                assert all(n <= 8 for kind, n in faces if kind == 'CUDA_MEMCPY_KIND_DTOH')
                assert all(n <= 144 for kind, n in faces if kind == 'CUDA_MEMCPY_KIND_HTOD')
                assert sorted(n for kind, n in faces if kind == 'CUDA_MEMCPY_KIND_PTOP') == sizes
            ledger.append(dict(trace=path.name, face_bytes_per_direction=sum(sizes),
                geometry_render_d2h_bytes=0, display_dtod_bytes=sum(n for _, n in render)))
    assert len(ledger) == 2
    record_property('transfer_ledger', ledger)
    record_property('directory_bytes', size)


def test_host_mean_fields_are_supplied_only_when_due(tmp_path, record_property):
    args = params(tmp_path)
    clocks = "product_ids='mean_reynolds_streamlines.image','mean_favre_streamlines.image'," \
        "product_modes='steps','steps',product_steps=2,3"
    settings = configuration(statistics=True).replace('step_interval=8', clocks).replace('final_frame=t', 'final_frame=f')
    case, size = run_case(args, ROOT, 'gpu', 2, 'pf_host_means', 6, grid='32,32,32',
        insitu_config=settings, checkpoint_keep=1, checkpoint_interval=100, insitu_timing=True)
    output = receipts(case, 2, {2: ['mean_reynolds_streamlines'], 3: ['mean_favre_streamlines'],
        4: ['mean_reynolds_streamlines'], 6: ['mean_reynolds_streamlines', 'mean_favre_streamlines']})
    for rank in range(2):
        for step in (2, 3, 4, 6):
            record = json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            fields = set(record['available_fields'])
            expected = {'u', 'v', 'w', 'statistics_duration'}
            for kind in ('reynolds', 'favre'):
                if 'mean_'+kind+'_streamlines' in record['products']:
                    expected.update('mean_'+c+'_'+kind for c in ('u', 'v', 'w'))
            assert fields == expected
    record_property('directory_bytes', size)
