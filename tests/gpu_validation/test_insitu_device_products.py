"""Actual completed-step device products, without a host-volume oracle."""
import json
import re
import os
from pathlib import Path
from PIL import Image

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_output_insitu_restart import ROOT, arguments, configuration, check_render
from test_checkpoint_bundle import fault_library


def device_configuration(mode, statistics=True, interval=1, profile='all'):
    return configuration(statistics=statistics, interval=interval).replace('render=.true.,',
        f"render=.true.,derivative_backend='gpu',processing_backend='device',"
        f"postprocess_transport='{mode}',products='{profile}',rendering_pipeline='compatible',")


def resident_configuration(pipeline, mode='pinned', statistics=True, interval=1, profile='all'):
    return device_configuration(mode, statistics, interval, profile).replace(
        "rendering_pipeline='compatible'", f"rendering_pipeline='{pipeline}'").replace(
        'scripts/insitu/tgv_pipeline.py','scripts/insitu/device_render_pipeline.py')


def field_difference(first, second, atol=0.):
    worst=0.
    with h5py.File(first) as actual, h5py.File(second) as expected:
        names=[]
        actual.visititems(lambda name,obj: names.append(name) if isinstance(obj,h5py.Dataset) else None)
        other=[]
        expected.visititems(lambda name,obj: other.append(name) if isinstance(obj,h5py.Dataset) else None)
        assert names==other
        for name in names:
            a,b=actual[name][...],expected[name][...]
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name
            if a.dtype.kind in 'fc':
                error=float(np.max(np.abs(a-b))) if a.size else 0.
                assert error<=atol,(name,error,atol)
                worst=max(worst,error)
            else:
                np.testing.assert_array_equal(a,b,err_msg=name)
    return worst


def cross_backend_statistics_difference(gpu,cpu,cells=32):
    """Compare unique periodic nodes, not CPU seam copies or storage-role bits."""
    global_cells=cells
    worst=0.
    with h5py.File(gpu) as actual,h5py.File(cpu) as expected:
        a,b=actual['identity'][...],expected['identity'][...]
        np.testing.assert_array_equal(a[:-1],b[:-1])
        assert a[-1]==4 and b[-1]==3
        a,b=actual['metadata'][...],expected['metadata'][...]
        np.testing.assert_array_equal(a[[0,2,3]],b[[0,2,3]])
        assert a[1]==1 and b[1]==0
        error=float(np.max(abs(a[4:].view(np.float64)-b[4:].view(np.float64))))
        assert np.isfinite(error) and error<=2e-10,error
        worst=max(worst,error)
        np.testing.assert_array_equal(actual['partitions'][...],expected['partitions'][...])
        for component in range(1,35):
            name=f'q{component:04d}'
            a,b=actual[name][:global_cells,:global_cells,:global_cells],expected[name][:global_cells,:global_cells,:global_cells]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            error=float(np.max(abs(a-b)))
            assert error<=2e-10,(name,error)
            worst=max(worst,error)
        extras=actual['rank_extras'][...]
        assert np.isfinite(extras).all() and not np.count_nonzero(extras)
        seams=[]
        for partition in expected['partitions'][...].reshape(-1,8):
            origin,cells=partition[:3],partition[3:6]
            assert partition[6]==0
            full=cells+1
            owned=cells+(origin+cells==global_cells)
            mask=np.ones(tuple(full[::-1]),dtype=bool)
            mask[:owned[2],:owned[1],:owned[0]]=False
            selection=tuple(slice(int(o),int(o+n)) for o,n in zip(origin[::-1],full[::-1]))
            for component in range(1,35):
                seams.extend(expected[f'q{component:04d}'][selection][mask])
        stored=expected['rank_extras'][...]
        assert stored.size==len(seams)==extras.size and np.isfinite(stored).all()
        if stored.size:
            error=float(np.max(abs(stored-np.asarray(seams))))
            assert error<=2e-10,error
    return worst


@pytest.fixture(scope='module')
def resident_field_references(tmp_path_factory):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict device renderer candidate')
    references={}
    for ranks,axis in ((1,'x'),(2,'x'),(2,'y'),(2,'z')):
        args=arguments(tmp_path_factory.mktemp(f'resident_reference_{ranks}_{axis}'),axis)
        args.directory_budget_bytes=256*1024**2
        config=configuration(render=False,interval=1)
        cases={}
        for backend in ('cpu','gpu'):
            cases[backend],_=run_case(args,ROOT,backend,ranks,'reference',2,
                grid='32,32,32',insitu_config=config,checkpoint_interval=2)
        final='outdat/new/checkpoints/step000000000002/'
        errors={'state.h5':field_difference(cases['gpu']/final/'state.h5',
                    cases['cpu']/final/'state.h5',2e-10),
                'statistics.h5':cross_backend_statistics_difference(cases['gpu']/final/'statistics.h5',
                    cases['cpu']/final/'statistics.h5')}
        references[ranks,axis]=(cases['gpu'],errors)
    return references


@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('ranks,axis',[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def test_resident_numerical_and_state_gate(tmp_path,mode,pipeline,ranks,axis,
        resident_field_references,record_property):
    args=arguments(tmp_path,axis)
    args.directory_budget_bytes=256*1024**2
    args.runtime_timeout_seconds=300
    case,size=run_case(args,ROOT,'gpu',ranks,'resident_gate',2,grid='32,32,32',
        checkpoint_interval=2,insitu_config=resident_configuration(pipeline,mode),
        postprocess_transport=mode,resident_audit=True)
    log=(case/'run.log').read_text()
    audits=re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=(\d+) step=(\d+) product=(\S+) '
        r'field_maxabs=(\S+) display_maxabs=(\S+) points=(\d+)',log)
    assert len(audits)==ranks*2*6,audits
    assert len({(rank,step,name) for rank,step,name,*_ in audits})==len(audits)
    assert all(np.isfinite(float(field)) and float(field)<=2e-10 and float(display)==0.
               for _,_,_,field,display,_ in audits)
    endpoints=re.findall(r'ASTR_INSITU_RESIDENT_CROSSING rank=\d+ endpoint_maxabs=(\S+)',log)
    assert len(endpoints)==ranks*2 and all(float(e)<=2e-10 for e in endpoints)
    reference,errors=resident_field_references[ranks,axis]
    final='outdat/new/checkpoints/step000000000002/'
    for name in ('state.h5','statistics.h5'):
        field_difference(case/final/name,reference/final/name)
    assert not list((case/'outdat/render').rglob('*.vtp'))
    assert not list((case/'outdat/render').glob('sample.*.bin'))
    flows=re.findall(r'ASTR_INSITU_GPU_FLOW rank=\d+ frame_downloads=(\d+)',log)
    exports=re.findall(r'ASTR_INSITU_GPU_STATS rank=\d+ samples=\d+ full_output_downloads=(\d+)',log)
    assert len(flows)==len(exports)==ranks and all(int(v)==0 for v in flows+exports)
    record_property('cpu_gpu_authoritative_field_maxabs',errors)
    record_property('gpu_product_field_maxabs',max(float(row[3]) for row in audits))
    record_property('gpu_display_cast_maxabs',max(float(row[4]) for row in audits))
    record_property('directory_bytes',size)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('ranks,axis',[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def test_resident_exact_continuation(tmp_path,pipeline,mode,ranks,axis,record_property,fault_library,
                                     default_route=False):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict device renderer candidate')
    args=arguments(tmp_path,axis)
    args.directory_budget_bytes=256*1024**2
    args.runtime_timeout_seconds=300
    config=resident_configuration(pipeline,mode,interval=8)
    if default_route:
        config=config.replace(f"rendering_pipeline='{pipeline}',",'')
    continuous,size=run_case(args,ROOT,'gpu',ranks,'resident_continuous',12,
        grid='32,32,32',checkpoint_interval=5,insitu_config=config,
        postprocess_transport=mode,publication_fault=(fault_library,'protect_batch',5,0))
    source=continuous/'outdat/new/checkpoints/step000000000005'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert before['insitu_control.bin'][176:192].decode().strip()==pipeline
    resumed,resumed_size=run_case(args,ROOT,'gpu',ranks,'resident_resumed',12,
        grid='32,32,32',restore=source,checkpoint_interval=5,
        insitu_config=resident_configuration(pipeline,mode,interval=8) if default_route else config,
        postprocess_transport=mode)
    final='outdat/new/checkpoints/step000000000012/'
    for name in ('state.h5','statistics.h5'):
        field_difference(continuous/final/name,resumed/final/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (continuous/final/name).read_bytes()==(resumed/final/name).read_bytes(),name
    from run_output_restart_validation import archive_schedule_payload
    assert archive_schedule_payload(continuous/final/'archives.bin')==archive_schedule_payload(
        resumed/final/'archives.bin')
    for case in (continuous,resumed):
        output=case/'outdat/render'
        assert not list(output.rglob('*.vtp')) and not list(output.glob('sample.*.bin'))
        for rank in range(ranks):
            lifecycle=json.loads((output/f'lifecycle_rank{rank}.json').read_text())
            assert lifecycle['finalized'] and [row[0] for row in lifecycle['frames']]==[8,12]
            for step in (8,12):
                actual=json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                expected=json.loads((continuous/'outdat/render'/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                assert actual==expected
    pictures=sorted((resumed/'outdat/render').glob('*.jpeg'))
    assert len(pictures)==12
    for path in pictures:
        with Image.open(path) as actual,Image.open(continuous/'outdat/render'/path.name) as expected:
            np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
        assert path.with_suffix('.eps').read_bytes()==(
            continuous/'outdat/render'/path.with_suffix('.eps').name).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    record_property('directory_bytes',[size,resumed_size])


def test_resident_default_exact_continuation(tmp_path,record_property,fault_library):
    test_resident_exact_continuation(tmp_path,'standard-device','device-aware',2,'x',
        record_property,fault_library,default_route=True)


def test_resident_partition_image_seams(record_property):
    root=os.environ.get('ASTR_INSITU_RESIDENT_IMAGE_ROOT')
    if not root:
        pytest.skip('Select the immutable completed sixteen-case numerical matrix')
    from scipy.ndimage import distance_transform_edt
    cases=sorted(p for p in Path(root).glob('test_*/gpu_*_resident_gate') if not p.parent.is_symlink())
    assert len(cases)==16
    maxima={}
    for name in ('q_surface','velocity_slice','instantaneous_streamlines','crossing_streamlines',
                 'mean_reynolds_streamlines','mean_favre_streamlines'):
        worst=0.
        for step in (1,2):
            masks=[]
            for case in cases:
                with Image.open(case/'outdat/render'/f'{name}.step{step:08d}.jpeg') as image:
                    pixels=np.asarray(image.convert('RGB')).astype(int)[50:550,50:650]
                mask=pixels.max(2)-pixels.min(2)>40
                assert np.count_nonzero(mask)>100,(case,name)
                masks.append(mask)
            distance=distance_transform_edt(~masks[0])
            for case,mask in zip(cases,masks):
                error=max(float(distance[mask].max()),float(distance_transform_edt(~mask)[masks[0]].max()))
                assert error<=1.,(case,name,step,error)
                worst=max(worst,error)
        maxima[name]=worst
    record_property('partition_seam_max_pixels',maxima)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('ranks,axis',[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def test_resident_memory_and_lifecycle(tmp_path,pipeline,mode,ranks,axis,record_property):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict device renderer candidate')
    args=arguments(tmp_path,axis)
    args.directory_budget_bytes=256*1024**2
    args.runtime_timeout_seconds=300
    config=resident_configuration(pipeline,mode).replace('initial_frame=f','initial_frame=t')
    case,size=run_case(args,ROOT,'gpu',ranks,'resident_memcheck',2,enabled=False,
        grid='32,32,32',insitu_config=config,postprocess_transport=mode,no_field_io=True,memcheck=True)
    reports=list(case.glob('memcheck.*.log'))
    assert len(reports)==ranks and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)
    output=case/'outdat/render'
    for rank in range(ranks):
        lifecycle=json.loads((output/f'lifecycle_rank{rank}.json').read_text())
        assert lifecycle['finalized'] and [row[0] for row in lifecycle['frames']]==[0,1,2]
        assert len(json.loads((output/f'mesh_step00000000_rank{rank}.json').read_text())['products'])==4
        for step in (1,2):
            assert len(json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())['products'])==6
    assert not list(output.rglob('*.vtp')) and not list(output.glob('sample.*.bin'))
    record_property('directory_bytes',size)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
def test_resident_full_product_trace(tmp_path,pipeline,mode):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict device renderer candidate')
    args=arguments(tmp_path,'x')
    args.directory_budget_bytes=256*1024**2
    args.runtime_timeout_seconds=300
    args.nsys_trace_domains='cuda,nvtx,mpi'
    case,_=run_case(args,ROOT,'gpu',2,'resident_trace',2,enabled=False,
        grid='32,32,32',insitu_config=resident_configuration(pipeline,mode),
        postprocess_transport=mode,no_field_io=True,insitu_timing=True,nsys_trace=True,pixel_audit=True)
    assert len(list(case.glob('trace.rank*.sqlite')))==2


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
def test_resident_observed_resources(tmp_path,pipeline,mode,record_property):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict device renderer candidate')
    args=arguments(tmp_path,'y')
    args.statistics=False
    args.directory_budget_bytes=256*1024**2
    plain,_=run_case(args,ROOT,'gpu',2,'resident_resource_off',2,enabled=False,
        grid='32,32,32',insitu_config=configuration(render=False,statistics=False).replace('enabled=t','enabled=f'),
        no_field_io=True,monitor_resources=True)
    baseline=json.loads((plain/'resources.sampled.json').read_text())
    args.statistics=True
    case,size=run_case(args,ROOT,'gpu',2,'resident_resource_on',2,enabled=False,
        grid='32,32,32',insitu_config=resident_configuration(pipeline,mode),
        postprocess_transport=mode,no_field_io=True,monitor_resources=True,resource_baseline=baseline)
    report=json.loads((case/'resources.sampled.json').read_text())
    assert report['samples']>0 and report['sampling_period_seconds']==.02
    assert report['additional_host_peak_difference_bytes']<=4*1024**3
    assert all(value<=2*1024**3 for value in report['additional_device_peak_difference_bytes'].values())
    assert all(device['min_free_bytes']>=1024**3 for device in report['devices'].values())
    record_property('sampled_resources',report)
    record_property('directory_bytes',size)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('kind',['host','device'])
def test_resident_budget_rejection(tmp_path,pipeline,mode,kind):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE')!='1':
        pytest.skip('Select the strict device renderer candidate')
    args=arguments(tmp_path)
    args.directory_budget_bytes=256*1024**2
    config=resident_configuration(pipeline,mode)
    if kind=='host':
        config=config.replace('host_budget_bytes=4294967296','host_budget_bytes=134217728')
        message='node host increment'
    else:
        config=config.replace('device_budget_bytes=2147483648','device_budget_bytes=67108864')
        message='physical GPU increment'
    case,_=run_case(args,ROOT,'gpu',2,'resident_budget_'+kind,2,grid='32,32,32',
        insitu_config=config,postprocess_transport=mode,reject=message,failure_after_start=True)
    assert not (case/'outdat/render/lifecycle_rank0.json').exists()


@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
@pytest.mark.parametrize('ranks,axis', [(1,'x'), (2,'x'), (2,'y'), (2,'z')])
def test_resident_q_native_smoke(tmp_path, pipeline, ranks, axis):
    """First actual Catalyst Q candidate; not the full R6/R7 admission matrix."""
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE') != '1':
        pytest.skip('Select the explicitly built strict device renderer candidate')
    args = arguments(tmp_path, axis)
    args.statistics = False
    args.directory_budget_bytes = 256 * 1024**2
    args.runtime_timeout_seconds = 300
    config = resident_configuration(pipeline, 'pinned', statistics=False, profile='q_surface')
    memory_check = os.environ.get('ASTR_INSITU_RESIDENT_MEMCHECK') == '1'
    trace = os.environ.get('ASTR_INSITU_RESIDENT_TRACE') == '1'
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'resident_q', 2, enabled=False,
        grid='32,32,32', insitu_config=config, postprocess_transport='pinned', no_field_io=True,
        insitu_timing=True, memcheck=memory_check, nsys_trace=trace)
    output = case / 'outdat/render'
    log = (case / 'run.log').read_text()
    assert 'ASTR_INSITU_RESIDENT_BRIDGE' in log
    assert 'ASTR_INSITU_COMPACT_BRIDGE' not in log
    assert not list(output.rglob('*.vtp')) and not list(output.glob('*.pvtp'))
    assert not list((case / 'outdat/new').rglob('*.h5'))
    assert not list(output.glob('sample.*.bin'))
    for step in (1, 2):
        for rank in range(ranks):
            receipt = json.loads((output / f'mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['rendering_pipeline'] == pipeline and receipt['geometry_host_bytes'] == 0
            assert receipt['local_points'] > 0 and receipt['local_cells'] > 0
            assert receipt['products']['q_surface']['image'] == 'published'
        picture = output / f'q_surface.step{step:08d}.jpeg'
        assert picture.with_suffix('.eps').is_file()
        with Image.open(picture) as image:
            pixels = np.asarray(image.convert('RGB')).astype(int)
        assert pixels.shape == (600, 800, 3)
        domain = pixels[50:550, 50:650]
        assert np.count_nonzero(domain.max(axis=2)-domain.min(axis=2) > 40) > 1000
    for rank in range(ranks):
        lifecycle = json.loads((output / f'lifecycle_rank{rank}.json').read_text())
        assert lifecycle['finalized'] and [frame[0] for frame in lifecycle['frames']] == [1, 2]
    if memory_check:
        reports = list(case.glob('memcheck.*.log'))
        assert len(reports) == ranks
        assert all('ERROR SUMMARY: 0 errors' in path.read_text() for path in reports)


@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
@pytest.mark.parametrize('ranks,axis', [(1,'x'), (2,'x'), (2,'y'), (2,'z')])
def test_resident_all_products_smoke(tmp_path, pipeline, ranks, axis):
    if os.environ.get('ASTR_INSITU_RESIDENT_Q_CANDIDATE') != '1':
        pytest.skip('Select the explicitly built strict device renderer candidate')
    args = arguments(tmp_path, axis)
    args.directory_budget_bytes = 256*1024**2
    args.runtime_timeout_seconds = 300
    config = resident_configuration(pipeline, 'pinned')
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'resident_all', 2, enabled=False,
        grid='32,32,32', insitu_config=config, postprocess_transport='pinned', no_field_io=True,
        insitu_timing=True)
    names = {'q_surface','velocity_slice','instantaneous_streamlines','crossing_streamlines',
             'mean_reynolds_streamlines','mean_favre_streamlines'}
    output = case/'outdat/render'
    for rank in range(ranks):
        for step in (1,2):
            record = json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['rendering_pipeline']==pipeline and record['geometry_host_bytes']==0
            assert set(record['products'])==names
            for name, product in record['products'].items():
                assert product['image']=='published' and product['scalar_bar_visible']
                expected = 'mean_u_'+name.split('_')[1] if name.startswith('mean_') else 'u'
                assert product['color_field']==expected and product['color_range']==[-1.,1.]
    for name in names:
        for step in (1,2):
            picture = output/f'{name}.step{step:08d}.jpeg'
            assert picture.with_suffix('.eps').is_file()
            with Image.open(picture) as image:
                pixels=np.asarray(image.convert('RGB')).astype(int)
            assert pixels.shape==(600,800,3)
            domain=pixels[50:550,50:650]
            assert np.count_nonzero(domain.max(axis=2)-domain.min(axis=2)>40)>100
    log=(case/'run.log').read_text()
    assert 'ASTR_IS8_COMPACT_TRACE_READ' not in log and 'ASTR_INSITU_COMPACT_BRIDGE' not in log
    for rounds, count in re.findall(r'ASTR_INSITU_RESIDENT_TRAJECTORY .*?rounds=(\d+).*?control_read_bytes=(\d+)',log):
        assert int(count)<=int(rounds)*2048
    assert not list(output.rglob('*.vtp')) and not list(output.glob('sample.*.bin'))
    assert not list((case/'outdat/new').rglob('*.h5'))


def check_device_render(case, ranks, steps, statistics=True, mean_steps=None):
    pictures = check_render(case, ranks, steps, statistics, full_volume_downloads=False)
    log = (case / 'run.log').read_text()
    frames = re.findall(r'ASTR_INSITU_DEVICE_FRAME rank=(\d+) step=(\d+) transport=(\S+)', log)
    assert sorted((int(rank), int(step)) for rank, step, _ in frames) == [
        (rank, step) for rank in range(ranks) for step in steps]
    assert 'ASTR_INSITU_TEST_DEVICE_TRANSPORT' not in log
    for accepted,remaining,minimum in re.findall(
            r'reason=sub_minimum_remaining accepted=(\S+) remaining=(\S+) minimum=(\S+)',log):
        accepted,remaining,minimum=map(float,(accepted,remaining,minimum))
        assert 0 < remaining < minimum
        assert abs(accepted+remaining-np.pi) <= 2e-10
        assert minimum == .01*(2*np.pi/32)
    for rank in range(ranks):
        for step in steps:
            receipt = json.loads((case / 'outdat/render' / f'mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['processing_backend'] == 'device'
            assert {'q_surface', 'velocity_slice', 'instantaneous_streamlines', 'crossing_streamlines'} <= set(receipt['products'])
            covered=statistics and step>0 and (mean_steps is None or step in mean_steps)
            if covered:
                assert {'mean_reynolds_streamlines', 'mean_favre_streamlines'} <= set(receipt['products'])
            else:
                assert not {'mean_reynolds_streamlines', 'mean_favre_streamlines'} & set(receipt['products'])
            for name,product in receipt['products'].items():
                assert product['scalar_bar_visible'] is True
                assert product['scalar_bar_label_color']==[0.,0.,0.]
                assert product['color_range']==[-1.,1.]
                assert product['color_field']==('mean_u_'+name.split('_')[1] if name.startswith('mean_') else 'u')
    # Proxy visibility alone misses a long title consuming the entire color bar.
    for path in pictures:
        with Image.open(path) as image:
            region=np.asarray(image.convert('RGB'))[:350,660:790].astype(int)
        colored=region.max(axis=2)-region.min(axis=2)>50
        assert colored.sum(axis=0).max()>100,('Missing rendered color bar',path)
    return pictures


def check_geometry_fields(case, step, statistics=True, products=None, device_products=True):
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    checkpoint = case / f'outdat/new/checkpoints/step{step:012d}'
    def periodic(values):
        return np.pad(values[:32,:32,:32], ((0,1),)*3, mode='wrap')
    with h5py.File(checkpoint / 'state.h5') as state:
        density = state['q0001'][...]
        velocity = [periodic(state[f'q{c:04d}'][...]/density) for c in (2,3,4)]
    h = 2*np.pi/32
    gradient = np.zeros((32,32,32,3,3))
    for c in range(3):
        value = velocity[c][:32,:32,:32]
        for d,axis in enumerate((2,1,0)):
            gradient[...,c,d] = (.75*(np.roll(value,-1,axis)-np.roll(value,1,axis))
                -.15*(np.roll(value,-2,axis)-np.roll(value,2,axis))
                +(np.roll(value,-3,axis)-np.roll(value,3,axis))/60)/h
    q = periodic(-.5*np.einsum('...ij,...ji->...',gradient,gradient))
    means={}
    if statistics:
        with h5py.File(checkpoint / 'statistics.h5') as state:
            # The checkpoint prepends three window/time fields to device state slots.
            means = {f'mean_{component}_{kind}':periodic(state[f'q{index:04d}'][...])
                for index,(kind,component) in enumerate(
                    ((kind,component) for kind in ('reynolds','favre') for component in ('u','v','w')),10)}
    expected = dict(zip(('u','v','w'),velocity), Q_rs=q, **means)
    worst = 0.
    paths=sorted((case / 'outdat/render').glob(f'*.step{step:08d}.pvtp'))
    if products is None:
        assert len(paths)==6,paths
    else:
        assert {path.name.split('.step')[0] for path in paths}==set(products),paths
    for path in paths:
        reader=vtkXMLPPolyDataReader();reader.SetFileName(str(path));reader.Update()
        data=reader.GetOutput()
        assert data.GetNumberOfPoints()>0 and data.GetNumberOfCells()>0,path
        xyz=vtk_to_numpy(data.GetPoints().GetData())
        assert xyz.dtype==np.float64 and np.isfinite(xyz).all(),path
        scaled=np.clip(xyz/h,0,32)
        base=np.minimum(np.floor(scaled).astype(int),31)
        fraction=scaled-base
        for name,field in expected.items():
            array=data.GetPointData().GetArray(name)
            if array is None:
                assert name not in ('u','v','w'),(path,name)
                continue
            interpolated=np.zeros(len(xyz))
            for i in (0,1):
                for j in (0,1):
                    for k in (0,1):
                        weight=np.prod(np.where(np.array([i,j,k]),fraction,1-fraction),axis=1)
                        interpolated+=weight*field[base[:,2]+k,base[:,1]+j,base[:,0]+i]
            values=vtk_to_numpy(array)
            assert np.isfinite(values).all(),(path,name)
            error=float(np.max(abs(values-interpolated)))
            assert error<=2e-10,(path,name,error)
            worst=max(worst,error)
        if path.name.startswith('q_surface'):
            assert np.max(abs(vtk_to_numpy(data.GetPointData().GetArray('Q_rs'))-.25))<=2e-10
        if 'streamlines' in path.name and device_products:
            status=vtk_to_numpy(data.GetPointData().GetArray('termination_status'))
            residual=vtk_to_numpy(data.GetPointData().GetArray('untravelled_arc_length'))
            assert set(status.tolist()) <= {1.,3.,10.}
            assert np.isfinite(residual).all() and np.all(residual>=0)
            assert np.all((residual[status==3]>0)&(residual[status==3]<.01*h))
    return worst


@pytest.mark.parametrize('mode', ['pi','de'])
@pytest.mark.parametrize('index',range(4))
def test_immutable_product_field_oracle(mode,index,record_property):
    root=os.environ.get('ASTR_IS8_PRODUCT_ARTIFACTS')
    if not root:
        pytest.skip('Select immutable native products explicitly for offline field validation')
    cases=list((Path(root)/f'test_actual_device_products_{mode}{index}').glob('gpu_np*_products'))
    assert len(cases)==1,cases
    record_property('geometry_field_maxabs',check_geometry_fields(cases[0],2))


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
def test_actual_device_products(tmp_path, ranks, axis, mode, record_property):
    args = arguments(tmp_path, axis)
    args.directory_budget_bytes = 256 * 1024**2
    args.runtime_timeout_seconds = 240
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'products', 2, grid='32,32,32',
        insitu_config=device_configuration(mode), postprocess_transport=mode)
    check_device_render(case, ranks, [1, 2])
    record_property('geometry_field_maxabs',check_geometry_fields(case,2))
    control, _ = run_case(args, ROOT, 'gpu', ranks, 'plain', 2, grid='32,32,32',
        insitu_config=configuration(render=False, interval=1))
    for filename in ('state.h5', 'statistics.h5'):
        relative = 'outdat/new/checkpoints/step000000000002/' + filename
        with h5py.File(case / relative) as actual, h5py.File(control / relative) as expected:
            names = []
            actual.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
            for name in names:
                np.testing.assert_array_equal(actual[name][...], expected[name][...], err_msg=name)


@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
def test_exact_restart_and_transport_override(tmp_path, mode, axis):
    args = arguments(tmp_path, axis)
    args.directory_budget_bytes = 256 * 1024**2
    args.runtime_timeout_seconds = 240
    config = device_configuration(mode, interval=3)
    continuous, _ = run_case(args, ROOT, 'gpu', 2, 'continuous', 4, grid='32,32,32',
        checkpoint_interval=2, insitu_config=config, postprocess_transport=mode)
    check_device_render(continuous, 2, [3, 4])
    source = continuous / 'outdat/new/checkpoints/step000000000002'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert before['insitu_control.bin'][:8] == b'ASTRIR03'
    assert before['insitu_control.bin'][176:192] == b'compatible      '
    resumed, _ = run_case(args, ROOT, 'gpu', 2, 'resumed', 4, grid='32,32,32',
        restore=source, checkpoint_interval=2, insitu_config=config, postprocess_transport=mode)
    pictures = check_device_render(resumed, 2, [3, 4])
    final = 'outdat/new/checkpoints/step000000000004/'
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(continuous / final / name, resumed / final / name)
    assert (continuous / final / 'insitu_control.bin').read_bytes() == (resumed / final / 'insitu_control.bin').read_bytes()
    for path in pictures:
        with Image.open(path) as actual, Image.open(continuous / 'outdat/render' / path.name) as expected:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for path in (resumed / 'outdat/render').rglob('*.vtp'):
        expected = continuous / 'outdat/render' / path.relative_to(resumed / 'outdat/render')
        assert path.read_bytes() == expected.read_bytes(), path.name
    other = 'device-aware' if mode == 'pinned' else 'pinned'
    changed = device_configuration(other, interval=3)
    run_case(args, ROOT, 'gpu', 2, 'rejected_switch', 4, grid='32,32,32', restore=source,
        checkpoint_interval=2, insitu_config=changed, postprocess_transport=other,
        reject='native render configuration differs; select explicit override')
    switched, _ = run_case(args, ROOT, 'gpu', 2, 'switched', 4, grid='32,32,32',
        restore=source, override=True, checkpoint_interval=2, insitu_config=changed,
        postprocess_transport=other)
    check_device_render(switched, 2, [3, 4])
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(continuous / final / name, switched / final / name)
    # Only the explicit transport identity and its separated signature differ.
    original = (continuous / final / 'insitu_control.bin').read_bytes()
    replacement = (switched / final / 'insitu_control.bin').read_bytes()
    assert len(original) == len(replacement) == 361
    for region in (slice(0,128), slice(136,160), slice(176,None)):
        assert original[region] == replacement[region]
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device', 'default'])
@pytest.mark.parametrize('ranks', [1, 2])
def test_unbuilt_rendering_pipeline_has_no_fallback(tmp_path, pipeline, ranks):
    args = arguments(tmp_path)
    args.directory_budget_bytes = 256 * 1024**2
    config = device_configuration('pinned')
    if pipeline=='default':
        config=config.replace("rendering_pipeline='compatible',",'')
    else:
        config=config.replace("rendering_pipeline='compatible'",f"rendering_pipeline='{pipeline}'")
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'unbuilt_' + pipeline, 2,
        grid='32,32,32', insitu_config=config, postprocess_transport='pinned',
        reject='selected rendering pipeline is not built/admitted; no compatible fallback')
    log = (case / 'run.log').read_text()
    assert 'ASTR_INSITU_DEVICE_FRAME ' not in log
    assert not list((case / 'outdat').rglob('*.jpeg'))


@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
def test_native_memory_and_zero_mean_coverage(tmp_path, mode):
    args = arguments(tmp_path, 'y')
    args.directory_budget_bytes = 256 * 1024**2
    args.runtime_timeout_seconds = 300
    config = device_configuration(mode).replace('initial_frame=f', 'initial_frame=t')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'native_memory', 2, grid='32,32,32',
        insitu_config=config, postprocess_transport=mode, memcheck=True)
    check_device_render(case, 2, [0,1,2])
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)


@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('kind',['host','device'])
def test_native_device_observed_budget_rejection(tmp_path,mode,kind):
    args=arguments(tmp_path)
    args.directory_budget_bytes=256*1024**2
    config=device_configuration(mode)
    if kind=='host':
        config=config.replace('host_budget_bytes=4294967296','host_budget_bytes=134217728')
        message='node host increment'
    else:
        config=config.replace('device_budget_bytes=2147483648','device_budget_bytes=67108864')
        message='physical GPU increment'
    run_case(args,ROOT,'gpu',2,'native_budget_'+kind,2,grid='32,32,32',
        insitu_config=config,postprocess_transport=mode,reject=message,failure_after_start=True)
