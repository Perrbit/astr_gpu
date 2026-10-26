"""JPEG-only publication, encoding and explicitly selected local GPU checks."""
import ast
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import h5py
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/insitu'))
import image_publication as publication


@pytest.mark.parametrize('value,expected', [(None, False), ('1', True), ('0', False)])
def test_eps_selection(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('ASTR_INSITU_WRITE_EPS', raising=False)
    else:
        monkeypatch.setenv('ASTR_INSITU_WRITE_EPS', value)
    assert publication.eps_output_enabled() is expected


@pytest.mark.parametrize('value', ['', 'false', '2'])
def test_invalid_eps_selection(monkeypatch, value):
    monkeypatch.setenv('ASTR_INSITU_WRITE_EPS', value)
    with pytest.raises(ValueError, match='ASTR_INSITU_WRITE_EPS'):
        publication.eps_output_enabled()


def test_jpeg_only_publication_and_no_overwrite(tmp_path):
    picture = tmp_path / 'frame.jpeg'
    assert publication.publish_pair(picture, b'jpeg', None) is None
    assert picture.read_bytes() == b'jpeg'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['frame.jpeg']
    with pytest.raises(FileExistsError):
        publication.publish_pair(picture, b'other', None)
    assert picture.read_bytes() == b'jpeg'


@pytest.mark.parametrize('phase', ['stage_jpeg', 'publish_jpeg'])
@pytest.mark.parametrize('code', [errno.EACCES, errno.ENOSPC, errno.EIO])
def test_jpeg_only_failure_cleans_owned_files(tmp_path, monkeypatch, phase, code):
    def fail(*args):
        raise OSError(code, 'injected')

    monkeypatch.setattr(publication if phase == 'stage_jpeg' else publication.os,
                        'write_payload' if phase == 'stage_jpeg' else 'link', fail)
    result = publication.publish_pair(tmp_path / 'frame.jpeg', b'jpeg', None)
    assert result['phase'] == phase and result['errno'] == code
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('write_eps', [False, True])
def test_m12_render_encodes_only_selected_formats(tmp_path, monkeypatch, write_eps):
    pipeline = ROOT / 'scripts/insitu/device_render_pipeline.py'
    tree = ast.parse(pipeline.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'render_product')
    pixels = np.full((6, 8, 3), [170, 20, 40], dtype=np.uint8)
    scalars = SimpleNamespace(GetNumberOfComponents=lambda: 3)
    image = SimpleNamespace(GetDimensions=lambda: (8, 6, 1),
        GetPointData=lambda: SimpleNamespace(GetScalars=lambda: scalars))
    window = SimpleNamespace(MakeCurrent=lambda: None,
                             GetClassName=lambda: 'vtkEGLRenderWindow')

    class IntegerArray:
        def InsertNextValue(self, value): self.value = value
        def SetNumberOfTuples(self, count): self.value = 0
        def GetValue(self, index): return self.value

    def reduce(local, total, operation):
        total.value = local.value

    def collective_error(message):
        if message:
            raise RuntimeError(message)

    formats = []
    original_save = Image.Image.save

    def save(self, stream, format=None, **kwargs):
        formats.append(format)
        return original_save(self, stream, format=format, **kwargs)

    monkeypatch.setattr(Image.Image, 'save', save)
    namespace = dict(os=os, io=io, json=json, perf_counter=perf_counter, Image=Image,
        pv=SimpleNamespace(Render=Mock()),
        vtkPVProcessWindow=SimpleNamespace(GetRenderWindow=lambda: window),
        EGL=lambda: SimpleNamespace(current_uuid=lambda: 'gpu-id'), expected_uuid='gpu-id',
        collective_error=collective_error, capture_image=lambda view, size: (image, 95, 0),
        vtk_to_numpy=lambda data: pixels, rank=0, profile='m12_diagnostics', output=tmp_path,
        write_eps=write_eps, image_size=[8, 6], publish_pair=publication.publish_pair, timing=False,
        vtkIntArray=IntegerArray, vtkCommunicator=SimpleNamespace(MAX_OP=0, SUM_OP=1),
        controller=SimpleNamespace(AllReduce=reduce),
        dispatch=SimpleNamespace(isolate_image=lambda operation: operation,
            collective_call=lambda controller, operation: operation()))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(pipeline), 'exec'), namespace)
    view = SimpleNamespace(CameraPosition=[0., 0., 1.], CameraFocalPoint=[0., 0., 0.],
                           CameraViewUp=[0., 1., 0.], CameraParallelScale=1.)
    item = dict(view=view, image_size=[8, 6], color='u_Uinf', color_range=[0., 1.],
        legend=SimpleNamespace(Visibility=1),
        actor=SimpleNamespace(GetProperty=lambda: SimpleNamespace(GetLighting=lambda: False)),
        physical_bounds=[0., 1.] * 3, local_bounds=[0., 1.] * 3)
    result = namespace['render_product']('velocity_slice', item, 10, .2, 4, 2)
    assert result['image'] == 'published'
    assert formats == (['JPEG', 'EPS'] if write_eps else ['JPEG'])
    assert result['image_formats'] == (['jpeg', 'eps'] if write_eps else ['jpeg'])
    assert ('eps_source' in result) is write_eps
    assert (tmp_path / 'velocity_slice.step00000010.eps').exists() is write_eps
    assert not list(tmp_path.glob('*.partial'))
    with Image.open(tmp_path / 'velocity_slice.step00000010.jpeg') as decoded:
        decoded.load()
        assert decoded.size == (8, 6) and decoded.info['dpi'] == (300, 300)


@pytest.mark.parametrize('write_eps', [False, True])
def test_compatible_encoder_skips_eps_work(monkeypatch, write_eps):
    pipeline = ROOT / 'scripts/insitu/tgv_pipeline.py'
    function = next(node for node in ast.parse(pipeline.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'encode_image')
    jpeg = io.BytesIO()
    Image.new('RGB', (8, 6), (170, 20, 40)).save(jpeg, format='JPEG')
    payload = jpeg.getvalue()
    writer = Mock()
    writer.GetErrorCode.return_value = 0
    writer.GetResult.return_value = np.frombuffer(payload, dtype=np.uint8)
    namespace = dict(io=io, Image=Image, write_eps=write_eps, image_size=[8, 6],
                     vtkJPEGWriter=lambda: writer, vtk_to_numpy=lambda data: data)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(pipeline), 'exec'), namespace)
    original_open = Image.open
    calls = []

    def open_image(*args, **kwargs):
        calls.append(True)
        return original_open(*args, **kwargs)

    monkeypatch.setattr(Image, 'open', open_image)
    encoded, eps = namespace['encode_image'](
        SimpleNamespace(GetDimensions=lambda: (8, 6, 1)), 95, 0)
    assert encoded == payload
    assert bool(calls) is write_eps
    assert (eps is not None) is write_eps
    if write_eps:
        assert eps.startswith(b'%!PS-Adobe')


@pytest.mark.skipif(not all(os.environ.get(key) for key in (
    'ASTR_INSITU_JPEG_ONLY_EXE', 'ASTR_INSITU_JPEG_ONLY_SOURCE', 'ASTR_OUTPUT_INSITU_BACKEND')),
    reason='Select the local M12 executable, read-only source and Catalyst implementation')
def test_actual_m12_jpeg_only_matches_paired_output(tmp_path, record_property):
    from run_platform_insitu_fullsize import prepare_case
    from run_m12_mp7_nscbc_performance import runtime, mpi_command
    from insitu_resource_monitor import run_monitored
    from run_output_restart_validation import compare_fields

    ranks = 2  # The M12 HD small-grid profile admits NP=2/4 x-slabs, not NP=1.
    executable = Path(os.environ['ASTR_INSITU_JPEG_ONLY_EXE']).resolve(strict=True)
    args = SimpleNamespace(source=Path(os.environ['ASTR_INSITU_JPEG_ONLY_SOURCE']),
        library=Path(os.environ['ASTR_OUTPUT_INSITU_BACKEND']), m12_boundary='nscbc',
        m12_products='slices-wall', host_budget=16*1024**3, device_budget=6*1024**3,
        reserve=2*1024**3)
    mpiexec = Path(os.environ.get('ASTR_OUTPUT_MPIEXEC',
        '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'))
    cases, pictures, resources = {}, {}, {}
    for mode, eps in [('jpeg-only', '0'), ('paired', '1')]:
        case = tmp_path / mode
        prepare_case(args, case, (64, 32, 24), 2, True, checkpoint=True)
        config = case/'datin/input.insitu'
        config.write_text(config.read_text().replace('product_steps=10', 'product_steps=1'))
        env = runtime(f'{ranks},1,1', 'device-aware')
        env.update(ASTR_INSITU_CONFIG='datin/input.insitu', ASTR_INSITU_WRITE_EPS=eps,
            ASTR_GPU_MP_FACE_LAYOUT='x-contiguous', ASTR_GPU_MP_SPLIT_CACHE='on',
            ASTR_INSITU_M12_W_COLOR_MAX='0.01', ASTR_INSITU_TIMING='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for key in ('DISPLAY', 'PYTHONPATH', 'VTK_EGL_DEVICE_INDEX', 'CATALYST_IMPLEMENTATION_PREFER_ENV'):
            env.pop(key, None)
        command = mpi_command(mpiexec, ranks, 'device-aware') + [str(executable), 'run', 'datin/input.dat']
        with (case/'run.log').open('wb') as stream:
            resources[mode] = run_monitored(command, case, env, stream, case/'resources.json',
                device_reserve_bytes=args.reserve, timeout_seconds=300)
        log = (case/'run.log').read_text()
        assert 'The job is done!' in log
        assert 'ASTR_GPU_MP_POSITIVITY=flux' in log
        output = case/'outdat/render'
        expected = set()
        for step in (1, 2):
            for rank in range(ranks):
                receipt = json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                assert receipt['geometry_host_bytes'] == 0
                products = receipt['products']
                required = {'velocity_slice', 'temperature_slice', 'spanwise_slice'} | {
                    f'{field}.x{100*n}_{100*(n+1)}'
                    for field in ('velocity_slice', 'temperature_slice') for n in range(3)}
                assert required <= set(products)
                assert not any('streamlines' in name or 'q_surface' in name for name in products)
                for name, item in products.items():
                    assert item['image'] == 'published'
                    assert item['image_formats'] == (['jpeg'] if eps == '0' else ['jpeg', 'eps'])
                    expected.add(f'{name}.step{step:08d}.jpeg')
        pictures[mode] = {path.name: path for path in output.glob('*.jpeg')}
        assert set(pictures[mode]) == expected
        assert len(list(output.glob('*.eps'))) == (0 if eps == '0' else len(expected))
        assert not list(case.rglob('*.partial'))
        for path in pictures[mode].values():
            with Image.open(path) as image:
                image.load()
                assert image.info['dpi'] == (300, 300)
                assert np.any(np.asarray(image) < 245)
        assert sum(path.stat().st_size for path in case.rglob('*') if path.is_file()) <= 4*1024**3
        cases[mode] = case

    assert pictures['jpeg-only'].keys() == pictures['paired'].keys()
    for name, path in pictures['jpeg-only'].items():
        assert path.read_bytes() == pictures['paired'][name].read_bytes()
    assert (cases['jpeg-only']/'outdat/render/probes.csv').read_bytes() == (
        cases['paired']/'outdat/render/probes.csv').read_bytes()
    assert (cases['jpeg-only']/'flowstate.dat').read_bytes() == (
        cases['paired']/'flowstate.dat').read_bytes()
    state = Path('outdat/new/checkpoints/step000000000002/state.h5')
    compare_fields(cases['jpeg-only']/state, cases['paired']/state)
    with h5py.File(cases['jpeg-only']/state) as saved:
        assert all(np.isfinite(saved[name][...]).all() for name in saved if name.startswith('q'))
        assert all(np.all(saved[name][...] > 0.) for name in ('q0001', 'q0010', 'q0011'))
        extras = saved['rank_extras'][...]
        cursor = 0
        global_cells = saved['identity'][3:6]-1
        for partition in saved['partitions'][...].reshape(-1, 8):
            origin, cells = partition[:3], partition[3:6]
            halo, count = map(int, partition[6:8])
            shape = tuple((cells+2*halo+1)[::-1])
            owned = (cells+(origin+cells == global_cells))[::-1]
            nonowned = np.ones(shape, dtype=bool)
            nonowned[tuple(slice(halo, int(halo+n)) for n in owned)] = False
            coordinates = np.indices(shape)
            physical = np.ones(shape, dtype=bool)
            for d in range(3):
                index = coordinates[d]+origin[2-d]-halo
                physical &= (index >= 0) & (index <= global_cells[2-d])
            # Exterior primitive ghost caches may contain NaNs; preserve their
            # exact bits, but require every physical/shared-node cache to be finite.
            values = extras[cursor:cursor+count].reshape(11, -1)
            assert np.isfinite(values[:, physical[nonowned]]).all()
            cursor += count
        assert cursor == extras.size
        nonfinite_exterior_cache_values = int(np.count_nonzero(~np.isfinite(extras)))
    report = dict(status='passed', ranks=ranks, topology=[ranks, 1, 1], intervals=[64, 32, 24],
        steps=2, dt=.02, jpeg_count_per_run=len(pictures['jpeg-only']),
        jpeg_bytes_identical=True, authoritative_state_maxabs=0., full_state_bitwise_equal=True,
        physical_and_shared_caches_finite=True, probes_identical=True,
        nonfinite_exterior_cache_values=nonfinite_exterior_cache_values,
        jpeg_only_eps_count=0, geometry_host_bytes=0, resources=resources,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        pipeline_sha256=hashlib.sha256((ROOT/'scripts/insitu/device_render_pipeline.py').read_bytes()).hexdigest(),
        scope='local format and state-isolation check, not production physics or performance')
    (tmp_path/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    record_property('local_gpu_report', str(tmp_path/'report.json'))
