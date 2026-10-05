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


def device_configuration(mode, statistics=True, interval=1, profile='all'):
    return configuration(statistics=statistics, interval=interval).replace('render=.true.,',
        f"render=.true.,derivative_backend='gpu',processing_backend='device',"
        f"postprocess_transport='{mode}',products='{profile}',")


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


def check_geometry_fields(case, step):
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
    with h5py.File(checkpoint / 'statistics.h5') as state:
        # The checkpoint prepends three window/time fields to device state slots.
        means = {f'mean_{component}_{kind}':periodic(state[f'q{index:04d}'][...])
            for index,(kind,component) in enumerate(
                ((kind,component) for kind in ('reynolds','favre') for component in ('u','v','w')),10)}
    expected = dict(zip(('u','v','w'),velocity), Q_rs=q, **means)
    worst = 0.
    paths=sorted((case / 'outdat/render').glob(f'*.step{step:08d}.pvtp'))
    assert len(paths)==6,paths
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
        if 'streamlines' in path.name:
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
    assert before['insitu_control.bin'][:8] == b'ASTRIR02'
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
    assert len(original) == len(replacement) == 345
    for region in (slice(0,128), slice(136,160), slice(176,None)):
        assert original[region] == replacement[region]
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


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
