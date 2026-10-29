"""Small real-solver tests of the dynamic numerical-slice channel, without Catalyst."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get('ASTR_SLICE_EXE', ROOT / 'build_gpu_probe/bin/astr')).resolve()
MPIEXEC = Path(os.environ.get('ASTR_CATALOG_MPIEXEC',
    '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')).absolute()


def arguments(output):
    return SimpleNamespace(output=output, executable=EXE, mpiexec=MPIEXEC, case='tgv', mode='steps',
        restart_step=5, initial_dimension=0, legacy_statistics=False, statistics=False,
        initial_restart=False, filter_workspace='scalar', force='feedback', axis='x', no_samples=True,
        directory_budget_bytes=256*1024**2)


def product(name='midplane', fields=None, source='instantaneous'):
    return dict(id=name, kind='slice', enable_request=1, source=source,
        fields=fields or ['density', 'velocity_x', 'Q_rs', 'velocity_divergence', 'density_gradient_magnitude'],
        sampling=dict(mode='grid_index', axis='k', index=8, index_ranges=dict(i=[0, 16], j=[0, 16])),
        output=dict(channel='data', format='hdf5-xdmf'), schedule=dict(mode='steps', interval=4))


def segments(case):
    return sorted((case / 'outdat/new/slice_products').glob('*/*/segment*'))


def frames(segment):
    return sorted(segment.glob('step*.h5'))


@pytest.mark.parametrize('backend,ranks', [('cpu', 1), ('gpu', 1), ('gpu', 2)])
def test_real_tgv_slices_and_exact_restart(tmp_path, backend, ranks):
    args = arguments(tmp_path)
    reference, _ = run_case(args, ROOT, backend, ranks, 'continuous', 12,
        checkpoint_interval=5, product_catalog_records=[product()])
    actual_segments = segments(reference)
    assert len(actual_segments) == 1
    segment = actual_segments[0]
    assert [int(p.stem[4:]) for p in frames(segment)] == [0, 4, 8, 12]
    xml = ET.parse(segment / 'series.xdmf')
    assert len(xml.findall('.//Grid[@GridType="Uniform"]')) == 4
    for frame in frames(segment):
        with h5py.File(frame) as data:
            assert data['density'].shape == (17, 17)
            assert data['density'].dtype == np.dtype('float64')
            assert set(data) - {'clock', 'step', 'field_ids', 'index_ranges', 'reference_state'} == set(product()['fields'])
            assert np.isfinite(data['Q_rs']).all()
            assert 'velocity' not in data and 'coordinates' not in data
    with h5py.File(segment / 'geometry.h5') as geometry:
        assert geometry['coordinates'].shape == (17, 17, 3)
    prefix, _ = run_case(args, ROOT, backend, ranks, 'prefix', 5,
        checkpoint_interval=5, product_catalog_records=[product()])
    source = prefix / 'outdat/new/checkpoints/step000000000005'
    assert (source / 'insitu_control.bin').read_bytes()[:8] == b'ASTRIR07'
    # A conflicting disk list must not replace the restored, already effective definition.
    changed = product(fields=['pressure'])
    resumed, _ = run_case(args, ROOT, backend, ranks, 'resumed', 12, restore=source,
        checkpoint_interval=5, product_catalog_records=[changed])
    compare_fields(reference / 'outdat/new/checkpoints/step000000000012/state.h5',
                   resumed / 'outdat/new/checkpoints/step000000000012/state.h5')
    continuation = segments(resumed)
    assert len(continuation) == 1 and (continuation[0] / 'geometry.h5').is_file()
    assert [p.name for p in frames(continuation[0])] == [p.name for p in frames(segment)][2:]
    for frame in frames(continuation[0]):
        with h5py.File(frame) as actual, h5py.File(segment / frame.name) as expected:
            assert set(actual) == set(expected)
            for field in actual:
                np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
    status = json.loads((resumed / 'outdat/new/products.status.json').read_text())
    assert status['products'][0]['version'] == 1


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_existing_means_and_output_does_not_change_authority(tmp_path, backend):
    args = arguments(tmp_path)
    args.statistics = True
    mean_fields = ['velocity_x', 'velocity_y', 'velocity_z', 'Q_rs', 'velocity_divergence']
    definitions = [product(), product('time-reynolds', mean_fields, 'reynolds'),
                   product('time-favre', mean_fields, 'favre'), product('missing-pressure', ['pressure'], 'reynolds')]
    baseline, _ = run_case(args, ROOT, backend, 2, 'stats-only', 12, checkpoint_interval=5)
    observed, _ = run_case(args, ROOT, backend, 2, 'with-slices', 12,
        checkpoint_interval=5, product_catalog_records=definitions)
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(baseline / 'outdat/new/checkpoints/step000000000012' / name,
                       observed / 'outdat/new/checkpoints/step000000000012' / name)
    status = json.loads((observed / 'outdat/new/products.status.json').read_text())
    records = {entry['id']: entry for entry in status['products']}
    assert records['missing-pressure']['state'] == 'invalid-configuration'
    for name in ('time-reynolds', 'time-favre'):
        assert records[name]['state'] == 'active'
        segment = next(path for path in segments(observed) if path.parent.parent.name == name)
        assert [int(path.stem[4:]) for path in frames(segment)] == [1, 5, 9]
        with h5py.File(frames(segment)[-1]) as frame:
            assert 'duration=' in str(frame.attrs['source_measure'])
            assert np.isfinite(frame['Q_rs']).all()


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_mean_slice_exact_continuation(tmp_path,backend):
    args=arguments(tmp_path);args.statistics=True
    definitions=[product('reynolds',['velocity_x','Q_rs'],'reynolds'),
                 product('favre',['velocity_x','Q_rs'],'favre')]
    common=dict(checkpoint_interval=5,product_catalog_records=definitions)
    reference,_=run_case(args,ROOT,backend,2,'mean-continuous',12,**common)
    first,_=run_case(args,ROOT,backend,2,'mean-prefix',5,**common)
    resumed,_=run_case(args,ROOT,backend,2,'mean-resumed',12,
        restore=first/'outdat/new/checkpoints/step000000000005',**common)
    for name in ('state.h5','statistics.h5'):
        compare_fields(reference/'outdat/new/checkpoints/step000000000012'/name,
                       resumed/'outdat/new/checkpoints/step000000000012'/name)
    for current in segments(resumed):
        name=current.parent.parent.name
        original=next(path for path in segments(reference) if path.parent.parent.name==name)
        assert [path.name for path in frames(current)]==['step000000000009.h5']
        for frame in frames(current):
            with h5py.File(frame) as a,h5py.File(original/frame.name) as b:
                for field in a:
                    np.testing.assert_array_equal(a[field],b[field],err_msg=field)


def test_bad_product_directory_does_not_stop_other_products(tmp_path):
    args = arguments(tmp_path)
    case, _ = run_case(args, ROOT, 'gpu', 2, 'isolate', 4,
        product_catalog_records=[product('x'*256), product('good')])
    status = json.loads((case / 'outdat/new/products.status.json').read_text())
    assert status['products'][0]['state'] == 'resource-disabled'
    assert status['products'][1]['state'] == 'active'
    good = next(path for path in segments(case) if path.parent.parent.name == 'good')
    assert [path.name for path in frames(good)] == ['step000000000000.h5', 'step000000000004.h5']


def test_cpu_gpu_same_phase_diagnostics(tmp_path):
    args = arguments(tmp_path)
    cpu, _ = run_case(args, ROOT, 'cpu', 2, 'compare', 4, product_catalog_records=[product()])
    gpu, _ = run_case(args, ROOT, 'gpu', 2, 'compare', 4, product_catalog_records=[product()])
    left, right = segments(cpu)[0], segments(gpu)[0]
    for expected, actual in zip(frames(left), frames(right), strict=True):
        with h5py.File(expected) as a, h5py.File(actual) as b:
            for name in product()['fields']:
                assert np.max(np.abs(a[name][...]-b[name][...])) <= 2e-10, name


@pytest.mark.parametrize('case_name', ['channel', 'curve'])
def test_wall_and_curved_open_boundary(tmp_path, case_name):
    args = arguments(tmp_path)
    args.case = case_name
    if case_name == 'channel':
        args.force = 'fixed'
    definition = product(fields=['density', 'velocity_x', 'pressure', 'temperature',
                                 'velocity_gradient_xy', 'Q_rs', 'density_gradient_magnitude'])
    definition['sampling'] = dict(mode='grid_index', axis='j', index=0,
                                  index_ranges=dict(i=[0, 16], k=[0, 16]))
    definitions = [definition]
    cpu, _ = run_case(args, ROOT, 'cpu', 2, case_name, 4, product_catalog_records=definitions, enabled=False)
    gpu, _ = run_case(args, ROOT, 'gpu', 2, case_name, 4, product_catalog_records=definitions, enabled=False)
    for expected, actual in zip(frames(segments(cpu)[0]), frames(segments(gpu)[0]), strict=True):
        with h5py.File(expected) as a, h5py.File(actual) as b:
            for name in definition['fields']:
                scale = max(1., np.max(np.abs(a[name][...])))
                assert np.max(np.abs(a[name][...]-b[name][...])) / scale <= 2e-10, name


def test_m12_original_blowing_and_slice_diagnostics(tmp_path):
    from run_m12_local_compare import run
    args = SimpleNamespace(output=tmp_path, source=Path('/home/dell/workspace/M12_C13'),
        executable=Path(os.environ.get('ASTR_SLICE_M12_EXE', ROOT/'build_m12_gpu/bin/astr')).resolve(),
        mpiexec=MPIEXEC, memcheck=False, diagnostics=False, capture_rhs=True,
        halo_transport='pinned', insitu_write_eps='0')
    definition = product()
    definition['sampling'] = dict(mode='grid_index', axis='k', index=12,
                                  index_ranges=dict(i=[0, 64], j=[0, 32]))
    cpu = run(args, 'cpu', '2,1,1', case_label='cpu_slices', product_catalog_records=[definition])
    gpu = run(args, 'gpu', '2,1,1', case_label='gpu_slices', product_catalog_records=[definition])
    assert [p.name for p in frames(segments(cpu)[0])] == [p.name for p in frames(segments(gpu)[0])]
    for expected, actual in zip(frames(segments(cpu)[0]), frames(segments(gpu)[0]), strict=True):
        with h5py.File(expected) as a, h5py.File(actual) as b:
            for name in definition['fields']:
                scale = max(1., np.max(np.abs(a[name][...])))
                assert np.max(np.abs(a[name][...]-b[name][...])) / scale <= 2e-10, name


def test_air5_all_primitive_fields_and_diagnostics(tmp_path):
    from run_output_air5_restart_validation import launch, reference_scales, DOMAIN
    args = arguments(tmp_path)
    args.case = 'hbl';args.axis = 'z';args.top_mode = 'characteristic'
    args.reconstruction = 5;args.mean_statistics = False;args.sample_interval = 2
    names = ['density', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature',
             'vibrational_temperature', 'mass_fraction_N2', 'mass_fraction_O2', 'mass_fraction_N',
             'mass_fraction_O', 'mass_fraction_NO', 'Q_rs', 'density_gradient_magnitude']
    definition = product(fields=names)
    cpu, _ = launch(args, 'cpu', 2, 'slices', 4, product_catalog_records=[definition], checkpoint_enabled=False)
    gpu, _ = launch(args, 'gpu', 2, 'slices', 4, product_catalog_records=[definition], checkpoint_enabled=False)
    reference = reference_scales()
    scales = dict(density=reference['density'], pressure=reference['pressure'],
        temperature=reference['temperature'], vibrational_temperature=reference['temperature'],
        Q_rs=(reference['velocity']/DOMAIN[1])**2,
        density_gradient_magnitude=reference['density']/DOMAIN[1])
    scales.update({name: reference['velocity'] for name in ('velocity_x', 'velocity_y', 'velocity_z')})
    scales.update({name: 1. for name in names if name.startswith('mass_fraction_')})
    errors = {}
    for expected, actual in zip(frames(segments(cpu)[0]), frames(segments(gpu)[0]), strict=True):
        with h5py.File(expected) as a, h5py.File(actual) as b:
            for name in names:
                error = float(np.max(np.abs(a[name][...]-b[name][...])))
                errors.setdefault(actual.name, {})[name] = dict(maxabs=error, scale=scales[name],
                                                                normalized=error/scales[name])
                assert error / scales[name] <= 2e-10, (name, errors[actual.name][name])
    (tmp_path/'dimensional_diagnostic_comparison.json').write_text(json.dumps(errors, indent=2)+'\n')


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('bits',[32,64])
def test_native_field_precision_matches_completed_checkpoint(tmp_path,backend,bits):
    args = arguments(tmp_path)
    if bits==32:
        setting='ASTR_SLICE_FP32_EXE' if backend=='cpu' else 'ASTR_SLICE_FP32_GPU_EXE'
        args.executable = Path(os.environ.get(setting, ROOT/f'build_native_fp0_fp32_{backend}/bin/astr')).resolve()
    fields = ['density','velocity_x','velocity_y','velocity_z','pressure','temperature','Q_rs']
    case, _ = run_case(args, ROOT, backend, 2, f'native{bits}', 4, checkpoint_interval=4,
        product_catalog_records=[product(fields=fields)])
    frame = frames(segments(case)[0])[-1]
    mapping = dict(density=6,velocity_x=7,velocity_y=8,velocity_z=9,pressure=10,temperature=11)
    with h5py.File(frame) as data, h5py.File(case/'outdat/new/checkpoints/step000000000004/state.h5') as state:
        for name, component in mapping.items():
            assert data[name].dtype == np.dtype(f'float{bits}')
            expected = state[f'q{component:04d}'][8,:,:]
            np.testing.assert_array_equal(data[name],expected,err_msg=name)
        assert data['Q_rs'].dtype == np.dtype(f'float{bits}') and np.isfinite(data['Q_rs']).all()
    with h5py.File(segments(case)[0]/'geometry.h5') as geometry:
        assert geometry['coordinates'].dtype == np.dtype(f'float{bits}')
    xml = ET.parse(segments(case)[0]/'series.xdmf')
    assert {item.attrib['Precision'] for item in xml.findall('.//DataItem')} == {str(bits//8)}


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_air5_existing_solver_sample_means(tmp_path,backend):
    from run_output_air5_restart_validation import launch
    args=arguments(tmp_path);args.case='hbl';args.axis='z';args.top_mode='characteristic'
    args.reconstruction=5;args.mean_statistics=True;args.sample_interval=2
    r=product('sample-reynolds',['density','velocity_x','velocity_y','velocity_z','pressure'],'reynolds')
    f=product('sample-favre',['velocity_x','velocity_y','velocity_z','temperature'],'favre')
    for definition in (r,f):
        definition['statistics_source']='solver-samples'
        definition['schedule']['interval']=1
    case,_=launch(args,backend,2,'sample-means',4,interval=4,product_catalog_records=[r,f])
    checkpoint=case/'outdat/new/checkpoints/step000000000004/statistics.h5'
    with h5py.File(checkpoint) as statistics:
        for segment in segments(case):
            mean=frames(segment)[-1]
            with h5py.File(mean) as frame:
                measure=frame.attrs['source_measure']
                if isinstance(measure,bytes):
                    measure=measure.decode()
                samples=float(measure.split('samples=')[1].split(';')[0])
                count=int(statistics['metadata'][1])
                assert samples==count and count>0
                density=statistics['q0001'][8,:,:]
                if segment.parent.parent.name=='sample-reynolds':
                    mapping={'density':1,'velocity_x':28,'velocity_y':29,'velocity_z':30,'pressure':5}
                    for field,slot in mapping.items():
                        np.testing.assert_array_equal(frame[field],statistics[f'q{slot:04d}'][8,:,:]/count)
                else:
                    mapping={'velocity_x':2,'velocity_y':3,'velocity_z':4,'temperature':6}
                    for field,slot in mapping.items():
                        np.testing.assert_array_equal(frame[field],statistics[f'q{slot:04d}'][8,:,:]/density)


@pytest.mark.skipif(not os.environ.get('ASTR_SLICE_CATALYST_EXE'),reason='Select a Catalyst-enabled root build explicitly')
def test_numerical_slices_coexist_with_explicit_legacy_rendering(tmp_path):
    from test_output_insitu_restart import configuration
    args=arguments(tmp_path)
    args.executable=Path(os.environ['ASTR_SLICE_CATALYST_EXE']).resolve()
    common=dict(checkpoint_interval=4,insitu_config=configuration(render=True,statistics=False,interval=4))
    reference,_=run_case(args,ROOT,'gpu',2,'render-only',4,**common)
    observed,_=run_case(args,ROOT,'gpu',2,'render-plus-data',4,
        product_catalog_records=[product(fields=['density','pressure'])],**common)
    compare_fields(reference/'outdat/new/checkpoints/step000000000004/state.h5',
                   observed/'outdat/new/checkpoints/step000000000004/state.h5')
    images=sorted((reference/'outdat/render').glob('*.jpeg'))
    assert images
    from PIL import Image
    for expected in images:
        with Image.open(expected) as a,Image.open(observed/'outdat/render'/expected.name) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    assert len(frames(segments(observed)[0]))==2


def test_physical_time_slice_schedule(tmp_path):
    args=arguments(tmp_path)
    definition=product(fields=['pressure'])
    definition['schedule']=dict(mode='time',interval=.003)
    case,_=run_case(args,ROOT,'gpu',2,'physical-time',12,enabled=False,product_catalog_records=[definition])
    assert [int(path.stem[4:]) for path in frames(segments(case)[0])]==[0,3,6,9,12]


def test_existing_adaptive_window_controls_slice_schedule(tmp_path):
    args=arguments(tmp_path)
    definition=product(fields=['pressure'])
    definition['schedule']=dict(mode='steps',interval=5,windows=['burst'],dense_interval=1)
    window="""&adaptive_output
 enabled=t,window_ids='burst',window_modes='steps',window_steps(:,1)=3,7
/
"""
    common=dict(product_catalog_records=[definition],adaptive_config=window,checkpoint_interval=5)
    case,_=run_case(args,ROOT,'gpu',2,'adaptive-window',12,**common)
    assert [int(path.stem[4:]) for path in frames(segments(case)[0])]==[0,3,4,5,6,11]
    prefix,_=run_case(args,ROOT,'gpu',2,'adaptive-prefix',5,**common)
    resumed,_=run_case(args,ROOT,'gpu',2,'adaptive-resumed',12,
        restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    assert [int(path.stem[4:]) for path in frames(segments(resumed)[0])]==[6,11]
    compare_fields(case/'outdat/new/checkpoints/step000000000012/state.h5',
                   resumed/'outdat/new/checkpoints/step000000000012/state.h5')
