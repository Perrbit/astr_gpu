"""TB2 fixed-j instantaneous-minus-local-time-mean fields, no three-dimensional archive."""
import json
import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_index_slice_runtime import ROOT, arguments, product, segments, frames
from test_insitu_selected_runtime import configuration

EXE = os.environ.get('ASTR_SELECTED_RUNTIME_EXE')
pytestmark = pytest.mark.skipif(not EXE, reason='Select the root-CMake-built selected-statistics executable')
FIELDS = ['velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature']
TOPOLOGIES = [(1,1,1), (2,1,1), (1,2,1), (1,1,2), (2,2,1), (2,1,2), (1,2,2)]


def definitions():
    raw = product('raw', ['density', *FIELDS])
    raw['sampling'] = dict(mode='grid_index', axis='j', index=4,
                           index_ranges=dict(i=[0,16], k=[0,16]))
    raw['schedule']['interval'] = 1
    records = [raw]
    for source in ('reynolds', 'favre'):
        mean = deepcopy(raw)
        mean.update(id=source, source=source, statistics_source='selected-time')
        mean['fields'] = FIELDS if source == 'reynolds' else FIELDS[:3]
        records.append(mean)
        fluct = deepcopy(mean)
        fluct.update(id=source+'-fluctuation', fluctuation=True)
        records.append(fluct)
    return records


def launch(tmp_path, backend, topology, name, steps=12, restore=None, enabled=True):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    records = definitions()
    if not enabled:
        records = records[:1]
    return run_case(args, ROOT, backend, int(np.prod(topology)), name, steps,
                    restore=restore, checkpoint_interval=5, topology=topology,
                    insitu_config=configuration(), product_catalog_records=records, insitu_timing=True)[0]


def check_frames(case):
    available = {path.parent.parent.name: path for path in segments(case)}
    assert set(available) == {'raw','reynolds','favre','reynolds-fluctuation','favre-fluctuation'}, (case/'run.log').read_text()
    raw = {p.name: p for p in frames(available['raw'])}
    worst = 0.
    for source in ('reynolds','favre'):
        mean = {p.name: p for p in frames(available[source])}
        observed = frames(available[source+'-fluctuation'])
        assert len(observed) == 12
        for path in observed:
            with h5py.File(path) as a, h5py.File(mean[path.name]) as m, h5py.File(raw[path.name]) as r:
                assert 'quantity=instantaneous_minus_local_accumulated_mean' in str(a.attrs['source_measure'])
                assert 'actual_sample_range=' in str(a.attrs['source_measure'])
                for field in FIELDS if source == 'reynolds' else FIELDS[:3]:
                    expected = r[field][...] - m[field][...]
                    error = np.abs(a[field][...] - expected)
                    scaled = float(np.max(error / np.maximum(1., np.maximum(np.abs(expected), np.abs(a[field][...])))))
                    worst = max(worst, scaled)
                    assert scaled <= 2e-10, (path,field,scaled)
                    assert np.isfinite(a[field][...]).all()
    return worst


@pytest.mark.parametrize('backend', ['cpu','gpu'])
@pytest.mark.parametrize('topology', TOPOLOGIES)
def test_fixed_j_local_fluctuations(tmp_path, backend, topology, record_property):
    case = launch(tmp_path, backend, topology, 'fluctuations')
    record_property('maximum_scaled_difference', check_frames(case))
    record_property('case_directory', str(case))
    record_property('backend', backend)
    record_property('topology', ','.join(map(str,topology)))


@pytest.mark.parametrize('backend', ['cpu','gpu'])
@pytest.mark.parametrize('topology', TOPOLOGIES)
def test_sbli_fixed_j_local_fluctuations(tmp_path, backend, topology, record_property):
    from test_insitu_sbli_statistics import launch as launch_sbli

    records = definitions()
    for item in records:
        item['sampling']['index_ranges'] = dict(i=[0,64], k=[0,16])
    case = launch_sbli(tmp_path/backend, backend, topology, records=records)
    record_property('maximum_scaled_difference', check_frames(case))
    record_property('case_directory', str(case))
    record_property('backend', backend)
    record_property('topology', ','.join(map(str,topology)))


@pytest.mark.parametrize('backend', ['cpu','gpu'])
def test_fluctuation_exact_restart_and_output_switch(tmp_path, backend):
    topology = (1,1,2)
    reference = launch(tmp_path, backend, topology, 'continuous')
    prefix = launch(tmp_path, backend, topology, 'prefix', steps=5)
    resumed = launch(tmp_path, backend, topology, 'resumed', restore=prefix/'outdat/new/checkpoints/step000000000005')
    disabled = launch(tmp_path, backend, topology, 'disabled', enabled=False)
    expected = reference/'outdat/new/checkpoints/step000000000012/state.h5'
    for case in (resumed, disabled):
        compare_fields(expected, case/'outdat/new/checkpoints/step000000000012/state.h5')
    compare_fields(reference/'outdat/new/checkpoints/step000000000012/statistics.h5',
                   resumed/'outdat/new/checkpoints/step000000000012/statistics.h5')
    originals = {p.parent.parent.name:p for p in segments(reference)}
    for segment in segments(resumed):
        for path in frames(segment):
            with h5py.File(path) as actual, h5py.File(originals[segment.parent.parent.name]/path.name) as wanted:
                for field in actual:
                    np.testing.assert_array_equal(actual[field], wanted[field], err_msg=f'{path}/{field}')
    status = json.loads((resumed/'outdat/new/products.status.json').read_text())
    assert all(row['state'] == 'active' for row in status['products'])


def analyse_matrix(record, output, fields=None):
    """Only read passing, complete evidence and compare all requested primitive components."""
    from analyse_station_profiles import restart_logs
    from analyse_selected_statistics import datasets
    import re

    cases = {}
    if fields is None:
        fields = ['density',*FIELDS]
    for log,backend in restart_logs([record]):
        text = log.read_text()
        match = re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)',text)
        if not match or 'The job is done!' not in text:
            raise ValueError(f'Incomplete solver identity: {log}')
        key = backend, tuple(map(int,match.groups()))
        if key in cases:
            raise ValueError(f'Duplicate identity: {key}')
        cases[key] = log.parent
    assert set(cases)=={(b,t) for b in ('cpu','gpu') for t in TOPOLOGIES}
    result = dict(comparisons=[])
    for topology in TOPOLOGIES:
        left,right = datasets(cases['cpu',topology]),datasets(cases['gpu',topology])
        assert left.keys()==right.keys()
        absolute = scaled = 0.
        pairs = 0
        for name in left:
            assert left[name].keys()==right[name].keys()
            for frame in left[name]:
                pairs += 1
                with h5py.File(left[name][frame]) as a, h5py.File(right[name][frame]) as b:
                    np.testing.assert_array_equal(a['clock'],b['clock'])
                    np.testing.assert_array_equal(a['field_ids'],b['field_ids'])
                    for field in fields:
                        if field not in a:
                            continue
                        x,y = a[field][...],b[field][...]
                        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                        error = np.abs(x-y)
                        absolute = max(absolute,float(error.max()))
                        scaled = max(scaled,float((error/np.maximum(1.,np.maximum(np.abs(x),np.abs(y)))).max()))
        assert scaled<=2e-10, (topology,scaled,absolute)
        result['comparisons'].append(dict(topology=topology,frame_pairs=pairs,max_scaled=scaled,max_absolute=absolute))
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def test_fluctuation_gpu_memcheck(tmp_path):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    case,_ = run_case(args, ROOT, 'gpu', 2, 'memcheck', 2, checkpoint_enabled=False,
                      insitu_config=configuration(), product_catalog_records=definitions(), memcheck=True)
    logs = list(case.glob('memcheck.*.log'))
    assert len(logs)==2
    assert all('ERROR SUMMARY: 0 errors' in log.read_text() for log in logs)


@pytest.mark.skipif(not os.environ.get('ASTR_FLUCTUATION_RENDER_EXE'), reason='Select the strict Catalyst/device executable')
@pytest.mark.parametrize('ranks,pipeline', [(1,'standard-device'), (2,'standard-device'), (2,'direct-device'),
                                         (4,'standard-device')])
def test_fluctuation_device_images(tmp_path, ranks, pipeline):
    from PIL import Image
    from test_insitu_device_products import resident_configuration
    from test_runtime_catalog_images import image_segments

    args = arguments(tmp_path)
    args.executable = Path(os.environ['ASTR_FLUCTUATION_RENDER_EXE']).resolve()
    config = resident_configuration(pipeline, profile='catalog_stats')
    records = image_definitions()
    case,_ = run_case(args, ROOT, 'gpu', ranks, 'images', 2, grid='32,32,32', checkpoint_enabled=False,
                      insitu_config=config, product_catalog_records=records, postprocess_transport='pinned',
                      topology=(2,2,1) if ranks==4 else (ranks,1,1))
    status = json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']), status
    paths = image_segments(case)
    assert len(paths)==4
    for segment in paths:
        images = list(segment.glob('*.jpeg'))
        assert len(images)==2
        for path in images:
            with Image.open(path) as image:
                assert image.size==(800,600)
                assert np.any(np.asarray(image.convert('RGB'))<230)
        receipts = [json.loads(path.read_text()) for path in segment.glob('frame.*.rank*.json')]
        assert len(receipts)==2*ranks
        for receipt in receipts:
            assert receipt['field_d2h_bytes']==receipt['geometry_d2h_bytes']==receipt['geometry_host_bytes']==0
            stats = receipt['statistics']
            assert stats['provider']=='selected-time' and stats['quantity']=='instantaneous_minus_local_accumulated_mean'
            assert stats['sampling_phase']=='completed_step_clipped_time' and stats['covered_duration']>0
            assert stats['actual_sample_range'][1]<=receipt['time']


def image_definitions():
    from test_insitu_catalog_products import visual_product

    records = []
    for source,field in [('reynolds','velocity_x'),('favre','velocity_x'),
                          ('reynolds','pressure'),('reynolds','temperature')]:
        item = visual_product(name=source+'-'+field)
        item.update(source=source, statistics_source='selected-time', fluctuation=True, field=field)
        item['sampling'] = dict(mode='grid_index', axis='j', index=4,
                               index_ranges=dict(i=[0,32], k=[0,32]))
        item['schedule']['interval'] = 1
        item['display'].update(camera=[np.pi,-10.,np.pi], focal_point=[np.pi,np.pi/4,np.pi],
                               view_up=[0.,0.,1.], parallel_scale=4., color_range=[-1e-4,1e-4])
        records.append(item)
    return records


@pytest.mark.skipif(not os.environ.get('ASTR_FLUCTUATION_RENDER_EXE'), reason='Select the strict Catalyst/device executable')
def test_fluctuation_image_cost_without_oracles(tmp_path):
    from test_insitu_device_products import resident_configuration
    import re

    args = arguments(tmp_path)
    args.executable = Path(os.environ['ASTR_FLUCTUATION_RENDER_EXE']).resolve()
    common = dict(grid='32,32,32', checkpoint_enabled=False, postprocess_transport='pinned',
                  insitu_config=resident_configuration('standard-device', profile='catalog_stats'),
                  insitu_timing=True, monitor_resources=True)
    off,_ = run_case(args, ROOT, 'gpu', 2, 'without-images', 2, product_catalog_records=[], **common)
    baseline = json.loads((off/'resources.sampled.json').read_text())
    on,_ = run_case(args, ROOT, 'gpu', 2, 'with-images', 2, product_catalog_records=image_definitions(),
                    resource_baseline=baseline, **common)
    result = dict(scope='one local 32^3 GPU NP2 x two-step window, four images per step; not a stable performance benchmark', cases={})
    for name,case in [('off',off),('on',on)]:
        assert not list((case/'outdat/new').rglob('*.h5')), 'no numerical oracle/checkpoint I/O in timing runs'
        values = {}
        for line in (case/'run.log').read_text().splitlines():
            match = re.fullmatch(r'ASTR_INSITU_STAGE_TIMING completed_step_inclusive (\d+) (\d+)\s+(\S+)',line)
            if match:
                values[tuple(map(int,match.group(1,2)))] = float(match[3])
        assert set(values)=={(s,r) for s in (1,2) for r in (0,1)}
        result['cases'][name] = dict(first_step_max_rank_seconds=max(values[1,r] for r in (0,1)),
            second_step_max_rank_seconds=max(values[2,r] for r in (0,1)),
            two_steps_max_rank_sum_seconds=max(sum(values[s,r] for s in (1,2)) for r in (0,1)),
            memory=json.loads((case/'resources.sampled.json').read_text()))
    assert all(row['state']=='active' for row in json.loads((on/'outdat/new/products.status.json').read_text())['products'])
    (tmp_path/'cost.json').write_text(json.dumps(result,indent=2)+'\n')
