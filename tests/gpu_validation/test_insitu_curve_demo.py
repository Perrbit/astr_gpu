"""Q=0 and instantaneous physical streamlines before admitting larger grids."""
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest

from insitu_curve_boundary_reference import completed_wall_fields
from run_output_restart_validation import run_case, compare_fields
from test_insitu_curve_derivatives import ROOT, FINAL, reference
from test_insitu_device_curve_surface import configuration as surface_config, check_oracles
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_is4 import compare_numerical
from test_insitu_x4_observation import no_large_io

PRODUCTS = ('q_surface', 'instantaneous_streamlines')


def configuration(pipeline, transport):
    return surface_config(pipeline, transport).replace("products='q_surface'", "products='curve_demo'")


def scale_configuration(pipeline, transport):
    return configuration(pipeline, transport).replace(
        'device_budget_bytes=2147483648', 'device_budget_bytes=6442450944').replace(
        'host_budget_bytes=4294967296', 'host_budget_bytes=17179869184').replace(
        'device_reserve_bytes=1073741824', 'device_reserve_bytes=2147483648')


def check_frames(case, ranks, pipeline, steps, audit=True):
    log = (case/'run.log').read_text()
    audits = re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=(\d+) step=(\d+) product=(\S+) '
                        r'field_maxabs=(\S+) display_maxabs=(\S+) points=(\d+)', log)
    selected = [row for row in audits if row[2] in PRODUCTS]
    assert len(selected) == (ranks*len(steps)*2 if audit else 0), log
    assert all(np.isfinite(float(row[3])) and float(row[3]) <= 2e-10 and float(row[4]) == 0.
               for row in selected), selected
    for step in steps:
        for rank in range(ranks):
            record = json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['rendering_pipeline'] == pipeline and record['geometry_host_bytes'] == 0
            assert set(record['products']) == set(PRODUCTS)
        for product in PRODUCTS:
            path = case/f'outdat/render/{product}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as source:
                pixels = np.asarray(source.convert('RGB'))
                assert pixels.shape == (960, 1280, 3)
                assert np.count_nonzero(np.any(pixels < 220, axis=-1)) > 500
                # Fixed camera leaves this upper-right strip outside the geometry.
                # Check rendered color bar and labels, not proxy Visibility alone.
                legend = pixels[:450, 1150:]
                assert np.count_nonzero(np.ptp(legend, axis=-1) > 40) > 3000
                assert np.count_nonzero(np.all(legend < 150, axis=-1)) > 500
    assert not list((case/'outdat/render').glob('*.vtp'))


@pytest.mark.parametrize('mapping,pipeline,mode', [('periodic', 'standard-device', 'pinned'),
                                                 ('y-wavy', 'direct-device', 'device-aware')])
@pytest.mark.skipif(os.environ.get('ASTR_INSITU_X4_OBSERVE') != '1', reason='Explicit opt-in after numerical gates')
def test_qzero_transfers(tmp_path, mapping, pipeline, mode, record_property):
    from test_insitu_device_curve_trace_observation import transfer_ledger
    args = current_arguments(tmp_path, samples=False)
    args.runtime_timeout_seconds = 300
    case, _ = run_case(args, ROOT, 'gpu', 2, 'trace', 2,
        grid='32,32,32', tgv_mapping=mapping, enabled=False, checkpoint_enabled=False,
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, nsys_trace=True, pixel_audit=True)
    check_frames(case, 2, pipeline, (2,))
    no_large_io(case)
    record_property('transfer_ledger', json.dumps(transfer_ledger(case, 2, image_size=(1280, 960))))


def test_completed_qzero_transfer_receipt(record_property):
    from test_insitu_device_curve_trace_observation import transfer_ledger
    selected = os.environ.get('ASTR_INSITU_CURVE_DEMO_TRACE_REUSE')
    if not selected:
        pytest.skip('Select one immutable completed trace for parser-only reanalysis')
    case = Path(selected)
    config = (case/'datin/input.insitu').read_text()
    assert "products='curve_demo'" in config
    pipeline = re.search(r"rendering_pipeline='([^']+)'", config)[1]
    check_frames(case, 2, pipeline, (2,))
    no_large_io(case)
    record_property('transfer_ledger', json.dumps(transfer_ledger(case, 2, image_size=(1280, 960))))


@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
@pytest.mark.parametrize('pipeline,mode', [('standard-device', 'pinned'), ('direct-device', 'device-aware')])
def test_qzero_reference_restart(tmp_path, mapping, pipeline, mode, record_property):
    args = current_arguments(tmp_path, samples=False)
    args.runtime_timeout_seconds = 300
    kwargs = dict(grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1)
    on, _ = run_case(args, ROOT, 'gpu', 2, 'qzero', 4,
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True, **kwargs)
    check_frames(on, 2, pipeline, (2, 4))
    ref = reference if mapping == 'periodic' else completed_wall_fields
    record_property('independent_qzero_maxabs', check_oracles(on, 2, on, ref, iso=0.))
    cpu, _ = run_case(current_arguments(tmp_path, 'cpu', samples=False), ROOT, 'cpu', 2, 'cpu', 4, **kwargs)
    record_property('cpu_gpu_state_maxabs', compare_numerical(cpu/FINAL/'state.h5', on/FINAL/'state.h5'))
    record_property('cpu_gpu_qzero_maxabs', check_oracles(on, 2, cpu, ref, iso=0.))
    off, _ = run_case(args, ROOT, 'gpu', 2, 'off', 4, **kwargs)
    compare_fields(on/FINAL/'state.h5', off/FINAL/'state.h5')
    resumed, _ = run_case(args, ROOT, 'gpu', 2, 'resumed', 4,
        restore=on/'outdat/new/checkpoints/step000000000003',
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True, **kwargs)
    check_frames(resumed, 2, pipeline, (4,))
    compare_fields(on/FINAL/'state.h5', resumed/FINAL/'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (on/FINAL/name).read_bytes() == (resumed/FINAL/name).read_bytes()
    for product in PRODUCTS:
        for extension in ('jpeg', 'eps'):
            name = f'outdat/render/{product}.step00000004.{extension}'
            assert (on/name).read_bytes() == (resumed/name).read_bytes()


@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
@pytest.mark.parametrize('cells', [32, 64, 128, 256])
def test_qzero_memcheck(tmp_path, mapping, cells):
    args = current_arguments(tmp_path, samples=False)
    args.runtime_timeout_seconds = 1800 if cells == 256 else 300
    if cells >= 64:
        args.scale_timestep, args.maximum_cfl = 2e-5, .5
        args.directory_budget_bytes = (4 if cells == 256 else 2)*1024**3
    case, _ = run_case(args, ROOT, 'gpu', 2, 'memcheck', 2,
        grid=','.join([str(cells)]*3), tgv_mapping=mapping, enabled=False, checkpoint_enabled=False,
        insitu_config=(configuration if cells == 32 else scale_configuration)('direct-device', 'device-aware'),
        postprocess_transport='device-aware', resident_audit=True, memcheck=True,
        curve_wall_temperature=1. if mapping == 'y-wavy' and cells >= 64 else None)
    check_frames(case, 2, 'direct-device', (2,))
    no_large_io(case)
    logs = list(case.glob('memcheck.*.log'))
    assert len(logs) == 2
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)


@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
@pytest.mark.parametrize('pipeline,mode', [('standard-device', 'pinned'), ('direct-device', 'device-aware')])
@pytest.mark.parametrize('cells', [64, 128, 256])
def test_qzero_scale_reference_restart(tmp_path, mapping, pipeline, mode, cells, record_property):
    args = current_arguments(tmp_path, samples=False)
    args.runtime_timeout_seconds = 1800 if cells == 256 else 600
    args.directory_budget_bytes = (16 if cells == 256 else 2)*1024**3
    args.output_host_budget_bytes = (2048 if cells == 256 else 256 if cells == 128 else 64)*1024**2
    args.scale_timestep, args.maximum_cfl = 2e-5, .5
    # The large diagnostic needs only its step-3 restart source and final state.
    kwargs = dict(grid=','.join([str(cells)]*3), tgv_mapping=mapping,
                  checkpoint_interval=3 if cells == 256 else 1,
                  curve_wall_temperature=1. if mapping == 'y-wavy' else None)
    config = scale_configuration(pipeline, mode)
    on, size = run_case(args, ROOT, 'gpu', 2, f'qzero{cells}', 4,
        insitu_config=config, postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True, **kwargs)
    check_frames(on, 2, pipeline, (2, 4))
    ref = reference if mapping == 'periodic' else completed_wall_fields
    record_property('independent_qzero_maxabs', check_oracles(on, 2, on, ref, iso=0.))
    record_property('directory_bytes', size)
    cpu_args = current_arguments(tmp_path, 'cpu', samples=False)
    cpu_args.runtime_timeout_seconds = args.runtime_timeout_seconds
    cpu_args.directory_budget_bytes = args.directory_budget_bytes
    cpu_args.output_host_budget_bytes = args.output_host_budget_bytes
    cpu_args.scale_timestep, cpu_args.maximum_cfl = args.scale_timestep, args.maximum_cfl
    cpu, _ = run_case(cpu_args, ROOT, 'cpu', 2, f'cpu{cells}', 4, **kwargs)
    record_property('cpu_gpu_state_maxabs', compare_numerical(cpu/FINAL/'state.h5', on/FINAL/'state.h5'))
    record_property('cpu_gpu_qzero_maxabs', check_oracles(on, 2, cpu, ref, iso=0.))
    off, _ = run_case(args, ROOT, 'gpu', 2, f'off{cells}', 4, **kwargs)
    compare_fields(on/FINAL/'state.h5', off/FINAL/'state.h5')
    resumed, _ = run_case(args, ROOT, 'gpu', 2, f'resumed{cells}', 4,
        restore=on/'outdat/new/checkpoints/step000000000003',
        insitu_config=config, postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True, **kwargs)
    check_frames(resumed, 2, pipeline, (4,))
    compare_fields(on/FINAL/'state.h5', resumed/FINAL/'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (on/FINAL/name).read_bytes() == (resumed/FINAL/name).read_bytes()
    for product in PRODUCTS:
        for extension in ('jpeg', 'eps'):
            name = f'outdat/render/{product}.step00000004.{extension}'
            assert (on/name).read_bytes() == (resumed/name).read_bytes()


@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('cells', [64, 128, 256])
@pytest.mark.skipif(os.environ.get('ASTR_INSITU_X4_OBSERVE') != '1', reason='Numerical scale gate first')
def test_qzero_scale_observation(tmp_path, mapping, pipeline, mode, cells, record_property):
    from test_insitu_device_curve_trace_observation import transfer_ledger
    args = current_arguments(tmp_path, samples=False)
    args.runtime_timeout_seconds = 600
    args.directory_budget_bytes = (4 if cells == 256 else 2)*1024**3
    args.scale_timestep, args.maximum_cfl = 2e-5, .5
    args.resource_device_extra_budget_bytes = 6*1024**3
    args.resource_host_extra_budget_bytes = 16*1024**3
    args.resource_device_reserve_bytes = 2*1024**3
    args.nsys_trace_domains = 'cuda,nvtx,mpi'
    kwargs = dict(grid=','.join([str(cells)]*3), tgv_mapping=mapping, enabled=False, checkpoint_enabled=False,
                  postprocess_transport=mode, curve_wall_temperature=1. if mapping == 'y-wavy' else None)
    config = scale_configuration(pipeline, mode)
    off, _ = run_case(args, ROOT, 'gpu', 2, f'off{cells}', 2, monitor_resources=True,
        insitu_config='&insitu_run enabled=f /\n', **kwargs)
    baseline = json.loads((off/'resources.sampled.json').read_text())
    on, _ = run_case(args, ROOT, 'gpu', 2, f'on{cells}', 2, monitor_resources=True,
        resource_baseline=baseline, insitu_config=config, resident_audit=True, **kwargs)
    check_frames(on, 2, pipeline, (2,))
    resources = json.loads((on/'resources.sampled.json').read_text())
    assert resources['samples'] > 0 and resources['sampling_period_seconds'] == .02
    assert resources['additional_host_peak_difference_bytes'] <= 16*1024**3
    assert all(n <= 6*1024**3 for n in resources['additional_device_peak_difference_bytes'].values())
    assert all(d['min_free_bytes'] >= 2*1024**3 for d in resources['devices'].values())
    trace, _ = run_case(args, ROOT, 'gpu', 2, f'trace{cells}', 2, insitu_config=config,
        nsys_trace=True, pixel_audit=True, resident_audit=True, **kwargs)
    check_frames(trace, 2, pipeline, (2,))
    for case in (off, on, trace):
        no_large_io(case)
    record_property('external_20ms_resources', json.dumps(resources))
    record_property('transfer_ledger', json.dumps(transfer_ledger(trace, 2, image_size=(1280, 960))))
