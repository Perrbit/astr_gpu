"""Bounded TGV seed preset: solver isolation, resident products and exact restart."""
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_output_insitu_restart import ROOT, arguments
from test_insitu_device_products import resident_configuration, field_difference


@pytest.mark.parametrize('ranks,pipeline,mode,mapping', [
    (1, 'standard-device', 'pinned', None),
    (2, 'direct-device', 'device-aware', None),
    (2, 'direct-device', 'device-aware', 'periodic')])
def test_stratified_seeds_runtime(tmp_path, ranks, pipeline, mode, mapping, record_property):
    args = arguments(tmp_path)
    args.directory_budget_bytes = 128*1024**2
    args.runtime_timeout_seconds = 300
    config = resident_configuration(pipeline, mode, statistics=True, interval=2, profile='streamlines')
    config = config.replace('&insitu_run\n',
        "&insitu_run\n streamline_seeds='tgv-stratified',mean_streamline_render=t,\n")
    settings = dict(grid='32,32,32', insitu_config=config, checkpoint_interval=3,
                    tgv_mapping=mapping, postprocess_transport=mode, resident_audit=True)
    actual, size = run_case(args, ROOT, 'gpu', ranks, 'stratified', 4, **settings)
    final = 'outdat/new/checkpoints/step000000000004'
    plain, _ = run_case(args, ROOT, 'gpu', ranks, 'off', 4, grid='32,32,32',
                        tgv_mapping=mapping, checkpoint_interval=3)
    compare_fields(actual/final/'state.h5', plain/final/'state.h5')
    cpu_args = SimpleNamespace(**vars(args))
    cpu_args.executable = Path(os.environ.get('ASTR_OUTPUT_CPU_EXE', ROOT/'build_insitu_check/bin/astr'))
    cpu, _ = run_case(cpu_args, ROOT, 'cpu', ranks, 'cpu_reference', 4, grid='32,32,32',
                      tgv_mapping=mapping, checkpoint_interval=3)
    cpu_gpu_error = field_difference(actual/final/'state.h5', cpu/final/'state.h5', 2e-10)
    baseline_config = config.replace('tgv-stratified', 'line16')
    baseline, _ = run_case(args, ROOT, 'gpu', ranks, 'line16', 4,
                          **dict(settings, insitu_config=baseline_config))
    for name in ('state.h5', 'statistics.h5'):
        field_difference(actual/final/name, baseline/final/name)
    source = actual/'outdat/new/checkpoints/step000000000003'
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'stratified_resumed', 4, restore=source, **settings)
    for name in ('state.h5', 'statistics.h5'):
        field_difference(actual/final/name, resumed/final/name)
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual/final/name).read_bytes() == (resumed/final/name).read_bytes()
    for product in ('instantaneous_streamlines', 'crossing_streamlines',
                    'mean_reynolds_streamlines', 'mean_favre_streamlines'):
        filename = f'{product}.step00000004.jpeg'
        assert (actual/'outdat/render'/filename).read_bytes() == (resumed/'outdat/render'/filename).read_bytes()
        assert (actual/'outdat/render'/filename).with_suffix('.eps').read_bytes() == \
            (resumed/'outdat/render'/filename).with_suffix('.eps').read_bytes()
        with Image.open(actual/'outdat/render'/filename) as image:
            pixels = np.asarray(image.convert('RGB'))
        assert pixels.shape == (600, 800, 3)
        assert np.count_nonzero(np.ptp(pixels[:550, :650].astype(int), axis=2) > 40) > 100
    log = (actual/'run.log').read_text()
    audits = re.findall(r'ASTR_INSITU_RESIDENT_AUDIT .*?field_maxabs=(\S+) display_maxabs=(\S+)', log)
    assert len(audits) == ranks*2*4, log
    assert all(np.isfinite(float(v)) and float(v) <= 2e-10 and float(display) == 0. for v, display in audits)
    rows = re.findall(r'ASTR_INSITU_RESIDENT_TRAJECTORY .*?product=(\S+).*?rounds=(\d+).*?'
        r'control_read_bytes=(\d+).*?owner_query_read_bytes=(\d+) seed_layout=(\S+) seeds=(\d+) particles=(\d+)', log)
    assert len(rows) == ranks*2*4, log
    for product, rounds, control, owners, layout, seeds, particles in rows:
        diagnostic = product == 'crossing_streamlines'
        assert (layout, int(seeds), int(particles)) == (('line16', 16, 16) if diagnostic else ('tgv-stratified', 256, 512))
        assert int(control) <= int(rounds)*int(particles)*64
        assert int(owners) <= (int(rounds)+1)*int(particles)*8
    for rank in range(ranks):
        for step in (2, 4):
            receipt = json.loads((actual/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['geometry_host_bytes'] == 0
            for product, description in receipt['products'].items():
                diagnostic = product == 'crossing_streamlines'
                assert description['seeding'] == dict(layout='line16' if diagnostic else 'tgv-stratified',
                    seeds=16 if diagnostic else 256, direction_trajectories=16 if diagnostic else 512)
    assert not list((actual/'outdat/render').rglob('*.vtp'))
    assert 'frame_downloads=0' in log
    # Render state identity must reject a different seed layout at restart.
    run_case(args, ROOT, 'gpu', ranks, 'mismatched_seeds', 4, restore=source,
             reject='native render configuration differs', **dict(settings, insitu_config=baseline_config))
    record_property('directory_bytes', size)
    record_property('cpu_gpu_state_maxabs', cpu_gpu_error)
    record_property('seed_layout', 'tgv-stratified: 8x8x4 fixed physical sites, 512 directions')


def test_stratified_seeds_memcheck(tmp_path, record_property):
    args = arguments(tmp_path)
    args.directory_budget_bytes = 64*1024**2
    args.runtime_timeout_seconds = 600
    config = resident_configuration('direct-device', 'pinned', statistics=False, interval=2, profile='streamlines')
    config = config.replace('&insitu_run\n', "&insitu_run\n streamline_seeds='tgv-stratified',\n")
    case, size = run_case(args, ROOT, 'gpu', 2, 'stratified_memcheck', 2, grid='32,32,32',
        checkpoint_enabled=False, insitu_config=config, postprocess_transport='pinned', memcheck=True)
    assert 'seed_layout=tgv-stratified seeds=256 particles=512' in (case/'run.log').read_text()
    record_property('directory_bytes', size)
