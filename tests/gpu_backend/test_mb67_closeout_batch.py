"""Targeted offline checks of the closeout reader and approved trajectory policy."""
import struct

import pytest
from pathlib import Path

import run_mb67_closeout_batch as batch
from run_mb67_closeout_batch import compare_snapshot, compare_sensor, monitor, wall_curves


def snapshot(path, energy):
    path.write_bytes(struct.pack('<5i', 1, 1, 1, 0, 5)+struct.pack('<40d', *([1.]*8+[0.]*24+[energy]*8)))


def test_mixed_absolute_threshold_is_not_a_precision_fallback(tmp_path):
    reference = tmp_path/'reference.bin'; actual = tmp_path/'actual.bin'
    snapshot(reference, 1.); snapshot(actual, 1.+1e-6)
    result = compare_snapshot(actual, reference, 'fp64', 2e-6)
    assert 9e-7 < result['max_abs'][4] < 1.1e-6
    with pytest.raises(AssertionError, match='absolute gate'):
        compare_snapshot(actual, reference, 'fp64', 1e-7)


def test_raw_sensor_difference_does_not_hide_integer_mask_changes(tmp_path):
    reference = tmp_path/'reference.bin'; actual = tmp_path/'actual.bin'
    reference.write_bytes(struct.pack('<3idB', 0, 0, 0, .1, 1))
    actual.write_bytes(struct.pack('<3idB', 0, 0, 0, .9, 1))
    assert compare_sensor(actual, reference) == pytest.approx(.8)
    actual.write_bytes(struct.pack('<3idB', 0, 0, 0, .9, 0))
    with pytest.raises(AssertionError, match='masks differ'):
        compare_sensor(actual, reference)


def test_monitor_separates_all_rates_from_identity_flags(tmp_path):
    (tmp_path/'run.log').write_text('ASTR_AP_MONITOR step=1 time=1e-3 kinetic_energy=.1 rates=1 2 3 4 5 6 7 8 '
                                  'active=T F F F F F F F windows=1 0 0 0 0 0 0 0 wall_seconds=2e-4\n')
    row, = monitor(tmp_path)
    assert row['rates'] == list(range(1, 9))
    assert row['active'] == ['T']+['F']*7 and row['windows'] == ['1']+['0']*7


def test_wall_product_lookup_uses_actual_version_segment_layout(tmp_path):
    path = tmp_path/'outdat/new/wall_products/lower-pressure/version0001/segmentABCD/frame.step000000000002.csv'
    path.parent.mkdir(parents=True); path.write_text('step,time,value\n2,1e-3,1\n')
    assert wall_curves(tmp_path) == {'lower-pressure/frame.step000000000002.csv': path}


def test_quiet_does_not_parse_an_unselected_insitu_file(tmp_path, monkeypatch):
    template = tmp_path/'template'; (template/'datin').mkdir(parents=True)
    (template/'datin/input.run').write_text('&run_control\n max_steps=4\n/\n')
    (template/'datin/input.case').write_text('&case_options\n use_gpu=.true.\n/\n')
    (template/'datin/input.output').write_text('&checkpoint\n enabled=.true.\n/\n&volume\n enabled=.true.\n/\n&slices\n enabled=.false.\n/\n')
    unused = '&insitu_run enabled=f /\n'
    (template/'datin/input.insitu').write_text(unused)
    family = dict(precision='fp64', topology=[1, 1, 1], transport='pinned', backend='cuda', steps=4, catalog=False)
    def command(argv, cwd, env, log):
        log.write_text('ASTR_GPU_CPP rank=0 backend=cuda device=0\n')
        assert 'ASTR_INSITU_CONFIG' not in env
    monkeypatch.setattr(batch, 'command', command)
    case = batch.run_case(tmp_path/'build', template, tmp_path/'quiet', family, 'quiet', Path('/unused/launcher'))
    assert (case/'datin/input.insitu').read_text() == unused
