"""Completed-step point monitors reuse the numerical slice providers."""
import csv
import json

import h5py
import numpy as np
import pytest

from test_index_slice_runtime import ROOT, arguments, frames, product, segments
from run_output_restart_validation import compare_fields, run_case


def probe(name='point'):
    return dict(id=name, kind='probe', enable_request=1, source='instantaneous',
                fields=['velocity_x', 'pressure', 'velocity_gradient_xx'],
                sampling=dict(mode='grid_node', indices=[8, 6, 8]),
                output=dict(channel='data', format='csv'),
                schedule=dict(mode='steps', interval=4))


def read_rows(case):
    paths = list((case / 'outdat/new/monitor_products').glob('*/*/segment*/samples.csv'))
    assert len(paths) == 1
    lines = paths[0].read_text().splitlines()
    assert 'source_bytes=8' in lines[0] and 'phase=completed_step' in lines[0]
    return list(csv.DictReader(line for line in lines if not line.startswith('#')))


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_probe_matches_same_state_archive_and_does_not_change_flow(tmp_path, backend):
    args = arguments(tmp_path)
    point = probe()
    plane = product(fields=point['fields'])
    baseline, _ = run_case(args, ROOT, backend, 2, 'off', 4)
    observed, _ = run_case(args, ROOT, backend, 2, 'with-point', 4,
                           product_catalog_records=[point, plane])
    compare_fields(baseline / 'outdat/new/checkpoints/step000000000004/state.h5',
                   observed / 'outdat/new/checkpoints/step000000000004/state.h5')
    rows = read_rows(observed)
    assert [int(row['step']) for row in rows] == [0, 4]
    for row, frame in zip(rows, frames(segments(observed)[0]), strict=True):
        with h5py.File(frame) as values:
            for field in point['fields']:
                np.testing.assert_array_equal(float(row[field]), values[field][6, 8])
            assert float(row['time']) == values['clock'][0]
    with h5py.File(segments(observed)[0] / 'geometry.h5') as values:
        np.testing.assert_array_equal([float(rows[0][a]) for a in 'xyz'], values['coordinates'][6, 8])
    status = json.loads((observed / 'outdat/new/products.status.json').read_text())
    assert all(item['state'] == 'active' for item in status['products'])


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_probe_exact_continuation_uses_saved_definition_and_clock(tmp_path, backend):
    args = arguments(tmp_path)
    common = dict(checkpoint_interval=5, product_catalog_records=[probe()])
    continuous, _ = run_case(args, ROOT, backend, 2, 'continuous', 12, **common)
    first, _ = run_case(args, ROOT, backend, 2, 'first', 5, **common)
    conflict = probe()
    conflict['sampling']['indices'] = [1, 1, 1]
    resumed, _ = run_case(args, ROOT, backend, 2, 'resumed', 12, checkpoint_interval=5,
                          restore=first / 'outdat/new/checkpoints/step000000000005',
                          product_catalog_records=[conflict])
    expected, actual = read_rows(continuous), read_rows(resumed)
    assert [int(row['step']) for row in actual] == [8, 12]
    assert actual == expected[2:]
    compare_fields(continuous / 'outdat/new/checkpoints/step000000000012/state.h5',
                   resumed / 'outdat/new/checkpoints/step000000000012/state.h5')
