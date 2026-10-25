"""Offline report arithmetic only; no solver, GPU or figure rendering."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/gpu_validation'))
from plot_native_precision_summary import precision_summary


def source(directory):
    pairs = {}
    for scheme in ('643e', '643c'):
        pairs[scheme] = {name: {} for name in ('kenergy', 'enstrophy', 'dissipation')}
        for kind, values in (('fp64', (1., 2.)), ('fp32', (1.25, 2.))):
            table = np.column_stack(([40, 80], [0.01, 0.02], [0.01, 0.02],
                                     values, values, values))
            np.savetxt(directory/f'{scheme}_{kind}_curves.csv', table, delimiter=',',
                header='step,native_time,nominal_time,kenergy,enstrophy,dissipation', comments='')
    (directory/'accuracy.json').write_text(json.dumps(dict(pairs=pairs)))


def test_relative_curve_difference_is_not_peak_difference(tmp_path):
    source(tmp_path)
    result = precision_summary(tmp_path)
    for pair in result['pairs'].values():
        for metric in pair.values():
            assert metric['max_pointwise_relative_percent']==25.
            assert metric['max_pointwise_relative_step']==40
            assert metric['peak_relative_percent']==0.
            assert metric['final_fp64']==metric['final_fp32']==2.


def test_different_sample_identity_is_rejected(tmp_path):
    source(tmp_path)
    path = tmp_path/'643e_fp32_curves.csv'
    path.write_text(path.read_text().replace('4.000000000000000000e+01', '4.100000000000000000e+01'))
    with pytest.raises(ValueError, match='sample identities'):
        precision_summary(tmp_path)


def test_nonpositive_reference_has_no_fabricated_relative_error(tmp_path):
    source(tmp_path)
    path = tmp_path/'643e_fp64_curves.csv'
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    data[0, 3] = 0.
    np.savetxt(path, data, delimiter=',',
        header='step,native_time,nominal_time,kenergy,enstrophy,dissipation', comments='')
    with pytest.raises(ValueError, match='positive reference'):
        precision_summary(tmp_path)
