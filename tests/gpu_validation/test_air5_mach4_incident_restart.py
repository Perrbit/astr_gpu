import h5py
import numpy as np
import pytest

from prepare_air5_mach4_case import prepare as precursor, seed_profile
from prepare_air5_mach4_incident_restart import prepare


def baseline(tmp_path):
    case = tmp_path/'hbl'
    precursor(case, grid='7,7,7')
    _, meta = seed_profile()
    (case/'outdat').mkdir(exist_ok=True)
    (case/'outdat/auxiliary.txt').write_text('fixture\n')
    with h5py.File(case/'outdat/flowfield.h5', 'w') as f:
        f['air5_compensation_version'] = 1
        f['nstep'] = 3000
        f['time'] = 4.09e-6
        for i, value in enumerate(meta['upstream_q'], 1):
            f[f'acq{i:02}'] = np.full((8, 8, 8), value)
            f[f'acc{i:02}'] = np.full((8, 8, 8), np.spacing(value)/4)
    return case


def test_incident_preserves_checkpoint_and_matches_approved_jump(tmp_path):
    case = baseline(tmp_path)
    original = (case/'outdat/flowfield.h5').read_bytes()
    out = tmp_path/'incident'
    meta = prepare(case, out)
    assert (out/'outdat/flowfield.h5').read_bytes() == original
    assert (case/'outdat/flowfield.h5').read_bytes() == original
    assert 'air5sbli\n' in (out/'datin/input.air5_c4').read_text()
    assert meta['upstream_mach'] == pytest.approx(4)
    assert meta['pressure'][1] == pytest.approx(63346.3128585)
    assert meta['max_scaled_normal_flux_residual'] < 2e-12
    assert 0 < meta['geometric_wall_intersection_x'] < meta['domain'][0]
    with pytest.raises(FileExistsError):
        prepare(case, out)


def test_mismatched_edge_rejected_without_output(tmp_path):
    case = baseline(tmp_path)
    with h5py.File(case/'outdat/flowfield.h5', 'r+') as f:
        f['acq02'][0, -1, 0] *= 1.01
    with pytest.raises(ValueError, match='top edge'):
        prepare(case, tmp_path/'incident')
    assert not (tmp_path/'incident').exists()
