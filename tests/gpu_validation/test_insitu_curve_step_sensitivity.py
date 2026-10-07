"""Approved half-step diagnostics, not a new trajectory accuracy threshold."""
import json
import re

import numpy as np
import pytest

from run_output_restart_validation import run_case
from test_insitu_curve_demo import ROOT, PRODUCTS, scale_configuration, check_frames
from test_insitu_device_curve_streamlines import read_trace_oracle
from test_insitu_device_curve_wall_fields import current_arguments


def terminal_states(text, scale, step=2):
    rows = re.findall(r'ASTR_INSITU_CURVE_STEP_SENSITIVITY step=(\d+) product=(\S+) scale=(\S+) '
        r'particle=(\d+) status=(\d+) x=(\S+) y=(\S+) z=(\S+) length=(\S+)', text)
    selected = [r for r in rows if int(r[0]) == step and r[1] == 'instantaneous_streamlines']
    if len(selected) != 32 or {int(r[3]) for r in selected} != set(range(32)):
        raise ValueError('Missing or repeated terminal particle identity')
    result = np.empty((32, 5))
    for r in selected:
        values = np.asarray([float(r[5]), float(r[6]), float(r[7]), float(r[8]), int(r[4])])
        if float(r[2]) != scale or not np.isfinite(values).all() or not 0 <= values[3] <= np.pi+2e-10:
            raise ValueError('Invalid sensitivity terminal state')
        if values[4] not in (1, 3, 10) or (values[4] == 3 and np.pi-values[3] >= .01*scale*(2*np.pi/32)):
            raise ValueError('Unapproved trajectory termination in sensitivity diagnostic')
        result[int(r[3])] = values
    return result


def trajectories(case, scale):
    groups = [[] for _ in range(32)]
    for rank in range(2):
        xyz, velocity, accepted, particle, edges = read_trace_oracle(
            case/f'outdat/render/curve_trace_oracle.instantaneous_streamlines.step2.rank{rank}.bin')
        assert np.isfinite(accepted).all() and np.isfinite(velocity).all()
        np.testing.assert_array_equal(xyz, accepted[:, :3])
        assert (particle >= 0).all() and (particle < 32).all()
        assert (edges >= 0).all() and (edges < len(xyz)).all()
        np.testing.assert_array_equal(particle[edges[:, 0]], particle[edges[:, 1]])
        ds = accepted[edges[:, 1], 3]-accepted[edges[:, 0], 3]
        assert (ds > 0).all() and (ds <= .5*scale*(2*np.pi/32)+2e-10).all()
        for pid in range(32):
            groups[pid].extend(accepted[particle == pid, :4])
    result = []
    for group in groups:
        points = np.asarray(sorted(group, key=lambda p: p[3]))
        assert len(points) >= 2 and abs(points[0, 3]) <= 2e-10
        ds = np.diff(points[:, 3])
        repeated = ds == 0.
        assert np.max(abs(np.diff(points[:, :3], axis=0)[repeated]), initial=0.) <= 2e-10
        points = points[np.r_[True, ~repeated]]
        assert np.diff(points[:, 3]).min() > 0.
        result.append(points)
    return result


def compare_paths(first, second):
    """Polyline sampling adds interpolation error; do not call it RK45 error."""
    errors = []
    for a, b in zip(first, second, strict=True):
        stop = min(a[-1, 3], b[-1, 3])
        s = np.unique(np.r_[a[a[:, 3] <= stop, 3], b[b[:, 3] <= stop, 3], stop])
        diff = np.stack([np.interp(s, a[:, 3], a[:, d])-np.interp(s, b[:, 3], b[:, d])
                         for d in range(3)], axis=-1)
        errors.append(float(np.linalg.norm(diff, axis=-1).max()))
    return errors


def test_path_comparison_is_arc_length_based():
    a = np.array([[0., 0., 0., 0.], [1., 0., 0., 1.]])
    b = np.array([[0., 0., 0., 0.], [.25, 0., 0., .25], [1., 0., 0., 1.]])
    assert compare_paths([a], [b]) == [0.]


@pytest.mark.parametrize('duplicate', [False, True])
def test_terminal_state_incomplete_or_duplicate_rejected(duplicate):
    row = 'ASTR_INSITU_CURVE_STEP_SENSITIVITY step=2 product=instantaneous_streamlines scale=1 '
    text = ''.join(row+f'particle={0 if duplicate else i} status=10 x=1 y=2 z=3 length=3.141592653589793\n'
                   for i in range(32 if duplicate else 31))
    with pytest.raises(ValueError, match='identity'):
        terminal_states(text, 1.)


@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
@pytest.mark.parametrize('cells', [64, 128])
def test_actual_state_half_step(tmp_path, mapping, cells, record_property):
    args = current_arguments(tmp_path, samples=False)
    args.runtime_timeout_seconds = 600
    args.directory_budget_bytes = 2*1024**3
    args.scale_timestep, args.maximum_cfl = 2e-5, .5
    states, paths, cases = [], [], []
    for scale in (1., .5):
        case, _ = run_case(args, ROOT, 'gpu', 2, f'scale{scale}', 2,
            grid=','.join([str(cells)]*3), tgv_mapping=mapping, enabled=False, checkpoint_enabled=False,
            insitu_config=scale_configuration('direct-device', 'device-aware'),
            postprocess_transport='device-aware', resident_audit=True, curve_trace_oracle=True,
            curve_trace_step_scale=scale)
        check_frames(case, 2, 'direct-device', (2,))
        assert not list((case/'outdat').rglob('*.h5'))
        assert 'runtime_zero_readback_evidence=0' in (case/'run.log').read_text()
        states.append(terminal_states((case/'run.log').read_text(), scale))
        paths.append(trajectories(case, scale))
        cases.append(str(case))
    report = dict(cells=cells, mapping=mapping, cases=cases,
        scope='Same solver steps and state; half initial/min/max physical RK45 steps, unchanged tolerance 1e-8',
        endpoint_maxabs=float(np.max(abs(states[0][:, :3]-states[1][:, :3]))),
        accepted_length_maxabs=float(np.max(abs(states[0][:, 3]-states[1][:, 3]))),
        changed_termination_particles=np.flatnonzero(states[0][:, 4] != states[1][:, 4]).tolist(),
        polyline_common_arc_maxnorm=compare_paths(*paths),
        original_terminal_states=states[0].tolist(), half_terminal_states=states[1].tolist(),
        note='Polyline difference includes linear chord interpolation; no new accuracy gate is inferred')
    (tmp_path/'sensitivity.json').write_text(json.dumps(report, indent=2)+'\n')
    record_property('step_sensitivity', json.dumps(report))
