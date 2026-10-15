"""Actual CPU/GPU ROS-2 optional-history transaction and continuation probe."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference
from run_air5_state_workload import parse_results

ROOT = Path(__file__).resolve().parents[2]


def test_optional_history_transaction_and_continuation(tmp_path):
    probe = Path(os.environ.get('ASTR_CHEMISTRY_COMPENSATION_GPU_PROBE_EXE',
                                ROOT / 'build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe'))
    result = subprocess.run([str(probe), 'history-contract'], capture_output=True,
                            text=True, check=True, timeout=180)
    (tmp_path / 'history-contract.log').write_text(result.stdout + result.stderr)
    for name in ('FIRST_CALL_BASELINE_EXACT', 'INTERVAL_CLIP', 'FINAL_REMAINDER_NOT_REUSED',
                 'REJECTION_AND_PREFIX_ROLLBACK', 'INVALID_BITWISE_ROLLBACK',
                 'NONDEFAULT_CPU_GPU_CONTINUATION_EXACT'):
        assert f'HISTORY_{name}_PASS' in result.stdout


@pytest.mark.parametrize('segments,mode', [(1, 'reset'), (4, 'history-from-input')])
def test_cpu_state_batch_needs_no_gpu_and_resets_each_cell(tmp_path, segments, mode):
    model = Air5RadauReference(ROOT / 'chemMech/air5_kimjo12.json')
    rho = .05
    momentum = rho * np.array([2000., 0., 0.])
    species = rho * np.array([.767, .233, 0., 0., 0.])
    ev = model.ev_from_tv(species, 2000.)
    q5 = model.q5_from_state(rho, momentum, species, ev, 3000.)
    row = np.r_[rho, momentum, q5, species, ev, np.zeros(6)]
    if mode == 'history-from-input':
        row = np.r_[row, 1e-12]
    input_path, output_path = tmp_path / 'batch.in', tmp_path / 'batch.out'
    with input_path.open('w') as stream:
        stream.write('2 5e-11 0\n')
        np.savetxt(stream, np.tile(row, (2, 1)), fmt='%.17e')
    probe = Path(os.environ.get('ASTR_CHEMISTRY_COMPENSATION_GPU_PROBE_EXE',
                                ROOT / 'build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    subprocess.run([str(probe), 'batch-cpu', str(input_path), str(output_path),
                    '1e-11', '1e-15', str(segments), mode], env=env,
                   capture_output=True, text=True, check=True, timeout=180)
    parse_results(output_path, 2)
    values = np.loadtxt(output_path)
    np.testing.assert_array_equal(values[0, 1:], values[1, 1:])
    assert values[0, 2] > 0
    assert abs(np.sum(values[0, 6:11]) / rho - 1.) <= 128*np.finfo(float).eps


@pytest.mark.parametrize('backend', ['batch-cpu', 'batch'])
def test_explicit_component_tolerances_reproduce_legacy_batch(tmp_path, backend):
    model = Air5RadauReference(ROOT / 'chemMech/air5_kimjo12.json')
    rows, tolerances = [], []
    for rho in (.05, .07):
        momentum = rho * np.array([2000., 0., 0.])
        species = rho * np.array([.767, .233, 0., 0., 0.])
        ev = model.ev_from_tv(species, 2000.)
        q5 = model.q5_from_state(rho, momentum, species, ev, 3000.)
        rows.append(np.r_[rho, momentum, q5, species, ev, np.zeros(6)])
        tolerances.append(np.r_[np.full(5, 1e-13*rho), 1e-13*max(abs(ev), 1.)])
    input_path = tmp_path / 'batch.in'
    with input_path.open('w') as stream:
        stream.write('2 5e-11 0\n')
        np.savetxt(stream, rows, fmt='%.17e')
    tolerance_path = tmp_path / 'absolute_tolerances.in'
    with tolerance_path.open('w') as stream:
        stream.write('2\n')
        np.savetxt(stream, tolerances, fmt='%.17e')
    probe = Path(os.environ.get('ASTR_CHEMISTRY_COMPENSATION_GPU_PROBE_EXE',
                                ROOT / 'build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if backend == 'batch-cpu' else '0',
               ASTR_AIR5_SOURCE_EVALUATION='joint')
    outputs = []
    for name, extra in (('legacy', []), ('vector', [str(tolerance_path)])):
        output_path = tmp_path / f'{name}.out'
        subprocess.run([str(probe), backend, str(input_path), str(output_path), *extra],
                       env=env, capture_output=True, text=True, check=True, timeout=180)
        parse_results(output_path, 2)
        outputs.append(output_path.read_bytes())
    assert outputs[0] == outputs[1]
