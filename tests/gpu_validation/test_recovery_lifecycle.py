"""Exercise recovery before step checks, shared-node averaging and NSCBC RHS."""

import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from prepare_m12_local_case import prepare, replace_value
from run_m12_boundary_compare import active_state, configuration
from run_m12_mp7_nscbc_performance import runtime, mpi_command


PROBE = os.environ.get('ASTR_MP_WALL_PROBE')
EXE = os.environ.get('ASTR_M12_CRASHFIX_EXECUTABLE')
MPIEXEC = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'


def assert_finite_rhs(path):
    with path.open('rb') as stream:
        im,jm,km,nq = np.fromfile(stream, dtype='<i4', count=4)
        values = np.fromfile(stream, dtype='<f8')
    assert nq == 5 and values.size == (im+1)*(jm+1)*(km+1)*nq
    assert np.isfinite(values).all(), str(path)


def assert_positive_recovery_snapshots(case, expected):
    paths = list((case/'diagnostics').glob('state.post_crashfix.*.bin'))
    assert len(paths) == expected
    for path in paths:
        q = active_state(path)
        internal = q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
        assert np.isfinite(q).all() and q[...,0].min() > 0 and internal.min() > 0, str(path)


@pytest.mark.skipif(not PROBE, reason='set ASTR_MP_WALL_PROBE')
@pytest.mark.parametrize('mode,ranks', [('crashfix-check', 1), ('crashfix-check', 2),
    ('crashfix-check', 4), ('crashfix-filter', 1), ('crashfix-filter', 2), ('crashfix-filter', 4)])
def test_recovery_entry_points(tmp_path, mode, ranks):
    env = runtime(f'{ranks},1,1')
    env['ASTR_GPU_MP_POSITIVITY'] = 'off'
    command = [*mpi_command(MPIEXEC, ranks, 'pinned'), str(Path(PROBE).resolve()), 'rank', mode]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (tmp_path/'run.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    assert 'GPU expanding crashfix: PASS' in result.stdout


def staged_case(path, steps, filtered):
    prepare(Path('/home/dell/workspace/M12_C13'), path, 'gpu')
    configuration(path, 'nscbc', (64,32,24), steps, archive=False)
    controller = path/'datin/controller'
    lines = controller.read_text().splitlines()
    replace_value(lines, 'lwsequ,lwslic', 'f,f,f,t')
    controller.write_text('\n'.join(lines)+'\n')
    if filtered:
        input_path = path/'datin/input.dat'
        lines = input_path.read_text().splitlines()
        replace_value(lines, 'nondimen,diffterm', 't,t,t,f,f,f,f,f,t')
        input_path.write_text('\n'.join(lines)+'\n')
    wall = path/'datin/wallbs.dat'
    lines = wall.read_text().splitlines()
    lines[2] = '0.0, 0.0, 20.0, 40.0, 60.0'
    wall.write_text('\n'.join(lines)+'\n')
    with h5py.File(path/'datin/grid.2d') as grid:
        x = grid['x'][...]
    s = np.clip(((x-x.min())/(x.max()-x.min())-.8)/.1, 0., 1.)
    eta = np.linspace(0., 1., x.shape[0])[:,None]
    sy = np.clip(np.minimum(eta-.125, .875-eta)/.125, 0., 1.)
    weight = s*s*(3.-2.*s)*sy*sy*(3.-2.*sy)
    with h5py.File(path/'datin/flowini2d.h5', 'r+') as field:
        a = np.sqrt(1.4*field['p'][...]/field['ro'][...])
        field['u1'][...] = (1.-weight)*field['u1'][...]-weight*.5*a


@pytest.mark.skipif(not EXE, reason='set ASTR_M12_CRASHFIX_EXECUTABLE')
@pytest.mark.parametrize('ranks,filtered', [(1,False), (2,False), (1,True), (2,True)])
def test_staged_nscbc_repairs_before_rhs(tmp_path, ranks, filtered):
    staged_case(tmp_path/'case', 6, filtered)
    case = tmp_path/'case'
    env = runtime(f'{ranks},1,1')
    env.update(ASTR_GPU_MP_POSITIVITY='off', ASTR_NSCBC_FARFIELD_MODE='incoming_only',
               ASTR_NSCBC_FARFIELD_RHO='1', ASTR_NSCBC_FARFIELD_U='1',
               ASTR_NSCBC_FARFIELD_V='0', ASTR_NSCBC_FARFIELD_W='0', ASTR_NSCBC_FARFIELD_T='1',
               ASTR_VALIDATION_PROFILE_STAGE_STATE='1', ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
               ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_STEP_END='5')
    command = [*mpi_command(MPIEXEC, ranks, 'pinned'), str(Path(EXE).resolve()), 'run', 'datin/input.dat']
    result = subprocess.run(command, cwd=case, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (case/'run.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-8000:]
    assert 'ASTR_MP_INPUT_CRASHFIX' in result.stdout
    # All assembled RHS snapshots must be finite, not merely the repaired q.
    snapshots = list((case/'diagnostics').glob('state.full.*.bin'))
    assert len(snapshots) == 6*3*ranks
    for path in snapshots:
        assert_finite_rhs(path)
    for path in (case/'diagnostics').glob('state.post_update.*.bin'):
        q = active_state(path)
        assert np.isfinite(q).all()
    assert_positive_recovery_snapshots(case, 6*3*ranks)


@pytest.mark.skipif(not EXE, reason='set ASTR_M12_CRASHFIX_EXECUTABLE')
@pytest.mark.parametrize('ranks', [1, 2])
def test_staged_rhs_is_rebuilt_after_late_repair(tmp_path, ranks):
    case = tmp_path/'case'
    prepare(Path('/home/dell/workspace/M12_C13'), case, 'gpu')
    configuration(case, 'nscbc', (64,32,24), 3, archive=False)
    controller = case/'datin/controller'
    lines = controller.read_text().splitlines()
    replace_value(lines, 'lwsequ,lwslic', 'f,f,f,t')
    controller.write_text('\n'.join(lines)+'\n')
    wall = case/'datin/wallbs.dat'
    lines = wall.read_text().splitlines()
    lines[2] = '0.0, 0.0, 20.0, 40.0, 60.0'
    wall.write_text('\n'.join(lines)+'\n')
    with h5py.File(case/'datin/flowini2d.h5', 'r+') as field:
        field['u1'][-1,32] += 1.
    env = runtime(f'{ranks},1,1')
    env.update(ASTR_GPU_MP_POSITIVITY='off', ASTR_NSCBC_FARFIELD_MODE='incoming_only',
               ASTR_NSCBC_FARFIELD_RHO='1', ASTR_NSCBC_FARFIELD_U='1',
               ASTR_NSCBC_FARFIELD_V='0', ASTR_NSCBC_FARFIELD_W='0', ASTR_NSCBC_FARFIELD_T='1',
               ASTR_VALIDATION_PROFILE_STAGE_STATE='1', ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
               ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_STEP_END='2')
    command = [*mpi_command(MPIEXEC, ranks, 'pinned'), str(Path(EXE).resolve()), 'run', 'datin/input.dat']
    result = subprocess.run(command, cwd=case, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (case/'run.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-8000:]
    assert 'ASTR_NSCBC_RHS_REBUILT' in result.stdout
    paths = list((case/'diagnostics').glob('state.full.*.bin'))
    assert len(paths) == 3*3*ranks
    for path in paths:
        assert_finite_rhs(path)
    assert_positive_recovery_snapshots(case, 3*3*ranks)
