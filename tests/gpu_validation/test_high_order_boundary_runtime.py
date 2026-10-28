"""Actual ASTR first-stage RHS convergence for the approved Cartesian chain.

Python supplies a manufactured field and analytic derivatives; ASTR performs
the complete boundary preparation and every native time step.
"""
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from prepare_s1_flatplate_case import write_input, write_controller
from run_m12_mp7_nscbc_performance import runtime, mpi_command
from test_sbli_joint_nscbc import MPIEXEC
from compare_flowstate import read_flowstate
from prepare_hcw_laminar_sbli_case import prepare_case

EXE = os.environ.get('ASTR_SBLI_GRCBC_EXE')
pytestmark = pytest.mark.skipif(not EXE, reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake executable')
RHO, MACH, GAMMA, RE = 1024., 2., 1.4, 1.
NY, NZ = 16, 8


def manufactured(nx, ny=NY, axis='x', rho=RHO):
    z, y, x = np.meshgrid(np.linspace(0, .125, NZ+1), np.linspace(0, 1, ny+1),
                          np.linspace(0, 1, nx+1), indexing='ij')
    amplitude = .05*np.exp(4*(x-1))
    u = y+amplitude*y*(1-y)
    ux = 4*amplitude*y*(1-y)
    uxx = 16*amplitude*y*(1-y)
    uy = 1+amplitude*(1-2*y)
    uyy = -2*amplitude
    uxy = 4*amplitude*(1-2*y)
    p = rho/(GAMMA*MACH*MACH)
    mu = 1/RE
    rhs = np.zeros(x.shape+(5,))
    rhs[..., 0] = -rho*ux
    rhs[..., 1] = -2*rho*u*ux+mu*(4/3*uxx+uyy)
    rhs[..., 2] = mu*uxy/3
    rhs[..., 4] = -(GAMMA*p/(GAMMA-1)+1.5*rho*u*u)*ux+mu*(4/3*(ux*ux+u*uxx)+uy*uy+u*uyy)
    # At x+, p equals the prescribed outlet target. Only the incoming acoustic
    # normal derivative is replaced by zero. R_minus*L_minus is normalization independent.
    sound = 1/MACH
    increment = np.where(u[..., -1] < sound, rho*(sound-u[..., -1])*ux[..., -1]/(2*sound), 0.)
    rhs[..., -1, 0] += increment
    rhs[..., -1, 1] += increment*(u[..., -1]-sound)
    enthalpy = GAMMA*p/((GAMMA-1)*rho)+.5*u[..., -1]**2
    rhs[..., -1, 4] += increment*(enthalpy-u[..., -1]*sound)
    if axis == 'y':
        exponential = np.exp(4*(y-1))
        u = y+.05*(exponential-np.exp(-4)-y*(1-np.exp(-4)))
        uy = 1+.05*(4*exponential-(1-np.exp(-4)))
        uyy = .8*exponential
        rhs[...] = 0.
        rhs[..., 1] = mu*uyy
        rhs[..., 4] = mu*(uy*uy+u*uyy)
    rhs[..., 0, :] = 0.    # Prescribed inlet.
    rhs[:, 0, :, :] = 0.   # Algebraic isothermal no-slip wall.
    return x, y, z, u, rhs


def run_manufactured(case, backend, nx, ny=NY, axis='x', rho=RHO):
    datin = case/'datin'
    datin.mkdir(parents=True)
    (case/'outdat/new').mkdir(parents=True)
    (case/'diagnostics').mkdir()
    x, y, z, u, reference = manufactured(nx, ny, axis, rho)
    with h5py.File(datin/'grid.flatplate.h5', 'w') as file:
        for name, value in [('x', x), ('y', y), ('z', z)]:
            file[name] = value
    with h5py.File(datin/'flowini3d.h5', 'w') as file:
        for name, value in [('ro', np.full_like(x, rho)), ('u1', u), ('u2', np.zeros_like(x)),
                            ('u3', np.zeros_like(x)), ('t', np.ones_like(x))]:
            file[name] = value
    with (datin/'inlet.prof').open('w') as file:
        file.write('Manufactured density=provided pressure=provided\nrho u v temperature pressure\n1 1 1 1\ny rho u v temperature pressure\n')
        for value in u[0, :, 0]:
            file.write(f'{rho:.16e} {value:.16e} 0.0000000000000000e+00 1.0000000000000000e+00 '
                       f'{rho/(GAMMA*MACH*MACH):.16e}\n')
    write_input(datin/'input.flatplate', nx, ny, NZ, 't' if backend == 'gpu' else 'f',
                't', 'f', '743e', 't', .001, RE, MACH, 288., 1., 52, 11, 3, 0, 'prof', 1,
                x_max_bctype=22, outlet_pressure=rho)
    # The legacy boundary tokenizer has 20-character words; keep this test's
    # large pressure ratio explicit without truncating a long exponent token.
    path = datin/'input.flatplate'
    path.write_text(path.read_text().replace(f'22,{rho:.16e}', f'22,{rho:g}'))
    write_controller(datin/'controller', 1, 1, 1e-8, feqlist=1)
    (datin/'input.output').write_text("""&output
 directory='outdat/new'
 host_budget_bytes=134217728,device_budget_bytes=134217728,buffer_bytes=4194304,device_reserve_bytes=0
/
&checkpoint
 enabled=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    env = runtime('1,1,1')
    env.pop('ASTR_WALL_BLOWING_MODE', None)
    env.update(ASTR_GPU_MP_POSITIVITY='off', ASTR_GPU_MP_SPLIT_CACHE='off',
               ASTR_NSCBC_FARFIELD_MODE='sbli_grcbc', ASTR_NSCBC_FARFIELD_SHOCK_X='10',
               ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='40', ASTR_NSCBC_FARFIELD_RHO=str(rho),
               ASTR_NSCBC_FARFIELD_U='1', ASTR_NSCBC_FARFIELD_V='0', ASTR_NSCBC_FARFIELD_W='0',
               ASTR_NSCBC_FARFIELD_T='1', ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
               ASTR_VALIDATION_RHS_STEP='0')
    with (case/'run.log').open('w') as log:
        result = subprocess.run(mpi_command(MPIEXEC, 1, 'pinned')+[str(Path(EXE).resolve()), 'run',
                                 'datin/input.flatplate'], cwd=case, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=120)
    text = (case/'run.log').read_text()
    assert result.returncode == 0 and 'The job is done!' in text, text[-8000:]
    assert 'sbli-grcbc-h1 high-order boundary chain' in text
    path = case/'diagnostics/state.full.step00000000.rk01.rank00000000.bin'
    with path.open('rb') as file:
        header = np.fromfile(file, np.int32, 4)
        assert list(header) == [nx, ny, NZ, 5]
        raw = np.fromfile(file, np.float64).reshape((nx+1, ny+1, NZ+1, 5), order='F')
    # Uniform Cartesian coordinate Jacobian. Snapshot convention is J*dq/dt.
    jacobian = (1/nx)*(1/ny)*(.125/NZ)
    actual = raw.transpose(2, 1, 0, 3)/jacobian
    mask = np.zeros(x.shape, dtype=bool)
    if axis == 'x':
        mask[:, 1:, 1:7] = True
        mask[:, 1:, -7:] = True
    else:
        # The outlet filter modifies nonpolynomial transverse data. The fixed
        # middle x plane isolates the y operator without dropping any y row.
        mask[:, 1:7, nx//2] = True
        mask[:, -7:, nx//2] = True
    component_errors = np.max(np.abs(actual-reference)[mask], axis=0)
    return actual, raw, component_errors


def test_actual_full_rhs_x_convergence(tmp_path):
    errors = []
    reports = []
    for nx in (32, 64, 128):
        cpu, cpu_raw, cpu_error = run_manufactured(tmp_path/f'cpu_{nx}', 'cpu', nx)
        gpu, gpu_raw, gpu_error = run_manufactured(tmp_path/f'gpu_{nx}', 'gpu', nx)
        parity = float(np.max(np.abs(cpu_raw-gpu_raw)/(1+np.abs(cpu_raw))))
        assert parity <= 2e-10, (nx, parity)
        errors.append(float(max(cpu_error.max(), gpu_error.max())))
        reports.append(dict(nx=nx, cpu_error=cpu_error.tolist(), gpu_error=gpu_error.tolist(),
                            raw_rhs_backend_scaled=parity,
                            physical_rhs_backend_abs=float(np.max(np.abs(cpu-gpu)))))
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    report = dict(scope='first-stage full RHS, x refinement, all y nodes including characteristic top/outlet',
                  density=RHO, fixed_yz_intervals=[NY, NZ], source='analytic constant-viscosity Navier-Stokes',
                  cases=reports, errors=errors, orders=orders.tolist())
    (tmp_path/'convergence.json').write_text(json.dumps(report, indent=2)+'\n')
    assert np.all(orders >= 5.5), report


def test_actual_full_rhs_y_convergence(tmp_path):
    errors, reports = [], []
    for ny in (32, 64, 128):
        cpu, cpu_raw, cpu_error = run_manufactured(tmp_path/f'cpu_{ny}', 'cpu', 16, ny, 'y', 1.)
        gpu, gpu_raw, gpu_error = run_manufactured(tmp_path/f'gpu_{ny}', 'gpu', 16, ny, 'y', 1.)
        parity = float(np.max(np.abs(cpu_raw-gpu_raw)/(1+np.abs(cpu_raw))))
        assert parity <= 2e-10, (ny, parity)
        errors.append(float(max(cpu_error.max(), gpu_error.max())))
        reports.append(dict(ny=ny, cpu_error=cpu_error.tolist(), gpu_error=gpu_error.tolist(),
                            raw_rhs_backend_scaled=parity,
                            physical_rhs_backend_abs=float(np.max(np.abs(cpu-gpu)))))
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    report = dict(scope='first-stage full RHS, y refinement at fixed middle x plane; both physical boundary strips',
                  density=1., fixed_xz_intervals=[16, NZ], source='analytic constant-viscosity Navier-Stokes',
                  cases=reports, errors=errors, orders=orders.tolist())
    (tmp_path/'convergence.json').write_text(json.dumps(report, indent=2)+'\n')
    assert np.all(orders >= 5.5), report


def test_hcw_native_short_window(tmp_path):
    """Real HCW targets, two-rank ASTR steps, no rendering or production run."""
    states, statistics, reports = [], [], []
    for backend in ('cpu', 'gpu'):
        case = tmp_path/backend
        manifest = prepare_case(case, use_gpu='t' if backend == 'gpu' else 'f', im=64, jm=48, km=5)
        dt = manifest['run_plan']['dt_nondimensional']
        write_controller(case/'datin/controller', 11, 1, dt, feqlist=1)
        config = case/'datin/input.output'
        config.write_text(config.read_text().replace('interval_steps=3000', 'interval_steps=12'))
        env = runtime('2,1,1')
        env.pop('ASTR_WALL_BLOWING_MODE', None)
        for line in (case/'runtime.env').read_text().splitlines():
            name, value = line.removeprefix('export ').split('=', 1)
            env[name] = value
        with (case/'run.log').open('w') as log:
            result = subprocess.run(mpi_command(MPIEXEC, 2, 'pinned')+[str(Path(EXE).resolve()), 'run',
                                     'datin/input.hcw'], cwd=case, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=180)
        text = (case/'run.log').read_text()
        assert result.returncode == 0 and 'The job is done!' in text, text[-8000:]
        assert 'sbli-grcbc-h1 high-order boundary chain' in text
        checkpoint = case/'outdat/new/checkpoints/step000000000012'
        with h5py.File(checkpoint/'state.h5') as file:
            state = np.stack([file[f'q{i:04d}'][...] for i in range(1, 12)], axis=-1)
        assert b'sbli-grcbc-h1' in (checkpoint/'control.bin').read_bytes()
        rho = state[..., 0]
        internal = state[..., 4]-.5*np.sum(state[..., 1:4]**2, axis=-1)/rho
        temperature = .4*internal*1.4*manifest['flow']['mach']**2/rho
        assert np.isfinite(state).all() and rho.min() > 0 and internal.min() > 0
        states.append(state)
        statistics.append(read_flowstate(case))
        reports.append(dict(backend=backend, dt=dt, complete_steps=12, time_nondimensional=12*dt,
                            rho_min=float(rho.min()), temperature_min=float(temperature.min())))
    field_error = float(np.max(np.abs(states[1]-states[0])/(1+np.abs(states[0]))))
    assert statistics[0][0] == statistics[1][0]
    np.testing.assert_array_equal(statistics[0][1][:, 0], statistics[1][1][:, 0])
    statistics_error = float(np.max(np.abs(statistics[1][1]-statistics[0][1])))
    (tmp_path/'hcw_short_window.json').write_text(json.dumps(dict(
        scope='64x48x5, NP2 x, native 12-step window; not a developed-SBLI physical gate',
        cases=reports, maximum_scaled_field_error=field_error,
        maximum_absolute_statistics_error=statistics_error), indent=2)+'\n')
    assert field_error <= 2e-10 and statistics_error <= 2e-10
