"""Approved T3: a 32-node periodic filter, not a CFD or MPI solver."""
import json
import math
import os
from fractions import Fraction
from pathlib import Path
import subprocess

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference

ROOT = Path(__file__).resolve().parents[2]
COEFFICIENTS = np.array([105/512, -15/128, 45/1024, -5/512, 1/1024])


def fixture(name):
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    x = 2*np.pi*np.arange(32)/32
    rho = np.full(32, .05)
    velocity = np.zeros((32, 3))
    temperature, tv = np.full(32, 3000.), np.full(32, 2000.)
    fractions = np.tile([.55, .15, .1, .12, .08], (32, 1))
    if name == 'smooth':
        rho *= 1+.1*np.sin(x)
        velocity[:, 0] = 500+20*np.cos(x)
        velocity[:, 1] = 10*np.sin(x)
        temperature += 100*np.cos(x)
    elif name == 'cancellation':
        velocity[:16, 1] = 10*np.sin(x[:16])
        velocity[16:, 1] = -velocity[:16, 1]
    elif name in ('zero', 'trace', 'trace-step'):
        fractions[:] = [.767, .233, 0., 0., 0.]
        rho *= 1+.1*np.sin(x)
        if name in ('trace', 'trace-step'):
            fractions[:, 3] = 1e-8*(1+.1*np.cos(x))
            if name == 'trace-step':
                fractions[:, 3] = 1e-8*np.where(np.arange(32) >= 16, 1.1, .9)
            fractions[:, 2] = np.where(np.arange(32) >= 16, 1e-18, 0.)
            fractions[:, 0] = 1-fractions[:, 1:].sum(axis=1)
    elif name == 'energy':
        temperature[:] = 600.
        tv[:] = 600.
        velocity[16, 0] = 4000.
    elif name == 'vibration':
        tv[:] = 300.01
        tv[16] = 4000.
    elif name == 'density':
        rho[:16] = .005
    elif name == 'pressure-floor':
        rho[:] = .01
        fractions[:] = [1., 0., 0., 0., 0.]
        temperature[:] = 2500./(.01*model.cv_tr[0])
        tv[:] = 1000.
    elif name != 'constant':
        raise ValueError(name)
    states = []
    for r, v, t, vtemp, y in zip(rho, velocity, temperature, tv, fractions):
        species, momentum = r*y, r*v
        ev = model.ev_from_tv(species, vtemp)
        q5 = model.q5_from_state(r, momentum, species, ev, t)
        states.append(np.r_[r, momentum, q5, species, ev])
    states = np.array(states)
    if name == 'pressure-floor':
        # A legal near-cone state inside the operation-derived freeze band.
        exact_energy = (Fraction(float(states[0, 10]))+2500+
                        sum(Fraction(float(z))*Fraction(float(e))
                            for z, e in zip(states[0, 5:10], model.formation_energy)))
        energy = float(exact_energy)
        if Fraction(energy) < exact_energy:
            energy = np.nextafter(energy, np.inf)
        for _ in range(16):
            energy = np.nextafter(energy, np.inf)
        states[:, 4] = energy
    return model, states


def linear_filter(q):
    result = (193/256)*q
    for offset, coefficient in enumerate(COEFFICIENTS, 1):
        result = result+coefficient*(np.roll(q, offset, axis=0)+np.roll(q, -offset, axis=0))
    return result


def sums(q):
    return np.array([math.fsum(map(float, column)) for column in q.T])


def check_state_and_conservation(model, before, after):
    assert np.isfinite(after).all()
    assert np.all(after[:, 0] > 0.)
    assert np.all(after[:, 5:11] >= 0.)
    closure = max(abs(math.fsum(row[5:10]/row[0])-1) for row in after)
    assert closure <= 128*np.finfo(float).eps
    initial, final = sums(before), sums(after)
    absolute_scale = np.array([math.fsum(map(float, abs(column))) for column in before.T])
    # Two face updates plus initial/final 32-term reductions: 2*32+4 operations.
    unit_roundoff = np.finfo(float).eps/2
    gamma = (68*unit_roundoff)/(1-68*unit_roundoff)
    relative_drifts, zero_total_drifts = [], []
    for component in range(11):
        if initial[component] != 0.:
            drift = abs(final[component]-initial[component])/abs(initial[component])
            assert drift <= 1e-14
            relative_drifts.append(drift)
        else:
            assert abs(final[component]) <= gamma*absolute_scale[component]
            zero_total_drifts.append({'component': component+1,
                                      'absolute_drift': abs(float(final[component])),
                                      'roundoff_bound': float(gamma*absolute_scale[component])})
    element_drifts = []
    for element in (np.array([1., 0., 1., 0., 14/30]), np.array([0., 1., 0., 1., 16/30])):
        old = math.fsum(before[:, 5:10]@element)
        new = math.fsum(after[:, 5:10]@element)
        if old == 0.:
            assert new == 0.
            drift = 0.
        else:
            drift = abs(new-old)/old
        assert drift <= 1e-14
        element_drifts.append(drift)
    temperatures, vibration_temperatures, pressures = [], [], []
    for q in after:
        t, vtemp = model.temperatures(q[0], q[1:4], q[4], q[5:11, None])
        pressure = model.pressure(q[5:10], t[0])
        assert 300 <= t[0] <= 8000 and 300 <= vtemp[0] <= 8000
        assert 1e3 <= pressure <= 1e6
        temperatures.append(float(t[0]))
        vibration_temperatures.append(float(vtemp[0]))
        pressures.append(float(pressure))
    return {'max_closure_error': closure, 'max_relative_conservation_drift': max(relative_drifts),
            'zero_total_drifts': zero_total_drifts, 'element_relative_drifts': element_drifts,
            'temperature_range': [min(temperatures), max(temperatures)],
            'vibration_temperature_range': [min(vibration_temperatures), max(vibration_temperatures)],
            'pressure_range': [min(pressures), max(pressures)]}


@pytest.mark.parametrize('name', ['constant', 'smooth', 'cancellation', 'zero', 'trace', 'trace-step',
                                 'energy', 'vibration', 'density', 'pressure-floor'])
def test_periodic_filter_cpu_gpu_products_and_conservation(tmp_path, name):
    model, before = fixture(name)
    input_path = tmp_path/'states.in'
    with input_path.open('w') as stream:
        stream.write('32\n')
        np.savetxt(stream, before, fmt='%.17e')
    probe = Path(os.environ.get('ASTR_AIR5_FILTER_FLUX_PROBE_EXE',
                                ROOT/'build_gpu_probe/bin/air5_filter_flux_probe'))
    products = []
    metrics = {'fixture': name, 'nodes': 32, 'components': 11, 'backends': {}}
    for backend in ('cpu', 'gpu'):
        output = tmp_path/f'{backend}.out'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if backend == 'cpu' else '0')
        with (tmp_path/f'{backend}.log').open('w') as log:
            subprocess.run([str(probe), backend, str(input_path), str(output)], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        values = np.loadtxt(output, ndmin=2)
        assert values.shape == (32, 37)
        assert np.all(values[:, 1:3] == 0)
        after, high, flux = values[:, 3:14], values[:, 14:25], values[:, 25:36]
        lower = -.5*before[:, 5:10]
        upper = .5*np.roll(before[:, 5:10], -1, axis=0)
        assert np.all(flux[:, 5:10] >= lower)
        assert np.all(flux[:, 5:10] <= upper)
        mass_errors = np.array([abs(math.fsum(row[5:10])-row[0]) for row in flux])
        mass_scales = np.array([abs(row[0])+math.fsum(abs(row[5:10])) for row in flux])
        assert np.all(mass_errors <= 128*np.finfo(float).eps*mass_scales)
        assert np.max(abs(after-(before+flux-np.roll(flux, 1, axis=0)))) <= 2e-10
        assert np.max(abs(before+high-np.roll(high, 1, axis=0)-linear_filter(before))) <= 2e-10
        metrics['backends'][backend] = check_state_and_conservation(model, before, after)
        metrics['backends'][backend].update({
            'min_species_density': float(after[:, 5:10].min()),
            'max_face_mass_error': float(mass_errors.max()),
            'energy_limited_faces': int(np.count_nonzero(values[:, -1] < 1.)),
            'min_energy_ratio': float(values[:, -1].min()),
            'component_face_change_max': np.max(abs(flux-high), axis=0).tolist(),
            'component_linear_update_max': np.max(abs(linear_filter(before)-before), axis=0).tolist(),
            'component_departure_from_linear_max': np.max(abs(after-linear_filter(before)), axis=0).tolist(),
        })
        products.append(after)
        if name in ('constant', 'smooth', 'cancellation'):
            assert np.max(abs(after-linear_filter(before))) <= 2e-10
            assert np.all(values[:, -1] == 1.)
        if name == 'zero':
            np.testing.assert_array_equal(after[:, 7:10], np.zeros((32, 3)))
        if name in ('trace', 'trace-step'):
            assert np.min(linear_filter(before)[:, 7]) < 0.
            assert np.any(flux[:, 7] != high[:, 7])
            assert np.all(values[:, -1] == 1.)
        if name in ('energy', 'vibration'):
            assert np.any(values[:, -1] < 1.)
        if name == 'pressure-floor':
            np.testing.assert_array_equal(after, before)
            np.testing.assert_array_equal(flux, np.zeros_like(flux))
            np.testing.assert_array_equal(values[:, -1], np.zeros(32))
        if name == 'cancellation':
            assert sums(before)[2] == 0. and np.any(before[:, 2] != 0.)
    metrics['cpu_gpu_max_abs_difference'] = float(np.max(abs(products[0]-products[1])))
    assert metrics['cpu_gpu_max_abs_difference'] <= 2e-10
    (tmp_path/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
