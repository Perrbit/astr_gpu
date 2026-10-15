"""Approved T3 boundary probe: 33 nodes, two fixed endpoints, no CFD or MPI."""
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference

ROOT = Path(__file__).resolve().parents[2]
NODES = 33
WEIGHTS = np.r_[.5, np.ones(NODES-2), .5]
D5 = np.zeros((NODES-7, NODES-2), dtype=np.int64)
for row in range(len(D5)):
    D5[row, row:row+6] = [-1, 5, -10, 10, -5, 1]
INCREMENT = np.zeros((NODES, NODES), dtype=np.int64)
INCREMENT[1:-1, 1:-1] = -(D5.T@D5)


def linear_filter(before):
    result = before.copy()
    for i in range(1, NODES-1):
        for component in range(11):
            result[i, component] += math.fsum(
                int(INCREMENT[i, j])*(float(before[j, component])-float(before[i, component]))
                for j in np.flatnonzero(INCREMENT[i]))/1024
    return result


def fixture(name):
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    x = np.arange(NODES)/32
    rho = np.full(NODES, .05)
    velocity = np.zeros((NODES, 3))
    temperature, tv = np.full(NODES, 3000.), np.full(NODES, 2000.)
    fractions = np.tile([.55, .15, .1, .12, .08], (NODES, 1))
    if name == 'smooth':
        rho *= 1+.1*np.sin(2*np.pi*x)
        velocity[:, 0] = 500+20*np.cos(2*np.pi*x)
        temperature += 100*np.cos(2*np.pi*x)
    elif name == 'cancellation':
        velocity[1:16, 1] = np.arange(1, 16)
        velocity[17:32, 1] = -velocity[1:16, 1][::-1]
    elif name in ('zero', 'trace', 'trace-step'):
        fractions[:] = [.767, .233, 0., 0., 0.]
        if name != 'zero':
            middle = (np.arange(NODES) >= 4) & (np.arange(NODES) <= 28)
            fractions[:, 2] = np.where(middle, 1e-18, 0.)
            fractions[:, 3] = 1e-8*(1+.1*np.cos(2*np.pi*x))
            if name == 'trace-step':
                fractions[:, 3] = 1e-8*np.where(middle, 1.1, .9)
            fractions[:, 0] = 1-fractions[:, 1:].sum(axis=1)
    elif name == 'energy':
        temperature[:] = 600.
        tv[:] = 600.
        velocity[[2, 30], 0] = 4000.
    elif name == 'vibration':
        tv[:] = 300.01
        tv[[2, 30]] = 4000.
    elif name == 'density':
        rho[:4] = .005
        rho[29:] = .005
    elif name != 'constant':
        raise ValueError(name)
    states = []
    for r, v, t, vtemp, y in zip(rho, velocity, temperature, tv, fractions):
        species, momentum = r*y, r*v
        ev = model.ev_from_tv(species, vtemp)
        q5 = model.q5_from_state(r, momentum, species, ev, t)
        states.append(np.r_[r, momentum, q5, species, ev])
    return model, np.array(states)


def weighted_sums(q):
    return np.array([math.fsum(map(float, WEIGHTS*column)) for column in q.T])


def run_probe(tmp_path, before):
    probe = Path(os.environ.get('ASTR_AIR5_FILTER_BOUNDARY_PROBE_EXE',
                                ROOT/'build_gpu_probe/bin/air5_filter_boundary_probe'))
    input_path = tmp_path/'states.in'
    with input_path.open('w') as stream:
        stream.write('33\n')
        np.savetxt(stream, before, fmt='%.17e')
    products = {}
    for backend in ('cpu', 'gpu'):
        output = tmp_path/f'{backend}.out'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if backend == 'cpu' else '0')
        with (tmp_path/f'{backend}.log').open('w') as log:
            subprocess.run([str(probe), backend, str(input_path), str(output)], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        values = np.loadtxt(output, ndmin=2)
        assert values.shape == (NODES, 37)
        assert np.all(values[:, 1:3] == 0)
        products[backend] = values
    assert np.max(abs(products['cpu'][:, 3:]-products['gpu'][:, 3:])) <= 2e-10
    return products


@pytest.mark.parametrize('degree', range(11))
def test_boundary_polynomial_moments_and_interior_equivalence(tmp_path, degree):
    _, before = fixture('constant')
    x = np.arange(NODES)/32
    before[:, 1] = x**degree
    expected = linear_filter(before)
    metrics = {'degree': degree, 'boundary_order': 5, 'interior_order': 10, 'backends': {}}
    for backend, values in run_probe(tmp_path, before).items():
        after, high, flux = values[:, 3:14], values[:, 14:25], values[:, 25:36]
        np.testing.assert_array_equal(after[[0, -1]], before[[0, -1]])
        np.testing.assert_array_equal(high[[0, -2, -1]], np.zeros((3, 11)))
        assert np.max(abs(after-expected)) <= 2e-10
        assert np.max(abs(high-flux)) <= 2e-10
        assert np.all(values[:, -1] == 1.)
        change = after[:, 1]-before[:, 1]
        if degree <= 4:
            np.testing.assert_array_equal(change, np.zeros(NODES))
        if degree == 5:
            assert change[1] == 15*(1/32)**5/128
            assert change[1] != 0.
        if degree <= 9:
            np.testing.assert_array_equal(change[6:-6], np.zeros(NODES-12))
        else:
            assert np.max(abs(change[6:-6])) > 0.
        metrics['backends'][backend] = {'max_linear_difference': float(np.max(abs(after-expected))),
                                        'first_interior_correction': float(change[1])}
    (tmp_path/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')


@pytest.mark.parametrize('name', ['constant', 'smooth', 'cancellation', 'zero', 'trace',
                                 'trace-step', 'energy', 'vibration', 'density'])
def test_physical_boundary_filter_cpu_gpu_conservation_and_admissibility(tmp_path, name):
    model, before = fixture(name)
    linear = linear_filter(before)
    products = run_probe(tmp_path, before)
    metrics = {'fixture': name, 'nodes': NODES, 'components': 11, 'backends': {}}
    for backend, values in products.items():
        after, high, flux = values[:, 3:14], values[:, 14:25], values[:, 25:36]
        assert np.isfinite(after).all() and np.all(after[:, 0] > 0.)
        assert np.all(after[:, 5:11] >= 0.)
        np.testing.assert_array_equal(after[[0, -1]], before[[0, -1]])
        np.testing.assert_array_equal(high[[0, -2, -1]], np.zeros((3, 11)))
        np.testing.assert_array_equal(flux[[0, -2, -1]], np.zeros((3, 11)))
        reconstructed = before.copy()
        reconstructed[1:-1] = before[1:-1]+high[1:-1]-high[:-2]
        assert np.max(abs(reconstructed-linear)) <= 2e-10
        np.testing.assert_array_equal(after[1:-1], before[1:-1]+flux[1:-1]-flux[:-2])
        assert np.all(flux[1:-2, 5:10] >= -.5*before[1:-2, 5:10])
        assert np.all(flux[1:-2, 5:10] <= .5*before[2:-1, 5:10])
        mass_error = np.array([abs(math.fsum(row[5:10])-row[0]) for row in flux])
        mass_scale = np.array([abs(row[0])+math.fsum(abs(row[5:10])) for row in flux])
        assert np.all(mass_error <= 128*np.finfo(float).eps*mass_scale)
        closure = max(abs(math.fsum(row[5:10]/row[0])-1) for row in after)
        assert closure <= 128*np.finfo(float).eps
        old, new = weighted_sums(before), weighted_sums(after)
        scale = np.array([math.fsum(map(float, WEIGHTS*abs(column))) for column in before.T])
        operations = 2*NODES+4
        roundoff = np.finfo(float).eps/2
        gamma = operations*roundoff/(1-operations*roundoff)
        drifts, zero_totals = [], []
        for component in range(11):
            if old[component] != 0.:
                drift = abs(new[component]-old[component])/abs(old[component])
                assert drift <= 1e-14
                drifts.append(float(drift))
            else:
                assert abs(new[component]) <= gamma*scale[component]
                zero_totals.append({'component': component+1, 'drift': float(abs(new[component])),
                                    'bound': float(gamma*scale[component])})
        element_drifts = []
        for element in range(2):
            atom_weights = model.atom_counts[:, element]/model.molar_mass
            old_atoms = math.fsum(map(float, WEIGHTS*(before[:, 5:10]@atom_weights)))
            new_atoms = math.fsum(map(float, WEIGHTS*(after[:, 5:10]@atom_weights)))
            drift = abs(new_atoms-old_atoms)/old_atoms
            assert drift <= 1e-14
            element_drifts.append(float(drift))
        temperatures, vibration_temperatures, pressures = [], [], []
        for state in after:
            t, tv = model.temperatures(state[0], state[1:4], state[4], state[5:11, None])
            p = model.pressure(state[5:10], t[0])
            assert 300 <= t[0] <= 8000 and 300 <= tv[0] <= 8000
            assert 1e3 <= p <= 1e6
            temperatures.append(float(t[0]))
            vibration_temperatures.append(float(tv[0]))
            pressures.append(float(p))
        if name in ('constant', 'smooth', 'cancellation'):
            assert np.max(abs(after-linear)) <= 2e-10
            assert np.all(values[:, -1] == 1.)
        if name == 'zero':
            np.testing.assert_array_equal(after[:, 7:10], np.zeros((NODES, 3)))
        if name in ('trace', 'trace-step'):
            assert min(linear[1:6, 7].min(), linear[-6:-1, 7].min()) < 0.
            assert np.any(flux[1:6, 7] != high[1:6, 7])
            assert np.any(flux[-6:-1, 7] != high[-6:-1, 7])
            assert np.all(values[:, -1] == 1.)
        if name in ('energy', 'vibration'):
            assert np.any(values[1:6, -1] < 1.)
            assert np.any(values[-6:-1, -1] < 1.)
        if name == 'cancellation':
            assert old[2] == 0. and np.any(before[:, 2] != 0.)
        metrics['backends'][backend] = {
            'max_closure_error': closure, 'max_relative_conservation_drift': max(drifts),
            'zero_total_drifts': zero_totals, 'element_relative_drifts': element_drifts,
            'min_species_density': float(after[:, 5:10].min()),
            'max_face_mass_error': float(mass_error.max()),
            'min_energy_ratio': float(values[:, -1].min()),
            'energy_limited_faces': int(np.count_nonzero(values[:, -1] < 1.)),
            'component_departure_from_linear_max': np.max(abs(after-linear), axis=0).tolist(),
            'temperature_range': [min(temperatures), max(temperatures)],
            'vibration_temperature_range': [min(vibration_temperatures), max(vibration_temperatures)],
            'pressure_range': [min(pressures), max(pressures)]}
    metrics['cpu_gpu_max_abs_difference'] = float(np.max(abs(products['cpu'][:, 3:14]-products['gpu'][:, 3:14])))
    (tmp_path/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
