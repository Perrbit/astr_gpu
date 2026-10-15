"""Exact-rational signs at the T3 double-temperature lower energy bounds."""
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from test_air5_filter_flux_probe import ROOT
from air5_radau_reference import Air5RadauReference


def exact_margins(row, carry=None):
    q, tr, v = list(map(Fraction, map(float, row[:11]))), row[11:16], row[16:]
    if carry is not None:
        q = [h-Fraction(float(c)) for h, c in zip(q, carry)]
    vibration = q[10]-sum(q[s+5]*Fraction(float(v[s])) for s in range(5))
    thermal = (q[4]-q[10]-sum(q[s+5]*Fraction(float(tr[s])) for s in range(5))
               -sum(q[s]**2 for s in (1, 2, 3))/(2*q[0]))
    return vibration, thermal


def exact_pressure_margin(state, formation, floor, carry=None):
    q = list(map(Fraction, map(float, state)))
    if carry is not None:
        q = [h-Fraction(float(c)) for h, c in zip(q, carry)]
    return (q[4]-q[10]-sum(q[s+5]*Fraction(float(formation[s])) for s in range(5))
            -sum(q[s]**2 for s in (1, 2, 3))/(2*q[0])-Fraction(float(floor)))


def predicate_states():
    # Frozen FP64 node (15,15,12), T3-P1 vibration/axis3; no artifact dependency.
    q = np.array([.05, 0., 0., 0., 369522.0630378398, .027500000000000004,
                  .0075, .005000000000000001, .006, .004, 2.938687330386796])
    tr = np.array([-87155.17987089539, -76539.78238703348, 33574650.584154926,
                   15416544.26113556, 2926591.565453831])
    v = np.array([12.342261482012496, 302.69249224928086, 0., 0., 82.27036194046147])
    rows, names = [], []
    for direction in (0, -1, 1):
        trial = q.copy()
        if direction:
            trial[10] = np.nextafter(trial[10], np.inf*direction)
        rows.append(np.r_[trial, tr, v]); names.append(f'captured_vibration_{direction}')
    for mode in ('vibration', 'thermal'):
        state = np.array([1., 1., 0., 0., 1.5, .5, .5, 0., 0., 0., 1.])
        tr0, v0 = np.zeros(5), np.ones(5)
        for direction in (0, -1, 1):
            trial = state.copy()
            component = 10 if mode == 'vibration' else 4
            if direction:
                trial[component] = np.nextafter(trial[component], np.inf*direction)
            rows.append(np.r_[trial, tr0, v0]); names.append(f'exact_zero_{mode}_{direction}')
    random = np.random.default_rng(2215)
    for i in range(64):
        state = q.copy()
        state[0] = random.uniform(.01, .1)
        fractions = random.random(5); fractions /= fractions.sum()
        state[5:10] = state[0]*fractions
        state[1:4] = state[0]*random.uniform(-4000, 4000, 3)
        state[10] = state[5:10]@v
        state[4] = state[10]+state[5:10]@tr+state[1:4]@state[1:4]/(2*state[0])
        component = 4 if i % 2 else 10
        state[component] = np.nextafter(state[component], np.inf if i % 4 < 2 else -np.inf)
        rows.append(np.r_[state, tr, v]); names.append(f'near_bound_{i}')
    return names, np.array(rows)


def run_predicate(tmp_path, names, rows, expected, margins, mode=None):
    source, result = tmp_path/'input.dat', tmp_path/'result.dat'
    with source.open('w') as stream:
        stream.write(str(len(rows))+'\n')
        np.savetxt(stream, rows, fmt='%.17e')
    exe = Path(os.environ.get('ASTR_AIR5_FILTER_PREDICATE_PROBE_EXE',
                              ROOT/'build_gpu_probe/bin/air5_filter_predicate_probe'))
    with (tmp_path/'run.log').open('w') as log:
        command = [str(exe), str(source), str(result)]+([mode] if mode else [])
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=60, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
    verdicts = np.loadtxt(result, dtype=int, ndmin=2)
    differences = [names[i] for i in range(len(rows)) if np.any(verdicts[i, 1:] != expected[i])]
    (tmp_path/'exact_signs.json').write_text(json.dumps({'names': names, 'margins': margins,
        'expected': expected, 'cpu_gpu': verdicts[:, 1:].tolist(), 'mismatches': differences}, indent=2)+'\n')
    assert not differences, differences


def test_predicate_matches_exact_rational_signs(tmp_path):
    names, rows = predicate_states()
    margins = [exact_margins(row) for row in rows]
    expected = [int(v >= 0 and t >= 0) for v, t in margins]
    run_predicate(tmp_path, names, rows, expected, [[str(v), str(t)] for v, t in margins])


def test_pressure_convex_guard_exact_signs_and_excluded_state(tmp_path):
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    rows, names = [], []
    state = np.array([1., 1., 0., 0., -288135.5, .5, .5, 0., 0., 0., 1.])
    assert exact_pressure_margin(state, model.formation_energy, 2500.) == 0
    for direction in (0, -1, 1):
        trial = state.copy()
        if direction:
            trial[4] = np.nextafter(trial[4], direction*np.inf)
        rows.append(np.r_[trial, np.zeros(10)]); names.append(f'pressure_zero_{direction}')
    random = np.random.default_rng(2216)
    for i in range(64):
        q = state.copy()
        q[0] = random.uniform(.001, .1)
        y = random.random(5); y /= y.sum()
        q[5:10] = q[0]*y
        q[1:4] = q[0]*random.uniform(-4000, 4000, 3)
        q[10] = random.uniform(1, 10000)
        q[4] = q[10]+q[5:10]@model.formation_energy+q[1:4]@q[1:4]/(2*q[0])+2500.
        q[4] = np.nextafter(q[4], np.inf if i % 2 else -np.inf)
        rows.append(np.r_[q, np.zeros(10)]); names.append(f'pressure_near_{i}')
    captured = np.array([.0011381776792390686, 0., 0., 0., 6280.8291820285685,
                         .0006259977235814877, .00017072665188586032, .00011381776792390688,
                         .0001365813215086882, 9.105421433912554e-05, 154.97323472845767])
    rows.append(np.r_[captured, np.zeros(10)]); names.append('captured_density_failure')
    for name, rho, y in (('molecular_included', .01, [1., 0., 0., 0., 0.]),
                         ('atom_rich_excluded', .001, [.01, 0., .99, 0., 0.])):
        species = rho*np.array(y)
        temperature = 1100./(species@model.gas_constant)
        ev = model.ev_from_tv(species, 1000.)
        energy = model.q5_from_state(rho, np.zeros(3), species, ev, temperature)
        q = np.r_[rho, np.zeros(3), energy, species, ev]
        assert 300 <= temperature <= 8000
        assert abs(model.pressure(species, temperature)-1100.) <= 2e-10
        margin = exact_pressure_margin(q, model.formation_energy, 2500.)
        assert (margin >= 0) == (name == 'molecular_included')
        rows.append(np.r_[q, np.zeros(10)]); names.append(name)
    rows = np.array(rows)
    margins = [exact_pressure_margin(row[:11], model.formation_energy, 2500.) for row in rows]
    assert margins[names.index('captured_density_failure')] < 0
    run_predicate(tmp_path, names, rows, [int(m >= 0) for m in margins],
                  [str(m) for m in margins], 'pressure')


def test_pressure_represented_exact_signs(tmp_path):
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    base = np.array([1., 1., 0., 0., -288135.5, .5, .5, 0., 0., 0., 1.])
    rows, names = [], []
    for component in (0, 1, 4, 5, 10):
        for sign in (-1, 0, 1):
            carry = np.zeros(11)
            carry[component] = sign*abs(np.spacing(base[component]))/4
            np.testing.assert_array_equal(base-carry, base)
            rows.append(np.r_[base, carry]); names.append(f'carry_ulp_{component}_{sign}')
    random = np.random.default_rng(2217)
    for i in range(64):
        high = base.copy()
        high[0] = random.uniform(.001, .1)
        y = random.random(5); y /= y.sum()
        high[5:10] = high[0]*y
        high[1:4] = high[0]*random.uniform(-4000, 4000, 3)
        high[10] = random.uniform(1, 10000)
        high[4] = high[10]+high[5:10]@model.formation_energy+high[1:4]@high[1:4]/(2*high[0])+2500.
        carry = random.uniform(-.5, .5, 11)*abs(np.spacing(high))
        rows.append(np.r_[high, carry]); names.append(f'carry_random_{i}')
    rows = np.array(rows)
    margins = [exact_pressure_margin(row[:11], model.formation_energy, 2500., row[11:]) for row in rows]
    run_predicate(tmp_path, names, rows, [int(m >= 0) for m in margins],
                  [str(m) for m in margins], 'pressure-represented')
