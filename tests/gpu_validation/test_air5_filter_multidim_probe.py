"""T3-P1 native pure-filter gates; Python supplies/checks data, not CFD steps."""
import json
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference
from test_air5_filter_flux_probe import COEFFICIENTS, ROOT
from test_air5_filter_predicate_probe import exact_margins, exact_pressure_margin

NAMES = ('constant', 'smooth', 'zero', 'trace-step', 'energy', 'vibration',
         'density', 'cancellation', 'carry-only')


def fixture3(name):
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    indices = np.indices((32, 32, 32))
    phase = 2*np.pi*indices/32
    wave = np.sin(phase).sum(axis=0)/3
    rho = np.full((32, 32, 32), .05)
    velocity = np.zeros((32, 32, 32, 3))
    temperature, tv = np.full_like(rho, 3000.), np.full_like(rho, 2000.)
    fractions = np.broadcast_to([.55, .15, .1, .12, .08], rho.shape+(5,)).copy()
    octants = (indices >= 16).sum(axis=0) % 2 == 0
    if name == 'smooth':
        rho *= 1+.1*wave
        velocity[..., 0] = 500+20*np.cos(phase).sum(axis=0)/3
        velocity[..., 1] = 10*wave
        temperature += 100*np.cos(phase).sum(axis=0)/3
    elif name in ('zero', 'trace-step'):
        fractions[:] = [.767, .233, 0., 0., 0.]
        rho *= 1+.1*wave
        if name == 'trace-step':
            fractions[..., 3] = 1e-8*np.where(octants, 1.1, .9)
            fractions[..., 2] = np.where(octants, 1e-18, 0.)
            fractions[..., 0] = 1-fractions[..., 1:].sum(axis=-1)
    elif name == 'energy':
        temperature[:] = 600.
        tv[:] = 600.
        velocity[16, 16, 16, 0] = 4000.
    elif name == 'vibration':
        tv[:] = 300.01
        tv[16, 16, 16] = 4000.
    elif name == 'density':
        rho[octants] = .005
    elif name == 'cancellation':
        velocity[..., 1] = np.where(octants, 10., -10.)
    elif name == 'carry-only':
        velocity[..., 0] = 1/rho
    elif name != 'constant':
        raise ValueError(name)
    species, momentum = rho[..., None]*fractions, rho[..., None]*velocity
    molecular_ev = np.array([model.species_vibrational_energy(s, 2000.) for s in range(5)])
    if name in ('energy', 'vibration'):
        molecular_ev = np.zeros(rho.shape+(5,))
        for s, theta in enumerate(model.theta_v):
            if theta:
                molecular_ev[..., s] = model.gas_constant[s]*theta/np.expm1(theta/tv)
    ev = np.sum(species*molecular_ev, axis=-1)
    energy = (np.sum(momentum**2, axis=-1)/(2*rho)+np.sum(species*model.cv_tr, axis=-1)*temperature
              +ev+np.sum(species*model.formation_energy, axis=-1))
    q = np.concatenate((rho[..., None], momentum, energy[..., None], species, ev[..., None]), axis=-1)
    carry = np.zeros_like(q)
    if name == 'carry-only':
        carry[..., 1] = np.where(octants, 2.**-55, -2.**-55)
        np.testing.assert_array_equal(q-carry, q)
    return model, q, carry


def raw_faces(q, carry, axis):
    result = np.zeros_like(q)
    for m, coefficient in enumerate(COEFFICIENTS, 1):
        pairs = np.zeros_like(q)
        for r in range(m):
            pair = np.roll(q, -r-1, axis=axis)-np.roll(q, r, axis=axis)
            pair -= np.roll(carry, -r-1, axis=axis)-np.roll(carry, r, axis=axis)
            pairs += pair
        result += coefficient*pairs
    return result


def compensated_update(q, carry, flux, axis):
    h, c = q.copy(), carry.copy()
    incoming = np.roll(flux, 1, axis=axis)
    identity = (flux == 0) & (incoming == 0)
    for increment in (flux, -incoming):
        adjusted = increment-c
        updated = h+adjusted
        c = (updated-h)-adjusted
        h = updated
    h[identity], c[identity] = q[identity], carry[identity]
    return h, c


def read_products(prefix, ranks):
    fields = [np.empty((32, 32, 32, 11)) for _ in range(6)]
    local_fields = []
    floors = None
    for rank in range(ranks):
        with Path(f'{prefix}.rank{rank}.bin').open('rb') as stream:
            header = np.fromfile(stream, dtype=np.int32, count=8)
            n, offset = header[:3], header[3:6]
            assert tuple(header[6:]) == (rank, ranks)
            pieces = []
            for product in range(6):
                shape = tuple(n+1 if product < 2 else n)
                values = np.fromfile(stream, dtype=np.float64, count=11*math.prod(shape))
                pieces.append(np.moveaxis(values.reshape((11,)+shape, order='F'), 0, -1))
                region = tuple(slice(int(o), int(o+length)) for o, length in zip(offset, n))
                fields[product][region] = pieces[-1][tuple(slice(0, int(length)) for length in n)]
            local_floors = np.fromfile(stream, dtype=np.float64, count=16)
            assert local_floors.shape == (16,)
            if floors is not None:
                np.testing.assert_array_equal(local_floors, floors)
            floors = local_floors
            assert stream.read() == b''
            local_fields.append((n, offset, pieces[:2]))
    for n, offset, pieces in local_fields:
        coordinates = [(np.arange(int(length)+1)+int(o)) % 32 for length, o in zip(n, offset)]
        for component in range(2):
            np.testing.assert_array_equal(pieces[component], fields[component][np.ix_(*coordinates)])
    return fields+[floors]


def check_energy(q, carry, floors):
    state = q-carry
    tr, v, formation, pressure_floor = floors[:5], floors[5:10], floors[10:15], floors[15]
    assert pressure_floor == 2500.
    vibration = state[..., 10].copy()
    thermal = state[..., 4]-state[..., 10]
    for s in range(5):
        vibration -= state[..., s+5]*v[s]
        thermal -= state[..., s+5]*tr[s]
    kinetic = np.sum(state[..., 1:4]**2, axis=-1)/(2*state[..., 0])
    thermal -= kinetic
    pressure_margin = state[..., 4]-state[..., 10]-pressure_floor
    for s in range(5):
        pressure_margin -= state[..., s+5]*formation[s]
    pressure_margin -= kinetic
    absolute = abs(q)+abs(carry)
    v_scale = absolute[..., 10]+np.sum(absolute[..., 5:10]*abs(v), axis=-1)
    tr_scale = (absolute[..., 4]+absolute[..., 10]+
                np.sum(absolute[..., 5:10]*abs(tr), axis=-1)+abs(kinetic))
    # Includes q-carry, products, residuals, kinetic and scale operations.
    operations = 64*np.finfo(float).eps/2
    gamma = operations/(1-operations)
    vb, tb = gamma*v_scale, gamma*tr_scale
    assert np.all(vibration >= -vb) and np.all(thermal >= -tb)
    near = np.argwhere((abs(vibration) <= vb) | (abs(thermal) <= tb))
    for index in near:
        location = tuple(index)
        vr, trr = exact_margins(np.r_[q[location], floors], carry[location])
        assert vr >= 0 and trr >= 0, (location, float(vr), float(trr))
    pressure_scale = (absolute[..., 4]+absolute[..., 10]+pressure_floor+
                      np.sum(absolute[..., 5:10]*abs(formation), axis=-1)+abs(kinetic))
    pb = gamma*pressure_scale
    assert np.all(pressure_margin >= -pb)
    pressure_near = np.argwhere(abs(pressure_margin) <= pb)
    for index in pressure_near:
        location = tuple(index)
        margin = exact_pressure_margin(q[location], formation, pressure_floor, carry[location])
        assert margin >= 0, (location, float(margin))
    return {'exact_boundary_nodes': len(near), 'exact_pressure_boundary_nodes': len(pressure_near),
            'min_convex_pressure_margin': float(pressure_margin.min())}


def represented_sums(q, carry):
    q, carry = q.reshape(-1, 11), carry.reshape(-1, 11)
    return np.array([math.fsum([*q[:, c], *(-carry[:, c])]) for c in range(11)])


def check_products(model, before, before_c, products, last_axis, axes, name):
    q, carry, source, source_c, high, flux, floors = products
    assert np.isfinite(q).all() and np.isfinite(carry).all()
    state = q-carry
    assert np.all(state[..., 0] > 0) and np.all(state[..., 5:11] >= 0)
    energy = check_energy(q, carry, floors)
    np.testing.assert_array_equal(floors[10:15], model.formation_energy)
    internal = (state[..., 4]-state[..., 10]-state[..., 5:10]@model.formation_energy
                -np.sum(state[..., 1:4]**2, axis=-1)/(2*state[..., 0]))
    temperature = internal/(state[..., 5:10]@model.cv_tr)
    pressure = temperature*(state[..., 5:10]@model.gas_constant)
    assert np.all((pressure >= 1000.) & (pressure <= 1e6))
    closure = np.max(abs(np.sum(state[..., 5:10]/state[..., 0, None], axis=-1)-1))
    assert closure <= 128*np.finfo(float).eps
    initial, final = represented_sums(before, before_c), represented_sums(q, carry)
    absolute_scale = np.sum(abs(before)+abs(before_c), axis=(0, 1, 2))
    operations = 2*32**3+6*len(axes)
    roundoff = operations*np.finfo(float).eps/2
    gamma = roundoff/(1-roundoff)
    drifts, zero_totals = [], []
    for c in range(11):
        if initial[c] != 0:
            drift = abs(final[c]-initial[c])/abs(initial[c])
            assert drift <= 1e-14, (c, drift)
            drifts.append(float(drift))
        else:
            assert abs(final[c]) <= gamma*absolute_scale[c]
            zero_totals.append({'component': c+1, 'drift': float(abs(final[c])),
                                'bound': float(gamma*absolute_scale[c])})
    element_drifts = []
    for element, atomic_mass in enumerate((.014, .016)):
        weights = model.atom_counts[:, element]*atomic_mass/model.molar_mass
        old = math.fsum(np.sum(before[..., 5:10]*weights, axis=-1).ravel())
        old -= math.fsum(np.sum(before_c[..., 5:10]*weights, axis=-1).ravel())
        new = math.fsum(np.sum(q[..., 5:10]*weights, axis=-1).ravel())
        new -= math.fsum(np.sum(carry[..., 5:10]*weights, axis=-1).ravel())
        element_drifts.append(abs(new-old)/old)
    assert max(element_drifts) <= 1e-14
    np.testing.assert_allclose(high, raw_faces(source, source_c, last_axis), rtol=0, atol=2e-10)
    left = source-source_c
    right = np.roll(left, -1, axis=last_axis)
    assert np.all(flux[..., 5:10] >= -.5*left[..., 5:10])
    assert np.all(flux[..., 5:10] <= .5*right[..., 5:10])
    mass_error = abs(np.sum(flux[..., 5:10], axis=-1)-flux[..., 0])
    mass_scale = np.sum(abs(flux[..., 5:10]), axis=-1)+abs(flux[..., 0])
    assert np.all(mass_error <= 128*np.finfo(float).eps*mass_scale)
    if np.any(before_c) or np.any(carry):
        expected, expected_c = compensated_update(source, source_c, flux, last_axis)
        np.testing.assert_array_equal(q, expected)
        np.testing.assert_array_equal(carry, expected_c)
    else:
        expected = (source+flux)-np.roll(flux, 1, axis=last_axis)
        np.testing.assert_array_equal(q, expected)
    if name in ('constant', 'smooth', 'cancellation', 'carry-only'):
        expected, expected_c = before.copy(), before_c.copy()
        for axis in axes:
            raw = raw_faces(expected, expected_c, axis)
            expected, expected_c = compensated_update(expected, expected_c, raw, axis)
        assert np.max(abs((q-expected)-(carry-expected_c))) <= 2e-10
        if name == 'carry-only' and np.any(before_c):
            scale = abs(before_c).max()
            # Tiny carry is checked at its own operation-derived scale, not 2e-10.
            tiny_bound = 512*np.finfo(float).eps*scale
            assert np.max(abs((q-expected)-(carry-expected_c))) <= tiny_bound
            assert np.any(carry != before_c)
            assert np.any(high != 0)
    if name == 'zero':
        np.testing.assert_array_equal(state[..., 7:10], np.zeros((32, 32, 32, 3)))
    return {**energy, 'pressure_range': [float(pressure.min()), float(pressure.max())],
            'max_closure_error': float(closure), 'max_relative_conservation_drift': max(drifts),
            'zero_totals': zero_totals, 'element_relative_drifts': element_drifts,
            'max_face_mass_error': float(mass_error.max()),
            'component_final_change_max': np.max(abs((q-before)-(carry-before_c)), axis=(0, 1, 2)).tolist()}


def run_pair(directory, name, storage, use_carry, slab, sequence):
    campaign = Path(os.environ.get('ASTR_AIR5_FILTER_CAMPAIGN_ROOT', directory.parent)).resolve(strict=True)
    campaign_budget = 24*1024**3
    used = sum(p.stat().st_size for p in campaign.rglob('*') if p.is_file() and not p.is_symlink())
    if used+64*1024**2 > campaign_budget:
        raise RuntimeError('T3 campaign lacks its next 64 MiB directory budget')
    directory.mkdir()
    model, before, before_c = fixture3(name)
    if not use_carry:
        before_c[:] = 0
    with (directory/'input.bin').open('wb') as stream:
        stream.write(np.array([32]*3, dtype=np.int32).tobytes())
        stream.write(np.moveaxis(before, -1, 0).tobytes(order='F'))
        stream.write(np.moveaxis(before_c, -1, 0).tobytes(order='F'))
    exe = Path(os.environ.get('ASTR_AIR5_FILTER_MULTIDIM_PROBE_EXE',
                              ROOT/'build_gpu_probe/bin/air5_filter_multidim_probe'))
    executable_hash = hashlib.sha256(exe.read_bytes()).hexdigest()
    mpi = os.environ.get('ASTR_MPIEXEC', '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
    ranks = 1 if slab == 0 else 2
    axes = [int(a)-1 for a in str(sequence)]
    products, metrics = [], {'fixture': name, 'storage': storage, 'carry': use_carry,
                             'slab': slab, 'axes': axes, 'backends': {},
                             'executable_sha256': executable_hash}
    for backend in ('cpu', 'gpu'):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if backend == 'cpu' else '0,1',
                   OMP_NUM_THREADS='1')
        command = [mpi, '-np', str(ranks), '--bind-to', 'none', str(exe), backend, storage,
                   'on' if use_carry else 'off', str(directory/'input.bin'), str(directory/backend),
                   str(slab), str(sequence)]
        with (directory/f'{backend}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        log_text = (directory/f'{backend}.log').read_text()
        allocations = re.findall(r'PASS rank,host_bytes,device_bytes\s+(\d+)\s+(\d+)\s+(\d+)', log_text)
        assert len(allocations) == ranks
        assert max(int(value) for row in allocations for value in row[1:]) <= 64*1024**2
        fields = read_products(directory/backend, ranks)
        metrics['backends'][backend] = check_products(model, before, before_c, fields, axes[-1], axes, name)
        metrics['backends'][backend]['controlled_bytes'] = allocations
        limited = int(re.search(r'energy_limited_faces,min_ratio\s+(\d+)', log_text)[1])
        metrics['backends'][backend]['energy_limited_faces'] = limited
        if name in ('energy', 'vibration'):
            assert limited > 0
        if name in ('constant', 'smooth', 'cancellation', 'carry-only'):
            assert limited == 0
        products.append(fields[:2])
    differences = [float(abs(a-b).max()) for a, b in zip(*products)]
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == executable_hash
    assert max(differences) <= 2e-10
    metrics['cpu_gpu_max_abs_difference'] = differences
    size = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    assert size <= 64*1024**2
    metrics['directory_bytes'] = size
    metrics['campaign_root'] = str(campaign)
    metrics['campaign_budget_bytes'] = campaign_budget
    metrics['campaign_directory_bytes'] = sum(p.stat().st_size for p in campaign.rglob('*')
                                             if p.is_file() and not p.is_symlink())
    assert metrics['campaign_directory_bytes'] <= campaign_budget
    (directory/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    return products[0]


@pytest.fixture(scope='module')
def np1_references(tmp_path_factory):
    root = tmp_path_factory.mktemp('np1_matrix')
    paths = {}
    for name in NAMES:
        references = {}
        for storage, carry in (('full', False), ('scalar', False), ('full', True), ('scalar', True)):
            directory = root/f'{name}_{storage}_{int(carry)}'
            values = run_pair(directory, name, storage, carry, 0, 123)
            if carry in references:
                for a, b in zip(values, references[carry]):
                    np.testing.assert_array_equal(a, b)
            references[carry] = values
            paths[name, carry] = directory/'cpu'
    return paths


def test_np1_storage_and_carry(np1_references):
    assert len(np1_references) == 2*len(NAMES)


@pytest.mark.parametrize('name', NAMES)
def test_multiaxis_topology(tmp_path, np1_references, name):
    references = {carry: read_products(np1_references[name, carry], 1)[:2] for carry in (False, True)}
    for slab in (1, 2, 3):
        values = run_pair(tmp_path/f'np2_{slab}_scalar_on', name, 'scalar', True, slab, 123)
        for a, b in zip(values, references[True]):
            np.testing.assert_array_equal(a, b)
        if name in ('smooth', 'trace-step', 'energy', 'carry-only'):
            for storage, carry in (('full', True), ('full', False), ('scalar', False)):
                values = run_pair(tmp_path/f'np2_{slab}_{storage}_{int(carry)}', name, storage, carry, slab, 123)
                for a, b in zip(values, references[carry]):
                    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('name', ('smooth', 'carry-only'))
@pytest.mark.parametrize('axis', (1, 2, 3))
def test_isolated_axis(tmp_path, np1_references, name, axis):
    reference = run_pair(tmp_path/'np1', name, 'scalar', True, 0, axis)
    for slab in (1, 2, 3):
        values = run_pair(tmp_path/f'np2_{slab}', name, 'scalar', True, slab, axis)
        for a, b in zip(values, reference):
            np.testing.assert_array_equal(a, b)
    if name == 'smooth':
        values = run_pair(tmp_path/'np1_full', name, 'full', True, 0, axis)
        for a, b in zip(values, reference):
            np.testing.assert_array_equal(a, b)
