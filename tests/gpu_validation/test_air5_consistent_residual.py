from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from air5_radau_reference import Air5RadauReference
from check_air5_consistent_residual import (thermal_contributions, endpoint_terms,
    contact_counterfactuals, mass_gas_interval, symmetric_face_probe)
import check_air5_inlet_velocity_budget as budget
from run_air5_layered_stress_gate import (fixture_fields, require_contact_equilibrium,
    symmetric_species_error, check_shared_face_payloads, ROOT)


def test_mass_closure_does_not_imply_gas_moment_feasibility():
    lower = np.array([.4, .4, 0., 0., 0.])
    upper = np.array([.6, .6, 0., 0., 0.])
    gas = np.array([1., 2., 3., 4., 5.])
    result = mass_gas_interval(lower, upper, 1., gas)
    assert result['mass_feasible']
    np.testing.assert_allclose(result['gas_bounds'], [1.4, 1.6], atol=1e-15, rtol=0)
    assert not mass_gas_interval(lower, upper, 2., gas)['mass_feasible']
    assert not mass_gas_interval(lower, upper, .5, gas)['mass_feasible']
    with pytest.raises(ValueError):
        mass_gas_interval(upper, lower, 1., gas)


def test_mass_gas_extrema_are_permutation_and_orientation_covariant():
    from itertools import permutations
    lower = np.array([-.1, .02, 0., -.3, -1e-12])
    upper = np.array([.5, .8, 0., .1, 2e-12])
    gas = np.array([297., 260., 594., 520., 277.])
    reference = mass_gas_interval(lower, upper, .25, gas)['gas_bounds']
    for order in permutations(range(5)):
        idx = list(order)
        test = mass_gas_interval(lower[idx], upper[idx], .25, gas[idx])
        np.testing.assert_allclose(test['gas_bounds'], reference, rtol=0, atol=1e-13)
    reverse = mass_gas_interval(-upper, -lower, -.25, gas)['gas_bounds']
    np.testing.assert_allclose(reverse, -np.array(reference)[::-1], rtol=0, atol=1e-13)
    for scale in (2.**-100, 2.**100):
        test = mass_gas_interval(lower*scale, upper*scale, .25*scale, gas)
        np.testing.assert_allclose(np.array(test['gas_bounds'])/scale, reference, rtol=0, atol=1e-13)


def test_mass_gas_interval_zero_and_fixed_species():
    import math
    fixed = np.array([.7, .3, 0., 0., 1e-12])
    gas = np.arange(1., 6.)
    result = mass_gas_interval(fixed, fixed, math.fsum(fixed), gas)
    assert result['mass_feasible']
    np.testing.assert_allclose(result['gas_bounds'], [fixed@gas]*2, rtol=0, atol=1e-15)
    assert mass_gas_interval(np.zeros(5), np.zeros(5), 0., gas)['gas_bounds'] == [0., 0.]


def test_trace_relative_gate_does_not_hide_errors_below_bulk_absolute_tolerance():
    initial = np.zeros((2, 2, 2, 11))
    initial[..., 5] = 1.
    initial[..., 9] = 1e-20
    candidate = initial.copy()
    candidate[..., 9] += 1e-25
    with pytest.raises(ValueError, match='species-relative'):
        symmetric_species_error(initial, candidate, initial)
    candidate = initial.copy()
    candidate[..., 9] += 1e-31
    assert symmetric_species_error(initial, candidate, initial)[4] < 1e-9
    candidate[..., 7] = 1e-100
    with pytest.raises(ValueError, match='absent frozen species'):
        symmetric_species_error(initial, candidate, initial)


def test_face_probe_matches_shared_global_face_without_averaging(tmp_path):
    log = tmp_path/'probe.log'
    a = np.zeros(56)
    b = a.copy()
    a[0], b[0] = .96, .97
    b[27] = 3.75e-6
    def line(header, values):
        return 'AIR5_SYMMETRIC_FACE '+' '.join(map(str, (*header, *values)))+'\n'
    log.write_text(line((0, 2, 0, 23, 1), a)+line((0, 2, 1, 0, 0), b))
    result = symmetric_face_probe(log, 24, 2)['comparisons']
    assert len(result) == 1 and result[0]['global_face'] == 23
    assert result[0]['beta'] == [.96, .97]
    assert result[0]['final_max_abs_spread'][4] == 3.75e-6
    assert max(result[0]['base_left_max_abs_spread']) == 0
    log.write_text('AIR5_SYMMETRIC_FACE 0 2 0\n')
    with pytest.raises(ValueError, match='incomplete'):
        symmetric_face_probe(log, 24, 2)


@pytest.mark.parametrize('topology', [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_shared_payload_gate_catches_one_bit_change(tmp_path, topology):
    directory = tmp_path/'validation'
    directory.mkdir()
    count = 2*3*12
    for stage in range(1, 4):
        for axis in range(1, 4):
            first = [np.arange(count, dtype=float)+rank*1000+axis*10000+stage*100000
                     for rank in range(2)]
            last = [v+.5 for v in first]
            for rank in range(2):
                neighbor = 1-rank if topology[axis-1] == 2 else rank
                filename = directory/f'air5.shared_faces.axis{axis}.step00000000.rk{stage:02d}.rank{rank:08d}.bin'
                with filename.open('wb') as stream:
                    np.array([0, stage, rank, axis, 2, 3, 12], np.int32).tofile(stream)
                    np.array([first[rank], last[rank], first[neighbor], last[neighbor]]).tofile(stream)
    assert check_shared_face_payloads(tmp_path, topology) == dict(plane_pairs=36, bitwise_identical=True)
    filename = directory/'air5.shared_faces.axis1.step00000000.rk01.rank00000000.bin'
    with filename.open('r+b') as stream:
        stream.seek(7*4+2*count*8)
        original = np.fromfile(stream, np.float64, 1)
        stream.seek(7*4+2*count*8)
        np.nextafter(original, np.inf).tofile(stream)
    with pytest.raises(ValueError, match='not bitwise identical'):
        check_shared_face_payloads(tmp_path, topology)


def state(thermo, species, t, u):
    q = np.zeros(11)
    q[5:10] = species
    q[0] = sum(species)
    q[1] = q[0]*u
    ev = np.array([thermo.species_vibrational_energy(s, 1500.) for s in range(5)])
    q[10] = q[5:10]@ev
    q[4] = .5*q[0]*u*u+q[10]+q[5:10]@(thermo.formation_energy+t*thermo.cv_tr)
    return q


def test_exact_thermal_identity_includes_kinetic_and_composition_changes():
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    a = state(thermo, [.7, .2, .03, .02, .05], 4000., 2500.)
    b = state(thermo, [.65, .24, .04, .03, .05], 4050., 2520.)
    result = thermal_contributions(a, b, endpoint_terms(a, b), thermo)
    assert abs(result['temperature_difference_k']-50.) < 1e-9
    result = thermal_contributions(a, b, {'one': .2*(b-a), 'two': .8*(b-a)}, thermo)
    assert abs(result['closure_temperature_k']) < 1e-9
    with pytest.raises(ValueError, match='does not close'):
        thermal_contributions(a, b, {'incomplete': .9*(b-a)}, thermo)


def test_contact_fixtures_preserve_periodic_eos_and_control_only_trace():
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    for kind in ('contact', 'low_n2'):
        y, rho, tv = fixture_fields(kind, thermo)
        assert y.min() >= 0
        np.testing.assert_allclose(y.sum(axis=1), 1., atol=1e-15)
        np.testing.assert_array_equal(y[0], y[-1])
        np.testing.assert_allclose(rho*(y@thermo.gas_constant)*4000., 1e5)
        assert np.all(tv == 1500.)
    a, _, _ = fixture_fields('contact', thermo)
    b, _, _ = fixture_fields('contact', thermo, 0.)
    assert a[26, 4] == 1e-12 and b[:, 4].max() == 0
    np.testing.assert_array_equal(a[:, [0, 2, 3]], b[:, [0, 2, 3]])
    c, _, _ = fixture_fields('low_n2', thermo)
    assert c[:, 0].min() == 1e-12 and c[:, 0].max() == 2e-12
    with pytest.raises(ValueError):
        fixture_fields('contact', thermo, -1.)


def test_full_component_budget_reproduces_recorded_ssprk_stages(monkeypatch):
    q0, rhs, dt = np.arange(1., 12.), np.arange(1., 12.)*.01, .5
    before = [q0, q0+dt*rhs, q0+.5*dt*rhs]
    after = [q0+dt*rhs, q0+.5*dt*rhs, q0+dt*rhs]
    arrays = {('pre_chemistry', 1): q0, ('post_chemistry', 2): after[-1],
              ('post_transport', 1): after[-1]}
    for k in range(3):
        arrays[('pre_rhs', k+1)] = before[k]
        arrays[('post_update', k+1)] = after[k]
    monkeypatch.setattr(budget, 'path', lambda c, label, stage, r, s: (label, stage))
    monkeypatch.setattr(budget, '_active_array', lambda key: arrays[key].reshape(1, 1, 1, 11))
    monkeypatch.setattr(budget, 'read_rhs_snapshot', lambda _: SimpleNamespace(header=(0, 0, 0, 11), values=rhs))
    initial, end, terms, _ = budget.step_budget(Path('.'), 0, dt, 0, (0, 0, 0), 1., tuple(range(11)))
    np.testing.assert_array_equal(initial, q0)
    np.testing.assert_allclose(end-initial, dt*rhs)
    np.testing.assert_allclose(terms['convection_raw'], dt*rhs)
    np.testing.assert_allclose(sum(terms.values()), end-initial)
    with pytest.raises(ValueError, match='component'):
        budget.step_budget(Path('.'), 0, dt, 0, (0, 0, 0), 1., (11,))


def test_contact_diagnostic_cannot_promote_the_observed_pressure_defect():
    drift = dict(max_pressure_drift_pa=1.31117, max_temperature_drift_k=.443826,
                 max_velocity_drift_m_s=[.00202037, 0., 0.])
    with pytest.raises(ValueError, match='compatibility failed'):
        require_contact_equilibrium(drift)
    require_contact_equilibrium(dict(max_pressure_drift_pa=1.46e-10,
        max_temperature_drift_k=2.28e-12, max_velocity_drift_m_s=[3.27e-13, 0., 0.]))


def test_energy_only_repair_cannot_preserve_both_pressure_and_temperature():
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    raw = state(thermo, [.78, .22, 0., 0., 0.], 4000., 100.)
    limited = raw.copy()
    limited[5] += 1e-4
    limited[6] -= 1e-4
    report = contact_counterfactuals(raw, limited, thermo)
    assert report['restore_temperature_only']['max_temperature_difference_k'] < 1e-9
    assert report['restore_temperature_only']['max_pressure_difference_pa'] > 1.
    assert report['restore_pressure_only']['max_pressure_difference_pa'] < 1e-7
    assert report['restore_pressure_only']['max_temperature_difference_k'] > .01
    assert report['remove_o2_colimiting_with_n2_closure']['max_temperature_difference_k'] < 1e-9
    assert report['temperature_repair_pressure_identity_error_pa'] < 1e-7
    with pytest.raises(ValueError):
        contact_counterfactuals(raw[:10], limited, thermo)


def test_contact_counterfactual_is_identity_without_limiting():
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    q = state(thermo, [.78, .22, 0., 0., 0.], 4000., 100.)
    report = contact_counterfactuals(q, q, thermo)
    for name in ('original', 'restore_temperature_only', 'restore_pressure_only',
                 'remove_o2_colimiting_with_n2_closure'):
        assert report[name]['max_temperature_difference_k'] < 1e-9
        assert report[name]['max_pressure_difference_pa'] < 1e-7
