"""Read-only attribution of the retained compressive MMS density-RHS failure.

No CFD advancement or production splitting changes. Alternate epsilon values
are algebraic diagnostics at one retained, entirely subsonic x stencil.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_h1_inlet_closure_design import mp_weights_exact
from test_high_order_boundary_closures import boundary_flux_rows, steger_warming_physical_split
from test_high_order_boundary_runtime import manufactured


def analyze(evidence):
    case = evidence/'manufactured/test_variable_pressure_compres0/cpu_128'
    n, j, i = 128, 65, 125
    with h5py.File(case/'datin/flowini3d.h5') as file:
        rho = file['ro'][0, j, 121:129]
        velocity = file['u1'][0, j, 121:129]
    pressure = rho/5.6
    q = np.zeros((8, 5))
    q[:, 0] = rho
    q[:, 1] = rho*velocity
    q[:, 4] = pressure/.4+.5*rho*velocity**2
    with (case/'diagnostics/state.full.step00000000.rk01.rank00000000.bin').open('rb') as file:
        header = np.fromfile(file, np.int32, 4)
        np.testing.assert_array_equal(header, [n, n, 8, 5])
        raw = np.fromfile(file, np.float64).reshape((n+1, n+1, 9, 5), order='F')
    native = float(raw[i, j, 0, 0]*n*n*8/.125)
    reference = float(manufactured(n, n, pressure_compression=True)[4][0, j, i, 0])
    polynomial = np.r_[np.array(boundary_flux_rows()[2], dtype=float).ravel(), 0.]
    difference7 = np.array([-1, 7, -21, 35, -35, 21, -7, 1.])
    interior = np.array(mp_weights_exact(), dtype=float)
    alternatives = []
    for name, spacing in [('native_computational_epsilon', 1/n),
                          ('fixed_physical_epsilon', 1.), ('zero_epsilon_diagnostic', 0.)]:
        split = [steger_warming_physical_split(row, 0, spacing) for row in q]
        plus = np.array([pair[0][0] for pair in split])
        minus = np.array([pair[1][0] for pair in split])
        right = polynomial@plus[::-1]+(polynomial-.125*difference7)@minus[::-1]
        left = interior@plus+interior@minus[::-1]
        rhs = float(-n*(right-left))
        alternatives.append(dict(name=name, physical_epsilon=.04*spacing, rhs=rhs,
                                 error=rhs-reference, residual_vs_native=rhs-native))
    assert abs(alternatives[0]['residual_vs_native']) <= 2e-10, alternatives
    assert np.all((velocity > 0) & (velocity < .5)), velocity
    previous = json.loads((evidence/'pressure_previous_control/control_error.json').read_text())
    convergence = json.loads((case.parent/'pressure_convergence.json').read_text())
    candidate_error = convergence['cases'][-1]['backend']['cpu']['rhs_component_errors']
    np.testing.assert_array_equal(previous['component_errors'], candidate_error)
    return dict(scope='Single saved density-RHS stencil; not convergence or stability acceptance',
                case=str(case), index_zyx=[0, j, i], native_rhs=native,
                analytic_rhs=reference, native_error=native-reference,
                velocity_minus_sound=(velocity-.5).tolist(), alternatives=alternatives,
                previous_candidate_component_errors=previous['component_errors'],
                compression_candidate_component_errors=candidate_error,
                all_stencil_points_subsonic=True,
                interpretation='Native shrinking physical smoothing scale accounts for the observed error. '
                'The supersonic bypass is not entered on this stencil. Alternate epsilon results '
                'are offline only; neither a production correction nor an all-Mach acceptance.')


def analyze_metric_crossing(case):
    """Retained 256-grid counterexample; smooth all-Mach split is offline only."""
    n, j, i = 256, 131, 253
    with h5py.File(case/'datin/flowini3d.h5') as file:
        rho = file['ro'][0, j, n-7:n+1]
        velocity = file['u1'][0, j, n-7:n+1]
    pressure = rho/5.6
    q = np.zeros((8, 5))
    q[:, 0] = rho
    q[:, 1] = rho*velocity
    q[:, 4] = pressure/.4+.5*rho*velocity**2
    with (case/'diagnostics/state.full.step00000000.rk01.rank00000000.bin').open('rb') as file:
        np.testing.assert_array_equal(np.fromfile(file, np.int32, 4), [n, n, 8, 5])
        raw = np.fromfile(file, np.float64).reshape((n+1, n+1, 9, 5), order='F')
    native = float(raw[i, j, 0, 0]*n*n*64)
    reference = float(manufactured(n, n, pressure_compression=True)[4][0, j, i, 0])
    polynomial = np.r_[np.array(boundary_flux_rows()[2], dtype=float).ravel(), 0.]
    difference7 = np.array([-1, 7, -21, 35, -35, 21, -7, 1.])
    interior = np.array(mp_weights_exact(), dtype=float)
    split = [steger_warming_physical_split(row, 0, 1.) for row in q]
    native_plus = np.array([pair[0][0] for pair in split])
    native_minus = np.array([pair[1][0] for pair in split])
    old_split = [steger_warming_physical_split(row, 0, 1/n) for row in q]
    old_plus = np.array([pair[0][0] for pair in old_split])
    old_minus = np.array([pair[1][0] for pair in old_split])
    eigenvalues = np.array([velocity, velocity+.5, velocity-.5])
    positive = .5*(eigenvalues+np.sqrt(eigenvalues**2+.04**2))
    smooth_plus = rho*(positive[0]-(2*positive[0]-positive[1]-positive[2])/2.8)
    alternatives = []
    for name, plus, minus in [('native_supersonic_bypass', native_plus, native_minus),
                              ('old_epsilon_with_bypass_diagnostic', old_plus, old_minus),
                              ('smooth_all_mach_diagnostic', smooth_plus, rho*velocity-smooth_plus)]:
        right = polynomial@plus[::-1]+(polynomial-.125*difference7)@minus[::-1]
        left = interior@plus+interior@minus[::-1]
        rhs = float(-n*(right-left))
        alternatives.append(dict(name=name, rhs=rhs, error=rhs-reference,
                                 residual_vs_native=rhs-native))
    assert abs(alternatives[0]['residual_vs_native']) <= 2e-10, alternatives
    assert np.any(velocity < .5) and np.any(velocity > .5), velocity
    return dict(scope='Single saved sonic-crossing stencil; no CFD advancement or branch correction',
                case=str(case), index_zyx=[0, j, i], native_rhs=native,
                analytic_rhs=reference, physical_epsilon=.04,
                velocity_minus_sound=(velocity-.5).tolist(), alternatives=alternatives)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--crossing-case', type=Path)
    args = parser.parse_args()
    report = analyze(args.evidence)
    if args.crossing_case is not None:
        report['metric_sonic_crossing'] = analyze_metric_crossing(args.crossing_case)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
