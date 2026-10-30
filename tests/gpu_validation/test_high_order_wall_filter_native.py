"""Actual CPU candidate filter, not a replacement flow integrator."""
from fractions import Fraction
import os
import subprocess

import numpy as np
import pytest
from scipy.linalg import block_diag

from review_high_order_filter_eigenpair import exact_filter
from review_high_order_p6_filter_compatibility import current_normal, sample
from test_high_order_boundary_closures import derivative_matrix
from test_high_order_explicit_wall_candidates import state_projection
from test_high_order_local_filter_ownership import COEFFICIENTS
from test_high_order_wall_filter_coupling import native_filter_matrices
from test_high_order_boundary_geometry import ROOT, procedure


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('wall_filter_native')
    return {kind: native_filter_matrices(root/f'bc{kind}', (8, 9, 32, 64, 128), wall_kind=kind)
            for kind in (0, 41, 42, 421)}


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_actual_cpu_candidate_matches_approved_coefficients_and_deep_rows(native, kind):
    parameters = [Fraction(x) for x in COEFFICIENTS[41 if kind == 41 else 42]]
    for n, actual in native[kind].items():
        reference = np.array(exact_filter(native[0][n], parameters, kind)[:n+1, :n+1], float)
        np.testing.assert_allclose(actual, reference, rtol=0, atol=8*np.finfo(float).eps)
        np.testing.assert_array_equal(actual[5:n-4], native[0][n][5:n-4])
        np.testing.assert_array_equal(actual[[0, n]], native[0][n][[0, n]])
        np.testing.assert_allclose(actual, actual[::-1, ::-1], rtol=0, atol=8*np.finfo(float).eps)
        if kind == 421:
            np.testing.assert_array_equal(actual, native[42][n])


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_actual_cpu_coefficients_keep_original_joint_spectral_gate(native, kind):
    for n, filtering in native[kind].items():
        pd = state_projection(n, 'dirichlet')
        pt = state_projection(n, 'dirichlet' if kind == 41 else 'neumann')
        a = current_normal(n, 0., kind)
        diffusion = current_normal(n, 1., kind)-a
        blocks = [('normal', block_diag(filtering, filtering[1:-1]@pd, filtering[1:-1]@pt), a, diffusion)]
        tangent = pt if kind == 421 else pd
        derivative = n*derivative_matrix(n, True)
        tf = filtering[1:-1]@tangent
        blocks.append(('tangent', tf, np.zeros_like(tf), (derivative@derivative@tangent)[1:-1]))
        for viscosity in (0., 1e-6, 1e-5, 1e-4, 1/1600):
            for cfl in (None, 0., .0001, .001, .005, .01, .025, .05, .1, .15, .2):
                report = sample(blocks, viscosity, cfl, n)
                assert report['passes'], (kind, n, report)


@pytest.mark.parametrize('testing,mode,accepted,enabled',
                         [(True,'hermite7',True,True), (False,'hermite7',False,False),
                          (False,'off',True,False)])
def test_actual_admission_keeps_production_filter_closed(tmp_path, testing, mode, accepted, enabled):
    configure = procedure(ROOT/'src/bc.F90', 'subroutine', 'configure_high_order_wall')
    source = tmp_path/'admission.F90'
    source.write_text('''module commvar
  character(4) :: conschm='643e',difschm='643e',rkscheme='rk3'
  character(8) :: turbmode='none'
  logical :: lcomb=.false.,nondimen=.true.,lfilter=.true.,diffterm=.true.
  logical :: lihomo=.true.,ljhomo=.false.,lkhomo=.true.
end module
module admission_probe
  use boundary_high_order
  implicit none
  integer :: ndims=3,numq=5,num_species=0,num_modequ=0
  integer :: bctype(6)=[1,1,42,42,1,1]
  logical :: lio=.true.
  character(16) :: flowtype='channel'
contains
'''+configure+'''end module
program probe
  use admission_probe
  call configure_high_order_wall()
  print *, 'WALL_FILTER_ADMISSION_ENABLED',high_order_wall_enabled
end program
''')
    command = ['gfortran', '-cpp', '-ffree-line-length-none']
    if testing:
        command.append('-DASTR_BUILD_TESTING')
    command += [str(ROOT/'src/astr_precision.F90'), str(ROOT/'src/boundary_high_order.F90'),
                str(source), '-o', 'probe']
    build = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stdout+build.stderr
    env = dict(os.environ, ASTR_HIGH_ORDER_WALL_MODE=mode)
    result = subprocess.run([str(tmp_path/'probe')], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    (tmp_path/'run.log').write_text(result.stdout+result.stderr)
    if accepted:
        assert result.returncode == 0, result.stdout+result.stderr
        assert 'WALL_FILTER_ADMISSION_ENABLED '+('T' if enabled else 'F') in ' '.join(result.stdout.split())
    else:
        assert result.returncode != 0
        assert 'main filter is validation-only' in result.stdout+result.stderr


@pytest.mark.parametrize('kind', [0,41])
def test_seven_interval_physical_filter_is_outside_existing_template_domain(tmp_path, kind):
    with pytest.raises(AssertionError, match='physical filter consumes exterior ghosts'):
        native_filter_matrices(tmp_path/'unsupported', (7,), wall_kind=kind)
