"""Small analytic checks of offline FFT normalization and periodic correlations."""
import importlib.util
from pathlib import Path

import numpy as np

MODULE=Path(__file__).resolve().parents[2]/'scripts/gpu_validation/analyze_native_precision_tgv_spectra.py'
spec=importlib.util.spec_from_file_location('tgv_spectra',MODULE)
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(n=16):
    x=np.arange(n)*2*np.pi/n
    u=np.zeros((3,n,n,n))
    u[0]=np.sin(2*x)[:,None,None]
    return u


def test_single_fourier_mode_and_zero_variance():
    u=fixture()
    bins,weights,counts=analysis.spectral_grid(u.shape[1:])
    f,_=analysis.forward_velocity(u,workers=1)
    d=analysis.velocity_diagnostics(f,u.shape[1:],bins,weights)
    np.testing.assert_allclose(d['energy'].sum(),.25,atol=2e-16)
    np.testing.assert_allclose(d['energy'][2],.25,atol=2e-16)
    np.testing.assert_allclose(d['normalized_correlation'][0,0],np.cos(2*np.arange(16)*2*np.pi/16),atol=2e-15)
    np.testing.assert_allclose(d['normalized_correlation'][0,1],1,atol=2e-15)
    assert np.isnan(d['normalized_correlation'][1:]).all()
    assert counts.sum()==16**3


def test_random_field_correlations_match_periodic_roll():
    u=np.random.default_rng(8).normal(size=(3,12,12,12))
    f,means=analysis.forward_velocity(u,workers=1)
    bins,weights,_=analysis.spectral_grid(u.shape[1:])
    d=analysis.velocity_diagnostics(f,u.shape[1:],bins,weights)
    np.testing.assert_allclose(d['energy'].sum(),.5*np.mean(np.sum((u-means[:,None,None,None])**2,axis=0)),rtol=2e-15)
    for c in range(3):
        v=u[c]-means[c]
        for axis in range(3):
            for lag in [0,1,4,6]:
                expected=np.mean(v*np.roll(v,-lag,axis=axis))
                np.testing.assert_allclose(d['correlation'][c,axis,lag],expected,atol=4e-16)
    np.testing.assert_allclose(d['directional_power'].sum(axis=2),d['variance'][:,None]*np.ones((3,3)),rtol=2e-15)


def test_nyquist_mode_is_not_doubled():
    u=np.zeros((3,8,8,8));u[2]=(-1.)**np.arange(8)[None,None,:]
    f,_=analysis.forward_velocity(u,workers=1)
    bins,weights,_=analysis.spectral_grid(u.shape[1:])
    d=analysis.velocity_diagnostics(f,u.shape[1:],bins,weights)
    assert d['energy'][4]==.5
    np.testing.assert_array_equal(d['normalized_correlation'][2,2],(-1.)**np.arange(8))


def test_density_weighted_spectrum_keeps_absolute_energy():
    u=fixture();rho=np.full(u.shape[1:],4.)
    bins,weights,_=analysis.spectral_grid(u.shape[1:])
    e=analysis.density_weighted_spectrum(rho,u,bins,weights,workers=1)
    np.testing.assert_allclose(e.sum(),1,atol=4e-16)


def test_pair_error_and_alignment_have_distinct_meanings():
    u=fixture();f,_=analysis.forward_velocity(u,workers=1)
    bins,weights,_=analysis.spectral_grid(u.shape[1:])
    d=analysis.compare_transforms(f,2*f,bins,weights)
    assert d['spectral_alignment'][2]==1
    np.testing.assert_allclose(d['candidate_energy'][2]/d['reference_energy'][2],4)
    np.testing.assert_allclose(d['error_energy'][2]/d['reference_energy'][2],1)
    np.testing.assert_allclose(d['error_energy'],d['reference_energy']+d['candidate_energy']-2*d['cross_energy'],atol=1e-16)
