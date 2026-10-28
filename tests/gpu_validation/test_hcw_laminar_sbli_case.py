"""Physical input and geometry checks; these do not advance the HCW flow."""
import json
import math

import h5py
import numpy as np
import pytest

import prepare_hcw_laminar_sbli_case as hcw
from prepare_hcw_laminar_sbli_case import (
    case_parameters, cone_shock_angle, cone_surface_angle,
    coordinate_derivative_643e, initial_grid_diagnostics, prepare_case,
)


def test_cone_shooting_satisfies_surface_condition_and_differs_from_wedge():
    beta = math.radians(cone_shock_angle(7.0, 9.0))
    assert math.degrees(cone_surface_angle(7.0, beta)) == pytest.approx(9.0, abs=1.0e-7)
    assert math.asin(1.0 / 7.0) < beta < math.radians(15.0)
    parameters = case_parameters()
    assert parameters["shock"]["turning_deg"] < 9.0


def test_incident_shock_mass_momentum_energy_and_reflection_geometry():
    p = case_parameters()
    s, f, g = (p[key] for key in ("shock", "flow", "geometry"))
    beta = math.radians(s["beta_deg"])
    normal = np.array([math.sin(beta), math.cos(beta)])
    tangent = np.array([math.cos(beta), -math.sin(beta)])
    u1, u2 = np.array([1.0, 0.0]), np.array([s["u2_Uinf"], s["v2_solver_Uinf"]])
    pressure = 1.0 / (1.4 * f["mach"]**2)
    assert normal @ u1 == pytest.approx(s["rho2_rho1"] * (normal @ u2))
    assert tangent @ u1 == pytest.approx(tangent @ u2)
    assert pressure + (normal @ u1)**2 == pytest.approx(
        pressure * s["p2_p1"] + s["rho2_rho1"] * (normal @ u2)**2)
    cp_temperature = 1.0 / (0.4 * f["mach"]**2)
    assert cp_temperature + 0.5 == pytest.approx(cp_temperature * s["T2_T1"] + 0.5 * (u2 @ u2))
    assert g["inlet_x_m"] < g["top_incident_shock_entry_x_m"] < g["nominal_impingement_x_m"]
    assert g["outlet_x_m"] < g["ideal_reflected_shock_top_crossing_x_m"]
    assert s["ideal_regular_reflection_p3_p1"] > s["p2_p1"] > 1.0
    assert g["physical_plate_height_m"] - g["nominal_impingement_x_m"] * math.tan(beta) == pytest.approx(0.0)


def test_previous_height_and_outlet_proposal_is_rejected(monkeypatch):
    monkeypatch.setattr(hcw, "DOMAIN_HEIGHT_M", 0.02)
    monkeypatch.setattr(hcw, "OUTLET_X_M", 1.2)
    with pytest.raises(ValueError, match="Domain conflicts"):
        case_parameters()


def test_643e_coordinate_metrics_against_analytic_stretch():
    index = np.arange(257)
    line = 0.03 * np.sinh(3.5 * index / 256) / np.sinh(3.5)
    expected = 0.03 * 3.5 / 256 * np.cosh(3.5 * index / 256) / np.sinh(3.5)
    actual = coordinate_derivative_643e(line)
    np.testing.assert_allclose(actual[3:-3], expected[3:-3], rtol=1.0e-10)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-4)


def test_static_cfl_and_diffusion_match_uniform_field_rates():
    p = case_parameters()
    x = np.linspace(p["geometry"]["inlet_x_m"], 1.0, 17)
    y, z = np.linspace(0.0, 0.03, 17), np.linspace(0.0, 0.002, 9)
    field = {name: np.full((17, 17), value) for name, value in
             (("ro", 0.8), ("u1", 0.7), ("u2", -0.2), ("u3", 0.1), ("t", 1.25))}
    report = initial_grid_diagnostics(p, x, y, z, field, 0.04)
    spacings = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    expected_rates = (np.array([0.7, 0.2, 0.1]) + np.sqrt(1.25)/7.0) / spacings
    np.testing.assert_allclose(report["directional_acoustic_rates"], expected_rates, rtol=2.0e-12)
    assert report["acoustic_rate_upper_bound"] == pytest.approx(sum(expected_rates))
    mu = 1.25**1.5 * (1.0+110.3/240.0) / (1.25+110.3/240.0)
    expected_diffusion = (1.4/0.72) * mu / (p["flow"]["reynolds_reference"] * 0.8) * sum(spacings**-2)
    assert report["maximum_diffusive_rate"] == pytest.approx(expected_diffusion)
    assert report["selected_dt"] == pytest.approx(min(0.5/sum(expected_rates), 0.25/expected_diffusion))


def test_periodic_extent_smaller_than_fixed_halo_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="periodic z >=5"):
        prepare_case(tmp_path / "invalid", im=32, jm=48, km=4)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("km", [5, 8])
def test_prepared_case_has_joint_nscbc_and_unshocked_developing_blasius(tmp_path, km):
    destination = tmp_path / "review"
    manifest = prepare_case(destination, im=32, jm=48, km=km)
    datin = destination / "datin"
    text = (datin / "input.hcw").read_text()
    assert "# conschm,difschm,rkscheme\n743e,643e,rk3" in text
    assert "# recon_schem,lchardecomp,bfacmpld,shkcrt\n5,t,0.1d0,1.0000000000000000e-03" in text
    assert manifest["numerics"]["reconstruction"] == "MP7-LD"
    assert manifest["numerics"]["recon_schem"] == 5
    boundary = text.split("# bctype\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert boundary[0] == "11,prof"
    assert boundary[1].startswith("22,")
    assert float(boundary[1].split(",")[1]) == pytest.approx(manifest["shock"]["ideal_regular_reflection_p3_p1"])
    assert boundary[2:] == ["41,1.2500000000000000e+00", "52", "1", "1"]
    profile = np.loadtxt(datin / "inlet.prof", skiprows=4)
    with h5py.File(datin / "flowini3d.h5") as field:
        for name in ("ro", "u1", "u2", "u3", "t"):
            values = field[name][...]
            assert values.shape == (km + 1, 49, 33)
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values[0], values[-1])
        for name, column in (("ro", 0), ("u1", 1), ("u2", 2), ("t", 3)):
            np.testing.assert_allclose(field[name][0, :, 0], profile[:, column], atol=2.0e-14, rtol=0.0)
        np.testing.assert_allclose(field["ro"][...] * field["t"][...], 1.0, atol=3.0e-16)
        np.testing.assert_allclose(field["u1"][:, 0, :], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(field["t"][:, 0, :], 1.25, atol=1.0e-12)
        assert not np.allclose(field["u1"][0, :, 0], field["u1"][0, :, -1])
    assert "ASTR_NSCBC_FARFIELD_MODE=sbli_grcbc" in (destination / "runtime.env").read_text()
    assert manifest["boundaries"]["upper_mode"] == "sbli_grcbc"
    assert json.loads((destination / "case_manifest.json").read_text())["run_plan"]["executed"] is False
    assert manifest["grid"]["y_sinh_stretch"] == 3.5
    assert manifest["grid"]["required_z_partitions"] == 1
    diagnostics = manifest["static_diagnostics"]
    assert diagnostics["selected_initial_acoustic_cfl"] <= 0.5
    assert diagnostics["selected_initial_diffusive_number"] <= 0.25 * (1.0+1.0e-14)
    controller_dt = float((datin / "controller").read_text().split("# deltat\n", 1)[1].strip())
    assert controller_dt == manifest["run_plan"]["dt_nondimensional"]
    controller_maxstep = int((datin / "controller").read_text().split(
        "# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg\n", 1)[1].split(",", 1)[0])
    assert controller_maxstep == manifest["run_plan"]["controller_maxstep"] == 29999
    assert manifest["run_plan"]["complete_steps"] == 30000
