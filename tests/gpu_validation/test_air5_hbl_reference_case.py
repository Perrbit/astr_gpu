from pathlib import Path

import numpy as np

from tests.gpu_validation.air5_hbl_diagnostics import Air5HblDiagnostics
from tests.gpu_validation.air5_hbl_reference_case import (
    compare_hbl_diagnostics,
    format_comparison_report,
    generate_reference_diagnostics,
    generate_similarity_initial_field,
    htr_contract_on_grid,
    load_diagnostics,
    write_astr_air5_initial_field,
)


ROOT = Path(__file__).resolve().parents[2]
HTR_PROFILE = (
    ROOT / "tests/gpu_validation/data/htr_multispecies_tbl_mach6_similarity.dat"
)
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"


def _diagnostics(*, profile_scale: float = 1.0, trace_offset: float = 0.0):
    profile_mass_fraction = np.zeros((2, 4, 5))
    profile_mass_fraction[..., 0] = 0.77 - 3.0 * trace_offset
    profile_mass_fraction[..., 1] = 0.23
    profile_mass_fraction[..., 2:] = trace_offset
    return Air5HblDiagnostics(
        wall_x=np.array([0.0, 0.5, 1.0]),
        skin_friction=np.array([2.0e-3, 1.8e-3, 1.6e-3]) * profile_scale,
        heat_flux_total=np.array([-2.0e5, -1.8e5, -1.6e5]) * profile_scale,
        heat_flux_translational=np.array([-1.8e5, -1.6e5, -1.4e5]) * profile_scale,
        heat_flux_vibrational=np.array([-2.0e4, -2.0e4, -2.0e4]) * profile_scale,
        heat_flux_species=np.zeros(3),
        profile_x=np.array([0.25, 0.75]),
        profile_y=np.linspace(0.0, 1.0, 4),
        profile_velocity=np.ones((2, 4, 3)) * profile_scale,
        profile_temperature=np.ones((2, 4)) * 2000.0 * profile_scale,
        profile_tv=np.ones((2, 4)) * 1800.0 * profile_scale,
        profile_mass_fraction=profile_mass_fraction,
    )


def test_htr_contract_interpolates_to_requested_grid_without_changing_boundaries():
    y = np.linspace(0.0, 8.821662850689266e-5, 17)

    primitive, boundary, metadata = htr_contract_on_grid(
        HTR_PROFILE, y, pressure=101325.0
    )

    assert primitive.shape == (17, 8)
    assert metadata.x_origin > 0.0
    assert primitive[0, 0] == boundary.wall_velocity
    assert primitive[0, 2] == boundary.wall_temperature
    assert primitive[-1, 0] == boundary.edge_velocity[0]
    assert primitive[-1, 2] == boundary.edge_temperature
    mass_fraction = np.column_stack(
        (1.0 - np.sum(primitive[:, 4:8], axis=1), primitive[:, 4:8])
    )
    assert np.all(mass_fraction >= 0.0)
    np.testing.assert_allclose(np.sum(mass_fraction, axis=1), 1.0, atol=2.0e-15)


def test_reference_comparison_uses_relative_primary_and_absolute_trace_gates():
    reference = _diagnostics()
    candidate = _diagnostics(profile_scale=1.01, trace_offset=5.0e-6)

    result = compare_hbl_diagnostics(
        reference,
        candidate,
        profile_relative_tolerance=0.02,
        wall_relative_tolerance=0.05,
        trace_absolute_tolerance=1.0e-5,
    )

    assert result.passed
    assert result.profile_u_l2_relative < 0.02
    assert result.trace_species_max_absolute == 5.0e-6


def test_reference_comparison_fails_closed_when_trace_gate_is_exceeded():
    reference = _diagnostics()
    candidate = _diagnostics(trace_offset=1.1e-5)

    result = compare_hbl_diagnostics(
        reference,
        candidate,
        profile_relative_tolerance=0.02,
        wall_relative_tolerance=0.05,
        trace_absolute_tolerance=1.0e-5,
    )

    assert not result.passed
    assert result.trace_species_max_absolute > 1.0e-5


def test_reference_comparison_archive_and_report_are_reproducible(tmp_path: Path):
    diagnostics = _diagnostics()
    archive = tmp_path / "diagnostics.npz"
    np.savez(archive, **diagnostics.__dict__)

    loaded = load_diagnostics(archive)
    comparison = compare_hbl_diagnostics(
        diagnostics,
        loaded,
        profile_relative_tolerance=0.02,
        wall_relative_tolerance=0.05,
        trace_absolute_tolerance=1.0e-5,
    )
    report = format_comparison_report(
        comparison,
        profile_relative_tolerance=0.02,
        wall_relative_tolerance=0.05,
        trace_absolute_tolerance=1.0e-5,
    )

    assert comparison.passed
    assert report.startswith("status: pass\n")
    assert "profile_temperature_l2_relative: 0.0000000000000000e+00" in report
    assert "profile_relative_tolerance: 2.0000000000000000e-02" in report


def test_reference_generation_reports_local_coordinates_and_closed_residual():
    y = np.linspace(0.0, 8.821662850689266e-5, 9)

    evidence = generate_reference_diagnostics(
        HTR_PROFILE,
        MECHANISM,
        y=y,
        wall_x=np.array([0.0, 1.0e-6]),
        profile_x=np.array([1.0e-6]),
        pressure=101325.0,
        maximum_streamwise_step=1.0e-6,
        reference_points=17,
        max_function_evaluations=1200,
    )

    np.testing.assert_array_equal(evidence.diagnostics.wall_x, [0.0, 1.0e-6])
    np.testing.assert_array_equal(evidence.diagnostics.profile_x, [1.0e-6])
    assert evidence.diagnostics.profile_velocity.shape == (1, 9, 3)
    assert evidence.diagnostics.profile_mass_fraction.shape == (1, 9, 5)
    assert evidence.scaled_residual_max <= 1.0e-10


def test_reference_generation_allows_astr_edge_above_htr_profile():
    y = np.linspace(0.0, 8.825243000357576e-5, 9)

    evidence = generate_reference_diagnostics(
        HTR_PROFILE,
        MECHANISM,
        y=y,
        wall_x=np.array([0.0, 1.0e-6]),
        profile_x=np.array([1.0e-6]),
        pressure=101325.0,
        maximum_streamwise_step=1.0e-6,
        reference_points=17,
        max_function_evaluations=1200,
    )

    np.testing.assert_array_equal(evidence.diagnostics.profile_y, y)
    np.testing.assert_allclose(
        evidence.diagnostics.profile_velocity[0, -1, 0],
        2905.07,
        atol=2.0e-3,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        evidence.diagnostics.profile_temperature[0, -1],
        450.0,
        atol=2.0e-8,
        rtol=0.0,
    )
    assert evidence.scaled_residual_max <= 1.0e-10


def test_similarity_initial_field_matches_all_boundaries_and_thickens_downstream(
    tmp_path: Path,
):
    x = np.array([0.0, 8.825243000357576e-4], dtype=np.float64)
    y = np.linspace(0.0, 8.825243000357576e-5, 33)

    evidence = generate_similarity_initial_field(
        HTR_PROFILE,
        MECHANISM,
        x=x,
        y=y,
        pressure=101325.0,
    )

    assert evidence.conservative.shape == (2, 33, 11)
    assert np.all(np.isfinite(evidence.conservative))
    np.testing.assert_allclose(
        np.sum(evidence.conservative[:, :, 5:10], axis=2),
        evidence.conservative[:, :, 0],
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    assert evidence.inlet_scaled_error <= 2.0e-13
    assert evidence.end_thickness_scale > 1.3
    np.testing.assert_allclose(
        evidence.conservative[:, 0, :],
        np.repeat(evidence.conservative[0:1, 0, :], x.size, axis=0),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        evidence.conservative[:, -1, (0, 1, 3, 5, 6, 7, 8, 9, 10)],
        np.repeat(
            evidence.conservative[0:1, -1, (0, 1, 3, 5, 6, 7, 8, 9, 10)],
            x.size,
            axis=0,
        ),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    inlet_u = evidence.conservative[0, :, 1] / evidence.conservative[0, :, 0]
    outlet_u = evidence.conservative[-1, :, 1] / evidence.conservative[-1, :, 0]
    inlet_v = evidence.conservative[0, :, 2] / evidence.conservative[0, :, 0]
    outlet_v = evidence.conservative[-1, :, 2] / evidence.conservative[-1, :, 0]
    assert outlet_u[8] < inlet_u[8]
    np.testing.assert_allclose(
        outlet_v[-1],
        inlet_v[-1] / evidence.end_thickness_scale,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    assert outlet_v[0] == 0.0

    output = tmp_path / "air5_hbl_initial_field.dat"
    write_astr_air5_initial_field(output, evidence)
    lines = output.read_text(encoding="ascii").splitlines()
    assert lines[0] == "# ASTR_AIR5_HBL_INITIAL_FIELD_V1"
    assert lines[2] == "2 33"
    data = np.loadtxt(output, comments="#", skiprows=3)
    assert data.shape == (66, 13)
    np.testing.assert_allclose(data[0, :2], [x[0], y[0]], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data[-1, :2], [x[-1], y[-1]], atol=0.0, rtol=0.0)
