#!/usr/bin/env python3
"""Unit tests for the Sod exact-solution validation gate."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np

from tests.gpu_validation.sod_exact_profile import (
    CaseProfile,
    PrimitiveState,
    ValidationLimits,
    compute_diagnostics,
    evaluate_diagnostics,
    format_report,
    read_case_profile,
    solve_riemann,
)


class SodExactProfileTests(unittest.TestCase):
    def test_standard_sod_star_state_matches_reference(self) -> None:
        solution = solve_riemann(
            PrimitiveState(rho=1.0, velocity=0.0, pressure=1.0),
            PrimitiveState(rho=0.125, velocity=0.0, pressure=0.1),
            gamma=1.4,
        )

        self.assertAlmostEqual(solution.star_pressure, 0.30313017805064685, places=12)
        self.assertAlmostEqual(solution.star_velocity, 0.9274526200489499, places=12)
        self.assertAlmostEqual(solution.left_star_density, 0.4263194281784952, places=12)
        self.assertAlmostEqual(solution.right_star_density, 0.2655737117053071, places=12)

        x = np.array([-1.0, 1.0])
        fields = solution.sample(x=x, time=0.2, x0=0.0)
        np.testing.assert_allclose(fields["ro"], [1.0, 0.125], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(fields["u1"], [0.0, 0.0], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(fields["p"], [1.0, 0.1], rtol=0.0, atol=0.0)

    def test_reads_hdf5_profile_without_assuming_x_axis_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / "datin").mkdir()
            (case / "outdat").mkdir()
            x_line = np.linspace(-1.0, 1.0, 5)
            shape = (3, 2, x_line.size)
            x = np.broadcast_to(x_line, shape)

            with h5py.File(case / "datin/grid.h5", "w") as h5:
                h5.create_dataset("x", data=x)
            with h5py.File(case / "outdat/flowfield.h5", "w") as h5:
                h5.create_dataset("time", data=np.array([0.2]))
                h5.create_dataset("ro", data=2.0 + x)
                h5.create_dataset("u1", data=3.0 - x)
                h5.create_dataset("u2", data=np.zeros(shape))
                h5.create_dataset("u3", data=np.zeros(shape))
                h5.create_dataset("p", data=4.0 + 2.0 * x)

            profile = read_case_profile(case)

        np.testing.assert_allclose(profile.x, x_line)
        self.assertEqual(profile.time, 0.2)
        np.testing.assert_allclose(profile.fields["ro"], 2.0 + x_line)
        np.testing.assert_allclose(profile.fields["u1"], 3.0 - x_line)
        np.testing.assert_allclose(profile.fields["p"], 4.0 + 2.0 * x_line)

    def test_exact_sample_has_zero_error_and_grid_scale_discontinuities(self) -> None:
        solution = solve_riemann(
            PrimitiveState(rho=1.0, velocity=0.0, pressure=1.0),
            PrimitiveState(rho=0.125, velocity=0.0, pressure=0.1),
            gamma=1.4,
        )
        x = np.linspace(-2.5, 2.5, 4001)
        exact = solution.sample(x=x, time=0.2)
        zeros = np.zeros_like(x)
        profile = CaseProfile(
            x=x,
            time=0.2,
            fields={**exact, "u2": zeros, "u3": zeros},
        )

        diagnostics = compute_diagnostics(
            profile,
            solution,
            analysis_half_width=2.5,
            exclude_cells=3.0,
        )

        for error in diagnostics.smooth_errors.values():
            self.assertEqual(error.l1, 0.0)
            self.assertEqual(error.l2, 0.0)
            self.assertEqual(error.linf, 0.0)
        for bounds in diagnostics.bounds.values():
            self.assertEqual(bounds.lower_violation, 0.0)
            self.assertEqual(bounds.upper_violation, 0.0)
        self.assertLessEqual(diagnostics.transitions["contact"].thickness_cells, 1.0)
        self.assertLessEqual(diagnostics.transitions["shock"].thickness_cells, 1.0)
        self.assertLessEqual(diagnostics.transitions["contact"].position_error_cells, 0.5)
        self.assertLessEqual(diagnostics.transitions["shock"].position_error_cells, 0.5)

        self.assertEqual(
            evaluate_diagnostics(
                diagnostics,
                ValidationLimits(
                    max_smooth_l1=0.0,
                    max_smooth_l2=0.0,
                    max_smooth_linf=0.0,
                    max_bound_violation=0.0,
                    max_contact_thickness_cells=1.0,
                    max_shock_thickness_cells=1.0,
                    max_position_error_cells=0.5,
                ),
            ),
            [],
        )
        failures = evaluate_diagnostics(
            diagnostics,
            ValidationLimits(max_contact_thickness_cells=0.5),
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("contact thickness", failures[0])
        report = format_report(profile, solution, diagnostics, failures)
        self.assertIn("status: fail", report)
        self.assertIn("[smooth_errors]", report)
        self.assertIn("[bounds]", report)
        self.assertIn("contact thickness_cells=", report)
        self.assertIn(failures[0], report)

    def test_unresolved_density_transition_is_reported_as_a_gate_failure(self) -> None:
        solution = solve_riemann(
            PrimitiveState(rho=1.0, velocity=0.0, pressure=1.0),
            PrimitiveState(rho=0.125, velocity=0.0, pressure=0.1),
            gamma=1.4,
        )
        x = np.linspace(-2.5, 2.5, 101)
        fields = solution.sample(x=x, time=0.01)
        fields["ro"] = np.ones_like(x)
        fields["u2"] = np.zeros_like(x)
        fields["u3"] = np.zeros_like(x)

        diagnostics = compute_diagnostics(
            CaseProfile(x=x, time=0.01, fields=fields),
            solution,
        )
        failures = evaluate_diagnostics(diagnostics, ValidationLimits())

        self.assertFalse(diagnostics.transitions["contact"].resolved)
        self.assertFalse(diagnostics.transitions["shock"].resolved)
        self.assertIn("contact transition unresolved", failures)
        self.assertIn("shock transition unresolved", failures)

    def test_smooth_error_uses_full_analysis_profile_as_normalization_scale(self) -> None:
        solution = solve_riemann(
            PrimitiveState(rho=1.0, velocity=0.0, pressure=1.0),
            PrimitiveState(rho=0.125, velocity=0.0, pressure=0.1),
            gamma=1.4,
        )
        x = np.linspace(-2.5, 2.5, 4001)
        fields = solution.sample(x=x, time=0.2)
        fields["u1"] = fields["u1"] + 1.0e-6
        fields["u2"] = np.full_like(x, 1.0e-6)
        fields["u3"] = np.zeros_like(x)

        diagnostics = compute_diagnostics(
            CaseProfile(x=x, time=0.2, fields=fields),
            solution,
            exclude_cells=200.0,
        )

        expected = 1.0e-6 / solution.star_velocity
        self.assertAlmostEqual(diagnostics.smooth_errors["u1"].linf, expected, places=12)
        exact = solution.sample(x=x, time=0.2)
        momentum_scale = float(np.max(np.abs(exact["ro"] * exact["u1"])))
        expected_q3 = 1.0e-6 / momentum_scale
        self.assertAlmostEqual(diagnostics.smooth_errors["q3"].linf, expected_q3, places=12)


if __name__ == "__main__":
    unittest.main()
