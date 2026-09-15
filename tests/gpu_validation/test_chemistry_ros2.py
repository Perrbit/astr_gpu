from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from air5_radau_reference import Air5RadauReference


ROOT = Path(__file__).resolve().parents[2]


class ChemistryRos2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.exe = Path(cls.directory.name) / "chemistry_ros2_probe"
        sources = [
            ROOT / "src/chemistry_air5_data.F90",
            ROOT / "src/chemistry_model.F90",
            ROOT / "src/chemistry_thermo.F90",
            ROOT / "src/chemistry_relaxation.F90",
            ROOT / "src/chemistry_source.F90",
            ROOT / "src/chemistry_linear6.F90",
            ROOT / "src/chemistry_ros2.F90",
            ROOT / "tests/gpu_validation/chemistry_ros2_probe.F90",
        ]
        result = subprocess.run(
            ["gfortran", "-std=f2008", "-O0", "-g", "-fcheck=all",
             "-ffpe-trap=invalid,zero,overflow", *map(str, sources),
             "-o", str(cls.exe)],
            cwd=cls.directory.name, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        cls.reference = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")

    def run_probe(self, mode, *arguments):
        return subprocess.run(
            [str(self.exe), mode, *map(str, arguments)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_kpp_ros2_21_coefficients_are_frozen(self):
        values = list(map(float, self.run_probe("coefficients").stdout.split()))
        gamma = 1.0 + 1.0 / (2.0**0.5)
        expected = [
            gamma,
            1.0 / gamma,
            -2.0 / gamma,
            3.0 / (2.0 * gamma),
            1.0 / (2.0 * gamma),
            1.0 / (2.0 * gamma),
            1.0 / (2.0 * gamma),
        ]
        for actual, reference in zip(values, expected):
            self.assertAlmostEqual(actual, reference, places=15)

    def test_partial_pivot_lu_solves_and_rejects_singular_systems(self):
        solution_error, residual, solved_status, singular_status, expected_failure = (
            self.run_probe("linear").stdout.split()
        )
        self.assertLessEqual(float(solution_error), 2.0e-14)
        self.assertLessEqual(float(residual), 2.0e-14)
        self.assertEqual(int(solved_status), 0)
        self.assertEqual(int(singular_status), int(expected_failure))

    def test_lu_accepts_well_conditioned_rows_with_different_units(self):
        solution_error, residual, status = self.run_probe("scaled_linear").stdout.split()
        self.assertEqual(int(status), 0)
        self.assertLessEqual(float(solution_error), 2.0e-14)
        self.assertLessEqual(float(residual), 2.0e-14)

    def test_fixed_step_is_admissible_and_preserves_invariants(self):
        values = self.run_probe("fixed_step").stdout.split()
        status = int(values[0])
        minimum_state, mass_drift, nitrogen_drift, oxygen_drift = map(
            float, values[1:5]
        )
        maximum_change, maximum_error = map(float, values[5:7])
        self.assertEqual(status, 0)
        self.assertGreaterEqual(minimum_state, 0.0)
        self.assertLessEqual(mass_drift, 5.0e-13)
        self.assertLessEqual(nitrogen_drift, 5.0e-13)
        self.assertLessEqual(oxygen_drift, 5.0e-13)
        self.assertGreater(maximum_change, 0.0)
        self.assertGreaterEqual(maximum_error, 0.0)

    def test_fixed_step_rejects_nonpositive_and_nonfinite_step_sizes(self):
        actual, expected = [
            list(map(int, line.split()))
            for line in self.run_probe("invalid_step").stdout.splitlines()
        ]
        self.assertEqual(actual, expected)

    def test_adaptive_advance_reaches_requested_interval_with_diagnostics(self):
        values = self.run_probe("adaptive").stdout.split()
        status, accepted, rejected, rhs_calls, jacobian_calls = map(int, values[:5])
        suggested_step, minimum_state, mass_drift = map(float, values[5:8])
        nitrogen_drift, oxygen_drift, maximum_change = map(float, values[8:11])
        self.assertEqual(status, 0)
        self.assertGreaterEqual(accepted, 1)
        self.assertGreaterEqual(rejected, 0)
        self.assertGreaterEqual(rhs_calls, 2 * accepted)
        self.assertGreaterEqual(jacobian_calls, accepted)
        self.assertGreater(suggested_step, 0.0)
        self.assertGreaterEqual(minimum_state, 0.0)
        self.assertLessEqual(mass_drift, 5.0e-13)
        self.assertLessEqual(nitrogen_drift, 5.0e-13)
        self.assertLessEqual(oxygen_drift, 5.0e-13)
        self.assertGreater(maximum_change, 0.0)

    def test_adaptive_advance_fails_closed_at_attempt_limit(self):
        values = self.run_probe("attempt_limit").stdout.split()
        status, expected_status, accepted, rejected = map(int, values[:4])
        state_change = float(values[4])
        self.assertEqual(status, expected_status)
        self.assertEqual(accepted, 0)
        self.assertEqual(rejected, 1)
        self.assertEqual(state_change, 0.0)

    def test_adaptive_advance_rejects_invalid_controls(self):
        actual, expected = [
            list(map(int, line.split()))
            for line in self.run_probe("invalid_control").stdout.splitlines()
        ]
        self.assertEqual(actual, expected)

    def test_fixed_ros2_has_second_order_convergence_in_controlled_state(self):
        rho = 0.01
        momentum = rho * np.array([40.0, -5.0, 2.0])
        rho_species = rho * np.array([0.7653, 0.2347, 0.0, 0.0, 0.0])
        ev = self.reference.ev_from_tv(rho_species, 4000.0)
        state = np.r_[rho_species, ev]
        q5 = self.reference.q5_from_state(
            rho, momentum, rho_species, ev, 4000.0
        )
        atol = np.r_[np.full(5, rho * 1.0e-13), max(abs(ev), 1.0) * 1.0e-13]
        radau = self.reference.integrate_radau(
            rho, momentum, q5, state, 1.0e-3, rtol=1.0e-11, atol=atol
        ).y[:, -1]
        errors = []
        scale = atol + 1.0e-9 * np.abs(radau)
        for steps in (32, 64, 128):
            values = self.run_probe("fixed_trajectory", steps).stdout.split()
            self.assertEqual(int(values[0]), 0)
            final_state = np.array(list(map(float, values[1:])))
            errors.append(np.sqrt(np.mean(((final_state - radau) / scale) ** 2)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        observed_order = np.log(errors[1] / errors[2]) / np.log(2.0)
        self.assertGreaterEqual(observed_order, 1.8)

    def test_adaptive_ros2_matches_independent_radau_trajectory(self):
        rho = 0.05
        momentum = rho * np.array([40.0, -5.0, 2.0])
        rho_species = rho * np.array([0.55, 0.15, 0.10, 0.12, 0.08])
        ev = self.reference.ev_from_tv(rho_species, 1000.0)
        state = np.r_[rho_species, ev]
        q5 = self.reference.q5_from_state(
            rho, momentum, rho_species, ev, 6000.0
        )
        atol = np.r_[np.full(5, rho * 1.0e-13), max(abs(ev), 1.0) * 1.0e-13]
        radau = self.reference.integrate_radau(
            rho, momentum, q5, state, 2.0e-8, rtol=1.0e-11, atol=atol
        ).y[:, -1]
        lines = self.run_probe("adaptive_trajectory").stdout.splitlines()
        status, accepted, rejected, rhs_calls, jacobian_calls = map(
            int, lines[0].split()
        )
        final_state = np.fromstring(lines[1], sep=" ")
        self.assertEqual(status, 0)
        self.assertGreater(accepted, 0)
        self.assertGreaterEqual(rejected, 0)
        self.assertGreaterEqual(rhs_calls, 2 * accepted)
        self.assertGreaterEqual(jacobian_calls, accepted)
        np.testing.assert_allclose(
            final_state[:5] / rho,
            radau[:5] / rho,
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        ros_temperature, ros_tv = self.reference.temperatures(
            rho, momentum, q5, final_state[:, None]
        )
        ref_temperature, ref_tv = self.reference.temperatures(
            rho, momentum, q5, radau[:, None]
        )
        np.testing.assert_allclose(
            ros_temperature, ref_temperature, rtol=1.0e-7, atol=0.0
        )
        np.testing.assert_allclose(ros_tv, ref_tv, rtol=1.0e-7, atol=0.0)

    def test_adaptive_ros2_matches_frozen_component_radau_trajectories(self):
        rho = 0.05
        momentum = rho * np.array([40.0, -5.0, 2.0])
        rho_species = rho * np.array([0.55, 0.15, 0.10, 0.12, 0.08])
        ev = self.reference.ev_from_tv(rho_species, 1000.0)
        state = np.r_[rho_species, ev]
        q5 = self.reference.q5_from_state(
            rho, momentum, rho_species, ev, 6000.0
        )
        atol = np.r_[np.full(5, rho * 1.0e-13), max(abs(ev), 1.0) * 1.0e-13]
        for source_mode in ("chemical", "vt"):
            with self.subTest(source_mode=source_mode):
                radau = self.reference.integrate_radau(
                    rho,
                    momentum,
                    q5,
                    state,
                    2.0e-8,
                    rtol=1.0e-11,
                    atol=atol,
                    source_mode=source_mode,
                ).y[:, -1]
                lines = self.run_probe(
                    "component_trajectory", source_mode
                ).stdout.splitlines()
                status, accepted, rejected, rhs_calls, jacobian_calls = map(
                    int, lines[0].split()
                )
                final_state = np.fromstring(lines[1], sep=" ")
                self.assertEqual(status, 0)
                self.assertGreater(accepted, 0)
                self.assertGreaterEqual(rejected, 0)
                self.assertGreaterEqual(rhs_calls, 2 * accepted)
                self.assertGreaterEqual(jacobian_calls, accepted)
                np.testing.assert_allclose(
                    final_state[:5] / rho,
                    radau[:5] / rho,
                    rtol=1.0e-6,
                    atol=1.0e-10,
                )
                ros_temperature, ros_tv = self.reference.temperatures(
                    rho, momentum, q5, final_state[:, None]
                )
                ref_temperature, ref_tv = self.reference.temperatures(
                    rho, momentum, q5, radau[:, None]
                )
                np.testing.assert_allclose(
                    ros_temperature, ref_temperature, rtol=1.0e-7, atol=0.0
                )
                np.testing.assert_allclose(
                    ros_tv, ref_tv, rtol=1.0e-7, atol=0.0
                )

    def test_representative_state_matrix_matches_radau_and_preserves_invariants(self):
        cases = (
            (350.0, 350.0, 2.0e3, 1.0e-8, 1),
            (4000.0, 350.0, 2.0e3, 1.0e-9, 2),
            (6000.0, 1000.0, 1.0e5, 2.0e-8, 2),
            (7800.0, 1000.0, 9.0e5, 1.0e-10, 2),
            (1000.0, 4000.0, 1.0e4, 1.0e-8, 2),
            (3000.0, 6000.0, 9.0e5, 1.0e-10, 2),
            (7800.0, 7800.0, 9.0e5, 1.0e-10, 2),
        )
        compositions = {
            1: np.array([0.7653, 0.2347, 0.0, 0.0, 0.0]),
            2: np.array([0.55, 0.15, 0.10, 0.12, 0.08]),
        }
        for temperature, tv, pressure, duration, composition_index in cases:
            with self.subTest(temperature=temperature, tv=tv, pressure=pressure):
                lines = self.run_probe(
                    "matrix_trajectory",
                    temperature,
                    tv,
                    pressure,
                    duration,
                    composition_index,
                ).stdout.splitlines()
                header = lines[0].split()
                status, accepted, rejected, rhs_calls, jacobian_calls = map(
                    int, header[:5]
                )
                rho = float(header[5])
                momentum = np.array(list(map(float, header[6:9])))
                q5 = float(header[9])
                initial_state = np.fromstring(lines[1], sep=" ")
                final_state = np.fromstring(lines[2], sep=" ")
                self.assertEqual(status, 0)
                self.assertGreater(accepted, 0)
                self.assertGreaterEqual(rejected, 0)
                self.assertGreaterEqual(rhs_calls, 2 * accepted)
                self.assertGreaterEqual(jacobian_calls, accepted)
                self.assertTrue(np.all(final_state >= 0.0))
                mass_drift, nitrogen_drift, oxygen_drift = (
                    self.reference.invariant_drifts(initial_state, final_state)
                )
                self.assertLessEqual(mass_drift, 5.0e-13)
                self.assertLessEqual(nitrogen_drift, 5.0e-13)
                self.assertLessEqual(oxygen_drift, 5.0e-13)
                atol = np.r_[
                    np.full(5, rho * 1.0e-13),
                    max(abs(initial_state[5]), 1.0) * 1.0e-13,
                ]
                radau = self.reference.integrate_radau(
                    rho,
                    momentum,
                    q5,
                    initial_state,
                    duration,
                    rtol=1.0e-11,
                    atol=atol,
                ).y[:, -1]
                np.testing.assert_allclose(
                    final_state[:5] / rho,
                    radau[:5] / rho,
                    rtol=1.0e-6,
                    atol=1.0e-10,
                )
                ros_temperature, ros_tv = self.reference.temperatures(
                    rho, momentum, q5, final_state[:, None]
                )
                ref_temperature, ref_tv = self.reference.temperatures(
                    rho, momentum, q5, radau[:, None]
                )
                np.testing.assert_allclose(
                    ros_temperature, ref_temperature, rtol=1.0e-7, atol=0.0
                )
                np.testing.assert_allclose(
                    ros_tv, ref_tv, rtol=1.0e-7, atol=0.0
                )
                reconstructed_q5 = self.reference.q5_from_state(
                    rho,
                    momentum,
                    final_state[:5],
                    final_state[5],
                    ros_temperature[0],
                )
                self.assertLessEqual(
                    abs(reconstructed_q5 - q5) / max(abs(q5), 1.0), 5.0e-12
                )
                expected_rho = pressure / (
                    temperature
                    * np.dot(
                        compositions[composition_index], self.reference.gas_constant
                    )
                )
                self.assertLessEqual(abs(rho - expected_rho) / expected_rho, 1.0e-14)

    def test_temperature_and_pressure_domain_endpoints_accept_source_evaluation(self):
        for line in self.run_probe("domain_endpoint_source").stdout.splitlines():
            status, _case, source_scale = line.split()
            self.assertEqual(int(status), 0)
            self.assertTrue(np.isfinite(float(source_scale)))

    def test_ros2_rejects_out_of_domain_state_matrix_without_partial_results(self):
        for line in self.run_probe("invalid_state_matrix").stdout.splitlines():
            values = line.split()
            status, expected, accepted, rejected, rhs_calls, jacobian_calls, _case = (
                map(int, values[:7])
            )
            state_change = float(values[7])
            self.assertEqual(status, expected)
            self.assertEqual(accepted, 0)
            self.assertEqual(rejected, 0)
            self.assertEqual(rhs_calls, 0)
            self.assertEqual(jacobian_calls, 0)
            self.assertEqual(state_change, 0.0)

    def test_stiff_fixed_step_error_decreases_and_reaches_radau_gate(self):
        rho = 0.05
        momentum = rho * np.array([40.0, -5.0, 2.0])
        rho_species = rho * np.array([0.55, 0.15, 0.10, 0.12, 0.08])
        ev = self.reference.ev_from_tv(rho_species, 1000.0)
        state = np.r_[rho_species, ev]
        q5 = self.reference.q5_from_state(
            rho, momentum, rho_species, ev, 6000.0
        )
        atol = np.r_[np.full(5, rho * 1.0e-13), max(abs(ev), 1.0) * 1.0e-13]
        radau = self.reference.integrate_radau(
            rho, momentum, q5, state, 2.0e-8, rtol=1.0e-11, atol=atol
        ).y[:, -1]
        scale = atol + 1.0e-9 * np.abs(radau)
        errors = []
        trajectories = []
        for steps in (2048, 4096, 8192):
            values = self.run_probe("stiff_fixed_trajectory", steps).stdout.split()
            self.assertEqual(int(values[0]), 0)
            final_state = np.array(list(map(float, values[1:])))
            trajectories.append(final_state)
            errors.append(np.sqrt(np.mean(((final_state - radau) / scale) ** 2)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        np.testing.assert_allclose(
            trajectories[-1][:5] / rho,
            radau[:5] / rho,
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        ros_temperature, ros_tv = self.reference.temperatures(
            rho, momentum, q5, trajectories[-1][:, None]
        )
        ref_temperature, ref_tv = self.reference.temperatures(
            rho, momentum, q5, radau[:, None]
        )
        np.testing.assert_allclose(
            ros_temperature, ref_temperature, rtol=1.0e-7, atol=0.0
        )
        np.testing.assert_allclose(ros_tv, ref_tv, rtol=1.0e-7, atol=0.0)


if __name__ == "__main__":
    unittest.main()
