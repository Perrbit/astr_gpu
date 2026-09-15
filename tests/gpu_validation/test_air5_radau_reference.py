from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from air5_radau_reference import Air5RadauReference


ROOT = Path(__file__).resolve().parents[2]


class Air5RadauReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.exe = Path(cls.directory.name) / "chemistry_radau_probe"
        sources = [
            ROOT / "src/chemistry_air5_data.F90",
            ROOT / "src/chemistry_model.F90",
            ROOT / "src/chemistry_thermo.F90",
            ROOT / "src/chemistry_relaxation.F90",
            ROOT / "src/chemistry_source.F90",
            ROOT / "tests/gpu_validation/chemistry_radau_probe.F90",
        ]
        result = subprocess.run(
            ["gfortran", "-std=f2008", "-O0", "-g", "-fcheck=all",
             "-ffpe-trap=invalid,zero,overflow", *map(str, sources),
             "-o", str(cls.exe)],
            cwd=cls.directory.name, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def test_independent_rhs_matches_fortran_source(self):
        result = subprocess.run(
            [str(self.exe), "source"], check=True, capture_output=True, text=True
        )
        lines = result.stdout.splitlines()
        rho, q5 = map(float, lines[0].split())
        momentum = np.fromstring(lines[1], sep=" ")
        state = np.fromstring(lines[2], sep=" ")
        fortran_source = np.fromstring(lines[3], sep=" ")
        python_source = self.reference.rhs(0.0, state, rho, momentum, q5)
        source_scale = np.max(np.abs(fortran_source))
        np.testing.assert_array_less(
            np.abs(python_source - fortran_source),
            1.0e-13 * source_scale + 5.0e-12 * np.abs(fortran_source),
        )

    def test_independent_reaction_progress_matches_fortran_diagnostics(self):
        result = subprocess.run(
            [str(self.exe), "source"], check=True, capture_output=True, text=True
        )
        lines = result.stdout.splitlines()
        rho, q5 = map(float, lines[0].split())
        momentum = np.fromstring(lines[1], sep=" ")
        state = np.fromstring(lines[2], sep=" ")
        fortran_progress = np.fromstring(lines[4], sep=" ")
        _source, python_progress = self.reference.rhs_with_progress(
            0.0, state, rho, momentum, q5
        )
        progress_scale = np.max(np.abs(fortran_progress))
        tolerance = (
            1.0e-13 * progress_scale + 5.0e-12 * np.abs(fortran_progress)
        )
        self.assertTrue(
            np.all(np.abs(python_progress - fortran_progress) <= tolerance)
        )

    def test_independent_rhs_splits_chemical_and_vt_sources(self):
        rho, momentum, q5, state, _ = self.make_hot_state()
        coupled = self.reference.rhs(0.0, state, rho, momentum, q5)
        chemical = self.reference.rhs(
            0.0, state, rho, momentum, q5, source_mode="chemical"
        )
        vt = self.reference.rhs(0.0, state, rho, momentum, q5, source_mode="vt")
        np.testing.assert_allclose(coupled, chemical + vt, rtol=2.0e-15, atol=0.0)
        np.testing.assert_array_equal(vt[:5], np.zeros(5))
        with self.assertRaisesRegex(ValueError, "source mode"):
            self.reference.rhs(0.0, state, rho, momentum, q5, source_mode="unknown")

    def test_thermodynamic_roundtrip_is_independent_and_bounded(self):
        rho_species = np.array([0.0275, 0.0075, 0.005, 0.006, 0.004])
        ev = self.reference.ev_from_tv(rho_species, 1000.0)
        tv = self.reference.tv_from_ev(rho_species, ev)
        momentum = np.array([2.0, -0.25, 0.1])
        q5 = self.reference.q5_from_state(
            0.05, momentum, rho_species, ev, 6000.0
        )
        temperature = self.reference.temperature_from_q5(
            0.05, momentum, rho_species, ev, q5
        )
        self.assertLessEqual(abs(tv - 1000.0) / 1000.0, 1.0e-12)
        self.assertLessEqual(abs(temperature - 6000.0) / 6000.0, 1.0e-14)

    def test_out_of_domain_states_fail_without_clipping(self):
        rho_species = np.array([0.0275, 0.0075, 0.005, 0.006, 0.004])
        with self.assertRaisesRegex(ValueError, "temperature"):
            self.reference.ev_from_tv(rho_species, 200.0)
        rho_species[0] = -1.0e-12
        with self.assertRaisesRegex(ValueError, "species"):
            self.reference.ev_from_tv(rho_species, 1000.0)

    def make_hot_state(self):
        rho = 0.05
        momentum = np.zeros(3)
        rho_species = rho * np.array([0.55, 0.15, 0.10, 0.12, 0.08])
        ev = self.reference.ev_from_tv(rho_species, 1000.0)
        state = np.r_[rho_species, ev]
        q5 = self.reference.q5_from_state(rho, momentum, rho_species, ev, 6000.0)
        atol = np.r_[np.full(5, rho * 1.0e-13), max(abs(ev), 1.0) * 1.0e-13]
        return rho, momentum, q5, state, atol

    def test_radau_trajectory_converges_with_tolerance_and_preserves_invariants(self):
        rho, momentum, q5, state, atol = self.make_hot_state()
        loose = self.reference.integrate_radau(
            rho, momentum, q5, state, 2.0e-8, rtol=1.0e-7, atol=atol * 1.0e4
        )
        medium = self.reference.integrate_radau(
            rho, momentum, q5, state, 2.0e-8, rtol=1.0e-9, atol=atol * 1.0e2
        )
        tight = self.reference.integrate_radau(
            rho, momentum, q5, state, 2.0e-8, rtol=1.0e-11, atol=atol
        )
        scale = np.maximum(np.abs(tight.y[:, -1]), atol)
        loose_error = np.linalg.norm((loose.y[:, -1] - tight.y[:, -1]) / scale)
        medium_error = np.linalg.norm((medium.y[:, -1] - tight.y[:, -1]) / scale)
        self.assertLess(medium_error, loose_error)
        mass_drift, nitrogen_drift, oxygen_drift = self.reference.invariant_drifts(
            state, tight.y[:, -1]
        )
        self.assertLessEqual(mass_drift, 5.0e-13)
        self.assertLessEqual(nitrogen_drift, 5.0e-13)
        self.assertLessEqual(oxygen_drift, 5.0e-13)
        temperatures, vibrational_temperatures = self.reference.temperatures(
            rho, momentum, q5, tight.y
        )
        self.assertTrue(np.all((temperatures >= 300.0) & (temperatures <= 8000.0)))
        self.assertTrue(np.all(
            (vibrational_temperatures >= 300.0)
            & (vibrational_temperatures <= 8000.0)
        ))

    def test_radau_accepts_exact_zero_products_without_state_repair(self):
        rho = 0.01
        momentum = np.zeros(3)
        rho_species = rho * np.array([0.7653, 0.2347, 0.0, 0.0, 0.0])
        ev = self.reference.ev_from_tv(rho_species, 4000.0)
        state = np.r_[rho_species, ev]
        q5 = self.reference.q5_from_state(rho, momentum, rho_species, ev, 4000.0)
        atol = np.r_[np.full(5, rho * 1.0e-13), max(abs(ev), 1.0) * 1.0e-13]
        result = self.reference.integrate_radau(
            rho, momentum, q5, state, 1.0e-5, rtol=1.0e-11, atol=atol
        )
        self.assertTrue(np.all(result.y[:, -1] >= 0.0))
        self.assertGreater(result.y[2, -1], 0.0)
        self.assertGreater(result.y[3, -1], 0.0)


if __name__ == "__main__":
    unittest.main()
