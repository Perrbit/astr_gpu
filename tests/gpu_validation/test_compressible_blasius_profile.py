import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid

import generate_compressible_blasius_profile as profile
from check_blasius_mass_continuity import check_resolution


class BlasiusProfileTests(unittest.TestCase):
    def test_optional_transport_parameters_preserve_default(self):
        baseline = profile.solve_profile(2.0, 288.0, 1.676194, 20.0, 801)
        explicit = profile.solve_profile(2.0, 288.0, 1.676194, 20.0, 801,
                                         prandtl=0.72, sutherland_temperature=110.3)
        for first, second in zip(baseline, explicit):
            np.testing.assert_array_equal(first, second)
        other = profile.solve_profile(2.0, 288.0, 1.676194, 20.0, 801,
                                      prandtl=0.71, sutherland_temperature=110.4)
        self.assertGreater(np.max(np.abs(other[4] - baseline[4])), 1e-5)
        for args in ({"prandtl": 0.0}, {"sutherland_temperature": -1.0}):
            with self.assertRaises(ValueError):
                profile.solve_profile(2.0, 288.0, 1.676194, 20.0, 801, **args)

    def test_manufactured_variable_temperature(self):
        eta = np.linspace(0.0, 1.0, 101)
        temperature = 1.0 + 0.5 * eta
        y = eta + 0.25 * eta**2
        _, _, v, _, _ = profile.map_similarity_profile(
            np.append(y, 2.0), eta, 0.5 * eta**2, eta, temperature, 2.0, 1.0
        )
        # u=eta, f=eta^2/2 gives u*integral(T)-T*f=eta^2/2 exactly.
        np.testing.assert_allclose(v[:-1], eta**2 / 4.0, atol=1e-14, rtol=0.0)

    def test_constant_temperature_limit(self):
        eta = np.linspace(0.0, 1.0, 101)
        for constant in (1.0, 1.5):
            with self.subTest(temperature=constant):
                _, _, v, _, _ = profile.map_similarity_profile(
                    np.append(constant * eta, 2.0), eta, 0.5 * eta**2,
                    eta, np.full_like(eta, constant), 2.0, 1.0
                )
                np.testing.assert_allclose(v[:-1], constant * eta**2 / 4.0, atol=1e-14)

    def test_mass_continuity_converges(self):
        coarse, fine = (check_resolution(n) for n in (2001, 4001))
        self.assertLess(fine["existing_mass_residual"], 1e-7)
        self.assertLess(fine["existing_mass_residual"], 0.4 * coarse["existing_mass_residual"])

    def test_written_inlet_and_initial_field(self):
        for mach, wall in ((2.0, 1.676194), (5.0, 0.8)):
            with self.subTest(mach=mach), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reynolds, station, reference_temperature = 950.0, 10.0, 288.0
                eta, f, u, _, temperature = profile.solve_profile(
                    mach, reference_temperature, wall, 20.0, 801
                )
                height = cumulative_trapezoid(temperature, eta, initial=0.0)
                yline = np.linspace(0.0, 2.0 * height[-1], 257)
                xline = np.array([0.0, 1.0, 2.0])
                z, y, x = np.meshgrid([0.0, 1.0], yline, xline, indexing="ij")
                with h5py.File(root / "grid.h5", "w") as handle:
                    for name, data in (("x", x), ("y", y), ("z", z)):
                        handle[name] = data
                profile.write_profile(
                    root / "profile", root / "grid.h5", mach, reynolds,
                    reference_temperature, wall, station, "provided", "provided",
                    False, 0.0, 32.58, 20.0, 801,
                )
                profile.write_similarity_initial_field(
                    root / "initial.h5", root / "grid.h5", mach, reynolds,
                    reference_temperature, wall, -station, 20.0, 801,
                    False, 32.58, 40.0, 115.0, 0.0,
                )
                inlet = np.loadtxt(root / "profile", skiprows=4)
                with h5py.File(root / "initial.h5", "r") as handle:
                    for column, name in enumerate(("ro", "u1", "u2", "t")):
                        np.testing.assert_allclose(handle[name][0, :, 0], inlet[:, column], atol=1e-13)
                        np.testing.assert_array_equal(handle[name][0], handle[name][1])
                        self.assertTrue(np.isfinite(handle[name][:]).all())
                    for i, xpos in enumerate(xline):
                        scale = np.sqrt(2.0 * (station + xpos) / reynolds)
                        expected = np.interp(yline / scale, height, u * height - temperature * f)
                        expected /= np.sqrt(2.0 * reynolds * (station + xpos))
                        np.testing.assert_allclose(handle["u2"][0, :, i], expected, atol=1e-13)
                    np.testing.assert_allclose(handle["u2"][:, 0, :], 0.0, atol=1e-14)
                    np.testing.assert_allclose(handle["ro"][:] * handle["t"][:], 1.0, atol=1e-14)
                np.testing.assert_allclose(inlet[:, 4], inlet[:, 0] * inlet[:, 3] / (1.4 * mach**2))


if __name__ == "__main__":
    unittest.main()
