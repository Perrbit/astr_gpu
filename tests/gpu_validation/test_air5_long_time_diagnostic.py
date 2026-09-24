"""Metrics and warning contract; no CFD integrator is implemented here."""

import unittest

import numpy as np

from air5_radau_reference import Air5RadauReference
from run_air5_long_time_diagnostic import ROOT, metrics, volume_mean


class LongTimeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
        shape = (7, 7, 33)
        self.fields = {key: np.full(shape, value) for key, value in
                       dict(ro=.04, u1=3500., u2=0., u3=0., p=5000., t=500., tv=500.).items()}
        for i, y in enumerate((.765297, .234699, 1e-6, 1e-6, 2e-6), 1):
            self.fields[f'sp{i:03d}'] = np.full(shape, y)
        self.reference = (3500., 5000., .04)

    def measure(self):
        return metrics(self.fields, self.thermo, self.reference)

    def test_planar_is_zero(self):
        result = self.measure()
        self.assertEqual(result['max_transverse_velocity_over_uinf'], 0.)
        self.assertEqual(result['pressure_transverse_rms_over_pinf'], 0.)
        self.assertFalse(result['extrusion_warning'])

    def test_extrusion_is_warning_not_failure(self):
        self.fields['u2'][2, 2, 12] = 1e-6
        self.assertTrue(self.measure()['extrusion_warning'])

    def test_hard_failures(self):
        for key, value in (('sp003', -1e-30), ('ro', 0.), ('t', 299.),
                           ('tv', 8001.), ('p', 1e6+1), ('u2', np.nan)):
            with self.subTest(key=key):
                original = self.fields[key][0, 0, 0]
                self.fields[key][0, 0, 0] = value
                with self.assertRaises(ValueError):
                    self.measure()
                self.fields[key][0, 0, 0] = original

    def test_periodic_duplicate_excluded_from_rms_not_max(self):
        self.fields['u2'][-1, -1, :] = 1.
        result = self.measure()
        self.assertEqual(result['rms_transverse_velocity_over_uinf'], 0.)
        self.assertGreater(result['max_transverse_velocity_over_uinf'], 0.)

    def test_physical_normalization(self):
        self.fields['u2'][:] = 3.
        self.fields['u3'][:] = 4.
        result = self.measure()
        self.assertAlmostEqual(result['rms_transverse_velocity_over_uinf'], 5/3500)
        self.assertAlmostEqual(result['transverse_kinetic_energy_over_dynamic_pressure'], 25/3500**2)

    def test_x_endpoint_quadrature(self):
        a = np.zeros((6, 6, 33))
        a[:, :, 0] = 1.
        self.assertEqual(volume_mean(a), .5/32)


if __name__ == '__main__':
    unittest.main()
