"""Check the diagnosis arithmetic separately from solver acceptance."""

import unittest

import numpy as np

from analyze_m12_sensor_diagnostics import replay, validate_snapshot


class SensorDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.header = np.array([3, 3, 3, 1], dtype=np.int32)
        self.gradient = np.zeros((4, 4, 4, 3, 3))
        self.pressure = np.ones((6, 6, 6))

    def test_uniform_pressure_produces_exact_zero_sensor(self):
        self.gradient[..., 0, 0] = 0.2
        result = replay(self.header, self.gradient, self.pressure)
        np.testing.assert_array_equal(result['sensor'], 0.)

    def test_near_zero_gradient_regularization_is_not_removed(self):
        self.gradient[..., 0, 0] = 1e-17
        result = replay(self.header, self.gradient, self.pressure)
        expected = 1e-34 / (1e-34 + 1e-30)
        np.testing.assert_allclose(result['factor'], expected, rtol=1e-15, atol=0.)

    def test_unused_halo_corner_is_not_a_sensor_input(self):
        self.pressure[0, 0, 0] = np.nan
        result = replay(self.header, self.gradient, self.pressure)
        self.assertTrue(np.isfinite(result['sensor']).all())

    def test_nonfinite_used_face_halo_fails(self):
        self.pressure[1, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'pressure used'):
            replay(self.header, self.gradient, self.pressure)

    def test_mpi_face_consumes_halo_not_physical_clamp(self):
        self.gradient[..., 0, 0] = .2
        self.pressure[0, 1:5, 1:5] = 2.
        physical = replay(np.array([3, 3, 3, 1, 4, 4, 0]), self.gradient, self.pressure)
        partition = replay(np.array([3, 3, 3, 1, 2, 4, 0]), self.gradient, self.pressure)
        self.assertEqual(physical['sensor'][0, 1, 1], 0.)
        self.assertGreater(partition['sensor'][0, 1, 1], 0.)

    def test_d6_rejects_wrong_output_even_when_masks_match(self):
        mask = np.zeros(4, dtype=np.int8)
        with self.assertRaisesRegex(ValueError, 'explain sensor'):
            validate_snapshot(0., 0., [0., 3e-10], (mask, mask))

    def test_d6_rejects_mismatched_masks_even_when_replay_matches(self):
        with self.assertRaisesRegex(ValueError, 'activation masks'):
            validate_snapshot(0., 0., [0., 0.], (np.zeros(4), np.ones(4)))

    def test_d6_retains_input_accuracy_gate(self):
        mask = np.zeros(4, dtype=np.int8)
        with self.assertRaisesRegex(ValueError, 'gradient/used-pressure'):
            validate_snapshot(3e-10, 0., [0., 0.], (mask, mask))

    def test_nonfinite_replay_receipt_fails(self):
        mask = np.zeros(4, dtype=np.int8)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_snapshot(0., 0., [0., np.nan], (mask, mask))


if __name__ == '__main__':
    unittest.main()
