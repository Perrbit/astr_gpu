import unittest
import numpy as np

from run_air5_characteristic_backend_gate import compare, TOLERANCE


class BackendComparisonTests(unittest.TestCase):
    def test_identical(self):
        q = np.ones((3, 4, 2, 11))
        self.assertEqual(compare({'phase': q}, {'phase': q})['phase']['max_scaled'], 0.)

    def test_top_mismatch_is_not_excluded(self):
        q = np.ones((3, 4, 2, 11))
        other = q.copy()
        other[-1, -1, 0, 4] += 2*TOLERANCE
        with self.assertRaisesRegex(ValueError, 'phase mismatch'):
            compare({'phase': q}, {'phase': other})

    def test_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            compare({'phase': np.ones((3, 4, 2, 11))}, {'phase': np.ones((2, 4, 2, 11))})

    def test_nan_rejected(self):
        q = np.ones((3, 4, 2, 11))
        other = q.copy()
        other[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            compare({'phase': q}, {'phase': other})


if __name__ == '__main__':
    unittest.main()
