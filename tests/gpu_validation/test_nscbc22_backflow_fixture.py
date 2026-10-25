"""Check the manufactured outlet patches before invoking the flow solver."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from run_nscbc22_backflow_validation import GAMMA, seed_outlet_backflow


class BackflowFixtureTests(unittest.TestCase):
    def test_both_regimes_preserve_thermodynamics_wall_top_and_upstream(self):
        for mach in (.5, 1.5):
            with self.subTest(mach=mach), tempfile.TemporaryDirectory() as name:
                case = Path(name)
                (case/'datin').mkdir()
                x = np.broadcast_to(np.linspace(0., 10., 11), (33, 11))
                with h5py.File(case/'datin/grid.2d', 'w') as grid:
                    grid['x'] = x
                original = dict(ro=np.ones_like(x), p=np.full_like(x, .2), t=np.full_like(x, .8),
                                u1=np.ones_like(x), u2=np.zeros_like(x), u3=np.zeros_like(x))
                original['u1'][0] = 0.
                with h5py.File(case/'datin/flowini2d.h5', 'w') as field:
                    for key, value in original.items():
                        field[key] = value
                seed_outlet_backflow(case, mach)
                with h5py.File(case/'datin/flowini2d.h5') as field:
                    for key in ('ro', 'p', 't', 'u2', 'u3'):
                        np.testing.assert_array_equal(field[key][...], original[key])
                    u = field['u1'][...]
                np.testing.assert_array_equal(u[:, :9], original['u1'][:, :9])
                np.testing.assert_array_equal(u[[0, -1]], original['u1'][[0, -1]])
                np.testing.assert_allclose(u[16, -1]/np.sqrt(GAMMA*.2), -mach, rtol=0, atol=2e-15)
                weight = np.sin(np.pi*np.arange(33)/32.)**2
                expected = (1.-weight)*original['u1'][:, -1]-weight*mach*np.sqrt(GAMMA*.2)
                expected[[0, -1]] = original['u1'][[0, -1], -1]
                np.testing.assert_allclose(u[:, -1], expected, rtol=0, atol=2e-15)


if __name__ == '__main__':
    unittest.main()
