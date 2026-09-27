import unittest
import numpy as np
from run_air5_characteristic_source_refinement import refinement


class RefinementTests(unittest.TestCase):
    def test_decreasing(self):
        q=np.ones((2,3,11))
        self.assertTrue(refinement(q+.04,q+.01,q,np.ones(11))['passed'])

    def test_growing_component_rejected(self):
        q=np.ones((2,3,11)); fine=q.copy(); fine[...,2]+=.2
        self.assertFalse(refinement(q+.04,q+.01,fine,np.ones(11))['passed'])

    def test_roundoff_not_order(self):
        q=np.ones((2,3,11))
        self.assertFalse(refinement(q,q,q,np.ones(11))['passed'])

    def test_nonfinite_rejected(self):
        q=np.ones((2,3,11)); bad=q.copy(); bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            refinement(q,q,bad,np.ones(11))
