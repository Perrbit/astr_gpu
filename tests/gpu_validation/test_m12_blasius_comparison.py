import unittest

import numpy as np

from analyze_m12_blasius import crossing_height, inlet_station, span_average, wall_quantities


class M12BlasiusComparisonTests(unittest.TestCase):
    def test_crossing_uses_physical_coordinates(self):
        self.assertAlmostEqual(crossing_height(np.array([0., 1., 3.]),
                                              np.array([0., .5, 1.])), 2.96)
        with self.assertRaises(ValueError):
            crossing_height(np.array([0., 1.]), np.array([0., .8]))

    def test_one_inlet_scale_is_derived_without_final_fields(self):
        solution = {'velocity': np.array([0., .5, 1.]),
                    'height': np.array([0., 2., 4.])}
        distance, delta = inlet_station(np.array([0., 1., 2.]),
                                        np.array([0., .5, 1.]), solution, 100.)
        self.assertAlmostEqual(delta, 1.98)
        self.assertAlmostEqual(distance, 12.5)

    def test_periodic_endpoint_is_not_counted_twice(self):
        values = np.array([[0., 2.], [1., 3.], [0., 2.]])
        np.testing.assert_allclose(span_average(values, np.array([0., 1., 2.])), [.5, 2.5])

    def test_wall_stress_includes_normal_velocity_tangential_derivative(self):
        x = np.linspace(0., 2., 9)
        y = np.linspace(0., .4, 9)
        u = np.broadcast_to((3*y + y**2)[:, None], (9, 9))
        temperature = np.broadcast_to((10 + 2*y)[:, None], (9, 9))
        v = np.broadcast_to(.25*x[None, :], (9, 9))
        wall = wall_quantities(x, y, {'velocity_x': u, 'velocity_y': v,
                                     'temperature': temperature})
        np.testing.assert_allclose(wall['du_dy'], 3., atol=1e-12)
        np.testing.assert_allclose(wall['dv_dx'], .25, atol=1e-12)
        np.testing.assert_allclose(wall['dT_dy'], 2., atol=1e-12)
        s = 110.3/594.3
        mu = 10*np.sqrt(10)*(1+s)/(10+s)/1e4
        np.testing.assert_allclose(wall['cf'], 2*mu*3.25, rtol=1e-12)
        np.testing.assert_allclose(wall['qw'], mu*2/(.72*.4*12**2), rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
