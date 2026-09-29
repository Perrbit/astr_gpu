import numpy as np
import pytest

from check_air5_precursor_admission import thickness, monitor_windows


def test_integral_thickness_constant_density_linear_velocity():
    y = np.linspace(0, 1, 10001)
    r = thickness(y, y, y*y, 1., 1., y)
    assert r['delta99'] == pytest.approx(.99)
    assert r['displacement_thickness'] == pytest.approx(.5)
    assert r['momentum_thickness'] == pytest.approx(1/6)
    assert r['shape_factor'] == pytest.approx(3.)


def test_outermost_crossing_and_invalid_coordinates():
    y = np.arange(5.)
    u = np.array([0., 1.1, .95, 1., 1.])
    assert thickness(y, u, u*u, 1., 1., u)['delta99'] == pytest.approx(2.8)
    with pytest.raises(ValueError):
        thickness(y[::-1], u, u*u, 1., 1., u)


def test_monitor_windows_reject_incomplete_profile():
    rows = np.zeros((5, 20))
    rows[:, 1] = [0., 0., 0., 1e-6, 1e-6]
    rows[:, 4] = [0, 1, 2, 0, 1]
    with pytest.raises(ValueError, match='incomplete'):
        monitor_windows(rows, np.zeros((2, 9)), [0, 1, 2], 1.)


def test_stationary_uniform_monitor_has_zero_window_drift():
    rows, walls = [], []
    for t in np.arange(8)*1e-6:
        for j in range(3):
            row = np.zeros(20)
            row[1], row[4], row[6] = t, j, j/2
            row[8], row[9], row[12], row[13] = 1., j/2, 1500., 1500.
            rows.append(row)
        wall = np.zeros(9)
        wall[1], wall[6], wall[7] = t, 2., -3.
        walls.append(wall)
    r = monitor_windows(np.array(rows), np.array(walls), [0.,4.,8.], 1.)[0]
    assert [w['samples'] for w in r['windows']] == [4,4]
    assert r['changes'][0]['max_du_over_uinf'] == 0.
    assert r['changes'][0]['relative_to_later'] == {'mean_tau':0., 'mean_heat':0.}
    with pytest.raises(ValueError, match='fully sampled'):
        monitor_windows(np.array(rows), np.array(walls), [0.,4.,12.], 1.)
