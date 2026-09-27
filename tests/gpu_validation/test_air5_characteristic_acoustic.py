import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from run_air5_characteristic_acoustic import read_monitor, read_acoustic_probe
from check_air5_characteristic_acoustic_matrix import CASES, compare
from run_air5_characteristic_acoustic_sequence import sequence


class AcousticMonitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'monitor.dat'
        self.dt = 5e-9
        self.records = np.zeros(3, dtype=np.dtype({
            'names': ['step', 'values'], 'formats': ['<i4', ('<f8', 10)],
            'offsets': [0, 4], 'itemsize': 88}))
        self.records['step'] = np.arange(3)
        self.records['values'][:, 0] = np.arange(3)*self.dt
        self.records['values'][:, 2] = .01
        self.records['values'][:, 5] = 21622.

    def test_layout_and_current_step_time(self):
        self.records.tofile(self.path)
        actual = read_monitor(self.path, 3, self.dt)
        np.testing.assert_array_equal(actual[:, 2], [.01]*3)
        np.testing.assert_array_equal(actual[:, 5], [21622.]*3)

    def test_reject_truncation(self):
        self.records[:2].tofile(self.path)
        with self.assertRaisesRegex(ValueError, 'length'):
            read_monitor(self.path, 3, self.dt)

    def test_reject_shifted_phase(self):
        self.records['values'][:, 0] += self.dt
        self.records.tofile(self.path)
        with self.assertRaisesRegex(ValueError, 'phase'):
            read_monitor(self.path, 3, self.dt)

    def test_reject_bad_index(self):
        self.records['step'][1] = 3
        self.records.tofile(self.path)
        with self.assertRaisesRegex(ValueError, 'indices'):
            read_monitor(self.path, 3, self.dt)

    def test_reject_nonfinite_pressure(self):
        self.records['values'][1, 5] = np.nan
        self.records.tofile(self.path)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            read_monitor(self.path, 3, self.dt)

    def test_gpu_probe_layout_and_phase(self):
        data = np.column_stack((self.records['step'], self.records['values'][:, :6]))
        np.savetxt(self.path, data)
        np.testing.assert_array_equal(read_acoustic_probe(self.path, 3, self.dt), data[:, 1:])
        data[:, 1] += self.dt
        np.savetxt(self.path, data)
        with self.assertRaisesRegex(ValueError, 'phase'):
            read_acoustic_probe(self.path, 3, self.dt)


class AcousticMatrixTests(unittest.TestCase):
    def fixtures(self):
        entries = {}
        for name in CASES:
            factor = {'half_tau': .5, 'double_tau': 2.}.get(name, 1.)
            mode = 'prescribed' if name == 'prescribed' else 'characteristic'
            ny = 128 if name == 'coarse' else 256
            dt = 2.5e-9 if name == 'half_dt' else 5e-9
            extended = name == 'extended'
            tau = factor*.01/778.
            meta = dict(executable_sha256='fixture', rho=.05, pressure=21622., sound_speed=778.,
                gamma=1.4, epsilon=1e-5, probe_xyz=[.04, .00875, .001], sampling_phase='step-start',
                incident_center=1.6e-6, reflected_center=4.8e-6, window_halfwidth=1.2e-6,
                ny=ny, dt=dt, updates=round(7.5e-6/dt), tau_factor=factor, mode=mode, extended=extended,
                grid_intervals=[16, ny*(2 if extended else 1), 8], domain=[.08, .02 if extended else .01, .002],
                tau=tau, environment=dict(ASTR_AIR5_SOURCE_MODE='frozen', ASTR_AIR5_COMPENSATION='on',
                    ASTR_AIR5_TOP_MODE=mode, ASTR_AIR5_TOP_TAU=str(tau)))
            result = dict(R_peak=.02, R_L2=.02, incident_peak=.4, incident_L2=.0003,
                          maximum_sequential_closure=2e-16)
            entries[name] = (meta, result)
        return entries

    def test_complete_matrix(self):
        self.assertTrue(compare(self.fixtures())['passed'])

    def test_each_fixed_threshold_fails_closed(self):
        entries = self.fixtures()
        for name, key, value in (('baseline', 'R_peak', .051), ('half_tau', 'R_L2', .06),
                                 ('coarse', 'R_L2', .045), ('half_dt', 'R_L2', .026),
                                 ('extended', 'incident_peak', .5),
                                 ('baseline', 'maximum_sequential_closure', float('nan'))):
            with self.subTest(name=name, key=key):
                modified = deepcopy(entries)
                modified[name][1][key] = value
                with self.assertRaises(ValueError):
                    compare(modified)

    def test_reject_incompatible_control(self):
        entries = self.fixtures()
        entries['extended'][0]['tau'] *= 2
        with self.assertRaisesRegex(ValueError, 'relaxation'):
            compare(entries)

    def test_reject_missing_case(self):
        entries = self.fixtures()
        entries.pop('prescribed')
        with self.assertRaisesRegex(ValueError, 'seven-case'):
            compare(entries)

    def test_reject_unbridged_backend_mix(self):
        entries = self.fixtures()
        entries['half_tau'][0]['use_gpu'] = True
        with self.assertRaisesRegex(ValueError, 'mixed backends'):
            compare(entries)

    def test_prefix_never_claims_full_matrix(self):
        entries = self.fixtures()
        for length in range(1, len(CASES)):
            report = compare({key: entries[key] for key in CASES[:length]}, partial=True)
            self.assertFalse(report['complete'])

    def test_prefix_checks_pair_before_next_case(self):
        entries = self.fixtures()
        entries['half_tau'][1]['R_L2'] = .06
        with self.assertRaisesRegex(ValueError, 'difference'):
            compare({key: entries[key] for key in CASES[:2]}, partial=True)

    def test_prefix_rejects_skipped_gate(self):
        entries = self.fixtures()
        with self.assertRaises(ValueError):
            compare({key: entries[key] for key in ('baseline', 'double_tau')}, partial=True)

    def test_sequence_does_not_start_after_failed_pair(self):
        entries = self.fixtures()
        entries['half_tau'][1]['R_L2'] = .06
        module = 'run_air5_characteristic_acoustic_sequence'
        with tempfile.TemporaryDirectory() as directory:
            with patch(module+'.load_case', side_effect=[entries['baseline'], entries['half_tau']]), \
                 patch(module+'.prepare') as prepare_next, patch(module+'.run') as run_next:
                with self.assertRaisesRegex(ValueError, 'difference'):
                    sequence(Path('baseline'), Path('half_tau'), Path(directory)/'sequence')
                prepare_next.assert_not_called()
                run_next.assert_not_called()

    def test_sequence_order_and_complete_report(self):
        entries = self.fixtures()
        entries['baseline'][0]['executable'] = '/unused/astr'
        module = 'run_air5_characteristic_acoustic_sequence'
        with tempfile.TemporaryDirectory() as directory:
            with patch(module+'.load_case', side_effect=[entries[key] for key in CASES]), \
                 patch(module+'.prepare', side_effect=[entries[key][0] for key in CASES[2:]]) as prep, \
                 patch(module+'.run') as advance:
                report = sequence(Path('baseline'), Path('half_tau'), Path(directory)/'sequence')
                self.assertTrue(report['complete'])
                self.assertEqual([call.args[0].name for call in prep.call_args_list], list(CASES[2:]))
                self.assertEqual(advance.call_count, 5)

    def test_reuse_extended_rechecks_without_launch(self):
        entries = self.fixtures()
        entries['baseline'][0]['executable'] = '/unused/astr'
        module = 'run_air5_characteristic_acoustic_sequence'
        pending = [key for key in CASES[2:] if key != 'extended']
        with tempfile.TemporaryDirectory() as directory:
            with patch(module+'.load_case', side_effect=[entries[key] for key in CASES]) as load, \
                 patch(module+'.prepare', side_effect=[entries[key][0] for key in pending]) as prep, \
                 patch(module+'.run') as advance:
                report = sequence(Path('baseline'), Path('half_tau'), Path(directory)/'sequence',
                                  reuse_extended=Path('existing_extended'))
                self.assertTrue(report['complete'])
                self.assertEqual([call.args[0].name for call in prep.call_args_list], pending)
                self.assertEqual(load.call_args_list[3].args[0], Path('existing_extended'))
                self.assertEqual(advance.call_count, 4)

    def test_reused_extended_mismatch_stops_next_case(self):
        entries = self.fixtures()
        entries['baseline'][0]['executable'] = '/unused/astr'
        entries['extended'][0]['executable_sha256'] = 'different'
        module = 'run_air5_characteristic_acoustic_sequence'
        with tempfile.TemporaryDirectory() as directory:
            with patch(module+'.load_case', side_effect=[entries[key] for key in CASES[:4]]), \
                 patch(module+'.prepare', return_value=entries['double_tau'][0]) as prep, \
                 patch(module+'.run') as advance:
                with self.assertRaisesRegex(ValueError, 'executable_sha256'):
                    sequence(Path('baseline'), Path('half_tau'), Path(directory)/'sequence',
                             reuse_extended=Path('existing_extended'))
                self.assertEqual(prep.call_count, 1)
                self.assertEqual(advance.call_count, 1)


if __name__ == '__main__':
    unittest.main()
