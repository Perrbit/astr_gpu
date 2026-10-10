"""Configuration checks for the matched local boundary experiment."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from run_m12_boundary_compare import compare_pair, configuration, environment


class M12BoundaryCompareTest(unittest.TestCase):
    def test_long_run_checkpoint_and_total_step_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case/'datin').mkdir()
            (case/'outdat').mkdir()
            (case/'datin/input.dat').write_text('# bctype\n11,prof\n21\n41,10.d0\n50\n1\n1\n')
            (case/'datin/controller').write_text('# maxstep,feqchkpt\n9,1,9999,9999,1,9999\n')
            configuration(case,'nscbc',(450,130,96),10000,checkpoint_interval=1000)
            self.assertIn('9999,1,999999,999999,1,999999', (case/'datin/controller').read_text())
            output = (case/'datin/input.output').read_text()
            self.assertIn("enabled=t,mode='steps'", output)
            self.assertIn('interval_steps=1000,keep=2,initial_frame=f', output)
            self.assertIn('interval_steps=10000,initial_frame=f,final_frame=t', output)
            self.assertIn('host_budget_bytes=1073741824', output)

    def test_restart_requires_checkpoint_schedule(self):
        with self.assertRaisesRegex(ValueError,'restart requires'):
            configuration(Path('/unused'),'nscbc',(64,32,24),10,restore=Path('/checkpoint'))

    def test_reporting_does_not_enable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case/'datin').mkdir()
            (case/'outdat').mkdir()
            (case/'datin/input.dat').write_text('# bctype\n11,prof\n21\n41,10.d0\n50\n1\n1\n')
            (case/'datin/controller').write_text('# maxstep,feqchkpt\n9,1,9999,9999,1,9999\n')
            configuration(case, 'nscbc', (450,130,96), 2000)
            self.assertIn('1999,1,999999,999999,1,999999', (case/'datin/controller').read_text())
            self.assertIn('&checkpoint\n enabled=f', (case/'datin/input.output').read_text())
            self.assertIn('22,1.d0\n41,10.d0\n52', (case/'datin/input.dat').read_text())
            self.assertTrue((case/'outdat/new').is_dir())

    def test_same_random_mode_for_matched_pair(self):
        controls = environment('2,1,1')
        self.assertEqual(controls['ASTR_WALL_BLOWING_MODE'], 'legacy_random')
        self.assertEqual(controls['ASTR_GPU_PRECISION_MODE'], 'fp64')
        self.assertEqual(controls['ASTR_GPU_SYNC_MODE'], 'explicit')

    def test_comparison_checks_inputs_and_complete_step_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind, offset in (('extrapolation', 0.), ('nscbc', .25)):
                case = root/kind
                (case/'datin').mkdir(parents=True)
                for name in ('grid.2d', 'flowini2d.h5', 'inlet.prof', 'wallbs.dat', 'controller', 'input.output'):
                    (case/'datin'/name).write_bytes(b'matched')
                (case/'flowstate.dat').write_text(f'nstep time massflux\n1 .02 {1+offset}\n')
                path = case/'outdat/new/fields/segment00000000/step000000000002/data.h5'
                path.parent.mkdir(parents=True)
                with h5py.File(path, 'w') as archive:
                    archive.attrs['phase'] = b'completed_step'
                    metadata = np.zeros(11, dtype=np.int64)
                    metadata[1] = 2
                    metadata[2] = np.array([.04]).view(np.int64)[0]
                    archive['metadata'] = metadata
                    for name in ('density', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature'):
                        archive[name] = np.full((2,2,2), 1+offset)
            report = compare_pair(root, 2)
            self.assertEqual(report['final_field_time'], .04)
            self.assertEqual(report['statistics_last_time'], .02)
            self.assertEqual(report['fields']['density']['max_abs_difference'], .25)
            self.assertEqual(report['fields']['density']['global_ijk'], [0,0,0])
            (root/'nscbc/datin/wallbs.dat').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'unmatched comparison input'):
                compare_pair(root, 2)


if __name__ == '__main__':
    unittest.main()
