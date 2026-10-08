"""Reference sign handling must distinguish CPU/GPU from GPU-OFF/GPU-ON."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from run_m12_local_compare import check_rhs_sensor


class BackendComparisonTests(unittest.TestCase):
    def fixture(self, root, name, axis_sign):
        case = root/name
        directory = case/'diagnostics'
        directory.mkdir(parents=True)
        for label in ('conv_after_x', 'conv_after_y', 'conv_after_z', 'conv', 'full', 'sensor'):
            for index in range(30):
                path = directory/f'state.{label}.{index:02d}.bin'
                sensor = label == 'sensor'
                header = np.array([0, 0, 0] if sensor else [0, 0, 0, 1], dtype=np.int32)
                value = axis_sign if label.startswith('conv_after_') else 1.
                payload = header.tobytes() + np.array([value], dtype=np.float64).tobytes()
                if sensor:
                    payload += bytes([0])
                path.write_bytes(payload)
        return case

    @patch('run_m12_local_compare.analyze_pair', return_value={'passed': True})
    def test_same_gpu_backend_does_not_negate_axis_rhs(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            off = self.fixture(root, 'off', 1.)
            on = self.fixture(root, 'on', 1.)
            receipt = check_rhs_sensor(off, on, 1, 'gpu')
            self.assertEqual(max(receipt['max_abs'].values()), 0.)
            with self.assertRaisesRegex(ValueError, 'same-definition'):
                check_rhs_sensor(off, on, 1, 'cpu')

    @patch('run_m12_local_compare.analyze_pair', return_value={'passed': True})
    def test_cpu_reference_retains_pre_rhs_axis_sign(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu = self.fixture(root, 'cpu', -1.)
            gpu = self.fixture(root, 'gpu', 1.)
            receipt = check_rhs_sensor(cpu, gpu, 1)
            self.assertEqual(max(receipt['max_abs'].values()), 0.)
            with self.assertRaisesRegex(ValueError, 'same-definition'):
                check_rhs_sensor(cpu, gpu, 1, 'gpu')
