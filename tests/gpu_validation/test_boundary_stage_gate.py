"""Fail-closed parsing for the actual boundary-stage probe."""

import unittest

from run_boundary_stage_gate import validate_output


class StageOutputTests(unittest.TestCase):
    def test_finite_pass(self):
        self.assertEqual(validate_output(0, ' BOUNDARY_STAGE_PASS max_abs= 2.22D-16\n'),
                         2.22e-16)

    def test_help_is_not_pass(self):
        with self.assertRaises(ValueError):
            validate_output(0, 'Command Line Help\n')

    def test_nonzero_exit_is_not_pass(self):
        with self.assertRaises(ValueError):
            validate_output(1, 'BOUNDARY_STAGE_PASS max_abs= 0\n')

    def test_invalid_error(self):
        for value in ('nan', 'inf', '-1', '1e-11', 'broken'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_output(0, f'BOUNDARY_STAGE_PASS max_abs= {value}\n')

    def test_duplicate_marker(self):
        with self.assertRaises(ValueError):
            validate_output(0, 'BOUNDARY_STAGE_PASS max_abs= 0\n' * 2)

    def test_gpu_requires_its_own_marker(self):
        with self.assertRaises(ValueError):
            validate_output(0, 'BOUNDARY_STAGE_PASS max_abs= 0\n', gpu=True)
        self.assertEqual(validate_output(0, 'BOUNDARY_STAGE_GPU_PASS max_abs= 0\n', gpu=True), 0.)


if __name__ == '__main__':
    unittest.main()
