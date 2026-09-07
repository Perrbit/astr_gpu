"""Reject false-positive CLI exits and invalid numeric evidence."""

import unittest

from run_boundary_rhs_gate import CPU_MARKERS, check_output


class BoundaryRHSGateTests(unittest.TestCase):
    def output(self, value):
        return "\n".join(f"{marker} max_abs= {value}" for marker in CPU_MARKERS)

    def test_valid(self):
        self.assertEqual(len(check_output(self.output("3.7E-16"), CPU_MARKERS)), len(CPU_MARKERS))

    def test_invalid_numeric(self):
        for value in ("NaN", "Inf", "1e-9", "-1e-16"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_output(self.output(value), CPU_MARKERS)

    def test_missing_or_duplicate_marker(self):
        for output in ("Command Line Help", self.output("0") * 2):
            with self.assertRaises(ValueError):
                check_output(output, CPU_MARKERS)

    def test_sanitizer_summary_required(self):
        with self.assertRaises(ValueError):
            check_output(self.output("0"), CPU_MARKERS, sanitizer=True)
        check_output(self.output("0") + "\nERROR SUMMARY: 0 errors", CPU_MARKERS, sanitizer=True)


if __name__ == "__main__":
    unittest.main()
