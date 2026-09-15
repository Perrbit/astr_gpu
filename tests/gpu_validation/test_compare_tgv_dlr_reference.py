#!/usr/bin/env python3
"""Tests for DLR spectral TGV diagnostic comparison."""

from pathlib import Path
import tempfile
import unittest

from compare_tgv_dlr_reference import compare_histories, read_astr, read_reference


class CompareTgvDlrReferenceTests(unittest.TestCase):
    def test_interpolates_astr_history_and_finds_reference_peak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.gdiag"
            astr = root / "flowstate.dat"
            reference.write_text(
                "# Time Energy Dissipation Enstrophy\n"
                "0.0 0.125 0.001 0.3\n"
                "0.5 0.100 0.010 2.0\n"
                "1.0 0.080 0.004 1.0\n",
                encoding="ascii",
            )
            astr.write_text(
                "nstep time kenergy enstophy dissipation\n"
                "0 0.0 0.125 0.3 0.001\n"
                "1 0.25 0.1125 1.15 0.0055\n"
                "2 0.5 0.100 2.0 0.010\n"
                "3 0.75 0.090 1.5 0.007\n"
                "4 1.0 0.080 1.0 0.004\n",
                encoding="ascii",
            )

            metrics = compare_histories(read_reference(reference), read_astr(astr))

            self.assertEqual(metrics.samples, 3)
            self.assertAlmostEqual(metrics.reference_peak_time, 0.5)
            self.assertAlmostEqual(metrics.astr_peak_time, 0.5)
            self.assertAlmostEqual(metrics.energy_relative_l2, 0.0)
            self.assertAlmostEqual(metrics.dissipation_relative_l2, 0.0)
            self.assertAlmostEqual(metrics.enstrophy_relative_l2, 0.0)

    def test_rejects_history_that_does_not_cover_reference_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.gdiag"
            astr = root / "flowstate.dat"
            reference.write_text("0.0 0.125 0.001 0.3\n1.0 0.08 0.004 1.0\n", encoding="ascii")
            astr.write_text(
                "nstep time kenergy enstophy dissipation\n"
                "0 0.0 0.125 0.3 0.001\n1 0.5 0.1 2.0 0.01\n",
                encoding="ascii",
            )

            with self.assertRaisesRegex(ValueError, "does not cover"):
                compare_histories(read_reference(reference), read_astr(astr))

    def test_injects_analytic_tgv_initial_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            astr = Path(tmp) / "flowstate.dat"
            astr.write_text(
                "nstep time kenergy enstophy dissipation\n"
                "1 0.001 0.1249 0.376 0.00047\n",
                encoding="ascii",
            )

            samples = read_astr(astr)

            self.assertEqual(samples[0].time, 0.0)
            self.assertEqual(samples[0].energy, 0.125)
            self.assertEqual(samples[0].enstrophy, 0.375)
            self.assertAlmostEqual(samples[0].dissipation, 0.00046875)

    def test_does_not_inject_initial_sample_before_late_restart_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            astr = Path(tmp) / "flowstate.dat"
            astr.write_text(
                "nstep time kenergy enstophy dissipation\n"
                "104000 19.92536 0.021875 1.60164 0.0020035\n"
                "104001 19.92555 0.021874 1.60158 0.0020034\n",
                encoding="ascii",
            )

            samples = read_astr(astr)

            self.assertEqual(samples[0].time, 19.92536)

    def test_overlap_only_compares_reference_samples_covered_by_restart_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.gdiag"
            astr = root / "flowstate.dat"
            reference.write_text(
                "0.0 0.125 0.001 0.3\n"
                "1.0 0.100 0.004 1.0\n"
                "2.0 0.080 0.003 0.8\n"
                "3.0 0.060 0.002 0.6\n",
                encoding="ascii",
            )
            astr.write_text(
                "nstep time kenergy enstophy dissipation\n"
                "20 2.0 0.080 0.8 0.003\n"
                "25 2.5 0.070 0.7 0.0025\n"
                "30 3.0 0.060 0.6 0.002\n",
                encoding="ascii",
            )

            metrics = compare_histories(
                read_reference(reference), read_astr(astr), overlap_only=True
            )

            self.assertEqual(metrics.samples, 2)
            self.assertEqual(metrics.start_time, 2.0)
            self.assertEqual(metrics.end_time, 3.0)
            self.assertAlmostEqual(metrics.energy_relative_l2, 0.0)
            self.assertAlmostEqual(metrics.enstrophy_relative_l2, 0.0)
            self.assertAlmostEqual(metrics.dissipation_relative_l2, 0.0)


if __name__ == "__main__":
    unittest.main()
