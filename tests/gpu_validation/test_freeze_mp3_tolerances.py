import importlib.util
import math
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "gpu_validation" / "freeze_mp3_tolerances.py"


def load_module():
    if not SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("freeze_mp3_tolerances", SCRIPT)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


class FreezeMp3TolerancesTest(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(MODULE, f"missing {SCRIPT}")
        return MODULE

    def test_field_parser_returns_largest_linf(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "field.txt"
            path.write_text(
                "q1 linf=2.5e-9 l2=1e-10\nq5 linf=7.5e-8 l2=2e-9\n",
                encoding="utf-8",
            )
            self.assertEqual(module.read_field_max(path), 7.5e-8)

    def test_stats_parser_returns_largest_max_abs(self):
        module = self.require_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "stats.txt"
            path.write_text(
                "status: pass\n\n"
                "metric max_abs max_rel final_cpu final_gpu final_abs final_rel\n"
                "time 0 0 0 0 0 0\n"
                "kenergy 3.2e-8 3.2e-8 1 1 3.2e-8 3.2e-8\n",
                encoding="utf-8",
            )
            self.assertEqual(module.read_stats_max(path), 3.2e-8)

    def test_tolerance_uses_tenfold_margin_and_125_ceiling(self):
        module = self.require_module()
        self.assertTrue(
            math.isclose(
                module.frozen_tolerance(2.1e-9, 1.0e-12, 1.0e-5),
                5.0e-8,
            )
        )
        self.assertEqual(
            module.frozen_tolerance(0.0, 1.0e-12, 1.0e-5), 1.0e-12
        )
        self.assertEqual(
            module.frozen_tolerance(1.6e-7, 1.0e-12, 1.0e-5), 2.0e-6
        )

    def test_tolerance_rejects_nonfinite_negative_and_over_ceiling(self):
        module = self.require_module()
        for value in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                module.frozen_tolerance(value, 1.0e-12, 1.0e-5)
        with self.assertRaises(ValueError):
            module.frozen_tolerance(6.0e-6, 1.0e-12, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
