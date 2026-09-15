#!/usr/bin/env python3
"""Source contract for restart-safe GPU TGV diagnostic files."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src_gpu/statistic_gpu.cuf").read_text(encoding="ascii")


def subroutine_body(name: str) -> str:
    match = re.search(
        rf"subroutine\s+{name}\(\)(.*?)end\s+subroutine\s+{name}",
        SOURCE,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing subroutine {name}")
    return match.group(1)


class TgvGpuStatisticsRestartContractTests(unittest.TestCase):
    def test_tgv_diagnostic_writers_use_restart_aware_list_initialization(self) -> None:
        expected_files = {
            "gpu_write_flowstate": "flowstate.dat",
            "gpu_write_kenergy": "gpu_kenergy.dat",
            "gpu_write_enstophy": "gpu_enstophy.dat",
            "gpu_write_dissipation": "gpu_dissipation.dat",
        }
        for routine, filename in expected_files.items():
            with self.subTest(routine=routine):
                body = subroutine_body(routine).lower().replace(" ", "")
                self.assertIn("useutility,only:listinit", body)
                self.assertIn("calllistinit(", body)
                self.assertIn(f"filename='{filename}'", body)
                self.assertNotIn("status='replace'", body)


if __name__ == "__main__":
    unittest.main()
