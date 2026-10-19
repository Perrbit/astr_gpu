#!/usr/bin/env python3
"""Source contract for restart-safe GPU TGV diagnostic files."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src_gpu/statistic_gpu.cuf").read_text(encoding="ascii")


def subroutine_body(name: str) -> str:
    match = re.search(
        rf"subroutine\s+{name}\([^)]*\)(.*?)end\s+subroutine\s+{name}",
        SOURCE,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing subroutine {name}")
    return match.group(1)


class TgvGpuStatisticsRestartContractTests(unittest.TestCase):
    def test_tgv_outputs_share_one_complete_statistic_set(self) -> None:
        runtime = (ROOT / 'src_gpu/gpu_runtime.cuf').read_text()
        body = runtime.split('subroutine gpu_write_tgv_statistics()',1)[1].split(
            'end subroutine gpu_write_tgv_statistics',1)[0]
        for name in ('kenergy','enstophy','dissipation'):
            self.assertEqual(body.count(f'call gpu_{name}_scalar('),1)
            self.assertIn(f'call gpu_write_{name}(statistics(',body)
        self.assertIn('call gpu_write_flowstate(statistics)',body)

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
