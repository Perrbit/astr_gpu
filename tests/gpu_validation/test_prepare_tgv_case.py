#!/usr/bin/env python3
"""Unit tests for TGV validation case preparation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_tgv_case


class PrepareTgvCaseTests(unittest.TestCase):
    def test_set_restart_updates_start_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "input.tgv"
            input_file.write_text("# lrestar : start mode\nf\n", encoding="ascii")

            prepare_tgv_case.set_restart(input_file, "t")

            self.assertEqual(
                input_file.read_text(encoding="ascii"),
                "# lrestar : start mode\nt\n",
            )

    def test_reference_mach_update_preserves_temperature_and_reynolds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "input.tgv"
            input_file.write_text(
                "# ref_t,reynolds,mach\n273.15d0,1600.d0,0.1d0\n",
                encoding="ascii",
            )

            prepare_tgv_case.set_reference_mach(input_file, 0.3)

            self.assertEqual(
                input_file.read_text(encoding="ascii").splitlines()[1],
                "273.15d0,1600.d0,2.9999999999999999e-01",
            )

    def test_controller_steps_can_disable_list_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Path(tmp) / "controller"
            controller.write_text(
                "# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg\n"
                "10,20,30,40,1,60\n",
                encoding="ascii",
            )

            prepare_tgv_case.set_controller_steps(controller, 20, 9999, 9999)

            self.assertEqual(controller.read_text(encoding="ascii").splitlines()[1], "20,9999,30,40,9999,60")

    def test_controller_sequence_enables_flowfield_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = Path(tmp) / "controller"
            controller.write_text(
                "# lwsequ,lwslic,lavg,lcracon\n"
                "f,f,f,f\n"
                "# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg\n"
                "10,20,30,40,50,60\n",
                encoding="ascii",
            )

            prepare_tgv_case.set_controller_sequence(controller, "t", 4)

            lines = controller.read_text(encoding="ascii").splitlines()
            self.assertEqual(lines[1], "t,f,f,f")
            self.assertEqual(lines[3], "10,20,4,40,50,60")

    def test_bctype_accepts_semicolon_separated_full_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "input.tgv"
            input_file.write_text(
                "\n".join(
                    [
                        "# bctype",
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                        "1",
                    ]
                )
                + "\n"
            )

            prepare_tgv_case.set_bctype(
                input_file,
                "41, 273.15d0;41, 273.15d0;1;1;1;1",
            )

            self.assertEqual(
                input_file.read_text().splitlines()[1:7],
                ["41, 273.15d0", "41, 273.15d0", "1", "1", "1", "1"],
            )


if __name__ == "__main__":
    unittest.main()
