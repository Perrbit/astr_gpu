#!/usr/bin/env python3
"""Unit tests for TGV validation case preparation helpers."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_tgv_case


class PrepareTgvCaseTests(unittest.TestCase):
    def test_explicit_external_grid_omits_the_unused_template(self) -> None:
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';datin=source/'datin';datin.mkdir(parents=True)
            for name in ('input.tgv','controller'):
                shutil.copyfile(root/'examples/Taylor_Green_Vortex/datin'/name,datin/name)
            template=datin/'grid.h5';template.write_bytes(b'unused template resource')
            target=Path(tmp)/'prepared'
            argv=['prepare_tgv_case','--src-case',str(source),'--dst-case',str(target),
                  '--use-gpu','t','--maxstep','11','--feqchkpt','5','--grid','16,16,16',
                  '--lreadgrid','t','--restart','f','--gridfile','datin/grid.cartesian.h5',
                  '--omit-grid-resource']
            with patch.object(sys,'argv',argv):
                self.assertEqual(prepare_tgv_case.main(),0)
            self.assertFalse((target/'datin/grid.h5').exists())
            text=(target/'datin/input.tgv').read_text()
            self.assertIn('datin/grid.cartesian.h5',text)
            lines=text.splitlines()
            marker=next(i for i,line in enumerate(lines) if 'nondimen,diffterm,lfilter,lreadgrid' in line)
            flags=lines[prepare_tgv_case.next_data_line(lines,marker)].split(',')
            self.assertEqual(flags[3],'t')
            self.assertEqual(flags[8],'t')
            self.assertEqual(template.read_bytes(),b'unused template resource')
            rejected=Path(tmp)/'rejected'
            argv[argv.index('--dst-case')+1]=str(rejected)
            argv[argv.index('--gridfile')+1]='datin/grid.h5'
            with patch.object(sys,'argv',argv), self.assertRaisesRegex(ValueError,'alternate --gridfile'):
                prepare_tgv_case.main()
            self.assertFalse(rejected.exists())

    def test_default_validation_output_is_explicitly_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.output"
            prepare_tgv_case.write_validation_output_config(path)
            text = path.read_text(encoding="ascii")
            self.assertIn("&output", text)
            self.assertIn("directory='outdat/new'", text)
            self.assertEqual(text.count("enabled=f"), 3)
            self.assertNotIn("enabled=t", text)

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
