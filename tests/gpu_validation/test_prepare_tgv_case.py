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
