#!/usr/bin/env python3
"""Plot the valid ASTR/DLR overlap retained from A800 job 454679."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]

# Adjust these paths here when plotting another production run.
ASTR_FLOWSTATE = ROOT / (
    "tests/gpu_validation/out/a800_tgv_campaign_454679/"
    "t7_512_gpu_np4_t20/raw/flowstate.dat"
)
DLR_REFERENCE = ROOT / "documents/reference_data/tgv/spectral_Re1600_512.gdiag"
OUTPUT_DIR = ROOT / (
    "tests/gpu_validation/out/a800_tgv_campaign_454679/"
    "t7_512_gpu_np4_t20/dlr_overlap_comparison"
)
COMPARE_SCRIPT = ROOT / "tests/gpu_validation/compare_tgv_dlr_reference.py"


def main() -> int:
    command = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--reference",
        str(DLR_REFERENCE),
        "--astr-flowstate",
        str(ASTR_FLOWSTATE),
        "--output-dir",
        str(OUTPUT_DIR),
        "--overlap-only",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
