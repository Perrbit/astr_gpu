"""Compatibility imports for the authoritative TGV preset helpers."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/insitu"))
from tgv_streamlines import make_trace,check_crossing
