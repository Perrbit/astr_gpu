#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_viscous_profile}"
DIFFTERM=t
ANALYZE_VISCOUS=t
NP=2
TOPOLOGY=1,2,1

export ASTR_NSCBC_FARFIELD_MODE=nonreflecting
DIFFTERM="$DIFFTERM" ANALYZE_VISCOUS="$ANALYZE_VISCOUS" \
  OUT_DIR="$OUT_DIR/np1_profile" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_profile.sh"

ASTR_VALIDATION_NSCBC52_OWNER_TRACE=1 \
  OUT_DIR="$OUT_DIR/np2_owner" NP="$NP" TOPOLOGY="$TOPOLOGY" MAXSTEP=1 \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_viscous_uniform_compare.sh"

python3 - "$OUT_DIR/np2_owner/gpu/gpu.log" <<'PY'
import re
from pathlib import Path
import sys

records = re.findall(
    r"ASTR_NSCBC52_OWNER rank=(\d+).*?apply_up=(\d+)",
    Path(sys.argv[1]).read_text(),
)
if len(records) != 2 or sum(int(owner) for _, owner in records) != 1:
    raise SystemExit(f"NP=2 upper-y owner trace mismatch: {records}")
print("np2_owner_audit: PASS ranks=2 owners=1 topology=1,2,1")
PY
