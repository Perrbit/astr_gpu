#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_viscous_matrix}"
MAXSTEP="${MAXSTEP:-5}"
MATRIX=(
  '1:1,1,1'
  '2:2,1,1'
  '2:1,2,1'
  '2:1,1,2'
  '4:2,2,1'
  '4:2,1,2'
  '4:1,2,2'
  '8:2,2,2'
)

audit_owners() {
  local log="$1"
  local np="$2"
  local topology="$3"
  python3 - "$log" "$np" "$topology" <<'PY'
import re
from pathlib import Path
import sys

log = Path(sys.argv[1]).read_text()
np = int(sys.argv[2])
isize, jsize, ksize = map(int, sys.argv[3].split(","))
pattern = re.compile(
    r"ASTR_NSCBC52_OWNER rank=(\d+) irk=(\d+) jrk=(\d+) krk=(\d+) apply_up=(\d+)"
)
records = [tuple(map(int, match.groups())) for match in pattern.finditer(log)]
if len(records) != np or len({record[0] for record in records}) != np:
    raise SystemExit(f"owner trace must contain one record per rank: {records}")
owners = [record for record in records if record[4] == 1]
expected = isize * ksize
if len(owners) != expected:
    raise SystemExit(f"upper-y owner count mismatch: {len(owners)} != {expected}")
if any(record[4] != int(record[2] == jsize-1) for record in records):
    raise SystemExit(f"upper-y owner placement mismatch: {records}")
print(f"owner_audit: PASS ranks={np} owners={len(owners)} topology={sys.argv[3]}")
PY
}

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/matrix_summary.txt"
printf 'maxstep=%s\n' "$MAXSTEP" > "$SUMMARY"
for entry in "${MATRIX[@]}"; do
  np="${entry%%:*}"
  topology="${entry#*:}"
  tag="np${np}_${topology//,/x}"
  case_dir="$OUT_DIR/$tag"
  ASTR_VALIDATION_NSCBC52_OWNER_TRACE=1 \
    OUT_DIR="$case_dir" NP="$np" TOPOLOGY="$topology" MAXSTEP="$MAXSTEP" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_viscous_hbl_compare.sh"
  audit_owners "$case_dir/gpu/gpu.log" "$np" "$topology" | tee -a "$SUMMARY"
  printf 'pass np=%s topology=%s out=%s\n' "$np" "$topology" "$case_dir" | tee -a "$SUMMARY"
done
