#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE="${CASE:-uniform}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_${CASE}_matrix}"
GRID="${GRID:-}"
MAXSTEP="${MAXSTEP:-}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
REFLECTION_ATOL="${REFLECTION_ATOL:-1e-3}"
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

if [[ "$CASE" != uniform && "$CASE" != acoustic ]]; then
  echo "CASE must be uniform or acoustic" >&2
  exit 2
fi
if [[ -z "$GRID" ]]; then
  if [[ "$CASE" == uniform ]]; then
    GRID='32,24,32'
  else
    GRID='64,48,64'
  fi
fi
if [[ -z "$MAXSTEP" ]]; then
  if [[ "$CASE" == uniform ]]; then
    MAXSTEP='10'
  else
    MAXSTEP='440'
  fi
fi

topology_tag() {
  printf '%s' "${1//,/x}"
}

audit_owners() {
  local log="$1"
  local np="$2"
  local topology="$3"
  python3 - "$log" "$np" "$topology" <<'PY'
import re
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
np = int(sys.argv[2])
isize, jsize, ksize = map(int, sys.argv[3].split(","))
pattern = re.compile(
    r"ASTR_NSCBC52_OWNER rank=(\d+) irk=(\d+) jrk=(\d+) krk=(\d+) apply_up=(\d+)"
)
records = [tuple(map(int, match.groups())) for match in pattern.finditer(log_path.read_text())]
if len(records) != np or len({record[0] for record in records}) != np:
    raise SystemExit(f"owner trace must contain one record per rank: {records}")
expected_owners = isize * ksize
owners = [record for record in records if record[4] == 1]
if len(owners) != expected_owners:
    raise SystemExit(f"upper-y owner count mismatch: {len(owners)} != {expected_owners}")
for rank, irk, jrk, krk, apply_up in records:
    if not (0 <= irk < isize and 0 <= jrk < jsize and 0 <= krk < ksize):
        raise SystemExit(f"rank {rank} has invalid Cartesian coordinates")
    if apply_up not in (0, 1):
        raise SystemExit(f"rank {rank} has invalid apply_up={apply_up}")
    if apply_up == 1 and jrk != jsize - 1:
        raise SystemExit(f"rank {rank} claims upper-y ownership at jrk={jrk}")
    if jrk != jsize - 1 and apply_up != 0:
        raise SystemExit(f"nonowner rank {rank} has apply_up={apply_up}")
print(f"owner_audit: PASS ranks={np} owners={len(owners)} topology={isize},{jsize},{ksize}")
PY
}

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/matrix_summary.txt"
printf 'case=%s grid=%s maxstep=%s\n' "$CASE" "$GRID" "$MAXSTEP" > "$SUMMARY"
baseline_uniform=''
baseline_acoustic=''

for entry in "${MATRIX[@]}"; do
  np="${entry%%:*}"
  topology="${entry#*:}"
  tag="np${np}_$(topology_tag "$topology")"
  case_dir="$OUT_DIR/$tag"

  if [[ "$CASE" == uniform ]]; then
    ASTR_VALIDATION_NSCBC52_OWNER_TRACE=1 \
      OUT_DIR="$case_dir" NP="$np" TOPOLOGY="$topology" \
      GRID="$GRID" MAXSTEP="$MAXSTEP" ATOL="$FIELD_ATOL" RTOL="$FIELD_ATOL" \
      bash "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh"
    audit_owners "$case_dir/gpu/gpu.log" "$np" "$topology" | tee -a "$SUMMARY"
    snapshot="$case_dir/cpu/outdat/rk_complete_snapshot.h5"
    if [[ -z "$baseline_uniform" ]]; then
      baseline_uniform="$snapshot"
    else
      python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
        --cpu "$baseline_uniform" --gpu "$snapshot" \
        --report "$case_dir/cpu_vs_np1.txt" \
        --atol "$FIELD_ATOL" --rtol "$FIELD_ATOL" > /dev/null
    fi
  else
    ASTR_VALIDATION_NSCBC52_OWNER_TRACE=1 \
      OUT_DIR="$case_dir" NP="$np" TOPOLOGY="$topology" \
      GRID_LEVELS="$GRID" MAXSTEP="$MAXSTEP" REFLECTION_ATOL="$REFLECTION_ATOL" \
      bash "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh"
    IFS=, read -r im jm km <<< "$GRID"
    level_dir="$case_dir/${im}x${jm}x${km}"
    audit_owners "$level_dir/gpu_nonreflecting/run.log" "$np" "$topology" | tee -a "$SUMMARY"
    metrics="$level_dir/reflection_summary.json"
    if [[ -z "$baseline_acoustic" ]]; then
      baseline_acoustic="$metrics"
    else
      python3 - "$baseline_acoustic" "$metrics" "$REFLECTION_ATOL" <<'PY'
import json
from pathlib import Path
import sys

baseline = json.loads(Path(sys.argv[1]).read_text())
current = json.loads(Path(sys.argv[2]).read_text())
atol = float(sys.argv[3])
for key in ("cpu_nonreflecting", "gpu_nonreflecting"):
    difference = abs(float(current[key]) - float(baseline[key]))
    if difference > atol:
        raise SystemExit(f"topology reflection mismatch for {key}: {difference} > {atol}")
PY
    fi
  fi
  printf 'pass np=%s topology=%s out=%s\n' "$np" "$topology" "$case_dir" | tee -a "$SUMMARY"
done
