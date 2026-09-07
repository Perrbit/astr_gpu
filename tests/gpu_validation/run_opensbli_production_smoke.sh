#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new evidence directory}"
NX="${NX:-33}"
NY="${NY:-33}"
NZ="${NZ:-9}"
MAXSTEP="${MAXSTEP:-1}"
DELTAT="${DELTAT:-1.0e-6}"
PRANDTL="${PRANDTL:-0.72}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
CAPTURE_RHS="${CAPTURE_RHS:-0}"

if [[ -e "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

for kind in cpu gpu; do
  use_gpu=f
  exe="$CPU_EXE"
  if [[ "$kind" == gpu ]]; then
    use_gpu=t
    exe="$GPU_EXE"
  fi
  case_dir="$OUT_DIR/$kind"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_opensbli_astr_case.py" \
    --destination "$case_dir" --use-gpu "$use_gpu" --nx "$NX" --ny "$NY" --nz "$NZ" \
    --maxstep "$MAXSTEP" --feqchkpt "$MAXSTEP" --deltat "$DELTAT" --prandtl "$PRANDTL" \
    > "$OUT_DIR/${kind}_prepare.json"
  (
    cd "$case_dir"
    rhs_prefix=""
    if [[ "$CAPTURE_RHS" == "1" ]]; then
      rhs_prefix="$case_dir/rhs"
    fi
    OMPI_MCA_sharedfp=individual \
    ASTR_PERFECT_GAS_PRANDTL="$PRANDTL" \
    ASTR_SUTHERLAND_TEMPERATURE_K=110.4 \
    ASTR_VALIDATION_RHS_PREFIX="$rhs_prefix" \
    ASTR_CONSERVATIVE_BOUNDARY_FILE="$case_dir/datin/conservative_boundary.nml" \
      timeout --kill-after=10s 120s mpirun -np 1 "$exe" run datin/input.opensbli \
      > "$kind.log" 2>&1
  )
  grep -q 'ASTR_CONSERVATIVE_SBLI_MODE enabled' "$case_dir/$kind.log"
  grep -q 'The job is done!' "$case_dir/$kind.log"
  if grep -Eq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|NaN' "$case_dir/$kind.log"; then
    printf 'Nonfinite or crash marker in %s\n' "$case_dir/$kind.log" >&2
    exit 1
  fi
done

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu/outdat/flowfield.h5" \
  --gpu "$OUT_DIR/gpu/outdat/flowfield.h5" \
  --report "$OUT_DIR/field_compare.json" \
  --atol "$FIELD_ATOL" --rtol 0

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu/flowstate.dat" \
  --gpu "$OUT_DIR/gpu/flowstate.dat" \
  --report "$OUT_DIR/stats_compare.json" \
  --atol "$STATS_ATOL" --rtol 0
