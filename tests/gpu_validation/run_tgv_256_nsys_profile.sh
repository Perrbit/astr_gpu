#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/nsys_tgv_256_np1_np2}"
GRID="${GRID:-256,256,256}"
MAXSTEP="${MAXSTEP:-10}"
FEQCHKPT="${FEQCHKPT:-9999}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
DELTAT="${DELTAT:-}"
NP1="${NP1:-1}"
NP2="${NP2:-2}"
NP2_TOPOLOGY="${NP2_TOPOLOGY:-2,1,1}"
NSYS_TRACE="${NSYS_TRACE:-cuda,mpi}"
NSYS_STATS="${NSYS_STATS:-true}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not found in PATH" >&2
  exit 127
fi

prepare_case() {
  local dst_case="$1"
  local use_gpu="$2"
  local args=(
    "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py"
    --src-case "$CASE_DIR"
    --dst-case "$dst_case"
    --use-gpu "$use_gpu"
    --maxstep "$MAXSTEP"
    --feqchkpt "$FEQCHKPT"
    --lfilter "$LFILTER"
    --diffterm "$DIFFTERM"
    --scheme "$SCHEME"
    --grid "$GRID"
  )
  if [[ -n "$DELTAT" ]]; then
    args+=(--deltat "$DELTAT")
  fi
  python3 "${args[@]}"
}

profile_case() {
  local label="$1"
  local np="$2"
  local topology="$3"
  local case_dir="$OUT_DIR/$label"
  local profile_base="../${label}_nsys"
  local log_path="../${label}_profile.log"

  (
    cd "$case_dir"
    ASTR_FORCE_MPI_TOPOLOGY="$topology" \
      nsys profile --trace="$NSYS_TRACE" --stats="$NSYS_STATS" --force-overwrite=true \
        -o "$profile_base" \
        mpirun -np "$np" "$GPU_EXE" run datin/input.tgv > "$log_path" 2>&1
  )
}

mkdir -p "$OUT_DIR"

prepare_case "$OUT_DIR/np1" t
prepare_case "$OUT_DIR/np2_xslab" t

profile_case np1 "$NP1" "1,1,1"
profile_case np2_xslab "$NP2" "$NP2_TOPOLOGY"

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/np1" \
  --gpu "$OUT_DIR/np2_xslab" \
  --report "$OUT_DIR/np1_vs_np2_flowstate_compare.txt" \
  --atol "$ATOL" \
  --rtol "$RTOL"
