#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_c10_256_residency}"
GRID="${GRID:-256,256,256}"
MAXSTEP="${MAXSTEP:-2}"
DELTAT="${DELTAT:-1e-6}"
PROFILE_POINTS="${PROFILE_POINTS:-8001}"
START_KERNEL="${START_KERNEL:-characteristic_upwind_rhs_x_xyphysical_global_kernel}"
CASE_DIR="$OUT_DIR/case"
PROFILE_BASE="$OUT_DIR/c10_np1_nochk"
SQLITE="$OUT_DIR/c10_np1_nochk.sqlite"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required" >&2
  exit 127
fi
IFS=',' read -r IM JM KM <<< "$GRID"
if [[ -z "${KM:-}" || "$IM" -lt 8 || "$JM" -lt 8 || "$KM" -lt 8 ]]; then
  echo "GRID must contain three comma-separated dimensions of at least 8" >&2
  exit 2
fi

python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
  --dst-case "$CASE_DIR" --use-gpu t \
  --im "$IM" --jm "$JM" --km "$KM" \
  --diffterm t --lfilter f --conschm 543e --lchardecomp t \
  --shock-threshold 0.001 --reynolds 1.83052e6 --mach 5.0 \
  --reference-temperature 226.65 --wall-temperature 5.191440547760865 \
  --upper-bctype 51 --x-min-bctype 11 --ninit 3 \
  --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
  --warp-x 0.4 --warp-y 0.2 --maxstep "$MAXSTEP" \
  --feqchkpt 9999 --feqlist 9999 --deltat "$DELTAT"
python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
  --grid "$CASE_DIR/datin/grid.flatplate.h5" \
  --output "$CASE_DIR/datin/inlet.prof" --mach 5.0 --reynolds 1.83052e6 \
  --reference-temperature 226.65 --wall-temperature 5.191440547760865 \
  --station-x 1.0 --density-mode provided --pressure-mode provided \
  --field-output "$CASE_DIR/datin/flowini3d.h5" --virtual-leading-edge -2.0 \
  --field-oblique-shock --shock-angle-deg 35.0 --shock-x0 -1.0 \
  --shock-y0 0.18 --shock-y-min 0.18 --profile-oblique-shock \
  --profile-shock-y-min 0.18 --points "$PROFILE_POINTS"

mkdir -p "$CASE_DIR/tmp"
(
  cd "$CASE_DIR"
  TMPDIR="$CASE_DIR/tmp" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
      --force-overwrite=true -o "$PROFILE_BASE" \
      mpirun -np 1 "$GPU_EXE" run datin/input.flatplate > "$OUT_DIR/profile.log" 2>&1
)

grep -q 'The job is done!' "$OUT_DIR/profile.log"
if grep -Eq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$OUT_DIR/profile.log"; then
  echo "non-finite or crash marker found in $OUT_DIR/profile.log" >&2
  exit 1
fi

nsys export --type=sqlite --force-overwrite=true --output="$SQLITE" \
  "$PROFILE_BASE.nsys-rep"
python3 "$ROOT_DIR/tests/gpu_validation/analyze_nsys_rk_residency.py" \
  --input "$SQLITE" --start-kernel "$START_KERNEL" \
  --large-transfer-bytes 65536 --report "$OUT_DIR/residency_report.txt"
nsys stats --force-export=true \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum --format table \
  "$PROFILE_BASE.nsys-rep" > "$OUT_DIR/nsys_summary.txt"
