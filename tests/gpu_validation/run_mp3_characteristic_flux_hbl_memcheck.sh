#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp3_characteristic_flux_hbl_memcheck_$STAMP}"
GPU_IDS="${GPU_IDS:-0,1}"

[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"
[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
command -v compute-sanitizer >/dev/null || { echo "compute-sanitizer is unavailable" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite MP3 HBL memcheck: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"

prepare_case() {
  local label="$1"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$label" --use-gpu t \
    --im 64 --jm 64 --km 8 \
    --diffterm t --lchardecomp t --shock-threshold 0.001 \
    --conschm 543e --reynolds 1.83052e6 --mach 5.0 \
    --reference-temperature 226.65 \
    --wall-temperature 5.191440547760865 \
    --upper-bctype 51 --x-min-bctype 11 --ninit 3 \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --maxstep 1 --feqchkpt 9999 --sponge-im 0 --deltat 1.0e-5

  python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$OUT_DIR/$label/datin/grid.flatplate.h5" \
    --output "$OUT_DIR/$label/datin/inlet.prof" \
    --mach 5.0 --reynolds 1.83052e6 \
    --reference-temperature 226.65 \
    --wall-temperature 5.191440547760865 --station-x 1.0 \
    --density-mode provided --pressure-mode provided \
    --field-output "$OUT_DIR/$label/datin/flowini3d.h5" \
    --virtual-leading-edge -2.0 --shock-angle-deg 35.0 \
    --shock-x0 -1.0 --shock-y0 0.18 --shock-y-min 0.02 \
    --profile-oblique-shock --profile-shock-y-min 0.18 \
    --field-oblique-shock
}

run_memcheck() {
  local topology="$1" label="$2"
  prepare_case "$label"
  (
    cd "$OUT_DIR/$label"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    ASTR_FORCE_MPI_TOPOLOGY="$topology" \
    ASTR_GPU_PRECISION_MODE=mixed_workspace \
    ASTR_GPU_MIXED_CANDIDATE=characteristic_flux \
    ASTR_GPU_SYNC_MODE=explicit \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self,tcp OMPI_MCA_osc=pt2pt \
    OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      mpirun -np 2 compute-sanitizer --tool memcheck --error-exitcode 99 \
        "$GPU_EXE" run datin/input.flatplate > "$OUT_DIR/$label.log" 2>&1
  )

  rg -q 'The job is done!' "$OUT_DIR/$label.log"
  rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux$' \
    "$OUT_DIR/$label.log"
  local summaries
  summaries="$(rg -c 'ERROR SUMMARY: 0 errors' "$OUT_DIR/$label.log")"
  [[ "$summaries" -eq 2 ]] || {
    echo "$label: expected 2 clean sanitizer summaries, found $summaries" >&2
    exit 1
  }
}

run_memcheck 2,1,1 x_slab
run_memcheck 1,2,1 y_slab

echo "MP3 characteristic_flux Cartesian viscous HBL memcheck passed: $OUT_DIR"
