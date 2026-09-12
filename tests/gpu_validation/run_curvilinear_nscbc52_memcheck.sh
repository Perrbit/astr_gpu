#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_memcheck}"
GRID="${GRID:-32,24,32}"
GPU_IDS="${GPU_IDS:-0,1}"
MATRIX=(
  '1:1,1,1'
  '2:1,2,1'
)

if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite memcheck evidence directory: $OUT_DIR" >&2
  exit 2
fi
if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
command -v compute-sanitizer >/dev/null 2>&1 || {
  echo "compute-sanitizer not found" >&2
  exit 127
}

IFS=, read -r im jm km <<< "$GRID"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/memcheck_summary.txt"
printf 'grid=%s\n' "$GRID" > "$SUMMARY"

for entry in "${MATRIX[@]}"; do
  np="${entry%%:*}"
  topology="${entry#*:}"
  tag="np${np}_${topology//,/x}"
  case_dir="$OUT_DIR/$tag/case"
  log="$OUT_DIR/$tag/memcheck.log"

  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$case_dir" --use-gpu t \
    --im "$im" --jm "$jm" --km "$km" --conschm 643e \
    --diffterm f --lfilter f --upper-bctype 52 --ninit 3 \
    --uniform-profile --wall-temperature 1.0 --mach 0.3 \
    --maxstep 1 --feqchkpt 2 --deltat 1e-4
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$case_dir/datin/grid.flatplate.h5" \
    --report "$OUT_DIR/$tag/grid_generation.txt" --grid "$GRID" \
    --mapping y-wavy --amplitude 0.15
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$case_dir/datin/flowini3d.h5" --grid "$GRID" \
    --u1 0.0 --u2 0.0 --u3 0.0 --density 1.0 --temperature 1.0

  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" OMPI_MCA_sharedfp=individual \
      OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp OMPI_MCA_osc=pt2pt \
      OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_opal_cuda_support=0 \
      UCX_MEMTYPE_CACHE=n ASTR_FORCE_MPI_TOPOLOGY="$topology" \
      ASTR_NSCBC_FARFIELD_MODE=nonreflecting \
      mpirun --oversubscribe -np "$np" \
        compute-sanitizer --tool memcheck --error-exitcode 99 \
        "$GPU_EXE" run datin/input.flatplate > "$log" 2>&1
  )

  grep -q 'The job is done!' "$log"
  summaries="$(grep -c 'ERROR SUMMARY: 0 errors' "$log")"
  if [[ "$summaries" -ne "$np" ]]; then
    echo "expected $np clean sanitizer summaries, found $summaries: $log" >&2
    exit 1
  fi
  printf 'pass np=%s topology=%s summaries=%s log=%s\n' \
    "$np" "$topology" "$summaries" "$log" | tee -a "$SUMMARY"
done
