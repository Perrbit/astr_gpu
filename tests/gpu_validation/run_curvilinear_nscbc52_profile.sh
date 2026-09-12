#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_profile}"
GPU_ID="${GPU_ID:-0}"
GRID="${GRID:-32,24,32}"
RK_STEPS="${RK_STEPS:-2}"

if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite profile evidence directory: $OUT_DIR" >&2
  exit 2
fi
if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
if (( RK_STEPS < 1 )); then
  echo "RK_STEPS must be positive" >&2
  exit 2
fi
command -v nsys >/dev/null 2>&1 || {
  echo "nsys not found" >&2
  exit 127
}

IFS=, read -r im jm km <<< "$GRID"
case_dir="$OUT_DIR/case"
report_base="$OUT_DIR/nonreflecting"
mkdir -p "$OUT_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
  --dst-case "$case_dir" --use-gpu t \
  --im "$im" --jm "$jm" --km "$km" --conschm 643e \
  --diffterm f --lfilter f --upper-bctype 52 --ninit 3 \
  --uniform-profile --wall-temperature 1.0 --mach 0.3 \
  --maxstep "$((RK_STEPS-1))" --feqchkpt "$((RK_STEPS+1))" \
  --deltat 1e-4
python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
  --output "$case_dir/datin/grid.flatplate.h5" \
  --report "$OUT_DIR/grid_generation.txt" --grid "$GRID" \
  --mapping y-wavy --amplitude 0.15
python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
  --output "$case_dir/datin/flowini3d.h5" --grid "$GRID" \
  --u1 0.0 --u2 0.0 --u3 0.0 --density 1.0 --temperature 1.0

(
  cd "$case_dir"
  CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_sharedfp=individual \
    OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
    OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_opal_cuda_support=0 \
    UCX_MEMTYPE_CACHE=n ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_NSCBC_FARFIELD_MODE=nonreflecting \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
      --force-overwrite=true -o "$report_base" \
      mpirun -np 1 "$GPU_EXE" run datin/input.flatplate \
      > "$OUT_DIR/profile.log" 2>&1
)
grep -q 'The job is done!' "$OUT_DIR/profile.log"
nsys export --type=sqlite --force-overwrite=true \
  --output="$OUT_DIR/nonreflecting.sqlite" "$report_base.nsys-rep" \
  > "$OUT_DIR/export.log" 2>&1
python3 "$ROOT_DIR/tests/gpu_validation/analyze_curvilinear_nscbc52_nsys.py" \
  --input "$OUT_DIR/nonreflecting.sqlite" --rk-steps "$RK_STEPS" \
  --report "$OUT_DIR/profile_audit.txt"

printf 'CURVILINEAR_NSCBC52_PROFILE_PASS\n'
