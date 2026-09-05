#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_c21_aggregate}"
START_STAGE="${START_STAGE:-}"
STOP_AFTER="${STOP_AFTER:-}"
RUN_PERFORMANCE="${RUN_PERFORMANCE:-t}"
SUMMARY="$OUT_DIR/c21_stage_summary.tsv"
LOG_DIR="$OUT_DIR/logs"
STARTED=t

if [[ -n "$START_STAGE" ]]; then
  STARTED=f
fi

mkdir -p "$LOG_DIR"
printf 'stage\tstatus\tseconds\n' > "$SUMMARY"
git -C "$ROOT_DIR" rev-parse HEAD > "$OUT_DIR/git_head.txt"
git -C "$ROOT_DIR" status --short -- CMakeLists.txt src src_gpu \
  > "$OUT_DIR/solver_git_status.txt"
git -C "$ROOT_DIR" diff -- CMakeLists.txt src src_gpu | sha256sum | \
  awk '{print $1}' > "$OUT_DIR/solver_diff_sha256.txt"
date --iso-8601=seconds > "$OUT_DIR/run_started.txt"

run_stage() {
  local label="$1"
  shift
  local start elapsed

  if [[ "$STARTED" != "t" ]]; then
    if [[ "$label" == "$START_STAGE" ]]; then
      STARTED=t
    else
      printf '%s\tskipped-before-start\t0\n' "$label" >> "$SUMMARY"
      return
    fi
  fi

  printf '\n===== C21 stage: %s =====\n' "$label"
  start=$SECONDS
  "$@" 2>&1 | tee "$LOG_DIR/$label.log"
  elapsed=$((SECONDS-start))
  printf '%s\tpass\t%s\n' "$label" "$elapsed" >> "$SUMMARY"

  if [[ -n "$STOP_AFTER" && "$label" == "$STOP_AFTER" ]]; then
    cat "$SUMMARY"
    exit 0
  fi
}

run_builds() {
  cmake --build "$ROOT_DIR/build_cpu_probe" -j "${BUILD_JOBS:-8}"
  cmake --build "$ROOT_DIR/build_gpu_probe" -j "${BUILD_JOBS:-8}"
  sha256sum "$ROOT_DIR/build_cpu_probe/bin/astr" \
    "$ROOT_DIR/build_gpu_probe/bin/astr" > "$OUT_DIR/binary_sha256.txt"
}

run_c2() {
  env OUT_DIR="$OUT_DIR/c02_np2_x" NP=2 TOPOLOGY=2,1,1 MAXSTEP=5 \
    FEQCHKPT=5 LFILTER=t DIFFTERM=t \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"
  env OUT_DIR="$OUT_DIR/c02_np2_y" NP=2 TOPOLOGY=1,2,1 MAXSTEP=5 \
    FEQCHKPT=5 LFILTER=t DIFFTERM=t \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"
  env OUT_DIR="$OUT_DIR/c02_np2_z" NP=2 TOPOLOGY=1,1,2 MAXSTEP=5 \
    FEQCHKPT=5 LFILTER=t DIFFTERM=t \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"
}

run_c5() {
  env OUT_DIR="$OUT_DIR/c05_symmetry_x_np1" AXIS=x NP=1 TOPOLOGY=1,1,1 \
    MAXSTEP=1 "$ROOT_DIR/tests/gpu_validation/run_curvilinear_symmetry_compare.sh"
  env OUT_DIR="$OUT_DIR/c05_symmetry_x_np2" AXIS=x NP=2 TOPOLOGY=2,1,1 \
    MAXSTEP=1 "$ROOT_DIR/tests/gpu_validation/run_curvilinear_symmetry_compare.sh"
}

check_c11_policy() {
  rg -q 'CURVE-C11-HBL-FILTERED-SELECTIVE-ROE.*closed-negative' \
    "$ROOT_DIR/documents/GPU_VALIDATION_MATRIX.md"
  rg -q 'keeps shock-containing selective-Roe paths at `lfilter=f`' \
    "$ROOT_DIR/tests/gpu_validation/README.md"
  if rg --files "$ROOT_DIR/tests/gpu_validation" | \
       rg -q 'run_curvilinear_hbl_c11.*\.sh$'; then
    echo 'C11 must remain a closed negative gate without an executable driver' >&2
    return 1
  fi
  echo 'C11 closed-negative policy is present; the known non-finite CPU probe is not rerun.'
}

run_c15() {
  if [[ "$RUN_PERFORMANCE" != "t" ]]; then
    echo 'C15 performance/residency was explicitly skipped' >&2
    return 2
  fi
  env OUT_DIR="$OUT_DIR/c15_residency" GRID=256,256,256 MAXSTEP=2 \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c10_256_residency.sh"
  env OUT_DIR="$OUT_DIR/c15_benchmark" GRID=256,256,256 MAXSTEP=5 REPEATS=3 \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c10_256_benchmark.sh"
}

run_c20() {
  env OUT_DIR="$OUT_DIR/c20_profile_inflow11" \
    "$ROOT_DIR/tests/gpu_validation/run_profile_inflow11_state_compare.sh"
  env OUT_DIR="$OUT_DIR/c20_open_boundary_rejects" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_open_boundary_rejects.sh"
  env OUT_DIR="$OUT_DIR/c20_cartesian_openshock22" MAXSTEP=1 FEQCHKPT=1 \
    "$ROOT_DIR/tests/gpu_validation/run_openshock_s0b2_nscbc_compare.sh"
}

run_axis_memcheck() {
  local axis case_dir
  for axis in x y z; do
    case_dir="$OUT_DIR/c13_symmetry/$axis"_np1/gpu
    if [[ ! -d "$case_dir" ]]; then
      echo "missing C13 GPU case for $axis memcheck: $case_dir" >&2
      return 1
    fi
    (
      cd "$case_dir"
      env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
        mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
        "$ROOT_DIR/build_gpu_probe/bin/astr" run datin/input.tgv
    ) > "$OUT_DIR/c21_memcheck_${axis}.log" 2>&1
    rg -q 'ERROR SUMMARY: 0 errors' "$OUT_DIR/c21_memcheck_${axis}.log"
    echo "C21 $axis-direction representative memcheck: 0 errors"
  done
}

run_stage build run_builds
run_stage c00_conv env OUT_DIR="$OUT_DIR/c00_conv" MAXSTEP=1 FEQCHKPT=1 \
  LFILTER=f DIFFTERM=f \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"
run_stage c01_diff_filter env OUT_DIR="$OUT_DIR/c01_diff_filter" MAXSTEP=10 \
  FEQCHKPT=10 LFILTER=t DIFFTERM=t \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"
run_stage c02_np2_slabs run_c2
run_stage c03_freestream env OUT_DIR="$OUT_DIR/c03_freestream" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_freestream_matrix.sh"
run_stage c04_metrics env OUT_DIR="$OUT_DIR/c04_metrics" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_metric_convergence.sh"
run_stage c05_symmetry_x run_c5
run_stage c06_wall41_x env OUT_DIR="$OUT_DIR/c06_wall41_x" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall41_matrix.sh"
run_stage c07_hbl env OUT_DIR="$OUT_DIR/c07_hbl" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c7_matrix.sh"
run_stage c08_hbl_filter env OUT_DIR="$OUT_DIR/c08_hbl_filter" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c8_filter_matrix.sh"
run_stage c09_selective_roe env OUT_DIR="$OUT_DIR/c09_selective_roe" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c9_selective_roe_matrix.sh"
run_stage c10_viscous_roe env OUT_DIR="$OUT_DIR/c10_viscous_roe" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c10_viscous_selective_roe_matrix.sh"
run_stage c11_closed_negative check_c11_policy
run_stage c12_nscbc_sponge env OUT_DIR="$OUT_DIR/c12_nscbc_sponge" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c12_nscbc_sponge_matrix.sh"
run_stage c13_symmetry env OUT_DIR="$OUT_DIR/c13_symmetry" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_symmetry_six_face_matrix.sh"
run_stage c14_physical_refinement env OUT_DIR="$OUT_DIR/c14_physical_refinement" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_physical_refinement.sh"
run_stage c15_performance_residency run_c15
run_stage c16_wall41_six_face env OUT_DIR="$OUT_DIR/c16_wall41_six_face" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall41_six_face_matrix.sh"
run_stage c17_zeroextrap_six_face env OUT_DIR="$OUT_DIR/c17_zeroextrap_six_face" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_zeroextrap_six_face_matrix.sh"
run_stage c18_wall42 env OUT_DIR="$OUT_DIR/c18_wall42" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall42_matrix.sh"
run_stage c19_slip_blowing env OUT_DIR="$OUT_DIR/c19_slip_blowing" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall_c19_matrix.sh"
run_stage c20_open_boundaries run_c20
run_stage c21_xyz_memcheck run_axis_memcheck

date --iso-8601=seconds > "$OUT_DIR/run_completed.txt"
python3 "$ROOT_DIR/tests/gpu_validation/summarize_curvilinear_c21.py" \
  --input "$OUT_DIR" --output "$OUT_DIR/c21_aggregate_report.md"
cat "$SUMMARY"
