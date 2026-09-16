#!/bin/bash
#SBATCH --job-name=astr-a800-devmpi-scale
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-devmpi-scale-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-devmpi-scale-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cuda_aware_qualification_c801eeb}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_cuda_aware_qualification_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_device_aware_scaling_${SLURM_JOB_ID:-dryrun}}"
QUALIFICATION_JSON="$WORK_ROOT/results/a800_cuda_aware_qualification_460370/qualification.json"
EXPECTED_GPU_EXE_SHA256=c19d05a2b496161f267c19d493c8303cef8f8b0cc838231f61bdce120d7f1fbc
GPU_EXE="$BUILD_DIR/bin/astr"
BENCHMARK_DRIVER="$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
SUMMARY_DRIVER="$SRC_DIR/tests/gpu_validation/summarize_device_aware_scaling.py"
PREPARE_CASE="$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
MANIFEST="$RUN_ROOT/matrix_manifest.tsv"
MAXSTEP=21
DISCARD_STEPS=2
REPEATS=5
DELTAT=1.0e-4
FILTER_WORKSPACE=full
EXPECTED_CASES=19
EXPECTED_PAIRS=9
DRY_RUN="${ASTR_DEVICE_AWARE_SCALING_DRY_RUN:-0}"

case_matrix=(
  "strong_np1_111_pinned 1 1,1,1 512,512,512 512,512,512 pinned"
  "strong_np2_211_pinned_pipeline 2 2,1,1 512,512,512 256,512,512 pinned-pipeline"
  "strong_np2_211_device_aware 2 2,1,1 512,512,512 256,512,512 device-aware"
  "strong_np2_121_device_aware 2 1,2,1 512,512,512 512,256,512 device-aware"
  "strong_np2_121_pinned_pipeline 2 1,2,1 512,512,512 512,256,512 pinned-pipeline"
  "strong_np2_112_pinned_pipeline 2 1,1,2 512,512,512 512,512,256 pinned-pipeline"
  "strong_np2_112_device_aware 2 1,1,2 512,512,512 512,512,256 device-aware"
  "strong_np4_411_device_aware 4 4,1,1 512,512,512 128,512,512 device-aware"
  "strong_np4_411_pinned_pipeline 4 4,1,1 512,512,512 128,512,512 pinned-pipeline"
  "strong_np4_141_pinned_pipeline 4 1,4,1 512,512,512 512,128,512 pinned-pipeline"
  "strong_np4_141_device_aware 4 1,4,1 512,512,512 512,128,512 device-aware"
  "strong_np4_114_device_aware 4 1,1,4 512,512,512 512,512,128 device-aware"
  "strong_np4_114_pinned_pipeline 4 1,1,4 512,512,512 512,512,128 pinned-pipeline"
  "strong_np4_221_pinned_pipeline 4 2,2,1 512,512,512 256,256,512 pinned-pipeline"
  "strong_np4_221_device_aware 4 2,2,1 512,512,512 256,256,512 device-aware"
  "strong_np4_212_device_aware 4 2,1,2 512,512,512 256,512,256 device-aware"
  "strong_np4_212_pinned_pipeline 4 2,1,2 512,512,512 256,512,256 pinned-pipeline"
  "strong_np4_122_pinned_pipeline 4 1,2,2 512,512,512 512,256,256 pinned-pipeline"
  "strong_np4_122_device_aware 4 1,2,2 512,512,512 512,256,256 device-aware"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'A800 CUDA-aware MPI strong-scaling dry-run\n'
  printf 'source=%s\nbuild=%s\nrun_root=%s\n' "$SRC_DIR" "$BUILD_DIR" "$RUN_ROOT"
  printf 'grid=512,512,512\nmaxstep=%s\ndiscard_steps=%s\nrepeats=%s\n' \
    "$MAXSTEP" "$DISCARD_STEPS" "$REPEATS"
  printf 'filter_workspace=%s\nsync_mode=explicit\nfield_hdf5=disabled\n' "$FILTER_WORKSPACE"
  printf 'case=%s\n' "${case_matrix[@]}"
  exit 0
fi

for path in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$PYTHON_EXE" "$RUN_ROOT" \
  "$QUALIFICATION_JSON"; do
  case "$path/" in
    "$WORK_ROOT"/*) ;;
    *) printf 'path escapes approved work root: %s\n' "$path" >&2; exit 2 ;;
  esac
done

module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$(dirname "$PYTHON_EXE"):$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll='^hcoll,ucc,cuda'
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_sharedfp=individual
export UCX_MEMTYPE_CACHE=n
export UCX_CUDA_COPY_ENABLE_FABRIC=no
export UCX_CUDA_COPY_DMABUF=no
export UCX_CUDA_IPC_ENABLE_MNNVL=no
export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_PROTO_INFO=n

for required in "$GPU_EXE" "$PYTHON_EXE" "$QUALIFICATION_JSON" \
  "$BENCHMARK_DRIVER" "$SUMMARY_DRIVER" "$PREPARE_CASE"; do
  [[ -f "$required" ]] || { printf 'missing scaling dependency: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$GPU_EXE" && -x "$PYTHON_EXE" ]] || {
  printf 'required executable is not executable\n' >&2
  exit 2
}
[[ "$(sha256sum "$GPU_EXE" | awk '{print $1}')" == "$EXPECTED_GPU_EXE_SHA256" ]] || {
  printf 'GPU executable hash does not match the admitted solver\n' >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/timing" "$RUN_ROOT/preflight"

if ! ldd "$GPU_EXE" > "$RUN_ROOT/gpu_dependencies.txt" 2>&1 || \
   grep -q 'not found' "$RUN_ROOT/gpu_dependencies.txt"; then
  printf 'unresolved compute-node ASTR dependency\n' >&2
  exit 2
fi
grep -Fq "$HDF5_ROOT/lib/libhdf5_fortran" "$RUN_ROOT/gpu_dependencies.txt" || {
  printf 'ASTR does not use the approved HDF5 Fortran runtime\n' >&2
  exit 2
}
mapfile -t flag_files < <(find "$BUILD_DIR" -type f \( -name 'flags.make' -o -name 'link.txt' \) -print)
[[ "${#flag_files[@]}" -gt 0 ]] || { printf 'missing CMake compiler flag manifests\n' >&2; exit 2; }
if grep -H -E -- '(--use_fast_math|-ffast-math|-Ofast|-Mfprelaxed|-Mfpapprox|-gpu=[^[:space:]]*fastmath|(^|[[:space:]])-fast([[:space:]]|$))' \
  "${flag_files[@]}" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi
for input_file in \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/input.tgv" \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/controller" \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/parallel.info"; do
  [[ -f "$input_file" ]] || continue
  if LC_ALL=C grep -q $'\r' "$input_file"; then
    printf 'CRLF or bare carriage return found in %s\n' "$input_file" >&2
    exit 2
  fi
done
grep -q '"$HALO_TRANSPORT" != "device-aware"' "$BENCHMARK_DRIVER" || {
  printf 'benchmark driver does not accept device-aware transport\n' >&2
  exit 2
}
"$PYTHON_EXE" - "$QUALIFICATION_JSON" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
if data.get("production_qualified") is not True:
    raise SystemExit("A800 CUDA-aware MPI probe is not production_qualified")
PY

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs: %s\n' "$visible" >&2; exit 2; }
gpu1="${gpu[0]}"
gpu2="${gpu[0]},${gpu[1]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

gpu_ids_for_np() {
  case "$1" in
    1) printf '%s\n' "$gpu1" ;;
    2) printf '%s\n' "$gpu2" ;;
    4) printf '%s\n' "$gpu4" ;;
    *) return 2 ;;
  esac
}

assert_no_field_hdf5() {
  local root="$1" found
  found="$(find "$root" -type f \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  [[ -z "$found" ]] || { printf 'forbidden field HDF5 output:\n%s\n' "$found" >&2; return 1; }
}

run_compute_preflight() {
  local case_dir="$RUN_ROOT/preflight/case" log="$RUN_ROOT/preflight/run.log" status
  "$PYTHON_EXE" "$PREPARE_CASE" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid 64,64,64 --maxstep 1 --feqchkpt 9999 --feqlist 9999 \
    --deltat "$DELTAT" --lfilter t --diffterm t --scheme 643e --lreadgrid f || return $?
  rm -f "$case_dir/datin/grid.h5"
  (
    cd "$case_dir" || exit 2
    env CUDA_VISIBLE_DEVICES="$gpu4" UCX_PROTO_INFO=y \
      ASTR_FORCE_MPI_TOPOLOGY=2,2,1 ASTR_GPU_RK_TIMING=1 \
      ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=device-aware ASTR_GPU_FILTER_WORKSPACE=full \
      mpirun -np 4 "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$log" || return 1
  grep -aq 'cuda_ipc/cuda' "$log" || return 1
  [[ "$(grep -ac 'ASTR_GPU_PRE_MPI_DEVICE=' "$log")" -eq 4 ]] || return 1
  ! grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log" || return 1
  assert_no_field_hdf5 "$case_dir"
}

run_compute_preflight || {
  printf 'device-aware compute-node preflight failed; scaling matrix will not run\n' >&2
  exit 2
}

{
  date -Iseconds
  hostname
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$visible"
  printf 'UCX_TLS=%s\n' "$UCX_TLS"
  printf 'grid=512,512,512\nmaxstep=%s\ndiscard_steps=%s\nrepeats=%s\n' \
    "$MAXSTEP" "$DISCARD_STEPS" "$REPEATS"
  printf 'precision=FP64\nfilter_workspace=%s\nsync_mode=explicit\nfield_hdf5=disabled\n' \
    "$FILTER_WORKSPACE"
  sha256sum "$GPU_EXE" "$BENCHMARK_DRIVER" "$SUMMARY_DRIVER"
  module -t list 2>&1
  mpirun --version
  nvidia-smi
  nvidia-smi topo -m
  ldd "$GPU_EXE"
} > "$RUN_ROOT/environment.txt" 2>&1

printf 'label\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\n' > "$MANIFEST"
for entry in "${case_matrix[@]}"; do
  read -r label np topology grid local_grid backend <<< "$entry"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$np" "$topology" "$grid" "$local_grid" "$backend" >> "$MANIFEST"
done
[[ "$(($(wc -l < "$MANIFEST") - 1))" -eq "$EXPECTED_CASES" ]] || {
  printf 'internal matrix size mismatch\n' >&2
  exit 2
}
printf 'case\tstatus\texit_code\tstarted\tfinished\tdetail\n' > "$STATUS_FILE"

verify_case() {
  local label="$1" backend="$2" out="$RUN_ROOT/timing/$label" count=0 log
  [[ -s "$out/${label}_timings.tsv" ]] || return 1
  [[ "$(($(wc -l < "$out/${label}_timings.tsv") - 1))" -eq "$REPEATS" ]] || return 1
  while IFS= read -r -d '' log; do
    count=$((count + 1))
    grep -aq 'GPU filter workspace storage: full' "$log" || return 1
    grep -aq "ASTR_GPU_HALO_TRANSPORT_SELECTED=$backend" "$log" || return 1
    if [[ "$backend" == device-aware ]]; then
      grep -aq 'ASTR_GPU_PRE_MPI_DEVICE=' "$log" || return 1
    fi
  done < <(find "$out" -type f -name run.log -print0)
  [[ "$count" -eq $((REPEATS + 1)) ]] || return 1
  assert_no_field_hdf5 "$out"
}

run_benchmark() {
  local label="$1" np="$2" topology="$3" grid="$4" backend="$5" gpu_ids
  gpu_ids="$(gpu_ids_for_np "$np")" || return $?
  GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/timing/$label" LABEL="$label" \
    GRID="$grid" MAXSTEP="$MAXSTEP" DISCARD_STEPS="$DISCARD_STEPS" \
    REPEATS="$REPEATS" DELTAT="$DELTAT" NP="$np" TOPOLOGY="$topology" \
    GPU_IDS="$gpu_ids" FILTER_WORKSPACE="$FILTER_WORKSPACE" \
    HALO_TRANSPORT="$backend" SYNC_MODE=explicit PHASE_TIMING=0 FEQCHKPT=9999 \
    "$BENCHMARK_DRIVER" || return $?
  verify_case "$label" "$backend"
}

run_case() {
  local label="$1" np="$2" topology="$3" grid="$4" backend="$5"
  local started finished status
  started="$(date -Iseconds)"
  printf 'CASE_START %s %s\n' "$label" "$started"
  run_benchmark "$label" "$np" "$topology" "$grid" "$backend"
  status=$?
  finished="$(date -Iseconds)"
  if (( status == 0 )); then
    printf '%s\tPASS\t0\t%s\t%s\t%s\n' \
      "$label" "$started" "$finished" "$RUN_ROOT/timing/$label" >> "$STATUS_FILE"
  else
    printf '%s\tFAIL\t%s\t%s\t%s\t%s\n' \
      "$label" "$status" "$started" "$finished" "$RUN_ROOT/timing/$label" >> "$STATUS_FILE"
  fi
  printf 'CASE_END %s status=%s %s\n' "$label" "$status" "$finished"
  return 0
}

for entry in "${case_matrix[@]}"; do
  read -r label np topology grid local_grid backend <<< "$entry"
  run_case "$label" "$np" "$topology" "$grid" "$backend"
done

"$PYTHON_EXE" "$SUMMARY_DRIVER" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT/timing" \
  --output-tsv "$RUN_ROOT/scaling_results.tsv" \
  --pair-tsv "$RUN_ROOT/backend_comparison.tsv" \
  --summary "$RUN_ROOT/scaling_summary.md" \
  --expected-cases "$EXPECTED_CASES" --expected-pairs "$EXPECTED_PAIRS" \
  > "$RUN_ROOT/scaling_summary.stdout" 2>&1
summary_status=$?
passed="$(awk -F '\t' '$2=="PASS" {n++} END {print n+0}' "$STATUS_FILE")"
failed="$(awk -F '\t' '$2=="FAIL" {n++} END {print n+0}' "$STATUS_FILE")"
pairs=0
if [[ -s "$RUN_ROOT/backend_comparison.tsv" ]]; then
  pairs="$(($(wc -l < "$RUN_ROOT/backend_comparison.tsv") - 1))"
fi
{
  printf 'expected_cases=%s\npassed=%s\nfailed=%s\n' "$EXPECTED_CASES" "$passed" "$failed"
  printf 'expected_pairs=%s\ncompleted_pairs=%s\nsummary_status=%s\n' \
    "$EXPECTED_PAIRS" "$pairs" "$summary_status"
} > "$RUN_ROOT/job_status.txt"

if [[ "$passed" -eq "$EXPECTED_CASES" && "$failed" -eq 0 && \
      "$pairs" -eq "$EXPECTED_PAIRS" && "$summary_status" -eq 0 ]]; then
  printf 'A800_DEVICE_AWARE_SCALING=PASS root=%s\n' "$RUN_ROOT"
  exit 0
fi
printf 'A800_DEVICE_AWARE_SCALING=PARTIAL root=%s\n' "$RUN_ROOT" >&2
exit 1
