#!/bin/bash
#SBATCH --job-name=astr-a800-cpu-gpu
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-cpu-gpu-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-cpu-gpu-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
EXPECTED_SOURCE_COMMIT=8aefd6c5e50e6e804e4fe47b5ffa7bdd78a386e4
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cpu_gpu_8aefd6c}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-$WORK_ROOT/build_cpu_gpu_compare_8aefd6c_cpu_v5}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$WORK_ROOT/build_cpu_gpu_compare_8aefd6c_gpu_v5}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
QUALIFICATION_JSON="${QUALIFICATION_JSON:-$WORK_ROOT/results/a800_cuda_aware_qualification_460370/qualification.json}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_cpu_gpu_comparison_${SLURM_JOB_ID:-dryrun}}"
TIMINGS="$RUN_ROOT/wall_timings.tsv"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
REPEATS=5
MAXSTEP=20
DELTAT=1.0e-4
DRY_RUN="${ASTR_CPU_GPU_COMPARISON_DRY_RUN:-0}"

matrix=(
  "tgv_cpu_np1 cpu tgv_256 1 1,1,1 256,256,256"
  "tgv_cpu_np8 cpu tgv_256 8 2,2,2 256,256,256"
  "tgv_cpu_np16 cpu tgv_256 16 4,2,2 256,256,256"
  "tgv_cpu_np32 cpu tgv_256 32 4,4,2 256,256,256"
  "tgv_gpu_np1 gpu tgv_256 1 1,1,1 256,256,256"
  "tgv_gpu_np4 gpu tgv_256 4 2,2,1 256,256,256"
  "shuosher_cpu_np1 cpu shuosher_400x128x128 1 1,1,1 400,128,128"
  "shuosher_gpu_np1 gpu shuosher_400x128x128 1 1,1,1 400,128,128"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'A800 matched CPU/GPU comparison dry-run\n'
  printf 'source=%s\ncpu_build=%s\ngpu_build=%s\nrun_root=%s\n' \
    "$SRC_DIR" "$CPU_BUILD_DIR" "$GPU_BUILD_DIR" "$RUN_ROOT"
  printf 'repeats=%s\nmaxstep=%s\ndeltat=%s\nfield_output=disabled\n' \
    "$REPEATS" "$MAXSTEP" "$DELTAT"
  printf 'case=%s\n' "${matrix[@]}"
  exit 0
fi

for path in "$SRC_DIR" "$CPU_BUILD_DIR" "$GPU_BUILD_DIR" "$HDF5_ROOT" \
  "$PYTHON_EXE" "$QUALIFICATION_JSON" "$RUN_ROOT"; do
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

CPU_EXE="$CPU_BUILD_DIR/bin/astr"
GPU_EXE="$GPU_BUILD_DIR/bin/astr"
PREPARE="$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py"
SUMMARY="$SRC_DIR/tests/gpu_validation/summarize_cpu_gpu_comparison.py"
for required in "$CPU_EXE" "$GPU_EXE" "$PYTHON_EXE" "$PREPARE" "$SUMMARY" "$QUALIFICATION_JSON"; do
  [[ -f "$required" ]] || { printf 'missing benchmark dependency: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$CPU_EXE" && -x "$GPU_EXE" && -x "$PYTHON_EXE" ]] || {
  printf 'required executable is not executable\n' >&2; exit 2;
}
[[ "$(git -C "$SRC_DIR" rev-parse HEAD)" == "$EXPECTED_SOURCE_COMMIT" ]] || {
  printf 'source commit does not match benchmark contract\n' >&2; exit 2;
}
git -C "$SRC_DIR" status --short > "$RUN_ROOT.source_status.tmp"
if grep -Ev '^( M src/benchmark_runtime.F90|\?\? tests/gpu_validation/(run_zhongke_a800_cpu_gpu_comparison.sbatch|summarize_cpu_gpu_comparison.py|test_cpu_gpu_comparison_job.py))$' \
  "$RUN_ROOT.source_status.tmp" | grep -q .; then
  printf 'source checkout contains changes outside the benchmark whitelist\n' >&2
  cat "$RUN_ROOT.source_status.tmp" >&2
  exit 2
fi
rm -f "$RUN_ROOT.source_status.tmp"
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite run root: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"

for exe in "$CPU_EXE" "$GPU_EXE"; do
  ldd "$exe" | tee "$RUN_ROOT/$(basename "$(dirname "$(dirname "$exe")")")_dependencies.txt"
  ! ldd "$exe" | grep -q 'not found' || { printf 'unresolved executable dependency\n' >&2; exit 2; }
done
mapfile -t flag_files < <(find "$CPU_BUILD_DIR" "$GPU_BUILD_DIR" -type f \( -name flags.make -o -name link.txt \) -print)
[[ "${#flag_files[@]}" -gt 0 ]] || { printf 'missing CMake flag manifests\n' >&2; exit 2; }
if grep -H -E -- '(--use_fast_math|-ffast-math|-Ofast|-Mfprelaxed|-Mfpapprox|-gpu=[^[:space:]]*fastmath|(^|[[:space:]])-fast([[:space:]]|$))' \
  "${flag_files[@]}" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math option found\n' >&2; exit 2
fi
while IFS= read -r input_file; do
  LC_ALL=C grep -q $'\r' "$input_file" && { printf 'CRLF found in %s\n' "$input_file" >&2; exit 2; }
done < <(find "$SRC_DIR/examples/Taylor_Green_Vortex/datin" "$SRC_DIR/examples/Shuosher/datin" -type f)
"$PYTHON_EXE" - "$QUALIFICATION_JSON" <<'PY'
import json, sys
if json.load(open(sys.argv[1], encoding="ascii")).get("production_qualified") is not True:
    raise SystemExit("CUDA-aware MPI is not production-qualified")
PY

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs: %s\n' "$visible" >&2; exit 2; }
gpu1="${gpu[0]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

validate_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log" || return 1
  ! grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"
}

prepare_case() {
  local family="$1" use_gpu="$2" grid="$3" target="$4"
  local input_name flowtype lfilter diffterm conschm extra=()
  if [[ "$family" == tgv_256 ]]; then
    input_name=input.tgv; flowtype=tgv; lfilter=t; diffterm=t; conschm=643e
    extra=(--src-case "$SRC_DIR/examples/Taylor_Green_Vortex")
  else
    input_name=input.shuosher; flowtype=shuosher; lfilter=f; diffterm=f; conschm=543e
    extra=(--src-case "$SRC_DIR/examples/Shuosher" --homogeneous t,t,t \
      --bctype 1,1,1,1,1,1 --recon-schem 3 --lchardecomp t)
  fi
  "$PYTHON_EXE" "$PREPARE" "${extra[@]}" --dst-case "$target" \
    --input-name "$input_name" --flowtype "$flowtype" --use-gpu "$use_gpu" \
    --grid "$grid" --maxstep "$MAXSTEP" --feqchkpt 9999 --feqlist 9999 \
    --deltat "$DELTAT" --lfilter "$lfilter" --diffterm "$diffterm" \
    --scheme 643e --conschm "$conschm" --difschm 643e --lreadgrid f
  rm -f "$target/datin/grid.h5"
}

run_once() {
  local label="$1" backend="$2" family="$3" np="$4" topology="$5" grid="$6" repeat="$7" record="$8"
  local run_dir="$RUN_ROOT/$label/run_$repeat" exe use_gpu input gpu_ids start end status found
  if [[ "$backend" == cpu ]]; then
    exe="$CPU_EXE"; use_gpu=f; gpu_ids="$gpu1"
  else
    exe="$GPU_EXE"; use_gpu=t; input=""; [[ "$np" -eq 1 ]] && gpu_ids="$gpu1" || gpu_ids="$gpu4"
  fi
  prepare_case "$family" "$use_gpu" "$grid" "$run_dir"
  [[ "$family" == tgv_256 ]] && input=datin/input.tgv || input=datin/input.shuosher
  start="$(date +%s.%N)"
  (
    cd "$run_dir" || exit 2
    env CUDA_VISIBLE_DEVICES="$gpu_ids" ASTR_FORCE_MPI_TOPOLOGY="$topology" \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING="$([[ "$backend" == gpu ]] && echo 1 || echo 0)" \
      ASTR_GPU_RANK_RK_TIMING=0 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=device-aware ASTR_GPU_FILTER_WORKSPACE=full \
      mpirun -np "$np" --bind-to core --map-by core "$exe" run "$input" > run.log 2>&1
  )
  status=$?
  end="$(date +%s.%N)"
  (( status == 0 )) || return "$status"
  validate_log "$run_dir/run.log" || return 1
  found="$(find "$run_dir" -type f -name '*.h5' -print)"
  [[ -z "$found" ]] || { printf 'forbidden field output:\n%s\n' "$found" >&2; return 1; }
  [[ "$record" == t ]] || return 0
  "$PYTHON_EXE" - "$family" "$backend" "$np" "$topology" "$repeat" "$start" "$end" >> "$TIMINGS" <<'PY'
import sys
print("\t".join(sys.argv[1:6] + [f"{float(sys.argv[7])-float(sys.argv[6]):.9f}"]))
PY
}

run_preflight() {
  run_once preflight_tgv_cpu cpu tgv_256 1 1,1,1 64,64,64 0 f || return $?
  run_once preflight_tgv_gpu gpu tgv_256 1 1,1,1 64,64,64 0 f || return $?
  run_once preflight_shuosher_cpu cpu shuosher_preflight 1 1,1,1 64,32,32 0 f || return $?
  run_once preflight_shuosher_gpu gpu shuosher_preflight 1 1,1,1 64,32,32 0 f || return $?
}

mkdir -p "$RUN_ROOT"
printf 'case\tbackend\tnp\ttopology\trepeat\twall_seconds\n' > "$TIMINGS"
printf 'label\tstatus\texit_code\n' > "$STATUS_FILE"
run_preflight || { printf 'preflight\tFAIL\t%s\n' "$?" >> "$STATUS_FILE"; exit 2; }
printf 'preflight\tPASS\t0\n' >> "$STATUS_FILE"

{
  printf 'source_commit=%s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  printf 'node=%s\ncuda_visible_devices=%s\n' "$(hostname)" "$visible"
  lscpu
  module -t list 2>&1
  mpirun --version
  nvidia-smi
  nvidia-smi topo -m
  sha256sum "$CPU_EXE" "$GPU_EXE" "$0" "$SUMMARY" "$SRC_DIR/src/benchmark_runtime.F90"
} > "$RUN_ROOT/environment.txt" 2>&1

for entry in "${matrix[@]}"; do
  read -r label backend family np topology grid <<< "$entry"
  run_once "$label" "$backend" "$family" "$np" "$topology" "$grid" warmup f
  warmup_status=$?
  if (( warmup_status != 0 )); then
    printf '%s\tFAIL\t%s\n' "$label" "$warmup_status" >> "$STATUS_FILE"
    continue
  fi
  case_status=0
  for repeat in $(seq 1 "$REPEATS"); do
    run_once "$label" "$backend" "$family" "$np" "$topology" "$grid" "$repeat" t || { case_status=$?; break; }
  done
  if (( case_status == 0 )); then
    printf '%s\tPASS\t0\n' "$label" >> "$STATUS_FILE"
  else
    printf '%s\tFAIL\t%s\n' "$label" "$case_status" >> "$STATUS_FILE"
  fi
done

if awk -F'\t' 'NR>1 && $2!="PASS" {bad=1} END {exit bad}' "$STATUS_FILE"; then
  "$PYTHON_EXE" "$SUMMARY" --timings "$TIMINGS" \
    --output-tsv "$RUN_ROOT/comparison_summary.tsv" \
    --summary "$RUN_ROOT/comparison_summary.md"
else
  printf 'one or more benchmark cases failed; summary not generated\n' >&2
  exit 1
fi
