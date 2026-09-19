#!/bin/bash
#SBATCH --job-name=astr-cpu-rk-512
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-cpu-rk-512-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-cpu-rk-512-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cpu_rk_512}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-$WORK_ROOT/build_cpu_rk_512}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_cpu_rk_512_${SLURM_JOB_ID:-dryrun}}"
GRID=512,512,512
MAXSTEP=5
DELTAT=1.0e-4
DRY_RUN="${ASTR_CPU_RK_SCALING_DRY_RUN:-0}"

matrix=(
  "1 1,1,1"
  "2 2,1,1"
  "4 2,2,1"
  "8 2,2,2"
  "16 4,2,2"
  "32 4,4,2"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CPU complete-RK scaling dry-run\n'
  printf 'source=%s\nbuild=%s\nrun_root=%s\ngrid=%s\nmaxstep=%s\n' \
    "$SRC_DIR" "$CPU_BUILD_DIR" "$RUN_ROOT" "$GRID" "$MAXSTEP"
  printf 'case=%s\n' "${matrix[@]}"
  exit 0
fi

for path in "$SRC_DIR" "$CPU_BUILD_DIR" "$HDF5_ROOT" "$PYTHON_EXE" "$RUN_ROOT"; do
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
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_coll='^hcoll,ucc,cuda'
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_sharedfp=individual

CPU_EXE="$CPU_BUILD_DIR/bin/astr"
PREPARE="$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py"
SUMMARY="$SRC_DIR/tests/gpu_validation/summarize_cpu_rk_scaling.py"
for required in "$CPU_EXE" "$PYTHON_EXE" "$PREPARE" "$SUMMARY"; do
  [[ -f "$required" ]] || { printf 'missing benchmark dependency: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$CPU_EXE" && -x "$PYTHON_EXE" ]] || {
  printf 'required executable is not executable\n' >&2; exit 2;
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite run root: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"

ldd "$CPU_EXE" | tee "$RUN_ROOT/cpu_dependencies.txt"
! ldd "$CPU_EXE" | grep -q 'not found' || { printf 'unresolved executable dependency\n' >&2; exit 2; }
mapfile -t flag_files < <(find "$CPU_BUILD_DIR" -type f \( -name flags.make -o -name link.txt \) -print)
[[ "${#flag_files[@]}" -gt 0 ]] || { printf 'missing CMake flag manifests\n' >&2; exit 2; }
if grep -H -E -- '(--use_fast_math|-ffast-math|-Ofast|-Mfprelaxed|-Mfpapprox|-gpu=[^[:space:]]*fastmath|(^|[[:space:]])-fast([[:space:]]|$))' \
  "${flag_files[@]}" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math option found\n' >&2
  exit 2
fi
while IFS= read -r input_file; do
  LC_ALL=C grep -q $'\r' "$input_file" && { printf 'CRLF found in %s\n' "$input_file" >&2; exit 2; }
done < <(find "$SRC_DIR/examples/Taylor_Green_Vortex/datin" -type f)

prepare_case() {
  local target="$1" grid="$2" maxstep="$3"
  "$PYTHON_EXE" "$PREPARE" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" \
    --dst-case "$target" --input-name input.tgv --flowtype tgv --use-gpu f \
    --grid "$grid" --maxstep "$maxstep" --feqchkpt 9999 --feqlist 9999 \
    --deltat "$DELTAT" --lfilter t --diffterm t \
    --scheme 643e --conschm 643e --difschm 643e --lreadgrid f
  rm -f "$target/datin/grid.h5"
}

validate_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log" || return 1
  ! grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log" || return 1
  [[ "$(grep -ac '^ASTR_CPU_RK_TIMING ' "$log")" -eq 6 ]] || {
    printf 'expected six CPU RK timing records in %s\n' "$log" >&2
    return 1
  }
  awk '$1=="ASTR_CPU_RK_TIMING" {print $2}' "$log" | diff -u <(seq 0 5) - >/dev/null
}

run_case() {
  local np="$1" topology="$2" grid="$3" maxstep="$4" target="$5" log="$6"
  prepare_case "$target" "$grid" "$maxstep" || return $?
  (
    cd "$target" || exit 2
    env ASTR_FORCE_MPI_TOPOLOGY="$topology" \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_CPU_RK_TIMING=1 ASTR_GPU_RK_TIMING=0 \
      mpirun -np "$np" --bind-to core --map-by core \
      "$CPU_EXE" run datin/input.tgv > "$log" 2>&1
  ) || return $?
  validate_log "$log" || return 1
  local found
  found="$(find "$target" -type f -name '*.h5' -print)"
  [[ -z "$found" ]] || { printf 'forbidden field output:\n%s\n' "$found" >&2; return 1; }
}

run_preflight() {
  local target="$RUN_ROOT/preflight"
  mkdir -p "$target"
  run_case 2 2,1,1 64,64,64 5 "$target" "$RUN_ROOT/preflight.log"
}

run_preflight || { printf 'CPU timing preflight failed\n' >&2; exit 2; }

{
  printf 'source_commit=%s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  git -C "$SRC_DIR" status --short
  printf 'node=%s\n' "$(hostname)"
  lscpu
  module -t list 2>&1
  mpirun --version
  sha256sum "$CPU_EXE" "$0" "$SUMMARY" "$SRC_DIR/src/benchmark_runtime.F90" "$SRC_DIR/src/mainloop.F90"
} > "$RUN_ROOT/environment.txt" 2>&1

printf 'np\ttopology\tstatus\texit_code\n' > "$RUN_ROOT/case_status.tsv"
failed=0
for entry in "${matrix[@]}"; do
  read -r np topology <<< "$entry"
  target="$RUN_ROOT/np${np}"
  mkdir -p "$target"
  if run_case "$np" "$topology" "$GRID" "$MAXSTEP" "$target" "$RUN_ROOT/logs/np${np}.log"; then
    printf '%s\t%s\tPASS\t0\n' "$np" "$topology" >> "$RUN_ROOT/case_status.tsv"
  else
    status=$?
    printf '%s\t%s\tFAIL\t%s\n' "$np" "$topology" "$status" >> "$RUN_ROOT/case_status.tsv"
    failed=1
  fi
done

(( failed == 0 )) || { printf 'one or more CPU scaling cases failed\n' >&2; exit 1; }
"$PYTHON_EXE" "$SUMMARY" --logs "$RUN_ROOT/logs" \
  --output-tsv "$RUN_ROOT/cpu_rk_scaling.tsv" \
  --summary "$RUN_ROOT/cpu_rk_scaling.md"
cat "$RUN_ROOT/cpu_rk_scaling.md"
