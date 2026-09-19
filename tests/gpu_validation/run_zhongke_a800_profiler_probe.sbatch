#!/bin/bash
#SBATCH --job-name=astr-a800-prof-probe
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-prof-probe-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-prof-probe-%j.err

set -uo pipefail

ROOT=/data/user/hd56000/weiph
PROBE="$ROOT/a800_tgv_gpu_diagnostics_c801eeb/a800_profiler_probe"
CUDA_TRACE_CHECKER="$ROOT/a800_tgv_gpu_diagnostics_c801eeb/check_nsys_cuda_trace.py"
OUT="$ROOT/results/a800_profiler_probe_${SLURM_JOB_ID}"
NSYS24=/data/apps/nvhpc/24.11/Linux_x86_64/24.11/compilers/bin/nsys
NSYS25=/data/apps/nvhpc/25.5/Linux_x86_64/25.5/compilers/bin/nsys
NCU24=/data/apps/nvhpc/24.11/Linux_x86_64/24.11/compilers/bin/ncu
NCU25=/data/apps/nvhpc/25.5/Linux_x86_64/25.5/compilers/bin/ncu

module purge
module load gcc/11.5
module load nvhpc/25.5
mkdir -p "$OUT/tmp"
export TMPDIR="$OUT/tmp"
[[ -f "$CUDA_TRACE_CHECKER" ]] || {
  printf 'missing Nsight CUDA trace checker: %s\n' "$CUDA_TRACE_CHECKER" >&2
  exit 2
}

printf 'tool\tstatus\n' > "$OUT/status.tsv"

run_nsys() {
  local label="$1" executable="$2" report
  report="$OUT/$label"
  "$executable" profile --trace=cuda --sample=none --cpuctxsw=none \
    --stats=false --force-overwrite=true -o "$report" "$PROBE" \
    > "$OUT/$label.log" 2>&1
  local status=$?
  if (( status == 0 )); then
    "$executable" stats --force-export=true --report cuda_gpu_kern_sum \
      "$report.nsys-rep" > "$OUT/${label}_kernels.txt" 2>&1 || status=$?
    if (( status == 0 )); then
      python3 "$CUDA_TRACE_CHECKER" "$OUT/${label}.sqlite" \
        > "$OUT/${label}_cuda_trace_check.txt" 2>&1 || status=$?
    fi
  fi
  printf '%s\t%s\n' "$label" "$status" >> "$OUT/status.tsv"
}

run_ncu() {
  local label="$1" executable="$2" report
  report="$OUT/$label"
  "$executable" --set basic --kernel-name regex:saxpy_probe \
    --launch-count 1 --clock-control none --force-overwrite \
    --export "$report" "$PROBE" > "$OUT/$label.log" 2>&1
  local status=$?
  if (( status == 0 )); then
    "$executable" --import "$report.ncu-rep" --page raw --csv \
      > "$OUT/${label}_raw.csv" 2>&1 || status=$?
  fi
  printf '%s\t%s\n' "$label" "$status" >> "$OUT/status.tsv"
}

run_nsys nsys24 "$NSYS24"
run_nsys nsys25 "$NSYS25"
run_ncu ncu24 "$NCU24"
run_ncu ncu25 "$NCU25"

cat /proc/driver/nvidia/params > "$OUT/nvidia_driver_params.txt" 2>&1 || true
nvidia-smi -q > "$OUT/nvidia_smi_q.txt" 2>&1 || true
cat "$OUT/status.tsv"
