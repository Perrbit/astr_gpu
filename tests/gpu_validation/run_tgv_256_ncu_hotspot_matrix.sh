#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_256_ncu_hotspot_matrix}"
GRID="${GRID:-256,256,256}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-9999}"
GPU_ID="${GPU_ID:-0}"
SYNC_MODE="${SYNC_MODE:-explicit}"
NCU_SET="${NCU_SET:-full}"
KERNEL_FILTER="${KERNEL_FILTER:-.*}"
CASE_DIR="$OUT_DIR/ncu_case"
MANIFEST="$OUT_DIR/manifest.tsv"

if [[ "$GPU_EXE" != /* ]]; then
  GPU_EXE="$ROOT_DIR/$GPU_EXE"
fi
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
  CASE_DIR="$OUT_DIR/ncu_case"
  MANIFEST="$OUT_DIR/manifest.tsv"
fi

if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
command -v ncu >/dev/null 2>&1 || { echo "ncu not found" >&2; exit 127; }
if [[ "$FEQCHKPT" -le "$MAXSTEP" ]]; then
  echo "FEQCHKPT must exceed MAXSTEP so profiling does not write checkpoints" >&2
  exit 2
fi
if [[ "$SYNC_MODE" != "explicit" && "$SYNC_MODE" != "selective" ]]; then
  echo "SYNC_MODE must be explicit or selective" >&2
  exit 2
fi

kernels=(
  "conv_x|solver_gpu_convective_rhs_x_global_kernel_|0"
  "conv_y|solver_gpu_convective_rhs_y_global_kernel_|0"
  "conv_z|solver_gpu_convective_rhs_z_global_kernel_|0"
  "diff_flux|solver_gpu_diffusion_flux_global_kernel_|0"
  "diff_rhs_x|solver_gpu_diffusion_rhs_x_stored_global_kernel_|0"
  "diff_rhs_y|solver_gpu_diffusion_rhs_y_stored_global_kernel_|0"
  "diff_rhs_z|solver_gpu_diffusion_rhs_z_stored_global_kernel_|0"
  "filter_x|solver_gpu_filter_x_global_kernel_|0"
  "filter_y|solver_gpu_filter_y_pong_to_q_halo_global_kernel_|0"
  "filter_z|solver_gpu_filter_z_halo_global_kernel_|0"
  "primitive|solver_gpu_q_to_primitive_kernel_|1"
  "gradcal|gradcal_gpu_gradcal_dvel_kernel_|0"
)

mkdir -p "$OUT_DIR"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
  --dst-case "$CASE_DIR" --use-gpu t --grid "$GRID" \
  --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" \
  --lfilter t --diffterm t --scheme 643e
printf 'label\tkernel\treport\tsource\n' > "$MANIFEST"

for spec in "${kernels[@]}"; do
  IFS='|' read -r label kernel launch_skip <<< "$spec"
  if [[ ! "$label" =~ $KERNEL_FILTER ]]; then
    continue
  fi
  report="$OUT_DIR/$label"
  source_csv="$OUT_DIR/${label}_source.csv"
  echo "Profiling $label ($kernel)"
  (
    cd "$CASE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
      ASTR_GPU_SYNC_MODE="$SYNC_MODE" \
      ncu --target-processes all --target-processes-filter 'regex:^astr$' \
        --set "$NCU_SET" \
        --kernel-name "regex:$kernel" \
        --launch-skip "$launch_skip" --launch-count 1 \
        --import-source yes --source-folders "$ROOT_DIR/src_gpu" \
        --force-overwrite --export "$report" \
        mpirun -np 1 "$GPU_EXE" run datin/input.tgv \
        > "$OUT_DIR/${label}.log" 2>&1
  )
  ncu --import "${report}.ncu-rep" --page source \
    --print-source cuda,sass --csv --print-units base > "$source_csv"
  printf '%s\t%s\t%s\t%s\n' \
    "$label" "$kernel" "${report}.ncu-rep" "$source_csv" >> "$MANIFEST"
done

python3 "$ROOT_DIR/tests/gpu_validation/summarize_ncu_hotspot_matrix.py" \
  --manifest "$MANIFEST" \
  --tsv "$OUT_DIR/hotspot_matrix.tsv" \
  --markdown "$OUT_DIR/hotspot_matrix.md"
