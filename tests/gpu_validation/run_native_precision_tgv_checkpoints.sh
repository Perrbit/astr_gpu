#!/usr/bin/env bash
# Repeat the four long TGV cases with complete, rotating checkpoints.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=${1:?new checkpoint campaign directory required}
PYTHON=${PYTHON:-/usr/bin/python3}

printf 'START four 256^3 TGV cases, 40000 steps, checkpoint every 5000, keep two\n'
"$PYTHON" tests/gpu_validation/run_native_precision_tgv_long.py \
    --fp64 "$OUT/bin/astr_fp64" --fp32 "$OUT/bin/astr_fp32" \
    --out "$OUT" --mode run --steps 40000 --grid 256,256,256 \
    --interval 40 --checkpoint-interval 5000 >> "$OUT/campaign.log" 2>&1
printf 'COMPLETE four TGV runs; each retains checkpoints 35000 and 40000\n'
