#!/usr/bin/env bash
# Serial local FP8 timing followed by the approved long TGV comparisons.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-/usr/bin/python}
PERFORMANCE=${1:?performance evidence directory required}
ACCURACY=${2:?long TGV evidence directory required}

printf 'START native precision formal RK matrix\n'
"$PYTHON" tests/gpu_validation/run_native_precision_performance.py \
    --tgv-fp32 build_native_fp0_fp32_gpu/bin/astr \
    --tgv-fp64 build_native_fp0_fp64_gpu/bin/astr \
    --plate-fp32 build_native_fp3_fp32_gpu/bin/astr \
    --plate-fp64 build_native_fp3_fp64_gpu/bin/astr \
    --out "$PERFORMANCE" --mode timing --resume >> "$PERFORMANCE/timing.log" 2>&1

printf 'START native precision 256^3 TGV, four 40000-step runs\n'
"$PYTHON" tests/gpu_validation/run_native_precision_tgv_long.py \
    --fp32 build_native_fp0_fp32_gpu/bin/astr \
    --fp64 build_native_fp0_fp64_gpu/bin/astr \
    --out "$ACCURACY" --mode run >> "$PERFORMANCE/long_tgv.log" 2>&1

printf 'START paired TGV diagnostic figures\n'
"$PYTHON" - "$ACCURACY" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'scripts/gpu_validation')
from plot_native_precision_tgv import generate
source = Path(sys.argv[1]).resolve()
generate(source, source/'figures')
PY
printf 'COMPLETE native precision numerical campaign; reports await review\n'
