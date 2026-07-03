# GPU Validation

This directory contains validation drivers for the current ASTR CUDA Fortran port.

Current validated scope:

- Taylor-Green Vortex only
- single-rank and multi-rank MPI decompositions
- one physical GPU, two physical GPUs, and two-GPU oversubscription correctness smoke tests
- q(1:5) only
- explicit sixth-order central difference
- explicit tenth-order central filter
- detect-only crash check
- LF runtime input files
- runtime `use_gpu=t/f`; GPU support is compiled with `ASTR_WITH_CUDA=ON`
- file output, checkpoint, and HDF5 field writing remain CPU-owned output boundaries

Validation order:

1. L0 build and smoke
2. L1 module-level CPU/GPU field diff
3. L2 one-step and ten-step integration diff
4. L3 multi-rank topology correctness
5. L4 TGV physical diagnostics
6. L5 performance and residency profiling

Do not report GPU speedup until correctness, multi-rank topology coverage, and residency profiling all pass. Two-GPU oversubscription runs are correctness smoke tests only, not performance evidence.

## Build Contract

Build from the repository root `CMakeLists.txt`.

CPU baseline:

```bash
cmake -B build_cpu_probe -S /home/dell/workspace/astr_gpu -DCMAKE_Fortran_COMPILER=mpif90
cmake --build build_cpu_probe -j4
```

CUDA-capable binary:

```bash
cmake -B build_gpu_probe -S /home/dell/workspace/astr_gpu -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON
cmake --build build_gpu_probe -j4
```

`ASTR_WITH_CUDA=ON` compiles GPU support into the binary. The actual CPU/GPU path is selected at runtime by the input-file `use_gpu` flag.

## Line Ending Check

Run:

```bash
find examples -type f \( -name 'input.*' -o -name 'controller' \) -print0 \
  | xargs -0 file | rg 'CRLF'
```

Expected: no output.

## Native TGV Statistics Compare

Run CPU/GPU `flowstate.dat` statistics comparison with isolated case copies:

```bash
MAXSTEP=10 LFILTER=f DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_stats_compare_643e_diff_no_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_stats_compare_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh
```

The driver rewrites the copied TGV input to `643e,643e` so CPU and GPU use the explicit sixth-order central derivative policy. It also appends runtime `use_gpu=t` only for the GPU case.

Current expected result: both commands print `status: pass` with `flowstate.dat` differences within `1e-10`. In the GPU-resident path, `flowstate.dat` is written directly from GPU reductions rather than by CPU `statistic.F90`.

## Full-Field HDF5 Compare

Run CPU/GPU `outdat/flowfield.h5` comparison with reconstructed conservative variables:

```bash
MAXSTEP=10 LFILTER=f DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_field_compare_643e_diff_no_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_field_compare_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh
```

The driver compares primitive datasets `ro,u1,u2,u3,p,t`, reconstructs `q1:q5` from the HDF5 output, and records `ro/p/t` extrema. Current expected result: both commands print `status: pass` with `L_inf` within `1e-10`.

## TGV Physical Diagnostics

Generate CPU/GPU diagnostic plots from native `flowstate.dat` output:

```bash
python3 scripts/gpu_validation/plot_tgv_diagnostics.py
```

The script writes EPS and JPEG figures plus `tgv_diagnostics_summary.txt` under `tests/gpu_validation/out/tgv_diagnostics_l3` by default. It currently covers `kenergy`, `enstophy`, and `dissipation`.

## GPU Kinetic-Energy Scalar Reduction

Validate the GPU-resident TGV statistics slices:

```bash
MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

python3 tests/gpu_validation/compare_gpu_kenergy.py \
  --cpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/cpu \
  --gpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu_kenergy_compare.txt \
  --atol 1e-12 --rtol 1e-12

python3 tests/gpu_validation/compare_gpu_enstophy.py \
  --cpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/cpu \
  --gpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu_enstophy_compare.txt \
  --atol 1e-12 --rtol 1e-12

python3 tests/gpu_validation/compare_gpu_dissipation.py \
  --cpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/cpu \
  --gpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu_dissipation_compare.txt \
  --atol 1e-12 --rtol 1e-12
```

The GPU run writes native `flowstate.dat` plus `gpu_kenergy.dat`, `gpu_enstophy.dat`, and `gpu_dissipation.dat` from device-side reductions. The time loop keeps full flow variables resident on the GPU. File output, checkpoint, and HDF5 `flowfield` writing remain CPU-owned and are intentionally out of current GPU-porting scope; full-field D2H at those boundaries is acceptable.

For the explicit 10th-order filter, the GPU path uses ping-pong storage and halo stencil kernels. In single-rank homogeneous y/z directions it refreshes local halos for `qwork_d` and `q_d` before launching the y/z halo filter kernels; in multi-rank directions it uses the corresponding MPI halo exchange before the same halo filter kernels.

## GPU Single-Rank Qswap

Validate the first-stage GPU periodic halo and periodic-plane averaging path:

```bash
MAXSTEP=1 LFILTER=f DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_qswap_643e_diff_no_filter_1 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_qswap_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_qswap_field_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh
```

Current expected result: all commands print `status: pass`. The full-field check has reconstructed conservative-variable `q5` `L_inf` around `8.73e-12`.

## GPU-Resident Loop Check

Validate the current GPU-resident TGV path:

```bash
MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_resident_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_resident_field_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh
```

For a no-checkpoint runtime copy check, profile a short GPU case with `feqchkpt` greater than `maxstep`. The current 3-step profile under `tests/gpu_validation/out/tgv_gpu_resident_nsys_3_nochk` showed D2H copies only from reduction partial sums: 18 copies, each 65,536 bytes, and no full-field D2H in the compute loop.

## Phase A 2dvort Validation

Validate the first non-TGV explicit case. The driver starts from `examples/Vortex_Transport/datin/input.2dvort`, rewrites it to a 3D extruded periodic case, and forces explicit `643e,643e`:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_phasea_smoke \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_2dvort_phasea_compare.sh
```

Current expected result: both `flowstate.dat` and `flowfield.h5` comparisons print `status: pass`. The default filtered Phase A thresholds are `STATS_ATOL=1e-9`, `STATS_RTOL=1e-10`, `FIELD_ATOL=1e-9`, and `FIELD_RTOL=1e-10`. The most recent filtered run had max stats differences `maxq4=7.8810021050800005e-10` and `maxq5=2.7473134878164274e-11`; final field reconstructed errors were `q2=2.6846691536519529e-10`, `q3=2.6549052794524672e-10`, `q4=7.9986820163237011e-10`, and `q5=3.3623592798903701e-10`.

A stricter filtered field threshold of `1e-10` is still too tight for this Phase A case because CPU produces about `8e-10` of roundoff in the physically zero `u3/q4` component while GPU keeps that component exactly zero. The no-filter strict check below still passes to roundoff, so this is tracked as filtered non-TGV roundoff sensitivity rather than a baseline RHS/RK/copy-back failure.

Run the no-filter strict isolation check:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_phasea_no_filter_1 \
  MAXSTEP=1 FEQCHKPT=1 LFILTER=f STATS_ATOL=1e-10 FIELD_ATOL=1e-10 \
  tests/gpu_validation/run_2dvort_phasea_compare.sh
```

Current expected result: strict no-filter comparison prints `status: pass` for both reports. This isolates the RHS/RK/non-TGV initialization path from the explicit-filter roundoff sensitivity; the most recent no-filter field errors were at roundoff scale, with reconstructed `q5` `L_inf=7.1054273576010019e-15`.

Run the first `2dvort` multi-rank matrix:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_mpirank_matrix_np2 \
  tests/gpu_validation/run_2dvort_mpirank_matrix.sh
```

The matrix covers `NP=2` with `2x1x1`, `1x2x1`, and `1x1x2`. No-filter cases use `STATS_ATOL=1e-10` and `FIELD_ATOL=1e-10`. Filtered cases use `STATS_ATOL=1e-9` and `FILTER_FIELD_ATOL=5e-9`; the field tolerance is separated from the native statistics tolerance because interface-adjacent filtered HDF5 reconstructed energy differs by about `4.7e-9` while `flowstate.dat` remains within `1e-9`.

The optional `NP=4` core matrix can be run with:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_mpirank_matrix_np4 \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' FILTER_FIELD_ATOL=6e-9 \
  tests/gpu_validation/run_2dvort_mpirank_matrix.sh
```

The higher filtered field tolerance is required by the `2x2x1` reconstructed `q5` interface-adjacent difference, about `5.2e-9`; native filtered statistics still use `1e-9`.

The larger `256^3 NP=1/NP=2` Nsight profile can be generated with:

```bash
OUT_DIR=tests/gpu_validation/out/nsys_tgv_256_np1_np2 \
  tests/gpu_validation/run_tgv_256_nsys_profile.sh
```

Useful controls:

```bash
GRID=256,256,256 MAXSTEP=10 FEQCHKPT=9999 LFILTER=t DIFFTERM=t \
  NP2_TOPOLOGY=2,1,1 \
  tests/gpu_validation/run_tgv_256_nsys_profile.sh
```

The driver prepares GPU-only `NP=1` and `NP=2` case copies, profiles both with Nsight Systems, and compares their `flowstate.dat` statistics. Existing profile evidence was generated before this driver was added; rerun the driver before marking the reusable profile test as passed.

## GPU Multi-Rank X-Slab Validation

Validate the current two-rank, two-GPU TGV x-slab path:

```bash
MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_stats_compare_postgit \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_field_compare_postgit \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The post-git run had max statistic differences `kenergy=2.9254376698872875e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=8.5597761517730575e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=5.8832938520936295e-12`.

Validate the two-rank, two-GPU y-slab path:

```bash
MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_yslab_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_yslab_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The y-slab stats run had max differences `kenergy=2.9226621123257246e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=8.5489341300482025e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.1106675275368616e-12`.

Validate the two-rank, two-GPU z-slab path:

```bash
MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=1,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_zslab_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=1,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_zslab_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The z-slab stats run had max differences `kenergy=2.9240498911065060e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=5.7571135358980285e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.2243543652584776e-12`.

Validate the four-rank x-y combined topology on a two-GPU workstation. This is an oversubscription correctness test, not a performance test:

```bash
NP=4 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x2x1_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=4 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x2x1_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9254376698872875e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.5654148784233257e-12`. The GPU log should show ranks `0/2` sharing GPU 0 and ranks `1/3` sharing GPU 1.

Validate the four-rank x-z combined topology on a two-GPU workstation. This is also an oversubscription correctness test, not a performance test:

```bash
NP=4 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x1x2_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=4 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=2,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x1x2_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9254376698872875e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=5.9685589803848416e-12`. The GPU log should show ranks `0/2` sharing GPU 0 and ranks `1/3` sharing GPU 1.

Validate the four-rank y-z combined topology on a two-GPU workstation. This is also an oversubscription correctness test, not a performance test:

```bash
NP=4 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_1x2x2_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=4 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_1x2x2_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9226621123257246e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=5.9685589803848416e-12`. The GPU log should show ranks `0/2` sharing GPU 0 and ranks `1/3` sharing GPU 1.

Validate the eight-rank full x-y-z combined topology on a two-GPU workstation. This is also an oversubscription correctness test, not a performance test:

```bash
NP=8 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank8_2x2x2_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=8 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank8_2x2x2_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9240498911065060e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.4517280407017097e-12`. The GPU log should show ranks `0/2/4/6` sharing GPU 0 and ranks `1/3/5/7` sharing GPU 1.

Validate higher-rank full x-y-z topology combinations on a two-GPU workstation. These are native-statistics smoke tests only: they use `MAXSTEP=1`, disable checkpoint field output with `FEQCHKPT=99`, and are meant to exercise rank topology and face-halo composition under oversubscription. They are not field-output comparisons and not performance tests.

```bash
NP=16 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=4,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank16_4x2x2_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=16 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,4,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank16_2x4x2_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=16 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,4 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank16_2x2x4_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=32 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=4,4,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank32_4x4x2_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh
```

Current expected result: all four commands print `status: pass`. The `4x2x2`, `2x4x2`, and `2x2x4` smoke runs had max differences `kenergy=1.1227130336521896e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`. The `4x4x2` smoke run had max differences `kenergy=1.1241008124329710e-14`, `enstophy=8.2711615334574162e-15`, and `dissipation=4.2392304944183223e-17`.

For a two-rank no-checkpoint transfer audit, prepare a GPU-only case with `feqchkpt` greater than `maxstep` and profile MPI plus CUDA:

```bash
OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_nsys_nochk_postgit
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 tests/gpu_validation/prepare_tgv_case.py \
  --src-case examples/Taylor_Green_Vortex \
  --dst-case "$OUT_DIR/gpu" \
  --use-gpu t \
  --maxstep 1 \
  --feqchkpt 99 \
  --lfilter t \
  --diffterm t \
  --scheme 643e

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY=2,1,1 \
    nsys profile --trace=cuda,mpi --stats=true --force-overwrite=true \
    -o ../nsys_mpirank2_nochk \
    mpirun -np 2 /home/dell/workspace/astr_gpu/build_gpu_probe/bin/astr run datin/input.tgv
)

nsys stats --force-export true \
  --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$OUT_DIR/nsys_mem" \
  "$OUT_DIR/nsys_mpirank2_nochk.nsys-rep"
```

The post-git audit reported `Device-to-Host` count `116`, total `380.607 MB`, max `4.793 MB`, and `Host-to-Device` count `188`, total `867.120 MB`, max `104.333 MB`. For this `2x1x1` case, one local interior scalar field is about `8.39 MB`; therefore the observed maximum D2H is halo-buffer scale, not full-field scale. The largest H2D events are initial resident array setup and are not evidence of per-kernel or per-step full-field D2H. ASTR may still create the initial CPU-owned `outdat/flowfield.h5`; this audit is scoped to checkpoint-disabled GPU compute-loop transfers, not to removing every file output path.

## Multi-Rank Matrix Driver

Run the core multi-rank stats and field matrix:

```bash
OUT_DIR=tests/gpu_validation/out/tgv_mpirank_matrix \
  tests/gpu_validation/run_tgv_mpirank_matrix.sh
```

By default this covers `2x1x1`, `1x2x1`, `1x1x2`, `2x2x1`, `2x1x2`, `1x2x2`, and `2x2x2` for both native statistics and one-step field comparison. Higher-rank oversubscription smoke tests are opt-in:

```bash
RUN_SMOKE=t OUT_DIR=tests/gpu_validation/out/tgv_mpirank_matrix_smoke \
  tests/gpu_validation/run_tgv_mpirank_matrix.sh
```

Override `STATS_MATRIX`, `FIELD_MATRIX`, or `SMOKE_MATRIX` with entries of the form `NP:i,j,k` to run a narrower set.

## GPU Gradcal Dvel Compare

Validate the first-stage GPU `gradcal` path for TGV velocity gradients:

```bash
MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_gradcal_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

python3 tests/gpu_validation/compare_gpu_gradcal.py \
  --gpu tests/gpu_validation/out/tgv_gpu_gradcal_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_gradcal_643e_diff_filter_10/gpu_gradcal_compare.txt \
  --atol 1e-10 --rtol 1e-10
```

The GPU run used to write `gpu_gradcal_dvel_compare.dat` after CPU `gradcal()` and GPU `gradcal_dvel_kernel` both ran on the same refreshed state. That bridge validation passed and remains useful as a regression fixture. The current GPU-resident path no longer runs CPU `gradcal()` every step; `dvel_d` is computed on device for GPU statistics.
