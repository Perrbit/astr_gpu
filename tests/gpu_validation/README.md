# GPU Validation

This directory contains validation drivers for the current ASTR CUDA Fortran port.

Current validated scope:

- Taylor-Green Vortex, 3D extruded `2dvort`, and generated-velocity HIT within the explicit periodic non-reacting capability gate
- first x/y/z-direction non-periodic zero-extrapolation boundary slices, with the other directions periodic and filter/diffusion support on one MPI rank
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

The optional `NP=8` `2x2x2` statistics-only smoke can be run with:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_mpirank_matrix_np8_stats_smoke \
  MATRIX='8:2,2,2' RUN_FIELD=f FEQCHKPT_FIELD=99 \
  tests/gpu_validation/run_2dvort_mpirank_matrix.sh
```

This is a halo-routing smoke under two-GPU oversubscription. It does not compare HDF5 fields and should not be used as performance evidence.

## Phase A HIT Validation

Validate the generated-velocity HIT case. The driver starts from the TGV input template, rewrites `flowtype=hit`, generates `datin/velocity.h5`, and compares CPU/GPU `flowstate.dat` statistics:

```bash
OUT_DIR=tests/gpu_validation/out/hit_phasea_np1_20steps \
  MAXSTEP=20 FEQCHKPT=99 GRID=64,64,64 COMPARE_FIELD=f \
  tests/gpu_validation/run_hit_phasea_compare.sh
```

The generated velocity is an ABC-style periodic field. `hitini` must refresh halos before its divergence diagnostic because `grad()` consumes halo data; with the current code the log reports divergence `avg/min/max = 0`.

Validate HIT x-slab multi-rank HDF5 hyperslab reading and halo exchange:

```bash
OUT_DIR=tests/gpu_validation/out/hit_phasea_np2_xslab_5steps \
  MAXSTEP=5 FEQCHKPT=99 GRID=64,64,64 NP=2 TOPOLOGY=2,1,1 COMPARE_FIELD=f \
  tests/gpu_validation/run_hit_phasea_compare.sh
```

Validate HIT combined-direction halo-routing smoke under two-GPU oversubscription:

```bash
OUT_DIR=tests/gpu_validation/out/hit_phasea_np8_2x2x2_5steps \
  MAXSTEP=5 FEQCHKPT=99 GRID=64,64,64 NP=8 TOPOLOGY=2,2,2 COMPARE_FIELD=f \
  tests/gpu_validation/run_hit_phasea_compare.sh
```

Current expected result: all three commands print `status: pass` with `kenergy`, `enstophy`, and `dissipation` differences at about `1e-15` or below. HIT field-output comparison is intentionally disabled by default because HDF5 flowfield output remains a CPU-owned boundary in the current project scope.

## Phase B Zeroextrap Boundary Validation

Validate the first finite-domain boundary slice. The driver starts from the TGV template, sets `lihomo=f,ljhomo=t,lkhomo=t`, uses x-direction `bctype=50,50`, enables diffusion and the explicit filter, and compares CPU/GPU statistics plus `flowfield.h5` output:

```bash
OUT_DIR=tests/gpu_validation/out/xextrap_phaseb_np1_5steps_filter_diffusion \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: both `flowstate.dat` and field comparison print `status: pass` with differences at roundoff scale. This validation depends on the CPU `parallelini` fix that sets single-rank non-homogeneous active ranges to interior nodes, the GPU x-zeroextrap boundary kernel, x-zeroextrap RHS active-range guards, x-physical GPU `gradcal`, x-physical diffusion flux/RHS support, and CPU-compatible explicit-filter primitive timing. In the CPU code, `filterq` updates `q` but does not immediately refresh all primitive fields; the GPU x-zeroextrap filtered path therefore synchronizes primitive fields before filtering each RK substep, then preserves interior primitive fields after filtering while refreshing only boundary/halo primitive slices required by CPU `qswap`. `LFILTER=f` and `DIFFTERM=f` remain useful regression variants for isolating convection/RK from filter and diffusion effects.

The same driver can validate the y/z zeroextrap slices by setting `ZERO_AXIS=y` or `ZERO_AXIS=z`. These slices validate boundary application, active ranges, y/z physical `gradcal`, y/z physical convection RHS, y/z physical diffusion flux/RHS, explicit filter ping-pong halo semantics, CPU-compatible filtered primitive timing, and single-rank halo routing.

```bash
OUT_DIR=tests/gpu_validation/out/yzero_phaseb_filter_5steps \
  ZERO_AXIS=y MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/zzero_phaseb_filter_5steps \
  ZERO_AXIS=z MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: both y and z commands print `status: pass` for `flowstate.dat` and field comparison. Single-rank finite-domain slices remain useful isolation tests for the physical y/z boundary kernels and filtered primitive timing.

Validate the current multi-rank Phase B boundary matrix:

```bash
OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix \
  tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh
```

The default matrix validates all currently enabled multi-rank Phase B routes, including physical-direction decomposition for x/y/z zeroextrap. Physical-direction convection/diffusion stencils use physical one-sided templates only on true `MPI_PROC_NULL` faces and use halo-backed sixth-order central stencils on MPI internal interfaces.

```text
x:2:1,2,1 x:2:1,1,2 x:2:2,1,1
y:2:2,1,1 y:2:1,1,2 y:2:1,2,1
z:2:2,1,1 z:2:1,2,1 z:2:1,1,2
x:4:1,2,2 x:4:2,2,1 x:4:2,1,2
y:4:2,1,2 y:4:2,2,1 y:4:1,2,2
z:4:2,2,1 z:4:2,1,2 z:4:1,2,2
```

Current expected result: every matrix entry prints `status: pass` for `flowstate.dat` and field comparison at `STATS_ATOL=1e-9` and `FIELD_ATOL=1e-8`. Physical-direction decomposition is enabled for x-zero, y-zero, and z-zero in the default NP=2/NP=4 matrix.

Run the longer Phase B stability checks with:

```bash
OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix_5steps \
  MAXSTEP=5 FEQCHKPT=5 \
  tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix_20steps_stats \
  MAXSTEP=20 FEQCHKPT=99 RUN_FIELD=f \
  tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh
```

Current expected result: the 5-step matrix passes statistics and field comparison; the 20-step matrix passes statistics-only. The latest 5-step field run had max reconstructed `q5` error `7.3896444519050419e-13`; the latest 20-step statistics-only run had max differences `kenergy=4.4231285301066237e-13`, `enstophy=3.54605234065275e-13`, and `dissipation=4.427014310692812e-14`.

## Phase C Symmetry Boundary Slices

Validate CPU-compatible Cartesian `bctype=60,60` symmetry with the TGV template:

```bash
OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_x_5steps \
  BC_KIND=symmetry ZERO_AXIS=x NP=1 TOPOLOGY=1,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_y_5steps \
  BC_KIND=symmetry ZERO_AXIS=y NP=1 TOPOLOGY=1,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_z_5steps \
  BC_KIND=symmetry ZERO_AXIS=z NP=1 TOPOLOGY=1,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate the multi-rank Phase C matrix:

```bash
OUT_DIR=tests/gpu_validation/out/symmetry_phasec_mpirank_matrix \
  MAXSTEP=1 FEQCHKPT=1 RUN_FIELD=t \
  tests/gpu_validation/run_symmetry_phasec_mpirank_matrix.sh
```

The current expected result is 18/18 matrix entries passing across x/y/z symmetry and NP=2/NP=4 topologies. Direct five-step physical-direction checks also pass:

```bash
OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np2_yphysical_5steps \
  BC_KIND=symmetry ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np2_zphysical_5steps \
  BC_KIND=symmetry ZERO_AXIS=z NP=2 TOPOLOGY=1,1,2 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

This validates the CPU-compatible Cartesian symmetry path only. It does not imply general curvilinear symmetry normals, wall boundaries, farfield, NSCBC, immersed boundary, species, turbulence, chemistry, compact schemes, or GPU HDF5/checkpoint writing.

## Phase D Artificial TGV `bctype=41` Wall Slices

Use the TGV input template as an artificial boundary-path probe for isothermal no-slip walls in one Cartesian direction. This is not a physical TGV wall validation. It deliberately breaks periodicity in one Cartesian direction, sets the two faces in that direction to `bctype=41` with fixed wall temperature, keeps the other two directions periodic, and compares CPU/GPU boundary/filter/diffusion behavior.

```bash
WALL_AXIS=x MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=y MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=z MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh
```

Validate physical-direction MPI decomposition with:

```bash
WALL_AXIS=x NP=2 TOPOLOGY=2,1,1 MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=y NP=2 TOPOLOGY=1,2,1 MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=z NP=2 TOPOLOGY=1,1,2 MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh
```

Current expected result: all six commands pass `flowstate.dat` at `1e-9` and `flowfield.h5` at `1e-8`. The latest single-rank 5-step field maxima were reconstructed `q5=5.68e-13` for x, `6.25e-13` for y, and `4.55e-13` for z. The latest NP=2 physical-direction 5-step field maxima were reconstructed `q5=5.12e-13` for x, `5.68e-13` for y, and `5.68e-13` for z.

Validate the reusable wall41 MPI matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/wall41_phased_mpirank_matrix_np2_np4_1step \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=t \
  tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh
```

The default matrix covers 18 entries across x/y/z wall41 and NP=2/NP=4 topologies, including physical-direction and transverse decompositions. Current expected result: `matrix_summary.txt` has 18 pass lines, and each entry passes `flowstate.dat` at `1e-9` and `flowfield.h5` at `1e-8`.

Run the current NP=8 oversubscription smoke with:

```bash
OUT_DIR=tests/gpu_validation/out/wall41_phased_mpirank_matrix_np8_1step \
  MATRIX='x:8:2,2,2 y:8:2,2,2 z:8:2,2,2' \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=t \
  tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/wall41_phased_mpirank_matrix_np8_5step_stats \
  MATRIX='x:8:2,2,2 y:8:2,2,2 z:8:2,2,2' \
  MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=f \
  tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh
```

The summary from the latest NP=8 runs has three pass lines. The one-step field smoke passes for all three wall axes, and the five-step statistics-only smoke passes with max statistic differences below `5e-14`. This is two-GPU oversubscription correctness evidence, not performance scaling evidence.

The x/y/z statistics are not expected to match each other because the artificial wall direction changes which TGV structures are cut by no-slip/isothermal faces. This validates only the current explicit, Cartesian, no-species, no-turbulence, no-wall-blowing wall41 slices; it does not imply general wall-boundary physics, wall models, curvilinear wall normals, compact schemes, species, chemistry, or GPU HDF5/checkpoint writing.

## Phase D Channel `bctype=41` Wall Slice

Validate the first wall-bounded channel slice from `examples/Channel/datin/input.chl`. The current GPU support is intentionally narrow: y-direction `bctype=41,41` isothermal no-slip walls, x/z periodic, `numq=5`, no species, no turbulence model, explicit `643e,643e`, RK3, explicit filter/diffusion only.

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_np1_filter_diff_1step \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-8 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: one-step `flowstate.dat` and `flowfield.h5` comparisons print `status: pass`. This validates the GPU `bctype=41` y-wall kernel, y-physical gradient/diffusion/convection path, explicit filter ping-pong path, channel statistics (`massflux`, `fbcx`, `forcex`, `wrms`), and the GPU channel body-force source with variables resident on the device except for scalar reductions and CPU-owned output.

Run the short feedback check with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_np1_filter_diff_2steps_green \
  MAXSTEP=2 FEQCHKPT=2 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: statistics and field comparison pass at `1e-8`. The channel source path now reads `jacob_d` directly inside the GPU kernel, matching the rest of the GPU RHS kernels and avoiding the previous device-array lower-bound mismatch.

Run the current single-rank feedback gate with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_feedback_filter_np1_20steps_after_source_fix \
  MAXSTEP=20 FEQCHKPT=20 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-8 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: statistics and field comparison pass at `1e-8`; the most recent run had final differences near roundoff (`massflux=2.20e-13`, `forcex=3.91e-16`, reconstructed `q5=2.84e-13`).

For longer source/wall/diffusion validation without the channel feedback loop, use a fixed body force:

```bash
OUT_DIR=tests/gpu_validation/out/channel_fixedforce_nofilter_np1_100steps_dt5e4_after_source_fix \
  MAXSTEP=100 FEQCHKPT=100 GRID=32,32,32 DELTAT=5.d-4 \
  LFILTER=f DIFFTERM=t CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  COMPARE_STATS=t COMPARE_FIELD=t STATS_ATOL=1e-8 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: statistics and field comparison pass at `1e-8`; the latest 100-step run had reconstructed `q5=6.04e-14` and `u1=4.00e-15`. Do not use `LFILTER=t` beyond the current 20-step gate as a long-step convergence test on this small channel setup: the CPU baseline itself crashes around 23 filtered steps at `DELTAT=5.d-4`, and around 20 filtered steps for smaller `DELTAT`. That is a CPU baseline/filter-frequency stability limit, not a GPU equivalence failure.

For larger multi-rank long-step statistics-only validation, use the original channel grid size and timestep with fixed force and field output disabled:

```bash
OUT_DIR=tests/gpu_validation/out/channel_long_np2_128_100steps_stats \
  MATRIX='2:2,1,1 2:1,2,1 2:1,1,2' \
  GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 FEQCHKPT=9999 \
  LFILTER=f DIFFTERM=t RUN_FIELD=f \
  CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  STATS_ATOL=1e-8 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_long_np4_128_100steps_stats \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' \
  GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 FEQCHKPT=9999 \
  LFILTER=f DIFFTERM=t RUN_FIELD=f \
  CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  STATS_ATOL=1e-8 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_long_np8_128_100steps_stats \
  MATRIX='8:2,2,2' \
  GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 FEQCHKPT=9999 \
  LFILTER=f DIFFTERM=t RUN_FIELD=f \
  CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  STATS_ATOL=1e-8 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

Current expected result: NP=2 `2x1x1`, `1x2x1`, `1x1x2`; NP=4 `2x2x1`, `2x1x2`, `1x2x2`; and NP=8 `2x2x2` all print `status: pass` in `flowstate_compare.txt`. The latest run had maximum observed `massflux=4.998e-13`, `wrms=4.979e-15`, and `forcex=0` differences. This is a long-step CPU/GPU statistics equivalence gate under two-GPU oversubscription, not a production scaling result.

Validate the current `NP=2` channel slab matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np2_1step \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The default matrix covers `2x1x1`, `1x2x1`, and `1x1x2`. The `1x2x1` entry is the key physical-direction decomposition check: the lower and upper y-wall kernels must apply only on true physical faces, while the internal y interface uses halo-backed central stencils.

Run the short multi-rank feedback matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np2_2steps \
  MAXSTEP=2 FEQCHKPT=2 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

Current expected result: all three entries pass statistics at `1e-8` and field comparison at `1e-6`.

Validate the `NP=4` combined-direction channel matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np4_1step \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np4_2steps \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' \
  MAXSTEP=2 FEQCHKPT=2 STATS_ATOL=2e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The `MAXSTEP=1` matrix is strict at `STATS_ATOL=1e-8` and `FIELD_ATOL=1e-8`. The two-step matrix uses `STATS_ATOL=2e-8` because the channel feedback loop produces `~1.2e-8` mass-flux/force tail differences under combined decompositions.

Validate the `NP=8` `2x2x2` channel smoke with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np8_1step \
  MATRIX='8:2,2,2' \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np8_2steps \
  MATRIX='8:2,2,2' \
  MAXSTEP=2 FEQCHKPT=2 STATS_ATOL=2e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The `NP=8` run is a correctness smoke under two-GPU oversubscription. It is not performance evidence and should not be used to infer production scaling.

Validate the `NP=27` `3x3x3` fully interior-rank channel smoke with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_np27_3x3x3_1step \
  MATRIX='27:3,3,3' \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_np27_3x3x3_2steps \
  MATRIX='27:3,3,3' \
  MAXSTEP=2 FEQCHKPT=2 STATS_ATOL=3e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The `3x3x3` topology creates ranks fully wrapped by MPI neighbors in x/y/z, including y-interior ranks that do not touch either wall. This is a high-oversubscription correctness smoke on the current two-GPU machine, not performance evidence.

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
