# GPU Validation

This directory contains validation drivers for the current ASTR CUDA Fortran port.

Current validated scope:

- Taylor-Green Vortex, 3D extruded `2dvort`, generated-velocity HIT, forced 3-D LDC slices, and forced 3-D RTI explicit validation variants
- x/y/z zero-extrapolation and symmetry slices
- CPU-compatible Cartesian wall-family slices: `bctype=41` x/y/z, `bctype=42` x/y, `bctype=411` y, and `bctype=421` y
- channel `bctype=41` y-wall source/statistics path and RTI y-fixed source path
- single-rank and multi-rank MPI decompositions, including correctness smoke tests under two-GPU oversubscription
- one physical GPU, two physical GPUs, and two-GPU oversubscription correctness smoke tests
- q(1:5) only
- explicit sixth-order central difference
- explicit tenth-order central filter
- current implemented GPU convection paths are explicit central `643e` and controlled single-rank explicit upwind `543e` with `recon_schem=-1` (first order), `1` (WENO7), or `3` (MP7)
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

For reproducible wall-clock benchmark reporting, use:

```bash
OUT_DIR=tests/gpu_validation/out/channel_128_benchmark \
  tests/gpu_validation/run_channel_128_benchmark.sh
```

The benchmark driver runs the `128^3`, `DELTAT=7.5d-4`, `MAXSTEP=100`, fixed-force channel case by default. It records explicit CPU/GPU wall times in `benchmark_times.tsv`, writes speedups relative to the NP=1 CPU baseline in `benchmark_summary.md`, and keeps per-topology `flowstate_compare.txt` reports when `RUN_CPU_FOR_ALL=t`.

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

## Phase E Adiabatic Wall `bctype=42` Slices

Validate CPU-compatible Cartesian `bctype=42,42` no-slip adiabatic walls for x/y directions only. The CPU implementation `noslip_adibatic` currently implements `ndir=1..4`; z-direction wall42 is therefore outside the CPU-supported scope and is intentionally rejected by the validation driver.

```bash
OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_x_5steps \
  BC_KIND=adiabaticwall ZERO_AXIS=x \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_y_5steps \
  BC_KIND=adiabaticwall ZERO_AXIS=y \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate physical-direction MPI gating with:

```bash
OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_x_np2_physical \
  BC_KIND=adiabaticwall ZERO_AXIS=x NP=2 TOPOLOGY=2,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_y_np2_physical \
  BC_KIND=adiabaticwall ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: all four commands print `status: pass` for statistics and field comparison; reconstructed `q5` errors stay below `6.3e-13`. `BC_KIND=adiabaticwall ZERO_AXIS=z` exits before running because CPU `noslip_adibatic` has no `ndir=5/6` implementation.

## Phase F Slip-Isothermal Wall `bctype=411` Slice

Validate CPU-compatible Cartesian `bctype=411,411` slip-nonslip isothermal wall behavior for y direction only. The CPU implementation `slipisotwall` implements `ndir=3/4`; x/z directions are therefore outside the CPU-supported scope and are intentionally rejected by the validation driver. The lower wall uses `x <= xslip` for the slip segment and the upper wall follows the CPU no-slip isothermal branch. Wall blowing/suction, species, and turbulence models remain outside the GPU validation scope.

```bash
OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np1_5steps \
  BC_KIND=slipisotwall ZERO_AXIS=y \
  XSLIP=3.141592653589793d0 WALL_TEMP=273.15d0 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate physical-face MPI gating and transverse decompositions with:

```bash
OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np2_physical \
  BC_KIND=slipisotwall ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np2_xslab \
  BC_KIND=slipisotwall ZERO_AXIS=y NP=2 TOPOLOGY=2,1,1 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np2_zslab \
  BC_KIND=slipisotwall ZERO_AXIS=y NP=2 TOPOLOGY=1,1,2 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: all commands print `status: pass` for statistics and field comparison; reconstructed `q5` field errors stay below `6e-13`. `BC_KIND=slipisotwall ZERO_AXIS=x/z` exits before running because CPU `slipisotwall` has no `ndir=1/2/5/6` implementation.

## Phase G Slip-Adiabatic Wall `bctype=421` Slice

Validate CPU-compatible Cartesian `bctype=421,421` slip-nonslip adiabatic wall behavior for y direction only. The CPU implementation `slipadibwall` implements `ndir=3/4`; x/z directions are therefore outside the CPU-supported scope and are intentionally rejected by the validation driver. Although the input still reads `xslip`, the current CPU `slipadibwall` has the `xslip` split commented out, so the GPU validation follows the active CPU formula rather than the boundary-condition name.

```bash
OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np1_5steps \
  BC_KIND=slipadibwall ZERO_AXIS=y \
  XSLIP=3.141592653589793d0 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate physical-face MPI gating and transverse decompositions with:

```bash
OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np2_physical \
  BC_KIND=slipadibwall ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np2_xslab \
  BC_KIND=slipadibwall ZERO_AXIS=y NP=2 TOPOLOGY=2,1,1 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np2_zslab \
  BC_KIND=slipadibwall ZERO_AXIS=y NP=2 TOPOLOGY=1,1,2 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: all commands print `status: pass` for statistics and field comparison; reconstructed `q5` field errors stay below `6e-13`. `BC_KIND=slipadibwall ZERO_AXIS=x/z` exits before running because CPU `slipadibwall` has no `ndir=1/2/5/6` implementation.

## Phase H Wall-Family Regression Matrix

Use Phase H as the reusable wall-family regression gate after changing boundary, halo, filter, diffusion, or capability-gate code. It combines the currently supported Cartesian wall slices and also checks unsupported CPU-scope directions are still rejected.

```bash
OUT_DIR=tests/gpu_validation/out/wall_family_phaseh_matrix \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=t \
  tests/gpu_validation/run_wall_family_phaseh_matrix.sh
```

The default supported matrix covers:

| Boundary | Supported axes in Phase H | MPI coverage |
|---|---|---|
| `bctype=41` isothermal no-slip | x/y/z | physical-direction NP=2 slab for each axis |
| `bctype=42` adiabatic no-slip | x/y | physical-direction NP=2 slab for each supported axis |
| `bctype=411` slip-isothermal | y | physical y NP=2 plus transverse x/z NP=2 slabs |
| `bctype=421` slip-adiabatic | y | physical y NP=2 plus transverse x/z NP=2 slabs |

The default reject matrix checks `42-z`, `411-x`, `411-z`, `421-x`, and `421-z`. These cases should exit before running CPU/GPU solvers because the corresponding CPU boundary routines do not implement those directions. A rejected case passing is a validation failure, not progress.

Useful controls:

```bash
# Parse and list the default matrix without launching ASTR.
DRY_RUN=t tests/gpu_validation/run_wall_family_phaseh_matrix.sh

# Run a focused subset.
MATRIX='slipisotwall:y:2:1,2,1 slipadibwall:y:2:1,2,1' \
  tests/gpu_validation/run_wall_family_phaseh_matrix.sh

# Check only unsupported CPU-scope directions.
RUN_SUPPORTED=f RUN_REJECTS=t \
  tests/gpu_validation/run_wall_family_phaseh_matrix.sh

# Disable field comparison for a faster statistics-only smoke.
RUN_FIELD=f FEQCHKPT=99 tests/gpu_validation/run_wall_family_phaseh_matrix.sh
```

Current expected result: the full default matrix prints 11 supported pass lines and 5 reject pass lines in `matrix_summary.txt`. Each supported entry should print `status: pass` for both `flowstate_compare.txt` and `flowfield_compare.txt`. The current full run is `tests/gpu_validation/out/wall_family_phaseh_matrix_full`; it passed all supported entries and kept `42-z`, `411-x/z`, and `421-x/z` rejected with exit status 2. Phase H does not expand the physics contract beyond the already validated Cartesian slices; it prevents regressions and accidental scope creep.

## Phase I Lid-Driven-Cavity Gate

Use the LDC Phase I gate to keep the next case expansion explicit. The original `examples/Lid-Driven-Cavity` case is a 2D cavity with x/y non-periodic directions, isothermal walls, and a top-lid `bctype=0` UDF boundary. It is intentionally outside the current GPU contract.

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phasei_gate \
  tests/gpu_validation/run_ldcavity_phasei_gate.sh
```

Current expected result: the GPU run exits nonzero and `gate_summary.txt` reports `pass gpu-reject`. The current baseline rejects the original 2D case with `GPU first-stage supports 3D cases only`.

Useful probes:

```bash
# Confirm the CPU explicit LDC case still completes before using it as a future oracle.
OUT_DIR=tests/gpu_validation/out/ldcavity_phasei_gate_cpu_probe \
  RUN_CPU=t \
  tests/gpu_validation/run_ldcavity_phasei_gate.sh

# Force a 3D explicit probe to expose the next GPU gate after the 2D blocker.
OUT_DIR=tests/gpu_validation/out/ldcavity_phasei_gate_3d \
  GRID=32,32,32 \
  tests/gpu_validation/run_ldcavity_phasei_gate.sh
```

Current status: CPU `RUN_CPU=t` completes a one-step explicit LDC probe without `ieee_invalid`. The warning was traced to `gridcube(1.d0,1.d0,0.d0)` in Release/O2 builds: the old loop contained a non-taken `lz/real(ka,8)` expression when `ka==0`, which NVHPC could still evaluate speculatively and raise `0/0` invalid for 2-D zero-thickness grids. `src/gridgeneration.F90` now precomputes guarded `dx/dy/dz` values before the loop. The same file also keeps original 2-D LDC grid generation for `ka==0`, but uses a unit z length for forced 3-D LDC probes with `ka>0`. The latest original-case check is `OUT_DIR=tests/gpu_validation/out/ldcavity_phaseib_original_filter_reject RUN_CPU=f tests/gpu_validation/run_ldcavity_phasei_gate.sh`; the GPU run still rejects the original/default filtered case as expected. The GPU LDC boundary routine mirrors the CPU boundary order: x isothermal no-slip walls first, y lower isothermal no-slip next, and y upper moving lid last. Therefore the top-lid/side-wall corner velocity follows the CPU implementation and is overwritten to `(u,v,w)=(1,0,0)`.

Validate the narrow Phase I-A no-filter/no-diffusion LDC slice with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseia_compare \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: `flowstate_compare.txt` and `flowfield_compare.txt` both print `status: pass`. This script forces a 3-D LDC probe with `GRID=32,32,32`, `LFILTER=f`, `DIFFTERM=f`, and explicit `643e`. The latest 5-step check:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseia_script_5steps \
  MAXSTEP=5 FEQCHKPT=5 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

passed with max `flowstate.dat` difference `maxq5=2.8620661396416835e-11` and reconstructed field `q5` `L_inf=2.8421709430404007e-14`. This validates x/y physical boundary ownership plus z periodic halo for convection without diffusion.

Validate the Phase I-B no-filter diffusive LDC slice with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseib_diffusion_5steps \
  DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: `flowstate_compare.txt` and `flowfield_compare.txt` both print `status: pass`. The latest 5-step diffusive single-rank check passed with final `flowstate.dat` difference `maxq5=3.9619862945983186e-11` and reconstructed field `q5` `L_inf=8.5265128291212022e-14`.

Validate the current LDC multi-rank x/y physical halo route with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseib_diffusion_np4_221 \
  DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 NP=4 TOPOLOGY=2,2,1 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: the run passes statistics and HDF5 field comparison. This covers x/y internal MPI halos plus true physical x/y faces, while z remains periodic. Filtered LDC and the original 2-D LDC case remain outside the supported GPU contract.

Validate the current Phase I-C filter isolation with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_filter_only_1step_check \
  LFILTER=t DIFFTERM=f MAXSTEP=1 FEQCHKPT=1 \
  COMPARE_STATS=f COMPARE_FIELD=t \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: strict field comparison prints `status: pass`. The latest run had reconstructed `q5` `L_inf=1.9895196601282805e-13`, so the LDC multi-axis explicit filter path is correct in isolation.

Validate the combined `LFILTER=t,DIFFTERM=t` Phase I-C gate with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_filter_diff_1step_fixed_clean \
  LFILTER=t DIFFTERM=t MAXSTEP=1 FEQCHKPT=1 \
  COMPARE_STATS=f COMPARE_FIELD=t FIELD_ATOL=1e-10 FIELD_RTOL=1e-10 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: strict field comparison prints `status: pass`; the latest reconstructed `q5` maximum was `2.2737367544323206e-13`. The previous failure at `(i,j,k)=(0,31,1)` was caused by GPU x/y physical diffusion RHS kernels applying all three `is:ie/js:je/ks:ke` restrictions at once. CPU `diffrsdcal6` applies direction-specific RHS ranges, so GPU `xyphysical` diffusion RHS and `x_xphysical` diffusion RHS now follow the same ownership rule.

Use the 5-step strict stats plus field gate for the short regression:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_filter_diff_5step_fixed_clean \
  LFILTER=t DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 \
  COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-10 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: statistics and field comparisons both print `status: pass`; the latest final `flowstate.dat maxq5` difference was `8.1854523159563541e-12`, and reconstructed `q5 L_inf` was `5.6843418860808015e-13`. Do not use the current 20-step small-grid LDC run as a correctness gate: CPU/GPU diagnostics both become abnormal or crash-prone by roughly steps 16-20, so that setup is an oracle-quality problem rather than a GPU-only failure.

## Phase J Rayleigh-Taylor Explicit Variant

The canonical Phase J regression driver is:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_matrix_driver_current \
  tests/gpu_validation/run_rti_phasej_matrix.sh
```

Current expected result: `matrix_summary.txt` contains 13 `pass` lines. The default matrix covers single-rank `LFILTER=f/t,DIFFTERM=f/t`, `NP=2` x/y/z slabs, `NP=4` combined slabs, `NP=8 TOPOLOGY=2,2,2`, and 20-step single-rank plus `NP=4 TOPOLOGY=2,2,1`.

Validate the RTI explicit validation variant with:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_5step \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

This driver starts from `examples/Rayleigh–Taylor-Instability/datin/input.rti`, but intentionally rewrites it to a forced 3-D explicit oracle: `GRID=32,64,32`, `643e`, `rk3`, `numq=5`, no species, no turbulence, x/z periodic, y fixed `bctype=31`, and RTI gravity source. It is not a validation of the original compact 2-D RTI input.

Current expected result: statistics and field comparisons both print `status: pass`; the latest single-rank 5-step `LFILTER=t,DIFFTERM=t` run had final `flowstate.dat maxq5=3.5704772471945034e-13` and reconstructed `q5 L_inf=7.9936057773011271e-15`. The single-rank 1-step and 5-step matrix also passed for all `LFILTER=f/t` and `DIFFTERM=f/t` combinations.

Validate the current two-rank RTI halo slices with:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_np2_121 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=2 TOPOLOGY=1,2,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_np2_211 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=2 TOPOLOGY=2,1,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_np2_112 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=2 TOPOLOGY=1,1,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

Current expected result: all three NP=2 entries print `status: pass` for both reports. The `1,2,1` case covers true y fixed physical faces plus internal y halo, while `2,1,1` and `1,1,2` cover transverse x/z periodic MPI halos. The latest maximum observed reconstructed `q5 L_inf` was `8.8817841970012523e-15`, and the maximum observed final `flowstate.dat maxq5` difference was `4.4408920985006262e-13`.

The current extended RTI multi-rank matrix also passes:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_221 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=2,2,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_212 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=2,1,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_122 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=1,2,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np8_222 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=8 TOPOLOGY=2,2,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

Current expected result: all four entries print `status: pass` for both reports. The latest 5-step extended matrix had maximum reconstructed `q5 L_inf=1.0658141036401503e-14` and maximum `flowstate.dat maxq5=4.6984638402136625e-13`.

Use the 20-step RTI checks as the current stronger short-run regression:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/sr20 \
  MAXSTEP=20 FEQCHKPT=20 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=1 TOPOLOGY=1,1,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_221_20 \
  MAXSTEP=20 FEQCHKPT=20 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=2,2,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

Current expected result: both 20-step entries print `status: pass`. The latest single-rank run had reconstructed `q5 L_inf=1.0658141036401503e-14` and `flowstate.dat maxq5=4.9382720135326963e-13`; the latest `NP=4 TOPOLOGY=2,2,1` run had reconstructed `q5 L_inf=1.3322676295501878e-14` and `flowstate.dat maxq5=4.7073456244106637e-13`.

## Phase K Source Capability Regression

The canonical Phase K source regression driver is:

```bash
OUT_DIR=tests/gpu_validation/out/source_phasek_matrix_current \
  tests/gpu_validation/run_source_phasek_matrix.sh
```

Current expected result: `matrix_summary.txt` contains three `pass` lines:

- `tgv_nosource`: verifies the source dispatcher leaves no-source cases alone.
- `channel_source`: verifies the channel source path and `lihomo`-gated source dispatch.
- `rti_source`: verifies the RTI source path with `NP=2 TOPOLOGY=1,2,1`, including y fixed boundary plus internal y halo.

After changing the source capability layer, also rerun the full RTI matrix:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_matrix_after_capability \
  tests/gpu_validation/run_rti_phasej_matrix.sh
```

Current expected result: all 13 RTI entries pass. The source capability table currently covers only `GPU_SOURCE_NONE`, `GPU_SOURCE_CHANNEL`, and `GPU_SOURCE_RTI`; it is not chemistry, turbulence, or generic UDF source support.

## Phase S0-A Shock-Format Readiness

S0-A1 is the first completed shock-format plumbing gate. It uses a forced 3-D extruded Sod case to open the explicit upwind path without adding open boundaries, high-speed wall coupling, characteristic decomposition, sensors, diffusion, or filtering.

Validated S0-A1 contract:

- `flowtype='sod'` through the `sodini` initialization path
- controlled validation input, not `examples/sod/datin/input.sod` as-is
- `GRID=200,8,8`
- `deltat=5.d-4`, `maxstep=20`
- periodic x/y/z boundaries, with the run short enough to avoid wave interaction with the x-periodic boundary
- `conschm='543e'`, `difschm='643e'`
- `recon_schem=-1`, `lchardecomp=.false.`
- `diffterm=f`, `lfilter=f`
- CPU/GPU statistics tolerances `STATS_ATOL=1e-10`, `STATS_RTOL=1e-10`
- field tolerances `FIELD_ATOL=1e-10`, `FIELD_RTOL=1e-10`
- at minimum, report max differences for `q(:,:,:,1:5)`

Run the canonical gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a1_gate_200x8x8_20 \
  tests/gpu_validation/run_sod_phase_s0a1_compare.sh
```

Current status: pass. The controlled `sod` path generates a positive-volume 3-D grid, the GPU capability gate accepts only `NP=1`, and compact upwind plus characteristic decomposition remain rejected. The recorded 20-step run passed statistics and field checks at `1e-10`. Maximum statistic differences were `maxq2=4.3021142204224816e-15` and `maxq5=1.3322676295501878e-15`; maximum conservative-field differences were `q1=2.2204460492503131e-16`, `q2=8.8446089343311681e-17`, `q3=3.1150835073543049e-18`, `q4=3.1456319031046116e-18`, and `q5=1.7763568394002505e-15`.

This result validates first-order Steger-Warming flux-splitting plumbing only. It is not a high-order shock-accuracy claim.

S0-A2 adds the CPU-semantic WENO reconstruction with `recon_schem=1`, `lchardecomp=.false.`, and the same periodic Sod oracle. In the current CPU implementation this selection uses WENO7 on periodic interior interfaces; WENO5 is only a physical-boundary fallback inside `recons_exp`.

Run the canonical S0-A2 gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a2_gate_200x8x8_20 \
  tests/gpu_validation/run_sod_phase_s0a2_compare.sh
```

Current S0-A2 status: pass. The recorded 20-step run passed statistics and field checks at `1e-10`. Maximum statistic differences were `maxq1=4.7384318691001681e-13` and `maxq5=4.1477932199995848e-13`; maximum conservative-field differences were `q1=8.8817841970012523e-16`, `q2=6.5503158452884060e-17`, `q3=1.1111687016389874e-17`, `q4=1.1204301546017956e-17`, and `q5=1.3322676295501878e-15`. The GPU path reuses one haloed scalar `flux_work_d` across directions and conservative components. This is a porting-equivalence result; Sod exact-solution error and discontinuity resolution remain a separate accuracy gate.

Run the independent S0-A2 exact-solution gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a2_accuracy \
  tests/gpu_validation/run_sod_phase_s0a2_accuracy.sh
```

The accuracy driver uses `GRID=800,5,5`, `deltat=5.d-4`, and `maxstep=400`, giving `t=0.2` and `dx=0.0125`. Every 3-D dimension must be at least `hm=5`; the comparison driver rejects smaller dimensions before launching ASTR because the CPU program otherwise only warns and can continue into invalid halo accesses. The exact solver uses the standard ideal-gas Sod states, extracts the x profile from ASTR HDF5 output without assuming h5py axis order, and evaluates only the central Riemann problem so the periodic-boundary discontinuity is excluded.

The gate reports normalized `L1`, `L2`, and `Linf` errors for `ro`, `u1`, `p`, and `q1:q5` away from wave fronts; normalized primitive-variable over/undershoot; contact and shock 10%-90% density thickness; and 50% crossing-position error. The finite limits are `1e-3`, `6e-3`, `6e-2`, `2e-2`, 4 cells, 3 cells, and 1 cell, respectively. A missing 10%, 50%, or 90% density crossing is reported as an unresolved transition and fails even when numeric thresholds are disabled.

Current exact-solution status: pass for both CPU and GPU. The recorded calibration under `tests/gpu_validation/out/sod_s0a2_accuracy_calibration_800x5x5_400` had maximum smooth errors `L1=7.1864e-4`, `L2=4.0087e-3`, and `Linf=4.6338e-2`; maximum normalized bound violation `1.3889e-2`; contact thickness `2.9780` cells; shock thickness `2.1989` cells; and maximum position error `0.7527` cells. CPU/GPU reconstructed-field `Linf` was at most `q5=8.4377e-15`. Recorded step-400 times were `74.907 s` for NP=1 CPU and `8.102 s` for NP=1 GPU; these timings are supporting evidence from the accuracy run, not a controlled performance benchmark.

S0-A3 adds the CPU-semantic MP reconstruction with `recon_schem=3`. The controlled periodic path uses MP7 at every interface; CPU MP5 physical-boundary degradation is outside this gate. WENO7 and MP7 share the generic explicit-reconstruction kernels and the same scalar `flux_work_d`; the reconstruction selector is a kernel argument, and every positive-flux, negative-flux, and directional-RHS kernel remains explicitly synchronized.

Run the S0-A3 equivalence and exact-solution gates with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a3_gate_200x8x8_20 \
  tests/gpu_validation/run_sod_phase_s0a3_compare.sh

OUT_DIR=tests/gpu_validation/out/sod_s0a3_accuracy_800x5x5_400 \
  tests/gpu_validation/run_sod_phase_s0a3_accuracy.sh
```

Current S0-A3 status: pass. The 20-step CPU/GPU comparison passed statistics and fields at `1e-10`; maximum statistic differences were `maxq1=4.6985e-13` and `maxq5=4.6629e-13`, and maximum conservative-field difference was `q5=1.3323e-15`. The 400-step accuracy run passed the unchanged S0-A2 finite thresholds. Maximum smooth errors were `L1=7.2439e-4`, `L2=4.1951e-3`, and `Linf=4.8797e-2`; maximum normalized bound violation was `8.5021e-3`; contact and shock thicknesses were `2.9496` and `1.6939` cells; and maximum position error was `0.7272` cells. MP7 resolved the shock more sharply and reduced overshoot relative to this WENO7 run, while its largest smooth-region errors were slightly higher. Same-run CPU/GPU field `Linf` was at most `q5=1.2879e-14`. Recorded step-400 times were `57.408 s` for NP=1 CPU and `7.277 s` for NP=1 GPU; treat these as supporting run evidence, not a controlled benchmark. A one-step S0-A3 Compute Sanitizer memcheck also completed with `ERROR SUMMARY: 0 errors` under `mpirun -np 1` and the `ob1/self/pt2pt` MPI settings.

S0-A4 isolates the Ducros/pressure-curvature sensor before enabling Roe characteristic reconstruction. The driver forces Shu-Osher onto a positive-volume 3-D periodic grid because the initial Sod velocity is zero and would make the Ducros compression factor an all-zero oracle. It uses `GRID=400,8,8`, `deltat=1.d-4`, `maxstep=1`, `recon_schem=3`, `lchardecomp=f`, `LFILTER=f`, and `DIFFTERM=f`.

Run the S0-A4 single-rank sensor gate with:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_sensor_s0a4_compare \
  tests/gpu_validation/run_shuosher_sensor_s0a4_compare.sh
```

Current S0-A4 status: pass. `compare_shock_sensor.py` checks every raw-sensor value and requires the byte mask to match exactly. The maximum raw-field difference was `1.1102230246251565e-16`; CPU/GPU raw maxima were both `6.9999992499998120e-1`, averages were `4.3855622929106558e-3` and `4.3855622929106566e-3`, and both masks contained 1944 shock nodes with zero mismatches. One-step statistics passed at `1e-10`, and the largest conservative-field difference was `q5=4.4408920985006262e-16`. A three-stage one-step Compute Sanitizer run reported `ERROR SUMMARY: 0 errors`. This is a validation-only capability selected by `ASTR_SHOCK_SENSOR_DUMP`: the mask is not yet consumed by MP7/Roe fluxes.

S0-A5 extends the raw sensor to the existing generic `hm`-layer HaloTransport for `NP=2`, `TOPOLOGY=2,1,1`. The driver sets `ASTR_SHUOSHER_SHOCK_X=0.d0`, shifting the test discontinuity beside the global x-slab interface while preserving the normal Shu-Osher initialization when the variable is absent. Each rank writes its local sensor dump with global offsets; `compare_shock_sensor.py` merges the rank files, verifies overlapping nodes, and compares the global CPU/GPU field and mask.

Run the S0-A5 sensor-halo gate with:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_sensor_s0a5_mpi_compare \
  tests/gpu_validation/run_shuosher_sensor_s0a5_mpi_compare.sh
```

Current S0-A5 status: pass. The merged global shape is `401,9,9`; raw `L_inf=1.1102230246251565e-16`, mask mismatches are zero, and both global masks contain 1944 shock nodes. The one-step multi-rank field comparison passed at `1e-10`, with `q5 L_inf=2.7089441800853820e-14`. The two-rank sanitizer command requires `OMPI_MCA_btl=self,tcp`, `OMPI_MCA_coll='^hcoll,ucc'`, `OMPI_MCA_opal_cuda_support=0`, and `UCX_MEMTYPE_CACHE=n`; with those MPI components disabled, both ranks report `ERROR SUMMARY: 0 errors`. This establishes raw-sensor halo correctness only for `2x1x1`; Roe characteristic reconstruction remains separate.

S0-A6 consumes the CPU-equivalent expanded mask in a single-rank selective Roe characteristic path. The controlled gate uses the same forced-3D periodic Shu-Osher configuration, but sets `lchardecomp=t`. Sensor-active interfaces project the five split Euler fluxes with the local Roe left eigenvectors, reconstruct each characteristic component with MP7, and project the interface flux back with the right eigenvectors. Other interfaces retain physical-space MP7. A reusable five-component `flux_characteristic_work_d` stores one direction at a time before the conservative flux difference.

Run the S0-A6 characteristic gate with:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a6_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a6_compare.sh
```

Current S0-A6 status: pass for `NP=1`. The one-step raw sensor and mask retain `L_inf=1.1102230246251565e-16`, zero mismatches, and 1944 shock nodes. The one-step conservative field maximum is `q5=4.4408920985006262e-16`; the maximum statistic difference is `4.9098503041022923e-12`. A three-step repetition passed with `q5 L_inf=1.4210854715202004e-14`. The prescribed x block remains `(512,1,1)`; a file-local NVHPC `-gpu=maxregcount:128` cap is required because the initial kernel used 255 registers/thread and could not launch at 512 threads. The characteristic kernel now caches each five-component Steger-Warming split for the original seven positive and seven negative MP7 stencil positions before Roe projection. This preserves the operator and adds no global workspace, kernel, or synchronization. On S0-A6, x/y/z characteristic-flux means are `1.051/1.236/1.440 ms` versus `2.026/3.116/3.948 ms` before caching; x-kernel achieved occupancy rises from about 14% to 22%. Compute Sanitizer passed the three-step GPU run with `ERROR SUMMARY: 0 errors`. This capability is limited to periodic Shu-Osher with `ASTR_SHOCK_SENSOR_DUMP`; S0-A7 through S0-A10 separately validate multi-rank characteristic reconstruction.

## Phase S0-B0 X Physical Reconstruction

Run the controlled x-physical explicit-MP7 gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_phase_s0b0_xphysical \
  tests/gpu_validation/run_sod_phase_s0b0_xphysical_compare.sh
```

The gate uses forced-3D Sod, `bctype=50,50,1,1,1,1`, `conschm=543e`, `recon_schem=3`, and no characteristic decomposition, diffusion, or filtering. It is not an open-boundary case. The x interface/RHS kernels preserve CPU active ranges and `npdci` semantics: physical-rank interfaces use two-point average, SUW3, MP5, then MP7; the opposite MPI face retains the exchanged `hm` halo. The 20-step same-topology CPU/GPU gate passes for NP=1 and `NP=2 TOPOLOGY=2,1,1`; NP=2 full-field `q5 L_inf=5.8841820305133297e-15`, and both memcheck ranks report zero errors.

The corresponding selective-Roe physical-face gate uses forced-3D Shu-Osher:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0b0_xphysical \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=3 FEQCHKPT=3 \
  tests/gpu_validation/run_shuosher_characteristic_s0b0_xphysical_compare.sh
```

It keeps `bctype=50,50`, MP7, `lchardecomp=t`, and no diffusion/filter. The
driver checks rank-local raw sensor and mask data for MPI runs, then statistics
and full fields. NP=1 and `NP=2 TOPOLOGY=2,1,1` pass at three steps; NP=2 has
raw `L_inf=1.1102230246251565e-16`, zero mask mismatches, and `q5
L_inf=7.1054273576010019e-15`. This gate also corrects the CPU reference:
`npdci=4` must clamp both physical sides in `ducrossensor`; leaving it untreated
reads undefined physical-face halos. The GPU sensor matches that rule and skips
the periodic x `ssf` fill on a physical x face. It is not inflow/outflow, NSCBC,
or sponge validation.

## Phase S0-B1 Classical Open Boundary

```bash
OUT_DIR=tests/gpu_validation/out/openshock_s0b1 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_openshock_s0b1_compare.sh
```

This gate is a stationary Mach-3 normal shock at x=0, with x-min `11,free` and
x-max `21,10.333333333333333`. The second outlet value is `p2/pinf`; setting
`pout=1` would impose upstream pressure at a subsonic outlet and invalidate the
stationary state. Ten-step NP=1 and `NP=2 TOPOLOGY=2,1,1` comparisons pass with
`q5 L_inf=8.8817841970012523e-16` and `9.9920072216264089e-16`, respectively.
NSCBC, sponge, diffusion, filtering, and characteristic Roe are excluded.

## Phase S0-B2 NSCBC Open Boundary

`run_openshock_s0b2_nscbc_compare.sh` prepares the restricted `12/22`
OpenShock input. The GPU implementation includes characteristic boundary RHS
terms, separate inlet-domain/outlet-plane Mach reductions, and the CPU
sixth-order explicit outlet y-filter. CPU `time_integration_rk` refreshes
halos with a full-rank pre-`boucon` `qswap` when `bctype=22` is present; this
defines the CPU filter input without changing the normal post-boundary swap.
Ten steps pass for NP=1 (`q5 L_inf=8.8817841970012523e-16`) and NP=2
`TOPOLOGY=2,1,1` (`q5 L_inf=9.9920072216264089e-16`). The NP=2 memcheck run,
with the documented `ob1/self,tcp/pt2pt` MPI settings, reports zero errors on
both ranks. Scope remains Cartesian x faces, periodic y/z, no species, no
sponge, no global filter, and physical-space MP7 only.

## Phase S0-B3 x-Max Sponge

```bash
OUT_DIR=tests/gpu_validation/out/openshock_s0b3_sponge_np2 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_openshock_s0b3_sponge_compare.sh
```

This gate retains the S0-B2 `12/22` NSCBC pair and sets only `spg_im=80` on
`GRID=400,8,8`. It is not a reference-state relaxation: after each RK update
the CPU-equivalent device path refreshes all applicable q halos, then smooths the
x-max active layer in conservative variables with the geometric coefficient
field and the six direct neighbors. The temporary values use `qwork_d`; no
additional full-field device allocation or stage-level D2H/H2D copy occurs.
Ten-step NP=1 and NP=2 `2x1x1` comparisons pass with final `q5
L_inf=8.8817841970012523e-15`; the one-step two-rank Compute Sanitizer run
reports zero errors on both ranks. Only x-max `layer` mode is enabled.

## Phase S0-B4 Combined Sponge MPI

```bash
OUT_DIR=tests/gpu_validation/out/openshock_s0b4_sponge_2x2x2 \
  NP=8 TOPOLOGY=2,2,2 GRID=400,16,16 MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_openshock_s0b3_sponge_compare.sh
```

This is the combined x/y/z MPI oracle for the B3 x-max layer. It corrected a
CPU reference error: the six-neighbor stencil requires current halos in every
direction, so `spongefilter_layer` now calls full `dataswap(q)` before each
layer. CUDA Fortran uses the matching q-only three-direction halo exchange.
The ten-step gate passes with final `q5 L_inf=8.8817841970012523e-15`; an
eight-rank one-step Compute Sanitizer run reports zero errors for every rank.
This workstation test oversubscribes two GPUs and is not a scaling measurement.

## Phase S1-A0/A1 Explicit Flat Plate

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a_20steps \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The driver creates an isolated `64x64x8` Cartesian, z-extruded `bl` case and
a nonuniform `datin/inlet.prof`. It uses non-dimensional `Mach=0.3`, explicit
central `643e/643e`, RK3, diffusion, no global filter, x `11,prof/21`, y
`41/51`, and periodic z. The CPU-active `51` branch is an extrapolative
upper-boundary update; its characteristic implementation is commented out in
the CPU source and is not claimed here. GPU uploads the profile once after
`flowinit`, keeps it on device, computes the matching four BL statistics on
device, and compares the final full field plus `massflux`, `fbcx`,
`wallheatflux`, and `wrms`. The 20-step gate passes at `1e-10`; a valid
one-rank Compute Sanitizer run reports zero errors.

S1-A1 uses the same case with explicit MP7 convection (`543e/643e`). Select
it through `CONSCHM=543e`:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_20steps \
  CONSCHM=543e MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The CPU reference now accepts the single-rank double-physical-face encoding
`npdci=npdcj=4`: `recons_exp` applies the same two-point, SUW3, MP5, then MP7
sequence at both ends, and `convrsduwd` uses the physical `[0,dim]` stencil
range for y/z as it already did for x. CUDA uses matching x/y face degradation
and restricts every x/y/z explicit-upwind RHS update to CPU active ranges
`is:ie`, `js:je`, and `ks:ke`. The Steger-Warming split reads local and
exchanged-halo metric entries directly; physical `j=jm` no longer reads the
periodic `jacob(:,0,:)`. The 20-step A1 gate passes with final
`q5 L_inf=1.7763568394002505e-14`, maximum statistic difference
`9.6256336235001072e-14`, and one-rank memcheck reports zero errors. Scope is
Cartesian, profile inflow, explicit MP7, and the active extrapolative `51`
upper boundary; it is not a curvilinear, high-Mach, multi-axis-MPI, or
characteristic-farfield claim.

The same S1-A1 gate also passes in two x slabs. The profile and both y physical
faces remain local to each rank; only x uses the existing solution halo exchange:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np2_20steps \
  CONSCHM=543e NP=2 TOPOLOGY=2,1,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The final parallel field result is `q5 L_inf=1.7763568394002505e-14` and the
maximum statistic difference is `5.3623772089395061e-14`. Both ranks report
`ERROR SUMMARY: 0 errors` in the one-step MPI Compute Sanitizer gate.

The same controlled MP7 case passes in two y slabs. This is the physical-y
decomposition gate: the two ranks exchange solution and geometry halos at the
internal y interface, while only the global lower-y rank contributes `fbcx`,
`wallheatflux`, and wall area to the BL statistic reduction.

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np2_yslab_20steps \
  CONSCHM=543e NP=2 TOPOLOGY=1,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The 20-step result has `q5 L_inf=1.7763568394002505e-14` and maximum statistic
difference `5.4956039718945249e-14`; the two-rank memcheck reports
`ERROR SUMMARY: 0 errors` for both ranks. `REYNOLDS` is an optional driver
parameter for diagnostic runs; the validated physical case uses `REYNOLDS=1000`.
The periodic z direction also passes in two z slabs:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np2_zslab_20steps \
  CONSCHM=543e NP=2 TOPOLOGY=1,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The 20-step result has `q5 L_inf=1.7763568394002505e-14`, maximum statistic
difference `5.4733995114020217e-14`, and a zero-error two-rank memcheck.

The first combined topology is also validated under two-GPU oversubscription:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np4_2x2x1_20steps \
  CONSCHM=543e NP=4 TOPOLOGY=2,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

This `2x2x1` gate passes with `q5 L_inf=1.7763568394002505e-14`, maximum
statistic difference `4.9737991503207013e-14`, and four zero-error memcheck
reports. The x/z combined topology is also validated under the same two-GPU
oversubscription:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np4_2x1x2_20steps \
  CONSCHM=543e NP=4 TOPOLOGY=2,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The `2x1x2` gate passes with `q5 L_inf=1.7763568394002505e-14`, maximum
statistic difference `5.0071058410594561e-14`, and four zero-error memcheck
reports. The y/z combined topology is also validated:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np4_1x2x2_20steps \
  CONSCHM=543e NP=4 TOPOLOGY=1,2,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The `1x2x2` gate has the same final `q5 L_inf=1.7763568394002505e-14`,
maximum statistic difference `5.0071058410594561e-14`, and four zero-error
memcheck reports. The full three-direction topology needs `KM=16`: the default
`KM=8` would make each z slab four points wide, below `hm=5` for MP7.

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np8_2x2x2_20steps \
  IM=64 JM=64 KM=16 CONSCHM=543e NP=8 TOPOLOGY=2,2,2 \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The `2x2x2` gate passes with final `q5 L_inf=1.7763568394002505e-14`, maximum
statistic difference `4.9737991503207013e-14`, and eight zero-error memcheck
reports. All four- and eight-rank cases are two-GPU-oversubscribed correctness
evidence only. Curvilinear geometry, high-Mach conditions, and characteristic
farfield behavior remain outside this contract.

## Phase S1-B0 HBL-Inspired Mach 3 Gate

`examples/Hypersonic_Boundary_Layer` is not directly GPU-runnable: its inputs
are two-dimensional, compact, and mostly dimensional. The S1-B0 driver creates
a separate non-dimensional, z-extruded explicit counterpart with the M3 x
extent, wall-normal clustering, `Mach=3`, `Re=100000`, and the M3 heated-wall
ratio `Twall/Tinf=568.89/226.65`:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b0_20steps \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

It uses explicit MP7 `543e/643e`, a deterministic heated profile, `11,prof`,
`21`, lower `41`, upper `51`, diffusion, no global filter, no characteristic
decomposition, and periodic z. The 20-step CPU/GPU gate passes with
`q5 L_inf=5.5511151231257827e-16`, maximum statistic difference
`7.8936857050848630e-14`, and a one-rank Compute Sanitizer report of zero
errors. This validates high-Mach parameters and the heated-wall explicit path,
not the original two-dimensional compact HBL case, characteristic farfield, or
SBLI behavior.

S1-B1 extends the same controlled HBL gate to individual MPI slabs:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b1_np2_x20 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b1_np2_y20 \
  NP=2 TOPOLOGY=1,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b1_np2_z20 \
  KM=16 NP=2 TOPOLOGY=1,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

All three runs pass with final `q5 L_inf=5.5511151231257827e-16`; their
maximum statistic differences are `6.3726801613483985e-14`,
`5.9729998724833422e-14`, and `7.8936857050848630e-14` for x/y/z,
respectively. The z slab requires `KM=16` because `KM=8` would leave only four
active z points per rank, below `hm=5`. A two-rank physical-y Compute Sanitizer
run reports zero errors on both ranks. Combined-axis and production-scaling
claims remain out of scope.

S1-B2 covers the three NP=4 two-axis combinations:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b2_np4_2x2x1_20 \
  NP=4 TOPOLOGY=2,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b2_np4_2x1x2_20 \
  KM=16 NP=4 TOPOLOGY=2,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b2_np4_1x2x2_20 \
  KM=16 NP=4 TOPOLOGY=1,2,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

All three runs pass with final `q5 L_inf=5.5511151231257827e-16`; maximum
statistic differences are `5.0182080713057076e-14`,
`6.3726801613483985e-14`, and `5.9729998724833422e-14`, respectively. The
`1x2x2` four-rank Compute Sanitizer run reports zero errors for every rank.
These are two-GPU-oversubscribed correctness tests. The full `2x2x2` HBL gate
remains separate.

S1-B3 closes the controlled HBL topology matrix with full x/y/z decomposition:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b3_np8_2x2x2_20 \
  KM=16 NP=8 TOPOLOGY=2,2,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

The `96x96x16` grid gives each rank `48x48x8` active points. The 20-step run
passes with `q5 L_inf=5.5511151231257827e-16`, maximum statistic difference
`5.0182080713057076e-14`, and zero-error Compute Sanitizer summaries from all
eight ranks. This is two-GPU-oversubscribed full-halo correctness evidence, not
a multi-GPU scaling result.

## Phase S1-C1 Mach 5 Sutherland Similarity Inlet

S1-C1 is the first HBL gate whose inlet comes from a coupled, isothermal-wall
compressible Blasius solution rather than an analytic exponential seed. The
generator uses `gamma=1.4`, `Pr=0.72`, Sutherland viscosity with
`Tref=226.65 K`, `Mach=5`, `Re=1.83052e6`, and
`Twall/Tinf=1176.64/226.65=5.191440547760865`. Its virtual leading-edge
station is `STATION_X=1.0`; the generated profile is imposed at the domain
inlet, whose x coordinate is `-1`.

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c1_m5_similarity_20 \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh
```

The `192x192x8` explicit MP7 case has a computed
`delta_99=9.5190551023136317e-3`. The 20-step CPU/GPU result passes with
`q5 L_inf=4.4408920985006262e-16` and maximum statistic difference
`3.5171865420124959e-13`.

The profile first line selects its density contract. `density=reconstruct` is
the backward-compatible default: ASTR sets `p=pinf` and reconstructs density
from temperature. `density=provided` preserves file density and computes
pressure from the ideal-gas equation of state; it rejects the input if that
pressure differs from `pinf` by more than `1e-10`. The similarity generator
writes the mathematically equivalent `rho=1/T` in both modes.

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c1_density_reconstruct \
  PROFILE_DENSITY_MODE=reconstruct MAXSTEP=2 \
  tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c1_density_provided \
  PROFILE_DENSITY_MODE=provided MAXSTEP=2 \
  tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh
```

Both modes pass their CPU/GPU comparisons. Comparing their CPU fields gives
maximum `T L_inf=6.2172489379008766e-15`; the difference is floating-point
round-off from evaluating `1/T` directly or through `pinf` and the equation of
state. This gate does not prove a complete physical flat-plate solution:
`blini` still copies the inlet profile over x at initialization, `51` is not a
characteristic farfield, and mesh/time/streamwise-development convergence and
external `Cf`/heat-transfer validation remain pending.

## Phase S1-C2 Mach 5 Similarity Field

C2 removes C1's x-uniform initialization by writing the existing CPU-readable
`datin/flowini3d.h5` and selecting `ninit=3`. The generator solves the C1
similarity ODE once, then maps it at every x with the local
`sqrt(2*x_s/Re)` scale. `VIRTUAL_LEADING_EDGE=-2` makes the `[-1,10]` domain
cover similarity stations `x_s=[1,12]`; HDF initialization remains CPU-owned,
followed by the normal one-time device upload.

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_similarity_field_20 \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh
```

The `192x192x8` 20-step CPU/GPU run passes with
`q5 L_inf=7.7715611723760958e-16` and maximum statistic difference
`3.3406610810970960e-13`. This validates the HDF initial-field reader and GPU
resident-field handoff, not combined-axis multi-rank HDF I/O, mesh/time
convergence, characteristic farfield treatment, or external skin-friction/
heat-transfer comparison.

The same C2 HDF field is validated in NP=2 x/y/z slabs:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_field_np2_x20 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_field_np2_y20 \
  NP=2 TOPOLOGY=1,2,1 MAXSTEP=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_field_np2_z20 \
  KM=16 NP=2 TOPOLOGY=1,1,2 MAXSTEP=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh
```

The x/y runs have maximum statistic differences `1.9806378759312793e-13` and
`2.9809488211185453e-13`; z has `3.3406610810970960e-13` and field
`q5 L_inf=9.9920072216264089e-16`. `KM=16` is required for z because each
rank otherwise has only four active z points, below `hm=5`.

## Phase S1-C3 Mach 5 NSCBC Farfield

S1-C3 is the candidate upper `bctype=52` NSCBC farfield gate for the same
Mach-5 flat-plate family. CPU probing initially showed that
`bc:farfield_nscbc` could apply its transverse filter on the `jmax` boundary
using stale k-halo values before a current halo refresh. In that state an
initially constant upper physical boundary was filtered into a nonconstant
range such as `0.84375..1.078125`, which is a halo artifact rather than a valid
characteristic-farfield result.

Under the CPU bug decision gate, GPU support did not reproduce that artifact.
The CPU oracle now refreshes halos before `boucon` whenever either
`bctype=22` or `bctype=52` is present, and the GPU `52` path is validated
against that corrected oracle:

```bash
tests/gpu_validation/run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh

OUT_DIR=/tmp/astr_s1c3_52_20 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh
```

Current expected result: both statistics and full-field comparisons print
`status: pass`. The latest default `192x192x8` two-step gate had reconstructed
`q5 L_inf=4.4408920985006262e-16`. The latest 20-step gate had reconstructed
`q5 L_inf=1.4432899320127035e-15` and maximum statistic difference
`3.4716673980028645e-13`.

## Phase S2-A0 Oblique-Shock HBL Compatibility

S2-A0 is the first deliberately narrow shock/boundary-layer compatibility gate.
It does not claim a resolved physical SBLI. It keeps the S1 Mach-5
Sutherland-viscosity flat-plate contract, initializes from `ninit=3`, and adds
an analytic oblique-shock overlay to the x-varying compressible Blasius HDF
field. The overlay updates `rho`, `T`, `u`, and `v` together using perfect-gas
oblique-shock ratios so CPU `readflowini3d` reconstructs a consistent pressure
from `rho*T`.

```bash
tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh

IM=64 JM=64 KM=8 MAXSTEP=2 FEQCHKPT=2 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_oblique_shock_smoke \
  tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh

OUT_DIR=tests/gpu_validation/out/s2_hbl_oblique_shock_mpirank_matrix \
  tests/gpu_validation/run_s2_hbl_oblique_shock_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/s2_hbl_inlet_sustained_shock \
  tests/gpu_validation/run_s2_hbl_inlet_sustained_shock_compare.sh
```

Current expected result: statistics and full-field comparisons print
`status: pass`. The current `64x64x8` smoke gate passes at one and two steps
with reconstructed `q5 L_inf=8.8817841970012523e-16` for the two-step run. The
default `192x192x8` two-step gate also passes with reconstructed
`q5 L_inf=1.3322676295501878e-15` and maximum statistic difference
`1.0755840662568517e-12`. The same two-step default gate now passes NP=2 x/y/z
slabs, NP=4 xy/xz/yz planes, and `NP=8 TOPOLOGY=2,2,2`; z-decomposed cases use
`KM=16` so local `km>=hm`. All MPI runs retain `q5 L_inf=1.3322676295501878e-15`.
The S2-B0 long subset currently passes `MAXSTEP=20` for NP=1 and
`NP=8 TOPOLOGY=2,2,2`; both have reconstructed `q5 L_inf=3.3306690738754696e-15`,
with largest statistic difference `1.2061462939527701e-12`.
The optional stress subset also passes `MAXSTEP=100` for NP=1 and NP=8; both
have reconstructed `q5 L_inf=5.7731597280508140e-15`, with largest statistic
difference `1.2216894162975223e-12`. Run it through the matrix driver with
`RUN_STRESS=t`.

S2-B1 adds a sustained compressed-inlet variant through
`run_s2_hbl_inlet_sustained_shock_compare.sh`. This wrapper enables
`PROFILE_OBLIQUE_SHOCK=t`, writes `density=provided pressure=provided` in
`inlet.prof`, and places the analytic shock line at the inlet so the upper
profile continuously injects the compressed post-shock state. `inletprofile`
reads the optional fifth pressure column only when the first profile line
contains `pressure=provided`; for non-dimensional pressure-provided density
profiles, it rejects inputs whose pressure is inconsistent with `rho*T`. The
current `64x64x8` two-step smoke has `q5 L_inf=8.8817841970012523e-16`. The
default `192x192x8` NP=1 two-step gate has `q5 L_inf=1.3322676295501878e-15`
and maximum statistic difference `7.8381745538536052e-13`. The `NP=2
TOPOLOGY=1,2,1` y-slab gate has the same `q5 L_inf` and maximum statistic
difference `7.8292927696566039e-13`, covering fifth-column profile scatter.
This remains a boundary-forced compatibility gate, not a full physical SBLI
validation.

This gate exposed a GPU `bctype=21` outlet compatibility bug that was hidden by
smooth S1 fields: CPU extrapolates sound speed as
`extrapolate(sos(T1),sos(T2))`, while the GPU previously used
`sqrt(extrapolate(T))/Mach`. The GPU outlet now extrapolates sound speed
directly.

S0-A7 extends the same selected Roe path to `NP=2 TOPOLOGY=2,1,1`. The driver places the jump at the global x-slab interface using `ASTR_SHUOSHER_SHOCK_X=0.d0`; raw `ssf` is exchanged through the generic `hm` transport before each rank expands its local mask and evaluates characteristic interfaces.

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a7_mpi_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a7_mpi_compare.sh
```

Current S0-A7 status: pass for x slabs. The three-step comparison has global raw sensor `L_inf=1.1102230246251565e-16`, zero mask mismatches, `q5 L_inf=9.1482377229112899e-14`, and maximum statistic difference `4.9098503041022923e-12`. The two-rank memcheck configuration used for S0-A5 reports zero errors on both S0-A7 ranks. This does not validate y/z characteristic interfaces or general multi-rank Roe support.

S0-A8 validates y slabs with `NP=2 TOPOLOGY=1,2,1` and `GRID=400,16,8`, so every local y domain satisfies `jm>=hm`. CPU raw `ssf` overlaps consistently at the duplicated y activity face, but its expanded `lshock` is locally owned; use rankwise CPU/GPU comparison instead of a globally merged mask.

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a8_y_mpi_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a8_y_mpi_compare.sh
```

Current S0-A8 status: pass. The three-step gate has rankwise raw `L_inf=1.1102230246251565e-16`, zero local mask mismatches, `q5 L_inf=8.8817841970012523e-16`, and maximum statistic difference `4.4906300900038332e-12`. Both Compute Sanitizer ranks report zero errors. Do not use `GRID=400,8,8` for this topology because local `jm=4<hm=5` causes halo pack ranges to include an undefined local halo value.

S0-A9 completes the corresponding z-slab gate with `NP=2 TOPOLOGY=1,1,2` and `GRID=400,8,16`, keeping local `km>=hm`. It uses the same rankwise raw/mask validation contract as S0-A8.

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a9_z_mpi_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a9_z_mpi_compare.sh
```

Current S0-A9 status: pass. The three-step gate has zero local mask mismatches, `q5 L_inf=8.8817841970012523e-16`, and both Compute Sanitizer ranks report zero errors. The single-axis `NP=2` gates do not validate combined `2x2x2` topology.

The one-step S0-A2 path also passed Compute Sanitizer memcheck with `ERROR SUMMARY: 0 errors` using `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self`, and `OMPI_MCA_osc=pt2pt` to prevent OpenMPI UCX initialization from generating CUDA-context false positives. Post-S0-A2 regressions passed for S0-A1, filtered/diffusive TGV, all three Phase K source entries, and all 13 Phase J RTI entries.

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

The driver prepares GPU-only `NP=1` and `NP=2` case copies, profiles both with Nsight Systems, and compares their `flowstate.dat` statistics. The reusable driver has a recorded pass in `documents/GPU_VALIDATION_MATRIX.md`; rerun it when profiling, halo exchange, residency, or build configuration changes.

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
