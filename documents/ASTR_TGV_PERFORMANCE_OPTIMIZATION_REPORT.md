# ASTR TGV Single-GPU Performance Optimization Report

## 1. Result

The second TGV single-GPU optimization gate passes on the local RTX 4000 Ada
workstation. For the fixed `256^3` FP64 case, the median complete-RK time is
reduced from the original `0.762855769 s` to `0.580044424 s`. This is a
`23.964%` time reduction, a `1.31517x` speedup, and a `31.517%` throughput
increase. Relative to the frozen first-round baseline of `0.622173751 s`, the
second round reduces time by `6.771%`.

The result retains the sixth-order `643e` explicit central scheme, the
tenth-order explicit filter, diffusion, RK3 stage ordering, GPU-resident
fields, and an explicit `cudaDeviceSynchronize` after every kernel. No MPI
halo transport or numerical boundary contract was changed.

## 2. Benchmark Contract

| Item | Fixed value |
|---|---|
| Precision | FP64 |
| Case | 3-D periodic Taylor-Green vortex |
| Grid | `256,256,256` |
| MPI topology | `NP=1`, `1x1x1` |
| Convection/diffusion | `643e/643e`, sixth-order explicit central |
| Filter | tenth-order explicit, enabled |
| Diffusion | enabled |
| Time integration | three-stage RK3 |
| Controller | `maxstep=10`, therefore 11 complete RK advances |
| Measured samples | discard first in-process advance, retain 10 per process |
| Repeats | one process warm-up plus five independent measured processes |
| Output during measured loop | checkpoint and full-field output disabled |
| Timing scope | first-stage preparation plus the remaining complete RK advance |

The timing scope excludes process startup, grid construction, initial field
output, and finalization. Those operations are not changed by the optimized
GPU compute path and would obscure the complete-RK result.

## 3. Test Platform

- GPU: NVIDIA RTX 4000 Ada Generation, 19195 MiB, compute capability 8.9.
- Driver: 595.84.
- Compiler: NVHPC `nvfortran 26.1`.
- MPI: Open MPI `4.1.9a1`.
- Build: top-level `CMakeLists.txt`, `ASTR_WITH_CUDA=ON`, `RELEASE`.
- Baseline source: commit `30a5f4d` plus disabled-by-default RK timing
  instrumentation, before the kernel changes in this report.

## 4. Performance Acceptance

| Metric | Baseline | Optimized | Acceptance | Result |
|---|---:|---:|---:|---|
| Median complete-RK time, overall | 0.762855769 s | 0.580044424 s | at least 10% lower | pass, 23.964% |
| Median complete-RK time, round 2 | 0.622173751 s | 0.580044424 s | at least 3% lower | pass, 6.771% |
| Run-to-run spread | 2.105% | 2.027% | at most 5% | pass |
| Median throughput | 2.199264e7 cell-RK/s | 2.892402e7 cell-RK/s | informational | +31.517% |
| Peak sampled device memory | 9766 MiB | 9738 MiB | growth at most 5% | pass, -0.287% |
| Peak sampled GPU utilization | 100% | 100% | informational | observed |

The five final process medians are `0.579141140`, `0.580044424`,
`0.580992526`, `0.590146962`, and `0.578387397 s/RK`.

Raw local evidence is under:

- `tests/gpu_validation/out/tgv_256_performance_baseline_30a5f4d/`
- `tests/gpu_validation/out/tgv_256_performance_optimized_final_source/`
- `tests/gpu_validation/out/tgv_256_deriv6_reciprocal_candidate/`
- `tests/gpu_validation/out/tgv_256_num1d60_candidate/`

## 5. Retained Optimizations

1. Periodic x/y/z convection, diffusion-RHS, and filter kernels retain the
   prescribed block shapes `(512,1,1)`, `(32,16,1)`, and `(64,1,8)`, but map
   the active plane or volume linearly across the available threads. This
   removes inactive lanes caused by launching a 512-thread x block for 257
   grid nodes and improves contiguous Fortran-order access where possible.
2. Conservative-to-primitive conversion keeps velocity and pressure in local
   variables instead of writing them to global arrays and immediately reading
   them back for kinetic energy, pressure, and temperature.
3. Fully periodic filtered cases skip the redundant first-stage primitive
   refresh. The primitive state is already current after initial GPU upload or
   the final refresh of the preceding RK advance. Physical-boundary cases keep
   the original refresh.
4. The full-volume primitive kernel uses `(32,4,4)`, while the diffusion-flux
   kernel retains its original `(8,8,8)` launch.
5. The x-periodic primitive halo kernel maps halo position as the fastest
   thread-varying index and reuses local primitive values. This changes only
   traversal order, not the halo points or formulas.
6. `ASTR_GPU_RK_TIMING=1` enables complete-RK timing. It is disabled by
   default and does not affect normal runs.
7. All GPU sixth-order central derivatives now import `num1d60` from the CPU
   `constdef` module and multiply the outer difference by that compile-time
   parameter. This follows the existing CPU `derivative.F90` convention and
   removes repeated device reciprocal sequences from diffusion flux,
   convection, diffusion RHS, and `gradcal`. WENO reconstruction divisions
   and unrelated physical-model divisions are unchanged.
8. `run_tgv_256_ncu_hotspot_matrix.sh` profiles 12 common kernels against one
   reusable case. `summarize_ncu_hotspot_matrix.py` records throughput,
   registers, occupancy, instruction count, excessive L2 sectors, branch
   uniformity when available, normalized warp-stall shares, and the three
   highest sampled source lines.

## 6. Profile Evidence

Nsight Systems was collected for three complete RK advances before and after
optimization. The following values are total kernel time over nine launches.

| Kernel group | Baseline | Optimized | Reduction |
|---|---:|---:|---:|
| convection x/y/z | 541.986 ms | 470.565 ms | 13.18% |
| diffusion RHS x/y/z | 465.184 ms | 403.775 ms | 13.20% |
| filter x/y/z | 204.525 ms | 174.663 ms | 14.60% |
| full-volume primitive conversion | 82.782 ms, 12 calls | 60.039 ms, 9 calls | 27.47% time |
| x-periodic primitive halo | 63.476 ms | 11.324 ms | 82.16% |
| diffusion flux | 239.067 ms | 239.352 ms | unchanged |
| all GPU kernels | 1944.413 ms | 1708.811 ms | 12.12% |

Nsight Compute identifies `diffusion_flux_global_kernel` as an FP64
compute-bound residual hotspot: compute throughput is `87.22%`, DRAM
throughput is `32.31%`, register use is 124 per thread, and achieved occupancy
is `32.21%`. A trial one-dimensional launch reduced DRAM throughput and
achieved occupancy and did not improve the screening time, so it was removed
from the final source.

Source-correlated Nsight Compute data showed that the original `/60.d0` in
`deriv6` generated a `MUFU.RCP64H` reciprocal sequence at every call site.
Replacing that division with multiplication by the compile-time
`constdef::num1d60` parameter changes the profiled diffusion-flux kernel as
follows:

| Metric | Before reciprocal change | After reciprocal change | Change |
|---|---:|---:|---:|
| NCU replay duration | 46.07 ms | 27.49 ms | -40.33% |
| executed instructions | 689.61 million | 430.70 million | -37.55% |
| compute throughput | 87.22% | 83.62% | -3.60 percentage points |
| DRAM throughput | 32.31% | 51.85% | +19.54 percentage points |
| registers per thread | 124 | 128 | +4 |
| achieved occupancy | 32.21% | 31.96% | -0.25 percentage points |

The matching Nsight Systems trace confirms the change under normal execution,
without metric replay. Across nine launches, diffusion-flux time decreases
from `239.351772` to `164.919657 ms` (`31.10%`), while total GPU-kernel time
decreases from `1708.810503` to `1536.671839 ms` (`10.07%`). The residency
audit still reports zero forbidden H2D/D2H transfers at or above 64 KiB.

The remaining `MUFU.RCP64H` instructions correspond to other divisions in
viscosity, heat conduction, and trace removal, not the sixth-order outer
coefficient. A `(32,4,4)` diffusion-flux block candidate produced
`0.682961920 s/RK`, 0.093% slower than the retained `(8,8,8)` baseline, and
was removed.

The second-round NCU matrix covers 12 kernels and all three stencil
directions. NCU replay time is diagnostic and is not used as wall-clock
performance evidence.

| Kernel | NCU duration | SM peak | DRAM peak | Registers | Occupancy | Dominant not-issued stalls |
|---|---:|---:|---:|---:|---:|---|
| convection x | 28.184 ms | 82.55% | 40.28% | 128 | 31.86% | TEX 47.89%, short SB 43.51% |
| convection y | 27.213 ms | 81.75% | 41.43% | 128 | 32.37% | short SB 43.99%, TEX 43.66% |
| convection z | 32.366 ms | 82.96% | 35.38% | 128 | 32.39% | short SB 47.66%, TEX 41.10% |
| diffusion flux | 27.453 ms | 83.63% | 51.82% | 128 | 31.97% | TEX 47.44%, short SB 43.11% |
| diffusion RHS x | 15.941 ms | 83.36% | 64.52% | 128 | 32.21% | TEX 57.57%, short SB 31.00% |
| diffusion RHS y | 16.894 ms | 78.87% | 60.29% | 128 | 31.67% | short SB 44.02%, TEX 34.87% |
| diffusion RHS z | 16.775 ms | 79.49% | 61.19% | 128 | 31.77% | short SB 39.97%, TEX 37.55% |
| filter x | 8.627 ms | 82.50% | 58.81% | 53 | 59.52% | TEX 55.46%, short SB 30.79% |
| filter y | 9.606 ms | 73.69% | 52.50% | 80 | 31.76% | short SB 46.66%, TEX 37.18% |
| filter z | 9.713 ms | 73.52% | 52.69% | 80 | 31.78% | short SB 46.56%, TEX 36.38% |
| primitive conversion | 6.670 ms | 91.39% | 80.21% | 64 | 53.25% | short SB 73.27% |
| gradcal | 35.463 ms | 85.38% | 44.83% | 88 | 32.21% | short SB 77.12% |

Source correlation identifies the `/60.d0` sixth-order term as the highest
sampled line in convection x/y/z and `gradcal`. Replacing it with the existing
`num1d60` parameter changes two representative kernels as follows:

| Kernel | NCU duration before/after | Instructions before/after | Registers before/after | `MUFU.RCP64H` sites before/after |
|---|---:|---:|---:|---:|
| convection x | 28.184 / 17.581 ms | 383.47 / 252.50 million | 128 / 128 | 5 / 0 |
| gradcal | 35.463 / 17.972 ms | 671.53 / 403.93 million | 88 / 72 | 24 / 0 |

The normal-execution Nsight Systems trace confirms that this is not an NCU
replay artifact. Across three complete RK advances, convection x/y/z falls
from `470.194833` to `334.916856 ms` and `gradcal` falls from `65.577973` to
`46.082315 ms`. Total GPU-kernel time falls from `1536.671839` to
`1382.294955 ms`, a `10.046%` reduction. Diffusion RHS and filter totals are
unchanged to within `0.02 ms`, consistent with the measured scope.

The final reports are under:

- `tests/gpu_validation/out/tgv_256_profile_optimized_final_source/`
- `tests/gpu_validation/out/tgv_256_profile_optimized_final_source_ncu/`
- `tests/gpu_validation/out/tgv_256_deriv6_reciprocal_ncu/`
- `tests/gpu_validation/out/tgv_256_ncu_hotspot_matrix_retained/`
- `tests/gpu_validation/out/tgv_256_ncu_num1d60_candidate/`
- `tests/gpu_validation/out/tgv_256_ncu_num1d60_gradcal_filtered/`
- `tests/gpu_validation/out/tgv_256_num1d60_nsys/`
- `tests/gpu_validation/out/tgv_256_profile_30a5f4d/`

## 7. Correctness, Safety, and Residency

The final binary passes 10-step CPU/GPU comparisons at `atol=rtol=1e-10`:

| Gate | Largest reported field error | Largest statistic error | Result |
|---|---:|---:|---|
| NP=1 | `q5=2.8421709430404007e-13` | `4.6740389336719090e-14` | pass |
| NP=2, `2x1x1` | `q5=3.1263880373444408e-13` | `4.6671000397680018e-14` | pass |

The final `num1d60` candidate repeats the NP=1 and NP=2 10-step field and
statistic gates. All reported differences remain below `1e-10`.

A five-step `64^3` channel filter-plus-diffusion regression also passes its
statistics gate. Its existing full-checkpoint comparison still fails on the
x/z periodic duplicate planes, as documented in Section 8. When those
two boundary planes on each periodic array axis are trimmed, the retained
periodic interior and all y-wall planes have `q5 L_inf=4.618527782440651e-14`.
This is supporting evidence for the physical-boundary derivative path, not a
full-field pass or closure of the checkpoint defect.

A representative `32^3`, one-step, filter-plus-diffusion Compute Sanitizer
run reports `ERROR SUMMARY: 0 errors`. OpenMPI UCX CUDA probing must be
disabled for this tool-only run with `OMPI_MCA_pml=ob1`,
`OMPI_MCA_btl=self`, and `OMPI_MCA_osc=pt2pt`; otherwise UCX probes CUDA
before ASTR creates a valid context and produces initialization-layer tool
errors.

The final acceptance outputs are under:

- `tests/gpu_validation/out/tgv_perf_final_np1_10/`
- `tests/gpu_validation/out/tgv_perf_final_np1_stats_10/`
- `tests/gpu_validation/out/tgv_perf_final_np2_xslab_field_10/`
- `tests/gpu_validation/out/tgv_perf_final_np2_xslab_stats_10/`
- `tests/gpu_validation/out/tgv_perf_final_memcheck_np1/`
- `tests/gpu_validation/out/tgv_num1d60_np1_field_10/`
- `tests/gpu_validation/out/tgv_num1d60_np1_stats_10/`
- `tests/gpu_validation/out/tgv_num1d60_np2_field_10/`
- `tests/gpu_validation/out/tgv_num1d60_np2_stats_10/`
- `tests/gpu_validation/out/channel_num1d60_filter_diff_5/`
- `tests/gpu_validation/out/tgv_num1d60_memcheck/`

The final no-checkpoint Nsight Systems residency audit starts at the first
filter kernel. It finds no forbidden transfer at or above 64 KiB. H2D has a
maximum size of 128 B. The 12 large D2H transfers are exactly 512 KiB each and
immediately follow the existing TGV kinetic-energy, enstrophy, or dissipation
partial-reduction kernels. They are diagnostic partial arrays, not complete
flow fields. The audit reports:

```text
large_h2d_d2h_count: 12
allowed_large_d2h_count: 12
forbidden_large_h2d_d2h_count: 0
h2d_max_bytes: 128
d2h_max_bytes: 524288
cuda_device_synchronize_count: 278
```

## 8. Non-goal Regression Observation

An additional one-step channel smoke test found a pre-existing checkpoint
comparison issue that is independent of the retained TGV optimizations. CPU
and GPU interior values agree to about `1e-15`, while differences are confined
to the duplicate planes in the two periodic directions. The physical wall
planes agree. History traces this behavior to the host-side checkpoint
preparation added in commit `16ded59`, where an NSCBC-oriented `boucon/qswap`
sequence is applied to every non-zero-dimensional GPU case. The current Goal
does not change `src/mainloop.F90`; this output-phase regression requires a
separate correctness decision and is not used as TGV performance evidence.

## 9. Reproduction

Build through the repository top-level CMake file:

```bash
cmake -S . -B build_gpu_probe \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DASTR_WITH_CUDA=ON
cmake --build build_gpu_probe -j 8
```

Run the five-repeat benchmark or either profiler:

```bash
OUT_DIR=/tmp/astr_tgv_perf \
  tests/gpu_validation/run_tgv_256_performance_benchmark.sh

PROFILE_TOOL=nsys OUT_DIR=/tmp/astr_tgv_nsys \
  tests/gpu_validation/run_tgv_256_performance_profile.sh

PROFILE_TOOL=ncu MAXSTEP=0 OUT_DIR=/tmp/astr_tgv_ncu \
  tests/gpu_validation/run_tgv_256_performance_profile.sh

OUT_DIR=/tmp/astr_tgv_ncu_matrix \
  tests/gpu_validation/run_tgv_256_ncu_hotspot_matrix.sh

OUT_DIR=/tmp/astr_tgv_memcheck \
  tests/gpu_validation/run_tgv_gpu_memcheck.sh
```

The benchmark prepares one case and reuses it for all repeats. Initial grid
and field files are overwritten in place while timing and monitor logs remain
separate, avoiding about 1.2 GB of duplicate HDF5 data per repeat.

The NCU driver writes the binary report plus terminal-readable exports:

- `diffusion_flux_full.ncu-rep`: complete report for later import;
- `ncu_details.txt`: section and rule output;
- `ncu_source.csv`: Fortran/SASS source correlation and line-level metrics.

Equivalent direct CLI inspection commands are:

```bash
ncu --list-sets
ncu --list-sections
ncu --import diffusion_flux_full.ncu-rep --page details --print-details all
ncu --import diffusion_flux_full.ncu-rep \
  --page source --print-source cuda,sass --csv --print-units base
```

For A800, create a fresh build directory with the same top-level command and
run the same scripts. Do not reuse the Ada workstation CMake cache. Select the
device with `GPU_ID`; for example:

```bash
GPU_ID=0 OUT_DIR=/tmp/astr_tgv_a800_perf \
  tests/gpu_validation/run_tgv_256_performance_benchmark.sh
GPU_ID=0 PROFILE_TOOL=nsys OUT_DIR=/tmp/astr_tgv_a800_nsys \
  tests/gpu_validation/run_tgv_256_performance_profile.sh
GPU_ID=0 PROFILE_TOOL=ncu MAXSTEP=0 OUT_DIR=/tmp/astr_tgv_a800_ncu \
  tests/gpu_validation/run_tgv_256_performance_profile.sh
```

The workstation result is not an A800 performance claim. The A800 run must
record its own five-repeat median, spread, memory, Nsight Systems hotspots,
and Nsight Compute metrics before it is used in a paper or presentation.

## 10. Selective Synchronization Experiment

`ASTR_GPU_SYNC_MODE=selective` is an opt-in mode limited to three-dimensional
fully periodic TGV. The default remains `explicit`. Both modes call
`cudaGetLastError` after every kernel. Selective mode relies on default-stream
ordering between device-only kernels and retains a device synchronization at
the following visibility boundaries:

- complete-RK preparation and complete-RK timing boundaries;
- partial reductions before host diagnostic copies;
- MPI pack completion before host-staged halo exchange;
- full or boundary field copies to the host;
- final asynchronous error closure.

The final selective Nsight Systems trace contains the same 275 kernels and the
same transfer structure as the explicit trace. Synchronization calls after the
first filter kernel decrease from 275 to 15, while forbidden transfers at or
above 64 KiB remain zero. This reduction does not improve the fixed `256^3`
workload on the RTX 4000 Ada workstation:

| Mode | Synchronizations in profiled RK interval | Five-run median | Spread | Relative to explicit |
|---|---:|---:|---:|---:|
| explicit | 275 | 0.682327627 s/RK | 1.751% | baseline |
| selective | 15 | 0.683028141 s/RK | 2.544% | 0.103% slower |
| compute-only experimental candidate | 152 | 0.705707483 s/RK | 6.677% | 3.426% slower |

The compute-only candidate retained local halo synchronization and was removed
after measurement. The retained selective mode is not the performance default.
It exists to make synchronization dependencies auditable and to permit a
controlled A800 rerun. The workstation result shows that this large FP64 case
is dominated by kernel execution rather than host synchronization latency.

Numerical behavior is unchanged. NP=1 and NP=2 x-slab ten-step CPU/GPU field
and statistic comparisons pass at `1e-10`; direct explicit/selective field
comparisons have zero difference. The selective one-step memcheck reports
`ERROR SUMMARY: 0 errors`. A non-TGV request is rejected before GPU stepping.

Evidence directories are:

- `tests/gpu_validation/out/tgv_256_sync_selective_candidate1/`
- `tests/gpu_validation/out/tgv_256_profile_selective_candidate1/`
- `tests/gpu_validation/out/tgv_256_sync_compute_candidate2/`
- `tests/gpu_validation/out/tgv_256_profile_compute_candidate2/`
- `tests/gpu_validation/out/tgv_sync_selective_final_np1_field_10/`
- `tests/gpu_validation/out/tgv_sync_selective_final_np1_stats_10/`
- `tests/gpu_validation/out/tgv_sync_selective_final_np2_field_10/`
- `tests/gpu_validation/out/tgv_sync_selective_final_np2_stats_10/`
- `tests/gpu_validation/out/tgv_sync_selective_final_memcheck/`

## 11. Next Optimization Boundary

The shared sixth-order reciprocal hotspot is closed in diffusion, convection,
and `gradcal`. The next candidates should target repeated velocity/metric
loads in convection and diffusion flux, the 128-register diffusion RHS path,
and y/z filter access. Any such candidate needs a new NCU source comparison
because the current matrix shows mixed compute, TEX-throttle, and scoreboard
limits rather than one universal bottleneck. Further removal of required
visibility-boundary synchronization is not justified by the current
measurements. Changing precision, the numerical format, or MPI halo transport
remains a separate design decision.
