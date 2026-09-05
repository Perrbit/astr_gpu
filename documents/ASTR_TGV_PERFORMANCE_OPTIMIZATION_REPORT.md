# ASTR TGV Single-GPU Performance Optimization Report

## 1. Result

The first TGV single-GPU optimization gate passes on the local RTX 4000 Ada
workstation. For the fixed `256^3` FP64 case, the median complete-RK time is
reduced from `0.762855769 s` to `0.682327627 s`. This is a `10.556%` time
reduction, a `1.11802x` speedup, and an `11.802%` throughput increase.

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
| Median complete-RK time | 0.762855769 s | 0.682327627 s | at least 10% lower | pass, 10.556% |
| Run-to-run spread | 2.105% | 1.751% | at most 5% | pass |
| Median throughput | 2.199264e7 cell-RK/s | 2.458821e7 cell-RK/s | informational | +11.802% |
| Peak sampled device memory | 9646 MiB | 9768 MiB | growth at most 5% | pass, +1.265% |
| Peak sampled GPU utilization | 100% | 100% | informational | observed |

The five optimized process medians are `0.683465901`, `0.687250710`,
`0.679855284`, `0.675305744`, and `0.682327627 s/RK`.

Raw local evidence is under:

- `tests/gpu_validation/out/tgv_256_performance_baseline_30a5f4d/`
- `tests/gpu_validation/out/tgv_256_performance_optimized_final_source/`

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

The final reports are under:

- `tests/gpu_validation/out/tgv_256_profile_optimized_final_source/`
- `tests/gpu_validation/out/tgv_256_profile_optimized_final_source_ncu/`
- `tests/gpu_validation/out/tgv_256_profile_30a5f4d/`

## 7. Correctness, Safety, and Residency

The final binary passes 10-step CPU/GPU comparisons at `atol=rtol=1e-10`:

| Gate | Largest reported field error | Largest statistic error | Result |
|---|---:|---:|---|
| NP=1 | `q5=3.4106051316484809e-13` | `4.6740389336719090e-14` | pass |
| NP=2, `2x1x1` | `q5=3.4106051316484809e-13` | `4.6671000397680018e-14` | pass |

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

## 10. Next Optimization Boundary

The largest remaining kernel is the compute-bound diffusion-flux calculation.
One next optimization can target its arithmetic and register lifetime with an
isolated kernel-level correctness oracle. A separate approved Goal can remove
redundant explicit synchronizations while preserving stream-order data
dependencies, MPI communication boundaries, error reporting, and the existing
NP=1/NP=2 field and statistic thresholds. Synchronization removal must be
staged and benchmarked independently; changing the numerical format or MPI
halo transport remains outside that Goal.
