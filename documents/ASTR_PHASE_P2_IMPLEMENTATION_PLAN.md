# ASTR Phase P2 Shock And SBLI Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish uncontaminated single-GPU performance and shock-activity baselines for the selective-Roe path before changing any numerical kernel.

**Architecture:** Use two complementary cases. A three-dimensional periodic Shu-Osher extrusion isolates sensor and characteristic-flux costs, while a three-dimensional SBLI case measures the complete wall, diffusion, NSCBC, sponge, and selective-Roe path. Shock activity is extracted from a separate one-step dump so the timed loop contains no diagnostic copies or reductions.

**Tech Stack:** Bash, Python 3, NumPy, CUDA Fortran, MPI, Nsight Systems, Nsight Compute, `nvidia-smi`.

## Global Constraints

- Build only from `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Keep the CPU and GPU numerical implementation unchanged during P2-0.
- Keep explicit synchronization after every kernel.
- Keep FP64, MP7/selective Roe, the current shock sensor, and existing boundary semantics.
- Run performance baselines with one MPI rank bound to one physical GPU.
- Do not write checkpoints or full fields in a timed run.
- Do not infer A800 performance from RTX 4000 Ada measurements.
- Keep `OMPI_MCA_sharedfp=individual` as the local P2 runner default because
  HPC-X `sharedfp=sm` blocks in HDF5 `MPI_File_open`; allow platforms to
  override the component after verifying their MPI/HDF5 stack.

---

### Task 1: Record Hardware-Specific P1 Decisions

**Files:**
- Modify: `documents/ASTR_GPU_PERFORMANCE_OPTIMIZATION_PLAN.md`
- Modify: `documents/ASTR_TGV_PERFORMANCE_OPTIMIZATION_REPORT.md`

**Interfaces:**
- Consumes: P1 timing and NCU evidence already recorded in both documents.
- Produces: An explicit decision that rejected P1 source variants remain valid A800 retest candidates, not retained workstation defaults.

- [x] State that each P1 rejection applies to the RTX 4000 Ada baseline and current compiler/toolchain.
- [x] Preserve candidate source mechanisms and evidence paths for A800 reconstruction.
- [x] Require a fresh A800 baseline, compiler record, five-repeat complete-RK timing, and NCU mechanism check before reconsideration.
- [x] Keep `3%` complete-RK improvement and all correctness gates unchanged.

### Task 2: Add Offline Shock-Activity Analysis

**Files:**
- Modify: `tests/gpu_validation/compare_shock_sensor.py`
- Modify: `tests/gpu_validation/test_compare_shock_sensor.py`
- Create: `tests/gpu_validation/summarize_shock_activity.py`

**Interfaces:**
- Consumes: `SensorDump.mask` returned by `read_sensor_dump_set(path)`.
- Produces: `summarize_shock_activity(mask)` with active-node and x/y/z active-interface counts and fractions.

- [x] Add a failing unit test using a small known 3-D mask.
- [x] Verify the test fails because activity summarization is absent.
- [x] Implement adjacent-node OR plus endpoint semantics for x/y/z launched interfaces.
- [x] Add a CLI that writes an ASCII report from one existing sensor dump or merged rank dump set.
- [x] Run the focused unit test and the complete Python validation suite.

### Task 3: Support Output-Free Prepared Cases

**Files:**
- Modify: `tests/gpu_validation/prepare_tgv_case.py`
- Modify: `tests/gpu_validation/test_prepare_tgv_case.py`

**Interfaces:**
- Consumes: Existing six-column controller format.
- Produces: Optional `--feqlist N`, preserving existing behavior when omitted.

- [x] Add a failing test proving `set_controller_steps` can set `maxstep`, `feqchkpt`, and `feqlist` together.
- [x] Verify the test fails with the old function signature.
- [x] Add the optional argument and CLI validation.
- [x] Run the focused preparer tests.

### Task 4: Add Dual P2 Baseline Runners

**Files:**
- Create: `tests/gpu_validation/summarize_gpu_rk_performance.py`
- Create: `tests/gpu_validation/test_summarize_gpu_rk_performance.py`
- Create: `tests/gpu_validation/run_p2_shock_performance_benchmark.sh`
- Create: `tests/gpu_validation/run_p2_shock_activity.sh`

**Interfaces:**
- Consumes: `CASE=shuosher|sbli`, the top-level GPU executable, existing case preparers, and `ASTR_GPU_RK_TIMING`.
- Produces: Five-repeat timing TSV, Markdown summary, GPU utilization/memory samples, and a separate shock-activity report.

- [x] Add failing summary tests for five contiguous repeats, retained sample count, and case metadata.
- [x] Implement the generic summary parser and report generator.
- [x] Implement one warm-up plus at least five measured runs, with `FEQCHKPT` and `FEQLIST` greater than `MAXSTEP`.
- [x] Default Shu-Osher to `256,64,32` and SBLI to `256,192,32`, `MAXSTEP=20`, `DISCARD_STEPS=1`, and explicit synchronization.
- [x] Reject unsupported cases, fewer than five repeats, output frequencies inside the measured loop, and non-explicit synchronization.
- [x] Implement a separate one-step activity runner that never contributes samples to performance TSV files.
- [x] Run `bash -n` and small-grid smoke preparations before launching production-size baselines.

### Task 5: Add P2 Nsight Capture

**Files:**
- Create: `tests/gpu_validation/run_p2_shock_performance_profile.sh`
- Modify: `documents/ASTR_GPU_PERFORMANCE_OPTIMIZATION_PLAN.md`

**Interfaces:**
- Consumes: A prepared P2 case and `PROFILE_TOOL=nsys|ncu`.
- Produces: Raw `.nsys-rep` or `.ncu-rep`, profiler log, and reproducible command metadata.

- [x] Capture one Nsight Systems run without checkpoints or full-field output.
- [x] Capture the highest-weight characteristic-flux kernel with Nsight Compute; sensor kernels remain below 1% in Nsight Systems and do not justify full NCU replay in P2-0.
- [x] Record duration, registers/thread, spill traffic, achieved occupancy, branch efficiency, and DRAM/L2 metrics.
- [x] Select the first optimization candidate only after ranking complete-RK kernel weights and shock-active fractions.

### Task 6: Baseline Verification

**Files:**
- Modify: `documents/ASTR_GPU_PERFORMANCE_OPTIMIZATION_PLAN.md`

**Interfaces:**
- Consumes: P2 timing, activity, Nsight, and existing CPU/GPU comparison scripts.
- Produces: A measured P2-0 baseline and an evidence-backed P2-1 candidate recommendation.

- [x] Build CPU and GPU executables from the top-level CMake project.
- [x] Run unit tests and shell syntax checks.
- [x] Run small-grid Shu-Osher and SBLI correctness smoke gates.
- [x] Run Compute Sanitizer before profiling production-size cases.
- [x] Run five-repeat NP=1 timing for both baselines.
- [x] Run separate mask-activity diagnostics.
- [x] Capture Nsight Systems and Nsight Compute evidence.
- [x] Report the measured hotspot and request approval before modifying a numerical kernel.

### Task 7: Screen P2-1A Y-Direction Split

**Scope:**

- Compute all y-interface physical-space MP7 fluxes into the existing work array.
- Overwrite only shock-active interfaces with Roe characteristic fluxes.
- Preserve the existing y RHS kernel and explicit synchronization after every kernel.

- [x] Add and observe a failing source contract for periodic and xy-physical paths.
- [x] Implement the minimum y-only split without adding device arrays.
- [x] Build from the top-level CMake project and pass 47 Python tests.
- [x] Pass NP=1 Shu-Osher and SBLI one-step CPU/GPU field gates.
- [x] Pass NP=2 y-slab sensor, field, and statistics comparison.
- [x] Pass Compute Sanitizer with `0 errors`.
- [x] Capture Nsight Systems and Nsight Compute mechanism evidence.
- [x] Run five measured complete-RK repeats for both formal baselines.
- [x] Reject and remove the candidate after Shu-Osher slowed `21.588%` and SBLI slowed `1.154%`.
- [x] Preserve raw evidence and document the candidate as an A800-local retest option.

### Task 8: Retain P2-1B Physical Five-Component Reuse

**Scope:**

- Keep the existing single sensor-coupled characteristic interface kernel.
- In its nonshock branch, evaluate all five physical Steger-Warming split
  components once per stencil point instead of repeating common state work for
  each component.
- Reuse the existing thread-local `split_plus(5,7)` and
  `split_minus(5,7)` arrays; add no full-field work array or kernel launch.
- Preserve `npdc=3` halo sampling in both nonshock and Roe branches for future
  decompositions containing an interior rank.

- [x] Add and observe a failing source contract for the vector physical helper.
- [x] Preserve the physical-path `sqrt(tmp)/mach` sound speed and component formulas.
- [x] Add and observe a failing contract for `npdc=3` internal-rank halo sampling.
- [x] Apply the same `npdc=3` halo contract to the Roe-active branch.
- [x] Build with the top-level CMake project.
- [x] Pass NP=1 one-step Shu-Osher and SBLI field comparisons.
- [x] Pass NP=2 x/y/z and NP=8 `2x2x2` halo comparisons.
- [x] Pass ten-step Shu-Osher and ten-step SBLI HDF5 field comparison.
- [x] Pass SBLI Compute Sanitizer with `0 errors`.
- [x] Use a same-workspace A/B build to prove the ten-step SBLI online
  `massflux` discrepancy predates P2-1B; candidate and baseline GPU outputs are
  bitwise identical.
- [x] Confirm with Nsight Compute that SBLI y-kernel local spilling falls about
  `78.1%`, while registers remain 128/thread.
- [x] After statistics-phase correction, confirm frozen complete-RK medians
  improve `27.902%` for Shu-Osher and `48.684%` for SBLI with spreads below
  `5%` and no peak-memory increase.
- [x] Retain P2-1B and isolate the GPU online `massflux` preparation discrepancy
  from field-evolution correctness before changing performance kernels again.

### Task 9: Close The SBLI Statistics-Phase Defect

**Scope:**

- Match the CPU order in which `bctype=52` applies the upper-y transverse x/z
  filters before `rkfirst` statistics observe the conservative state.
- Snapshot the complete pre-statistics RK state and restore it before the
  formal RK advance, so the correction changes diagnostics but not evolution.
- Reuse the existing `qsave_d` RK storage for the physical-domain snapshot;
  do not allocate another full-field device array.

- [x] Add failing source contracts for farfield-case snapshot coverage, x/z
  filter order, restored-state re-preparation, and `qsave_d` reuse.
- [x] Reproduce the defect as a statistics-phase mismatch rather than a
  reduction or HDF5-field error.
- [x] Apply x filter, halo exchange, z filter, primitive refresh, and final halo
  exchange before GPU `rkfirst` statistics.
- [x] Restore the unfiltered complete RK state and run the normal boundary
  preparation again before time integration.
- [x] Pass ten-step SBLI same-phase field and online statistics at the existing
  `1e-10` combined tolerance; `massflux` maximum absolute error is
  `5.0293103015519591e-13`.
- [x] Confirm peak SBLI memory remains 1,510 MiB after replacing the initial
  dedicated snapshot allocation with `qsave_d` reuse.

### Task 10: Reprofile The Corrected P2-1B Baseline

- [x] Freeze the corrected P2-1B executable and timing TSV files.
- [x] Recollect separate shock activity for Shu-Osher `256x64x32` and SBLI
  `256x192x32`.
- [x] Recollect NP=1 Nsight Systems profiles for both cases.
- [x] Recollect full Nsight Compute reports for SBLI x/y/z characteristic flux
  and Shu-Osher x characteristic flux.
- [x] Confirm sensor plus expanded-mask kernels remain below a `3%`
  complete-RK Amdahl upper bound.

### Task 11: Screen P2-1C Sequential Split Storage

**Scope:** Reuse one thread-local `split_flux(5,7)` array for positive and
negative fluxes while preserving Roe state, reconstruction, physical-boundary
clamping, FP64, fixed blocks, and explicit synchronization.

- [x] Add and observe failing contracts for one-array storage and positive then
  negative reconstruction order.
- [x] Pass one-step Shu-Osher and SBLI CPU/GPU field and statistics gates.
- [x] Run five complete-RK repeats for both formal cases.
- [x] Confirm with Nsight Compute that registers remain 128/thread and local
  spilling increases from `15,059,748` to `18,007,902` in the SBLI y kernel.
- [x] Reject the candidate because same-sample complete-RK gains are only
  `1.975%` for Shu-Osher and `1.208%` for SBLI, below the required `3%`.
- [x] Remove candidate source and tests while retaining timing, NCU, and source
  evidence under the candidate output directory.

### Task 12: Close Sensor/Mask, Characteristic Flux, And Halo-Wait Classes

- [x] Extend the P2 profile driver for NP, topology, visible-GPU list, CUDA/MPI
  trace selection, and reproducible MPI reports.
- [x] Add a tested SQLite timeline summarizer that pairs raw and expanded
  sensor kernels per GPU and counts only in-window MPI events.
- [x] Measure NP=2 y-slab sensor-halo critical-path upper bounds of `3.488%`
  for Shu-Osher and `1.421%` for SBLI.
- [x] Classify the P2-local sensor pack/unpack kernels as below `3%`; retain the
  host staging and MPI portion as P3 evidence rather than changing the
  communication backend in P2.
- [x] Pass the top-level CPU/GPU builds, 54 Python tests, all shell syntax
  checks, Sod/Shu-Osher/SBLI ten-step gates, Shu-Osher NP=2 x/y/z and NP=8
  `2x2x2`, SBLI wall statistics, and Compute Sanitizer with zero errors.
- [x] Close P2 with P2-1B as the only retained shock-path kernel optimization.
