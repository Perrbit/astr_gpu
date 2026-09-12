# GPU mixed-precision MP0-MP1 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the runtime precision contract and validate the first FP32 temporary workspace for periodic physical-space `543e` reconstruction without changing the FP64 authoritative solver state.

**Architecture:** A new precision-mode module owns MPI-consistent runtime selection. `commarray_gpu` allocates either the existing FP64 scalar flux workspace or a new FP32 scalar flux workspace for eligible periodic physical-space reconstruction. FP64 reconstruction arithmetic writes directly to FP32 storage, and the flux-difference kernels convert workspace values back to FP64 while accumulating into `qrhs_d`.

**Tech Stack:** NVHPC CUDA Fortran, MPI, CMake, pytest, existing ASTR GPU validation scripts.

## Global constraints

- Work only on `feature/gpu_dev`; do not create a worktree.
- Build through `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Keep `q_d`, `qrhs_d`, `qsave_d`, primitive fields, geometry, halo data, boundaries, Runge-Kutta updates, statistics, and output in FP64.
- Keep `ASTR_GPU_PRECISION_MODE=fp64` as the default; `mixed_workspace` is opt-in.
- Do not use global `-r4`, fast-math, FP16, BF16, TF32, or Tensor Cores.
- Do not add whole-field conversion kernels or host transfers.
- Preserve explicit synchronization after every kernel.
- Stop on invalid or MPI-inconsistent precision modes.
- Do not stage, commit, or push during this implementation unless separately requested.

---

### Task 1: Runtime precision contract

**Files:**

- Create: `src_gpu/precision_mode_gpu.cuf`
- Create: `tests/gpu_validation/precision_mode_setup_test.cuf`
- Create: `tests/gpu_validation/test_mixed_precision_workspace_contract.py`
- Modify: `src/CMakeLists.txt`

**Interfaces:**

- Produces: `configure_gpu_precision_mode()`
- Produces: `gpu_mixed_workspace_requested() -> logical`
- Produces: `gpu_precision_mode_name() -> character(len=32)`

- [x] **Step 1: Write source-contract tests that require the two accepted values, FP64 default, two MPI reductions, hard invalid-mode failure, and CMake source ordering.**
- [x] **Step 2: Run the focused pytest and verify it fails because `precision_mode_gpu.cuf` is absent.**
- [x] **Step 3: Implement the MPI-consistent runtime module and its probe target.**
- [x] **Step 4: Run pytest, configure the CUDA build, build the probe, and verify default, `fp64`, `mixed_workspace`, invalid, and rank-inconsistent behavior.**

### Task 2: Precision-aware workspace allocation

**Files:**

- Modify: `src_gpu/commarray_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_workspace_contract.py`

**Interfaces:**

- Produces: `real(4), allocatable, device :: flux_work_sp_d(:,:,:)`
- Produces: `gpu_mixed_flux_workspace_enabled() -> logical`
- Produces log records `ASTR_GPU_PRECISION_MODE`, `ASTR_GPU_ACTIVE_MIXED_WORKSPACE`, and `ASTR_GPU_FLUX_WORKSPACE_BYTES`

- [x] **Step 1: Extend the test to require mutually exclusive FP64/FP32 allocation and correct 8-byte/4-byte accounting.**
- [x] **Step 2: Run the test and verify allocation assertions fail.**
- [x] **Step 3: Allocate FP32 only when `mixed_workspace`, `conschm=543e`, all three directions are homogeneous, and `lchardecomp=f`; otherwise retain FP64.**
- [x] **Step 4: Rebuild and verify the default path reports FP64 while eligible mixed mode reports half-sized scalar workspace storage.**

### Task 3: Periodic physical-space FP32 flux kernels

**Files:**

- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `src_gpu/case_capability_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_workspace_contract.py`

**Interfaces:**

- Produces: `_sp` variants of periodic x/y/z reconstruction and RHS kernels.
- Produces: `gpu_tgv_explicit_upwind_supported() -> logical` for periodic, non-reacting, `543e/643e`, WENO7 or MP7, no filter, no diffusion, no characteristic decomposition.

- [x] **Step 1: Extend the test to require direct FP64-to-FP32 stores, explicit FP32-to-FP64 RHS reads, no conversion kernel, and one synchronization after every `_sp` launch.**
- [x] **Step 2: Run the test and verify kernel and dispatch assertions fail.**
- [x] **Step 3: Add six periodic `_sp` kernels; retain reconstruction arithmetic in FP64 and round only at workspace stores.**
- [x] **Step 4: Dispatch the `_sp` kernels only when `gpu_mixed_flux_workspace_enabled()` is true; preserve existing physical-boundary and characteristic paths.**
- [x] **Step 5: Admit the smooth periodic TGV `543e` gate for both FP64 and mixed modes.**
- [x] **Step 6: Build the complete CPU and GPU binaries and rerun the focused contract test.**

### Task 4: Numerical and performance admission

**Files:**

- Create: `tests/gpu_validation/run_tgv_upwind_mixed_precision_compare.sh`
- Create: `tests/gpu_validation/run_tgv_upwind_mixed_precision_benchmark.sh`
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `docs/superpowers/specs/2026-09-12-gpu-mixed-precision-workspace-design.md`

**Interfaces:**

- Compare three trajectories: CPU FP64, GPU FP64 workspace, and GPU mixed workspace.
- Report field $L_\infty$, $L_2$, relative norms, statistics, workspace bytes, and complete-step timing.

- [x] **Step 1: Write the comparison driver with a one-step field gate and a longer energy/statistics gate.**
- [x] **Step 2: Verify CPU versus GPU FP64 first; stop immediately if the new TGV `543e` route is not a valid FP64 oracle.**
- [x] **Step 3: Run mixed mode and establish evidence-based provisional tolerances from observed FP32 workspace error rather than reusing `1e-10`.**
- [x] **Step 4: Run Compute Sanitizer on the mixed periodic path.**
- [x] **Step 5: Run at least five FP64/mixed complete-step repetitions and report median, spread, workspace bytes, and peak device memory without imposing a fixed speedup threshold.**
- [x] **Step 6: Update documentation with observed evidence and classify the candidate as promoted, viable but not promoted, or rejected.**
