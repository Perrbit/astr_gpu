# ASTR CUDA Fortran Porting Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ASTR from CPU Fortran/MPI toward CUDA Fortran by first proving a numerically equivalent single-rank Taylor-Green Vortex GPU path.

**Architecture:** The first-stage GPU path is deliberately narrow: TGV, one MPI rank, one GPU, five conservative flow variables, explicit sixth-order central difference, explicit tenth-order central filter, and geometry-aware stencils using `dxi_d` and `jacob_d`. Correctness comes before performance: every kernel synchronizes and checks errors until L1/L2 validation passes.

**Tech Stack:** Fortran 90/95, MPI, HDF5, CUDA Fortran, NVHPC 26.1, CMake, Compute Sanitizer, Nsight Systems, Nsight Compute.

## Global Constraints

- First-stage acceptance boundary: `examples/Taylor_Green_Vortex`, one MPI rank, one GPU, `num_species=0`, `turbmode=none`.
- First-stage RHS covers only `q(1:5)`: density, three momentum components, and total energy.
- Derivatives use explicit sixth-order central difference only; compact finite-difference line solves are out of scope.
- Filtering uses explicit tenth-order central filter only; compact filter line solves are out of scope.
- Directional thread blocks are fixed for first-stage kernels: x `(512,1,1)`, y `(32,16,1)`, z `(64,1,8)`.
- Gradient, convection, and diffusion kernels use `dxi_d` and `jacob_d`; do not hard-code TGV `dx/dy/dz` spacings.
- RK must preserve the CPU `qsave=q*jacob` and `q=qsave/jacob` semantics; do not assume `jacob=1`.
- Every first-stage kernel launch must be followed by explicit synchronization and error checking.
- Crash check is detect-only: report invalid state and stop; do not repair the field.
- Runtime input files must use LF line endings; CRLF can make NVHPC read `\r` as part of string tokens.
- `use_gpu` is a runtime input-file option, not a CMake option.
- CPU/GPU code must stay separated: `src/` may only contain the runtime flag, input parsing/broadcast, top-level GPU lifecycle hooks, and one time-integration dispatch. CUDA arrays, kernels, synchronization, launch geometry, and GPU numerical algorithms must live under `src_gpu/`.
- `src/` files must depend on a narrow GPU facade only; they must not import individual GPU implementation modules such as GPU array, variable, solver, filter, derivative, or crashcheck modules.
- First-stage validation is single MPI rank only. CUDA-aware MPI and multi-rank halo exchange are later phases.
- A/B filter validation is required: A uses `lfilter=f`; B restores default TGV with `lfilter=t`.
- This plan does not modify source code by itself; it defines the implementation sequence.

---

## 1. Confirmed Project Language

The canonical glossary is in [`CONTEXT.md`](../CONTEXT.md).

Important terms:

- **First-stage acceptance boundary**: single-rank, single-GPU TGV numerical equivalence target.
- **Explicit sixth-order central difference**: derivative scheme, no tridiagonal solve.
- **Explicit tenth-order central filter**: filter scheme, no tridiagonal solve.
- **Directional thread block**: x `(512,1,1)`, y `(32,16,1)`, z `(64,1,8)`.
- **First-stage conservative variable set**: `q(1:5)` only.
- **Single-rank GPU validation**: no cross-rank halo exchange in first-stage acceptance.
- **Tiered numerical tolerance**: stricter algebra/copy tolerance, looser stencil tolerance.
- **Synchronous correctness kernel**: synchronize and check after every kernel.
- **Geometry-aware stencil path**: use `dxi` and `jacob`, not hard-coded uniform spacing.
- **Detect-only crash check**: stop on invalid state; no crashfix.

## 2. Confirmed ADRs

- [`0001-use-explicit-central-schemes-for-first-stage-gpu-port.md`](../docs/adr/0001-use-explicit-central-schemes-for-first-stage-gpu-port.md)
- [`0002-use-directional-thread-blocks.md`](../docs/adr/0002-use-directional-thread-blocks.md)
- [`0003-limit-first-stage-rhs-to-five-flow-variables.md`](../docs/adr/0003-limit-first-stage-rhs-to-five-flow-variables.md)
- [`0004-validate-single-rank-before-gpu-mpi.md`](../docs/adr/0004-validate-single-rank-before-gpu-mpi.md)
- [`0005-use-tiered-numerical-tolerances.md`](../docs/adr/0005-use-tiered-numerical-tolerances.md)
- [`0006-synchronize-after-every-kernel-in-first-stage.md`](../docs/adr/0006-synchronize-after-every-kernel-in-first-stage.md)
- [`0007-keep-geometry-aware-stencils-in-first-stage.md`](../docs/adr/0007-keep-geometry-aware-stencils-in-first-stage.md)
- [`0008-use-detect-only-crash-check-in-first-stage.md`](../docs/adr/0008-use-detect-only-crash-check-in-first-stage.md)
- [`0009-require-lf-runtime-input-files.md`](../docs/adr/0009-require-lf-runtime-input-files.md)
- [`0010-keep-use-gpu-as-runtime-input.md`](../docs/adr/0010-keep-use-gpu-as-runtime-input.md)

## 3. Current Technical Diagnosis

The current GPU path is a runnable first-stage smoke path, but it is not yet an accepted CFD result path.

Known blockers:

- CPU `rhscal` computes `qrhs = -conv + diff`; the GPU path must preserve this sign.
- CPU RK uses `qsave=q*jacob` and divides by `jacob`; GPU RK must carry `jacob_d` from the first numerical implementation.
- GPU path now uses first-stage single-rank GPU `qswap` for periodic halo fill plus periodic-plane averaging. Cross-rank GPU halo exchange is still out of scope.
- Current project scope does not include GPU porting of file output, checkpoint, or HDF5 `flowfield` writers. Treat these CPU-owned output paths as explicit boundaries, not as near-term migration blockers.
- Default TGV single-rank smoke now runs on GPU with explicit filter and diffusion enabled. Native `flowstate.dat` statistics and full-field `flowfield.h5` primitive/reconstructed-`q` `L_inf/L2` metrics pass for 10-step TGV with and without filtering at `1e-10` tolerance.
- The TGV GPU time loop now keeps full flow variables resident across kernels and across time steps. `flowstate.dat` is written directly from GPU reductions; full-field D2H is limited to CPU-owned output paths such as checkpoint/`flowfield.h5` writing, which are intentionally out of current GPU-porting scope.
- First GPU-resident statistics slice is implemented for TGV kinetic energy: `src_gpu/statistic_gpu.cuf` computes `kenergy` from `rho_d/vel_d`, copies only partial sums/scalar data to host, writes native `flowstate.dat` and `gpu_kenergy.dat`, and matches CPU `flowstate.dat` `kenergy` within `4.67e-14` on the 10-step filtered TGV validation.
- GPU-resident TGV enstrophy scalar reduction is implemented from `rho_d/dvel_d`, writes native `flowstate.dat` and `gpu_enstophy.dat`, and matches CPU `flowstate.dat` `enstophy` within `4.66e-14` on the 10-step filtered TGV validation.
- GPU-resident TGV dissipation scalar reduction is implemented from `tmp_d/dvel_d`, writes native `flowstate.dat` and `gpu_dissipation.dat`, and matches CPU `flowstate.dat` `dissipation` within `1.76e-16` on the 10-step filtered TGV validation.
- First-stage GPU `gradcal` now computes TGV velocity gradients `dvel_d` in `src_gpu/gradcal_gpu.cuf`; CPU/GPU `dvel` comparison passes with max error `2.22e-16` on the 10-step filtered TGV validation. Species and turbulence gradients remain out of first-stage scope.
- Runtime GPU use has been confirmed with `nvitop`: during a 10-step TGV GPU run, GPU 0 showed the ASTR compute process, about `1018-1129 MiB` GPU memory use, nonzero `%SM`, and GPU utilization peaking around `78%`.

Therefore, no performance result is accepted until L1/L2 numerical validation passes.

## 4. Target File Responsibilities

### Existing CPU Reference Files

- `src/mainloop.F90`: CPU RK sequence and reference order of filter, boundary, gradient, RHS, RK, updatefvar.
- `src/solver.F90`: CPU `rhscal`, `convrsdcal6`, and `diffrsdcal6` reference.
- `src/comsolver.F90`: CPU `gradcal` and filter setup reference.
- `src/fludyna.F90`: CPU primitive/conservative conversion reference.
- `src/filter.F90`: CPU explicit filter coefficients and reference behavior.
- `src/commarray.F90`: CPU array shapes and halo conventions.
- `src/commvar.F90`: runtime flags and physical/numerical parameters.

### GPU Implementation Files

- `src_gpu/commarray_gpu.cuf`: device arrays and copy routines.
- `src_gpu/commvar_gpu.cuf`: device constants and RK coefficients.
- `src_gpu/gpu_runtime.cuf`: the only GPU facade imported by CPU-side `src/` files; wraps GPU lifecycle and time-integration entry points.
- `src_gpu/fludyna_gpu.cuf`: GPU `q <-> rho/vel/prs/tmp` conversion.
- `src_gpu/bc_gpu.cuf`: single-rank periodic halo fill.
- `src_gpu/filter_gpu.cuf`: explicit tenth-order filter kernels.
- `src_gpu/gradcal_gpu.cuf`: explicit sixth-order gradient kernels.
- `src_gpu/solver_gpu.cuf`: convection, diffusion, and RHS assembly.
- `src_gpu/commfunc_gpu.cuf`: RK kernels and common device utilities.
- `src_gpu/crashcheck_gpu.cuf`: detect-only crash checking.
- `src_gpu/mainloop_gpu.cuf`: GPU RK driver.

### Validation and Diagnostic Files To Add

- `documents/ASTR_CUDA_FORTRAN_PORTING_ROADMAP.md`: this plan.
- `documents/GPU_VALIDATION_MATRIX.md`: validation table with pass/fail, tolerances, and commands.
- `tests/gpu_validation/README.md`: how to run L1/L2 validation.
- `tests/gpu_validation/compare_flowstate.py`: compare native CPU/GPU `flowstate.dat` statistics.
- `tests/gpu_validation/compare_flowfield_h5.py`: compare CPU/GPU HDF5 primitive fields and reconstructed `q`.
- `tests/gpu_validation/run_tgv_stats_compare.sh`: run isolated CPU/GPU native-statistics checks.
- `tests/gpu_validation/run_tgv_field_compare.sh`: run isolated CPU/GPU full-field HDF5 checks.

## 5. Validation Standards

### L0: Build and Smoke

L0 only proves the executable builds and starts.

Required checks:

```bash
cmake -B build_cpu -DCMAKE_Fortran_COMPILER=mpif90 .
cmake --build build_cpu -j4
cmake -B build_gpu -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON .
cmake --build build_gpu -j4
```

Expected result:

- Both builds exit with code 0.
- GPU build includes `src_gpu/*.cuf`.
- No numerical correctness claim is made from L0.

### L1: Module-Level CPU/GPU Diff

Algebra/copy checks:

```text
abs_err <= max(1e-14, 1e-12 * abs(cpu_ref))
```

Stencil/RHS checks:

```text
abs_err <= max(1e-12, 1e-10 * abs(cpu_ref))
```

Required modules:

1. `updatefvar`
2. periodic halo fill
3. explicit tenth-order filter
4. `gradcal`
5. convection only
6. diffusion only
7. full RHS: `-conv + diff`
8. RK3 single substep

### L2: Integration Diff

Required checks:

- one RK substep
- one full step
- ten full steps
- `min(rho)`, `min(prs)`, `min(tmp)`
- `L_inf(q_gpu-q_cpu)`
- `L2(q_gpu-q_cpu)`

L2 must be run twice:

- A path: `lfilter=f`
- B path: `lfilter=t`

### L3: TGV Physical Diagnostics

Only after L1/L2 pass:

- total kinetic energy
- kinetic energy decay rate
- vorticity diagnostic
- density, pressure, and temperature extrema

Every diagnostic plot must cite:

- generating script
- input field path
- output image path
- exact CPU/GPU run configuration

### L4: Performance

Only after L1/L2/L3 pass:

- kernel-only time
- H2D/D2H time
- synchronization cost
- total wall time
- CPU baseline compiler/rank/thread settings

## 6. Phase Plan

### Phase 0: Freeze Scope and Inputs

**Files:**

- Create: `.gitattributes`
- Modify: `examples/**/datin/input.*`
- Modify: `examples/**/datin/controller`
- Modify: `examples/Taylor_Green_Vortex/datin/input.tgv`
- Modify: `examples/Taylor_Green_Vortex/datin/controller`
- Create: `documents/GPU_VALIDATION_MATRIX.md`
- Create: `tests/gpu_validation/README.md`

**Interfaces:**

- Consumes: confirmed ADRs and `CONTEXT.md`.
- Produces: LF-normalized runtime inputs and stable TGV A/B validation inputs.

- [x] **Step 0: Enforce LF line endings for runtime inputs**

Create `.gitattributes`:

```gitattributes
examples/**/datin/input.* text eol=lf
examples/**/datin/controller text eol=lf
*.md text eol=lf
*.sh text eol=lf
```

Normalize existing runtime inputs:

```bash
find examples -type f \( -name 'input.*' -o -name 'controller' \) -print0 \
  | xargs -0 perl -pi -e 's/\r$//'
```

Verify:

```bash
find examples -type f \( -name 'input.*' -o -name 'controller' \) -print0 \
  | xargs -0 file | rg 'CRLF'
```

Expected: no output.

- [x] **Step 1: Create validation matrix document**

Create `documents/GPU_VALIDATION_MATRIX.md` with this table:

```markdown
# GPU Validation Matrix

| ID | Case | MPI ranks | GPU | lfilter | Module | Tolerance | Status | Evidence |
|---|---|---:|---:|---|---|---|---|---|
| L0-LF | Runtime inputs | n/a | n/a | n/a | line endings | no CRLF | pending | |
| L0-CPU | TGV build | 1 | 0 | n/a | build_cpu | exit 0 | pending | |
| L0-GPU | TGV build | 1 | 1 | n/a | build_gpu | exit 0 | pending | |
| L1-FVAR | TGV fixed field | 1 | 1 | n/a | updatefvar | max(1e-14,1e-12*ref) | pending | |
| L1-HALO | TGV fixed field | 1 | 1 | n/a | periodic halo | max(1e-14,1e-12*ref) | pending | |
| L1-FILTER-A | TGV fixed field | 1 | 1 | f | filter skipped | n/a | pending | |
| L1-FILTER-B | TGV fixed field | 1 | 1 | t | explicit filter | max(1e-12,1e-10*ref) | pending | |
| L1-GRAD | TGV fixed field | 1 | 1 | f/t | gradcal `dvel` | max(1e-12,1e-10*ref) | pass | first-stage TGV `dvel` max error `2.22e-16`; species/turbulence gradients out of scope |
| L1-CONV | TGV fixed field | 1 | 1 | f/t | -conv | max(1e-12,1e-10*ref) | pending | |
| L1-DIFF | TGV fixed field | 1 | 1 | f/t | diff | max(1e-12,1e-10*ref) | pending | |
| L1-RHS | TGV fixed field | 1 | 1 | f/t | -conv+diff | max(1e-12,1e-10*ref) | pending | |
| L1-RK | TGV fixed field | 1 | 1 | f/t | RK3 substep | max(1e-12,1e-10*ref) | pending | |
| L2-A | TGV | 1 | 1 | f | 10 step integration | `flowfield.h5` primitive/q `L_inf` <= `1e-10` | pass | no-filter full-field compare passed; reconstructed `q5` max error `2.27e-13` |
| L2-B | TGV | 1 | 1 | t | 10 step integration | `flowfield.h5` primitive/q `L_inf` <= `1e-10` | pass | filter full-field compare passed; reconstructed `q5` max error `8.73e-12` |
| L3 | TGV | 1 | 1 | t | native physical diagnostics | EPS/JPEG generated; CPU/GPU statistic deltas recorded | pass | `kenergy` and `enstophy` max abs deltas `0`; `dissipation` max abs delta `1.00e-16` |
```

- [x] **Step 2: Create validation README**

Create `tests/gpu_validation/README.md`:

```markdown
# GPU Validation

This directory contains validation drivers for the first-stage ASTR CUDA Fortran port.

First-stage scope:

- Taylor-Green Vortex only
- one MPI rank
- one GPU
- q(1:5) only
- explicit sixth-order central difference
- explicit tenth-order central filter
- detect-only crash check

Validation order:

1. L0 build and smoke
2. L1 module-level CPU/GPU field diff
3. L2 one-step and ten-step integration diff
4. L3 TGV physical diagnostics
5. L4 performance profiling

Do not report GPU speedup until L1, L2, and L3 pass.
```

- [x] **Step 3: Define A/B TGV input policy**

Record in `documents/GPU_VALIDATION_MATRIX.md`:

```markdown
## TGV Filter Policy

Path A sets `lfilter=f` in both CPU and GPU runs. It validates the minimum RHS/RK/updatefvar loop.

Path B sets `lfilter=t` in both CPU and GPU runs. It validates the default TGV path after explicit GPU filter is implemented.
```

- [x] **Step 4: Verify no source files changed in Phase 0**

Run:

```bash
git status --short src src_gpu
```

Expected:

```text
```

If output is not empty, list the touched source files in the validation matrix before proceeding.

Executed result:

```text
git status --short src src_gpu
```

produced no output after `.gitattributes` was narrowed to runtime inputs and `src/` was restored to HEAD.

### Phase 1A: Bootstrap GPU Scaffold From CPU Baseline

The current CPU baseline does not contain `src_gpu/`, a CUDA-capable build switch, or GPU source integration. This phase must run before Phase 1.

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `src/CMakeLists.txt`
- Modify: `src/commvar.F90`
- Modify: `src/readwrite.F90`
- Modify: `src/astr.F90`
- Create: `src_gpu/gpu_check.cuf`
- Create: `src_gpu/commarray_gpu.cuf`
- Create: `src_gpu/commvar_gpu.cuf`
- Create: `src_gpu/mainloop_gpu.cuf`
- Create: `src_gpu/gpu_runtime.cuf`

**Interfaces:**

- Produces: `ASTR_WITH_CUDA` build option, `_CUDA` compile definition, a narrow `gpu_runtime` facade, minimal GPU initialization hooks, and a CUDA-capable binary that preserves CPU behavior when runtime `use_gpu=f`.
- Consumes: CPU baseline modules and first-stage scope constraints.

- [x] **Step 1: Write a failing GPU scaffold configure/build check**

Run:

```bash
cmake -B build_gpu -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON .
cmake --build build_gpu -j4
```

Expected before implementation: configure or build does not provide a working GPU-enabled target with `src_gpu/*.cuf`.

Observed before implementation: `ASTR_WITH_CUDA` was only an unused CMake cache variable and no `src_gpu` source directory existed.

- [x] **Step 2: Add CUDA-capable CMake option and source list**

Add an `ASTR_WITH_CUDA` option. When enabled under NVHPC, add `-cuda -gpu=cc89 -gpu=rdc`, define `_CUDA`, and compile the first minimal `src_gpu/*.cuf` modules. Do not name this option `USE_GPU`.

- [x] **Step 3: Add minimal `use_gpu` runtime flag**

Add `logical :: use_gpu = .false.` to `commvar.F90` and parse it from the existing TGV input parameter line. This is a runtime switch only. Preserve CPU behavior when the flag is absent or false.

- [x] **Step 4: Add minimal GPU modules**

Create GPU modules that compile and expose only no-op initialization and synchronization helpers. Do not implement numerical kernels in this phase.

The CPU-side files import only `gpu_runtime`; direct imports of `commarray_gpu`, `commvar_gpu`, `mainloop_gpu`, solver, filter, derivative, crashcheck, CUDA launch, or synchronization details remain confined to `src_gpu/`.

- [x] **Step 5: Verify CPU build still passes**

Run:

```bash
cmake -B build_cpu -DCMAKE_Fortran_COMPILER=mpif90 .
cmake --build build_cpu -j4
```

Expected: build exits with code 0.

Executed result: `cmake --build build_cpu_probe -j4` exited with code 0 and built target `astr`.

- [x] **Step 6: Verify GPU scaffold build passes**

Run:

```bash
cmake -B build_gpu -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON .
cmake --build build_gpu -j4
```

Expected: build exits with code 0.

Executed result: `cmake --build build_gpu_probe -j4` exited with code 0 and built target `astr`; `build_gpu_probe/src/CMakeFiles/astr.dir/__/src_gpu/*.cuf.o` contains `commarray_gpu`, `commvar_gpu`, `gpu_check`, and `mainloop_gpu`.

- [x] **Step 7: Verify runtime `use_gpu=t` dispatch reaches GPU scaffold**

Run a `/tmp` copy of `examples/Taylor_Green_Vortex/datin/input.tgv` with the ninth parameter set to `use_gpu=t`.

Expected: TGV initialization completes, CPU RK is bypassed, and execution stops at the explicit scaffold message because numerical GPU kernels are not implemented yet.

Executed result: `mpirun -np 1 /home/dell/workspace/astr_gpu/build_gpu_probe/bin/astr run datin/input.tgv` printed `** 3-D TGV initialised.` and stopped at `GPU numerical kernels are not implemented after Phase 1A scaffold`.

### Phase 1: Correctness Infrastructure

**Files:**

- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `src_gpu/filter_gpu.cuf`
- Modify: `src_gpu/gradcal_gpu.cuf`
- Modify: `src_gpu/crashcheck_gpu.cuf`
- Create: `src_gpu/gpu_check.cuf`

**Interfaces:**

- Produces: `gpu_check_status(label, istat)` and `sync_after_kernel(label)`.
- Consumes: CUDA Fortran `cudaDeviceSynchronize()` and `cudaGetLastError()`.

- [ ] **Step 1: Add kernel error-checking helper**

Create `src_gpu/gpu_check.cuf`:

```fortran
module gpu_check
  use cudafor
  implicit none
contains
  subroutine sync_after_kernel(label)
    character(len=*), intent(in) :: label
    integer :: istat

    istat = cudaGetLastError()
    if (istat /= cudaSuccess) then
      print *, 'GPU kernel launch failed: ', trim(label), istat
      stop 'GPU kernel launch failed'
    endif

    istat = cudaDeviceSynchronize()
    if (istat /= cudaSuccess) then
      print *, 'GPU kernel synchronize failed: ', trim(label), istat
      stop 'GPU kernel synchronize failed'
    endif
  end subroutine sync_after_kernel
end module gpu_check
```

- [ ] **Step 2: Add helper to GPU build list**

Modify `src/CMakeLists.txt` so `ASTR_GPU_SOURCES` includes:

```cmake
${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_check.cuf
```

Place it before modules that use `gpu_check`.

- [ ] **Step 3: Replace raw synchronizations**

In every first-stage GPU wrapper, replace:

```fortran
istat = cudaDeviceSynchronize()
```

after a kernel with:

```fortran
call sync_after_kernel('module_name.kernel_name')
```

The label must name the wrapper and kernel, for example:

```fortran
call sync_after_kernel('rhscal_gpu.conv_i_kernel')
```

- [ ] **Step 4: Build GPU**

Run:

```bash
cmake -B build_gpu -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON .
cmake --build build_gpu -j4
```

Expected: build exits with code 0.

### Phase 2: Enforce First-Stage Numerical Scope

**Files:**

- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `src_gpu/commfunc_gpu.cuf`
- Modify: `src_gpu/commvar_gpu.cuf`

**Interfaces:**

- Consumes: TGV input values from `commvar`.
- Produces: fail-fast checks for unsupported first-stage configurations.

- [x] **Step 1: Add first-stage scope guard**

In `mainloop_gpu.cuf`, before entering the RK loop, add a guard equivalent to:

```fortran
if (numq /= 5) stop 'GPU first-stage supports numq=5 only'
if (ndims /= 3) stop 'GPU first-stage supports 3D TGV only'
```

Also reject unsupported chemistry and turbulence flags by importing the existing `num_species`, `num_modequ`, and `turbmode` values from `commvar`:

```fortran
if (num_species /= 0) stop 'GPU first-stage supports num_species=0 only'
if (num_modequ /= 0) stop 'GPU first-stage supports num_modequ=0 only'
if (trim(turbmode) /= 'none') stop 'GPU first-stage supports turbmode=none only'
```

- [x] **Step 2: Add single-rank guard**

Use the existing `mpisize` value from `parallel.F90`.

Runtime behavior:

```fortran
use parallel, only: mpisize

if (mpisize /= 1) stop 'GPU first-stage supports one MPI rank only'
```

- [x] **Step 3: Reject the temporary `jacob=1` RK assumption**

Do not implement a GPU RK path that assumes `jacob=1`. A scaffold smoke test on `examples/Taylor_Green_Vortex` measured:

```text
max_abs_jacob_minus_one ~= 0.9998817204411304
```

Therefore the first numerical GPU RK implementation must use the same geometry semantics as CPU RK:

```fortran
qsave = q * jacob
q = qsave / jacob
```

- [x] **Step 4: Build and run TGV smoke**

Run:

```bash
cmake --build build_gpu -j4
cd examples/Taylor_Green_Vortex
mpirun -np 1 ../../build_gpu/bin/astr run datin/input.tgv
```

Expected:

- If input is within first-stage scope, run starts.
- If unsupported settings are present, run stops with the explicit guard message.

Executed result: CUDA-capable TGV run with runtime `use_gpu=t` entered the GPU scaffold after initialization and stopped at the expected `GPU numerical kernels are not implemented after Phase 1A scaffold` message.

### Phase 3: Explicit Sixth-Order Difference Kernels

**Files:**

- Modify: `src_gpu/derivative_gpu.cuf`
- Modify: `src_gpu/gradcal_gpu.cuf`
- Modify: `src_gpu/solver_gpu.cuf`

**Interfaces:**

- Produces: x/y/z explicit sixth-order stencil kernels using directional thread blocks.
- Consumes: `dxi_d`, `jacob_d`, `q_d`, `vel_d`, `prs_d`, `tmp_d`.

- [ ] **Step 1: Remove compact derivative from first-stage path**

Ensure first-stage derivative wrappers do not call Thomas solvers or compact scheme arrays.

Accepted behavior:

```text
difscheme compact requested in GPU first-stage -> stop with explicit message
```

- [ ] **Step 2: Use directional thread blocks**

Set launch block shapes:

```fortran
block_x = dim3(512, 1, 1)
block_y = dim3(32, 16, 1)
block_z = dim3(64, 1, 8)
```

Use these in derivative, gradient, convection, diffusion, and filter kernels according to direction.

Implemented smoke slice: `src_gpu/solver_gpu.cuf` now launches x `(512,1,1)`, y `(32,16,1)`, and z `(64,1,8)` direction kernels for convection, diffusion divergence, and ping-pong explicit filtering. L1 derivative/gradient/RHS comparison remains pending.

- [ ] **Step 3: Implement x-direction explicit sixth-order stencil**

Stencil formula:

```text
df_i = 0.75*(f(i+1)-f(i-1))
     - 0.15*(f(i+2)-f(i-2))
     + (1/60)*(f(i+3)-f(i-3))
```

Use halo values for `i-3:i+3`. Do not solve a linear system.

- [ ] **Step 4: Implement y and z direction stencils**

Use the same coefficients as x. Only the index direction changes.

- [ ] **Step 5: Verify L1 gradient**

Run the L1 gradient comparison driver.

Expected:

```text
max_abs_err <= max(1e-12, 1e-10*abs(cpu_ref))
```

### Phase 4: RHS Sign and Five-Variable Flux

**Files:**

- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `tests/gpu_validation/compare_flowfield_h5.py`
- Modify: future `tests/gpu_validation/run_tgv_l1.sh`

**Interfaces:**

- Produces: GPU RHS equivalent to CPU `qrhs = -conv + diff`.
- Consumes: CPU reference field dumps.

- [ ] **Step 1: Make RHS sign explicit**

In `rhscal_gpu`, implement one of these two accepted forms:

Accepted form A:

```text
conv kernels accumulate +conv
negate kernel converts qrhs to -conv
diff kernels add +diff
```

Accepted form B:

```text
conv kernels accumulate -conv directly
diff kernels add +diff
```

Do not leave RHS sign implicit.

Implemented smoke slice: GPU convection accumulates `-conv` directly, and GPU diffusion adds `+diff`, matching CPU `qrhs = -conv + diff` at the sign-convention level. Elementwise RHS validation remains pending.

- [ ] **Step 2: Restrict flux to q(1:5)**

Convection and diffusion GPU RHS must only write:

```fortran
qrhs_d(:,:,:,1:5)
```

If `numq /= 5`, stop before launching RHS kernels.

- [ ] **Step 3: Compare convection only**

Run CPU/GPU with diffusion disabled and filter disabled.

Expected comparison:

```text
gpu_rhs == -cpu_conv
```

within stencil tolerance.

- [ ] **Step 4: Compare diffusion only**

Run CPU/GPU with convection disabled by test harness or diagnostic mode and diffusion enabled.

Expected comparison:

```text
gpu_rhs == cpu_diff
```

within stencil tolerance.

- [ ] **Step 5: Compare full RHS**

Run CPU/GPU with convection and diffusion enabled.

Expected comparison:

```text
gpu_rhs == -cpu_conv + cpu_diff
```

within stencil tolerance.

### Phase 5: Explicit Tenth-Order Filter

**Files:**

- Modify: `src_gpu/filter_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: future `tests/gpu_validation/run_tgv_l1.sh`

**Interfaces:**

- Produces: GPU explicit tenth-order central filter, no compact solve.
- Consumes: CPU explicit filter coefficients from `filter.F90`.

- [ ] **Step 1: Disable compact filter path in first-stage GPU**

If compact filter is requested, stop with:

```text
GPU first-stage supports explicit tenth-order filter only
```

- [ ] **Step 2: Implement explicit tenth-order central filter**

Use the explicit filter coefficients from CPU `filter.F90`.

The GPU filter must not call:

```fortran
tridiagonal_thomas_solver_gpu
```

Implemented smoke slice: GPU filter uses explicit coefficients matching `filter.F90`'s `coef10e` and ping-pong arrays (`q_d`/`qwork_d`): x writes `q_d -> qwork_d`, y writes `qwork_d -> q_d`, z writes `q_d -> qwork_d`, then copies `qwork_d -> q_d` as the standard post-filter state. Elementwise filter validation remains pending.

- [ ] **Step 3: Insert filter in GPU RK sequence**

Match CPU order:

```text
for each RK substep:
  filter if lfilter
  boundary/halo
  gradcal
  rhscal
  RK update
  updatefvar
  crashcheck
```

- [ ] **Step 4: Run A/B validation**

Path A:

```text
lfilter=f
```

Path B:

```text
lfilter=t
```

Expected: both paths pass L1/L2 before moving to diagnostics.

### Phase 6: Detect-Only Crash Check

**Files:**

- Modify: `src_gpu/crashcheck_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`

**Interfaces:**

- Produces: invalid-state detection and stop.
- Does not produce repaired fields.

- [ ] **Step 1: Remove first-stage crashfix behavior**

First-stage GPU crashcheck must not modify `q_d`, `rho_d`, `prs_d`, or `tmp_d`.

- [ ] **Step 2: Report invalid state**

On invalid state, report:

```text
rank
i, j, k
rho, prs, tmp
q(1:5)
RK substep
nstep
```

- [ ] **Step 3: Stop immediately**

Use:

```fortran
stop 'GPU crashcheck detected invalid state'
```

Do not continue the simulation after invalid state.

### Phase 7: L2 Integration Validation

**Files:**

- Create: `tests/gpu_validation/run_tgv_field_compare.sh`
- Create: `tests/gpu_validation/compare_flowfield_h5.py`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`

**Interfaces:**

- Consumes: CPU and GPU checkpoint/field output.
- Produces: one-step and ten-step error reports for native statistics and full-field HDF5 output.

- [x] **Step 1: Add L2 run script**

Implemented as `tests/gpu_validation/run_tgv_field_compare.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_field_compare}"
MAXSTEP="${MAXSTEP:-10}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" --dst-case "$OUT_DIR/cpu" \
  --use-gpu f --maxstep "$MAXSTEP" --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" --scheme "$SCHEME"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" --dst-case "$OUT_DIR/gpu" \
  --use-gpu t --maxstep "$MAXSTEP" --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" --scheme "$SCHEME"

(
  cd "$OUT_DIR/cpu"
  mpirun -np 1 "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  mpirun -np 1 "$GPU_EXE" run datin/input.tgv > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt"

cat "$OUT_DIR/flowfield_compare.txt"
```

- [x] **Step 2: Record L2 outputs**

For each run, record:

```text
L_inf(q)
L2(q)
min/max(rho)
min/max(prs)
min/max(tmp)
first failing component and index, if any
```

- [x] **Step 3: Update validation matrix**

`L2-A` and `L2-B` are set to pass in `documents/GPU_VALIDATION_MATRIX.md` because both no-filter and filter full-field HDF5 comparisons are inside the agreed `1e-10` tolerance and no invalid state was detected.

### Phase 8: L3 Physical Diagnostics

**Files:**

- Create: `scripts/gpu_validation/plot_tgv_diagnostics.py`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`

**Interfaces:**

- Consumes: CPU/GPU validated TGV outputs.
- Produces: EPS and JPEG plots if plots are generated.

- [x] **Step 1: Create diagnostic script**

Implemented as `scripts/gpu_validation/plot_tgv_diagnostics.py`. The script reads CPU/GPU `flowstate.dat`, plots `kenergy`, `enstophy`, and `dissipation`, and writes `tests/gpu_validation/out/tgv_diagnostics_l3/tgv_diagnostics_summary.txt`.

Use project plotting rules:

```python
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.0
plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
```

Do not add a plot title.

- [x] **Step 2: Output formats**

Every figure must be written as:

```text
.eps
.jpeg
```

- [x] **Step 3: Record traceability**

Near each diagnostic result, record:

```text
script path
input field path
output figure path
CPU run command
GPU run command
```

Current traceability is recorded in `tests/gpu_validation/out/tgv_diagnostics_l3/tgv_diagnostics_summary.txt`. Generated figure paths are also listed there.

### Phase 9: Multi-Rank GPU Validation

The multi-rank design is now split into a dedicated plan:

- [`ASTR_GPU_MULTI_RANK_PORTING_PLAN.md`](ASTR_GPU_MULTI_RANK_PORTING_PLAN.md)

The first multi-rank baseline is no longer a full-field CPU `qswap` bridge. It uses qswap-compatible host-staged halo buffers: GPU pack, halo-buffer D2H, blocking `MPI_Sendrecv`, halo-buffer H2D, GPU unpack, and GPU primitive-halo refresh. This keeps the full field resident during the compute loop while avoiding mandatory CUDA-aware/HIP-aware MPI in the first correctness path.

First acceptance target:

```text
single node
2 MPI ranks
2 visible GPUs
one rank per GPU
isize=2, jsize=1, ksize=1
TGV periodic, q(1:5)
```

Use:

```bash
nsys profile --trace=cuda,mpi mpirun -np 2 ./astr run datin/input.tgv
```

Expected: profile exists; do not optimize until correctness passes.

### Phase 10: Performance Optimization

This phase starts only after correctness is accepted.

**Files:**

- Modify: performance-critical `src_gpu/*.cuf` only after profiling evidence.
- Modify: `documents/GPU_VALIDATION_MATRIX.md`.

**Interfaces:**

- Consumes: validated GPU path.
- Produces: measured speedups with correctness regression.

- [ ] **Step 1: Run Compute Sanitizer**

Debug build:

```bash
compute-sanitizer --tool memcheck mpirun -np 1 ./astr run datin/input.tgv
```

Expected: no invalid global read/write in project kernels.

- [ ] **Step 2: Run Nsight Systems**

```bash
nsys profile --stats=true mpirun -np 1 ./astr run datin/input.tgv
```

Expected: report shows kernel timeline and transfer timeline.

- [ ] **Step 3: Run Nsight Compute**

Use lineinfo build, not `-G`:

```bash
ncu --set full --kernel-name regex:conv_.* ./astr run datin/input.tgv
```

Expected: report includes memory throughput, occupancy, and stall reasons.

- [ ] **Step 4: Re-run L1/L2 after each optimization**

Every performance patch must preserve:

```text
L1 pass
L2-A pass
L2-B pass
detect-only crashcheck clean
```

## 7. Explicit Non-Goals for First Stage

Do not implement these in first-stage acceptance:

- compact derivative schemes
- compact filter schemes
- Thomas solver validation
- `num_species > 0`
- chemistry source terms
- turbulence model equations
- immersed boundary support
- non-periodic wall boundary conditions
- multi-rank GPU halo exchange
- CUDA-aware MPI
- performance claims before correctness

## 8. Review Gates

### Gate A: Scope Gate

Pass criteria:

- first-stage guard rejects unsupported configurations
- TGV `numq=5` path starts
- single-rank guard is active

### Gate B: RHS Gate

Pass criteria:

- convection-only comparison proves sign convention
- diffusion-only comparison passes
- full RHS comparison passes

### Gate C: Filter Gate

Pass criteria:

- A path with `lfilter=f` passes
- B path with explicit `lfilter=t` passes
- no compact filter solve is used

### Gate D: Integration Gate

Pass criteria:

- one substep passes
- one full step passes
- ten full steps pass
- crashcheck detects no invalid state

### Gate E: Diagnostics Gate

Pass criteria:

- L3 TGV diagnostics generated
- all figures have script/input/output traceability
- no performance claim is mixed into correctness diagnostics

## 9. Self-Review Against User Decisions

- First-stage target is TGV, one MPI rank, one GPU: covered by Global Constraints, Phase 0, Phase 2.
- A/B filter validation: covered by Global Constraints, Phase 5, Gate C.
- Explicit sixth-order central difference only: covered by ADR 0001, Phase 3.
- Explicit tenth-order central filter only: covered by ADR 0001, Phase 5.
- Directional thread blocks x `(512,1,1)`, y `(32,16,1)`, z `(64,1,8)`: covered by ADR 0002, Phase 3.
- `numq=5` only: covered by ADR 0003, Phase 2, Phase 4.
- Single-rank first: covered by ADR 0004, Phase 2, Phase 9.
- Tiered tolerances: covered by ADR 0005, Section 5.
- Synchronize after every kernel: covered by ADR 0006, Phase 1.
- Use `dxi_d` and `jacob_d`: covered by ADR 0007, Global Constraints, Phase 3/4.
- Detect-only crashcheck: covered by ADR 0008, Phase 6.
- This round writes docs only: this roadmap, `CONTEXT.md`, and ADRs are documentation artifacts.

## 10. Execution Options

Plan complete once this document is accepted.

Recommended execution mode:

1. Subagent-driven implementation, one phase or gate at a time.
2. Review after each gate.
3. No performance work before Gate D passes.

Inline execution is acceptable only if each gate is still reviewed before moving on.
