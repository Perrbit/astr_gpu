# GPU Mixed-Precision Phase MP2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and locally validate mutually exclusive FP32 diagnostic-derivative and viscous-flux GPU workspaces without changing FP64 authoritative state, boundary algebra, MPI payloads, or explicit synchronization semantics.

**Architecture:** Keep `ASTR_GPU_PRECISION_MODE` as the top-level precision switch and add one MPI-consistent candidate selector. Allocate either the FP64 workspace or its candidate-specific FP32 replacement, duplicate only the affected CUDA Fortran writer/reader kernels, and convert FP32 values to FP64 at projection, differencing, reduction, and MPI-buffer boundaries. The derivative workspace is consumed only by diagnostics/statistics; the solver independently recomputes diffusion gradients from `vel_d/tmp_d`.

**Tech Stack:** CUDA Fortran with NVHPC 26.1, Fortran MPI, CMake, HDF5 validation output, Python/pytest contract and comparison tools, Compute Sanitizer, Nsight-compatible complete-RK timing.

## Global Constraints

- Build only through `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Work in the current `feature/gpu_dev` tree; do not create a worktree.
- Keep `ASTR_GPU_PRECISION_MODE=fp64` behavior unchanged.
- Permit exactly one of `flux`, `derivative`, or `viscous_flux`; do not implement combinations.
- Keep `q_d`, `qrhs_d`, `qsave_d`, primitives, geometry, RK, boundary characteristic algebra, diagnostics accumulation, checkpoints, and HDF5 output in FP64.
- Keep every MPI halo payload and host/device communication buffer in FP64.
- Preserve explicit `sync_after_kernel` after every kernel launch.
- Preserve x/y/z thread blocks `(512,1,1)`, `(32,16,1)`, and `(64,1,8)` where directional kernels use those established geometries.
- Do not alter `src/` CPU numerical behavior for MP2.
- Do not relax a numerical or physical gate after observing a failure.
- Do not stage or commit files unless the user explicitly requests Git operations.

---

### Task 1: MPI-consistent mixed-candidate contract

**Files:**
- Create: `src_gpu/mixed_candidate_gpu.cuf`
- Create: `tests/gpu_validation/mixed_candidate_setup_test.cuf`
- Create: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`
- Modify: `src/CMakeLists.txt`

**Interfaces:**
- Consumes: `precision_mode_gpu::gpu_mixed_workspace_requested()`
- Produces: `configure_gpu_mixed_candidate()`
- Produces: `gpu_mixed_candidate_id() -> integer`
- Produces: `gpu_mixed_candidate_name() -> character(len=32)`
- Produces: `gpu_mixed_candidate_requested(candidate_id) -> logical`
- Constants: `GPU_MIXED_FLUX=0`, `GPU_MIXED_DERIVATIVE=1`, `GPU_MIXED_VISCOUS_FLUX=2`

- [x] **Step 1: Write a source-contract test for accepted values, default, MPI agreement, and hard failures.**

```python
def test_candidate_contract():
    compact = MODE.replace(" ", "").lower()
    assert "astr_gpu_mixed_candidate" in compact
    assert "case('flux')" in compact
    assert "case('derivative')" in compact
    assert "case('viscous_flux')" in compact
    assert compact.count("mpi_allreduce") >= 2
    assert "lowest/=highest" in compact
    assert "mpi_abort" in compact
```

- [x] **Step 2: Run the focused test and confirm RED.**

Run: `python3 -m pytest -q tests/gpu_validation/test_mixed_precision_mp2_contract.py`

Expected: FAIL because `mixed_candidate_gpu.cuf` does not exist.

- [x] **Step 3: Implement the selector with `flux` as the unset default.**

```fortran
choice=GPU_MIXED_FLUX
call get_environment_variable('ASTR_GPU_MIXED_CANDIDATE',value,status=status)
if(status==0) then
  select case(trim(adjustl(value)))
  case('flux');         choice=GPU_MIXED_FLUX
  case('derivative');   choice=GPU_MIXED_DERIVATIVE
  case('viscous_flux'); choice=GPU_MIXED_VISCOUS_FLUX
  case default;         choice=-1
  end select
endif
call mpi_allreduce(choice,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
call mpi_allreduce(choice,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
if(lowest<0 .or. lowest/=highest) call mpi_abort(MPI_COMM_WORLD,1,ierr)
```

- [x] **Step 4: Reject `fp64` plus an explicitly non-default candidate and add rank-zero logging.**

```fortran
if((.not.gpu_mixed_workspace_requested()) .and. status==0 .and. &
   choice/=GPU_MIXED_FLUX) call mpi_abort(MPI_COMM_WORLD,1,ierr)
if(rank==0) write(*,'(A,A)') 'ASTR_GPU_MIXED_CANDIDATE=', &
  trim(gpu_mixed_candidate_name())
```

- [x] **Step 5: Add the module before `commarray_gpu.cuf` and add the probe target.**

- [x] **Step 6: Build and test default, three valid candidates, invalid value, FP64 mismatch, and NP=2 rank disagreement.**

Run: `cmake --build build_gpu_probe --target mixed_candidate_setup_test -j2`

Expected: valid modes exit 0; invalid and inconsistent modes exit nonzero through MPI abort.

---

### Task 2: Mutually exclusive allocation and byte accounting

**Files:**
- Modify: `src_gpu/commarray_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`

**Interfaces:**
- Produces: `real(4), allocatable, device :: dvel_sp_d(:,:,:,:,:), dtmp_sp_d(:,:,:,:)`
- Produces: `real(4), allocatable, device :: sigma_sp_d(:,:,:,:), qflux_sp_d(:,:,:,:)`
- Produces: `gpu_mixed_derivative_workspace_enabled() -> logical`
- Produces: `gpu_mixed_viscous_flux_workspace_enabled() -> logical`
- Logs: `ASTR_GPU_ACTIVE_MIXED_WORKSPACE`, `ASTR_GPU_MIXED_WORKSPACE_BYTES`, `ASTR_GPU_FP64_WORKSPACE_BYTES`

- [x] **Step 1: Add failing assertions that each candidate omits its corresponding FP64 allocation.**

```python
assert "allocate(dvel_sp_d(0:im,0:jm,0:km,1:3,1:3))" in compact
assert "allocate(dtmp_sp_d(0:im,0:jm,0:km,1:3))" in compact
assert "allocate(sigma_sp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:6))" in compact
assert "allocate(qflux_sp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3))" in compact
```

- [x] **Step 2: Run the contract test and confirm RED.**

- [x] **Step 3: Implement eligibility and hard-fail unsupported cases.**

```fortran
derivative_active = gpu_mixed_workspace_requested() .and. &
  gpu_mixed_candidate_requested(GPU_MIXED_DERIVATIVE)
viscous_active = gpu_mixed_workspace_requested() .and. &
  gpu_mixed_candidate_requested(GPU_MIXED_VISCOUS_FLUX)
if((derivative_active .or. viscous_active) .and. &
   ((.not.diffterm) .or. trim(difschm)/='643e')) &
  call mpi_abort(MPI_COMM_WORLD,1,ierr)
if(derivative_active .and. &
   (shock_sensor_validation_enabled() .or. lchardecomp)) &
  call mpi_abort(MPI_COMM_WORLD,1,ierr)
```

Apply the same hard-fail policy when `numq/=5`, `num_species/=0`, or the
configured time integrator is not the validated RK3 path. These checks must
use the existing runtime variables rather than introducing duplicate controls.

- [x] **Step 4: Allocate exactly one precision for each workspace pair.**

```fortran
if(derivative_active) then
  allocate(dvel_sp_d(0:im,0:jm,0:km,1:3,1:3))
  allocate(dtmp_sp_d(0:im,0:jm,0:km,1:3))
else
  allocate(dvel_d(0:im,0:jm,0:km,1:3,1:3))
  allocate(dtmp_d(0:im,0:jm,0:km,1:3))
endif
```

- [x] **Step 5: Add exact integer-byte formulas and release all optional arrays.**

- [x] **Step 6: Build the full executable and verify allocation logs for FP64, flux, derivative, and viscous-flux modes.**

---

### Task 3: FP32 derivative writer kernels

**Files:**
- Modify: `src_gpu/gradcal_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`

**Interfaces:**
- Produces five `_sp_kernel` variants of `gradcal_dvel_kernel` and its x/y/z/xy-physical variants.
- `gradcal_gpu()` dispatches either all-FP64 or all-FP32 derivative writers.

- [x] **Step 1: Require all five SP writer kernels and post-launch synchronizations in the contract test.**

- [x] **Step 2: Confirm the test fails before implementation.**

- [x] **Step 3: Duplicate only final stores, retaining FP64 stencil and metric arithmetic.**

```fortran
real(8) :: value
value = du_dxi(1,a)*dxi_d(i,j,k,1,b) + &
        du_dxi(2,a)*dxi_d(i,j,k,2,b) + &
        du_dxi(3,a)*dxi_d(i,j,k,3,b)
dvel_sp_d(i,j,k,a,b)=real(value,4)
```

- [x] **Step 4: Dispatch with the established `block3/grid3` geometry and call `sync_after_kernel` after every launch.**

- [x] **Step 5: Build `astr` and run a one-step periodic viscous TGV smoke test.**

---

### Task 4: FP32 diagnostic-derivative consumers

**Files:**
- Modify: `src_gpu/statistic_gpu.cuf`
- Modify: `src_gpu/production_statistics_gpu.cuf`
- Modify: `src_gpu/gradcal_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`

**Interfaces:**
- Produces derivative-SP enstrophy, dissipation, Channel, S1/HBL, and compact-wall statistics kernels.
- Makes the gradcal CPU/GPU comparison output precision-aware.
- Preserves FP64 constitutive algebra and FP64 reductions.

- [x] **Step 1: Add failing tests that enumerate every known derivative consumer.**

- [x] **Step 2: Confirm RED and compare the enumerated list against `rg -n 'dvel_d|dtmp_d' src_gpu`; document that solver diffusion kernels are not consumers.**

- [x] **Step 3: Implement SP statistics readers with explicit promotion.**

```fortran
dvel(a,b)=real(dvel_sp_d(i,j,k,a,b),8)
dtmp(b)=real(dtmp_sp_d(i,j,k,b),8)
```

- [x] **Step 4: Implement the SP compact-wall reader and precision-aware gradcal comparison; keep partial sums and host comparisons FP64.**

- [x] **Step 5: Dispatch each diagnostic wrapper to exactly one precision-specific kernel and preserve one synchronization after each selected launch.**

- [x] **Step 6: Build and run targeted source-contract tests.**

---

### Task 5: Derivative candidate local admission

**Files:**
- Create: `tests/gpu_validation/run_mp2_derivative_tgv_compare.sh`
- Create: `tests/gpu_validation/run_mp2_derivative_hbl_compare.sh`
- Create: `tests/gpu_validation/run_mp2_derivative_curve_compare.sh`
- Create: `tests/gpu_validation/run_mp2_derivative_mpi_matrix.sh`
- Create: `tests/gpu_validation/run_mp2_derivative_memcheck.sh`
- Create: `tests/gpu_validation/run_mp2_derivative_benchmark.sh`
- Create: `tests/gpu_validation/summarize_mp2_benchmark.py`
- Test: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`

**Interfaces:**
- Compares `ASTR_GPU_PRECISION_MODE=fp64` against `mixed_workspace` plus candidate `derivative`.
- Reuses existing field, flowstate, HBL physics, CURVE acoustic/uniform, and rank-timing tools.

- [x] **Step 1: Write drivers that create separate, non-overwriting evidence directories and verify mode logs.**

- [x] **Step 2: Run a pilot periodic viscous TGV case and freeze field/statistics thresholds from observed FP32 workspace error.**

- [x] **Step 3: Run Cartesian Sutherland HBL and require field, wall-friction, wall-heat-flux, and profile agreement.**

- [x] **Step 4: Run CURVE-C23 uniform, acoustic, and viscous HBL checks at matched RK phase.**

- [x] **Step 5: Run NP=2 `2x1x1` and NP=4 `2x2x1` where the case capability is eligible.**

- [x] **Step 6: Run Compute Sanitizer and require `ERROR SUMMARY: 0 errors`.**

- [x] **Step 7: Run five interleaved FP64/derivative repeats and report complete-RK median and spread without imposing a speedup threshold.**

- [x] **Step 8: Verify measured allocation bytes equal the analytical FP64-to-FP32 reduction and no FP64 derivative mirror is allocated.**

- [x] **Step 9: Stop for user review if any physical trend or error growth is unexplained.** No candidate-dependent physical discrepancy was observed; the known hot-wall transient heat-flux/reference mismatch is identical in both modes and is not promoted to a steady-HBL validation claim.

Local evidence (2026-09-12): periodic TGV fields were bitwise identical and
statistics remained below `1e-8` for NP=1/2/4; Cartesian and CURVE HBL runtime
wall diagnostics differed by at most `3.91e-13` (`fbcx`) and `3.70e-15`
(`wallheatflux`), while post-processed wall/profile quantities were identical.
Compute Sanitizer reported zero errors. Five interleaved `64^3` timing repeats
gave `8.706775 ms` FP64 and `8.891394 ms` derivative median complete-RK time,
with `3.956%` and `3.732%` spread. The measured derivative workspace changed
from `26,364,000` to `13,182,000` bytes per rank, exactly matching the
analytical 50% storage reduction.

---

### Task 6: FP32 viscous-flux writers and RHS readers

**Files:**
- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`

**Interfaces:**
- Produces five `_sp` diffusion-flux writer kernels.
- Produces thirteen `_sp` stored-diffusion RHS kernels: three periodic,
  one x-physical, three y-physical, three z-physical, and three xy-physical.
- Uses FP64 `dvel_d/dtmp_d` and FP64 RHS accumulation.

- [x] **Step 1: Add failing tests for all writer and reader variants plus synchronization labels.**

- [x] **Step 2: Confirm RED.**

- [x] **Step 3: Compute stress and heat flux in FP64 and cast only final stores.**

```fortran
sigma_sp_d(i,j,k,1)=real(tau11,4)
qflux_sp_d(i,j,k,1)=real(hcc*dtmp(1)+tau11*u1+tau12*u2+tau13*u3,4)
```

- [x] **Step 4: Promote each SP stencil value before metric projection and differencing in RHS kernels.**

- [x] **Step 5: Dispatch all periodic and physical-boundary variants without changing launch geometry or sync ordering.**

- [x] **Step 6: Build the full executable, but do not run the candidate until
  Task 7 provides FP32 local-periodic and MPI halo refresh.** Even NP=1
  periodic diffusion consumes `hm`-layer `sigma/qflux`, so a pre-halo smoke
  test would access the unallocated FP64 workspace or require an artificial
  non-production bypass.

---

### Task 7: FP32 field halo with FP64 MPI payload

**Files:**
- Modify: `src_gpu/halo_exchange_gpu.cuf`
- Modify: `tests/gpu_validation/test_mixed_precision_mp2_contract.py`

**Interfaces:**
- Produces: `exchange_diffusion_flux_sp_halo_gpu(work)`
- Produces x/y/z FP32 pack, unpack, and local periodic refresh kernels.
- Reuses existing FP64 device and host field buffers, MPI tags, counts, and neighbor ordering.

- [x] **Step 1: Add failing tests requiring FP32 field arguments and FP64 send/receive buffers.**

- [x] **Step 2: Confirm RED.**

- [x] **Step 3: Implement directional pack conversion.**

```fortran
real(4),device :: field(...)
real(8),device :: sendbuf(...)
sendbuf(h,j,k,m)=real(field(h+1,j,k,m),8)
```

- [x] **Step 4: Implement directional unpack conversion and FP32 local periodic halo refresh.**

```fortran
field(im+1+h,j,k,m)=real(recv_right(h,j,k,m),4)
```

- [x] **Step 5: Preserve existing explicit synchronization points and optional interior-work callback ordering.**

- [x] **Step 6: Run NP=1, NP=2, and NP=4 field comparisons and verify logs show FP64 halo buffers.**

---

### Task 8: Viscous-flux admission and documentation closure

**Files:**
- Create: `tests/gpu_validation/run_mp2_viscous_flux_tgv_compare.sh`
- Create: `tests/gpu_validation/run_mp2_viscous_flux_hbl_compare.sh`
- Create: `tests/gpu_validation/run_mp2_viscous_flux_curve_compare.sh`
- Create: `tests/gpu_validation/run_mp2_viscous_flux_mpi_matrix.sh`
- Create: `tests/gpu_validation/run_mp2_viscous_flux_memcheck.sh`
- Create: `tests/gpu_validation/run_mp2_viscous_flux_benchmark.sh`
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`
- Modify: `docs/superpowers/specs/2026-09-12-gpu-mixed-precision-workspace-design.md`
- Modify: this plan

**Interfaces:**
- Produces complete local evidence and classification for both MP2 candidates.

- [x] **Step 1: Run viscous TGV pilot and freeze candidate-specific thresholds.**

- [x] **Step 2: Run HBL field, wall-friction, heat-flux, and profile gates.**

- [x] **Step 3: Run CURVE-C23 uniform, acoustic, and viscous HBL gates.**

- [x] **Step 4: Run NP=2 and NP=4 halo correctness gates.**

- [x] **Step 5: Run Compute Sanitizer and require zero invalid accesses.**

- [x] **Step 6: Run five interleaved FP64/viscous-flux timing repetitions and verify byte accounting.**

- [x] **Step 7: Run all MP0/MP1/MP2 contract tests, shell syntax checks, NVHPC build, and `git diff --check`.**

Run:

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_mixed_precision_workspace_contract.py \
  tests/gpu_validation/test_mixed_precision_mp2_contract.py \
  tests/gpu_validation/test_upwind_flux_pair_contract.py
cmake --build build_gpu_probe --target mixed_candidate_setup_test astr -j2
git diff --check
```

- [x] **Step 8: Update documentation with exact commands, errors, physical diagnostics, bytes, medians, spreads, and `local-pass-not-promoted` or `rejected` status.**

- [x] **Step 9: Audit the Goal requirement by requirement; mark complete only when both candidates have every local gate.**

Local viscous-flux evidence (2026-09-12): the fixed TGV gates are `1e-9`
for fields and `1e-10` for statistics. NP=1/2/4 ten-step TGV runs pass with
maximum conservative-field difference `3.70e-13` and maximum statistic
difference `6.11e-16`. Cartesian HBL has maximum field/derived-diagnostic
differences `4.26e-12/1.97e-12`; CURVE-C23 uniform, acoustic, and viscous-HBL
gates pass. Compute Sanitizer reports zero errors. Five interleaved `64^3`
runs give FP64/candidate median complete-RK times `8.736809/9.417326 ms`, with
`1.544%/3.305%` spread. Workspace bytes change from `30,375,000` to
`15,187,500`, exactly 50%. Classification is `local-pass-not-promoted` because
the candidate is `7.789%` slower locally despite the verified storage reduction.
