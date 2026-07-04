# ASTR GPU Phase C Symmetry Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use writing-plans before implementation and execute task-by-task with verification after each task. Do not perform git operations unless the user explicitly requests them.

**Goal:** Extend the GPU boundary path from zeroextrap-only support to a reusable boundary dispatcher and add `bctype=60` symmetry support for x/y/z, single-rank and multi-rank physical-direction decomposition.

**Architecture:** Keep CPU and GPU code separated. Add a GPU boundary capability layer in `src_gpu/boundary_gpu.cuf` that identifies a single finite physical boundary type and dispatches per-face kernels only on `MPI_PROC_NULL` faces. Reuse existing Phase B physical-direction gradcal/convection/diffusion/filter/halo logic; symmetry changes only the boundary-state update, not the explicit stencil formulas.

**Tech Stack:** CUDA Fortran with NVHPC, MPI, existing root `CMakeLists.txt`, explicit `643e` central schemes, explicit 10th-order filter, host-staged MPI halo exchange.

## Global Constraints

- Build only through `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- `use_gpu` remains a runtime input option.
- Keep CPU and GPU code separated; allow only low-minimum edits in `src/`.
- No compact schemes, tridiagonal, or pentadiagonal solvers in this phase.
- Every GPU kernel launch must be followed by explicit `sync_after_kernel`.
- HDF5 field output, checkpoint, and file-output GPU writing remain out of scope.
- GPU validation uses CPU-owned field output only as an oracle.
- The current Phase C target is `numq=5`, `num_species=0`, `num_modequ=0`, `turbmode=none`, `rk3`, `conschm=643e`, `difschm=643e`.
- Do not claim support for wall, inflow, outflow, farfield, NSCBC, user-defined BC, immersed boundary, species, chemistry, or turbulence.

---

## CPU Symmetry Contract To Match

CPU entry: `src/bc.F90:symmetry(ndir)`.

For `num_species=0` and current GPU ideal-gas path, implement only primitive and conservative variables:

- Extrapolation helper: `extrapolate(a,b,dv=0)` is already matched by GPU `extrap2(a,b) = (4*a - b)/3`.
- `ndir=1` (`i=0`, active only when `mpileft == MPI_PROC_NULL`): extrapolate `vel/prs/tmp`, remove normal component using `bnorm_i0(j,k,:)`, compute `rho = prs/tmp*const2`, write `q`, zero `qrhs`.
- `ndir=2` (`i=im`, active only when `mpiright == MPI_PROC_NULL`): same using `bnorm_im(j,k,:)`.
- `ndir=3` (`j=0`, active only when `mpidown == MPI_PROC_NULL`): extrapolate `vel/prs/tmp`, then CPU writes `vel=(ue,0,we)` instead of the computed projected vector; compute `rho = prs/tmp*const2`, write `q`, zero `qrhs`.
- `ndir=4` (`j=jm`, active only when `mpiup == MPI_PROC_NULL`): same as `ndir=3`, with `vel=(ue,0,we)`.
- `ndir=5` (`k=0`, active only when `mpiback == MPI_PROC_NULL`): extrapolate `vel/prs/tmp`, then CPU writes `vel=(ue,ve,0)`; compute `rho = prs/tmp*const2`, write `q`, zero `qrhs`.
- `ndir=6` (`k=km`, active only when `mpifront == MPI_PROC_NULL`): extrapolate `vel/prs/rho`, remove normal component using `bnorm_km(i,j,:)`, compute `tmp = prs/rho*const2`, write `q`, zero `qrhs`.

Important: this plan intentionally matches CPU behavior, including the j-face and kmin hard-coded normal velocity treatment. It does not redefine the physics of symmetry.

---

## Files

- Modify: `src_gpu/boundary_gpu.cuf`
  - Add boundary capability detection for `bctype=60`.
  - Add symmetry kernels for x/y/z faces.
  - Replace zeroextrap-only dispatcher with boundary-type dispatch.
- Modify: `src_gpu/mainloop_gpu.cuf`
  - Replace `gpu_zeroextrap_axis/gpu_uses_zeroextrap` gate names with boundary-generic capability where needed.
  - Allow one symmetry direction with the other two directions periodic.
- Modify: `src_gpu/qswap_gpu.cuf`
  - Check whether primitive boundary refresh helpers currently assume only zeroextrap; rename or generalize if required.
- Modify: `src_gpu/commarray_gpu.cuf`
  - If exact normal projection for `i0/im/kmax` needs device boundary normal arrays, allocate/copy `bnorm_i0_d`, `bnorm_im_d`, and `bnorm_km_d`.
  - If validation is limited to Cartesian TGV symmetry first, this can be deferred, but the code must not claim curvilinear symmetry support without these arrays.
- Modify: `tests/gpu_validation/run_xextrap_phaseb_compare.sh`
  - Add `BC_KIND=zeroextrap|symmetry` while preserving old default behavior.
  - Map `BC_KIND=symmetry` and `ZERO_AXIS=x/y/z` to `bctype=60,60` in the physical direction.
- Create: `tests/gpu_validation/run_symmetry_phasec_mpirank_matrix.sh`
  - Mirror zeroextrap matrix driver with `BC_KIND=symmetry`.
- Modify: `tests/gpu_validation/README.md`
  - Document Phase C commands and scope.
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
  - Add Phase C validation rows after tests pass.
- Modify: `CONTEXT.md`
  - Record the boundary capability and current limitations after tests pass.

---

## Task 1: RED Tests For Symmetry Gate

**Files:**
- Modify: `tests/gpu_validation/run_xextrap_phaseb_compare.sh`
- Create: `tests/gpu_validation/run_symmetry_phasec_mpirank_matrix.sh`

**Interfaces:**
- Consumes: existing `prepare_tgv_case.py`, `compare_flowstate.py`, `compare_flowfield_h5.py`.
- Produces: `BC_KIND` driver option and a reusable symmetry matrix driver.

- [ ] Add `BC_KIND="${BC_KIND:-zeroextrap}"` to `run_xextrap_phaseb_compare.sh`.
- [ ] For `BC_KIND=zeroextrap`, keep the current `bctype=50,50` mapping exactly unchanged.
- [ ] For `BC_KIND=symmetry`, map:
  - `ZERO_AXIS=x`: `HOMOGENEOUS=f,t,t`, `BCTYPE=60,60,1,1,1,1`
  - `ZERO_AXIS=y`: `HOMOGENEOUS=t,f,t`, `BCTYPE=1,1,60,60,1,1`
  - `ZERO_AXIS=z`: `HOMOGENEOUS=t,t,f`, `BCTYPE=1,1,1,1,60,60`
- [ ] Reject unknown `BC_KIND` with exit code `2`.
- [ ] Add matrix script default:
  ```text
  x:2:1,2,1 x:2:1,1,2 x:2:2,1,1
  y:2:2,1,1 y:2:1,1,2 y:2:1,2,1
  z:2:2,1,1 z:2:1,2,1 z:2:1,1,2
  x:4:1,2,2 x:4:2,2,1 x:4:2,1,2
  y:4:2,1,2 y:4:2,2,1 y:4:1,2,2
  z:4:2,2,1 z:4:2,1,2 z:4:1,2,2
  ```
- [ ] Run the expected RED test:
  ```bash
  OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_red \
    BC_KIND=symmetry ZERO_AXIS=x NP=1 TOPOLOGY=1,1,1 \
    MAXSTEP=1 FEQCHKPT=1 GRID=64,64,64 \
    LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
    STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
    tests/gpu_validation/run_xextrap_phaseb_compare.sh
  ```
  Expected: CPU runs; GPU stops at the current capability gate because `bctype=60` is not supported yet.

---

## Task 2: Boundary Capability Detection

**Files:**
- Modify: `src_gpu/boundary_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`

**Interfaces:**
- Produces:
  - `integer function gpu_physical_boundary_axis()`
  - `integer function gpu_physical_boundary_kind()`
  - `logical function gpu_uses_physical_boundary()`
  - Keep compatibility wrappers for old zeroextrap names until all call sites are migrated.

- [ ] Add boundary kind constants in `boundary_gpu.cuf`:
  ```fortran
  integer,parameter :: GPU_BC_NONE=0, GPU_BC_ZEROEXTRAP=50, GPU_BC_SYMMETRY=60
  ```
- [ ] Implement detection for exactly one non-homogeneous direction:
  - x finite: `.not.lihomo`, `ljhomo`, `lkhomo`, `bctype(1)==bctype(2)` and other faces periodic.
  - y finite: `lihomo`, `.not.ljhomo`, `lkhomo`, matching pair on faces 3/4.
  - z finite: `lihomo`, `ljhomo`, `.not.lkhomo`, matching pair on faces 5/6.
- [ ] Accept only pair type `50` or `60`; return none otherwise.
- [ ] Update `validate_first_stage_gpu()`:
  - Replace `zeroextrap_case` with `physical_boundary_case`.
  - Error message should say: `GPU first-stage supports periodic x/y/z or one zeroextrap/symmetry direction with other directions periodic only`.
- [ ] Build:
  ```bash
  cmake --build build_gpu_probe -j 8
  ```

---

## Task 3: Symmetry Boundary Kernels

**Files:**
- Modify: `src_gpu/boundary_gpu.cuf`
- Modify: `src_gpu/commarray_gpu.cuf` only if device boundary normals are required for exact i/kmax projection.

**Interfaces:**
- Consumes: `q_d`, `qrhs_d`, `rho_d`, `vel_d`, `prs_d`, `tmp_d`, `const2`, `const6`, MPI face apply flags.
- Produces:
  - `symmetry_x_kernel(im,jm,km,hm,const2,const6,apply_left,apply_right)`
  - `symmetry_y_kernel(im,jm,km,hm,const2,const6,apply_down,apply_up)`
  - `symmetry_z_kernel(im,jm,km,hm,const2,const6,apply_back,apply_front)`

- [ ] Add a device helper:
  ```fortran
  attributes(device) subroutine set_qrhs_zero(i,j,k)
    use commarray_gpu, only: qrhs_d
    integer,value :: i,j,k
    integer :: m
    do m=1,5
      qrhs_d(i,j,k,m) = 0.d0
    enddo
  end subroutine set_qrhs_zero
  ```
- [ ] Implement x symmetry:
  - left face samples `i=1,2`; right face samples `i=im-1,im-2`.
  - pressure and temperature use `extrap2`.
  - density uses `rho = prs/tmp*const2`.
  - For Cartesian-first implementation, set `uu=0` on x faces and preserve tangential `vv/ww`; if device normals are added in the same task, remove `v_n` using `bnorm_i0_d`/`bnorm_im_d` to match CPU.
- [ ] Implement y symmetry:
  - down face samples `j=1,2`; up face samples `j=jm-1,jm-2`.
  - CPU-compatible velocity: `vv=0`, `uu/ww` extrapolated.
  - density uses `rho = prs/tmp*const2`.
- [ ] Implement z symmetry:
  - back face samples `k=1,2`; front face samples `k=km-1,km-2`.
  - back face CPU-compatible velocity: `ww=0`, `uu/vv` extrapolated.
  - front face should use CPU-compatible kmax behavior. For Cartesian validation, `ww=0` and `tmp=prs/rho*const2`; for curvilinear support, use `bnorm_km_d` and extrapolated `rho`.
- [ ] Add every kernel launch to `apply_boundary_conditions_gpu()` with `sync_after_kernel('symmetry_*_kernel')`.
- [ ] Build:
  ```bash
  cmake --build build_gpu_probe -j 8
  ```

---

## Task 4: Reuse Phase B Physical Stencils For Symmetry

**Files:**
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `src_gpu/gradcal_gpu.cuf` only if naming still hardcodes zeroextrap assumptions in public function names.
- Modify: `src_gpu/solver_gpu.cuf` only if call-site selection needs boundary-kind-general names.

**Interfaces:**
- Consumes: `gpu_physical_boundary_axis()`.
- Produces: main loop uses existing x/y/z physical kernels for both zeroextrap and symmetry finite-domain cases.

- [ ] Rename local variable `zero_axis` in `time_integration_rk_gpu()` to `physical_axis` or keep it but source it from the boundary-generic function.
- [ ] Keep the existing physical-direction kernels:
  - `convective_rhs_*_xphysical_global_kernel`
  - `convective_rhs_*_yphysical_global_kernel`
  - `convective_rhs_*_zphysical_global_kernel`
  - `diffusion_flux_*physical_global_kernel`
  - `diffusion_rhs_*physical_stored_global_kernel`
- [ ] Verify they are not mathematically tied to zeroextrap; they only need the physical axis and `MPI_PROC_NULL` face flags.
- [ ] Build:
  ```bash
  cmake --build build_gpu_probe -j 8
  ```

---

## Task 5: GREEN NP=1 Symmetry Validation

**Files:**
- Test-only.

**Interfaces:**
- Consumes: `BC_KIND=symmetry` validation driver.
- Produces: CPU/GPU field and statistics reports for x/y/z symmetry.

- [ ] Run x symmetry:
  ```bash
  OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_x_5steps \
    BC_KIND=symmetry ZERO_AXIS=x NP=1 TOPOLOGY=1,1,1 \
    MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
    LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
    STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
    tests/gpu_validation/run_xextrap_phaseb_compare.sh
  ```
- [ ] Run y symmetry with `ZERO_AXIS=y`.
- [ ] Run z symmetry with `ZERO_AXIS=z`.
- [ ] Expected: each command prints `status: pass` for `flowstate.dat` and `flowfield.h5` comparison.
- [ ] If field comparison fails but statistics pass, stop and inspect boundary planes first; do not relax tolerances.

---

## Task 6: GREEN NP=2/NP=4 Symmetry Matrix

**Files:**
- Test-only.

**Interfaces:**
- Consumes: `tests/gpu_validation/run_symmetry_phasec_mpirank_matrix.sh`.
- Produces: matrix summary proving physical-direction and periodic-direction decomposition.

- [ ] Run:
  ```bash
  OUT_DIR=tests/gpu_validation/out/symmetry_phasec_mpirank_matrix \
    MAXSTEP=1 FEQCHKPT=1 RUN_FIELD=t \
    tests/gpu_validation/run_symmetry_phasec_mpirank_matrix.sh
  ```
- [ ] Expected: 18/18 matrix entries pass.
- [ ] Run longer direct physical-direction checks:
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

---

## Task 7: Regression Matrix

**Files:**
- Test-only.

**Interfaces:**
- Ensures Phase C did not break Phase A/B.

- [ ] Build:
  ```bash
  cmake --build build_gpu_probe -j 8
  ```
- [ ] Re-run zeroextrap matrix:
  ```bash
  OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix_after_phasec \
    MAXSTEP=1 FEQCHKPT=1 RUN_FIELD=t \
    tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh
  ```
- [ ] Re-run periodic TGV smoke:
  ```bash
  OUT_DIR=tests/gpu_validation/out/tgv_phasec_regression \
    MAXSTEP=10 LFILTER=t DIFFTERM=t \
    tests/gpu_validation/run_tgv_stats_compare.sh
  ```

---

## Task 8: Documentation Update

**Files:**
- Modify: `CONTEXT.md`
- Modify: `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
- Modify: `tests/gpu_validation/README.md`

**Interfaces:**
- Produces documented Phase C scope and validation evidence.

- [x] Add a Phase C section:
  - `bctype=60` symmetry supported for x/y/z.
  - CPU-compatible quirks are intentionally matched.
  - `bctype=41/42/411/421`, `11/21/51`, `12/22/52/23`, `31`, `0`, and immersed boundary remain unsupported on GPU.
- [x] Record exact output directories and maximum observed field/stat errors from Tasks 5-7.
- [x] State that HDF5 output remains CPU-owned.

## Execution Results

Phase C implementation and validation were completed on `feature/gpu_dev` without git operations.

- Build passed with `cmake --build build_gpu_probe -j 8`.
- RED validation before implementation failed at the GPU capability gate for `BC_KIND=symmetry`, as expected.
- NP=1 symmetry validation passed for x/y/z at `MAXSTEP=5`, `LFILTER=t`, `DIFFTERM=t`, `COMPARE_STATS=t`, and `COMPARE_FIELD=t`. Max reconstructed `q5` field errors were about `5.68e-13` for x, `5.12e-13` for y, and `5.12e-13` for z.
- Multi-rank symmetry matrix passed 18/18 entries with `OUT_DIR=tests/gpu_validation/out/symmetry_phasec_mpirank_matrix MAXSTEP=1 FEQCHKPT=1 RUN_FIELD=t`.
- Longer physical-direction checks passed:
  - y physical decomposition: `OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np2_yphysical_5steps`, `NP=2`, `TOPOLOGY=1,2,1`, max reconstructed `q5=5.6843418860808015e-13`.
  - z physical decomposition: `OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np2_zphysical_5steps`, `NP=2`, `TOPOLOGY=1,1,2`, max reconstructed `q5=5.1159076974727213e-13`.
- Regression passed:
  - `OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix_after_phasec MAXSTEP=1 FEQCHKPT=1 RUN_FIELD=t tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh` passed 18/18.
  - `OUT_DIR=tests/gpu_validation/out/tgv_phasec_regression MAXSTEP=10 LFILTER=t DIFFTERM=t tests/gpu_validation/run_tgv_stats_compare.sh` passed with max diffs `kenergy=4.6726511548911276e-14`, `enstophy=4.6573855883025317e-14`, and `dissipation=1.3709736471079204e-16`.

---

## Self-Review

- Scope is one boundary type only: `bctype=60`.
- The plan does not add wall, inflow, outflow, farfield, NSCBC, chemistry, species, turbulence, compact schemes, or GPU file output.
- The CPU symmetry contract is explicit, including inconsistent-looking but real CPU behavior on j/k faces.
- Validation includes RED, NP=1, NP=2/NP=4, direct physical-direction 5-step checks, and Phase B regression.
