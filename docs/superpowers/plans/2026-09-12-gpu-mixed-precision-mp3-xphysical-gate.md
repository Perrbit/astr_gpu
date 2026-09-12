# GPU Mixed-Precision MP3 X-Physical Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admit and validate the FP32 `characteristic_flux` workspace for the bounded Shu-Osher S0-B0 x-physical case at NP=1 and NP=2.

**Architecture:** Preserve all FP64 sensing, Roe reconstruction, RHS, RK, and MPI state. Add only x-physical FP32 workspace writer/reader kernels, dispatch them under the existing S0-B0 capability, and validate the candidate against both CPU and GPU FP64 references with a dedicated driver.

**Tech Stack:** CUDA Fortran with NVHPC, Fortran MPI, Bash validation drivers, Python `unittest`, HDF5 field comparison, Compute Sanitizer.

## Global Constraints

- Build only through `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Keep the FP64 production default and abort on an ineligible MP3 request.
- Admit only `flowtype='shuosher'`, x `bctype=50,50`, periodic y/z, `543e`, `recon_schem=3`, `lchardecomp=t`, `diffterm=f`, `lfilter=f`, and RK3.
- Keep `q_d`, sensor, mask, Roe algebra, RHS, RK state, MPI payloads, diagnostics, and output unchanged.
- Cast only the final five-component interface flux to FP32 and promote both loads before FP64 differencing.
- Follow every new kernel launch with `sync_after_kernel`.
- Do not repair a CPU or FP64 GPU baseline defect without explicit user approval.

---

### Task 1: Freeze the source and driver contracts

**Files:**
- Modify: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`

**Interfaces:**
- Consumes: existing MP3 selector, S0-B0 case capability, periodic MP3 kernels.
- Produces: static requirements for x-physical kernels, bounded eligibility, dispatch, comparison, matrix, and memcheck drivers.

- [x] **Step 1: Add failing contract tests**

Require `commarray_gpu.cuf` to reference
`gpu_shock_characteristic_s0b0_xphysical_supported`, require the two
`*_x_physical_global_sp_kernel` names in `solver_gpu.cuf` and `mainloop_gpu.cuf`,
require the main loop to dispatch them only under
`characteristic_xphysical_case .and. mixed_characteristic_flux_workspace`, and
require dedicated compare, matrix, and memcheck scripts.

- [x] **Step 2: Verify the tests fail for the missing extension**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract
```

Expected: FAIL because the x-physical SP kernels and dedicated drivers do not
exist and the eligibility gate remains periodic-only.

### Task 2: Implement bounded x-physical MP3 execution

**Files:**
- Modify: `src_gpu/commarray_gpu.cuf`
- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Test: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`

**Interfaces:**
- Consumes: `gpu_shock_characteristic_s0b0_xphysical_supported()`,
  `characteristic_reconstruction_interface_flux`,
  `flux_characteristic_work_sp_d`, and `qrhs_d`.
- Produces: `characteristic_upwind_flux_x_physical_global_sp_kernel` and
  `characteristic_upwind_rhs_x_physical_global_sp_kernel`.

- [x] **Step 1: Add the alternate eligibility branch**

Import `gpu_shock_characteristic_s0b0_xphysical_supported` and define the
admitted geometry as:

```fortran
periodic_case = lihomo .and. ljhomo .and. lkhomo
xphysical_case = gpu_shock_characteristic_s0b0_xphysical_supported()
```

Keep all common precision, scheme, state-size, and RK constraints, then require
`periodic_case .or. xphysical_case`.

- [x] **Step 2: Add the FP32 x-physical writer**

Mirror the existing FP64 x-physical launch bounds and reconstruction call:

```fortran
call characteristic_reconstruction_interface_flux( &
     i,j,k,1,im,jm,km,reconstruction_scheme,gamma,hm,npdci,1,fh)
flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)
```

- [x] **Step 3: Add the FP32 x-physical reader**

Use the same `is:ie` active range as the FP64 reader and accumulate:

```fortran
qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
     (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
      real(flux_characteristic_work_sp_d(i-1,j,k,m),8))
```

- [x] **Step 4: Dispatch the new kernels**

Under `characteristic_xphysical_case`, select the new writer and reader when
`mixed_characteristic_flux_workspace` is true; otherwise retain the existing
FP64 launches. Keep explicit synchronization after both launches. Leave y/z on
the existing periodic MP3 kernels.

- [x] **Step 5: Verify contracts and root build**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract
cmake --build build_gpu_probe --target astr mixed_candidate_setup_test -j2
```

Expected: contract tests PASS and both targets build successfully.

### Task 3: Add executable physical-boundary gates

**Files:**
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_xphysical_compare.sh`
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_xphysical_matrix.sh`
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_xphysical_memcheck.sh`
- Modify: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`

**Interfaces:**
- Consumes: root CPU/GPU `astr` executables, `prepare_tgv_case.py`,
  `compare_shock_sensor.py`, `compare_flowstate.py`, and
  `compare_flowfield_h5.py`.
- Produces: three-way NP=1/2 numerical evidence and NP=2 memcheck evidence.

- [x] **Step 1: Implement the three-way comparison driver**

Prepare CPU, GPU FP64, and GPU MP3 copies of Shu-Osher with
`--homogeneous f,t,t`, `--bctype 50,50,1,1,1,1`, `--lchardecomp t`, no
diffusion/filter, `GRID=400,8,8`, `MAXSTEP=3`, and `DELTAT=1.d-4`. Require
completion and exact MP3 active-mode log markers. Compare CPU/GPU FP64 at
`1e-10`; compare GPU FP64/MP3 with explicit candidate tolerances; require exact
GPU sensor/mask equality.

- [x] **Step 2: Implement the frozen matrix driver**

Require an absolute `TOLERANCE_FILE`, source it once, then run NP=1
`1,1,1` and NP=2 `2,1,1` without calibration. Refuse to overwrite evidence.

- [x] **Step 3: Implement NP=2 memcheck**

Prepare a reduced x-physical case, run two MPI ranks with Compute Sanitizer
memcheck and `--error-exitcode 99`, and require two zero-error summaries, the
active MP3 marker, and normal completion.

- [x] **Step 4: Run NP=1 calibration and freeze tolerances**

Run the compare driver once with `CALIBRATE=t`, extract the maximum GPU
FP64/MP3 field and statistics errors, and write a new evidence-local tolerance
file using the `1-2-5` rounding rule. The threshold must be at least `2e-6` and
must not exceed `1e-5`.

- [x] **Step 5: Run the frozen numerical and safety gates**

Run:

```bash
TOLERANCE_FILE=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/mp3_xphysical_calibration_20260912/mp3_xphysical_tolerances.env \
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/mp3_xphysical_matrix_20260912 \
  tests/gpu_validation/run_mp3_characteristic_flux_xphysical_matrix.sh
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/mp3_xphysical_memcheck_20260912 \
  tests/gpu_validation/run_mp3_characteristic_flux_xphysical_memcheck.sh
```

Expected: NP=1 and NP=2 comparisons PASS, sensor/mask differences are zero,
and both sanitizer ranks report `ERROR SUMMARY: 0 errors`.

### Task 4: Synchronize evidence and run release checks

**Files:**
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`
- Modify: `docs/superpowers/specs/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux-design.md`
- Modify: `docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md`

**Interfaces:**
- Consumes: fresh NP=1/2 comparison reports, allocation logs, and memcheck logs.
- Produces: one consistent bounded statement of MP3 x-physical status.

- [x] **Step 1: Record measured evidence**

Record the frozen thresholds, maximum field/statistics errors, exact
sensor/mask result, sanitizer count, evidence paths, and classification. State
explicitly that `11/21`, NSCBC, diffusion, filtering, CURVE, long-time behavior,
and production promotion remain excluded.

- [x] **Step 2: Run the full contract suite**

Run:

```bash
python3 -m unittest discover -s tests/gpu_validation -p 'test_*.py'
```

Expected: all tests PASS.

- [x] **Step 3: Run final build and whitespace checks**

Run:

```bash
cmake --build build_gpu_probe --target astr mixed_candidate_setup_test -j2
git diff --check
```

Expected: build succeeds and `git diff --check` emits no diagnostics.
