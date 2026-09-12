# Curvilinear NSCBC Viscous-Source Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate the discrete viscous residual as a source term in the existing static curved upper-eta non-reflecting GCBC on CPU and CUDA Fortran paths.

**Architecture:** Preserve the existing inviscid CURVE-C22 boundary and sixth-order diffusion implementations. On the upper-y owner, capture only the five-component boundary RHS before diffusion, form the actual viscous residual from the post-diffusion difference, project it with the local eta characteristic matrices, and subtract only the incoming characteristic components. Keep the GPU face buffer device resident and retain explicit synchronization after every new kernel.

**Tech Stack:** Fortran 2008, CUDA Fortran with NVHPC, MPI, HDF5, CMake, Bash, Python 3, NumPy, h5py, pytest, Compute Sanitizer, Nsight Systems.

## Global Constraints

- Build only from `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Work directly in the current `feature/gpu_dev` checkout; do not create a worktree.
- Do not modify, stage, or delete unrelated manuals, platform files, presentations, rendered output, or generated test results.
- Do not run `git add`, `git commit`, or `git push` unless the user separately requests it.
- Preserve CURVE-C22 inviscid numerical behavior and runtime string `ASTR_NSCBC_FARFIELD_MODE=nonreflecting`.
- Add a separate viscous capability; do not broaden the inviscid predicate.
- Restrict the first production slice to static single-block CURVE, upper-y `bctype=52`, lower-y `41`, x `11/21`, periodic z, five equations, no species, no modes, no turbulence, nondimensional Sutherland transport, `difschm=643e`, `diffterm=t`, and `lfilter=f`.
- Preserve FP64 arithmetic, local eta-normal projection, physical-face diffusion closure, fixed-`hm` MPI halo exchange, and explicit Runge-Kutta integration.
- The GPU face buffer must remain resident and must not introduce full-field host/device copies or a new MPI protocol.
- Every new CUDA kernel launch is followed by `sync_after_kernel`.
- Treat algebra, build, CPU/GPU equivalence, uniform preservation, acoustic reflection, MPI topology, memory safety, and residency as separate gates.
- Stop for user review if a CPU logic defect, ambiguous viscous boundary condition, or change to an existing physical boundary contract is discovered.

---

## File Map

| Path | Responsibility |
| --- | --- |
| `src/bc.F90` | CPU incoming-source projection helper, face snapshot, upper-y correction, ready-state guard |
| `src/solver.F90` | Capture/apply calls around the existing diffusion stage |
| `src_gpu/nscbc_characteristic_policy_gpu.cuf` | Device incoming-source projection helper |
| `src_gpu/boundary_gpu.cuf` | Device face buffer, capture/correction kernels, wrappers, release logic |
| `src_gpu/mainloop_gpu.cuf` | Capture before diffusion and correction after diffusion |
| `src_gpu/case_capability_gpu.cuf` | Separate restricted viscous predicate |
| `tests/gpu_validation/boundary_rhs_manufactured.F90` | CPU sign and wave-mask algebra oracle |
| `tests/gpu_validation/nscbc_characteristic_policy_probe.cuf` | GPU algebra probe |
| `tests/gpu_validation/compare_nscbc_characteristic_policy.py` | CPU/GPU algebra comparison |
| `tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py` | Static source, ordering, allocation, and sync contracts |
| `tests/gpu_validation/run_curvilinear_nscbc52_viscous_uniform_compare.sh` | Uniform viscous curved-field gate |
| `tests/gpu_validation/run_curvilinear_nscbc52_viscous_acoustic_compare.sh` | Viscous acoustic reflection gate |
| `tests/gpu_validation/run_curvilinear_nscbc52_viscous_matrix.sh` | NP=1/2/4/8 HBL matrix |
| `tests/gpu_validation/run_curvilinear_nscbc52_viscous_memcheck.sh` | NP=1/2 sanitizer gate |
| `tests/gpu_validation/run_curvilinear_nscbc52_viscous_profile.sh` | Nsight ownership and residency gate |
| `tests/gpu_validation/analyze_curvilinear_nscbc52_nsys.py` | Kernel and transfer audit |
| `tests/gpu_validation/README.md` | Reproduction commands |
| `documents/GPU_VALIDATION_MATRIX.md` | CURVE-C23 evidence |
| `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md` | Current capability statement |
| `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md` | Roadmap synchronization |

### Task 1: Lock the Capability and Stage Contract

**Files:**
- Create: `tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py`
- Modify: `src_gpu/case_capability_gpu.cuf`
- Test: `tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py`

**Interfaces:**
- Produces: `gpu_curvilinear_nscbc52_viscous_nonreflecting_supported()`
- Preserves: `gpu_curvilinear_nscbc52_nonreflecting_supported()` as inviscid-only

- [x] **Step 1: Write failing static contracts**

Require a distinct viscous predicate, `diffterm`, `difschm=643e`, no filter,
five equations, the approved boundary set, capture-before-diffusion ordering,
correction-after-diffusion ordering, owner-only wrappers, a face allocation,
and explicit synchronization after both new kernels.

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py
```

Expected: FAIL because the predicate and stage calls are absent.

- [x] **Step 3: Add the restricted named predicate**

Implement `gpu_curvilinear_nscbc52_viscous_nonreflecting_supported()` with the
exact global constraints. Retain `.not.diffterm` in the CURVE-C22 predicate.

- [x] **Step 4: Re-run the focused test**

Expected: capability assertions pass; orchestration assertions remain RED
until Tasks 3 and 4.

### Task 2: Prove the Incoming Viscous-Source Projection

**Files:**
- Modify: `src/bc.F90`
- Modify: `src_gpu/nscbc_characteristic_policy_gpu.cuf`
- Modify: `tests/gpu_validation/boundary_rhs_manufactured.F90`
- Modify: `tests/gpu_validation/nscbc_characteristic_policy_probe.cuf`
- Modify: `tests/gpu_validation/compare_nscbc_characteristic_policy.py`
- Test: `tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh`

**Interfaces:**
- Produces CPU: `nscbc_remove_incoming_source(rhs,source,pnor,pinv,jacobian,metric,velocity,css,lambda,incoming)`
- Produces GPU: `nscbc_remove_incoming_source_gpu(rhs,source,pnor,pinv,jacobian,metric,velocity,css,lambda,incoming)`

- [x] **Step 1: Add failing manufactured cases**

For subsonic inflow/outflow, supersonic inflow/outflow, and roundoff-level
normal velocity, construct a nonzero five-component source and expected result:

```fortran
source_characteristic=matmul(pinv,source)/jacobian
where(.not.incoming) source_characteristic=0.d0
expected=rhs-jacobian*matmul(pnor,source_characteristic)
```

Require incoming components to be removed, outgoing/stationary components to
remain unchanged, and the reversed sign to fail the expected-result check.

- [x] **Step 2: Verify RED**

Run:

```bash
tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh
```

Expected: compile failure because the production helper names do not exist.

- [x] **Step 3: Implement the CPU helper**

Reuse the CURVE-C22 metric norm, normal speed, wave tolerance, and incoming
mask. Project `source` with `pinv/jacobian`, zero non-incoming entries, and
subtract `jacobian*matmul(pnor,masked_source)` from `rhs`.

- [x] **Step 4: Implement the matching device helper**

Use fixed-size local arrays and explicit loops suitable for CUDA Fortran. Do
not allocate or call a host routine from device code.

- [x] **Step 5: Verify GREEN**

Run the policy probe and the existing Python contract tests. Expected: all CPU
and GPU algebra cases agree within `1e-12`.

### Task 3: Integrate the CPU Face Snapshot Around Diffusion

**Files:**
- Modify: `src/bc.F90`
- Modify: `src/solver.F90`
- Modify: `tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py`

**Interfaces:**
- Produces: `nscbc_farfield_viscous_source_enabled()`
- Produces: `capture_nscbc_farfield_y_upper_prediff_rhs()`
- Produces: `apply_nscbc_farfield_y_upper_viscous_source()`

- [x] **Step 1: Add failing CPU stage tests**

Require the capture call after `qrhs=-qrhs`, the existing diffusion call after
capture, the correction before the `full` validation snapshot, and a ready
flag that is cleared after consumption.

- [x] **Step 2: Verify RED**

Run the focused pytest file. Expected: CPU ordering assertions fail.

- [x] **Step 3: Add the face-only CPU state**

Add a lazily allocated `real(8)` buffer shaped `(0:im,0:km,1:5)` and a logical
ready flag in `bc`. Capture only on `npdcj==2` or `npdcj==4`. Reject an apply
without a matching capture.

- [x] **Step 4: Add the CPU correction loop**

At every local upper-face point, compute
`viscous_rhs=qrhs(i,jm,k,1:5)-prediff(i,k,1:5)`, rebuild `pnor/pinv` with the
same local state and `dxi(i,jm,k,2,:)` as CURVE-C22, then call the Task 2
helper. Reject non-finite values in validation mode.

- [x] **Step 5: Wire `solver::rhscal`**

Use local `use bc, only:` imports. Capture after convective sign conversion,
call the unchanged `diffrsdcal6`, then apply the correction before
`write_rhs_validation_snapshot('full')`.

- [x] **Step 6: Verify GREEN**

Run focused pytest and the CPU algebra probe. Expected: all pass.

### Task 4: Integrate the Resident GPU Face Buffer

**Files:**
- Modify: `src_gpu/boundary_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `src_gpu/case_capability_gpu.cuf`
- Modify: `tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py`

**Interfaces:**
- Produces: `capture_nscbc_farfield_y_upper_prediff_rhs_gpu()`
- Produces: `apply_nscbc_farfield_y_upper_viscous_source_gpu()`
- Owns: device buffer `nscbc52_prediff_rhs_d(0:im,0:km,1:5)`

- [x] **Step 1: Extend failing GPU contracts**

Require lazy face allocation, release in `release_boundary_gpu`, upper-y owner
checks, one capture kernel, one correction kernel, and exact launch order
around the existing diffusion block.

- [x] **Step 2: Verify RED**

Run focused pytest. Expected: GPU allocation and ordering assertions fail.

- [x] **Step 3: Add capture state and wrapper**

Allocate the device face buffer only for the viscous capability. The capture
kernel copies `qrhs_d(i,jm,k,1:5)`. Set a host ready flag only after the
explicit post-kernel synchronization succeeds.

- [x] **Step 4: Add the correction kernel and wrapper**

Compute the RHS difference, rebuild the existing `pnor/pinv`, call the Task 2
device helper, and write the corrected upper-face `qrhs_d`. Clear the ready
flag after the synchronized launch. Non-owner ranks return without launch.

- [x] **Step 5: Wire the GPU main loop**

Capture immediately before the existing diffusion-flux launch block and apply
the correction after all three diffusion RHS kernels but before case sources.
Route the separate viscous capability through first-stage validation and the
same conservative boundary/full-face diffusion booleans as the approved HBL
path.

- [x] **Step 6: Verify GREEN and build both backends**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_viscous_source_contract.py
cmake -S . -B build_cpu_probe -DASTR_WITH_CUDA=OFF -DBUILD_TESTING=OFF
cmake --build build_cpu_probe -j2
cmake -S . -B build_gpu_probe -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=ON -DBUILD_TESTING=OFF
cmake --build build_gpu_probe -j2
```

Expected: contracts and both top-level builds pass.

### Task 5: Close Uniform and Viscous Acoustic Gates

**Files:**
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_viscous_uniform_compare.sh`
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_viscous_acoustic_compare.sh`
- Modify: `tests/gpu_validation/test_curvilinear_acoustic_tools.py`
- Modify: `tests/gpu_validation/README.md`

**Interfaces:**
- Consumes: existing curved grid/pulse generators and reflection analyzer
- Produces: matched CPU/GPU uniform and viscous acoustic reports

- [x] **Step 1: Add failing runner contracts**

Require `DIFFTERM=t`, `DIFFSCHEME=643e`, Sutherland transport, nonreflecting
mode, matched CPU/GPU inputs, explicit `OUT_DIR`, finite-field checks, and
full-field comparisons.

- [x] **Step 2: Verify RED**

Run the focused Python tests. Expected: failure because the new runners are
absent.

- [x] **Step 3: Implement the uniform runner**

Prepare the existing curved acoustic box with a constant field, run CPU and
GPU with viscosity, and require uniform drift <=`1e-12` and field error
<=`1e-10`.

- [x] **Step 4: Implement the viscous acoustic runner**

Run the established three grids with the compact-support pulse. Compare each
GPU result with its same-grid CPU result. Require reflection agreement and
field error <=`1e-10`; record the measured reflection trend without importing
the inviscid absolute threshold.

- [x] **Step 5: Run both gates**

Expected: both pass with finite fields and measured reports under the selected
output directories. If reflection does not decrease with refinement, stop and
report the physical/numerical issue instead of weakening the test.

### Task 6: Close MPI, Sanitizer, and Residency Gates

**Files:**
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_viscous_matrix.sh`
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_viscous_memcheck.sh`
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_viscous_profile.sh`
- Modify: `tests/gpu_validation/analyze_curvilinear_nscbc52_nsys.py`
- Modify: `tests/gpu_validation/test_analyze_curvilinear_nscbc52_nsys.py`

**Interfaces:**
- Consumes: Task 5 validated single-rank case
- Produces: topology, sanitizer, owner-launch, and transfer reports

- [x] **Step 1: Add failing analyzer and driver tests**

Require one capture and one correction per viscous RHS on upper-y owners, zero
such kernels on non-owners, and zero H2D/D2H operations >=64 KiB after the
first characteristic RHS.

- [x] **Step 2: Verify RED**

Run analyzer and contract pytest files. Expected: new runner/token assertions
fail.

- [x] **Step 3: Implement the topology matrix**

Run NP=1, NP=2 x/y/z slabs, NP=4 `2x2x1`, `2x1x2`, and `1x2x2`, plus NP=8
`2x2x2`. Compare same-topology CPU/GPU fields and statistics at `1e-10`, check
upper-y owner counts, wall invariants, positive Jacobians, and finite fields.

- [x] **Step 4: Implement sanitizer and profile runners**

Run Compute Sanitizer for NP=1 and NP=2 `1x2x1`. Capture a two-step NP=1 trace
and an NP=2 owner trace. Preserve raw logs and SQLite output below explicit
ignored output directories.

- [x] **Step 5: Run all runtime gates**

Expected: topology matrix passes, sanitizer reports zero errors, owner launch
counts match the number of physical upper-y ranks, and no full-field transfer
is introduced.

### Task 7: Regression and Evidence-Bounded Documentation

**Files:**
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`
- Modify: `docs/superpowers/plans/2026-09-12-curvilinear-nscbc-viscous-source.md`

**Interfaces:**
- Consumes: measured outputs from Tasks 2-6
- Produces: CURVE-C23 capability and limitations record

- [x] **Step 1: Run focused Python and shell syntax checks**

Run all CURVE-C22/C23 Python tests, `bash -n` on every new runner, and
`git diff --check`.

- [x] **Step 2: Run focused CURVE-C22 regression**

Run the existing inviscid policy probe, a short uniform comparison, and the
required-reject gates. Retain the frozen C22 acoustic/topology evidence when
the shared inviscid numerical operator is unchanged; repeat those expensive
gates only if that operator changes. Expected: measured CURVE-C22 behavior
remains within its frozen thresholds.

- [x] **Step 3: Run the final CURVE-C23 matrix**

Rebuild once from the top-level CMake file and run algebra, uniform, acoustic,
HBL topology, sanitizer, and profile gates against those binaries.

- [x] **Step 4: Update documentation from measured evidence**

Record exact grids, step counts, topologies, maximum errors, reflection
coefficients, sanitizer summaries, kernel counts, and transfer sizes. State
only that the tested upper-eta boundary is viscous-source-coupled. Retain the
explicit non-claim for a general complete viscous open boundary and production
SBLI fidelity.

- [x] **Step 5: Final audit without Git mutation**

Run:

```bash
git status --short
git diff --check
git diff -- src/bc.F90 src/solver.F90 src_gpu tests/gpu_validation documents docs/superpowers
```

Expected: no whitespace errors, generated results remain ignored, unrelated
untracked files remain untouched, and no Git staging, commit, or push occurs.
