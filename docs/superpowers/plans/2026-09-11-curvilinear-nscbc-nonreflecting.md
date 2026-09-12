# Curvilinear NSCBC Non-Reflecting Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate a Kim-Lee-style source-balanced non-reflecting GCBC on the static single-block curved upper eta face for CPU and CUDA Fortran ASTR.

**Architecture:** Add a third internal farfield policy without changing the existing pressure-relaxation or target-relaxation contracts. CPU and GPU retain ASTR's one-sided normal flux, metric correction, and transverse flux assembly. Locally incoming normal amplitudes cancel the metric and transverse source in characteristic space, making the total incoming characteristic derivative zero without discarding multidimensional source terms. Curved admission remains restricted to the approved inviscid five-equation upper-y cases.

**Tech Stack:** Fortran 2008, CUDA Fortran with NVHPC, MPI, HDF5, CMake, Bash, Python 3, NumPy, h5py, pytest, Compute Sanitizer, Nsight Systems.

## Global Constraints

- Build only from `/home/dell/workspace/astr_gpu/CMakeLists.txt` with `ASTR_WITH_CUDA=OFF/ON`.
- Work in the current `feature/gpu_dev` worktree; do not create another worktree.
- Do not revert, overwrite, or stage unrelated dirty-worktree changes.
- Do not run `git commit` or `git push` unless the user separately requests it.
- Preserve `compatibility`, `incoming_only`, and `sbli_shock` numerical behavior.
- Add only `ASTR_NSCBC_FARFIELD_MODE=nonreflecting` for the new source-balanced policy.
- Restrict the first capability to 3D, `numq=5`, `num_species=0`, `num_modequ=0`, static single-block CURVE, upper-y `bctype=52`, z periodic, `diffterm=f`, and `lfilter=f`.
- Do not read target-state values, derive a relaxation length, or execute the upper-face `gmachmax2` reduction in `nonreflecting` mode.
- Preserve local eta-normal projection, metric correction, x/z transverse flux derivatives, FP64 arithmetic, resident device fields, and explicit synchronization after every CUDA kernel.
- Treat build, CPU/GPU equivalence, uniform-flow preservation, acoustic reflection, MPI decomposition, and memory safety as separate gates.
- Stop and request user review if implementation exposes a CPU logic defect or requires changing an existing physical boundary contract.
- Scientific plots have no title, use `science/ieee/std-colors` without an extra font family, disable grids, and write both EPS and JPEG.

---

## File Map

| Path | Responsibility |
| --- | --- |
| `src/bc.F90` | Runtime policy parsing, CPU source-balanced incoming characteristic closure, CPU upper-y GCBC policy selection |
| `src/mainloop.F90` | Refresh NSCBC transverse halos before `boucon` while leaving new-mode filtering disabled |
| `src/test.F90` | Expose the CPU characteristic-policy probe through test mode `bcgc` |
| `src_gpu/nscbc_characteristic_policy_gpu.cuf` | Focused device routine that masks incoming characteristic amplitudes |
| `src_gpu/boundary_gpu.cuf` | Preserve the legacy upper-y kernel and add a target-free nonreflecting kernel/wrapper |
| `src_gpu/mainloop_gpu.cuf` | Select the non-reflecting execution sequence and skip legacy Mach/filter work while preserving RK snapshot semantics |
| `src_gpu/case_capability_gpu.cuf` | Admit only the approved curved-52 capability and MPI topologies |
| `src/CMakeLists.txt` | Order the new device policy module before `boundary_gpu.cuf` and build its probe |
| `tests/gpu_validation/boundary_rhs_manufactured.F90` | CPU algebra cases and machine-readable output |
| `tests/gpu_validation/nscbc_characteristic_policy_probe.cuf` | GPU algebra cases using the production device routine |
| `tests/gpu_validation/compare_nscbc_characteristic_policy.py` | CPU/GPU algebra comparison |
| `tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py` | Static source and capability contracts |
| `tests/gpu_validation/check_curvilinear_nscbc_geometry.py` | Jacobian, metric norm, and upper-face orientation gate |
| `tests/gpu_validation/generate_curvilinear_acoustic_pulse.py` | Linear three-dimensional acoustic initial field |
| `tests/gpu_validation/analyze_curvilinear_acoustic_reflection.py` | Characteristic-energy and reflection-coefficient calculation |
| `tests/gpu_validation/test_curvilinear_acoustic_tools.py` | Unit tests for geometry, pulse, and reflection analysis |
| `tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh` | Build and compare CPU/GPU algebra probes |
| `tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh` | NP=1 uniform-flow preservation and field comparison |
| `tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh` | Matched nonreflecting and compatibility acoustic runs |
| `tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh` | NP=1/2/4/8 topology matrix |
| `tests/gpu_validation/run_curvilinear_nscbc52_memcheck.sh` | Targeted NP=1 and NP=2 sanitizer runs |
| `tests/gpu_validation/run_curvilinear_nscbc52_profile.sh` | Two-step NP=1 Nsight Systems capture and audit |
| `tests/gpu_validation/analyze_curvilinear_nscbc52_nsys.py` | Reject legacy Mach/filter kernels and count non-reflecting RHS launches |
| `tests/gpu_validation/README.md` | Reproduction commands and evidence boundaries |
| `documents/GPU_VALIDATION_MATRIX.md` | Final gate results |
| `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md` | Current capability statement and next viscous target |
| `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md` | Top-level roadmap synchronization |

### Task 1: Lock the Runtime Policy Contract

**Files:**
- Create: `tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py`
- Modify: `src/bc.F90`
- Test: `tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py`

**Interfaces:**
- Produces: `NSCBC_FARFIELD_POLICY_COMPATIBILITY`, `NSCBC_FARFIELD_POLICY_TARGET_RELAXATION`, `NSCBC_FARFIELD_POLICY_NONREFLECTING`
- Produces: `integer function nscbc_farfield_policy_id()`
- Produces: `logical function nscbc_farfield_nonreflecting_enabled()`
- Preserves: `nscbc_farfield_incoming_only_enabled()` and `nscbc_farfield_sbli_shock_enabled()`

- [x] **Step 1: Write the failing static policy tests**

Add source-level assertions that require the new string, three named policies,
policy broadcast, and legacy strings:

```python
def test_farfield_parser_has_distinct_zero_incoming_policy() -> None:
    bc = source("src/bc.F90")
    for name in (
        "NSCBC_FARFIELD_POLICY_COMPATIBILITY",
        "NSCBC_FARFIELD_POLICY_TARGET_RELAXATION",
        "NSCBC_FARFIELD_POLICY_NONREFLECTING",
    ):
        assert name in bc
    assert "case('nonreflecting')" in bc.replace(" ", "").lower()
    assert "call bcast(nscbc_farfield_policy)" in bc.lower()


def test_legacy_mode_strings_remain_supported() -> None:
    parser = function_body(source("src/bc.F90"), "configure_nscbc_farfield")
    for mode in ("compatibility", "incoming_only", "sbli_shock"):
        assert f"case('{mode}')" in parser.replace(" ", "").lower()
```

- [x] **Step 2: Run the contract test and verify RED**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py
```

Expected: failure because `nonreflecting` and the named policy values do not
exist.

- [x] **Step 3: Add named policy state and parser behavior**

At module scope in `src/bc.F90`, add:

```fortran
integer,parameter :: NSCBC_FARFIELD_POLICY_COMPATIBILITY=0
integer,parameter :: NSCBC_FARFIELD_POLICY_TARGET_RELAXATION=1
integer,parameter :: NSCBC_FARFIELD_POLICY_NONREFLECTING=2
integer,save :: nscbc_farfield_policy=NSCBC_FARFIELD_POLICY_COMPATIBILITY
```

Extend `configure_nscbc_farfield` so the established booleans remain derived
compatibility state:

```fortran
case('compatibility')
  nscbc_farfield_policy=NSCBC_FARFIELD_POLICY_COMPATIBILITY
  nscbc_farfield_incoming_only=.false.
  nscbc_farfield_sbli_shock=.false.
case('incoming_only')
  nscbc_farfield_policy=NSCBC_FARFIELD_POLICY_TARGET_RELAXATION
  nscbc_farfield_incoming_only=.true.
  nscbc_farfield_sbli_shock=.false.
case('sbli_shock')
  nscbc_farfield_policy=NSCBC_FARFIELD_POLICY_TARGET_RELAXATION
  nscbc_farfield_incoming_only=.true.
  nscbc_farfield_sbli_shock=.true.
case('nonreflecting')
  nscbc_farfield_policy=NSCBC_FARFIELD_POLICY_NONREFLECTING
  nscbc_farfield_incoming_only=.false.
  nscbc_farfield_sbli_shock=.false.
case default
  stop 'ASTR_NSCBC_FARFIELD_MODE must be compatibility, incoming_only, sbli_shock, or nonreflecting'
end select
```

Broadcast `nscbc_farfield_policy`. Read and validate target values only when
the policy is `NSCBC_FARFIELD_POLICY_TARGET_RELAXATION`. Add the two query
functions listed in the Interfaces block.

- [x] **Step 4: Run the static contract and configure both builds**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py
cmake -S . -B build_cpu_probe -DASTR_WITH_CUDA=OFF -DBUILD_TESTING=OFF
cmake --build build_cpu_probe -j2
cmake -S . -B build_gpu_probe -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=ON -DBUILD_TESTING=OFF
cmake --build build_gpu_probe -j2
```

Expected: policy tests pass and both top-level CMake builds complete.

- [x] **Step 5: Review checkpoint**

Inspect only the Task 1 diff. Reject the task if any legacy branch changed its
target parsing, pressure formula, shock-target selection, or default mode.

### Task 2: Implement and Probe the CPU Source-Balanced Incoming Closure

**Files:**
- Modify: `src/bc.F90`
- Modify: `src/test.F90`
- Modify: `tests/gpu_validation/boundary_rhs_manufactured.F90`
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh`

**Interfaces:**
- Consumes: `NSCBC_FARFIELD_POLICY_NONREFLECTING`
- Produces: `subroutine nscbc_farfield_balance_incoming_lodi(lodi,source,pinv,jacobian,metric,velocity,css,lambda,incoming)`
- Produces: test mode `bcgc`

- [x] **Step 1: Add a failing CPU algebra probe**

Add `check_nscbc_characteristic_policy` to
`boundary_rhs_manufactured.F90`. Exercise these four states with a normalized
non-Cartesian metric direction:

```fortran
metric=[0.3d0,0.9d0,-0.2d0]
normal=metric/sqrt(sum(metric*metric))
lodi0=[1.d0,2.d0,3.d0,4.d0,5.d0]

call run_case('subsonic_outflow',metric, 0.2d0*normal,1.d0,lodi0)
call run_case('subsonic_inflow', metric,-0.2d0*normal,1.d0,lodi0)
call run_case('supersonic_outflow',metric, 2.0d0*normal,1.d0,lodi0)
call run_case('supersonic_inflow', metric,-2.0d0*normal,1.d0,lodi0)
```

Each case prints one machine-readable line containing the case name, five
eigenvalues, five mask integers, and five modified amplitudes. Add `bcgc` to
`src/test.F90::codetest`.

- [x] **Step 2: Run the CPU probe and verify RED**

Run:

```bash
mpirun -np 1 build_cpu_probe/bin/astr test bcgc
```

Expected: compile or link failure because
`nscbc_farfield_zero_incoming_lodi` is missing.

- [x] **Step 3: Implement the CPU policy helper**

The implemented helper computes the characteristic projection of the complete
metric-plus-transverse source and sets only incoming normal amplitudes to its
negative:

```fortran
subroutine nscbc_farfield_balance_incoming_lodi(lodi,source,pinv,jacobian, &
                                                 metric,velocity,css,lambda,incoming)
  real(8),intent(inout) :: lodi(5)
  real(8),intent(in) :: source(5),pinv(5,5),jacobian,metric(3),velocity(3),css
  real(8),intent(out) :: lambda(5)
  logical,intent(out) :: incoming(5)
  real(8) :: norm,normal_speed,wave_tolerance,source_characteristic(5)
  integer :: m

  norm=sqrt(sum(metric*metric))
  if(norm<=0.d0) error stop 'invalid upper-y NSCBC metric norm'
  if(css<=0.d0) error stop 'invalid upper-y NSCBC sound speed'
  normal_speed=sum(metric*velocity)/norm
  lambda=[normal_speed,normal_speed,normal_speed,normal_speed+css,normal_speed-css]
  wave_tolerance=64.d0*epsilon(1.d0)*max(css,abs(normal_speed))
  incoming=lambda<-wave_tolerance
  source_characteristic=matmul(pinv,source)/jacobian
  do m=1,5
    if(incoming(m)) lodi(m)=-source_characteristic(m)
  enddo
end subroutine nscbc_farfield_balance_incoming_lodi
```

- [x] **Step 4: Select the helper only for the new CPU policy**

In `farfield_nscbc(ndir=4)`, use mutually exclusive branches:

```fortran
select case(nscbc_farfield_policy)
case(NSCBC_FARFIELD_POLICY_NONREFLECTING)
  call nscbc_farfield_y_upper_source(i,j,k,Rest,source)
  call nscbc_farfield_balance_incoming_lodi(LODi,source,pinv, &
       jacob(i,j,k),dxi(i,j,k,2,:),vel(i,j,k,:),css,lambda,incoming)
case(NSCBC_FARFIELD_POLICY_TARGET_RELAXATION)
  call nscbc_farfield_y_upper_incoming_lodi(LODi,pinv,i,j,k,css,gmachmax2)
case default
  kinout=0.25d0*(1.d0-gmachmax2)*css/(ymax-ymin)
  LODi(5)=kinout*(prs(i,j,k)-pinf)
end select
```

Guard the face Mach loop and `pmax` so they do not execute for
`NSCBC_FARFIELD_POLICY_NONREFLECTING`. Do not change `LODi1`, `Rest`, `pnor`,
or transverse derivative assembly.

- [x] **Step 5: Pass the CPU algebra probe**

Run:

```bash
cmake --build build_cpu_probe -j2
mpirun -np 1 build_cpu_probe/bin/astr test bcgc
```

Expected: four `NSCBC_CPU_POLICY` lines and
`NSCBC_CPU_POLICY_PASS`; no target-state or relaxation diagnostic is printed.

- [x] **Step 6: Run existing Cartesian farfield regressions**

Run:

```bash
MAXSTEP=1 FEQCHKPT=1 tests/gpu_validation/run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh
MAXSTEP=1 FEQCHKPT=1 tests/gpu_validation/run_s2_hbl_nscbc52_incoming_only_compare.sh
MAXSTEP=1 FEQCHKPT=1 tests/gpu_validation/run_s2_sbli_physical_nscbc_compare.sh
```

Expected: all existing CPU/GPU comparisons pass at their established
tolerances. Any changed legacy field blocks further work.

### Task 3: Implement the GPU Policy and Execution Sequence

**Files:**
- Create: `src_gpu/nscbc_characteristic_policy_gpu.cuf`
- Create: `tests/gpu_validation/nscbc_characteristic_policy_probe.cuf`
- Create: `tests/gpu_validation/compare_nscbc_characteristic_policy.py`
- Modify: `src/CMakeLists.txt`
- Modify: `src_gpu/boundary_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh`
- Test: `tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py`

**Interfaces:**
- Produces: `attributes(device) subroutine nscbc_balance_incoming_lodi_gpu(...)`
- Consumes: named integer policy in the unchanged legacy-kernel calculation path
- Produces: `nscbc_farfield_y_upper_nonreflecting_rhs_kernel`
- Produces: `apply_nscbc_farfield_y_upper_nonreflecting_rhs_gpu()`
- Produces: executable target `nscbc_characteristic_policy_probe`

- [x] **Step 1: Extend static tests for GPU execution semantics**

Require a dedicated target-free production kernel to call the focused device
routine and retain explicit synchronization. Require the main loop to branch
before the Mach reduction and filters:

```python
def test_gpu_nonreflecting_path_skips_relaxation_work() -> None:
    mainloop = function_body(source("src_gpu/mainloop_gpu.cuf"), "time_integration_rk_gpu")
    assert "nscbc_farfield_nonreflecting_enabled" in mainloop
    assert mainloop.index("if(flatplate_nscbc_nonreflecting_case)") < mainloop.index(
        "call gpu_nscbc_farfield_y_upper_mach2"
    )
    boundary = source("src_gpu/boundary_gpu.cuf")
    body = function_body(
        boundary, "nscbc_farfield_y_upper_nonreflecting_rhs_kernel"
    )
    assert "call nscbc_balance_incoming_lodi_gpu" in body
    assert "rho_target" not in body
    assert "gmachmax2" not in body
    wrapper = function_body(
        boundary, "apply_nscbc_farfield_y_upper_nonreflecting_rhs_gpu"
    )
    assert "sync_after_kernel('nscbc_farfield_y_upper_nonreflecting_rhs_kernel')" in wrapper
```

- [x] **Step 2: Run the contract test and verify RED**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py
```

Expected: failure because the GPU policy module and branch are absent.

- [x] **Step 3: Add the focused device policy module**

Implement `nscbc_balance_incoming_lodi_gpu` with the same normalization,
eigenvalue order, source projection, and machine-zero deadband as the CPU helper. The admitted
production path relies on the one-time static metric gate and the existing
primitive-state validity checks; the device routine must not substitute a
Cartesian normal or silently clamp invalid values.

- [x] **Step 4: Add a dedicated target-free production kernel**

Replace the old `incoming_only` boolean kernel argument with the named integer
policy, while retaining the same two legacy calculations and arithmetic order.
Then add a new kernel with no target, relaxation, domain-length, or
Mach-reduction arguments:

```fortran
attributes(global) subroutine nscbc_farfield_y_upper_nonreflecting_rhs_kernel( &
    im,jm,km,hm,npdci,npdck,gamma,mach,apply_up)
```

Copy the established normal flux, metric correction, characteristic
projection, reconstruction, transverse x/z derivative, and `qrhs_d` sign
operations without changing their order. Replace only the boundary-policy
block with:

```fortran
call nscbc_balance_incoming_lodi_gpu(lodi,source,pinv,jacob_d(i,jm,k), &
     ddi,vel_d(i,jm,k,:),css,lambda,incoming)
```

The wrapper takes no physical target or `gmachmax2` argument. This deliberate
small duplication avoids changing the already validated legacy kernel while
ensuring the new launch contains no target-state or empirical-length data.

- [x] **Step 5: Split the GPU main-loop path**

For `flatplate_nscbc_nonreflecting_case`, execute only:

```fortran
call apply_nscbc_farfield_y_upper_nonreflecting_rhs_gpu()
```

followed by the wrapper's explicit synchronization and the existing required
solution halo exchange. Do not call:

```fortran
gpu_nscbc_farfield_y_upper_mach2
apply_nscbc_farfield_y_upper_filter_x_gpu
apply_nscbc_farfield_y_upper_filter_z_gpu
nscbc_farfield_q_to_primitive_kernel_postfilter
```

Keep the current sequence byte-for-byte equivalent for all older policies.

- [x] **Step 6: Build and compare the GPU algebra probe**

Add the new module before `boundary_gpu.cuf` in `ASTR_GPU_SOURCES`, and add the
standalone probe target. The probe runs the same four states as Task 2 and
prints `NSCBC_GPU_POLICY` records.

Run:

```bash
cmake -S . -B build_gpu_probe -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=ON -DBUILD_TESTING=OFF
cmake --build build_gpu_probe --target astr nscbc_characteristic_policy_probe -j2
tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh
```

Expected: all eigenvalues and modified amplitudes agree within `1e-12`; all
four mask patterns match exactly.

- [x] **Step 7: Re-run the three legacy farfield regressions**

Run the Task 2 Step 6 commands again. Expected: unchanged passing results.

### Task 4: Add Curved Geometry Admission and Uniform-Flow Gate

**Files:**
- Create: `tests/gpu_validation/check_curvilinear_nscbc_geometry.py`
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh`
- Modify: `src_gpu/case_capability_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `src/mainloop.F90`
- Modify: `tests/gpu_validation/run_curvilinear_open_boundary_rejects.sh`
- Test: `tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py`

**Interfaces:**
- Produces: `logical function gpu_curvilinear_nscbc52_nonreflecting_supported()`
- Reuses: `tests/gpu_validation/generate_curvilinear_tgv_grid.py --mapping y-wavy --amplitude 0.15`

- [x] **Step 1: Write failing capability tests**

Require the predicate to include every approved condition and require old
modes to retain the curved-52 reject. Add negative driver cases for
`diffterm=t`, `lfilter=t`, wrong upper type, nonperiodic z, and
`incoming_only` on the same curved grid.

- [x] **Step 2: Run contract and reject tests to verify RED**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py
tests/gpu_validation/run_curvilinear_open_boundary_rejects.sh
```

Expected: the new positive case is rejected because curved `52` is not yet
admitted; all existing negative cases still reject.

- [x] **Step 3: Implement the narrow capability predicate**

The predicate must require the common equation, boundary, and numerical
restrictions, plus one of two admitted initialization contracts:

```fortran
ndims==3 .and. nondimen .and. &
numq==5 .and. num_species==0 .and. num_modequ==0 .and. &
(.not.diffterm) .and. (.not.lfilter) .and. &
bctype(3)==41 .and. bctype(4)==52 .and. all(bctype(5:6)==1) .and. &
nscbc_farfield_nonreflecting_enabled() .and. &
(flatplate_configuration .or. acoustic_box_configuration)
```

`flatplate_configuration` retains `flowtype='bl'`, x `11/21`, z homogeneous,
and profile inflow. `acoustic_box_configuration` uses `flowtype='tgv'`,
periodic x/z, and a supplied weak-acoustic initial field.

Admit only the NP=1/2/4/8 topologies listed in the design. Do not weaken any
other curved open-boundary reject.

- [x] **Step 4: Implement the geometry checker**

Read `x/y/z` from the generated HDF5 grid. Compute the analytic Jacobian for
the selected mapping, the analytic upper eta gradient, and the vector from the
first interior plane to the upper boundary. Reject unless:

```python
np.all(np.isfinite(jacobian))
np.min(jacobian) > 0.0
np.min(np.linalg.norm(grad_eta, axis=-1)) > 0.0
np.min(np.sum(grad_eta * (upper - interior), axis=-1)) > 0.0
```

Write extrema and the orientation minimum to a text report.

- [x] **Step 5: Build the uniform-flow driver**

Prepare matched CPU/GPU cases with `32x24x32`, `a=0.15`, `maxstep=10`,
`lfilter=f`, `diffterm=f`, and `ASTR_NSCBC_FARFIELD_MODE=nonreflecting`.
Use a stationary constant perfect-gas state compatible with the wall and
inlet. Save the initial and final complete-RK fields.

Run the geometry checker before either solver. If it fails, exit without
launching ASTR.

- [x] **Step 6: Run the uniform-flow gate**

Run:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_nscbc52_uniform_np1 \
  tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh
```

Expected:

```text
geometry: PASS
cpu full-field drift <= 1e-10
gpu full-field drift <= 1e-10
upper-two-plane drift <= 1e-10
cpu/gpu full-field max_abs <= 1e-10
```

Stop the goal and report the offending term if this gate fails. Do not proceed
to acoustic tuning.

### Task 5: Implement the Acoustic Pulse and Reflection Diagnostic

**Files:**
- Create: `tests/gpu_validation/generate_curvilinear_acoustic_pulse.py`
- Create: `tests/gpu_validation/analyze_curvilinear_acoustic_reflection.py`
- Create: `tests/gpu_validation/test_curvilinear_acoustic_tools.py`
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh`

**Interfaces:**
- Produces: HDF5 initial field for a weak compact-support plane acoustic packet
- Produces: `reflection_metrics.json`, `characteristic_history.csv`, `reflection.eps`, `reflection.jpeg`
- Consumes: complete-RK field snapshots and the curvilinear grid

- [x] **Step 1: Write failing unit tests for the acoustic definitions**

Test a pure outgoing plane wave:

```python
def test_outgoing_wave_has_zero_incoming_invariant() -> None:
    p_prime = np.array([1.0e-4])
    rho0 = 1.0
    c0 = 1.0
    un_prime = p_prime / (rho0 * c0)
    w_plus, w_minus = acoustic_invariants(p_prime, un_prime, rho0, c0)
    assert np.allclose(w_plus, 2.0 * p_prime)
    assert np.allclose(w_minus, 0.0)
```

Also test a pure incoming wave, zero denominator rejection, mismatched time
windows, non-finite fields, and EPS/JPEG path generation.

- [x] **Step 2: Run the unit tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_acoustic_tools.py
```

Expected: import failure because the generator and analyzer do not exist.

- [x] **Step 3: Implement the weak acoustic initial field**

Use a base state `(rho0,p0,T0,u0,v0,w0)` and the accepted plane-y compact
pressure perturbation

```python
p_prime = amplitude * np.cos(0.5 * np.pi * distance / width)**4
p_prime[np.abs(distance) >= width] = 0.0
rho_prime = p_prime / c0**2
velocity_prime = p_prime[..., None] * direction / (rho0 * c0)
```

Require `amplitude/p0 <= 1e-3`, a normalized direction with positive upward
component, positive reconstructed density/pressure/temperature, and a packet
center far enough from all physical boundaries for the configured measurement
window. Reject the case before file output unless the compact support spans at
least 15 intervals on the generated physical y coordinates.

- [x] **Step 4: Implement the reflection analyzer**

At the fixed computational probe plane below `jm`, construct the local physical
normal from grid metrics and calculate:

```python
w_plus = p_prime + rho0 * c0 * un_prime
w_minus = p_prime - rho0 * c0 * un_prime
e_plus = integrate_surface_time(w_plus**2 / (4.0 * rho0 * c0**2))
e_minus = integrate_surface_time(w_minus**2 / (4.0 * rho0 * c0**2))
reflection = np.sqrt(e_minus / e_plus)
```

The analyzer must require explicit incident and reflected time windows, probe
index, base state, and physical surface measure. It stops on missing metadata,
non-finite data, non-positive `e_plus`, or overlapping windows.

- [x] **Step 5: Implement required figure output**

Use exactly:

```python
import scienceplots

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.0
plt.rcParams.update({
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})
```

Do not set a font family or title. Save `reflection.eps` and
`reflection.jpeg`. Keep `OUTPUT_DIR` as an explicit path variable near the top
of the script.

- [x] **Step 6: Pass unit tests**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_acoustic_tools.py
```

Expected: all tests pass.

- [x] **Step 7: Run matched NP=1 acoustic cases on three grids**

Run:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_nscbc52_acoustic_np1 \
GRID_LEVELS='64,48,64;80,60,80;96,72,96' \
  tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh
```

The driver runs CPU/GPU `nonreflecting` and matched `compatibility` controls.
Expected for every grid:

```text
R_nonreflecting <= 0.05
R_nonreflecting <= 0.25 * R_compatibility
abs(R_cpu - R_gpu) <= 1e-3
```

Require the three nonreflecting values to be non-increasing with refinement.
If the absolute reflection gate passes but the relative control gate fails,
stop and ask whether the control definition should change; do not tune the
boundary or measurement window solely to force the ratio.

Measured CPU/GPU `nonreflecting` reflection coefficients were
`0.01620257924`, `0.00662146177`, and `0.00393554835`. The matched CPU
`compatibility` ratios were `0.17063313018`, `0.11807481438`, and
`0.10561771228`. The maximum CPU/GPU reflection difference was
`1.88e-12`, and the maximum matched final-field difference was `2.20e-13`.

### Task 6: Close the MPI Topology Matrix

**Files:**
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh`
- Modify: `tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh`
- Modify: `tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh`

**Interfaces:**
- Consumes: `NP`, `TOPOLOGY`, `OUT_DIR`, `FIELD_ATOL`, `REFLECTION_ATOL`
- Produces: one summary row per topology and a nonzero exit status on the first failed gate

- [x] **Step 1: Write the matrix with exact topology ownership**

Use:

```bash
MATRIX=(
  '1:1,1,1'
  '2:2,1,1'
  '2:1,2,1'
  '2:1,1,2'
  '4:2,2,1'
  '4:2,1,2'
  '4:1,2,2'
  '8:2,2,2'
)
```

Each case gets a separate output directory. A failed case is reported with its
topology and terminates the correctness matrix; it is not silently skipped.

- [x] **Step 2: Add boundary-owner assertions**

For each run, parse the rank logs and require:

```text
number of GCBC-owning ranks = isize * ksize
all owners have jrk = jsize - 1
all nonowners launch zero upper-face GCBC kernels
```

- [x] **Step 3: Run the uniform-flow matrix**

Run:

```bash
CASE=uniform MAXSTEP=10 \
OUT_DIR=tests/gpu_validation/out/curvilinear_nscbc52_uniform_matrix \
  tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh
```

Expected: matched CPU/GPU field errors and CPU topology differences are all
`<=1e-10`.

- [x] **Step 4: Run the acoustic matrix**

Run the medium acoustic grid for all topologies:

```bash
CASE=acoustic GRID=64,48,64 \
OUT_DIR=tests/gpu_validation/out/curvilinear_nscbc52_acoustic_matrix \
  tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh
```

Expected: CPU/GPU reflection differences and differences from NP=1 are
`<=1e-3`; field comparisons remain `<=1e-10` at the matched checkpoint.

All eight topologies passed. The largest CPU/GPU reflection difference was
`4.19e-13`, the largest matched final-field difference was `2.21e-13`, and
the CPU/GPU topology spreads were `1.32e-12` and `1.13e-12`. Owner records
matched `isize*ksize`, and only ranks with `jrk=jsize-1` owned the upper face.

### Task 7: Close Memory Safety and Runtime Work Elimination

**Files:**
- Create: `tests/gpu_validation/run_curvilinear_nscbc52_memcheck.sh`
- Modify: `tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py`

**Interfaces:**
- Produces: NP=1 and NP=2 Compute Sanitizer logs
- Produces: one Nsight Systems report and text audit of forbidden work

- [x] **Step 1: Add the sanitizer driver**

Run the one-step uniform case under `compute-sanitizer --tool memcheck` for:

```text
NP=1 TOPOLOGY=1,1,1
NP=2 TOPOLOGY=1,2,1
```

Use the established OpenMPI transport isolation already documented by the
repository. Preserve one log per rank.

- [x] **Step 2: Run Compute Sanitizer**

Run:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_nscbc52_memcheck \
  tests/gpu_validation/run_curvilinear_nscbc52_memcheck.sh
```

Expected: every rank reports `ERROR SUMMARY: 0 errors` and exits zero.

- [x] **Step 3: Capture a short Nsight Systems trace**

Run NP=1 for two complete RK steps with CUDA tracing. ASTR includes the initial
step in `do while(nstep<=maxstep)`, so the driver sets `maxstep=RK_STEPS-1`.
Export the report to SQLite and reject if the run contains any of:

```text
nscbc_farfield_y_upper_mach2_partial_kernel
nscbc_farfield_y_upper_filter_x_kernel
nscbc_farfield_y_upper_filter_z_kernel
```

Require exactly one `nscbc_farfield_y_upper_nonreflecting_rhs_kernel` launch per RK substage
owned by the upper-face rank.

The retained trace contains six non-reflecting RHS launches and zero forbidden
Mach/x-filter/z-filter launches. The bctype=52 RK snapshot remains enabled to
preserve CPU/GPU phase semantics; only the legacy boundary-plane filters are
excluded from the new mode.

- [x] **Step 4: Audit explicit synchronization**

The static test must pair every new production kernel launch with its named
`sync_after_kernel` call. Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py
```

Expected: pass.

### Task 8: Run Final Regressions and Record Evidence

**Files:**
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`

**Interfaces:**
- Consumes: raw reports from Tasks 1-7
- Produces: evidence-bounded capability statement and reproduction commands

- [x] **Step 1: Run focused Python and static tests**

Run:

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_curvilinear_nscbc_nonreflecting_contract.py \
  tests/gpu_validation/test_curvilinear_acoustic_tools.py \
  tests/gpu_validation/test_curvilinear_freestream_tools.py
```

Expected: all selected tests pass.

- [x] **Step 2: Rebuild from the top-level CMake file**

Run:

```bash
cmake -S . -B build_cpu_probe -DASTR_WITH_CUDA=OFF -DBUILD_TESTING=OFF
cmake --build build_cpu_probe -j2
cmake -S . -B build_gpu_probe -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=ON -DBUILD_TESTING=OFF
cmake --build build_gpu_probe -j2
```

Expected: both builds complete without new warnings attributable to this
feature.

- [x] **Step 3: Run the final focused regression set**

Run:

```bash
tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh
tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh
tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh
tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh
tests/gpu_validation/run_curvilinear_nscbc52_memcheck.sh
MAXSTEP=1 tests/gpu_validation/run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh
MAXSTEP=1 tests/gpu_validation/run_s2_hbl_nscbc52_incoming_only_compare.sh
MAXSTEP=1 tests/gpu_validation/run_s2_sbli_physical_nscbc_compare.sh
tests/gpu_validation/run_curvilinear_open_boundary_rejects.sh
```

Expected: every gate passes. Preserve raw logs and metric reports under the
explicit `OUT_DIR` trees; do not stage generated fields, profiles, SQLite
traces, or plot artifacts.

- [x] **Step 4: Update documentation from measured results**

Record exact grids, steps, topologies, maximum field errors, uniform-flow
drifts, reflection coefficients, sanitizer summaries, and trace findings.
State only:

```text
Validated: static single-block, non-reacting, inviscid, curved upper-eta
source-balanced non-reflecting GCBC for the tested NP=1/2/4/8 matrix.

Not validated: viscous GCBC, target-state relaxation on curved faces, general
six-face open boundaries, moving/multiblock grids, chemistry, or production
SBLI farfield fidelity.
```

- [x] **Step 5: Final worktree audit without Git mutation**

Run:

```bash
git status --short
git diff --check
git diff -- src/bc.F90 src/mainloop.F90 src/test.F90 \
  src_gpu/nscbc_characteristic_policy_gpu.cuf \
  src_gpu/boundary_gpu.cuf src_gpu/mainloop_gpu.cuf \
  src_gpu/case_capability_gpu.cuf src/CMakeLists.txt \
  tests/gpu_validation documents docs/superpowers
```

Expected: no whitespace errors, no generated result files in the intended
source set, and no unrelated file reverted or overwritten. Report completion
without committing or pushing.
