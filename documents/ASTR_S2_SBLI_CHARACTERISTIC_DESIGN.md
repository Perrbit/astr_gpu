# ASTR S2 Characteristic-Boundary And Shock-Sensor-Coupled SBLI Design

## 1. Scope

This document designs the next Phase S2 path after the current oblique-shock HBL
compatibility gates. The target is a controlled, single-species, non-reacting,
laminar or DNS-like shock-boundary-layer interaction path with:

- explicit MP7/WENO-family shock-capable convection;
- Ducros/pressure-curvature shock sensing;
- selective Roe characteristic reconstruction only on sensor-active interfaces;
- simple boundary conditions first, with characteristic inflow/outflow/farfield
  retained as later case-specific branches where they are actually needed;
- x-max sponge as an optional stabilizing layer;
- no compact finite-difference, compact filter, RANS/LES, chemistry, combustion,
  multi-species transport, GPU HDF5 write, or checkpoint write support.

The design is a GPU porting and validation plan. It is not a claim that the
current S2-A0/S2-B1 cases are physical SBLI validations.

Maintainer feedback for the current phase is that farfield and NSCBC details
vary strongly by case and should not block the common GPU porting path. The
near-term plan therefore keeps the simplest already validated boundary contract
and prioritizes generic shock-sensor, selective-Roe, halo, source, and statistics
plumbing. The stricter characteristic-boundary design below is preserved as a
later branch, not the next implementation dependency.

## 2. Current Building Blocks

### 2.1 Shock-Sensor And Selective Roe Path

Current GPU support already includes:

- `src_gpu/shock_sensor_gpu.cuf`
  - `raw_shock_sensor_kernel`
  - `exchange_field_halo_gpu(shock_sensor_d,1)`
  - `expand_shock_sensor_kernel`
- `src_gpu/solver_gpu.cuf`
  - `characteristic_reconstruction_interface_flux`
  - x/y/z characteristic interface-flux kernels
  - one reusable `flux_characteristic_work_d`
- validation gates S0-A4 through S0-A10 for sensor-only, sensor halo, and
  selective Roe MP7 characteristic reconstruction under periodic Shu-Osher.

The important ownership rule remains unchanged: compute the expanded cell mask
once per RK substage, then form interface activity inside each directional flux
kernel as the adjacent-cell OR. Do not introduce separate x/y/z interface masks
in the baseline.

### 2.2 Characteristic Boundary Building Blocks

Current CPU and GPU code already contain restricted characteristic-boundary
pieces:

- CPU:
  - `bc:inflow_nscbc`
  - `bc:outflow_nscbc`
  - `bc:farfield_nscbc`
- GPU:
  - `nscbc_pmatrix`
  - `nscbc_inflow_x_rhs_kernel`
  - `nscbc_outflow_x_rhs_kernel`
  - `nscbc_farfield_y_upper_rhs_kernel`
  - `gpu_nscbc_*_mach2` reductions

The existing GPU support is intentionally narrow: x `12/22` OpenShock and upper
y `52` HBL-style farfield. It is not yet a general characteristic-boundary
framework.

### 2.3 Existing S2 Gates

- S2-A0: x-varying Mach-5 Blasius HDF initial field plus analytic oblique-shock
  overlay.
- S2-B1: same field with an optional fifth pressure column in `inlet.prof`, so
  the upper inlet profile continuously injects a compressed state.

Both are CPU/GPU compatibility gates. They do not generate a physical incident
shock through a characteristic boundary and do not consume the shock sensor in
the HBL/SBLI path.

## 3. Design Principles

1. Keep CPU/GPU oracle equivalence, but do not blindly port known questionable
   CPU behavior. A confirmed CPU bug or physically questionable boundary rule
   must go through the project decision gate before implementation.
2. Start from Cartesian, non-dimensional, single-species cases. Curvilinear
   metrics, dimensional gas models, and species are later gates.
3. Keep the sensor-coupled shock format and characteristic/open boundaries as
   separate toggles until each pair has a CPU/GPU oracle. In the near term,
   implement the common shock-format path under the simplest boundary contract
   before reopening farfield-specific logic.
4. Every launched kernel remains followed by explicit synchronization, matching
   the project rule.
5. Device-resident variables remain the default. Host transfers are limited to
   validation dumps, scalar reductions, and the existing host-staged MPI halo
   backend where no device-aware path is implemented.
6. S2 validation must distinguish:
   - porting equivalence against CPU;
   - shock-format numerical accuracy;
   - physical SBLI credibility.

## 4. Boundary-Condition Policy

### 4.1 Incoming-Wave Rule

The later physical-boundary target design is:

- outgoing characteristic waves are computed from the interior and left
  unchanged;
- only incoming characteristic waves are prescribed or relaxed toward a target
  state;
- supersonic inflow prescribes all incoming physical state variables;
- supersonic outflow should not pressure-relax the interior state;
- subsonic outlet/farfield may relax the incoming acoustic wave to a target
  pressure or farfield state.

The current CPU `farfield_nscbc` has commented flow-direction branches and
unconditional acoustic overwrite paths. Per maintainer feedback, this is not a
near-term blocker because boundary behavior is case-specific; the current GPU
port should avoid depending on this farfield path and keep the simplest boundary
contract until a selected case requires more.

### 4.2 Normal-Mach Reduction

For future production characteristic boundaries, the relaxation coefficient
should be based on the maximum normal Mach number of the active boundary face,
not a global whole-domain Mach unless the selected CPU oracle deliberately uses
that global reduction. The initial CPU/GPU compatibility gates may keep the CPU
formula for equivalence, but the physical SBLI branch should expose this as a
separate decision:

```text
compatibility mode:
  match CPU reduction exactly

physical SBLI mode:
  reduce normal Mach over the active boundary face only
```

### 4.3 Inlet State For Incident-Shock SBLI

The physically meaningful next inlet options are:

1. **Characteristic profile inflow with pressure-provided target state**:
   use the existing five-column `inlet.prof` contract as target data for
   incoming waves, not as a hard overwrite of all variables at every substage.
2. **Incident-shock boundary state generator**:
   compute pre-shock/post-shock target states from Mach number, shock angle, and
   inlet intersection height, then provide target values to the characteristic
   inflow kernel.

The first implementation should use option 1 because the profile reader and
validation scripts already exist. Option 2 is the later physics-facing wrapper.

### 4.4 Farfield

For the current flat-plate/SBLI geometry, y-min is the wall and y-max is the
upper boundary. The near-term route should prefer the simplest validated upper
boundary when the selected gate allows it. If a later case truly requires a
farfield, that farfield step is not "general 52"; it is:

- upper y only;
- Cartesian metric first;
- non-reacting five-equation state;
- target pressure/farfield state from `pinf` or the supplied profile/farfield
  state;
- incoming-wave-only correction in physical SBLI mode.

The existing transverse sixth-order explicit filter around CPU `52` remains a
compatibility risk. It should not be on the critical path for the common
shock-sensor/selective-Roe implementation. If retained for a later case, it must
be validated separately from the NSCBC wave update. If a future physical SBLI
branch disables this filter, document it as a CPU-oracle divergence before
implementation.

### 4.5 Outlet And Sponge

The recommended S2 outlet progression is:

1. x `21` classical outflow for compatibility gates.
2. x `22` NSCBC outflow for characteristic-boundary gates.
3. x-max sponge layer plus x `22` for longer SBLI runs.

Do not combine new characteristic inflow, y-farfield, selective Roe, and sponge
in the same first gate. Under the current maintainer guidance, the first gate
should avoid new characteristic/farfield work entirely unless a selected case
cannot run with the simpler boundary contract.

## 5. GPU Execution Sequence

The target S2 shock-sensor-coupled RK substage sequence is:

```text
q_to_primitive_kernel
explicit synchronization
apply physical primitive/q boundary states needed before halo exchange
explicit synchronization
exchange_solution_halo_gpu

zero_rhs_kernel
explicit synchronization

if characteristic boundary is enabled:
  optional NSCBC boundary-plane filter compatibility step
  explicit synchronization
  q_to_primitive_kernel
  explicit synchronization
  exchange_solution_halo_gpu
  boundary normal-Mach reduction
  characteristic boundary RHS kernels
  explicit synchronization

gradcal_gpu
explicit synchronization
raw_shock_sensor_kernel
explicit synchronization
exchange_field_halo_gpu(shock_sensor_d,1)
expand_shock_sensor_kernel
explicit synchronization

adaptive selective-Roe interface flux x
explicit synchronization
flux difference x
explicit synchronization
adaptive selective-Roe interface flux y
explicit synchronization
flux difference y
explicit synchronization
adaptive selective-Roe interface flux z
explicit synchronization
flux difference z
explicit synchronization

diffusion/source/sponge as allowed by the current gate
explicit synchronization after each kernel
RK update
```

This preserves the current dependency boundary: the sensor uses complete
gradients and current halos before mask expansion, and the characteristic flux
kernel consumes the completed mask.

## 6. Capability Gates

### S2-C0: Boundary Policy Audit Gate

Purpose: record the boundary-policy decision before implementation.

Tasks:

- Document that detailed `farfield_nscbc` and case-specific NSCBC policy are
  deferred for the common GPU path.
- Keep simple already validated boundary conditions for the first generic SBLI
  coupling gates.
- Preserve CPU-compatible versus physical SBLI boundary modes as later design
  branches, not immediate implementation requirements.

Acceptance:

- No code path depends on the known questionable CPU unconditional acoustic
  overwrite for the next common-path gate.
- The selected validation scripts state that they use the simple-boundary
  contract rather than true characteristic farfield behavior.

### S2-C1: Characteristic Profile Inflow Oracle

Purpose: replace hard profile overwrites at x-min with a controlled
characteristic profile-inflow RHS.

Initial case:

- `flowtype=bl`
- x-min `12`
- x-max still `21`
- y `41/51` or y `41/52` kept unchanged
- no `lchardecomp`
- no sponge
- five-column pressure-provided inlet profile

Implementation:

- Add a GPU capability function such as
  `gpu_s2_hbl_profile_nscbc_inflow_supported`.
- Add an x-min profile-target variant of `nscbc_inflow_x_rhs_kernel`.
- Upload `rho_prof/vel_prof/tmp_prof/prs_prof` as already done for S1/S2
  profile inflow.
- In compatibility mode, match CPU `inflow_nscbc` target-wave formulas.
- In physical mode, apply the incoming-wave rule.

Validation:

- NP=1 two-step field/statistics.
- NP=2 x/y/z slabs.
- Pressure-provided profile scatter check remains mandatory.

### S2-C2: Upper Farfield Incoming-Wave Gate

Purpose: isolate y-upper `52` as a characteristic farfield under the HBL/SBLI
profile setup.

Initial case:

- x-min either `11,prof` or S2-C1 x `12`
- x-max `21`
- y-min `41`
- y-max `52`
- no shock-sensor-coupled convection yet

Implementation:

- Keep the existing GPU upper-y `52` compatibility kernel as the CPU-oracle
  path.
- Add a named physical branch only after the incoming-wave policy is approved.
- Keep the current pre-boundary halo refresh requirement.

Validation:

- NP=1 20-step statistics and field.
- NP=2 y-slab and z-slab, because y owns the physical farfield and z exercises
  transverse halos.
- If the filter is enabled, add a constant-field preservation oracle before
  accepting production use.

### S2-C3: HBL With Selective Roe Sensor, No New Characteristic Boundary

Purpose: couple the shock sensor and selective Roe characteristic flux to the
current S2-B1 sustained compressed-inlet case while keeping boundaries already
validated.

Current implementation status:

- The first simple-boundary CUDA path is implemented for `bl`, `543e/643e`,
  `lchardecomp=t`, `lfilter=f`, `diffterm=f`, `11,prof/21/41/51`, and
  pressure-provided sustained compressed inlet profiles.
- Passing gates cover NP=1, all three NP=2 slab orientations, all NP=4
  two-axis orientations (`2x2x1`, `2x1x2`, `1x2x2`), and NP=8 `2x2x2`.
- The `64x64x8` NP=1 and `64x64x16` NP=8 `2x2x2` simple-boundary gates also
  pass 20 and 100 steps at `1e-10`. Final conservative-field differences are
  `q5 L_inf=2.00e-15/3.33e-15` for NP=1 and `1.33e-15/4.88e-15` for NP=8.
- The former y-slab discrepancy was caused by `steger_warming_split_vector_at`
  periodizing `jacob_d` while reading physical/MPI state halos. Its scalar
  counterpart correctly reads the exchanged geometry halo directly. The vector
  path now follows that rule; one- and three-step y-slab full-field gates pass
  at `1e-10`.
- Diffusion, filtering, NSCBC/farfield, and sponge remain outside this
  completed slice; the multi-rank topology matrix is complete only for the
  stated simple-boundary contract.

Initial case:

- S2-B1 setup
- `conschm='543e'`
- `recon_schem=3`
- `lchardecomp=t`
- simplest validated boundary contract first, preferably x `11,prof/21`,
  y `41/51`, periodic z when this is sufficient for the gate
- no diffusion first, then diffusion after equivalence is proven
- no global filter

Implementation:

- Add a capability function for `flowtype='bl'`, pressure-provided profile,
  S2-B1 boundaries, `lchardecomp=t`, and local extents `>= hm`.
- Reuse `compute_shock_sensor_gpu`.
- Reuse the S0-A6 through S0-A10 characteristic flux kernels, but extend their
  physical-boundary active-range handling to x/y physical HBL ranges.
- For the first C3 gate, use the existing shock mask dump mechanism so CPU/GPU
  raw sensor and local masks are compared rankwise.

Validation:

- NP=1 one-step sensor/mask plus field.
- NP=1 three-step field/statistics.
- NP=2 x/y/z slabs with rankwise sensor/mask comparison.
- NP=4 `2x2x1`, `2x1x2`, and `1x2x2`, followed by NP=8 `2x2x2`.

### S2-C3-S: Simple-Boundary x-Max Sponge

Purpose: exercise the existing device-resident x-max sponge with the completed
C3 common path without introducing unresolved characteristic-boundary policy.

Current implementation status:

- The gate retains the C3 `11,prof/21/41/51` boundary contract and permits
  only `spg_def='layer'`, positive `spg_im`, and zero sponge extents elsewhere.
- After each RK update, GPU exchanges q halo data, applies the existing
  two-kernel `qwork_d` sponge ping-pong with explicit synchronization, then
  refreshes primitive variables, matching CPU `spongefilter` ordering.
- `SPONGE_IM=16` passes 1, 20, and 100 steps at `1e-10` for NP=1
  (`64x64x8`) and NP=8 `2x2x2` (`64x64x16`). The final 100-step `q5 L_inf`
  values are `7.5495165674510645e-15` and `8.6597395920762210e-15`.

This is a common-path sponge equivalence gate only. It is not C4/C5, does not
validate a characteristic boundary, and does not authorize y/z/circular sponge
or unvalidated NP=2/4 sponge decompositions.

### S2-C4: Characteristic Inflow/Farfield Plus Selective Roe

Purpose: combine C1, C2, and C3 after the common simple-boundary selective-Roe
path has passed independently and a selected case actually requires
characteristic boundaries.

Initial case:

- x-min characteristic profile inflow
- x-max classical `21`; `22` remains deferred
- y-min `41`
- y-max `52`
- S2-B1 or incident-shock target state
- `lchardecomp=t`

Validation:

- CPU/GPU full fields and statistics at `1e-10`.
- NP=1, NP=2 slabs, NP=4 two-axis planes, and NP=8 `2x2x2` smoke
  coverage; 100-step NP=1 and NP=8 long-run closure.
- Compare statistics, wall heat flux, and sensor activity counts.
- Report whether sensor-active interfaces remain localized near the shock and
  interaction region.

Current implementation status:

- The controlled `12/21/41/52` route is implemented for the explicit
  Cartesian `543e/643e` MP7 Roe contract with `lchardecomp=t`, `lfilter=f`,
  `diffterm=f`, pressure-provided compressed profile inflow, and no sponge.
- The GPU upper-y NSCBC path matches CPU transverse-filter ordering. Its
  farfield normal-Mach reduction now contributes only from ranks that own the
  true upper-y physical face; including an internal y-slab interface caused a
  long-step CPU/GPU divergence and is invalid.
- The final 100-step NP=8 `2x2x2` comparison passes at `1e-10`, with
  `q5 L_inf=9.3258734068513149e-15` and maximum statistic difference
  `5.9863225487788441e-12`.

### S2-C5: Sponge-Stabilized Long Run

Purpose: make longer SBLI-style runs possible after characteristic boundaries
and selective Roe are already coupled.

Initial case:

- x-max sponge only.
- no y/circular sponge.
- no chemistry/species/RANS/LES.

Validation:

- NP=1 and NP=8 100-step stats/field at finite tolerances.
- Sponge q-only halo refresh must be validated with z/y decompositions.

Current implementation status:

- `run_s2_hbl_selective_roe_s2c5_sponge_compare.sh` enables only the
  existing x-max `layer` sponge with `SPONGE_IM=16`; all other sponge extents
  remain zero. The C5 predicate accepts only NP=1 and NP=8 `2x2x2`.
- The GPU sponge keeps the device-resident two-kernel `qwork_d` ping-pong and
  explicit synchronization. Its update range is restricted to the CPU active
  `is:ie`, `js:je`, `ks:ke` domain rather than the allocation extent.
- The strict 100-step CPU/GPU field and statistics gates pass for NP=1 and
  NP=8. Final `q5 L_inf` is `8.8817841970012523e-15` for NP=1 and
  `9.3258734068513149e-15` for NP=8; the latter maximum statistic difference
  is `5.9636739990764909e-12`.

## 7. Required Code Changes

The expected implementation files are:

- `src_gpu/case_capability_gpu.cuf`
  - add S2-C capability predicates.
- `src_gpu/mainloop_gpu.cuf`
  - route S2-C characteristic boundary and selective Roe combinations.
- `src_gpu/boundary_gpu.cuf`
  - add profile-target characteristic inflow kernel;
  - split compatibility and physical incoming-wave-only boundary modes;
  - add face-local normal Mach reduction helpers if physical mode is accepted.
- `src_gpu/solver_gpu.cuf`
  - extend selective Roe physical active ranges to HBL/SBLI x/y physical faces.
- `src_gpu/shock_sensor_gpu.cuf`
  - extend physical-face sensor clamping beyond the current S0-B x-face gate if
    y-wall/farfield masks require it.
- `tests/gpu_validation/`
  - add S2-C scripts for C1 through C5;
  - add rankwise sensor/mask comparison for HBL/SBLI gates;
  - reuse `generate_compressible_blasius_profile.py`.
- `documents/GPU_VALIDATION_MATRIX.md`
  - add rows only after each gate passes.

CPU changes should be limited and decision-gated. If a confirmed CPU bug is
found, report it first and choose either:

- fix CPU/common oracle first, then port; or
- add a named compatibility branch and keep the physical branch separate.

## 8. Validation Matrix

| Gate | Boundary | Sensor/flux | MPI | Expected evidence |
|---|---|---|---|---|
| S2-C1 | x `12`, x `21`, y `41/51` | MP7 physical-space | NP=1, NP=2 slabs | deferred characteristic-inflow oracle |
| S2-C2 | x `11/21`, y `41/52` | MP7 physical-space | NP=1, NP=2 y/z | deferred farfield oracle |
| S2-C3 | S2-B1/simple boundaries | sensor + selective Roe | NP=1, all NP=2 slabs, all NP=4 two-axis slabs, NP=8 `2x2x2` | rankwise sensor/mask plus field/statistics |
| S2-C3-S | S2-C3 plus x-max layer sponge | sensor + selective Roe + sponge | NP=1, NP=8 `2x2x2` | 1/20/100-step field/statistics |
| S2-C4 | x `12`, y `52`, x `21/22` | sensor + selective Roe | NP=1, NP=8 | later case-specific characteristic boundary gate |
| S2-C5 | C4 + x-max sponge | sensor + selective Roe | NP=1, NP=8 | 100-step finite stability and sponge halo check |

## 9. Open Decisions

1. Which later production case actually requires true characteristic farfield
   rather than the simple-boundary common path?
2. Should physical SBLI mode restore incoming-wave-only logic when that later
   case is selected, or first match CPU behavior for one compatibility gate?
3. Should y-upper `52` retain the CPU transverse explicit filter in physical
   mode?
4. Should normal-Mach relaxation use boundary-face reduction in physical mode?
5. Should incident-shock target state be generated from analytic shock angle in
   the boundary kernel, or generated into a pressure-provided profile by the
   case-preparation script?
6. What finite tolerance is acceptable for longer shock/SBLI runs after the
   exact CPU/GPU roundoff gates are passed?

## 10. Recommended Next Action

Start with S2-C3 under the simplest validated boundary contract. This follows
the maintainer guidance to avoid overfitting farfield/NSCBC behavior before the
common GPU path is useful:

- it reuses the pressure-provided inlet profile;
- it exercises the generic shock sensor and selective-Roe coupling that future
  SBLI cases need;
- it does not depend on unresolved `farfield_nscbc` policy;
- it can be validated by CPU/GPU field, statistics, and rankwise sensor/mask
  output before making any physical SBLI claim.
