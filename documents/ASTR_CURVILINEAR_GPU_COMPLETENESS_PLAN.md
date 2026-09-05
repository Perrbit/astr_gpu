# ASTR Static Curvilinear GPU Completeness Plan

## 1. Scope

This closure targets static, single-block, three-dimensional structured
curvilinear grids for the nonreacting five-equation ideal-gas solver with
Sutherland viscosity, explicit spatial schemes, RK3, fixed-`hm` halos, and
explicit synchronization after every CUDA kernel.

Included acceptance topologies are NP=1, NP=2 slabs, NP=4 planes, and NP=8
`2x2x2`. CPU and GPU complete-RK fields should generally agree within
`1e-10`; each physical boundary also needs its own invariant oracle. Jacobian
positivity, finite fields, Compute Sanitizer, and no-checkpoint residency are
separate gates.

Moving/ALE/GCL grids, multiple blocks, nonconforming interfaces, immersed
boundaries, compact schemes, GPU HDF5, species, chemistry, RANS, and LES are
outside this closure.

## 2. Evidence Rules

1. Match complete-RK phases. CPU production checkpoints are not substituted
   for `ASTR_VALIDATION_RK_SNAPSHOT` when their boundary preparation differs.
2. Validate boundary physics independently of CPU/GPU field equality.
3. Use geometric projection for physical no-penetration conditions on curved
   faces.
4. Preserve only CPU-defined directions and combinations. Missing branches
   are explicit rejects, not inferred support.
5. Stop for an explicit decision when CPU behavior is erroneous, ambiguous,
   or incompatible with the stated physical boundary type.

## 3. Capability Status

| Capability | Current status | Required remaining evidence |
|---|---|---|
| Periodic metrics, convection, diffusion, filter | complete through C0-C4 | retain regression |
| `60` symmetry, physical projection | complete through C13 on six faces | retain regression |
| `41` isothermal no-slip | zero-blowing six-face subset complete through C16; y-min physical-normal blowing complete through C19 | retain regression; x/z blowing is outside the CPU-defined route |
| `50` zero extrapolation | complete through C17 on six faces | retain regression |
| `42` adiabatic no-slip | zero-blowing x/y subset complete through C18; y-min physical-normal blowing complete through C19 | retain z reject |
| `411/421` slip families | y-face geometric semantics complete through C19 | retain x/z rejects because CPU has no branches |
| Optional y-min wall blowing | physical-normal, global-index-deterministic CPU/GPU route complete through C19 | retain topology-invariance regression |
| `11/12/21/22/51/52` and x-max sponge | complete through C20 for the documented restricted contracts | retain inlet-state and curved-open-boundary reject regressions |
| Ducros/selective Roe/MP7/Sutherland | complete for controlled C9-C12 contracts | retain shock/filter negative gate |
| Geometry, MPI, residency, performance | complete through C21 for the tested static single-block contracts | retain aggregate regression before release |

## 4. Remaining Sequence

### C19: Slip-wall and wall-blowing contract (complete)

CPU and GPU now extrapolate all three velocity components on the y-face slip
sections and project the result onto the local discrete tangent plane,
`u_t=u*-(u* dot n)n`. The `411` no-slip section remains zero velocity. The
`421` lower section combines tangential slip with prescribed normal blowing;
its upper section is tangential slip without blowing. No x/z `411/421`
branches were invented because CPU defines only y directions.

The signed `wallbs.dat` amplitude retains the CPU nondimensional scaling
`vwall=A uinf fx gz (1+r)`, and the resulting scalar is applied as
`u_wall=vwall n` on y-min `41/42` and on the applicable `411/421` sections.
Positive amplitude follows the stored inward unit normal. The 10%
perturbation uses global i/k indices and a fixed integer hash, so the same
physical nodes receive the same forcing for NP=1/2/4/8. The 15-entry y-wavy
matrix passes CPU/GPU, invariant, and topology checks. Maximum CPU/GPU field,
statistic, and topology differences are `8.5265128291212022e-14`,
`4.6865289426989420e-14`, and `5.6843418860808015e-14`; the maximum NP=1
normal residual is `6.0281640790194047e-17`. Five-step filter-plus-diffusion
checks pass for `411` and blowing `421`, and the latter reports zero memcheck
errors. The matrix also includes negative-amplitude suction. A three-step
no-checkpoint trace contains no transfer at or above
64 KiB after the first filter kernel.

### C20: Open-boundary inventory (complete)

| Type | CPU-defined directions | C20 classification and GPU contract |
|---|---|---|
| `11` | x-min | Prescribed global primitive state, not a characteristic condition. CPU now writes `rho_in/vel_in/prs_in/spc_in` and reconstructs temperature from the EOS; GPU updates all resident primitive and conservative fields. A deliberately stale `ninit=3` field verifies replacement of the complete x-min state. |
| `12` | x-min | Metric characteristic projection, but with case-specific relaxation. Retained only in the validated Cartesian OpenShock pair and C12 `12/21,41/52` composition. |
| `21` | x-max and y-max | Computational-direction outlet whose pressure correction is not a general physical-normal correction. Current GPU cases use x-max only, and C20 rejects a non-axis-aligned physical outlet face before CUDA launch. |
| `22` | x-min, x-max, y-max, and z-max | Direction branches are incomplete and only the Cartesian x pair has dedicated GPU evidence. C20 rejects `22` whenever the grid has nonzero off-diagonal inverse metrics. |
| `51` | y-min, y-max, z-min, and z-max | Uses Cartesian component logic and contains case-specific branches. GPU retains only the upper-y boundary-layer composition and requires its physical face to be axis aligned. |
| `52` | y-min, y-max, z-min, and z-max | Metric NSCBC implementation with case-dependent policy. GPU retains only the validated upper-y incoming-wave C12 composition; no arbitrary-direction claim is made. |

The `bctype=11` stale-state gate passes with maximum x-min primitive error
`4.4408920985006262e-15`, EOS residual `2.2204460492503131e-16`, and CPU/GPU
full-field `q5` error `7.1054273576010019e-15`. Dedicated nonorthogonal tests
reject curved `21`, nonorthogonal `22`, and curved `51`. Cartesian `22`, C7
`11/21,41/51`, and C12 `12/21,41/52` one-step regressions pass without
relaxing thresholds. The targeted `bctype=11` Compute Sanitizer run reports
`ERROR SUMMARY: 0 errors` with the established OpenMPI `ob1/self/pt2pt`
isolation. C20 does not standardize the deferred CPU NSCBC policy.

### C21: Aggregate closure (complete)

The reusable `run_curvilinear_c21_aggregate.sh` driver rebuilt the CPU and
GPU executables from the same source tree and ran all C0-C20 supported
matrices, the C11 closed-negative policy check, and the required curved-open
boundary rejects. All 23 aggregate stages passed. The run produced 136 field
reports, 128 statistic reports, 226 boundary-invariant reports, and 70
explicit finite-field checks.

The aggregate maximum CPU/GPU field error is
`q5 L_inf=6.2527760746888816e-13`. The maximum statistic difference is
`massflux=2.5093260802577788e-11`, and the maximum required boundary residual
is the C17 extrapolated pressure residual
`7.6170181273482740e-12`. The minimum checked numerical Jacobian is positive
at `2.5296638468231652e-7`. Representative x/y/z Compute Sanitizer runs each
report zero errors.

The `256^3` no-checkpoint trace contains 221 kernels after the selected first
RHS kernel and no H2D/D2H transfer at or above 64 KiB. H2D is limited to two
176-byte operations and D2H to two 48-byte operations. Three-repeat timing
gives GPU NP=1/NP=2 median wall times of `58.490/38.989 s`, or `1.5002x`
two-GPU speedup and `75.01%` parallel efficiency on the current machine.

The complete evidence is recorded in
`documents/ASTR_CURVE_C21_REGRESSION_REPORT.md` and
`documents/GPU_VALIDATION_MATRIX.md`.

## 5. Completion Condition

Static single-block CURVE support is complete only when every in-scope,
CPU-defined unambiguous boundary and numerical composition is either covered
by a passing reusable gate or rejected before launch with a documented
reason. A CPU/GPU match to a physically invalid curved boundary is not a
passing gate.

CURVE-C21 satisfies this completion condition for the scope in Section 1.
