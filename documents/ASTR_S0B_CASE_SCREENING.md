# ASTR S0-B Open-Boundary Case Screening

## Decision

Select a dedicated forced-3D `OpenShock` validation gate for Phase S0-B.

The gate is not a new production benchmark. It is a controlled non-reacting,
no-turbulence, no-species case that separates the GPU migration of classical
inflow/outflow, NSCBC, and sponge behavior before a high-speed boundary layer
or shock-boundary-layer interaction case is attempted.

## Selected Gate

The initial implementation uses a uniform Cartesian box with x as the streamwise
direction and periodic y/z directions. The initial state contains a stationary
normal shock at x=0. Its upstream state is identical to the reference state at
x-min; the downstream state follows the ideal-gas normal-shock relations. The
x-max target pressure must be `pout=p2/pinf`; otherwise the subsonic `21`
outlet drives the state away from its stationary downstream pressure. The existing `sodini` and
`shuosherini` initializers are not suitable because their x-min states do not
match the generic CPU inflow reference state.

Initial numerical contract:

- Forced 3-D grid with each active extent at least `hm`; initial target
  `GRID=400,8,8`.
- `conschm='543e'`, `difschm='643e'`, `recon_schem=3`, `rk3`, and
  `lchardecomp=t`.
- Before any `11/21` or `12/22` run, port the CPU-compatible MP5 physical-face
  degradation sequence for the x-direction explicit upwind reconstruction. The
  CPU `flux:recons_exp` contract is face-distance dependent: the first interface
  uses the two-point split-flux average, the second uses SUW3, the third uses
  MP5, and only subsequent interfaces use MP7. CPU `solver:iwind8` clips the
  source samples to the physical active range. The periodic MP7 stencil
  otherwise wraps through `idx_periodic` and is not a valid open-boundary
  operator.
- `diffterm=f`, `lfilter=f`, `num_species=0`, `turbmode='none'`, and no
  immersed boundary.
- x-min/x-max use `bctype=11/21` for the first classical inflow/outflow gate;
  y/z remain periodic.
- The follow-on NSCBC gate changes only x-min/x-max to `bctype=12/22`.
- The sponge gate then enables only an x-max layer. The CPU algorithm is a
  conservative-variable six-neighbor explicit smoothing operation, not a
  relaxation to a prescribed reference state.

The implementation must add a capability-specific initializer and validation
driver rather than overloading a production `flowtype` contract. The name
`OpenShock` is used only in documentation; source identifiers should use the
`s0b_open_shock` prefix to keep the controlled gate distinct from future
production shock cases.

## Candidate Screening

| Candidate | Decision | Reason |
|---|---|---|
| Dedicated forced-3D OpenShock | selected | Isolates one streamwise inflow/outflow pair, keeps y/z periodic, and has a CPU/GPU same-topology oracle. |
| `Hypersonic_Boundary_Layer` | defer to S1 | 2-D curvilinear grid, compact convection, wall plus farfield boundaries, and profile input couple several unsupported features. |
| `MixingLayer` | defer | Requires inflow, NSCBC outflow, two farfields, sponge, compact schemes, and case-specific initialization together. |
| `SWLBI` | defer to S2 | Already combines compact schemes, inlet/outlet, wall, farfield, sponge, and shock-wall physics. |
| `supersonic_backstep` | reject for S0-B | Adds immersed-boundary geometry and wall interactions. |
| `Riemann2D` | reject for S0-B | Uses UDF boundary type `0`, has no open-boundary contract, and is a 2-D configuration. |
| `Shuosher` / `sod` | not direct candidates | Current initial x-min states do not match generic CPU inflow reference data; periodic gates remain useful S0-A oracles. |

## Required Boundary Refactor

The current GPU boundary capability only accepts one physical axis with the
same boundary kind at both faces. It therefore rejects every valid inflow/outflow
pair before a GPU kernel is launched. Phase S0-B must replace this restriction
with a face-specific capability resolver:

1. Validate the physical axis independently from lower and upper face kinds.
2. Dispatch only on true physical faces (`MPI_PROC_NULL`).
3. Preserve internal MPI faces as halo-exchange interfaces.
4. Keep the existing single-kind wall/zeroextrap paths unchanged until their
   face-specific replacements pass their own regressions.
5. Apply explicit synchronization after every boundary, sponge, and RHS kernel.

`bctype=11/21` is the first implementation pair. `bctype=12/22` remains a
separate NSCBC subphase because the CPU routines depend on directional flux
derivatives, characteristic matrices, and initialized inflow profile storage.
For S0-B2, retain the CPU `outflow_nscbc` use of `spafilter6exp` as an explicit
boundary-stabilization exception. It is a sixth-order explicit outlet-plane
filter, is outside the ten-order global `lfilter` ping-pong path, and may run
when `lfilter=f`; it must not be silently replaced by the global filter.

## S0-B0 Status

The x physical explicit reconstruction prerequisite is implemented for the
controlled forced-3D Sod gate with `bctype=50,50`. The GPU x flux and RHS
kernels use `is:ie` and `npdci`, retain the opposite MPI-side `hm` halo, and
match the CPU face sequence of two-point average, SUW3, MP5, and MP7. The
20-step same-topology NP=1 and `NP=2 TOPOLOGY=2,1,1` CPU/GPU field and
statistics gates pass; the NP=2 field comparison has `q5 L_inf=5.8841820305133297e-15`
and both Compute Sanitizer ranks report zero errors. This does not yet cover
inflow/outflow behavior.

The physical-x selective-Roe prerequisite is now accepted for the forced-3D
Shu-Osher gate under the same `bctype=50,50` contract. The implementation uses
the MP5 physical-face degradation sequence for Roe reconstruction, computes the
raw Ducros sensor with physical-face pressure clamping, avoids periodic `ssf`
halo fills at physical x faces, and clamps x mask expansion only on
`MPI_PROC_NULL` faces. The CPU reference required a corresponding correctness
fix: `commcal:ducrossensor` handled `npdci=1/2` but omitted `npdci=4`, so a
single-rank physical domain read `prs(-1)` or `prs(im+1)` and expanded through
uninitialized `ssf` halo values. `npdci=4` now clamps both physical sides.

`MAXSTEP=3` passes for NP=1 and `NP=2 TOPOLOGY=2,1,1`: rankwise raw-sensor
`L_inf=1.1102230246251565e-16`, mask mismatches zero, and final full-field
`q5 L_inf=7.1054273576010019e-15`. The NP=2 GPU memcheck reports
`ERROR SUMMARY: 0 errors` on both ranks. This remains a finite-domain
zero-extrapolation prerequisite, not support for `11/21`, `12/22`, or sponge.

## Validation Sequence

1. S0-B0: validate the MP5 physical-face degradation against CPU with a
   controlled x-physical shock gate before introducing inflow/outflow behavior.
2. CPU baseline: prove the OpenShock initial state stays positive and the shock
   position remains bounded for short and long runs.
3. S0-B1: same-topology `NP=1` CPU/GPU comparison for `11/21`; compare complete
   `q1:q5`, primitive fields, statistics, and boundary-plane extrema.
4. S0-B2: repeat with `12/22` NSCBC; report the same quantities and a pressure
   probe/reflection diagnostic near x-max.
5. S0-B3: add only the x-max sponge; compare CPU/GPU at NP=1 and `2x1x1`, and
   demonstrate that the sponge does not alter the x-min imposed state.
6. S0-B4: repeat in a combined topology such as `2x2x2`; the physical x faces
   must remain on `MPI_PROC_NULL` ranks while y/z remain normal periodic halo
   exchanges.
7. Run Compute Sanitizer for each newly introduced boundary kernel and profile
   the completed S0-B gate before opening S1.

## S0-B1 Status

The classical x-face pair is implemented for the stationary forced-3D
`openshock` gate: `bctype(1:2)=11,21`, `turbinf='free'`, and Mach-3
`pout=10.333333333333333`. The new initializer applies normal-shock density
and pressure ratios at x=0; the GPU reproduces the CPU free-stream inflow and
supersonic/subsonic outflow branches only on their `MPI_PROC_NULL` faces.
Ten-step CPU/GPU comparisons pass for NP=1 (`q5 L_inf=8.8817841970012523e-16`)
and `NP=2 TOPOLOGY=2,1,1` (`q5 L_inf=9.9920072216264089e-16`). Both NP=2
Compute Sanitizer processes report `ERROR SUMMARY: 0 errors`. NSCBC, sponge,
diffusion, filtering, and characteristic Roe remain outside S0-B1.

## S0-B2 Status

The CUDA Fortran `12/22` implementation is accepted for the controlled
`openshock` contract: Cartesian,
non-dimensional Euler, no species, `543e/643e`, MP7 physical-space
reconstruction, `lfilter=f`, periodic y/z, and NP=1 or `2x1x1`. It implements
the CPU characteristic matrices, explicit second-order transverse flux terms,
separate inlet-domain and outlet-plane Mach-max reductions, and the CPU
sixth-order explicit y-plane outlet filter using the existing `qwork_d`
workspace.

The CPU reference now executes a full-rank `qswap` before `boucon` whenever a
`bctype=22` outlet is present. This refreshes the transverse halo before the
CPU outlet filter while retaining the original post-boundary `qswap`. It avoids
the prior stale-halo dependence without allowing a physical x-max rank to enter
MPI exchange alone. The ten-step NP=1 comparison passes with final `q5
L_inf=8.8817841970012523e-16`; NP=2 `TOPOLOGY=2,1,1` passes with final `q5
L_inf=9.9920072216264089e-16`. Both NP=2 Compute Sanitizer ranks report
`ERROR SUMMARY: 0 errors`.

This acceptance is limited to the stated Cartesian x-slab contract. Curved
NSCBC faces, y/z NSCBC, species, sponge, and characteristic Roe remain outside
S0-B2.

## S0-B3 Status

The controlled `openshock` sponge gate is accepted with the S0-B2 `12/22`
NSCBC pair and only `spg_im=80` on `GRID=400,8,8`; all other sponge ranges are
zero. After each RK update, every rank refreshes all applicable `q` halos
before the x-max rank evaluates the CPU-equivalent second-order
explicit six-neighbor smoothing on its active sponge cells,
`q_new=(1-sigma)q+sigma*(q_{i+1}+q_{i-1}+q_{j+1}+q_{j-1}+q_{k+1}+q_{k-1})/6`.
The geometric CPU coefficients, bounded by the CPU `dampfac=0.05`, are copied
once to a device coefficient array. `qwork_d` supplies the local staging space;
there is no added full-domain field or per-stage host/device copy.

The capability gate rejects x-min, y, z, global/circular, filtered, diffusive,
species, curved, and characteristic-Roe variants. Ten-step CPU/GPU full-field
and `flowstate.dat` comparisons pass at NP=1 and NP=2 `TOPOLOGY=2,1,1`, both
with final `q5 L_inf=8.8817841970012523e-15`. The one-step NP=2 Compute
Sanitizer run reports `ERROR SUMMARY: 0 errors` on both ranks. This is only
x-max layer support.

## S0-B4 Status

The combined-topology gate is accepted at NP=8 `TOPOLOGY=2,2,2` with
`GRID=400,16,16`, so local active extents are `(200,8,8)` and remain at least
`hm`. It exposed a CPU reference defect: `spongefilter_layer` previously
called directional `dataswap(q,direction=axis)` but always reads all six
neighbors. A y/z MPI halo could therefore be stale after the RK update. Each
layer now calls the full `dataswap(q)` before the stencil. CUDA Fortran mirrors
this through a q-only three-direction sponge halo exchange before launching the
local x-max kernels; it does not add primitive refreshes, full-domain scratch,
or host transfers.

The ten-step CPU/GPU full-field and `flowstate.dat` gate passes with final
`q5 L_inf=8.8817841970012523e-15`. The one-step eight-rank Compute Sanitizer
run reports `ERROR SUMMARY: 0 errors` for every rank. On the current two-GPU
workstation this is correctness evidence under rank oversubscription, not a
multi-GPU scaling result.

## Explicit Exclusions

This gate does not validate MP-LD, diffusion, central filtering, shock-region filtering, wall boundaries,
farfield boundaries, chemistry, turbulence, curvilinear geometry, or SBLI.
Those are separate subsequent gates and must not be inferred from OpenShock
CPU/GPU agreement.
