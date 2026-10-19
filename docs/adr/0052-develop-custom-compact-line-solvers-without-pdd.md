# Develop custom compact line solvers without PDD

On 2026-10-19 the user reopened compact-scheme design for flat-plate-type grids
and future large multi-GPU runs. The selected direction is custom blocked
CR/PCR and two-level PCR, borrowing DistD2 data grouping and RHS fusion without
depending on an external banded-solver library or using PDD coupling truncation.

The user selected two stages: first replace the rank-local solver with blocked
CR/PCR while preserving the existing MPI interface closures; then develop a
separate globally coupled path using local subsystem PCR and distributed
interface PCR. GPU calculation tiles must not introduce extra numerical closures.
The two stages have separate operator identities, references, and acceptance.
The user subsequently confirmed that the final compact production implementation
is unified global two-level PCR and is used by default when compact schemes are
selected. Rank-local closure remains a development/reference stage, not a
required permanent production option or a silent fallback. Existing explicit
scheme defaults are unchanged.

The selected operator scope is the existing 643c central derivative and compact
filter, validated in that order and then together. Compact upwind reconstruction
and new pentadiagonal discretizations are deferred; reusing a banded-solver
design does not authorize introducing a new numerical scheme.

The CPU algorithm is a reference, not an immutable requirement. The user
explicitly allows improving clear efficiency defects and numerical or physical
errors. Equivalent performance changes retain a same-operator reference;
mathematical corrections require documented evidence and an agreed corrected
reference, with shared defects addressed on CPU and GPU rather than reproduced
for compatibility. Model or scheme preferences are not automatically defects.
Numerical acceptance is defined separately from implementation and performance.

The first delivery covers non-reacting five-variable flow across both stages.
The line-solver interface must not hard-code five components. AIR5 remains a
separately reviewed extension, not a first-delivery acceptance requirement.

Integration proceeds from line-solver components to periodic TGV and then a
stretched-grid non-reacting flat plate. TGV exercises compact central flux
derivatives and gradients; the flat plate retains explicit MP7-LD convection
while compact gradient/viscous operations and filtering are admitted separately.

The first implementation remains CUDA Fortran under `src_gpu/`, built through
the root CMake entry. Public interfaces stay backend-neutral, but HIP/DCU device
kernels will be a later implementation rather than a claim of direct source
compatibility. This module does not add a CUDA C++ or in-situ dependency.

Authoritative flow-array layouts remain unchanged. Directional line grouping
is limited to private RHS and intermediate workspaces, with RHS formation and
writeback fused where useful. Workspace reuse and layout costs are part of the
operator evaluation rather than hidden behind line-kernel-only timings.

Establish a per-kernel synchronized diagnostic baseline first, then allow a
validated dependency-driven path inside the compact module while retaining
diagnostic synchronization. Required thread barriers, inter-block dependencies,
MPI readiness and workspace lifetimes remain explicit; the existing outer-loop
synchronization policy is not broadened by this decision.

In the second stage, periodic directions use genuinely cyclic global matrices.
Periodic endpoint replicas are not independent unknowns, and internal MPI
interfaces use the same interior compact relation. Establish a corresponding
CPU global reference; the old rank-local operator is not the equivalence oracle
for this new operator. Existing flow-array storage remains unchanged.

An undecomposed direction still uses the global operator, without distributed
communication. Checkpoint metadata must identify the compact operator/version
and numerically relevant settings so exact restart cannot silently change the
operator. This adds neither a sidecar file nor repeated full-field verification;
checkpoints lacking that identity follow the non-exact policy below.

The reduced interface system supports explicitly selected pinned host-staged
and device-aware MPI transports. Establish the host-staged reference first,
then validate the device path and compare complete operator costs on supported
platforms. PCR exchanges reduced interface data rather than gathering complete
lines or fields; known-field halos needed to form the RHS remain separate.
Reuse low-level readiness and buffer-lifetime mechanisms where suitable, not
the flow halo protocol's shared-node averaging or neighbor-only schedule.
Keep the linear-system protocol isolated from flow exchanges, do not silently
replace a requested device-aware transport, and add no NCCL dependency. The
planned ASTR_GPU_COMPACT_TRANSPORT setting selects pinned/device-aware with
pinned as the portable default. Communication contexts and tags are private to
the compact solver and are established with its directional plans.

The first delivery supports legal line lengths including odd and non-power-of-two
sizes, non-power-of-two process counts along each direction, and the existing
quotient/remainder partitions with unequal local interval counts. Numerical
stencil and partition requirements still apply. Computational padding must not
add physical grid points, discard unknowns or change the real cyclic coupling;
system sizes count unique unknowns, not shared-node or periodic replicas.

Private workspaces are preallocated, reused across directions and operated on
in batches of independent grid lines. Identical matrices share preprocessed
coefficients; distinct operators, lengths and boundary matrices must remain
distinguishable. Batching retains all within-line coupling and uses consistent
line identities and batch ordering within each directional process group.
Retain scalar/full compact-filter storage choices with scalar as the default.
Choose batch capacities during implementation and preflight against actual
resources; complete operator timings include every batch, layout operation and
communication. Resource accounting is not an arbitrary new allocation cap.

Use layered numerical acceptance: the normwise linear-system backward residual
must be at most 1e-12, evaluated against the original matrix including its
physical or cyclic boundary rows. Same-operator CPU/GPU and topology short tests
must have per-component maximum differences normalized by
`max(1,max(abs(reference)))` of at most 2e-10; report unnormalized differences too.
Use an independent CPU global solver for the globally coupled operator, and
check analytic discretization convergence separately for interior and actual
boundary closures. These are validation checks, not recurring production
full-field readbacks. They do not require bitwise agreement between different
elimination orders or establish physical validation by numerical agreement.

The first flat-plate delivery includes both the conventional
`11/prof,21,41,50,1,1` boundary combination and the characteristic
`11/prof,22,41,52,1,1` combination, retaining explicit MP7-LD convection.
Validate compact gradients/viscous operations and filtering with characteristic
RHS construction, boundary-specific derivatives, mode-dependent transverse
filtering and same-stage caches. The user selected `incoming_only` for bc52,
with explicitly identified global constant rho/u/v/w/T targets for incoming
characteristic relaxation. `nonreflecting` and imposed-shock target modes are
outside this first delivery. Existing boundary defaults remain unchanged, and
existing explicit admissions do not qualify these new compact combinations.

The approved local plan includes component and small-grid numerical gates,
then short performance tests on 256-cubed TGV and a 450-by-130-by-96-interval
flat plate. Use NP=1/2 with one rank per physical GPU for performance, with
10 warmup steps, 30 timed steps and three repetitions. Shared-device higher-rank
small cases validate topology and uneven partitions, not multi-GPU efficiency.
Separate field-comparison runs from timing, disable visualization and intermediate
large outputs, and report setup, complete operator, RK and memory costs. This
scope does not authorize remote jobs or long production simulations.

Disable wall blowing for the flat-plate topology-equivalence baseline and
performance timing. Separately validate ten CPU/GPU steps with legacy random
blowing at a fixed topology, identical random supply and matching wall-call
counts. Preserve the existing random algorithm and read-only source case;
rank-dependent random drives are not a same-input cross-topology field test.
Timing without blowing is not the complete forced production-case timing.

The flat-plate test retains the M12 inlet and initial-state resources, ideal-gas
gamma=1.4, Sutherland viscosity, Mach 12, Reynolds 1e4, reference temperature
594.3 K and Tw/Tinf=10. Explicitly identify the nondimensional bc52 target
rho/u/v/w/T as (1,1,0,0,1). Start the new compact tests with dt=0.005 and require
acoustic/advective CFL at most 0.5 in short tests, assessing viscous restrictions
separately. This is a candidate step subject to preflight, not a transfer of the
old explicit scheme's stability result or validation of real high-enthalpy air.

Existing new-format checkpoints lacking a compact-operator identity may supply
authoritative fields to a non-exact restart with explicitly selected new compact
inputs. Rebuild derived data and identify the unknown source operator and
non-exact switch; do not accept that source for exact continuation. This does
not restore retired checkpoint formats or bypass geometry, model, field or RNG
contracts. Retain the default exact_restart=false. Rebuild compact factors,
interface plans and workspaces instead of adding them to checkpoint storage.
Validate new exact continuation with 32-cubed TGV on CPU/GPU at NP=1/2,
comparing 12 continuous steps against 5+7 steps within the same backend and
topology. Add no production writeback reread, CRC or sidecar file.

For the covered non-reacting five-variable paths, connect the same global
compact operator to both CPU and GPU main-program execution. The runtime
usegpu choice selects an execution backend, not a different spatial operator.
The CPU implementation remains an independent solve route and need not copy
the GPU two-level PCR algorithm. Scope CPU edits to compact initialization,
derivatives/filtering and necessary MPI integration; retain explicit CPU
behavior and CPU-only builds without a CUDA dependency.

Before local tests, estimate controlled allocations against actually available
device memory, host memory and disk, reuse preallocated storage and record use.
The user selected no extra arbitrary directory-size cutoff or automatic timeout;
the agreed finite step counts and repetitions remain unchanged. Real resource,
runtime and numerical failures still stop downstream work, without silently
reducing the grid, precision or changing the operator. Do not query resources
after every kernel or interfere with other processes or remote jobs.

All D1-D22 requirements were resolved before implementation. The plan records
interface/file boundaries, CP0-CP10 dependencies, numerical/restart/topology
gates and local timing scope. Separate implementation goals covered CP0-CP1
and then CP2 and CP3. Independent components and the explicitly identified local
derivative/filter development path are verified as recorded in
`documents/ASTR_GPU_COMPACT_SCHEME_RESULTS.md`. CP2 uses a testing-build-only
local admission, not the final globally coupled production dispatch. CP3 includes
the user-approved post-filter primitive refresh correction for non-reacting
643c/643c, plus scalar/full packed batches that write q only after their inputs
and right-hand sides are captured. Existing explicit-filter ping-pong remains
unchanged. A subsequent CP4 goal verified the separate globally coupled line
component: unique ownership, local endpoint responses, distributed block PCR,
rank-two cyclic correction and exact shared-node copies. Both explicitly chosen
interface transports passed independent global-matrix checks. Local UCX IPC
mapping cleanup remains a separate toolchain issue; the explicit no-IPC
device-aware configuration passed leak checks without an application host-staged
fallback. A further CP5 goal connected both main-program backends for fresh,
periodic perfect-gas 643c/643c TGV. CPU uses interior Thomas elimination and a
direct solve of the retained endpoints, independently of GPU PCR. The flow
derivative type is switched after geometry initialization, since periodic flow
values and nonperiodic coordinate increments are not interchangeable. Both
backends consume ASTR's actual partition offsets, including its high-rank
remainder assignment. Supported compact flow inputs now select global coupling
without a development environment variable; the explicit default is unchanged.
Physical boundaries, restart identity, performance work and later stages are
not part of CP5. Git writes remain unauthorized.

Requirements continue in `documents/ASTR_GPU_COMPACT_SCHEME_PLAN.md`. This
decision authorizes design, not solver implementation or a change to the
existing explicit production default or the historical first-stage ADR.
