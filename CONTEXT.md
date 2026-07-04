# ASTR GPU Porting

This context defines the project language for migrating ASTR from CPU Fortran/MPI to CUDA Fortran while keeping numerical validation explicit.

## Language

**First-stage acceptance boundary**:
The initial GPU validation target is Taylor-Green Vortex on one MPI rank and one GPU, with no species transport and no turbulence model. It is the boundary for proving CPU/GPU numerical equivalence before broader ASTR coverage.
_Avoid_: Full ASTR GPU port, complete GPU migration

**Explicit sixth-order central difference**:
The finite-difference derivative scheme for the first-stage GPU port is the explicit sixth-order central scheme. It does not solve compact finite-difference linear systems.
_Avoid_: Compact difference, tridiagonal derivative solve

**Explicit tenth-order central filter**:
The filter scheme for the first-stage GPU port is the explicit tenth-order central filter. It does not solve compact filter linear systems.
_Avoid_: Compact filter, tridiagonal filter solve

**Directional thread block**:
A CUDA thread block shape selected by derivative/filter direction: x uses `(512,1,1)`, y uses `(32,16,1)`, and z uses `(64,1,8)`.
_Avoid_: Universal 3D block, fixed `(8,8,8)` block

**First-stage conservative variable set**:
The first-stage GPU RHS covers only the five compressible-flow conservative variables: density, three momentum components, and total energy. Species, modal energy, chemistry, and turbulence equations are outside this boundary.
_Avoid_: Full `numq` coverage, chemistry-ready RHS

**Single-rank GPU validation**:
The first-stage validation runs on one MPI rank and one GPU. It validates GPU numerical kernels without cross-rank halo exchange.
_Avoid_: CUDA-aware MPI validation, multi-rank GPU acceptance

**Tiered numerical tolerance**:
The validation tolerance separates exact-like algebra/copy checks from stencil and time-integration checks. It is meant to expose logic errors without mistaking harmless floating-point ordering differences for porting failures.
_Avoid_: Universal `1e-14` tolerance

**Synchronous correctness kernel**:
A first-stage GPU kernel launch followed immediately by explicit synchronization and error checking. It prioritizes fault localization over throughput.
_Avoid_: Asynchronous performance launch

**Geometry-aware stencil path**:
The first-stage GPU derivative, convection, and diffusion path uses ASTR geometry arrays such as `dxi` and `jacob` rather than hard-coded uniform-grid spacings.
_Avoid_: Hard-coded TGV spacing, `dx/dy/dz`-only GPU path

**Detect-only crash check**:
A first-stage crash check that reports NaN, negative density, pressure, or temperature and stops the run. It does not repair the field.
_Avoid_: Crashfix, automatic field repair

**LF input contract**:
ASTR runtime input files use LF line endings so NVHPC does not read carriage returns as part of string tokens.
_Avoid_: CRLF input files, Windows line endings in runtime inputs

**Runtime GPU switch**:
`use_gpu` is a runtime input-file option that selects CPU or GPU execution inside a CUDA-capable binary. It is not a CMake or compiler option.
_Avoid_: `-DUSE_GPU`, compile-time `use_gpu`

**Host-staged halo buffer**:
The first multi-rank GPU halo-exchange baseline. Each rank packs device halo data into a GPU buffer, copies only that halo buffer to host, exchanges host halo buffers with MPI, copies the received halo buffer back to device, and unpacks it into device halo cells. It keeps full field variables resident on GPU while avoiding CUDA-aware or HIP-aware MPI as a first dependency.
_Avoid_: Full-field D2H halo bridge, mandatory CUDA-aware MPI, device-MPI-only baseline

**GPU validation slab decomposition**:
The first multi-rank GPU validation topology for two available GPUs: two MPI ranks with one rank bound to one GPU, decomposed as `isize=2, jsize=1, ksize=1` for 3D TGV. It intentionally allows a 1D slab decomposition even though the legacy CPU 3D auto-decomposition prefers all three directions to have more than one rank.
_Avoid_: Requiring `2x2x2` for first multi-rank validation, treating two-GPU TGV as impossible

**Validation-controlled topology override**:
The first multi-rank validation may override the legacy 3D MPI auto-decomposition through an explicit validation control such as `ASTR_FORCE_MPI_TOPOLOGY=2,1,1`. This lets CPU and GPU validation runs use the same `2x1x1` topology while leaving default CPU decomposition behavior unchanged.
_Avoid_: `use_gpu`-only topology override, changing CPU default decomposition behavior, comparing different CPU/GPU topologies

**Node-local GPU binding**:
The first multi-rank GPU device-selection policy. Each MPI rank computes its node-local rank with a shared-memory MPI communicator and selects `device_id = mod(node_local_rank, visible_device_count)`. The first two-rank validation requires two visible GPUs and does not silently oversubscribe one GPU.
_Avoid_: Global-rank GPU binding, input-file GPU maps, silent multi-rank single-GPU oversubscription

**Single-node two-GPU validation**:
The first multi-rank GPU validation environment. It runs exactly two MPI ranks on one physical node with two visible GPUs, using node-local GPU binding and `2x1x1` slab decomposition. Cross-node GPU validation is a later extension and is not part of the first multi-rank acceptance target.
_Avoid_: Multi-node first validation, cross-node launcher assumptions

**Conservative-only halo exchange**:
The first multi-rank GPU halo exchange transfers only `q_d(1:5)` halo data. Primitive halo fields (`rho_d`, `vel_d`, `prs_d`, `tmp_d`) are refreshed on the GPU after unpacking `q_d`, rather than exchanged directly.
_Avoid_: Exchanging primitive halo fields, duplicating `q` and primitive communication

**Qswap-compatible halo exchange**:
The first multi-rank GPU halo exchange matches CPU `parallel:qswap`. Each exchanged direction transports `hm+1` layers of `q_d(1:5)`: `hm` layers fill the exterior halo and the extra interface plane synchronizes/averages duplicate periodic or rank-interface planes. Primitive halo fields are refreshed on GPU after unpacking.
_Avoid_: Exchanging only `hm` layers for `qswap`, skipping interface-plane averaging, treating halo exchange as plain endpoint copy

**Physical-face stencil gating**:
When a non-periodic direction is decomposed across MPI ranks, a local array boundary is not automatically a physical boundary. GPU physical-direction stencils must use one-sided/low-order physical templates only when the corresponding neighbor is `MPI_PROC_NULL`; internal MPI interfaces use halo-backed explicit sixth-order central stencils. The current validated instances cover x/y/z zeroextrap with MPI decomposition in the corresponding physical direction.
_Avoid_: Treating every local `i=0/im`, `j=0/jm`, or `k=0/km` as a physical face, using physical templates on MPI internal interfaces, validating only statistics when field output exposes interface errors

**Dataswap-compatible filter halo**:
The explicit filter path uses `dataswap` semantics rather than `qswap` semantics. It exchanges or locally refreshes exactly `hm` halo layers and does not average duplicate endpoint planes. In single-rank homogeneous y/z directions, GPU refreshes local halos for the ping-pong filter arrays before launching the same halo stencil kernels used after MPI filter exchange. In single-rank y/z zeroextrap slices, GPU preserves the physical-direction exterior halo values required by CPU `filterq` rather than inventing one-sided filter stencils.
_Avoid_: Reusing qswap endpoint averaging for filter halos, mapping endpoint index `n` to `0` inside filter stencils, treating local periodic filter handling as a different numerical operator from MPI filter handling, replacing CPU explicit center-filter semantics with one-sided physical-boundary filters

**Blocking staged halo exchange**:
The first multi-rank GPU communication implementation uses blocking `MPI_Sendrecv` on host-staged halo buffers, in the same directional order as CPU `parallel:qswap`. Nonblocking MPI, overlap, and device-aware MPI are later performance backends behind the same halo-exchange interface.
_Avoid_: First implementation based on overlap, mandatory CUDA-aware MPI, changing main-loop semantics for performance

**Private GPU halo MPI tags**:
The GPU halo exchange uses a fixed private MPI tag range instead of modifying `parallel:mpitag`. This keeps GPU communication local to the halo module and avoids collisions with CPU-side communication state.
_Avoid_: Reusing and mutating global `mpitag` inside GPU halo exchange

**CPU-aligned exchange cadence**:
The multi-rank GPU path follows the CPU `mainloop:time_integration_rk` exchange cadence. For TGV, each RK substep performs qswap-compatible halo exchange after filter/boundary handling and before `gradcal`/RHS assembly. There is no extra final halo exchange at the end of the RK step unless a later CPU-owned output boundary explicitly needs synchronized host data.
_Avoid_: One exchange per outer time step, unconditional final exchange after RK update

**Global GPU statistic reduction**:
The multi-rank GPU statistics path computes local numerator sums on each GPU, copies only scalar or partial-sum data to host, applies MPI global reduction, and lets the I/O rank write native `flowstate.dat`. Per-rank local averages are not valid global diagnostics.
_Avoid_: Each rank writing local `flowstate.dat`, averaging already-normalized rank values, device-aware reduction as the first baseline

**Same-topology CPU/GPU validation**:
The first multi-rank GPU acceptance comparison uses CPU and GPU runs with the same MPI rank count and decomposition, starting with `mpirun -np 2` and `2x1x1`. Single-rank CPU output may be recorded as reference context but is not the primary pass/fail oracle for multi-rank GPU correctness.
_Avoid_: Comparing multi-rank GPU directly against single-rank CPU as the primary acceptance test

**Multi-rank first-stage physics boundary**:
The first multi-rank GPU validation keeps the same physics boundary as the single-rank acceptance path: TGV, homogeneous periodic directions, `numq=5`, no immersed boundary, no species transport, no turbulence model, and no chemistry. It validates cross-rank stencil correctness before expanding physical models.
_Avoid_: Opening boundary-condition, species, turbulence, chemistry, or immersed-boundary paths during first multi-rank validation

**Topology-general halo exchange**:
The multi-rank GPU halo exchange interface is designed for x, y, and z directions even when the first two-GPU validation only exercises `2x1x1`. Each direction chooses MPI exchange when the corresponding rank count is greater than one, otherwise local homogeneous periodic handling is used.
_Avoid_: X-only exchange APIs, two-GPU-only naming, validation topology baked into module names

**Backend-neutral GPU facade**:
The first multi-rank implementation may use CUDA Fortran internally, but the public GPU facade and module names should stay backend-neutral. This leaves room for future HIP/DCU backends without changing `src/` call sites.
_Avoid_: CUDA-specific public routine names, TGV/two-GPU topology baked into facade names

**Full-GPU migration architecture**:
The long-term ASTR GPU migration boundary in which CPU `src/` keeps orchestration, runtime input, output, and case lifecycle responsibilities while GPU backend implementations own resident compute-loop data, kernels, halo exchange, and reductions behind backend-neutral facades.
_Avoid_: CUDA-only source architecture, rewriting `src/` around CUDA Fortran internals, treating the TGV backend as the full architecture

**GPU-authoritative compute loop**:
The full-GPU migration data-ownership rule that resident compute-loop fields are authoritative on the GPU between initialization and explicit CPU-owned output/control boundaries. Whole-field CPU/GPU transfers are allowed at initialization and output/checkpoint boundaries, while per-kernel or per-module whole-field round trips are outside the architecture.
_Avoid_: CPU-authoritative GPU acceleration, per-kernel full-field D2H/H2D bridge, treating host arrays as the live compute state during GPU execution

**HaloTransport backend**:
A replaceable GPU halo-communication backend behind the same halo-exchange semantics. The current baseline is host-staged blocking MPI; later backends may use nonblocking host-staged MPI, pinned host buffers, CUDA-aware MPI, HIP-aware MPI, or multi-node topology-aware transport without changing CPU orchestration call sites.
_Avoid_: Hard-wired CUDA-aware MPI, embedding transport policy in solver kernels, treating host-staged MPI as the only architecture

**Full-GPU module expansion order**:
The full ASTR GPU migration order that first stabilizes the backend-neutral architecture and non-reacting flow path, then expands to species transport, turbulence models, chemistry, and immersed-boundary support. This keeps high-complexity physics from driving the architecture before resident data ownership, halo transport, and validation contracts are stable.
_Avoid_: Chemistry-first GPU migration, immersed-boundary-first migration, adding all physics models before the GPU architecture skeleton is stable

**Deferred turbulence-model phase**:
RANS/LES turbulence models are deliberately out of scope for the current GPU migration track. GPU execution should continue to accept only `turbmode='none'`; non-`none` turbulence modes remain explicit rejects unless the turbulence phase is reopened.
_Avoid_: Accidental RANS/LES support claims, treating shock-boundary-layer work as turbulence-model support, weakening the `turbmode='none'` gate

**Combustion GPU target**:
The future combustion GPU track includes species-bearing conservative variables, species halo/filter/diffusion ownership, and chemistry source-term execution. It is broader than adding a single source kernel to the current `numq=5` path. This track is currently deferred; Cantera remains a CPU oracle/reference rather than a selected GPU backend.
_Avoid_: Calling the current source dispatcher chemistry-ready, treating `srccomb` as a drop-in GPU kernel, enabling chemistry without species transport ownership

**Deferred species-transport phase**:
With chemistry and combustion deferred, multi-species transport is optional rather than a blocker for shock/SBLI work. GPU execution should continue to reject `num_species > 0` until a concrete non-reacting multi-species or later combustion requirement reopens the phase.
_Avoid_: Porting species only because it appears before shock in an old phase list, treating single-species shock/SBLI as blocked by species transport, weakening the `num_species=0` gate

**Shock/SBLI GPU target**:
The future shock and shock-boundary-layer GPU track covers shock-capable numerical formats, shock sensors, shock-region filtering or added dissipation, and inlet/outlet/sponge/high-speed wall boundaries for laminar or DNS-like cases with `turbmode='none'`.
_Avoid_: Mixing this with RANS/LES support, using current explicit central-filter validation as shock-capturing validation, treating compact schemes as reopened

**Shock-format readiness gate**:
The first shock/SBLI GPU gate is a non-reacting, no-wall shock-format case that isolates explicit upwind reconstruction, flux splitting, and shock-format RHS behavior before opening sponge, inlet/outlet, high-speed wall, or full SBLI coupling. The selected first target is a forced 3D extruded Sod case via the `sodini` initialization path: x contains the Sod discontinuity, y/z are uniform thin directions, and the first validation grid is `200,8,8`. `examples/sod/datin/input.sod` must not be trusted as-is because it currently declares `flowtype=shuosher`, uses a 1D grid, uses compact upwind, and enables characteristic decomposition. S0-A1 uses periodic boundaries in x/y/z and runs the first gate with `deltat=5.d-4`, `maxstep=20`, so the final time is `0.01` and the Sod wave system should not interact with the x-periodic boundary. S0-A1 keeps `diffterm=f` and `lfilter=f`; diffusion, explicit central filtering, shock-region filtering, and artificial/sensor-based dissipation are later gates. S0-A1 uses `conschm='543e'`; `difschm='643e'` is retained as an explicit-center placeholder but does not participate while `diffterm=f`. The first oracle is same-input CPU/GPU statistics plus output-boundary field comparison with `STATS_ATOL=1e-10`, `STATS_RTOL=1e-10`, `FIELD_ATOL=1e-10`, and `FIELD_RTOL=1e-10`; compare at least max differences for `q(:,:,:,1:5)`. If the first implementation exposes a specific, explainable floating-point difference, tolerances may be revisited to `1e-9` by evidence rather than preemptively. S0-B owns outlet, sponge, NSCBC, or other open-boundary treatment. S0-A1 is explicitly scoped to `conschm(4:4)='e'`; compact upwind (`conschm(4:4)='c'`) and characteristic decomposition (`lchardecomp=.true.`) remain unsupported. The first implementation gate uses `recon_schem=-1` first-order Steger-Warming flux splitting as a plumbing/correctness gate only; WENO/MP/MP-LD accuracy gates come later. The `sod` grid path must provide positive-volume 3D geometry for forced 3D GPU validation instead of reusing `grid1d`'s zero-thickness z coordinate. Shock sensors are only included in S0-A1 if the selected explicit reconstruction requires them.
_Avoid_: Starting Phase S with full SBLI, diagnosing wall/sponge/stability issues before the shock format itself is validated, using the current `input.sod` without correcting/overriding its flowtype, reopening compact schemes or characteristic decomposition in the first shock gate, treating the first-order S0-A1 gate as final shock-format accuracy validation, treating 1D GPU support as required for the first shock gate, using long periodic Sod runs where waves wrap around the domain, enabling diffusion/filtering in the first shock-format plumbing gate

**CPU-owned output boundary**:
The full-GPU migration boundary where HDF5 flowfield output, checkpoint/restart files, slice/list/monitor output, and controller reload remain CPU-owned operations. GPU execution may perform whole-field D2H at these explicit boundaries without violating the GPU-authoritative compute-loop rule.
_Avoid_: Treating HDF5/checkpoint GPU writing as a near-term compute migration requirement, counting output-boundary D2H as per-kernel data residency failure

**Layered GPU validation matrix**:
The full-GPU migration validation strategy that separates build/runtime contracts, module equivalence, time integration, multi-rank correctness, performance/residency profiling, and physics-expansion cases. Each layer uses an explicit oracle such as same-topology CPU/GPU statistics, field comparison at output boundaries, single-rank versus multi-rank GPU statistics, or Nsight transfer budgets.
_Avoid_: Ad hoc validation commands, comparing different topologies as the primary oracle, treating one successful run as full architecture acceptance

**Full-GPU architecture skeleton phase**:
The next architecture phase after the validated TGV CUDA Fortran baseline. It turns the current experimental TGV GPU path into a reusable full-GPU architecture baseline by stabilizing facades, device data ownership, HaloTransport boundaries, kernel module taxonomy, and reusable validation scripts before adding complex physics modules.
_Avoid_: Chemistry-first next phase, adding new physics before architecture boundaries are stable, treating the current TGV implementation as already architecture-complete

**Stable src_gpu CUDA backend directory**:
The near-term code organization rule that keeps the current `src_gpu/` directory as the CUDA Fortran backend implementation while architecture work stabilizes backend-neutral facades and internal module boundaries. Backend-specific directory splits such as `src_backend_cuda/` or `src_backend_hip/` are deferred until a second backend or stronger evidence justifies the churn.
_Avoid_: Immediate large directory migration, CUDA-specific imports in `src/`, directory reshuffling as a substitute for facade boundaries

**Second GPU validation case**:
The first non-TGV case used to test whether the full-GPU architecture skeleton generalizes beyond the Taylor-Green Vortex baseline. It should remain non-reacting, `num_species=0`, no turbulence, no chemistry, no immersed boundary, and primarily `numq=5`, while preferably exposing non-TGV initialization, boundary-condition, or geometry behavior.
_Avoid_: Chemistry or immersed-boundary cases as the second validation case, TGV flame as the architecture-generalization test, picking a case that opens many physics dimensions at once

**Phase A 2dvort validation case**:
The current second-case GPU validation target. It starts from `examples/Vortex_Transport/datin/input.2dvort` but rewrites the runtime case to a 3D extruded periodic explicit case: `flowtype=2dvort`, `643e,643e`, `numq=5`, `num_species=0`, no turbulence, no chemistry, no immersed boundary. It validates generic non-TGV initialization and `flowstate.dat` output while avoiding compact schemes and wall/forcing physics.
_Avoid_: Original 2D `km=0` 2dvort as the GPU acceptance case, channel-flow wall physics as the next low-risk step, compact-filter CPU baselines for Phase A

**Phase A explicit-filter tolerance**:
The filtered `2dvort` Phase A comparison currently uses separate thresholds from the mature TGV baseline: `flowstate.dat` passes at `1e-9` absolute and final `flowfield.h5` reconstructed `q` passes at `1e-9` absolute after matching the CPU-owned output boundary semantics. The no-filter `2dvort` isolation check passes at `1e-10`. A stricter filtered field check at `1e-10` is still blocked by CPU roundoff in the physically zero `u3/q4` component, about `8e-10`, while the GPU keeps that component exactly zero.
_Avoid_: Reporting filtered `2dvort` as a TGV-level `1e-10` field match, hiding the no-filter strict comparison, treating the relaxed Phase A threshold as final acceptance for all cases, manufacturing spanwise momentum noise only to match CPU roundoff

**Phase A 2dvort multi-rank tolerance**:
The first `2dvort` multi-rank validation uses same-topology CPU/GPU comparisons for `NP=2` with `2x1x1`, `1x2x1`, and `1x1x2`. No-filter field output remains strict at `1e-10`; filtered `flowstate.dat` remains `1e-9`; filtered HDF5 field output uses `5e-9` because x/y interface-adjacent reconstructed energy differences reach about `4.7e-9` while native statistics remain at `1e-11` to `1e-10` scale except for physically zero `u3/q4` CPU roundoff.
_Avoid_: Collapsing statistics and HDF5 output tolerances into one number, treating HDF5 interface-adjacent filtered field tolerance as proof of performance readiness, using single-rank CPU as the primary multi-rank oracle

**Phase A 2dvort NP4 core evidence**:
The optional `NP=4` `2dvort` core matrix has passed for `2x2x1`, `2x1x2`, and `1x2x2` with the same no-filter `1e-10` and filtered-statistics `1e-9` contracts. Filtered HDF5 field output needs `FILTER_FIELD_ATOL=6e-9`; the observed maximum was reconstructed `q5=5.1636668274568365e-09` for `2x2x1`, again near decomposition interfaces.
_Avoid_: Promoting `6e-9` to the single-rank or NP2 default without recording why, treating two-GPU oversubscribed NP4 as performance evidence

**Phase A 2dvort NP8 smoke evidence**:
The optional `NP=8` `2x2x2` `2dvort` smoke has passed as a statistics-only same-topology CPU/GPU comparison with field output disabled: no-filter uses `1e-10`; filtered uses `1e-9`. This validates combined x/y/z halo routing under two-GPU oversubscription but is not an HDF5 field-output contract and not performance evidence.
_Avoid_: Treating NP8 oversubscription as production scaling proof, using this smoke to relax field-output tolerances, skipping same-topology CPU/GPU comparison

**Third-case screening result**:
Most remaining `examples/` cases are not low-risk under the current GPU capability gate: many are compact-scheme defaults, `numq=3`, 1D/2D, wall/inflow/shock/boundary-condition heavy, or chemistry/species cases. The next code step should be capability-gate cleanup, replacing hard-coded `flowtype=tgv/2dvort` checks with explicit supported-feature checks, before accepting another genuinely new flowtype.

**Phase D channel wall slice**:
The first channel/wall GPU validation target is `examples/Channel` with y-direction `bctype=41,41` isothermal no-slip walls and periodic x/z directions. The current slice includes deterministic channel initialization, GPU y-wall boundary application, resident geometry `x_d`, GPU channel statistics, and GPU channel body-force source. The GPU source kernel must read `jacob_d` directly from `commarray_gpu`, matching the global RHS kernels; passing the halo-bounded `jacob_d` through a `0:im,0:jm,0:km` dummy caused a force-proportional source error. Single-rank `MAXSTEP=20` feedback validation now passes strictly, and fixed-force no-filter validation passes to 100 steps at roundoff scale. At `GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 LFILTER=f DIFFTERM=t CHANNEL_FORCE_MODE=fixed`, stats-only multi-rank validation passes for NP=2 `2x1x1/1x2x1/1x1x2`, NP=4 `2x2x1/2x1x2/1x2x2`, and NP=8 `2x2x2` with max observed `massflux` difference about `5.0e-13` and `forcex=0`. `NP=2`, `NP=4`, `NP=8 2x2x2`, and `NP=27 3x3x3` short filtered-feedback checks also pass. It is not a general wall-boundary implementation.
_Avoid_: Treating `bctype=41` support as all wall BCs, passing halo-bounded device arrays through mismatched lower-bound kernel dummies, assuming wall blowing/suction, turbulence wall models, or production larger-rank channel performance are covered

**Artificial TGV wall41 slices**:
`tests/gpu_validation/run_wall41_phased_compare.sh` creates a non-physical TGV-template probe with one Cartesian direction set to `bctype=41,41` and the other two directions periodic. GPU x/y/z artificial wall slices pass single-rank and NP=2 physical-direction 5-step `LFILTER=t,DIFFTERM=t` CPU/GPU statistics and field comparison at `GRID=32,32,32`. `tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh` passes the default 18-entry NP=2/NP=4 field matrix and NP=8 `2x2x2` smoke. This validates explicit Cartesian isothermal no-slip boundary kernels, physical-face `MPI_PROC_NULL` gating, and existing physical-direction RHS/filter/diffusion routing for the artificial slice. It is not physical TGV-wall validation and not a general wall-boundary implementation.
_Avoid_: Comparing x/y/z artificial-wall statistics as if they should be equal, treating artificial TGV wall slices as channel-wall physics, assuming wall blowing/suction, turbulence wall models, curvilinear normals, species, chemistry, compact schemes, or GPU HDF5/checkpoint writing are covered, treating two-GPU NP=8 oversubscription as performance scaling evidence

**Phase E wall42 slices**:
GPU `bctype=42` support is limited to CPU-compatible Cartesian no-slip adiabatic x/y wall slices. CPU `noslip_adibatic` implements only `ndir=1..4`; z wall42 has no CPU branch and is intentionally rejected by `BC_KIND=adiabaticwall ZERO_AXIS=z`. The implemented GPU rule matches the supported CPU scope: pressure and temperature are second-order extrapolated from the interior, velocity is set to zero, density is reconstructed by EOS, and q/qrhs are rebuilt on device. Single-rank x/y and NP=2 physical-direction x/y 5-step `LFILTER=t,DIFFTERM=t` statistics plus field comparisons pass at `GRID=32,32,32`.
_Avoid_: Claiming wall42 z support, wall blowing/suction support, turbulence wall model support, or general wall-boundary support from the current x/y Cartesian validation slice

**Phase F wall411 slice**:
GPU `bctype=411` support is limited to CPU-compatible Cartesian slip-nonslip isothermal y wall slices. CPU `slipisotwall` implements only `ndir=3/4`; x/z wall411 have no CPU branch and are intentionally rejected by `BC_KIND=slipisotwall ZERO_AXIS=x/z`. The lower wall matches CPU's `x <= xslip` split: the slip segment extrapolates streamwise velocity and temperature from the interior, sets wall-normal and spanwise velocity to zero, and extrapolates pressure; the no-slip segment uses zero velocity and fixed `twall`. The upper wall follows the CPU no-slip isothermal branch. Single-rank y, NP=2 physical y, and NP=2 transverse x/z slab 3-5 step `LFILTER=t,DIFFTERM=t` statistics plus field comparisons pass at `GRID=32,32,32`.
_Avoid_: Claiming wall411 x/z support, wall blowing/suction support, turbulence wall model support, species support, or general slip-wall physics from the current y Cartesian validation slice

**Phase G wall421 slice**:
GPU `bctype=421` support is limited to CPU-compatible Cartesian slip-nonslip adiabatic y wall slices. CPU `slipadibwall` implements only `ndir=3/4`; x/z wall421 have no CPU branch and are intentionally rejected by `BC_KIND=slipadibwall ZERO_AXIS=x/z`. The active CPU implementation has the `xslip` branch commented out, so GPU follows the active formula: lower wall extrapolates pressure, temperature, and streamwise velocity from the interior while setting wall-normal/spanwise velocity to zero; upper wall extrapolates pressure, temperature, streamwise velocity, and wall-normal velocity while setting spanwise velocity to zero. Single-rank y, NP=2 physical y, and NP=2 transverse x/z slab 3-5 step `LFILTER=t,DIFFTERM=t` statistics plus field comparisons pass at `GRID=32,32,32`.
_Avoid_: Claiming wall421 x/z support, active xslip split behavior, wall blowing/suction support, turbulence wall model support, species support, or general slip-wall physics from the current y Cartesian validation slice

**Phase H wall-family regression matrix**:
`tests/gpu_validation/run_wall_family_phaseh_matrix.sh` is the canonical regression entry point for the currently supported wall family. The default supported matrix covers wall41 x/y/z physical NP=2 slabs, wall42 x/y physical NP=2 slabs, wall411 y physical plus transverse x/z NP=2 slabs, and wall421 y physical plus transverse x/z NP=2 slabs. The default reject matrix checks `42-z`, `411-x/z`, and `421-x/z`; those cases must stay rejected until the corresponding CPU boundary routines and GPU kernels both implement those directions. The full default run under `tests/gpu_validation/out/wall_family_phaseh_matrix_full` passed 11 supported stats/field entries and 5 expected rejects.
_Avoid_: Treating Phase H as new wall physics, removing expected rejects to make the matrix green, or claiming general wall-boundary support from the current Cartesian CPU-compatible slices

**Phase I LDC gate**:
`examples/Lid-Driven-Cavity` is the next true case candidate because it avoids species, chemistry, turbulence, and shock-capturing while opening multi-axis physical boundaries and the top-lid `bctype=0` UDF boundary. `tests/gpu_validation/run_ldcavity_phasei_gate.sh` records the current RED state: the original LDC input is rejected by the GPU path as non-3D, and forced 3-D probes with filter or diffusion enabled are rejected by explicit LDC pending gates. A one-step explicit CPU LDC probe completes without `ieee_invalid`; the earlier Release/O2 warning was traced to `gridcube(1.d0,1.d0,0.d0)` and fixed by guarded `dx/dy/dz` precomputation in `src/gridgeneration.F90`. LDC grid generation keeps `lz=0` for the original `ka==0` 2-D case and uses `lz=1` for forced `ka>0` 3-D oracle probes. The GPU boundary capability layer identifies the LDC x/y physical plus top-lid UDF pattern separately from generic unsupported boundary combinations. The LDC boundary function applies x isothermal no-slip walls, then y lower isothermal no-slip, then the y upper moving lid. This order matches CPU `boucon`/`udf_bc`: the top lid is applied after the side walls, so the top-left/top-right corner velocity is overwritten to `(u,v,w)=(1,0,0)`.
_Avoid_: Treating LDC as fully GPU-supported, enabling filtered LDC before multi-axis filter ownership is implemented, hiding CPU oracle warnings, or adding a shock/upwind format while the current blocker is boundary architecture

**Phase I-A LDC no-filter/no-diffusion slice**:
`tests/gpu_validation/run_ldcavity_phaseia_compare.sh` validates a forced 3-D `examples/Lid-Driven-Cavity` probe at `GRID=32,32,32`, explicit `643e`, `LFILTER=f`, and `DIFFTERM=f`. This slice supports x/y physical boundaries plus z periodic halo, applies the top lid last to preserve CPU corner semantics, and uses dedicated x+y physical convection flux kernels so neither x nor y is treated as periodic. Single-rank 1-step and 5-step CPU/GPU statistics plus field comparisons pass; the 5-step run has max `flowstate.dat` difference `maxq5=2.8620661396416835e-11` and reconstructed field `q5` `L_inf=2.8421709430404007e-14`.
_Avoid_: Claiming default LDC support, filtered LDC support, or full cavity physics from this no-filter/no-diffusion slice

**Phase I-B LDC no-filter/diffusion slice**:
Forced 3-D LDC now supports `LFILTER=f,DIFFTERM=t` using dedicated x+y physical diffusion gradient, stored diffusion flux, and x/y/z diffusion RHS kernels. The z direction remains periodic or MPI-exchanged; x/y use physical one-sided stencils only at true physical faces. Single-rank 1-step and 5-step CPU/GPU statistics plus field comparisons pass at `GRID=32,32,32`; the 5-step run has final `flowstate.dat` max `maxq5=3.9619862945983186e-11` and reconstructed field `q5` `L_inf=8.5265128291212022e-14`. `NP=4, TOPOLOGY=2,2,1` 5-step diffusive LDC also passes, covering x/y internal halos and true physical faces together.
_Avoid_: Treating original 2-D LDC, compact schemes, species/turbulence/shock paths, or GPU HDF5/checkpoint writing as covered by Phase I-B

**Phase I-C LDC filter/diffusion status**:
Forced 3-D LDC now passes strict `LFILTER=t,DIFFTERM=t` CPU/GPU field and statistics comparison at `GRID=32,32,32`. The previous combined-path mismatch was localized near the top-lid/side-wall corner at `(i,j,k)=(0,31,1)` and traced to GPU x/y physical diffusion RHS kernels applying all three `is:ie/js:je/ks:ke` restrictions at once. CPU `diffrsdcal6` applies direction-specific ranges: x RHS restricts only `is:ie`, y RHS restricts only `js:je`, and z RHS restricts only `ks:ke`; GPU `xyphysical` diffusion RHS and `x_xphysical` diffusion RHS now follow that rule. Clean validation: `OUT_DIR=tests/gpu_validation/out/ldcavity_filter_diff_1step_fixed_clean LFILTER=t DIFFTERM=t MAXSTEP=1 FEQCHKPT=1 COMPARE_STATS=f COMPARE_FIELD=t FIELD_ATOL=1e-10 FIELD_RTOL=1e-10 tests/gpu_validation/run_ldcavity_phaseia_compare.sh` passed with reconstructed `q5 L_inf=2.2737367544323206e-13`; `OUT_DIR=tests/gpu_validation/out/ldcavity_filter_diff_5step_fixed_clean LFILTER=t DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 COMPARE_STATS=t COMPARE_FIELD=t FIELD_ATOL=1e-10 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 ...` passed with final `flowstate.dat` max `maxq5=8.1854523159563541e-12` and reconstructed `q5 L_inf=5.6843418860808015e-13`. A 20-step run is still not a valid gate on the current small LDC setup because CPU/GPU diagnostics both enter abnormal/crash-prone territory by about 16-20 steps.
_Avoid_: Reintroducing all-axis compute-range restrictions into direction-split diffusion RHS kernels, using the 20-step unstable small-grid run as a GPU correctness failure, or treating this as support for compact schemes/species/turbulence/shock paths

**Phase J RTI explicit validation variant**:
`examples/Rayleigh–Taylor-Instability` is the next non-shock explicit case after LDC. This is an explicit validation variant, not a reproduction of the original compact 2-D input: the validation forces `GRID=32,64,32`, explicit `643e`, `rk3`, `numq=5`, no species, no turbulence, no shock/upwind, x/z periodic, y physical fixed `bctype=31`, and RTI gravity source. `src/gridgeneration.F90` preserves the original RTI zero-thickness grid when `ka==0`, but uses `lz=0.25` when `ka>0`; otherwise the forced 3-D oracle has zero volume. GPU adds y-only `bctype=31` fixed-boundary support and an RTI source kernel equivalent to CPU `src_rti`. `tests/gpu_validation/run_rti_phasej_matrix.sh` is now the canonical Phase J regression driver. The current driver run under `tests/gpu_validation/out/rti_phasej_matrix_driver_current` passed all 13 default entries: single-rank `LFILTER=f/t,DIFFTERM=f/t`, `NP=2` x/y/z slabs, `NP=4` combined slabs, `NP=8 TOPOLOGY=2,2,2`, and 20-step single-rank plus `NP=4 TOPOLOGY=2,2,1`.
_Avoid_: Calling the original compact RTI input supported, enabling x/z `bctype=31` fixed boundaries whose CPU branches are empty, treating this as shock/upwind/species/turbulence support, or using original `km=0` as the GPU correctness gate

**Phase K GPU case-source dispatcher**:
GPU source terms now enter the main loop through `solver_gpu:apply_case_sources_gpu()`, mirroring the CPU `solver:rhscal` source-dispatch position after convection/diffusion RHS assembly and before RK update. Source capability selection is centralized in `case_capability_gpu:gpu_case_source_kind()`, currently covering channel (`src_chan` equivalent, only when `lihomo`) and RTI (`src_rti` equivalent), with `GPU_SOURCE_NONE` for no-source cases. `tests/gpu_validation/run_source_phasek_matrix.sh` is the canonical Phase K source regression driver; the current run under `tests/gpu_validation/out/source_phasek_matrix_current` passed TGV no-source, channel source, and RTI source entries. The full RTI matrix also passes after capability extraction under `tests/gpu_validation/out/rti_phasej_matrix_after_capability`.
_Avoid_: Reintroducing `flowtype` source branches in `mainloop_gpu`, treating source dispatch as chemistry/turbulence support, or moving source application before diffusion/convection RHS assembly

**Channel filtered long-step limit**:
The small `GRID=32,32,32` channel validation with `LFILTER=t` is a strict CPU/GPU equivalence gate through 20 steps after the channel source fix. It is not a good time-step convergence study beyond that point: the CPU baseline itself crashes around 23 filtered steps at `DELTAT=5.d-4`, and around 20 filtered steps for smaller `DELTAT`. Use fixed-force `LFILTER=f,DIFFTERM=t` for 100-step source/wall/diffusion residency validation until the filtered channel baseline is redesigned.
_Avoid_: Loosening tolerances to hide a CPU baseline crash, claiming filtered long-step convergence from GPU-only completion when CPU `crashcheck` stops earlier, using the resident GPU path's lack of host `crashcheck` as stability evidence
_Avoid_: Jumping directly to channel flow, treating shock-tube/upwind or wall-bounded examples as equivalent to the current periodic explicit non-reacting path

**Capability-gated explicit periodic case**:
The GPU runtime gate is increasingly capability-based rather than a hard-coded `flowtype` whitelist. The broad shared periodic path still requires 3D, `numq=5`, no species, no modal equations, no turbulence model, `rk3`, explicit `643e,643e`, homogeneous x/y/z, consistent MPI topology, and positive Jacobian; this covers TGV, 3D extruded `2dvort`, and HIT. Later validated gates add specific non-periodic boundaries, wall slices, LDC, RTI, and source dispatch. Phase S0-A1 is planned as a separate controlled shock-format capability for forced 3D Sod with `conschm='543e'`, `recon_schem=-1`, no diffusion, and no filter.
_Avoid_: Treating `flowtype` names as the support contract, accidentally accepting compact or unsupported non-periodic cases because their name is known, treating planned S0-A1 as already implemented

**Phase A HIT validation case**:
The current third explicit non-reacting validation target is `flowtype=hit` using `examples/Taylor_Green_Vortex` as a template plus a generated deterministic `datin/velocity.h5`. The generator writes a periodic ABC-style velocity field with zero discrete divergence after `hitini` refreshes halos. Validation compares CPU/GPU `flowstate.dat` statistics for `kenergy`, `enstophy`, and `dissipation`; HDF5 field-output comparison remains outside the current HIT contract.
_Avoid_: Calling HIT validated while using a divergent generated velocity field, treating CPU/GPU agreement as physical correctness without checking `hitini` divergence, making HIT depend on GPU HDF5 output

**HIT initialization halo contract**:
`src/initialisation.F90:hitini` reads `vel(0:im,0:jm,0:km,1:3)` from `datin/velocity.h5` and then must refresh CPU halos before calling `grad()` for the divergence diagnostic. `grad()` consumes `-hm:im+hm` style halo arrays, so the diagnostic is not meaningful without `dataswap(vel)`.
_Avoid_: Trusting the original HIT divergence print before halo refresh, changing GPU kernels to compensate for a CPU initialization diagnostic bug

**Phase B x-zeroextrap boundary slice**:
The first non-periodic GPU boundary slice is a regular 3D `numq=5` explicit case with x-direction `bctype(1:2)=50,50`, y/z periodic, and single MPI rank. GPU applies the same second-order zero-extrapolation physical boundary update as CPU `bc:zeroextrap`, restricts RHS assembly to CPU `is:ie/js:je/ks:ke` active ranges, uses x-physical `gradcal`, has x-physical diffusion flux/RHS support for `diffterm=t`, and now supports the explicit 10th-order filter for `lfilter=t`.
_Avoid_: Treating this as general wall/farfield support, using periodic x-gradient or periodic x-diffusion diagnostics as the oracle, assuming this single-rank x-physical filter path already covers multi-rank non-periodic boundaries

**Phase B y/z zeroextrap filter slices**:
The next non-periodic boundary tracer supports single-rank y-direction or z-direction `bctype=50,50` with the other two directions periodic and `lfilter=t,diffterm=t`. GPU applies y/z zeroextrap physical boundary kernels, skips periodic qswap in the physical direction, keeps periodic qswap in the other two directions, and uses y/z physical `gradcal`, y/z physical convection RHS, y/z physical diffusion flux/RHS, and CPU-compatible explicit filter ping-pong halo handling over CPU direction-specific active ranges. `MAXSTEP=5` CPU/GPU statistics and `flowfield.h5` comparisons pass at roundoff scale for `ZERO_AXIS=y/z` with `lfilter=t,diffterm=t`.
_Avoid_: Treating y/z zeroextrap filtering as a one-sided physical-boundary filter, refreshing all primitive fields immediately after filterq, treating this as multi-rank non-periodic boundary support

**Phase B zeroextrap multi-rank slices**:
The first multi-rank non-periodic boundary support allows exactly one physical zeroextrap direction and now supports decomposition in that physical direction as well as the remaining homogeneous periodic directions. GPU physical-direction boundary kernels, gradcal, convection RHS, diffusion flux/RHS, qswap-compatible halo exchange, filter ping-pong halo exchange, and diffusion field halo exchange gate true physical faces with `MPI_PROC_NULL`; internal MPI interfaces use exchanged halos. The reusable one-step field matrix now covers x/y/z zeroextrap with NP=2 and NP=4 topologies including `2x1x1`, `1x2x1`, `1x1x2`, `2x2x1`, `2x1x2`, and `1x2x2` where applicable. Earlier matrix variants pass at `MAXSTEP=5` with field comparison and at `MAXSTEP=20` for statistics-only validation.
_Avoid_: Claiming full wall/farfield/general boundary-condition support, refreshing full-domain primitives after filterq, treating the current host-staged blocking MPI path as a performance/scaling result

**Phase C symmetry boundary slice**:
The next non-periodic GPU boundary support is CPU-compatible `bctype=60` symmetry for exactly one physical x/y/z direction, with the other two directions homogeneous periodic. It reuses the finite-physical-axis routing introduced for zeroextrap, so true physical faces are gated by `MPI_PROC_NULL` and internal MPI interfaces use exchanged halos with sixth-order central stencils. The implemented Cartesian boundary action matches the CPU `bc:symmetry` behavior used by the TGV-template validation: normal velocity is set to zero on the corresponding physical face, tangential primitive fields are second-order extrapolated, q is rebuilt on device, and boundary qrhs is explicitly zeroed. The current validation is not a general curvilinear normal-projection implementation because GPU boundary-normal arrays are not yet part of the device boundary contract.
_Avoid_: Treating `bctype=60` support as wall/farfield/NSCBC support, claiming arbitrary curvilinear symmetry until boundary normals are resident on device, using physical one-sided stencils on MPI internal interfaces

**Filtered primitive timing contract**:
CPU `filterq` updates conservative variables `q` but does not immediately refresh all primitive fields. CPU `updatefvar` happens after RK updates, so each filtered RK substep uses primitive fields from the pre-filter state for interior operations, while boundary and halo primitive slices are refreshed by `boucon/qswap`. The GPU filtered zeroextrap paths mirror this by refreshing primitive fields before filtering each unprepared RK substep, then preserving interior primitive fields after filtering and refreshing only CPU-compatible boundary/halo primitive slices.
_Avoid_: Calling a full-domain primitive refresh immediately after `filterq`, using filtered `q` to rebuild all primitive fields before `gradcal/rhscal`, treating q/primitive consistency as automatic inside a filtered CPU RK substep

**Single-rank non-homogeneous CPU active range**:
For `lihomo=f,isize=1` and the analogous y/z cases, CPU `parallelini` must set active ranges to interior nodes (`1:im-1`, `1:jm-1`, `1:km-1`) while physical boundary planes are owned by `boucon`. Leaving these module variables unset makes the CPU baseline effectively skip interior RHS for a single-rank finite-domain test.
_Avoid_: Validating GPU non-periodic boundary support against an uninitialized CPU active range, letting boundary planes receive convective RHS after `boucon` zeroes their boundary residual

**GPU device field ownership table**:
The explicit inventory of device-resident fields, their allocation extents, producer/consumer modules, halo semantics, and host-transfer policy. It is the guardrail that keeps the compute loop GPU-authoritative while still allowing CPU-owned initialization and output/checkpoint boundaries.
_Avoid_: Adding new device arrays without ownership records, hidden whole-field transfers, treating host arrays as live compute state during GPU execution

**HaloTransport semantic/transport split**:
The rule that qswap, dataswap, halo width, local periodic fallback, primitive refresh, and tag ownership are solver semantics, while host-staged blocking, nonblocking, pinned, CUDA-aware MPI, HIP-aware MPI, and multi-node scheduling are transport backends. New communication optimizations must preserve the semantic layer.
_Avoid_: Encoding CUDA-aware MPI into solver logic, hiding `hm` versus `hm+1` behind a byte transport, changing qswap/dataswap behavior while optimizing transfer paths

**Reusable GPU validation drivers**:
The current repeatable GPU validation entry points are `tests/gpu_validation/run_tgv_mpirank_matrix.sh` for the core multi-rank stats/field matrix and `tests/gpu_validation/run_tgv_256_nsys_profile.sh` for the `256^3 NP=1/NP=2` Nsight profile. Both have post-script pass evidence; high-rank oversubscription smoke remains opt-in with `RUN_SMOKE=t`.
_Avoid_: Treating old hand-run profile output as the only evidence, rerunning validation by manual case edits, marking oversubscription runs as performance proof

## Source Architecture Memory

The canonical source-structure note is `documents/ASTR_SRC_ARCHITECTURE_MEMORY.md`.

`src/astr.F90` is the only main program. The `run` path is:
`readinput -> mpisizedis -> parapp -> parallelini -> refcal -> fileini -> infodisp -> allocommarray -> ibprocess -> gridgen -> solvrinit -> geomcal -> spongelayerini -> flowinit -> steploop`.

`src/commvar.F90` owns global scalar configuration and runtime state. `src/commarray.F90` owns global field arrays. Most solver modules mutate these globals directly; new GPU work should treat their shapes, halo ranges, and update order as the CPU contract.

The CPU time-integration contract is in `src/mainloop.F90`: per RK substep it applies filter if enabled, boundary/halo work, `gradcal`, `rkfirst` only on substep 1, `rhscal`, RK update using `qsave=q*jacob`, sponge filtering, then `updatefvar`.

The CPU RHS contract is in `src/solver.F90`: `rhscal` accumulates convection into `qrhs`, negates it as `-conv`, then adds diffusion as `+diff`. Central convection/diffusion both use `src/derivative.F90` through `src/comsolver.F90`; explicit-vs-compact behavior is selected by the scheme suffix `e` or `c`.

Single-rank periodic halo semantics are not a plain endpoint copy. `parallel:qswap` fills both halo sides and then averages the duplicate periodic planes, e.g. `q(0)=0.5*(q(0)+q(im))` and `q(im)=q(0)`. GPU validation must match this before comparing statistics or fields.
