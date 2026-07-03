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

**Dataswap-compatible filter halo**:
The explicit filter path uses `dataswap` semantics rather than `qswap` semantics. It exchanges or locally refreshes exactly `hm` halo layers and does not average duplicate endpoint planes. In single-rank homogeneous y/z directions, GPU refreshes local halos for the ping-pong filter arrays before launching the same halo stencil kernels used after MPI filter exchange.
_Avoid_: Reusing qswap endpoint averaging for filter halos, mapping endpoint index `n` to `0` inside filter stencils, treating local periodic filter handling as a different numerical operator from MPI filter handling

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
