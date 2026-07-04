# ASTR Full-GPU Architecture Plan

## 1. Purpose

This document defines the scalable architecture plan for migrating ASTR from the current CUDA Fortran Taylor-Green Vortex baseline toward a full-GPU ASTR architecture.

The goal is not to turn the existing TGV implementation into a one-off CUDA path. The goal is to preserve the validated CUDA Fortran work while organizing it into backend-neutral boundaries that can later support HIP/DCU backends, larger multi-rank runs, and broader ASTR physics.

## 2. Current Baseline

The current GPU port has reached a validated non-reacting TGV baseline:

- TGV, `numq=5`, `num_species=0`, no turbulence, no chemistry, no immersed boundary.
- GPU-resident compute loop for the main TGV path.
- GPU implementations for the main TGV modules:
  - filter
  - gradcal
  - convection
  - diffusion
  - RK update
  - TGV statistics reductions
- Multi-rank host-staged halo exchange:
  - solution `q_d(1:5)` qswap-compatible halo exchange
  - filter halo exchange with dataswap semantics
  - diffusion-field halo exchange for `sigma_d` and `qflux_d`
- Validated topologies include:
  - `2x1x1`
  - `1x2x1`
  - `1x1x2`
  - `2x2x1`
  - `2x1x2`
  - `1x2x2`
  - `2x2x2`
  - selected oversubscription smoke tests such as `4x2x2`, `2x4x2`, `2x2x4`, and `4x4x2`
- A `256^3`, `NP=1` versus `NP=2`, `2x1x1` statistics comparison has passed within floating-point tail differences.
- Nsight Systems profiles have been generated for `256^3`, `MAXSTEP=10`, `NP=1` and `NP=2`.
- Phase A of the second-case expansion has started with a 3D extruded `2dvort` case:
  - `flowtype=2dvort`
  - explicit `643e,643e`
  - homogeneous periodic x/y/z
  - `numq=5`, `num_species=0`, no turbulence, no chemistry, no immersed boundary
  - `MAXSTEP=1` CPU/GPU statistics and field validation passes for no-filter strict comparison and explicit-filter Phase A thresholds.
  - The filtered `2dvort` HDF5 field comparison now passes at `1e-9` after matching the CPU-owned output boundary semantics. It is not yet a TGV-level `1e-10` contract because CPU produces about `8e-10` roundoff in the physically zero `u3/q4` component while GPU keeps it exactly zero.
  - The same 3D extruded `2dvort` case now has an `NP=2` multi-rank matrix pass for `2x1x1`, `1x2x1`, and `1x1x2`. No-filter field output remains strict at `1e-10`; filtered native statistics pass at `1e-9`; filtered HDF5 field output uses a separate `5e-9` tolerance for interface-adjacent reconstructed energy differences.
  - Optional `NP=4` combined-direction `2dvort` validation has also passed for `2x2x1`, `2x1x2`, and `1x2x2` with filtered HDF5 field tolerance `6e-9`; this is correctness smoke under two-GPU oversubscription, not performance evidence.
  - Optional `NP=8` `2x2x2` `2dvort` statistics-only smoke has passed with field comparison disabled; this validates combined x/y/z halo routing but remains oversubscription smoke, not a production scaling result.
  - Screening the remaining `examples/` inputs shows no safer genuinely new flowtype than `2dvort` under the current capability gate; most candidates introduce compact schemes, `numq=3`, shocks/upwind, wall/inflow boundary conditions, or chemistry/species.
- The third explicit periodic case expansion has started with HIT:
  - The GPU runtime gate has been changed from a `flowtype=tgv/2dvort` whitelist to an explicit capability gate.
  - `flowtype=hit` now runs through the same GPU-resident explicit periodic main loop when the case is 3D, `numq=5`, no species, no turbulence, `rk3`, explicit `643e,643e`, and homogeneous x/y/z.
  - A deterministic ABC-style `velocity.h5` generator has been added for validation. The generated initial field reports zero divergence after `hitini` refreshes CPU halos before calling `grad()`.
  - HIT CPU/GPU `flowstate.dat` validation has passed for `64^3 NP=1 MAXSTEP=20`, `NP=2 TOPOLOGY=2,1,1 MAXSTEP=5`, and `NP=8 TOPOLOGY=2,2,2 MAXSTEP=5` with differences at about `1e-15` or below.
  - HIT HDF5 field comparison remains disabled by default because HDF5 flowfield output is still a CPU-owned boundary, not a current GPU writing target.
- Phase B boundary expansion has started with the lowest-risk non-periodic slice:
  - x-direction `bctype(1:2)=50,50` zero extrapolation, y/z periodic, single MPI rank.
  - CPU `parallelini` now initializes single-rank non-homogeneous active ranges to interior nodes; otherwise the CPU baseline effectively skips interior RHS for this finite-domain test.
  - GPU x-zeroextrap RHS kernels use the same `is:ie/js:je/ks:ke` active ranges as CPU.
  - GPU `gradcal` now has an x-physical derivative path for this slice, so native `flowstate.dat` statistics are a valid oracle.
  - GPU diffusion now has x-physical flux and RHS paths for `diffterm=t`; diffusion field halo refresh skips invalid x-periodic fallback in this slice.
  - GPU explicit filtering now supports this slice for `lfilter=t`. The implementation follows CPU `filterq` timing: refresh primitive fields before a filtered RK substep, do not refresh all primitive fields immediately after filtering, and refresh only the CPU `qswap` boundary/halo primitive slices needed before `gradcal` and RHS assembly.
  - `MAXSTEP=5` CPU/GPU `flowstate.dat` and `flowfield.h5` comparisons pass at roundoff scale with `diffterm=f/t` and `lfilter=f/t`.
  - The first y/z zeroextrap tracer slices have also passed on one MPI rank with `lfilter=t,diffterm=t`. These validate y/z physical boundary kernels, y/z physical `gradcal`, y/z physical convection RHS, y/z physical diffusion flux/RHS, explicit filter ping-pong halo handling, CPU-compatible filtered primitive timing, and single-rank halo routing.
  - The first multi-rank zeroextrap slices now allow MPI decomposition in the physical zeroextrap direction and in the remaining periodic directions for x/y/z zeroextrap. The x/y/z physical convection/diffusion stencils now distinguish true physical faces from MPI internal interfaces and use halo-backed sixth-order central stencils on internal interfaces. `NP=2` and `NP=4` CPU/GPU statistics plus field comparisons pass for x/y/z zeroextrap with `lfilter=t,diffterm=t`; the expanded one-step matrix covers physical-direction decomposition for x, y, and z. The reusable matrix has also passed `MAXSTEP=5` with field output comparison and `MAXSTEP=20` as a statistics-only stability check for the earlier matrix variants.
- Phase C boundary expansion has started with CPU-compatible symmetry:
  - `bctype=60` symmetry is supported for exactly one physical x/y/z direction with the other two directions periodic.
  - The GPU implementation matches the CPU Cartesian behavior used by the TGV-template validation: zero the normal velocity component, extrapolate tangential primitive fields, rebuild `q`, and zero physical-boundary `qrhs`.
  - It reuses the finite-physical-axis infrastructure from zeroextrap, including `MPI_PROC_NULL` gating for true physical faces and halo-backed sixth-order central stencils on MPI internal interfaces.
  - Validation passed x/y/z single-rank five-step statistics and field comparison, the 18-entry NP=2/NP=4 symmetry matrix, direct y/z physical-direction NP=2 five-step checks, the zeroextrap Phase B regression matrix, and periodic TGV 10-step statistics regression.
  - This is not yet a general curvilinear symmetry implementation. Boundary-normal arrays such as CPU `bnorm_i0`, `bnorm_im`, and `bnorm_km` are not yet part of the GPU resident boundary contract.
- Phase D wall-boundary expansion has started with channel `bctype=41`:
  - The supported slice is `examples/Channel` with y-direction `bctype=41,41` isothermal no-slip walls and periodic x/z directions.
  - Channel initialization now uses a deterministic random seed so CPU/GPU validation runs start from the same perturbation field.
  - GPU support includes the y-wall boundary kernel, resident geometry `x_d`, channel statistics (`massflux`, `fbcx`, `forcex`, `wrms`), and the channel body-force source term in the GPU RHS path.
  - The channel source kernel reads `jacob_d` directly from `commarray_gpu`, matching the global RHS kernels. Passing the halo-bounded `jacob_d` through a `0:im,0:jm,0:km` dummy caused a force-proportional source error and is now avoided.
  - Single-rank filtered/diffusive feedback validation now passes through `MAXSTEP=20` at strict `1e-8` for both statistics and field output.
  - A fixed-force no-filter long-step gate now passes through `MAXSTEP=100` with `DIFFTERM=t`, `CHANNEL_FORCE_MODE=fixed`, and `CHANNEL_FORCE_FIXED=1.d-4`, with final field differences at roundoff scale.
  - `NP=2` channel slab validation now passes for `2x1x1`, `1x2x1`, and `1x1x2`. The `1x2x1` case validates physical y-direction decomposition: y-wall kernels are gated on true `MPI_PROC_NULL` faces, while the internal y interface uses halo-backed central stencils.
  - The `NP=2` two-step feedback matrix also passes statistics at `1e-8` and field comparison at `1e-6`.
  - `NP=4` channel combined-direction validation now passes for `2x2x1`, `2x1x2`, and `1x2x2`. One-step validation remains strict at `1e-8` for statistics and field output; two-step short-feedback validation passes with `STATS_ATOL=2e-8` and `FIELD_ATOL=1e-6`.
  - `NP=8` `2x2x2` channel validation passes as a two-GPU oversubscription correctness smoke with the same one-step and two-step contracts. It is not performance evidence.
  - `NP=27` `3x3x3` channel validation passes as a fully interior-rank halo smoke: one-step strict validation passes, and two-step short-feedback validation passes with `STATS_ATOL=3e-8` and `FIELD_ATOL=1e-6`. This topology confirms ranks fully wrapped by MPI neighbors in x/y/z, including y-interior ranks with no wall contact. It is severe two-GPU oversubscription, not performance evidence.
  - Filtered channel runs beyond the current 20-step gate are limited by the CPU baseline on the small validation grid: CPU `crashcheck` stops around 23 filtered steps at `DELTAT=5.d-4`, and around 20 filtered steps for smaller `DELTAT`. This must not be treated as GPU-only stability evidence.
- Phase J/K RTI and source-dispatch expansion has started:
  - `examples/Rayleigh–Taylor-Instability` is validated only as a forced 3-D explicit variant, not as the original compact 2-D input.
  - GPU support includes y-only fixed `bctype=31` boundary handling and the RTI gravity source term matching CPU `src_rti`.
  - `tests/gpu_validation/run_rti_phasej_matrix.sh` is the canonical RTI regression entry point; the current default run passes 13 entries covering single-rank filter/diffusion combinations, NP=2/NP=4/NP=8 halo topologies, and 20-step single-rank plus NP=4 checks.
  - GPU source terms now enter the RK loop through `solver_gpu:apply_case_sources_gpu()`, currently dispatching channel and RTI sources at the CPU-compatible RHS position.
  - `src_gpu/case_capability_gpu.cuf` is the first Phase K capability-table module. It currently centralizes source capability only: no source, channel source, or RTI source. Future boundary/case support checks can move into the same capability layer incrementally, but the current change intentionally avoids destabilizing the validated boundary kernels.
  - This is not yet a general wall-boundary implementation. Wall blowing/suction files, turbulence wall models, curvilinear wall-normal handling, larger-rank long channel validation, filtered-channel baseline redesign, and other wall `bctype` variants remain future work.
- Phase E-G wall-family expansion has added CPU-compatible Cartesian slices:
  - `bctype=42` adiabatic no-slip walls are supported for x/y only. z is intentionally rejected because CPU `noslip_adibatic` implements only `ndir=1..4`.
  - `bctype=411` slip-isothermal walls are supported for y only. x/z are intentionally rejected because CPU `slipisotwall` implements only `ndir=3/4`.
  - `bctype=421` slip-adiabatic walls are supported for y only. x/z are intentionally rejected because CPU `slipadibwall` implements only `ndir=3/4`; the active CPU formula has the `xslip` split commented out, and GPU follows that active formula.
  - These phases validate current CPU-compatible Cartesian behavior, not general wall-normal geometry, wall blowing/suction, turbulence wall models, species, chemistry, or compact schemes.
- Phase H wall-family regression is in place:
  - `tests/gpu_validation/run_wall_family_phaseh_matrix.sh` is the reusable wall-family regression entry point.
  - The supported matrix combines wall41 x/y/z, wall42 x/y, wall411 y, and wall421 y CPU-compatible slices with NP=2 physical or transverse slab coverage.
  - The reject matrix keeps unsupported CPU-scope directions closed: `42-z`, `411-x/z`, and `421-x/z`.
  - The full default matrix passed with 11 supported statistics/field entries and 5 expected rejects.
  - Phase H is a regression and scope-control layer. It does not add new boundary physics beyond the already validated slices.
- Phase I has started with `examples/Lid-Driven-Cavity` as the next true case candidate:
  - LDC is selected because it avoids species, chemistry, turbulence, and shock-capturing, but still exercises a real missing architecture feature: multiple physical boundary directions plus a top-lid `bctype=0` UDF boundary.
  - `tests/gpu_validation/run_ldcavity_phasei_gate.sh` records the current RED gate. The original 2D LDC input is rejected by the GPU path with `GPU first-stage supports 3D cases only`.
  - A forced `GRID=32,32,32` explicit probe now has a valid CPU oracle: LDC grid generation preserves the original zero-thickness z grid when `ka==0`, and uses unit z length when `ka>0`.
  - A one-step explicit CPU LDC probe now completes without `ieee_invalid`. The earlier warning was traced to Release/O2 evaluation of `gridcube(1.d0,1.d0,0.d0)` for a 2-D zero-thickness grid; `src/gridgeneration.F90` now precomputes guarded `dx/dy/dz` values so no non-taken `0/0` expression is present in the loop.
  - The forced 3-D CPU oracle completes and writes a grid with z range `0..1`.
  - The GPU boundary capability layer now identifies the LDC x/y physical plus top-lid UDF boundary pattern separately from generic unsupported boundary combinations, while keeping the old single-physical-axis supported paths unchanged.
  - A GPU LDC boundary routine has been added and is connected for the current no-filter LDC slices. It applies x isothermal no-slip walls, then the y lower isothermal no-slip wall, then the y upper moving lid. This intentionally preserves the CPU corner rule: `bc:boucon` applies `ndir=1..4` in order and `userdefine:udf_bc(ndir=4)` runs last, so top-lid velocity overwrites the static side-wall corner values.
  - The next implementation work is multi-axis physical filter ownership, not a new numerical scheme.
- Phase I-A/I-B have opened a deliberately narrow LDC execution slice:
  - `tests/gpu_validation/run_ldcavity_phaseia_compare.sh` validates forced 3-D LDC with `LFILTER=f` and `DIFFTERM=f/t`.
  - The GPU path now has dedicated x+y physical convection flux kernels. This is necessary because the existing x-physical kernels assume y is periodic, and the existing y-physical kernels assume x is periodic.
  - The GPU path now has dedicated x+y physical diffusion gradient, stored flux, and RHS kernels. This keeps x/y physical coordinates non-periodic while z remains periodic or MPI-exchanged.
  - The halo exchange layer now skips local periodic qswap/field qswap on every physical direction instead of relying on a single physical-axis id.
  - Single-rank 1-step and 5-step CPU/GPU statistics plus field comparisons pass at `GRID=32,32,32` for both no-diffusion and diffusive no-filter slices.
  - `NP=4, TOPOLOGY=2,2,1` 5-step diffusive LDC also passes statistics plus field comparison, covering x/y internal halos and true physical faces together.
  - The default filtered LDC path still rejects at the capability gate. Multi-axis explicit filtering remains the next LDC blocker.

The current baseline is a correctness-first CUDA Fortran implementation. It still intentionally uses explicit synchronization after kernels and host-staged halo buffers.

## 3. Architecture Decisions

The full-GPU migration will follow the decisions recorded in the project ADRs:

- `0013`: Use backend-neutral facades for full GPU migration.
- `0014`: Make GPU data authoritative inside the compute loop.
- `0015`: Use pluggable HaloTransport backends.
- `0016`: Expand full GPU coverage in ordered phases.
- `0017`: Keep file output as a CPU-owned boundary.
- `0018`: Use layered validation for full GPU migration.
- `0019`: Stabilize a full-GPU architecture skeleton before new physics.
- `0020`: Keep `src_gpu/` as the near-term CUDA backend directory.
- `0021`: Choose a non-reacting second validation case.

These decisions define the boundary between the short-term CUDA Fortran implementation and the long-term ASTR GPU architecture.

## 4. Core Architecture Principles

### 4.1 Backend-Neutral Facade

CPU-side `src/` must not become CUDA-specific. Public calls from `src/` into GPU execution must remain backend-neutral.

Allowed direction:

```text
src/
  CPU orchestration, runtime input, topology setup, output boundaries

src_gpu/
  current CUDA Fortran backend implementation
```

The CPU side should call facades such as:

```fortran
call gpu_runtime_init()
call gpu_case_supported()
call gpu_alloc_fields()
call gpu_upload_initial_state()
call gpu_step_rk()
call gpu_compute_statistics()
call gpu_download_for_output()
call gpu_finalize()
```

The CPU side should not directly depend on CUDA-specific APIs or CUDA-specific module names.

Current CUDA Fortran facade calls already used by `src/` are:

| Current facade | Current caller | Target architecture role |
|---|---|---|
| `gpu_bind_device()` | `src/astr.F90` | backend runtime initialization and device binding |
| `gpu_after_refcal()` | `src/astr.F90` | upload scalar/runtime metadata after reference calculation |
| `gpu_after_alloc()` | `src/astr.F90` | allocate device-owned fields after CPU allocation |
| `gpu_after_flowinit()` | `src/astr.F90` | upload initialized flow state and prepare GPU execution |
| `gpu_prepare_rkfirst_stats()` | `src/mainloop.F90` | prepare first-step statistics without CPU field ownership reversal |
| `gpu_exchange_solution_halo()` | `src/mainloop.F90` | backend-neutral solution halo exchange |
| `gpu_write_flow_statistics()` | `src/mainloop.F90` | GPU-resident TGV/HIT statistics or generic `maxq1..maxq5` flowstate output |
| `gpu_sync_flow_to_host()` | `src/mainloop.F90` | explicit output/checkpoint boundary download |
| `gpu_time_integration_rk()` | `src/mainloop.F90` | GPU-resident RK time integration |

Near-term architecture work should document the mapping before renaming public routines. A large facade rename is not required for the next phase unless it removes a real ambiguity.

### 4.2 GPU-Authoritative Compute Loop

During GPU execution, the resident compute-loop state is authoritative on the GPU.

GPU-authoritative fields include, in stages:

- `q`
- `qrhs`
- `qsave`
- `rho`, `vel`, `prs`, `tmp`
- geometry arrays required by GPU kernels
- filter ping-pong arrays
- diffusion flux arrays
- statistics partial sums
- future species, turbulence, chemistry, and immersed-boundary device data

Whole-field transfers are allowed only at explicit boundaries:

- initialization
- output
- checkpoint/restart
- CPU fallback transition
- debugging or validation hooks explicitly marked as such

Per-kernel or per-module whole-field D2H/H2D bridges are outside the architecture.

### 4.3 CPU-Owned Output Boundary

File output remains CPU-owned in the near and medium term:

- HDF5 `flowfield`
- checkpoint/restart files
- slice/list/monitor output
- controller reload
- existing CPU-owned output workflows

This means a whole-field D2H at an output boundary is acceptable. It must not be confused with a compute-loop residency failure.

Direct GPU HDF5 or checkpoint writing is not a next-phase requirement.

### 4.4 Pluggable HaloTransport

Halo semantics and halo transport must be separated.

The semantic layer defines what is exchanged:

- qswap-compatible `q` exchange
- dataswap-compatible filter exchange
- dataswap-compatible diffusion-field exchange
- interface-plane averaging where CPU `qswap` requires it
- `hm` versus `hm+1` transport width

The transport layer defines how it is exchanged:

```text
L0 host_staged_blocking
L1 host_staged_nonblocking
L2 pinned_host_staged
L3 CUDA-aware or HIP-aware MPI
L4 topology-aware multi-node transport
```

The current implementation is L0. Later transport backends should not change solver semantics or CPU orchestration call sites.

### 4.5 Ordered Physics Expansion

The migration must not jump directly from TGV into chemistry or immersed boundary support.

The planned expansion order is:

1. Architecture skeleton stabilization.
2. Non-reacting flow generalization.
3. Regular-grid boundary and source expansion for non-reacting cases.
4. Shock-capable non-reacting flow path, starting with Phase S0-A.
5. Optional species transport, reopened only for a concrete non-reacting or later combustion requirement.
6. Deferred turbulence models, reopened only if RANS/LES becomes a project requirement.
7. Deferred chemistry/combustion, reopened after species ownership and shock-capable flow paths are mature.
8. Immersed-boundary support.
9. Transport backend performance upgrades.

This keeps high-complexity physics from defining the architecture too early.

## 5. Target Module Boundaries

### 5.1 Runtime And Device Binding

Responsibilities:

- initialize GPU backend
- select device by node-local rank
- report oversubscription correctness mode when applicable
- guard unsupported GPU cases
- preserve runtime `use_gpu` semantics

Current CUDA backend location:

```text
src_gpu/gpu_runtime.cuf
src_gpu/device_runtime_gpu.cuf
src_gpu/commvar_gpu.cuf
```

### 5.2 Device Field Ownership

Responsibilities:

- allocate device fields
- upload initial state
- keep resident fields synchronized inside GPU execution
- provide explicit output-boundary download routines
- avoid hidden whole-field transfers inside solver modules

Current CUDA backend location:

```text
src_gpu/commarray_gpu.cuf
src_gpu/commvar_gpu.cuf
```

The field ownership table must include at least:

- solution fields: `q`, `qsave`, `qrhs`, `qwork`;
- primitive fields: `rho`, `vel`, `prs`, `tmp`;
- geometry and metric fields: `jacob`, `dxi`;
- filter ping-pong fields;
- diffusion fields: `sigma`, `qflux`;
- statistics partial sums and scalar reductions;
- future species, turbulence, chemistry, and immersed-boundary fields.

### 5.3 Solver Kernels

Responsibilities:

- filter
- gradcal
- convection
- diffusion
- RK update
- primitive refresh
- crash detection

Current CUDA backend location:

```text
src_gpu/gpu_check.cuf
src_gpu/solver_gpu.cuf
src_gpu/gradcal_gpu.cuf
src_gpu/mainloop_gpu.cuf
```

Near-term action:

Separate kernel taxonomy in documentation and interfaces before adding more physics. The code does not need an immediate directory split, but module responsibilities must be explicit.

### 5.4 Halo Exchange

Responsibilities:

- implement qswap-compatible solution exchange
- implement dataswap-compatible field exchange
- route x/y/z topology-general exchanges
- preserve private GPU halo MPI tags
- expose a future HaloTransport selection point

Current CUDA backend location:

```text
src_gpu/qswap_gpu.cuf
src_gpu/halo_exchange_gpu.cuf
```

Near-term action:

Split conceptual layers inside the module:

```text
pack/unpack kernels
halo semantics
L0 host-staged transport
future transport backend hook
```

Do not introduce CUDA-aware MPI as the only path.

### 5.5 Statistics And Reductions

Responsibilities:

- compute local numerator reductions on GPU
- move only scalar or partial-sum data to host
- perform MPI global reductions
- write native statistics from the I/O rank
- avoid rank-local normalized average errors

Current CUDA backend location:

```text
src_gpu/statistic_gpu.cuf
```

### 5.6 Validation Tooling

Responsibilities:

- produce reusable case preparation scripts
- support `NP`, `TOPOLOGY`, `MAXSTEP`, `FEQCHKPT`, `LFILTER`, `DIFFTERM`
- compare `flowstate.dat`
- compare HDF5 fields only at output boundaries
- record Nsight transfer and kernel summaries
- keep large runtime outputs out of git

Current location:

```text
tests/gpu_validation/
scripts/gpu_validation/
documents/GPU_VALIDATION_MATRIX.md
```

Current validation tooling gap:

- the `256^3 NP=1/NP=2` Nsight profile driver now has a post-script pass under `tests/gpu_validation/out/nsys_tgv_256_np1_np2_driver_current`;
- the reusable core multi-rank topology matrix now has a post-script pass under `tests/gpu_validation/out/tgv_mpirank_matrix_core_current`;
- generated runtime outputs and profile artifacts must stay out of git.

### 5.7 Build And Dependency Contract

The project must continue to build from the repository root `CMakeLists.txt`.

Current build contract:

- CPU build: MPI Fortran compiler, HDF5 Fortran/HL libraries.
- CUDA-capable build: NVHPC Fortran with CUDA Fortran support, MPI, HDF5, `-DASTR_WITH_CUDA=ON`.
- Runtime GPU selection: input-file `use_gpu=t/f`; `ASTR_WITH_CUDA` only controls whether GPU support is compiled into the binary.
- Topology override for validation: `ASTR_FORCE_MPI_TOPOLOGY=i,j,k`.

Profiling and runtime-observation dependencies:

- `nvidia-smi` and `nvitop` for device/process occupancy checks;
- Nsight Systems for CUDA/MPI timeline and D2H/H2D accounting;
- Python 3 validation scripts with HDF5/numerical dependencies used by `tests/gpu_validation`;
- plotting scripts may require `matplotlib` and `scienceplots`.

Future DCU/HIP work must not require changing CPU orchestration call sites. HIP/DCU dependencies belong behind a future backend implementation and HaloTransport backend, not inside `src/` solver orchestration.

Detailed follow-up documents:

- `documents/ASTR_GPU_DEVICE_FIELD_OWNERSHIP.md`
- `documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md`

### 5.8 Generated Artifact Policy

Do not commit generated runtime or profiling outputs:

- `tests/gpu_validation/out/`;
- `*.h5`, `*.dat`, `*.log`, `parallel.info`, `errnode.log`;
- `*.nsys-rep`, `*.sqlite`, exported Nsight CSV files;
- Python `__pycache__/` directories.

Validation documents should record compact summaries and exact commands, not large binary profiles or flowfield files.

## 6. Phased Roadmap

### Phase 0: Baseline Freeze

Goal:

Freeze the current TGV CUDA Fortran baseline as the reference point for future architecture work.

Tasks:

- Record the current validated topologies.
- Record `256^3 NP=1/NP=2` statistics comparison.
- Record Nsight Systems profile summaries.
- Keep old oversized runtime outputs out of git.
- Define a repeatable baseline validation command set.

Acceptance:

- TGV `NP=1` and `NP=2` pass statistics comparison.
- `2x2x2` small-step topology remains passing.
- Nsight report exists for `256^3 NP=1` and `NP=2`.

### Phase 1: Full-GPU Architecture Skeleton

Goal:

Turn the current TGV GPU implementation into a reusable architecture baseline.

Tasks:

- Define final facade names used by `src/`.
- Audit `src/` for CUDA-specific imports or assumptions.
- Document device field ownership.
- Document kernel taxonomy.
- Isolate HaloTransport L0 host-staged logic conceptually.
- Convert current validation commands into reusable scripts.
- Add validation report templates.

Acceptance:

- `src/` calls only backend-neutral GPU facades.
- No new CUDA-specific public API names are introduced in CPU orchestration.
- Existing TGV validation matrix remains passing.
- Profile and validation scripts can be rerun without manual case editing.

### Phase 2: Non-Reacting Flow Generalization

Goal:

Validate that the architecture is not TGV-specific.

Second-case criteria:

- non-reacting
- `num_species=0`
- no chemistry
- no turbulence
- no immersed boundary
- preferably non-TGV initialization, boundary, or geometry behavior

Initial candidate screen from `examples/`:

| Candidate | Input | Why useful | Main risk before GPU enablement |
|---|---|---|---|
| Sod | `examples/sod/datin/input.sod` | simple non-reacting shock tube; small CPU oracle | 1D path and boundary handling may differ from current 3D TGV assumptions |
| Shu-Osher | `examples/Shuosher/datin/input.shuosher` | non-reacting compressible wave/shock interaction | 1D path and shock-capturing behavior must be checked against explicit-scheme GPU support |
| Vortex transport | `examples/Vortex_Transport/datin/input.2dvort` | non-TGV vortical flow with different initialization | 2D path support and output/statistics oracle need definition |
| Riemann2D | `examples/Riemann2D/datin/input.riemann2d` | non-reacting 2D discontinuity problem | boundary conditions and numerical stability may expose unsupported code paths |
| Mixing layer | `examples/MixingLayer/datin/input.2d` | non-reacting shear-flow candidate | 2D boundary and statistics/output comparison need screening |
| Channel | `examples/Channel/datin/input.chl` | important future wall-bounded flow direction | turbulence/wall diagnostics may make it too early for Phase 2 |
| Rayleigh-Taylor | `examples/Rayleigh–Taylor-Instability/datin/input.rti` | selected Phase J non-shock explicit validation variant after LDC | original input is compact and 2-D; current GPU contract covers only the forced 3-D explicit variant with y fixed `bctype=31` and RTI source |

Phase 2 must first produce a candidate decision note with one selected case, one CPU oracle, and explicit reasons for rejecting the other candidates. Do not enable `use_gpu=t` for a candidate until its dimensionality, boundary conditions, model flags, and output oracle are known.

Tasks:

- Screen candidate examples.
- Choose one second validation case.
- Identify missing GPU support for its boundary and initialization path.
- Add only the minimum required GPU path.
- Validate same-topology CPU/GPU statistics and field output where appropriate.
- For boundary slices, validate field output first, then promote statistics after the corresponding non-periodic GPU `gradcal` diagnostics are implemented.

Acceptance:

- One non-TGV, non-reacting case runs through the GPU architecture skeleton.
- The solution does not introduce TGV-specific names into public GPU interfaces.
- The first non-periodic boundary slice preserves CPU `boucon` ownership of physical boundary planes and CPU `is/ie/js/je/ks/ke` active-range semantics.

### Phase 3: Species Transport

Goal:

Optional / deferred unless a non-reacting multi-species validation target becomes necessary. With chemistry and combustion deferred, species transport must not block the shock/SBLI GPU track.

Tasks:

- Keep the current GPU runtime gate rejecting `num_species > 0` until this phase is explicitly reopened.
- Reopen only for a concrete need such as passive scalar transport, non-reacting multi-species mixing, variable molecular-weight gas modeling, or later combustion prerequisites.
- When reopened, define device layout for species variables.
- Extend halo exchange to species fields.
- Extend diffusion and mixture-property support.
- Add species validation oracle.

Acceptance:

- Current acceptance: `num_species > 0` remains an explicit unsupported GPU path.
- Reopened acceptance: a selected non-reacting species case passes a defined CPU/GPU oracle.

### Phase 4: Turbulence Models

Goal:

Deferred / out of scope for the current migration track. RANS/LES model equations are not a near-term GPU target.

Tasks:

- Keep the GPU runtime gate strict: only `turbmode='none'` is accepted.
- Keep RANS/LES variables, model source terms, and wall-model diagnostics out of the current GPU resident data contract.
- Document this as a deliberate scope decision rather than an accidental missing feature.
- Reopen this phase only if RANS/LES becomes a project requirement.

Acceptance:

- GPU runs reject non-`none` turbulence modes clearly.
- The chemistry/combustion and shock/SBLI phases do not depend on RANS/LES support.

### Phase 5: Chemistry

Goal:

Deferred / planned for later. Add species transport and chemistry/combustion source-term capability only after the current non-reacting and shock-capable GPU paths are more mature.

Subphases:

- **Phase 5A: 0D/PSR chemistry source-only gate**
  - First chemistry acceptance target.
  - Use `examples/Perfectly_Stirred_Reactor` or `examples/air_reactor` style cases to isolate chemistry source evaluation from species convection, species diffusion, wall boundaries, and multi-dimensional halo effects.
  - Validate thermochemistry state update, reaction-rate evaluation, mass-fraction normalization, positivity, and CPU/GPU source-term oracle before opening transport coupling.
  - Current status: deferred. Cantera remains the CPU oracle/reference; no GPU chemistry backend has been selected.
- **Phase 5B: 1D flame transport gate**
  - Add species transport, species diffusion, and one-dimensional flame oracle after the source-only gate is stable.
- **Phase 5C: 3D flame gate**
  - Add `hitflame` or `tgvflame` style coupled flow/chemistry validation after species transport is resident and validated.
- **Phase 5D: chemistry with high-speed/shock coupling**
  - Combine chemistry with Phase S only after both standalone chemistry and standalone shock-capable paths have separate oracles.

Tasks:

- Keep the current GPU runtime gate rejecting chemistry/species cases until Phase 5 is reopened.
- Extend device data ownership from `numq=5` to species-bearing conservative variables.
- Extend species halo exchange, filter/diffusion participation, and boundary ownership rules.
- Decide chemistry backend strategy.
- Define mechanism/table ownership.
- Design batched source-term execution.
- Validate stiff source integration separately before full coupling.

Acceptance:

- Phase 5A: a selected 0D/PSR chemistry source-only case passes a defined CPU/GPU oracle.
- Later subphases: selected species-transport and coupled-flow cases pass their own CPU/GPU oracles.

### Phase S: Shock And High-Speed Wall-Bounded Flows

Goal:

Add GPU support for shock-capable numerical paths and shock-boundary-layer interaction style complex cases while keeping RANS/LES out of scope unless explicitly reopened.

Subphases:

- **Phase S0-A: shock-format readiness without walls**
  - First Phase S acceptance target.
  - Use a non-reacting, no-wall shock-format case to isolate upwind reconstruction, shock sensors, flux splitting, and shock-format RHS behavior.
  - The first selected case is a forced 3D extruded Sod case (`flowtype='sod'`) because it minimizes physics and boundary complexity while staying inside the current GPU 3D execution contract.
  - S0-A1 domain boundary: x carries the Sod discontinuity; y/z are uniform thin directions. The first validation grid is `GRID=200,8,8`.
  - S0-A1 time/boundary-condition boundary: use periodic boundaries in x/y/z and run `deltat=5.d-4`, `maxstep=20`, so the final time is `0.01` and the Sod wave system should not reach the x-periodic boundary.
  - Code screening note: `examples/sod/datin/input.sod` currently contains `flowtype=shuosher`, `GRID=200,0,0`, compact upwind, and `lchardecomp=t`; Phase S0-A1 must use a controlled validation input and the `sodini` initialization path explicitly rather than trusting that input file as-is.
  - Grid-generation note: current `sod` calls `grid1d(-5,5)`, and `grid1d` leaves z at zero even when `ka>0`. Phase S0-A1 therefore needs a positive-volume 3D extruded Sod grid path before GPU validation can pass the existing positive-Jacobian gate.
  - S0-A1 format boundary: support explicit upwind only with `conschm='543e'`. Compact upwind (`conschm(4:4)='c'`) remains out of scope.
  - S0-A1 reconstruction boundary: start with `recon_schem=-1` first-order Steger-Warming flux splitting as a plumbing/correctness gate. This validates the explicit upwind RHS chain and does not claim final shock-format accuracy.
  - S0-A1 physics-switch boundary: keep `diffterm=f` and `lfilter=f`; keep `difschm='643e'` as an explicit-center placeholder that does not participate while diffusion is disabled. Diffusion, explicit central filtering, shock-region filtering, and artificial/sensor-based dissipation are later gates.
  - S0-A1 validation oracle: compare same-input CPU/GPU statistics with `STATS_ATOL=1e-10`, `STATS_RTOL=1e-10`, and compare output-boundary fields with `FIELD_ATOL=1e-10`, `FIELD_RTOL=1e-10`. At minimum, report max differences for `q(:,:,:,1:5)`. If a specific, explainable floating-point difference appears, tolerances may be revisited to `1e-9` based on evidence rather than preemptively.
  - S0-A1 decomposition boundary: keep `lchardecomp=.false.`. Characteristic decomposition is a later optional gate, not part of the first shock-format implementation.
  - S0-A1 sensor boundary: port shock sensor logic only if the selected explicit reconstruction requires it; do not pull Ducros/MP-LD complexity into the first gate unless that reconstruction is explicitly chosen.
  - Later S0-A reconstruction gates should upgrade to WENO5/MP before MP-LD or sensor-coupled formats.
  - Later candidates include Shu-Osher and Riemann2D after dimensionality and boundary support are screened.
- **Phase S0-B: open-boundary and sponge readiness**
  - Add inlet/outlet/sponge/NSCBC behavior after the shock-format path itself is validated.
- **Phase S1: high-speed wall-bounded flow without full SBLI coupling**
  - Validate laminar or DNS-like hypersonic boundary-layer style cases with `turbmode='none'`.
- **Phase S2: shock-boundary-layer interaction**
  - Combine shock-format, open boundaries/sponge, and high-speed wall treatment only after the separated gates pass.

Tasks:

- Define the supported shock-capturing format family and its GPU data dependencies.
- Port shock sensor logic, including Ducros-style sensor paths where required by the selected format.
- Define shock-region filtering or added-dissipation policy separately from the existing explicit central filter.
- Add inlet/outlet/sponge/high-speed wall boundary support required by the selected validation case.
- Select a laminar or DNS-like shock/SBLI validation case with `turbmode='none'`.
- Keep compact finite-difference and compact filter solvers out of scope unless a separate decision reopens them.

Acceptance:

- Phase S0-A: the S0-A1 forced 3D Sod gate passes same-input CPU/GPU statistics and `q(:,:,:,1:5)` field comparison with the documented tolerances.
- Later subphases: open-boundary/sponge, high-speed wall, and SBLI gates pass separate oracles before combined validation.
- The validation distinguishes numerical-format stability from GPU porting correctness.
- RANS/LES remains explicitly rejected unless Phase 4 is reopened.

### Phase 6: Immersed Boundary

Goal:

Add irregular geometry and immersed-boundary support after regular-grid flow paths are stable.

Tasks:

- Define mask/interpolation/search device data ownership.
- Identify CPU-owned preprocessing boundaries.
- Port runtime forcing/interpolation kernels.
- Validate a minimal IB case.

Acceptance:

- One selected IB case passes CPU/GPU validation within an agreed tolerance.

### Phase 7: Transport Backend Optimization

Goal:

Replace or augment L0 host-staged blocking MPI with faster HaloTransport backends.

Tasks:

- Add pinned host buffers.
- Add nonblocking host-staged MPI.
- Evaluate overlap.
- Evaluate CUDA-aware MPI on NVIDIA.
- Keep HIP/DCU-aware transport as a parallel design target.

Acceptance:

- New transport backends preserve correctness.
- Nsight profiles show reduced communication overhead or better overlap.
- The L0 host-staged backend remains available as a portable correctness reference.

## 7. Validation Strategy

The full-GPU migration uses a layered validation matrix.

### L0 Build And Runtime Contracts

- CPU/GPU build.
- LF input contract.
- runtime `use_gpu`.
- device binding.
- topology override.
- root `CMakeLists.txt` build path.
- clean CPU and CUDA-capable configure/build commands.
- negative runtime contract: `use_gpu=t` in a non-CUDA binary must stop clearly.
- line-ending regression: `examples/**/input.*` and `controller` must remain LF.

### L1 Module Equivalence

- updatefvar.
- qswap/dataswap/halo.
- filter.
- gradcal.
- convection.
- diffusion.
- statistics.
- unsupported GPU case guards.
- HaloTransport semantic splits: solution qswap, filter dataswap, diffusion-field dataswap.

### L2 Time Integration

- one-step.
- ten-step.
- 100-step smoke where useful.
- filter on/off.
- diffusion on/off.

### L3 Multi-Rank Correctness

- `1x1x1`.
- `2x1x1`.
- `1x2x1`.
- `1x1x2`.
- `2x2x1`.
- `2x1x2`.
- `1x2x2`.
- `2x2x2`.
- selected high-rank oversubscription smoke tests.

### L4 Performance And Residency

- no-checkpoint Nsight profile.
- D2H/H2D budget.
- kernel time breakdown.
- MPI/halo transfer profile.
- GPU memory footprint.
- `nvitop` or `nvidia-smi` device-process observation for representative runs.
- `256^3 NP=1/NP=2` profile driver with reusable command-line controls.
- output-boundary transfer accounting separated from compute-loop residency accounting.

### L5 Physics Expansion

- validated non-TGV non-reacting explicit cases.
- regular-grid wall/source cases.
- shock-format readiness cases.
- high-speed wall-bounded and SBLI cases.
- optional species case if reopened.
- deferred turbulence and chemistry cases if reopened.
- immersed-boundary case.

## 8. Immediate Next Work

Recommended immediate work after this plan:

1. Implement Phase S0-A1 as the next feature gate: forced 3D extruded Sod, `GRID=200,8,8`, `deltat=5.d-4`, `maxstep=20`, all periodic, `conschm='543e'`, `recon_schem=-1`, `lchardecomp=.false.`, `diffterm=f`, and `lfilter=f`.
2. Add a positive-volume 3D Sod grid path so forced 3D `flowtype='sod'` does not reuse the current zero-thickness `grid1d` z coordinate.
3. Extend the GPU capability gate to accept only this controlled S0-A1 shock-format path; keep compact upwind, characteristic decomposition, sensors, species, turbulence, chemistry, and open boundaries rejected.
4. Implement the GPU first-order Steger-Warming explicit upwind RHS path by matching CPU `convrsduwd` and `recons_exp(..., recon_schem=-1)` semantics.
5. Add a reusable S0-A1 validation driver that performs CPU/GPU statistics comparison and `q(:,:,:,1:5)` field maximum-difference reporting at the documented `1e-10` tolerances.
6. Keep `tests/gpu_validation/run_source_phasek_matrix.sh`, `tests/gpu_validation/run_rti_phasej_matrix.sh`, and the wall-family/channel/TGV regression drivers as the required regression set after touching capability gates, source dispatch, boundary logic, or common solver kernels.
7. Keep `documents/GPU_VALIDATION_MATRIX.md`, `tests/gpu_validation/README.md`, `CONTEXT.md`, `documents/ASTR_GPU_DEVICE_FIELD_OWNERSHIP.md`, and `documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md` synchronized with each new device field, case capability, halo semantic, and validation result.

## 9. Explicit Non-Goals

The next architecture phase will not:

- rewrite `src/` as CUDA-specific code;
- move all directories into `src_backend_cuda/` immediately;
- require CUDA-aware MPI as the only transport;
- port HDF5/checkpoint writing to GPU;
- reopen compact finite differences or compact filters;
- start shock work with full SBLI, open-boundary, or sensor-coupled formats;
- start with chemistry;
- start with immersed boundary;
- treat two-GPU oversubscription runs as performance proof;
- accept per-kernel whole-field D2H/H2D bridges as normal GPU execution.

## 10. Risk Register

### Risk: TGV-Specific Architecture

The current implementation was developed through TGV. Public API names and data ownership rules must not bake in TGV assumptions.

Mitigation:

- enforce backend-neutral facade naming;
- select a second non-reacting validation case;
- keep TGV-specific logic behind case guards.

### Risk: Communication Overhead

The L0 host-staged halo path is portable but expensive. Nsight already shows halo-buffer D2H/H2D clearly in multi-rank profiles.

Mitigation:

- keep L0 as correctness baseline;
- add pinned and nonblocking staged backends;
- evaluate device-aware MPI later.

### Risk: Output Boundary Confusion

Whole-field D2H at HDF5/checkpoint boundaries can be mistaken for compute-loop residency failure.

Mitigation:

- document CPU-owned output boundary;
- profile with checkpoint disabled for residency audits;
- report output-boundary transfers separately.

### Risk: Shock-Format Scope Creep

Shock-capable cases can easily pull in open boundaries, sponge/NSCBC, shock sensors, characteristic decomposition, WENO/MP/MP-LD, high-speed walls, and SBLI coupling before the explicit upwind RHS path is validated.

Mitigation:

- keep S0-A1 limited to forced 3D Sod with periodic boundaries, first-order Steger-Warming, no filter, and no diffusion;
- promote WENO/MP, sensors, open boundaries, and wall coupling only through separate gates;
- require CPU/GPU statistics and field oracles for each shock subphase.

### Risk: Premature Physics Expansion

Species, turbulence, chemistry, and immersed boundary can each force major data model changes.

Mitigation:

- keep species, RANS/LES, and chemistry explicitly deferred unless reopened by a concrete requirement;
- add physics in ordered phases;
- require explicit validation oracles per phase.

## 11. Success Definition

The current full-GPU architecture phase is successful when:

- the current TGV path remains validated;
- `src/` depends only on backend-neutral GPU facades;
- GPU field ownership is documented and enforced;
- HaloTransport L0 is isolated as a backend rather than mixed into solver semantics;
- validation scripts are reusable;
- non-TGV explicit cases, regular-grid boundary slices, source dispatch, and wall-family regressions remain covered by reusable validation drivers;
- Nsight profiles can be generated reproducibly for `NP=1` and `NP=2`;
- Phase S0-A1 has a clear implementation and validation contract before any higher-order shock, open-boundary, high-speed wall, or SBLI work starts.
