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
  - `lfilter=f` and `diffterm=f` so the first boundary slice validates convection/RK/boundary ordering before non-periodic filter and diffusion stencils are opened.
  - CPU `parallelini` now initializes single-rank non-homogeneous active ranges to interior nodes; otherwise the CPU baseline effectively skips interior RHS for this finite-domain test.
  - GPU x-zeroextrap RHS kernels use the same `is:ie/js:je/ks:ke` active ranges as CPU.
  - `MAXSTEP=1` and `MAXSTEP=5` CPU/GPU `flowfield.h5` comparisons pass at roundoff scale; statistics are intentionally disabled for this slice because GPU diagnostic gradients remain periodic-first.

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
3. Species transport.
4. Turbulence models.
5. Chemistry.
6. Immersed-boundary support.
7. Transport backend performance upgrades.

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
| Rayleigh-Taylor | `examples/Rayleigh–Taylor-Instability/datin/input.rti` | non-TGV 3D instability candidate | initialization, boundary, and stability requirements need inspection |

Phase 2 must first produce a candidate decision note with one selected case, one CPU oracle, and explicit reasons for rejecting the other candidates. Do not enable `use_gpu=t` for a candidate until its dimensionality, boundary conditions, model flags, and output oracle are known.

Tasks:

- Screen candidate examples.
- Choose one second validation case.
- Identify missing GPU support for its boundary and initialization path.
- Add only the minimum required GPU path.
- Validate same-topology CPU/GPU statistics and field output where appropriate.
- For boundary slices, validate field output first and only promote statistics after non-periodic GPU `gradcal` diagnostics are implemented.

Acceptance:

- One non-TGV, non-reacting case runs through the GPU architecture skeleton.
- The solution does not introduce TGV-specific names into public GPU interfaces.
- The first non-periodic boundary slice preserves CPU `boucon` ownership of physical boundary planes and CPU `is/ie/js/je/ks/ke` active-range semantics.

### Phase 3: Species Transport

Goal:

Extend device data ownership and solver kernels from `numq=5` toward species variables.

Tasks:

- Define device layout for species variables.
- Extend halo exchange to species fields.
- Extend diffusion and mixture-property support.
- Add species validation oracle.

Acceptance:

- A selected non-reacting or weakly reacting species case passes a defined CPU/GPU oracle.

### Phase 4: Turbulence Models

Goal:

Add GPU-resident turbulence variables and model source terms.

Tasks:

- Identify turbulence model variable ownership.
- Add halo and boundary support for turbulence variables.
- Port model source terms incrementally.
- Validate wall or near-wall diagnostics where applicable.

Acceptance:

- One selected turbulence case passes CPU/GPU statistics and diagnostic comparisons.

### Phase 5: Chemistry

Goal:

Add chemistry source-term capability without destabilizing the main flow architecture.

Tasks:

- Decide chemistry backend strategy.
- Define mechanism/table ownership.
- Design batched source-term execution.
- Validate stiff source integration separately before full coupling.

Acceptance:

- A selected chemistry case passes source-term and coupled-flow validation.

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

- second non-reacting case.
- species case.
- turbulence case.
- chemistry case.
- immersed-boundary case.

## 8. Immediate Next Work

Recommended immediate work after this plan:

1. Use `tests/gpu_validation/run_tgv_256_nsys_profile.sh` as the repeatable `256^3 NP=1/NP=2` profile driver for future regression checks.
2. Use `tests/gpu_validation/run_tgv_mpirank_matrix.sh` as the repeatable core multi-rank validation matrix; high-rank oversubscription smoke remains opt-in with `RUN_SMOKE=t`.
3. Keep `tests/gpu_validation/README.md` aligned with current multi-rank TGV scope.
4. Audit `src/` for direct CUDA-specific dependencies and record current/target facade mapping.
5. Keep `documents/ASTR_GPU_DEVICE_FIELD_OWNERSHIP.md` synchronized with new device arrays.
6. Keep `documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md` synchronized with halo exchange changes.
7. Screen `examples/` for the second GPU validation case.
8. Update `documents/GPU_VALIDATION_MATRIX.md` with the latest profile and comparison results.
9. Make generated profile/runtime artifact exclusions explicit in `.gitignore`.

## 9. Explicit Non-Goals

The next architecture phase will not:

- rewrite `src/` as CUDA-specific code;
- move all directories into `src_backend_cuda/` immediately;
- require CUDA-aware MPI as the only transport;
- port HDF5/checkpoint writing to GPU;
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

### Risk: Premature Physics Expansion

Species, turbulence, chemistry, and immersed boundary can each force major data model changes.

Mitigation:

- stabilize architecture skeleton first;
- add physics in ordered phases;
- require explicit validation oracles per phase.

## 11. Success Definition

The full-GPU architecture skeleton phase is successful when:

- the current TGV path remains validated;
- `src/` depends only on backend-neutral GPU facades;
- GPU field ownership is documented and enforced;
- HaloTransport L0 is isolated as a backend rather than mixed into solver semantics;
- validation scripts are reusable;
- a second non-reacting case has a clear selection and validation plan;
- Nsight profiles can be generated reproducibly for `NP=1` and `NP=2`.
