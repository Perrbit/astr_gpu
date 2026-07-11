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
- filter ping-pong arrays, with an optional single-scalar-field work-array filter path for memory-constrained runs
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

#### 4.2.1 Explicit Filter Temporary Storage Policy

The current full-variable explicit-filter ping-pong implementation remains the default and validated path. It must be preserved as a fallback because it is simple, bandwidth-friendly, and already covered by the TGV, `2dvort`, HIT, boundary-slice, channel, LDC, and RTI validation history.

Add a second optional filter implementation for memory-constrained cases: a single 3-D scalar work array reused across conservative variables. This is not a replacement for the full `q` ping-pong path. It is an additional backend option with the same numerical operator and the same halo semantics.

Target execution model:

```text
for ivar = 1, numq
  filter one conservative component through q(:,:,:,ivar) and work(:,:,:)
  write the filtered result back to q(:,:,:,ivar)
end for
```

The optional work-array path must follow these rules:

- Preserve the explicit tenth-order center-filter coefficients and CPU `filterq` update timing.
- Preserve dataswap-compatible filter halo semantics: exchange or locally refresh exactly `hm` halo layers without qswap endpoint averaging.
- Keep current full-variable ping-pong kernels available and selectable until the work-array path has comparable validation coverage.
- Do not implement unsafe direct in-place overwrite of a 10th-order stencil field. A point may be overwritten only after no later stencil read needs its old value; this rule is difficult to maintain across GPU blocks and MPI faces, so it is not the near-term path.
- Validate with old-GPU/full-ping-pong versus new-GPU/work-array comparisons in addition to CPU/GPU comparisons.

Expected tradeoff:

- Memory: extra filter storage decreases from one full `numq` conservative field set to one scalar 3-D field, about `1/numq` of the current ping-pong temporary storage.
- Performance: kernel launches and repeated passes may increase, especially for small `numq=5`; this is acceptable only as an optional low-memory path until profiling proves otherwise.

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
  - S0-A1 status: implemented and passed for `NP=1`. The canonical `GRID=200,8,8`, 20-step CPU/GPU comparison passed statistics and full-field checks at `1e-10`; maximum conservative-field difference was `q5=1.7763568394002505e-15`.
  - S0-A2 status: implemented and passed for `NP=1`. It retains the S0-A1 periodic, no-diffusion, no-filter, no-characteristic-decomposition contract and sets `recon_schem=1`. CPU `recons_exp` uses WENO7 on these periodic interfaces. The canonical 20-step run passed at `1e-10`, with maximum conservative-field difference `q5=1.3322676295501878e-15`.
  - S0-A2 accuracy status: implemented and passed for CPU and GPU using `GRID=800,5,5`, `deltat=5.d-4`, `maxstep=400`, and the standard ideal-gas Sod exact solution. The gate separately measures smooth-region normalized errors, primitive-variable bound violations, contact/shock 10%-90% thickness, and wave-position error. The calibrated limits are `L1<=1e-3`, `L2<=6e-3`, `Linf<=6e-2`, bound violation `<=2e-2`, contact thickness `<=4` cells, shock thickness `<=3` cells, and position error `<=1` cell. The first passing run measured maxima `7.1864e-4`, `4.0087e-3`, `4.6338e-2`, `1.3889e-2`, `2.9780`, `2.1989`, and `0.7527`, respectively.
  - S0-A3 status: implemented and passed for `NP=1` with `recon_schem=3` MP7 under the same controlled periodic contract. The 20-step comparison passed at `1e-10`, with maximum conservative-field difference `q5=1.3323e-15`. The unchanged exact-solution thresholds also passed; maximum smooth `L1/L2/Linf` were `7.2439e-4/4.1951e-3/4.8797e-2`, maximum bound violation was `8.5021e-3`, contact/shock thicknesses were `2.9496/1.6939` cells, and maximum position error was `0.7272` cells.
  - S0-A4 sensor-only status: implemented and passed for `NP=1` on a forced 3D Shu-Osher case with `GRID=400,8,8`, `deltat=1.d-4`, `maxstep=1`, periodic boundaries, MP7 physical-space reconstruction, no diffusion, no filter, and no characteristic decomposition. The full raw-sensor field differed by at most `1.1102230246251565e-16`; CPU and GPU both marked 1944 shock nodes with zero mask mismatches. The one-step conservative-field difference was at most `q5=4.4408920985006262e-16`, and Compute Sanitizer reported zero errors.
  - S0-A5 MPI sensor-halo status: implemented and passed for `NP=2`, `TOPOLOGY=2,1,1` on the same `GRID=400,8,8` forced-3D Shu-Osher gate. `ASTR_SHUOSHER_SHOCK_X=0.d0` places the discontinuity beside the global x-slab interface so its `hm`-expanded region spans both ranks. The merged global raw sensor differs by at most `1.1102230246251565e-16`, masks match exactly with 1944 global shock nodes, and the one-step conservative-field maximum is `q5=2.7089441800853820e-14`. The two-rank Compute Sanitizer run reports zero errors when MPI CUDA probing components are disabled.
  - S0-A6 selective Roe status: implemented and passed for `NP=1` on forced-3D periodic Shu-Osher with `GRID=400,8,8`, `deltat=1.d-4`, `recon_schem=3`, `lchardecomp=t`, no diffusion, and no filter. The raw sensor remains CPU-equivalent (`L_inf=1.1102230246251565e-16`, exact 1944-node mask). Sensor-active interfaces use Roe projection, MP7 in characteristic space, and conservative-space back projection; smooth interfaces retain physical-space MP7. One-step statistics differ by at most `4.9098503041022923e-12` and `q5 L_inf=4.4408920985006262e-16`; the three-step run passes with `q5 L_inf=1.4210854715202004e-14`. Compute Sanitizer reports zero errors. The prescribed `(512,1,1)` x block requires a file-local `-gpu=maxregcount:128` cap because the unconstrained kernel used 255 registers/thread. The selected performance implementation generates each five-component Steger-Warming split once for the seven positive and seven negative MP7 stencil points, then projects those cached thread-local values into characteristic space. It adds no global field, kernel, or synchronization. Relative to the pre-cache baseline, x/y/z characteristic-flux mean times improve `1.93x/2.52x/2.74x`; the x kernel remains at 128 registers/thread and `1976` static stack bytes, but reduces actual local spill requests and raises achieved occupancy from about 14% to 22%.
  - S0-A7 MPI selective Roe status: implemented and passed for `NP=2`, `TOPOLOGY=2,1,1`. The Shu-Osher jump is placed at the global x-slab interface with `ASTR_SHUOSHER_SHOCK_X=0.d0`; raw `ssf` exchanges through the existing `hm`-layer field transport before local mask expansion and characteristic interface selection. The global raw field remains `L_inf=1.1102230246251565e-16` with an exact 1944-node mask. The three-step field comparison passes with `q5 L_inf=9.1482377229112899e-14`; both Compute Sanitizer ranks report zero errors. S0-A8/A9 subsequently validate equivalent y/z slabs under the rankwise duplicate-mask contract, and S0-A10 validates combined `2x2x2` routing.
  - S0-A8 MPI selective Roe status: implemented and passed for `NP=2`, `TOPOLOGY=1,2,1` on `GRID=400,16,8`, ensuring local `jm=8>=hm`. CPU `dataswap(ssf)` preserves raw-sensor overlap but assigns the expanded mask locally on the duplicated y interface, so this gate compares each CPU/GPU rank's local raw field and mask rather than merging masks across that face. Three steps pass with local sensor/mask agreement, `q5 L_inf=8.8817841970012523e-16`, and two zero-error Compute Sanitizer reports. The local-size precondition applies to every decomposed axis.
  - S0-A9 MPI selective Roe status: implemented and passed for `NP=2`, `TOPOLOGY=1,1,2` on `GRID=400,8,16`, ensuring local `km=8>=hm`. It uses the same rankwise raw-sensor and locally owned-mask comparison as S0-A8. Three steps pass with `q5 L_inf=8.8817841970012523e-16`, and both Compute Sanitizer ranks report zero errors.
  - S0-A10 combined MPI selective Roe status: implemented and passed for `NP=8`, `TOPOLOGY=2,2,2` on `GRID=400,16,16`, ensuring every local active extent is at least `hm` (`200,8,8`). It simultaneously exercises x/y/z raw-sensor halo transport and rankwise local-mask ownership. Three steps pass with zero mask mismatches and `q5 L_inf=8.8817841970012523e-16`; all eight Compute Sanitizer ranks report zero errors. On the current two-GPU workstation this is correctness-only oversubscription evidence, not a scaling result.
  - S0-A2/A3 storage: one haloed scalar `flux_work_d` is reused across direction and conservative component. Positive reconstruction writes it, negative reconstruction accumulates into it, and a third kernel applies the directional flux difference to `qrhs_d`; every kernel is explicitly synchronized. Generic explicit-reconstruction kernels select WENO7 or MP7 from `recon_schem` without allocating format-specific fields.
  - S0-A4/A5/A6 storage and scope: `shock_sensor_d` is a haloed one-component `real(8)` device field accepted by the generic `exchange_field_halo_gpu` interface; `shock_mask_d` is an active-domain one-byte mask. S0-A6 adds one reusable five-component `flux_characteristic_work_d` for directional final interface fluxes. `ASTR_SHOCK_SENSOR_DUMP` enables the controlled S0-A6 validation capability and sensor dump. S0-A7 through S0-A10 consume the same storage under x, y, z, and combined decompositions.
  - The completed S0-A1/A2/A3 flux gates, S0-A4/A5 sensor gates, and S0-A6 through S0-A10 selective-Roe gates do not cover MP5 physical-boundary degradation, MP-LD coupling, diffusion, or filtering. S0-B0 subsequently closes the `bctype=50,50` x physical-face MP7 and selective-Roe prerequisite for NP=1 and `2x1x1`, including face-clamped Ducros raw/mask semantics; it still does not cover inflow/outflow, NSCBC, sponge, diffusion, or filtering.
  - Shu-Osher is now used by the sensor-only gate. Riemann2D remains a later candidate after dimensionality and boundary support are screened.
- **Phase S0-B: open-boundary and sponge readiness**
  - Add inlet/outlet/sponge/NSCBC behavior after the shock-format path itself is validated.
  - The selected entry case is documented in `documents/ASTR_S0B_CASE_SCREENING.md`: a forced-3D `OpenShock` gate first uses classical `bctype=11/21`, then `12/22` NSCBC, then an x-max sponge. Existing hypersonic boundary-layer, mixing-layer, and SWLBI inputs are deferred because they combine unsupported compact, wall, farfield, sponge, curvilinear, or immersed-boundary features.
  - S0-B0 status: complete only for the finite-domain `bctype=50,50` prerequisite. Forced-3D Sod validates scalar MP7 physical-face reconstruction; forced-3D Shu-Osher validates the same face sequence under selective Roe and the Ducros sensor. Both NP=1 and `NP=2 TOPOLOGY=2,1,1` pass CPU/GPU field and sensor oracles, and the two-rank characteristic path passes memcheck. The CPU `npdci=4` sensor clamp was corrected because it previously read uninitialized physical-face halos. The next implementation target remains face-specific `11/21`, not a production open-boundary claim.
  - S0-B1 status: complete for the controlled stationary Mach-3 normal-shock gate only. `openshock` uses x-min `11,free`, x-max `21,pout`, and `pout=p2/pinf=10.333333333333333`; its GPU free-stream inflow and classical x-outflow kernels pass ten-step CPU/GPU field comparisons for NP=1 and `NP=2 TOPOLOGY=2,1,1`, with two-rank memcheck clean. This does not enable NSCBC, sponge, diffusion, filtering, or characteristic Roe for open boundaries.
  - S0-B2 status: complete for the restricted Cartesian `12/22` OpenShock gate. CUDA Fortran implements CPU-form characteristic matrices, explicit transverse terms, separate inlet-domain and outlet-plane Mach reductions, and the CPU sixth-order explicit outlet y-filter using `qwork_d`. CPU `time_integration_rk` now performs a full-rank pre-`boucon` `qswap` only when `bctype=22` is present, so the outlet filter reads current transverse halos without a rank-local MPI call; the existing post-boundary `qswap` remains. Ten steps pass at `1e-10` for NP=1 (`q5 L_inf=8.8817841970012523e-16`) and NP=2 `2x1x1` (`q5 L_inf=9.9920072216264089e-16`); both NP=2 Compute Sanitizer ranks report zero errors. This is not curved/y-z NSCBC, species, global sponge, or characteristic-Roe open-boundary support.
  - S0-B3 status: complete only for the x-max `spg_im=80` layer in the same explicit Cartesian `12/22` OpenShock gate. Every RK update refreshes the required q halos before the x-max rank applies the CPU-equivalent conservative-variable second-order six-neighbor smoothing with the uploaded geometric coefficient field, staging only its active layer through `qwork_d`, then recomputes primitive variables. There is no whole-field ping-pong allocation or stage-level host transfer. Ten-step NP=1 and NP=2 `2x1x1` field/statistics comparisons both pass with `q5 L_inf=8.8817841970012523e-15`; the NP=2 one-step Compute Sanitizer run is clean on both ranks. x-min/y/z/circular sponge, global filter, diffusion, species, curved geometry, and characteristic Roe remain separate work.
  - S0-B4 status: complete for the B3 x-max layer under the combined `NP=8 TOPOLOGY=2,2,2` gate on `GRID=400,16,16`. The test exposed and corrected a CPU correctness defect: the six-neighbor sponge stencil had refreshed only the nominal layer axis, leaving y/z MPI halos stale after RK updates. CPU now calls full `dataswap(q)` per layer; GPU uses a q-only three-direction sponge exchange before the local update. The ten-step field/statistics gate passes with `q5 L_inf=8.8817841970012523e-15`; all eight one-step sanitizer ranks report zero errors. This two-GPU oversubscribed execution is not performance evidence.
- **Phase S1: high-speed wall-bounded flow without full SBLI coupling**
  - Validate laminar or DNS-like hypersonic boundary-layer style cases with `turbmode='none'`.
  - S1-A0 status: complete for a controlled, non-dimensional, Cartesian, z-extruded `bl` gate. It uses `64x64x8`, `Mach=0.3`, `643e/643e`, `rk3`, diffusion enabled, no global filter, x `11,prof/21`, y `41/51`, and periodic z. GPU uploads the CPU-read `inlet.prof` profile once after `flowinit`; it applies profile inflow, CPU-active `51` upper-boundary extrapolation, and the existing lower isothermal wall in CPU `boucon` order. The dedicated xy physical gradient kernel permits device-resident `massflux/fbcx/wallheatflux/wrms` statistics. CPU/GPU fields and all four statistics pass for 2 and 20 steps at `1e-10`; the valid one-rank Compute Sanitizer run reports zero errors. This is an explicit-central boundary and statistics gate, not high-Mach validation, curved-grid support, a full characteristic farfield, or y-physical MP7 reconstruction.
  - S1-A1 status: complete for the same controlled gate with explicit MP7 convection (`543e/643e`, `recon_schem=3`). CPU `convrsduwd` and `flux:recons_exp` were corrected for single-rank `npdci=npdcj=4`: both physical ends use the two-point/SUW3/MP5/MP7 sequence and the y/z split ranges are `[0,dim]`. CUDA mirrors that x/y face degradation, constrains all three directional RHS paths to the CPU active ranges, and reads local plus exchanged-halo geometry metrics directly during Steger-Warming reconstruction. The 2- and 20-step CPU/GPU field/statistics gates pass at `1e-10`; the single-rank 20-step result has `q5 L_inf=1.7763568394002505e-14` and maximum statistic difference `9.6256336235001072e-14`. The same gate is complete for `NP=2 TOPOLOGY=2,1,1`, `NP=2 TOPOLOGY=1,2,1`, `NP=2 TOPOLOGY=1,1,2`, `NP=4 TOPOLOGY=2,2,1`, `NP=4 TOPOLOGY=2,1,2`, `NP=4 TOPOLOGY=1,2,2`, and `NP=8 TOPOLOGY=2,2,2`. The y-slab gate validates physical-wall ownership: only the global lower-y rank contributes BL wall shear, heat flux, and wall area to the device reduction. The y/z 20-step gates both have `q5 L_inf=1.7763568394002505e-14`; their maximum statistic differences are `5.4956039718945249e-14` and `5.4733995114020217e-14`. The oversubscribed `2x2x1` gate has `q5 L_inf=1.7763568394002505e-14`, maximum statistic difference `4.9737991503207013e-14`, and four zero-error memcheck reports. The oversubscribed x/z `2x1x2` and y/z `1x2x2` gates each have `q5 L_inf=1.7763568394002505e-14`, maximum statistic difference `5.0071058410594561e-14`, and four zero-error memcheck reports. The oversubscribed three-direction `2x2x2` gate uses `64x64x16`, so every local active extent is at least `hm` (`32x32x8`); it has `q5 L_inf=1.7763568394002505e-14`, maximum statistic difference `4.9737991503207013e-14`, and eight zero-error memcheck reports. Curved geometry, high-Mach stability, and full characteristic farfield support remain separate work.
  - S1-B0 status: complete for an `examples/Hypersonic_Boundary_Layer`-inspired, controlled 3-D explicit gate. The original HBL inputs are 2-D, compact, and mostly dimensional, so they are not GPU inputs. The new `96x96x8` z-extruded gate instead uses the original M3 x extent `[-1,10]`, wall-normal clustering, `Mach=3`, `Re=100000`, and the M3 wall-temperature ratio `568.89/226.65=2.5093503198764615`, with a deterministic heated boundary-layer profile. It retains the already validated non-dimensional explicit MP7 `543e/643e`, profile inflow, `11/21/41/51`, diffusion, no global filter, no characteristic decomposition, and periodic z contract. The 20-step CPU/GPU comparison passes with `q5 L_inf=5.5511151231257827e-16` and maximum statistic difference `7.8936857050848630e-14`; one-rank Compute Sanitizer reports zero errors. This is a high-Mach parameter and heated-wall porting gate only. It does not validate the original two-dimensional compact/dimensional HBL inputs, a true characteristic farfield, or SBLI physics.
  - S1-B1 status: complete for the controlled HBL gate in `NP=2` slabs. Twenty-step CPU/GPU field and statistic gates pass for x `TOPOLOGY=2,1,1`, y `1,2,1`, and z `1,1,2`. All three have `q5 L_inf=5.5511151231257827e-16`; maximum statistic differences are `6.3726801613483985e-14`, `5.9729998724833422e-14`, and `7.8936857050848630e-14`, respectively. The z gate uses `KM=16` so each local z extent is eight points and satisfies `local_n>=hm`; the x/y gates retain `KM=8`. The physical-y two-rank Compute Sanitizer run reports zero errors for both ranks. This establishes first MPI correctness for the high-Mach heated-wall parameters, not production multi-GPU scaling or combined-axis HBL decomposition.
  - S1-B2 status: complete for all controlled HBL `NP=4` two-axis decompositions. Twenty-step CPU/GPU gates pass for `2x2x1`, `2x1x2`, and `1x2x2`; each has final `q5 L_inf=5.5511151231257827e-16`. Their maximum statistic differences are `5.0182080713057076e-14`, `6.3726801613483985e-14`, and `5.9729998724833422e-14`, respectively. Both z-decomposed cases use `KM=16` and thus local `km=8>=hm`. The oversubscribed `1x2x2` Compute Sanitizer run reports zero errors from all four ranks. These are combined-halo correctness results on two shared GPUs, not a scaling claim; full `2x2x2` high-Mach HBL remains a separate gate.
  - S1-B3 status: complete for controlled HBL `NP=8 TOPOLOGY=2,2,2`. The `96x96x16` gate gives local active domains `48x48x8`, all at least `hm`. Its 20-step CPU/GPU comparison has `q5 L_inf=5.5511151231257827e-16` and maximum statistic difference `5.0182080713057076e-14`; all eight ranks report zero-error Compute Sanitizer summaries. As with the NP=4 cases, this is two-GPU-oversubscribed full three-direction halo correctness evidence only, not a performance or production scaling result.
  - S1-C1 status: complete for a single-rank `Mach=5` Sutherland-viscosity similarity-inlet gate. `generate_compressible_blasius_profile.py` solves the coupled zero-pressure-gradient, isothermal-wall compressible Blasius ODE with `gamma=1.4`, `Pr=0.72`, `T_ref=226.65 K`, `Re=1.83052e6`, and `Twall/Tinf=1176.64/226.65=5.191440547760865`. The `192x192x8` Cartesian explicit MP7 case uses the existing `11,prof/21/41/51` boundary contract, x extent `[-1,10]`, and a similarity station one reference length downstream of a virtual leading edge. The profile has `delta_99=9.5190551023136317e-3`, `delta_star=7.4566649943486243e-3`, and `theta=4.0299675024914713e-4`. The 20-step CPU/GPU comparison passes with `q5 L_inf=4.4408920985006262e-16` and maximum statistic difference `3.5171865420124959e-13`. `inletprofile` now supports both non-dimensional density contracts: its backward-compatible default reconstructs `rho` from `pinf,T`; a first-line `density=provided` declaration retains file density and stops unless its ideal-gas pressure is `pinf` within `1e-10`. Both modes pass two-step field/statistics comparison with maximum field difference `6.2172489379008766e-15`. This is a similarity-inlet and CPU/GPU consistency gate only: `blini` still copies one inlet profile throughout x, upper `51` remains extrapolative rather than characteristic, the grid is Cartesian, and no mesh/time/streamwise-development convergence or experimental comparison has yet been established.
  - S1-C2 status: complete for x-varying similarity-field initialization without GPU main-loop changes. The same generator maps one converged similarity solution at every x through `sqrt(2*x_s/Re)`, writes CPU-owned `datin/flowini3d.h5`, and selects existing `ninit=3`; `readflowini3d` initializes the host field once and the established `gpu_after_flowinit` upload keeps it device-resident thereafter. The virtual leading edge is `x=-2`, so the domain `[-1,10]` covers stations `x_s=[1,12]`. The `192x192x8`, 20-step NP=1 CPU/GPU gate passes with `q5 L_inf=7.7715611723760958e-16` and maximum statistic difference `3.3406610810970960e-13`. The same C2 field path passes NP=2 x and y slabs at the same grid, and the z slab at `192x192x16`; all retain `q5 L_inf=7.7715611723760958e-16`, with maximum statistic differences `1.9806378759312793e-13`, `2.9809488211185453e-13`, and `3.3406610810970960e-13`, respectively. This removes the C1 full-x initialization limitation and establishes single-axis HDF-input decomposition, but does not establish mesh/time convergence, characteristic farfield behavior, curved geometry, combined-axis HDF-input decomposition, or experimental agreement of skin friction and heat transfer.
  - S1-C3 status: complete for the current Cartesian upper `bctype=52` NSCBC farfield gate. Porting exposed a CPU oracle problem in `bc:farfield_nscbc`: for the `jmax` path, the local transverse filter could read stale k-halo values before a current halo refresh, so an initially constant physical upper boundary could be filtered to nonconstant values such as `0.84375..1.078125`. Following the CPU bug decision gate, this artifact was not reproduced in GPU code. CPU `time_integration_rk` now performs the same full-rank pre-`boucon` halo refresh when either `bctype=22` or `bctype=52` is present, so NSCBC transverse filters read current halos. GPU `52` support was then restored against the corrected oracle. `run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh` passes for the `192x192x8`, `Mach=5`, Sutherland-viscosity C1 setup with `maxstep=20`; the final reconstructed `q5 L_inf` is `1.4432899320127035e-15` and the maximum statistic difference is `3.4716673980028645e-13`. This validates the explicit Cartesian S1 farfield path only; curvilinear farfield, combined-axis C2 HDF-input decomposition, mesh/time convergence, and production SBLI farfield behavior remain separate gates.
- **Phase S2: shock-boundary-layer interaction**
  - Combine shock-format, open boundaries/sponge, and high-speed wall treatment only after the separated gates pass.
  - S2-A0 status: started with a deliberately narrow compatibility gate, not a resolved physical SBLI validation. `run_s2_hbl_oblique_shock_compare.sh` keeps the S1 Mach-5 Sutherland-viscosity flat-plate contract, uses `ninit=3`, and overlays an analytic perfect-gas oblique-shock state onto the x-varying compressible Blasius HDF field. The overlay updates `rho,T,u,v` consistently so CPU `readflowini3d` reconstructs pressure from `rho*T`. One- and two-step `64x64x8` CPU/GPU smoke gates pass at `1e-10`; the two-step smoke has reconstructed `q5 L_inf=8.8817841970012523e-16`. The default `192x192x8` two-step gate also passes with `q5 L_inf=1.3322676295501878e-15` and maximum statistic difference `1.0755840662568517e-12`. The same gate passes NP=2 x/y/z slabs, NP=4 xy/xz/yz planes, and `NP=8 TOPOLOGY=2,2,2`; z-decomposed cases use `KM=16` to keep local `km>=hm`, and all MPI runs retain `q5 L_inf=1.3322676295501878e-15`. S2-B0 long-step stability passes `MAXSTEP=20` for NP=1 and `NP=8 TOPOLOGY=2,2,2`; both have `q5 L_inf=3.3306690738754696e-15`, with largest statistic difference `1.2061462939527701e-12`. The optional `MAXSTEP=100` stress subset also passes for NP=1 and NP=8; both have `q5 L_inf=5.7731597280508140e-15`, with largest statistic difference `1.2216894162975223e-12`. `run_s2_hbl_oblique_shock_mpirank_matrix.sh` now captures this matrix, the 20-step long subset, and the optional 100-step stress subset via `RUN_STRESS=t`. This gate exposed and fixed a GPU `bctype=21` outlet compatibility bug: the GPU must extrapolate sound speed directly, matching CPU `extrapolate(sos(T1),sos(T2))`, rather than compute `sqrt(extrapolate(T))/Mach`. Full SBLI physics, characteristic farfield/inflow standardization, mesh/time convergence, and shock-sensor coupling remain open.
  - S2-B1 status: complete as a second compatibility gate for sustained compressed inflow, still not physical SBLI validation. `inletprofile` now supports an optional fifth profile column when the first line contains `pressure=provided`; for non-dimensional `density=provided pressure=provided`, it preserves both columns and rejects profiles whose pressure is inconsistent with `rho*T/(gamma*Mach^2)`. `generate_compressible_blasius_profile.py` can write this five-column profile and `run_s2_hbl_inlet_sustained_shock_compare.sh` sets `PROFILE_OBLIQUE_SHOCK=t`, `PROFILE_PRESSURE_MODE=provided`, and an inlet-entering shock line. This continuously injects a compressed upper inlet layer while retaining the same S2-A0 flat-plate/outlet/farfield contract. The `64x64x8` two-step smoke passes with `q5 L_inf=8.8817841970012523e-16`. The default `192x192x8` NP=1 two-step gate passes with `q5 L_inf=1.3322676295501878e-15` and maximum statistic difference `7.8381745538536052e-13`; the `NP=2 TOPOLOGY=1,2,1` y-slab gate passes with the same `q5 L_inf` and maximum statistic difference `7.8292927696566039e-13`, covering pressure-column `preadprofile` scatter. This gate establishes pressure-provided profile inflow compatibility only; it does not replace characteristic inflow/farfield, shock generation, convergence, or shock-sensor-coupled format validation.

#### Phase S0-A Sensor And Characteristic-Reconstruction GPU Execution Plan

This subsection defines the implementation contract for the explicit-upwind sensor and selective Roe characteristic-reconstruction gates. The single-rank sensor-only portion is complete as S0-A4, the first MPI raw-sensor halo gate is complete as S0-A5, and the single-rank selective Roe gate is complete as S0-A6.

Current S0-A4/A5 implementation boundary:

- `src_gpu/shock_sensor_gpu.cuf` computes the raw sensor, uses the local periodic `hm` fallback for S0-A4 or generic `exchange_field_halo_gpu(shock_sensor_d,1)` for S0-A5, expands and thresholds the mask, and explicitly synchronizes after every launched kernel.
- `ASTR_SHOCK_SENSOR_DUMP` enables the controlled validation capability and first-result CPU/GPU dump. Multi-rank runs write one rank-local file with global offsets; the validation tool merges and checks the overlapping interface plane. `lchardecomp=t` also owns the sensor storage because S0-A6 consumes its mask.
- The deterministic forced-3D Shu-Osher gate is used instead of Sod because the zero initial Sod velocity makes the Ducros compression factor identically zero and is therefore not a useful sensor oracle.
- S0-A6 uses physical-space MP7 outside the mask and Roe-characteristic MP7 inside it. It writes one five-component directional interface workspace, then applies directional flux divergence. S0-A5 does not claim MPI characteristic reconstruction.

CPU semantics to preserve:

1. `gradcal` completes the velocity-gradient tensor.
2. The raw cell sensor is computed as the Ducros compression ratio multiplied by the maximum normalized pressure curvature.
3. The haloed raw sensor `ssf` is exchanged before shock-region expansion.
4. The expansion takes an axial maximum over offsets `-hm+1:hm` in x, y, and z, then applies `shkcrt` to produce the cell mask `lshock`.
5. An interface is sensor-active when either adjacent cell is marked.
6. For explicit upwind convection, Roe characteristic projection is used only when both `lchardecomp` and the interface sensor mask are true; otherwise reconstruction remains in physical space.

The raw-sensor and expanded-mask stages cannot be collapsed into one ordinary GPU kernel without changing the dependency contract. The expansion needs raw values produced by neighboring thread blocks and, at MPI interfaces, values received from another rank. CUDA cooperative-grid synchronization or redundant tile recomputation is not the baseline because it complicates multi-rank behavior and future HIP/DCU portability.

Baseline execution sequence:

```text
complete GPU gradcal
raw_sensor_kernel
explicit device synchronization
exchange only hm layers of ssf
expand_and_threshold_sensor_kernel
explicit device synchronization
adaptive_interface_flux_kernel for one direction
explicit device synchronization
flux_difference_kernel for that direction
explicit device synchronization
repeat the reusable flux workspace for the next direction
```

The two sensor passes are accepted as dependency-driven passes. The following passes are explicitly rejected as the default design:

```text
physical_interface_flux_kernel over every interface
characteristic_override_kernel over the sensor-active interfaces
```

That overwrite design computes physical reconstruction unnecessarily in shock regions and writes selected interface fluxes twice. It may remain a profiling experiment, but it is not the implementation baseline.

The baseline `adaptive_interface_flux_kernel` performs exactly one reconstruction and one final-flux write per interface:

```text
interface_shock = shock_mask(left_cell) OR shock_mask(right_cell)

if lchardecomp AND interface_shock:
    Roe average
    project the five Euler split fluxes to characteristic space
    reconstruct in characteristic space
    project the interface flux back to conservative space
else:
    reconstruct the split fluxes in physical space
```

This kernel intentionally accepts warp divergence at mixed smooth/shock warps in the first implementation. The alternative of always using characteristic reconstruction is not valid because it changes the nonlinear numerical operator in smooth regions and adds substantial work.

Device storage contract:

- Use one haloed `real(8)` raw-sensor array, `shock_sensor_d`.
- Use one active-domain byte mask, `shock_mask_d`, represented as `integer(1)` rather than `real(8)` or default four-byte logical storage.
- Do not allocate separate x/y/z interface masks. Each directional flux kernel forms the adjacent-cell OR directly.
- Characteristic reconstruction couples all five Euler equations. If a five-component final-interface workspace is required, allocate only one reusable directional workspace and reuse it for x, y, and z; do not allocate three direction-specific copies.
- Retain the existing scalar `flux_work_d` path for S0-A1/A2/A3 while the characteristic capability is disabled.
- Do not introduce whole-field D2H/H2D transfers. Only the existing halo transport boundary may stage the `hm` sensor slabs through the host in the initial multi-rank backend.

Traversal and bandwidth controls:

- Compute and materialize the expanded cell mask once per RK substage. Do not recompute the axial neighborhood maximum independently in x-, y-, and z-flux kernels.
- Fuse thresholding and optional shock-node counting into `expand_and_threshold_sensor_kernel`; do not store a second expanded floating-point field.
- Structure sensor expansion so x-neighbor accesses remain coalesced and y/z planes are reused through cache or shared-memory tiling where profiling justifies it.
- Do not build a compacted shock-interface list in the baseline. Prefix scans, irregular index traffic, and backend-specific primitives are deferred until profiling proves that divergence dominates.
- A later optimization may fuse raw-sensor evaluation into the final velocity-gradient kernel only after proving that all nine gradient components are complete and visible at that point. This is an optimization, not a correctness dependency.
- A later multi-rank optimization may compute sensor-independent interior work while nonblocking sensor-halo transport is in flight. It must preserve the project rule that every launched kernel is explicitly synchronized and must not assume asynchronous MPI progress without measurement.

Branch-divergence fallback:

If Nsight Compute shows that the adaptive kernel loses materially more time to divergence or characteristic-path register pressure than it saves in memory traffic, evaluate two disjoint masked kernels:

```text
physical_interface_flux_kernel:
    return immediately when interface_shock is true

characteristic_interface_flux_kernel:
    return immediately when interface_shock is false
```

The two kernels must write mutually exclusive interfaces, so every interface is still reconstructed and written exactly once. This fallback is different from the rejected full-domain physical-plus-override design. Selection between the adaptive kernel and disjoint kernels is a measured backend decision and may differ between NVIDIA CUDA and future HIP/DCU builds.

Validation gates:

1. Sensor-only single-rank gate: **pass**. The forced-3D Shu-Osher case compares the complete raw field and mask, not only reductions.
2. Sensor-halo gate: **pass for `NP=2 TOPOLOGY=2,1,1`**. The Shu-Osher jump is placed beside the x-slab interface and merged CPU/GPU sensor fields plus masks are compared over the global domain.
3. Characteristic gate: **pass for `NP=1`** using controlled periodic Shu-Osher, not initial Sod. Sod has zero initial velocity and an all-zero Ducros gate, whereas Shu-Osher deterministically executes the selected Roe branch. Compare the complete sensor/mask plus statistics and `q(:,:,:,1:5)` fields.
4. Numerical-accuracy gate: retain the implemented `run_sod_phase_s0a2_accuracy.sh` Sod exact/profile checks for overshoot, undershoot, discontinuity thickness, position error, and smooth-region error. Reuse this oracle for each later reconstruction gate before claiming shock-format suitability.
5. Performance gate: report kernel time, achieved occupancy, register pressure, branch behavior, and DRAM traffic for the adaptive and disjoint-kernel variants. GPU utilization from `nvitop` is supporting evidence, not sufficient performance evidence by itself.
6. Multi-rank gate: **pass for `2x1x1`, `1x2x1`, `1x1x2`, and `2x2x2`**. The characteristic Shu-Osher jump crosses the x-slab boundary; y/z use rankwise CPU/GPU sensor and locally owned-mask comparison at duplicated active interfaces.

Tasks:

- Define the supported shock-capturing format family and its GPU data dependencies.
- Treat the single-rank and first x-slab MPI shock-sensor gates as complete; extend the same `hm`-layer raw-field transport to further decompositions only with an interface-crossing oracle.
- S0-A6 implements the adaptive selective-characteristic interface kernel. Its `(512,1,1)` x launch needs `maxregcount:128`, producing local-memory spills; profile it before considering disjoint masked kernels or a lower-register redesign.
- S0 performance baseline: Nsight Systems on the three-step single-rank S0-A6 gate attributes 96.6% of GPU kernel time to the x/y/z characteristic interface-flux kernels, while the associated RHS, sensor, and gradient kernels are each below 1%. Nsight Compute is available: before cached split fluxes the x kernel had 128 registers/thread, about 1.08M spill requests, about 14% achieved occupancy, and about 93.5% branch efficiency. The cached-split implementation improves x/y/z kernel means `1.93x/2.52x/2.74x`, reduces x spill requests to about 0.89M, and raises achieved occupancy to about 22%. Keep the 128-register cap; `maxregcount=96` is slower.
- Define shock-region filtering or added-dissipation policy separately from the existing explicit central filter.
- Add inlet/outlet/sponge/high-speed wall boundary support required by the selected validation case.
- Select a laminar or DNS-like shock/SBLI validation case with `turbmode='none'`.
- Keep compact finite-difference and compact filter solvers out of scope unless a separate decision reopens them.

Acceptance:

- Phase S0-A: the S0-A1/S0-A2/S0-A3 forced 3D Sod gates pass same-input CPU/GPU statistics and `q(:,:,:,1:5)` field comparison with the documented tolerances, the WENO7/MP7 paths pass the same finite-threshold exact-solution gate, and S0-A4/A5 pass the complete single-rank and x-slab MPI Shu-Osher raw-sensor/mask oracles.
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

1. Treat Phase S0-A1 through S0-A10 as complete controlled gates: first-order, WENO7, MP7, exact-solution, sensor, all one-axis MPI sensor/characteristic paths, and a `2x2x2` combined topology pass their documented CPU/GPU oracles.
2. Keep rankwise sensor/mask validation for duplicated y/z active interfaces and the `local_n>=hm` precondition for every decomposed direction.
3. Profiled S0-A6 branch behavior, memory traffic, and register spill. The `maxregcount=96` sensitivity build is numerically correct but slower with a larger stack; retain the `128` default and reduce live numerical state rather than tightening the compiler cap.
4. Keep S0-B open boundaries, S1 high-speed walls, MP-LD coupling, species, turbulence, chemistry, compact schemes, diffusion, and filtering outside the completed S0-A gates.
5. Keep `tests/gpu_validation/run_source_phasek_matrix.sh`, `tests/gpu_validation/run_rti_phasej_matrix.sh`, and the wall-family/channel/TGV regression drivers as the required regression set after touching capability gates, source dispatch, boundary logic, or common solver kernels.
6. Keep `documents/GPU_VALIDATION_MATRIX.md`, `tests/gpu_validation/README.md`, `CONTEXT.md`, `documents/ASTR_GPU_DEVICE_FIELD_OWNERSHIP.md`, and `documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md` synchronized with each new device field, case capability, halo semantic, and validation result.

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
