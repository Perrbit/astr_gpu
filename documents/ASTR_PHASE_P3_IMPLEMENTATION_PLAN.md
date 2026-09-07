# Phase P3 Multi-GPU HaloTransport Implementation Plan

Status: local P3 acceptance completed on 2026-09-07. L0 remains the default;
pinned blocking and periodic stored-diffusion overlap are retained opt-in modes.
Standalone nonblocking is rejected; CUDA-aware is deferred after current-stack
admission failures. Numerical, contract, host-only sanitizer, final performance
and request-state evidence gates pass within the documented local scope.
Reference: commit `566b5bccec8006c135fb2bc5f51012f42096347b`.

## Contract

Preserve FP64, RK3, explicit numerical schemes, direction-specific block sizes,
every explicit kernel synchronization, CPU orchestration, and boundary semantics.
Keep pageable blocking transport as the default reference. No new worktree.
Build through the repository's top-level CMake project.

Transport optimization does not change the data selected by pack kernels or the
order and arithmetic of unpack kernels. In particular, the fixed halo width
does not imply that every message has the same number of planes.

## Audited Exchange Inventory

| Consumer | Existing entry | Message width | Interface-plane action |
|---|---|---|---|
| Solution q | `exchange_solution_halo_gpu` | hm+1 | qswap-compatible averaging |
| Sponge q | `exchange_solution_sponge_halo_gpu` | hm+1 | same qswap kernels |
| Filter x q | `exchange_solution_filter_halo_gpu` | hm | halo only |
| Filter y qwork | `exchange_filter_y_work_halo_gpu` | hm | halo only |
| Filter z q | `exchange_filter_z_halo_gpu` | hm | halo only |
| Viscous stress | `exchange_field_halo_gpu(sigma_d,6)` | hm | halo only |
| Heat flux | `exchange_field_halo_gpu(qflux_d,3)` | hm | halo only |
| Shock sensor | `exchange_field_halo_gpu(shock_sensor_d,1)` | hm | halo only |

The generic field buffers have capacity for six variables; transmit only the
requested contiguous leading component range. Sensor MPI transport is already
shared with diffusion. Its single-rank periodic refresh remains a local kernel.

All nine axis-specific exchange callers live in
`src_gpu/halo_exchange_gpu.cuf`; paired MPI operations now live in
`src_gpu/halo_transport_gpu.cuf`. Each exchange has two directed messages. Tags are
21001/21002 for x, 21003/21004 for y, and 21005/21006 for z. Neighbors come from
`parallel`; physical endpoints use MPI_PROC_NULL. Preserve x/y/z ordering.
Do not read or upload a receive buffer for an absent neighbor.
All 18 receive uploads now check the matching neighbor before reading the host
buffer. MPI calls, widths, interface averaging and kernel synchronization are
unchanged. A source contract test first failed for all 18 missing guards and
then passed. It supplements, rather than replaces, actual MPI/CFD tests.

## Approved Design A

Keep pack/unpack, local periodic refresh, boundary decisions, and primitive
refresh in `halo_exchange_gpu.cuf`. Introduce `src_gpu/halo_transport_gpu.cuf`
for host-memory registration, transfer selection, and paired MPI operations.
Callers retain their current externally visible interfaces.

The runtime setting is `ASTR_GPU_HALO_TRANSPORT`. The current executable uses
`pageable` (default, blocking), `pinned` (opt-in, blocking) and
`pinned-overlap` (opt-in, periodic stored-diffusion core only).
The standalone `pinned-nonblocking` experiment was rejected after screening and
removed from active source; its executable/source evidence remains archived.
CUDA-aware is not selectable. The two opt-in host-staged modes have local admission.
Reject unknown values. Agree on the selected backend across ranks before use.

Prefer registering existing host allocations for pinned experiments, subject
to a CUDA Fortran compiler/runtime probe. Record successful registration bytes
and release registrations before freeing buffers. Do not allocate duplicate
full-field storage. A failed registration must be reported, with an explicitly
logged collective fallback before transport starts; never label pageable
measurements as pinned.

For paired nonblocking messages, post both receives, post both sends, and wait
for all four requests before consuming or reusing buffers. Preserve tags and
source/destination mapping, including two directions sharing one periodic peer.
Keep one exchange transaction outstanding initially: current tags and shared
field buffers are not safe for concurrent unrelated field exchanges.

CUDA-aware capability checks must inspect the MPI installation actually linked
by ASTR. The system `/usr/bin/ompi_info` is not the HPC-X installation used by
NVHPC. Build capability alone does not prove device-buffer communication works.
Use an isolated device-buffer probe before enabling the backend in a solver.
Fallback is allowed before communicating solver state. A failed or hung MPI
operation during a solver run is fatal; do not retry it silently with L0.

Overlap requires a separate dependency audit and measured MPI progress. Finish
each launched kernel with explicit synchronization. Never reuse a send buffer
or access a receive buffer while its request is outstanding. Preserve the
original per-cell arithmetic and boundary update order.

### Interior/Halo Dependency Boundary

`mainloop_gpu.cuf` currently finishes convective RHS, computes diffusion fluxes,
calls `exchange_diffusion_flux_halo_gpu`, then adds diffusion RHS in x/y/z order.
The halo helper completes the six-component sigma exchange before starting the
three-component heat-flux exchange, both through shared generic field buffers.
An outstanding sigma request therefore prevents reuse of those buffers for
heat flux. Interior work must not require either remote field value.

The periodic stored diffusion RHS uses offsets +/-1, +/-2 and +/-3. The
implemented interior/boundary split preserves each cell's x/y/z addition order and
write each contribution once. Physical-boundary variants have dedicated
closures and index ranges and cannot be treated as periodic simply by slicing
the launch grid. The implemented overlap excludes those physical-boundary paths.

The expanded shock sensor reads offsets -hm+1 through hm on each coordinate
line, with physical-face clamping. Its safe interior mask is not the diffusion
radius-three mask. The measured sensor-only fraction cannot bound all diffusion
or solution exchange opportunities. MPI progress while GPU kernels execute is
now evidenced by actual Testall flags in the profiling-only trace. Issuing
nonblocking requests alone would not prove it.

## Execution Gates

### Overlap Prototype Contract

The standalone progress probe now demonstrates a viable prerequisite: actively
progressing MPI_Testall during GPU work, then retaining the explicit post-kernel
cudaDeviceSynchronize. Waiting only in CUDA does not reproduce that progress
benefit on the tested platform. Probe speedups do not satisfy solver acceptance.

The first solver trial is implemented only for the fully periodic stored-diffusion
path. Split the owned cells into the radius-three core
`3 <= i <= im-3`, `3 <= j <= jm-3`, `3 <= k <= km-3`, and its complement.
Require nonempty core and distributed topology. Other paths retain blocking
transport. The core reads no remote sigma/qflux values; each cell retains x,
then y, then z accumulation into qrhs. Compute the core exactly once during the
first distributed sigma exchange, then the complement after all sigma/qflux
halo exchanges finish. Do not average, shorten, reorder or reuse live buffers.

The transport owns the four requests and host buffer lifetime through completion.
An optional useful-work callback may launch core kernels and explicitly progress
the owned requests before each existing kernel synchronization. No outstanding
request may survive the paired exchange routine; reject nested use. Retain
ASYNCHRONOUS buffer semantics, existing tags and collective registration fallback.
This is distinct from the rejected nonblocking-with-immediate-Waitall candidate.

Do not apply the periodic split to physical closures or the expanded shock sensor.
After split/complement coverage tests, require TGV NP=1/2 slabs/8 field and
statistics agreement, fallback tests for Shu-Osher/SBLI, sanitizer and a solver
NSYS trace. Only then run the controlled nine-case complete-RK screening against
pinned blocking. Reject and archive the prototype if launch/polling overhead
prevents the independent 3% gate. CUDA-aware screening remains a separate tier.

### Checklist

- [x] User approved design A.
- [x] Identify nine MPI paths, three transport families, and shared sensor path.
- [x] Freeze baseline executable hash, source revision, linked MPI/HDF5, GPU
  topology, clocks, environment, and input/controller settings.
  The baseline/source patch and final solver archive are retained. Current
  environment and observed clocks are recorded; clocks were not locked and
  missing historical samples are not reconstructed. Final input hashes agree.
- [x] Collect five-repeat L0 complete-RK baselines for TGV 256^3, Shu-Osher
  256x64x32, and SBLI 256x192x32 with NP=2 x/y/z slabs.
- [x] Collect separate NSYS CUDA/MPI traces, per-GPU peak memory and utilization.
- [x] Introduce and validate the L0 transport interface before changing memory
  or MPI behavior; compare against the frozen executable.
  Initial interface-only and final CFD matrices pass. The final direct TGV
  comparison against frozen pre-P3 GPU L0 has zero field difference.
- [x] Screen pinned blocking independently.
  Prerequisite registration/copy/release and exact-value pinned MPI tests now
  pass. Nine five-repeat screening groups and NP=1/2/8 CFD comparisons pass;
  the post-cleanup CFD and host-only solver sanitizer matrices also pass.
  Final performance reconciliation passes: nine valid groups, 2.829%--23.728%
  RK reductions, maximum per-device memory growth 3.436%. Default UCX memcheck reports
  context errors in MPI_Init; host-only results do not qualify that stack.
- [x] Screen pinned nonblocking independently.
  Rejected as a standalone change: nine five-repeat groups and same-binary
  controls yield at most 2.425% reproducible incremental time reduction.
  The branch was removed and evidence archived. This is not an overlap decision.
- [x] Audit interior/halo dependencies and screen overlap when its measured
  upper bound warrants it; otherwise document the bound.
  The periodic diffusion prototype is integrated as `pinned-overlap`. Ten-step
  TGV NP=1/2 x/y/z/8 field/statistics, NP=1/2/3 exact callback transport and a
  small z-slab solver memcheck pass. Shu-Osher x and SBLI z fallback comparisons
  pass. Actual solver Testall/kernel intersections are recorded. Full retained
  backend fallback coverage and final performance gates now pass.
  Same-binary TGV x/y/z five-repeat screens improve 2.240772/3.422865/3.261797%,
  with all spreads below 2%. The nine-case screen is complete. Shu-Osher x has
  an initial 1.334% adverse result, one noisy repeat and a valid reversed-order
  repeat without regression; all groups are retained. The candidate proceeded
  to final validation at that checkpoint. The improved y case also has a solver
  trace distinguishing request-record Testall calls from null-request polling.
  Final-source valid x/y/z increments are 2.616/3.085/2.944%, with actual
  completion-state evidence below. Retain as opt-in for the audited core only.
- [x] Probe CUDA-aware MPI and record current-stack admission limits.
  Small exact-value NP=1/2/3 tests pass, but default UCX large-message tests
  leave receive payloads unchanged. Disabling IPC passes payload checks but
  fails the strict zero-error sanitizer gate. Early device binding does not
  resolve either gate. No solver backend was added and no performance claim
  is made. This tier is deferred on the tested stack, not declared universally
  unsupported; see the baseline report and frozen diagnostic probe.
- [x] Run retained-backend correctness and sanitizer matrix.
  After endpoint cleanup, all 45 ten-step TGV/Shu/SBLI comparisons pass across
  three backends and five topologies. Nine additional wall41 physical-endpoint
  filter/diffusion checks pass at 1e-10 combined tolerance. Eighteen small
  solver memchecks pass with 36 zero-error rank logs under host-only MPI.
  The separate production halo executable now passes all three backends at
  NP=2/3, with deliberately conflicting shared planes, full-array ownership
  checks, three axes, periodic peers and physical chains, and 1/3/6 components.
  Its NP=3 host-only memchecks add nine zero-error rank logs. No pack/unpack
  kernel is duplicated in the fixture.
- [x] Reconcile formal performance acceptance with the final source hash after
  endpoint cleanup. Earlier screening numbers are not new post-cleanup timings.
  Final-source TGV x groups are recorded: pageable/pinned have spreads
  4.152/1.401%, and pinned reduces complete-RK time by 13.839%. Overlap has
  spread 8.020%, so that entire group remains inconclusive and required full
  paired repeats, rather than removal of individual slow processes.
  Shu-Osher/SBLI final-source x/y/z groups are now complete. The noisy Shu z
  pinned group is retained; a complete reverse-order three-backend repeat
  passes the spread gate. Per-device memory growth is at most 3.436%.
  TGV y/z now pass. TGV x repeat1 also exceeded the spread gate (5.125%) and
  remains archived; the entire repeat2 pair passes (1.394/1.170% spreads).
  All 34 groups and 204 process logs are audited, without trimming any run.
- [x] Strengthen overlap proof with request-completion state aligned to kernels.
  A profiling-only PMPI wrapper records the actual Testall flag without extra
  polling. Its controlled NP=2 probe passes, including exclusion of null-request
  polls. In final-source TGV 256^3 y-slab, 4237 pending markers and 24 completion
  markers occur during stored diffusion kernels on the same process. This
  establishes in-flight request/computation overlap in the profiled execution,
  not byte throughput or uninstrumented speedup. Formal timing excludes the wrapper.
- [x] Update baseline report, overall optimization plan, and validation README.
  Top-level CPU/GPU builds, optional probe builds, 68 Python tests and 96 shell
  syntax checks pass. The rebuilt solver and profiling wrapper add zero field
  difference in the small frozen-L0 reference comparison.

## Measurement And Acceptance

Use one rank per physical GPU for NP=2 performance. NP=8 2x2x2 sharing two GPUs
is correctness evidence only. Keep identical retained RK sample counts, warmup,
output settings, inputs, and monitoring across each five-repeat A/B pair.
Use the slowest-rank complete-RK time, not a sum of rank times. Profile separately
from uninstrumented performance runs. Record the host pinned-memory footprint
in addition to per-device peak memory.

A performance candidate must improve at least one formal benchmark by 3% or
more, with spread at most 5%, no stable regression above 1% in other formal
benchmarks, and no device-memory increase above 5%. A noisy run is inconclusive.
Overlap requires timeline evidence of concurrent communication and useful work.
Reject unsuccessful candidates and retain their reports and reproducible diffs.

Correctness includes top-level CPU/GPU builds, Python tests, each shell script
checked individually with bash -n, NP=1 and NP=2 x/y/z TGV/Shu-Osher/SBLI field
and statistics comparisons at existing 1e-10 combined tolerance, NP=8 2x2x2,
filter/diffusion/sensor halo checks, SBLI wall diagnostics, and Compute Sanitizer
zero errors. Add exact-value halo tests for hm versus hm+1, interface averaging,
MPI_PROC_NULL, periodic duplicate peers, and field component counts 1/3/6.
Extend topology checks to an interior rank before claiming that case covered.

Do not change CPU numerical bugs without user decision, relax tolerances, or
claim a backend complete from source-string tests alone. Completion requires
a supported, rejected, or evidence-based deferred decision for every stage,
and a passing L0 fallback matrix.
