# IS8-A Device Component Progress

Date: 2026-10-06. Branch: `feature/gpu_dev`. Approved bounded IS8-A0-A9 completed.
Current admission and final evidence: `ASTR_INSITU_IS8_ACCEPTANCE.md`.
Sections below retain dated component/failure history; their pending statements
do not supersede the latest acceptance record.
The independent CUDA/Viskores dependency and private VTK RK45 route were approved.
The solver and the validated ParaView installation have not been replaced.

## Verified Components

| Component | Result | Evidence |
|---|---|---|
| CUDA FP64 compiler compatibility | Maximum difference `1.1102230246251565e-16` | `insitu_device_compiler_probe` |
| Private adaptive Cash-Karp RK45 versus original VTK | 26 finite cases match exactly; 6 nonfinite cases rejected | `insitu_device_rk45_probe` |
| Borrowed CUDA velocity, Viskores grid interpolation and RK45 | 32 seed/field pairs; same-arclength maximum difference `1.3617407196422989e-18`; input host mirror absent | `insitu_viskores_rk45_probe` |
| Component regressions | 3 passed, no skips | `out/insitu_is8_device_bootstrap_20261005/pytest.xml` |
| Component memory checks | Three clean reports | `compiler_memcheck.log`, `rk45_memcheck.log`, `grid_memcheck_pt2pt.log` in the same directory |
| Actual device dispatch | Two CUDA field-generation and two `RK45StepWorklet` launches recorded | `grid_cuda_trace.nsys-rep` and `.sqlite` in the same directory |

The affine analytic field and its interpolated representation can produce zero
versus near-zero error estimates through rounding. VTK's unchanged exact-zero
branch then selects different adaptive steps. Three such differences remain
visible in the diagnostic output; the largest accepted-length difference is
`0.078539816339744828`. The grid comparison evaluates the analytic field at the
same accepted length, rather than comparing positions at unequal arc locations.
The separate differential test retains the original adaptive VTK comparison.
The approved constant-field cross-rank endpoint threshold remains unchanged.
Its synthetic distributed-grid component now passes; solver-field integration is
still pending.

The synthetic input is a `33*33*33` array of FP64 velocity vectors allocated by
`cudaMalloc` (862488 bytes) and generated on the GPU. It is borrowed using
`src_gpu/insitu_device_array.h`; the owner retains it until all consumers finish.
The input's host-allocation flag is checked before and after interpolation and
particle-result reads. No full input field is read through a host portal.
Nsight records 384 bytes of ordinary H2D seed upload, 65536 bytes of unified H2D
migration, and 131072 bytes of unified D2H migration. Thus the component is not
claimed to have zero total D2H traffic. The unified-memory records include page
granularity; host reads are limited to the 16 particle results per field.
The evidence directory is below 1 MiB, within the 256 MiB case limit.

## Dependency And Runtime Conditions

The separate build enables CUDA, FP64 and MPI through the default-off
`ASTR_VISKORES_FP64_MPI` dependency patch. `viskores_cont`, `viskores_worklet`
and their required libraries were built. The full ParaView installation is not
complete. The unused RK4/Euler `viskores_filter_flow` build was stopped after
the necessary worklet library became available; the private RK45 probe does
not link that flow library. Python 3.14 interpreter, headers and library are
pinned together in the separate configuration.

The initial sanitizer report `grid_memcheck.log` contains one CUDA invalid-context
error in UCX's MPI RMA component during MPI initialization, before the worklet.
It is retained, not discarded. This single-rank component needs MPI initialization
only, so its launcher explicitly uses Open MPI `ob1`, `self,tcp`, and `osc=pt2pt`.
Under that declared configuration the component memcheck is clean. This is not
a CUDA-aware MPI halo or distributed-particle acceptance result. Direct singleton
startup in the relocated HPC-X package also failed/hung; use the matching explicit
`mpirun --prefix` rather than relying on its original package prefix.

The root CMake CPU, CUDA and AIR5 CUDA `astr` targets built successfully during
this stage. The optional probes default to OFF; no runtime device product has
yet been enabled, and no Git write or remote-job operation was performed.

## Approved Accepted-Length Correction

VTK 9.6.2 `Filters/FlowPaths/vtkStreamTracer.cxx` adds
`abs(stepSize.Interval)` to `propagation` after `ComputeNextStep`. Adaptive
RK45 has already replaced that argument with the next suggested step. The
accepted length is returned separately in `stepTaken`.

`insitu_vtk_length_audit` demonstrates the difference with the approved
initial/minimum/maximum step and `1e-8` error control. A circular tangent of
radius 0.2 gives accepted length `0.019634954084936207`, next suggestion
`0.050634378597087477`. Counting the latter misrepresents travelled arc length
and can terminate the streamline early. This is an upstream postprocessing
counter issue, not a CPU ASTR time-integrator defect.

The user approved counting `abs(stepTaken)` in the new dependency's host reference
and new device trajectory driver. The old installation, RK45 algorithm, step
bounds, tolerance, seeds and cameras are preserved. The reproducible
`paraview-6.1.1-streamline-accepted-length.patch` applies only to the independent
source tree. A private root-CMake reference executable compiles the changed VTK
translation unit against existing module dependencies; no library was installed.

The original circular-field control fails the accepted-length check, with arc
deficit `0.030603952001448498`. The corrected reference reduces this deficit to
`9.3768870446098163e-8`, below the existing minimum step. Both constant-field
endpoints differ by `1.7763568394002505e-15`. The circle is a termination
diagnostic, not a new `2e-10` curved-trajectory physical gate.

`insitu_device_trace_probe` exercises 16 particles, constant/circular tangents,
actual-step accumulation and bounded continuation. Constant endpoint difference
is `1.7763568394002505e-15`; accepted-length counter difference and circular
counter deficit are zero. Eight-step save/resume and simulated ownership
handoff are individually exact. `trace_memcheck.log` reports zero errors;
`bootstrap_pytest.xml` records four passing regressions. These are not yet
distributed mesh, solver-state restart or complete rendering acceptance results.

## Initial Remaining IS8-A Work (Historical, 2026-10-05)

Private device sampling/shared-node ownership and halos; instantaneous and mean
velocity supply; slice and Q-surface device extraction; MPI particle continuation;
final-geometry-only EGL rendering; actual JPEG/EPS/VTK products; completed-step,
output-isolation, exact-restart, resource/rejection and matched timing gates.
CURVE, wall, AIR5 and production-scale device visualization are outside this goal.

## Distributed Trajectory Component

`insitu_viskores_trace_probe` generates local uniform-grid velocity on each GPU,
with three synthetic interpolation halo cells. It dispatches the bounded
`RK45TraceWorklet` and communicates only 16 continuation states. Extracted
segments are read for endpoint/straightness/seam checks. This is not a test of
solver halo generation, bidirectional physical TGV trajectories or rendering.

NP=1/2 regressions pass. NP=2 gives endpoint/straightness maximum difference
`1.7763568394002505e-15`, seam difference zero, 16 ownership transfers, 544
segment vertices including repeated seam vertices, and two dispatch rounds.
The input field never acquires a host mirror. Four actual trace-worklet launches
are recorded across two devices. `distributed_host_control_pytest.xml` records
three passing tests, including the previous interpolation component.

The initial two-rank sanitizer found 23 API errors per rank in MPI's optional
UCC/UCX initialization and host-buffer CUDA classification, not device-kernel
out-of-bounds accesses. Those failed logs remain as `trace_mpi_memcheck_*.log`.
For this host-particle-state-only component, the declared launcher disables
HCOLL/UCC and `opal_cuda_support`, with `ob1`, `self,tcp`, `osc=pt2pt`.
Both `trace_mpi_host_control_memcheck_*.log` reports are clean. This does not
qualify device-buffer MPI. The matching trace is
`distributed_host_control_trace.nsys-rep`; earlier `distributed_trace` retains
the original MPI configuration. No error suppression was added for this check.

## Historical Velocity Snapshot And Pre-Reboot P2P Failure

`src_gpu/insitu_device_velocity.cuf` implements a read-only CUDA Fortran snapshot:
copy density/momentum into private device storage, reconcile periodic upper
endpoints in x/y/z order, then construct interleaved FP64 velocity. The caller's
solver state is not modified. Parallel faces require queried CUDA-aware MPI,
use a private communicator and have no coded host-buffer fallback. Resource,
index, finite-density and finite-velocity checks fail explicitly. This module
was then linked only by its root-CMake component probe, not the solver.

`insitu_device_velocity_probe` uses a 32-cell-per-axis synthetic field with
deliberately inconsistent duplicate endpoints and checks all source values for
immutability on GPU. NP=1 passes exactly. NP=2 x fails with CUDA IPC enabled:
packed face values are nonzero but received face values remain zero, then the
canonical-state check aborts. Nonblocking calls and a receive synchronization
do not fix it. Disabling IPC passes exactly as a diagnostic only; that transport
may stage through host memory. That diagnostic does not qualify either backend
under the revised contract below: it lacks explicit staging, budget and transfer
accounting evidence.

The old `halo_cuda_aware_probe` also fails with the same IPC configuration at
33x33 tangential buffer size, despite passing at 3x2. Thus the observation is
not confined to the new snapshot implementation. The failed trace
`velocity_failure_trace.nsys-rep` records two 34848-byte peer copies. Disabling
the new UCX protocol selection or pinning the SDK UCX library does not resolve it.

The MPI-independent root-CMake `insitu_device_peer_probe` isolates direct CUDA
peer copies of 4356 doubles (34848 bytes) in both directions. Source-generation
controls have zero mismatches, but both peer transfers return CUDA success with
4356 mismatches. In the recorded run the first destination values were NaN and
the unchanged -999 sentinel instead of 101. `nvidia-smi topo -p2p r` nevertheless
reports OK. Recorded environment: kernel 7.0.0-31-generic, driver 595.91.07,
two RTX 4000 Ada cards; both PCI devices' IOMMU group types are `DMA-FQ`.

NVIDIA states that Linux bare-metal PCIe peer transfer is unsupported with IOMMU
enabled and warns of silent device-memory corruption:
[CUDA multi-GPU guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html),
[NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html).
IOMMU is the leading environment hypothesis, not a verified causal fix until
the system setting is changed and the same direct-copy/MPI controls pass.
No BIOS, boot parameters, driver, remote job or Git state was modified by the agent.

This restriction applied to the recorded pre-reboot environment. Hardware
reconfiguration/reboot needed human action. When that
route is revisited, rerun direct peer, the larger old MPI payload control, and
the private snapshot (NP=1/2 x/y/z), followed by memory/transfer checks.
Historical results remain historical evidence, not qualification of this current
kernel/driver/IOMMU combination. Full IS8-A remains unaccepted.

## Approved Transport Contract Revision (2026-10-05)

The user approved two explicitly selected backends while retaining IOMMU:
device-buffer MPI, and fixed pinned-host face-buffer MPI. They share GPU packing,
ownership completion, unpacking and numerical checks. No silent fallback or
visualization-only full-volume host mirror/canonicalization/reupload is allowed.
The second backend permits only halo/shared-node face D2H/H2D in addition to
final geometry/images and small state. It is not the old host-field pipeline.

This supersedes only the original goal's absolute prohibition on intermediate
field D2H. FP64, 32-cubed periodic TGV, NP=1/2, all products, tolerances, exact
restart, completed-step phase, safety and existing resource budgets remain.
Pinned slots must be bounded by maximum face size, reused, and charged to the
4 GiB/node additional host budget. Record selected mode, rank consensus, payload
bytes, capacities, copies, MPI time and observed library/managed-memory transfers.
Device-buffer MPI alone does not prove NVLink or physical GPU-direct transport.

At the time of this contract revision, the immediate next gate was implementing and qualifying the pinned face route
with NP=1/2 x/y/z snapshots, source immutability, sanitizer, budgets and actual
copy traces. At that point the snapshot module implemented device-buffer MPI
only; the new backend had not been implemented or accepted. Local IPC was
unqualified. Each backend has its own acceptance status; product acceptance on
one must not be reported as qualification of both. No system or remote changes
were prerequisites for continuing the pinned route.

## Reboot And Fixed-Slot Transport Progress

The user rebooted to kernel `7.0.0-34-generic` with
`iommu.passthrough=1`. Both GPU domains report `identity`; driver remains
595.91.07. Direct CUDA peer copies of 34848 bytes in both directions, the old
larger MPI payload probe, and private snapshots now pass. Kernel and domain
changed together, so this is not a single-variable causal test. The old failed
logs above are preserved.

`processing_backend=host/device` and independent explicit
`postprocess_transport=device-aware/pinned` are parsed and checked collectively.
75 configuration tests pass. Host remains the default; specifying a transport
for host, omitting a device transport, invalid values, inconsistent ranks and
unavailable budgets reject. The native device product entry remains guarded
until its extraction/rendering bridge is implemented. It never silently calls
the existing host-volume path.

The private transport now has a duplicated communicator, fixed device slots,
explicitly registered pinned slots for decomposed axes, allocation generations,
copy-byte counters, reuse and release checks. Snapshot, halo and diagnostic
tests cover NP=1 and NP=2 x/y/z for both modes. The 24-test matrix passes,
including invalid configuration/lifecycle/budget rejection and unchanged source
arrays. Maximum component derivative/Q difference is
`1.1102230246251565e-15`. Three periodic halo layers include edges and corners.
Pinned slot admission also sums allocations over the MPI shared-memory node;
two locally affordable slots that exceed the node budget are rejected before
allocation. `node_budget.xml` records that added rejection and the updated matrix.
Unused local axes may have `MPI_PROC_NULL`, as in the CPU topology; they execute
only local device copies and do not require an MPI neighbor.

The bounded maximum device double-slot allocation is 219024 bytes per rank.
Pinned mode allocates the same registered host capacity for NP=2, none for
NP=1; device-aware mode has no explicit host face slots. Each owned snapshot
face payload is 34848 bytes. An earlier snapshot-only Nsight capture records
six actual P2P transfers, total 209088 bytes across two ranks/three frames;
the paired pinned capture records six face D2H and six H2D transfers. These
traces predate the extended halo slot and mean supply and do not qualify all
final products or total transfer traffic. Small scalar copies are retained.

The root GPU solver now exposes a bounded read-only sampler from actual `q_d`
and `dxi_d`. It constructs velocity, private periodic halos and the same 14 FP64
diagnostics without a host field source. The shared derivative was factored
into `output_derivatives_gpu.cuf` without changing coefficients or physical
boundary branches. Statistical mean vectors copy the existing Reynolds/Favre
slots from the authoritative device accumulator, complete private endpoints,
and return device Vec3 buffers. Coverage is explicit; no instantaneous substitute
is returned for uncovered means.

Sixteen actual two-step cases cover instantaneous/mean supply, both modes and
NP=1/2 x/y/z. All diagnostic and mean differences from the existing controlled
reference are zero; all final state/cache and statistical HDF5 datasets are
exactly unchanged by enabling the diagnostic. This test-only oracle deliberately
downloads results, so it is not a no-volume-transfer product acceptance.
Evidence: `out/insitu_is8_transport_20261005/config.xml`,
`canonical_fields.xml`, `solver_sampling_v2.xml`, `solver_means.xml`.

Separate NP=2 pinned/device halo-Q component memchecks are clean. The first
actual mean-supply device-aware memcheck completed numerical checks but reported
UCX `cuCtxSetFlags/cuCtxGetDevice` invalid-context calls during MPI initialization.
The solver's host-halo selection did not prebind CUDA, while the independent
diagnostic requested CUDA-aware MPI. The pre-MPI prebind correction passes
both NP=2 actual mean/zero-coverage memchecks (four rank reports, zero errors).
No broad suppression or solver transport change was used. Formal independent
device configuration also requests prebinding before MPI; full collective
configuration validation still occurs later. Evidence:
`solver_means_memory_v2.xml` and `out/insitu_is8_solver_means_memory_20261005_v2/`.
Failed reports and the first unused-neighbor rejection remain available.

Real solver slice/contour consumption, real bidirectional x/y/z trajectories, final-geometry EGL
bridge, transport-aware exact restart and full joint resource/transfer/timing
acceptance remain pending. These component passes do not close IS8.

## Three-Axis Bidirectional Trajectory Component

`RK45TraceWorklet` now uses three-dimensional lower/upper ownership bounds.
The original x-only constructor remains equivalent. Synthetic constant fields
cover NP=1 and NP=2 x/y/z, with 16 forward and 16 reverse trajectories crossing
the selected partition. Six continuation tests pass, including the previous
forward-only x controls. Endpoint/straightness and seam differences remain
below `2e-10`; each two-rank bidirectional case requires 32 ownership transfers.
The RK45 coefficients, step bounds and error control are unchanged.
Evidence: `trace_axes.xml`. This remains a synthetic field test, not actual
TGV/mean trajectories or the product rendering gate.
The NP=2 z bidirectional memcheck has two clean rank reports; its logged
endpoint/straightness maximum is `2.3314683517128287e-15`, seam error zero.
`trace_axes_memcheck.*.log` records that safety check.

## Device Slice And Flying Edges Geometry Component

Root CMake now builds the original private Viskores `ContourFlyingEdges.cxx`
directly when `ASTR_INSITU_FLYING_EDGES_SOURCE` is explicitly set. Underlying
cont/worklet/filter_core/filter_vector_analysis modules are built from the
separate FP64/CUDA source. The unrelated unified contour filters need not be
built. `build_device_paraview.sh` supports an explicit `PV_DEVICE_TARGETS` list
only with `PV_DEVICE_INSTALL=0`; defaults still perform a full installation.
The old validated dependency installation is unchanged.

`insitu_device_geometry.h` borrows CUDA Vec3 velocity and Vec14 diagnostics,
forces CUDA, selects the approved z-node 4 plane, and extracts Q_rs=0.25 using
the original Flying Edges algorithm. No input host portal is used. Only final
product coordinates/fields/connectivity may be read for the compact bridge.

The synthetic TGV component covers NP=1 and NP=2 x/y/z, nonempty and empty
contours, and the empty-rank slice case. Two pytest items run eight launches.
The independent CPU oracle interpolates at final product coordinates, rather
than downloading input volumes. Maximum field/isovalue error is
`1.27675647831893e-15`; each topology has 1024 slice quads and 8944 contour
triangles. Summed contour area agrees within `2e-10` across partitions.
VTK coordinate, field and connectivity write/read comparisons are exact.
Generated source arrays are unchanged. NP=2 y memcheck reports zero errors
for both rank processes. `geometry_roundtrip.xml` and
`geometry_memcheck.*.log` record these checks.

`geometry_device_mpi.nsys-rep` confirms CUDA IndexPlane, SelectQAndU, Flying
Edges passes and field mapping. It also records approximately 1.1 MB of managed
device-to-host page migration, plus small scalar/metadata copies. This is not
zero-transfer evidence. The later `geometry_ranges.nsys-rep/.sqlite` adds
separate NVTX extraction and compact-output read ranges. Extraction has no
managed D2H migration and exactly three scalar D2H copies of 4, 4 and 8 bytes.
All 1114112 bytes of managed D2H pages occur in the final compact-geometry read
range. Input arrays remain device-only. `test_insitu_device_geometry_trace.py`
checks those operator/range receipts read-only; it is not a whole-solver trace.
Final actual-solver product attribution remains pending. An earlier direct singleton-MPI profiler launch
waited during initialization before CUDA work; that diagnostic was stopped and
replaced by an explicit matching `mpirun` launcher. Its failed trace is retained,
not counted as a performance sample.

This does not qualify the real solver geometry consumer, EGL image products,
native resource lifecycle, exact render restart or complete IS8-A5/A9.

## Native Consumer Integration And Terminal-Step Decision

The optional root build `ASTR_WITH_INSITU_DEVICE=ON` now links the private
Flying Edges implementation and device-product consumer. The default remains
OFF. Device rendering is orchestrated by `src_gpu/insitu_products_gpu.cuf`,
above both sampler and statistics modules, avoiding the dependency cycle
sampler -> statistics -> fields -> sampler. CUDA Fortran C addresses use
`TYPE(C_DEVPTR)` at the C bridge. Volume arrays remain on the device; the
adapter receives only final compact meshes. Static slice coordinates are
cached by dimensions, partition offset and CUDA device. Mean trajectories
carry their integrating Reynolds/Favre velocity separately from instantaneous
u/v/w sampled for other product fields.

Compact metadata uses Blueprint `state/fields`. An initial prototype put
numeric objects in Catalyst `state/parameters`, whose children ParaView 6.1.1
expects to be strings. That failed launch is retained in
`insitu_is8_products_smoke_20261005`; the schema was corrected without adding
a Catalyst Python dependency or downloading volume fields.

`compact_trace_latest.xml` records twelve passing synthetic driver checks:
NP=1 and NP=2 x/y/z, interpolated TGV, bidirectional constant fields and the
approved forward constant-field oracle. The driver uses a duplicated private
communicator. Valid empty-rank output is accepted while the global trajectory
must remain nonempty. This is not an actual mean-field/restart qualification.

The root CUDA+Catalyst+device `astr` build passes. The actual 32-cubed,
NP=1 pinned-face run produces all six products at completed step 1: slice,
Q surface, instantaneous and diagnostic constant-field streamlines, and
Reynolds/Favre streamlines. Six JPEG/EPS pairs and VTK products exist.
Constant-field endpoint/straightness error is `1.7763568394002505e-15`;
mean coverage is [0.0005,0.001] with duration 0.0005. Images and complete
lifecycle are not yet accepted. At step 2 the strict driver aborts with
`status=3 remaining=0.001779`; this failed test is retained as
`products_smoke_v2.xml` and its case directory is approximately 28 MiB,
below the 256 MiB case limit.

The remaining arc is below the approved minimum step
`0.01*(2*pi/32)=0.0019634954084936207`. The unmodified VTK
`vtkStreamTracer` reduces the requested maximum step to remaining propagation;
`vtkRungeKutta45` then returns `UNEXPECTED_VALUE` if minimum exceeds maximum.
The private adaptation preserves that rule, and the new strict driver treats
the return as fatal. This is not evidence of a field-transfer or MPI error.
The earlier circular-field reference documents a sub-minimum residual but
does not by itself approve its acceptance for every actual product.

Pending human decision: preserve VTK's last accepted vertex and explicitly
record terminal reason plus untravelled arc, admitting only this identified
sub-minimum condition; or retain fatal rejection. No minimum-step relaxation,
terminal-point extrapolation, relaxed analytic threshold or downstream full
matrix has been applied. A7 remains partial; A8 exact restart and transport-only
override, and A9 safety/resources/transfer/timing closure, remain outstanding.
The goal stays active; no Git write or remote-job operation was performed.

### Approved Terminal Semantics (2026-10-06)

The user selected A: retain VTK's final accepted vertex, raw terminal status and
untravelled arc for the identified sub-minimum terminal request. The new
classifier requires status 3, positive remainder strictly below the unchanged
minimum, an exactly remainder-sized request, and VTK's pre-step error sentinel.
Other errors remain fatal. No short-step integration or endpoint extrapolation
was introduced. Native streamline geometry carries `termination_status` and
`untravelled_arc_length`; per-seed logs name the sub-minimum reason. Targeted
tests also require the last accepted coordinate, length and step count to
remain unchanged. Four targeted terminal/RK45/reference probes passed.

### Native Products And Transport-Aware Restart (2026-10-06)

Eight actual two-step cases cover pinned/device-aware and NP=1 plus NP=2
x/y/z. Six product kinds produce paired JPEG/EPS and independently readable
VTK geometry. Render-off controls have exactly equal state/cache/statistics
datasets. Native flow-volume download counts remain zero. Independent final
geometry interpolation from checkpoint fields has maximum difference
`2.1649348980190553e-15`, including Q and both integrating mean velocities.
The initial offline oracle mistakenly omitted three checkpoint metadata fields
before device state slots; the corrected index begins at q0010, not q0007.
That failure was a test-reader offset, not a solver/mean-supply discrepancy.
Receipts: `products_actual.xml` and `products_independent_fields.xml` under
`out/insitu_is8_transport_20261005/`.

Six NP=2 x/y/z cases independently exercise each transport, continuous four
steps versus checkpoint step2+2, and explicit switching in both directions.
Same-backend state/cache/statistics/control, final VTK bytes and JPEG pixels
are equal. An unapproved transport switch is rejected. A transport-only override
keeps statistics and the step3/4 schedule; saved transport identity changes,
not clocks or field state. Source checkpoint files remain unchanged.
Device render control uses ASTRIR02 (345 bytes), with explicit processor and
transport identity; the old host ASTRIR01 remains unchanged.
Receipt: `products_restart.xml`, six passed.

Initial uncovered mean channels stay valid empty products, never instantaneous
substitutes. Two actual NP=2 y all-product memchecks have four clean rank logs;
receipt `products_memory.xml`. Approved final vertex, raw status and remaining
arc are present in native VTK and per-seed logs. Other status3 causes remain
fatal. Timing/range instrumentation and the final A9 matched observations are
being checked separately; these passes do not close the whole IS8 goal.

### Final Joint Admission (2026-10-06)

The approved two-transport 32³/FP64/643e periodic Cartesian TGV path now includes
completed-step device sampling, mean supply, all products, exact restart,
transport-only override, independent actual-field checks, safety, resource
rejection, per-object transfer attribution and matched old/new observations.
Root CPU/CUDA/device-OFF/AIR5 builds and representative host-path continuation
regressions pass. Both explicit standalone NP=2 launch modes reproduce the
matched case's state/statistics and all ten scheduled products.

Four native trace checks match the actual 998064-byte per-rank/per-direction
face payload with either P2P copies or declared pinned DTOH/HTOD. Device worklets
execute without full-volume host mirrors; managed workspace traffic and final
compact geometry reads remain counted. Scalar-bar proxy visibility proved
insufficient for long mean-field titles; bounded layout plus actual colored
pixel checks repair this display issue without changing fields or cameras.

See the acceptance record for exact receipts, resource/timing definitions and
exclusions. CURVE, walls, AIR5, production-scale device visualization and PF/AP
remain outside this completed goal. No remote/Git operation was performed.
