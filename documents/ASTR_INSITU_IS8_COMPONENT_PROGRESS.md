# IS8-A / IS8-R Device Component Progress

Date: 2026-10-06. Branch: `feature/gpu_dev`. Approved bounded IS8-A0-A9 and
IS8-R0-R11 completed. Current resident-rendering admission and final evidence:
`ASTR_INSITU_IS8_RESIDENT_ACCEPTANCE.md`. The earlier compact-host entry is
recorded in `ASTR_INSITU_IS8_ACCEPTANCE.md`.
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

## IS8-R Device-Resident Rendering Implementation (2026-10-06)

The new R0-R11 goal is active in the current feature checkout. The accepted
IS8-A route above remains the compatible baseline, not either new rendering
pipeline. New `standard-device` and `direct-device` selections require the
default-off native rendering build; the first Q-only bridge is under build/test,
not a full-product admission. No remote job or Git write is authorized.

### R0 Graphics Interoperability Component

Root CMake builds `insitu_graphics_interop_probe`. Its VTK EGL context is
selected by the current CUDA device UUID, before the render-window constructor.
For VTK 9.6.2, setting the inherited DeviceIndex after construction alone does
not update the private EGL device selection. The original GPU1 failure and the
corrected two-GPU pass are separate observations; the production adapter
already selects the environment before construction.

Each card passes three frames with one reused CUDA-registered GL buffer,
device-side triangle filling, map/unmap, color/depth/background pixel checks,
mapped-buffer exposure rejection and explicit release. Only 49152 RGBA image
bytes are read for the three 64x64 frames. Both memchecks report zero errors.
The Nsight component records one registration, three maps/unmaps and kernels,
three OpenGL draws/pixel reads and one unregister. It does not record CUDA
memory-copy/UVM tables. This is not whole-Catalyst residency evidence.

Evidence directory: `tests/gpu_validation/out/insitu_is8_r0_graphics_20261006/`.
The immutable R0 receipts are `gpu0_memcheck.log`, `gpu1_memcheck.log` and
`graphics_gpu0.nsys-rep/.sqlite`. The preserved installed VTK/EGL libraries
remain unchanged. The separate CUDA-enabled rendering dependency has built
Catalyst and Python wrappers. Native product integration and complete dependency
admission remain subject to the gates below.

### R1 Configuration And Restart Identity

The parser accepts `rendering_pipeline=compatible|standard-device|direct-device`.
New selections require rendering, device processing and GPU derivatives.
Pipeline identity is included in collective configuration comparison. A native
capability query admits compatible in ordinary builds. The separate default-off
`ASTR_WITH_INSITU_DEVICE_RENDERING` candidate adds the two new bridges, with a
Q-only product gate until R5 is connected; unbuilt routes fail explicitly rather
than silently selecting compatible.

The enabled render control record is ASTRIR03, with processing, transport and
pipeline identity. Old ASTRIR01/02 controls identify compatible only. Compatible
fingerprint semantics remain unchanged. Cross-pipeline restore is refused,
including explicit override; changing only the existing face transport still
requires its original explicit override.

`test_insitu_run_config.py`: 87 passed. Native receipts `r1_native.xml` record
four unavailable-pipeline refusals and one compatible exact restart/transport
override. `r1_host_restart.xml` records four host-render continuation and
invalid-pipeline-control checks, including q/statistics and product equivalence.
Strict geometry-writer policy and actual new-route capability still await
the R3/R4 integration; these passes do not close all of R1.

### R2 Common Surface View Component

`DeviceSurfaceOwner` retains typed FP64 coordinate/velocity/Q/u arrays, Id64
triangle indices and read tokens for the lifetime of an immutable device view.
Its GPU display worklets produce separate FP32 coordinates/speed and UInt32
indices. They check field finiteness, display overflow, array extents and index
range. GPU reductions return bounds/speed range and validation status only.
The owner admits empty geometry without a host container. This first component
covers triangles; slice triangulation, normals, colors and trajectory segments
remain to be integrated. It does not change extraction or Q mathematics.

`r2_geometry_final.xml`: six passed, covering empty/nonempty NP=1 and NP=2
x/y/z, the retained compatible geometry write/read controls, and deliberate
GPU nonfinite/index faults. The strict branch writes no geometry files. Maximum
independent FP64 interpolation difference is 1.27675647831893e-15, display
cast/index difference is zero, and the source fields are unchanged.
`r2_memcheck.*.log` records zero errors for both NP=2 y rank processes.

`r2_device_view.nsys-rep/.sqlite` records Flying Edges plus GPU display packing,
index conversion and independent GPU audit kernels. The device-view range has
exactly 72, 8 and 16 byte D2H summaries, not point/index arrays. The entire
component capture has no managed D2H migration; extraction has its existing
4/4/8 byte scalar D2H controls. `test_insitu_device_geometry_trace.py` independently
checks both the old compatible transfer receipt and this new strict receipt.
These are bounded component observations, not full solver/rendering admission.

### R3 Dependency Candidate

`paraview-6.1.1-device-rendering.patch` adds an opt-in CUDA-to-OpenGL buffer
upload and a restricted path in the standard VTK polydata mapper. It is not a
custom mapper. Contiguous device coordinates/colors/triangle indices go through
VTK's standard arrays, VBOs and IBOs; unsupported configurations throw instead
of entering host loops. `ASTR_VTK_STRICT_DEVICE_ACCESS=1` makes CPU value access
to vtkmDataArray fail and uses read-only device pointer acquisition. The CUDA
patch is in the independent source/build, not the preserved installation.

The new optional root target `insitu_standard_device_probe` exercises analytic
triangles using actual vtkmDataArray, vtkPolyData, vtkActor and the standard
vtkOpenGLPolyDataMapper. `ASTR_BUILD_INSITU_RENDER_PROBES` is OFF by default and
requires the existing CUDA/Catalyst/device probe switches. The private rendering
modules and probe compile through root CMake. Both GPUs pass three analytic
frames (green/blue/green foreground, red background triangle and black exterior),
host-coordinate-access rejection and explicit graphics release. VTK's OpenGL
object factory must be initialized by `vtk_module_autoinit`; without it the
generic actor/renderer never invokes the draw function and the original probe
correctly fails black. The mapper uses six reduced bounding-box values instead
of walking device cells to determine clipping bounds.

`r3_standard_component.xml` records the two-card component pass. Both
`r3_gpu0_memcheck.log` and `r3_gpu1_memcheck.log` report zero errors.
`r3_standard_gpu0.nsys-rep/.sqlite` records three registrations, nine map/unmap
pairs, nine D2D uploads (72-byte positions, 24-byte colors and indices per frame),
three draws, three image reads and three unregisters. It records no CUDA D2H
or managed migration. Image reads total 49152 RGBA bytes. These are analytic
component observations, not Catalyst end-to-end admission.

The independent dependency disables the optional Viskores filter collection
while retaining Core/DataModel CUDA arrays. ParaView's `PARAVIEW_USE_VISKORES`
otherwise explicitly requires VTKm filters even when their module cache value
is NO. A private dependency-link patch avoids the filter umbrella for array-only
consumers. Its build helper also preserves user CUDA compile flags: the original
`MODIFY_CUDA_FLAGS` call dropped the private glibc compatibility include and
failed compiling DataModel. `catalyst-paraview`, `pvpython` and
`paraview_all_python_modules` now build successfully; the actual `pvpython`
imports ParaView 6.1 and the wrapped view module without `PYTHONPATH` overrides.
Building `pvpython` alone previously failed imports because the wrappers were
missing. Instantiating a raw view without an active session is not an import
test and aborts; the corrected import-only check passes. This is a build-tree
runtime, not an installed new production prefix.

### R3/R4 Independent Analytic Components

`r3_r4_render_components.xml`: three parametrized tests pass, each exercising
both GPUs. The modes are Viskores/VTK arrays, actual Conduit external device
arrays, and `DirectDeviceMapper`. The Conduit control asserts that its returned
UInt32 connection buffer is the original CUDA allocation; it does not substitute
a hand-built cell array after conversion. Fixed-stride cell offsets and cell
types are VTK implicit arrays, not per-cell CPU geometry containers.

`r3_r4_{conduit,direct}_gpu{0,1}_memcheck.log`: all four reports contain zero
errors. `r3_conduit_latest` and `r4_direct_latest` Nsight receipts each show
three registrations, nine maps/unmaps, nine device-to-device uploads, three
draws/image reads and three unregisters; no CUDA D2H or managed-memory
migration. Both transfer/lifecycle trace regressions pass. These traces cover
the analytic component, not the full solver or MPI image composition.

The direct mapper uses the same CUDA/GL upload implementation, stock ParaView
shaders and cameras, but supplies raw device views rather than VTK geometry
arrays. It is not an independent renderer. The common generic device descriptor
contains no CUDA/GL resource handles and retains its private allocation owner.

### Native Q Candidate In Progress

The root-CMake default-off rendering switch compiles the narrow adapter,
shared GPU Q extraction/display owner, Conduit entry and direct actor registry.
Fortran passes the explicit pipeline independently of face transport. The new
`device_render_pipeline.py` excludes merge/calculator/geometry writers and uses
existing ParaView views, scalar bars, collective screenshot and JPEG/EPS
publication. The common GPU coloring uses a bounded 4096-entry control palette
from the unchanged ParaView Lab preset, not a readback of field/color arrays.
FP32 display coordinates/UInt8 colors remain distinct from authoritative FP64
extracted coordinates and fields.

The full root-CMake CUDA/Fortran candidate now builds. The initial Q failure
was Viskores's default managed allocation being rejected by the strict graphics
upload. An isolated `NO_VISKORES_MANAGED_MEMORY=1` control passes; the strict
route now disables that allocator mode itself, without requiring a user variable.
The matching no-override test passes eight cases: both routes, NP=1 and NP=2
x/y/z, two actual Q frames each. The second-frame JPEGs are identical across
those cases. This is a Q short-test result, not full-product admission.

Evidence is `out/insitu_is8_r3_native_q_20261006`: `native_q_policy.xml`
(8 passes); `native_q_memcheck_diagnostic.xml` (8 cases, 14 zero-error rank
reports with the isolated allocator override); `native_cuda_trace.xml` (8
passes without that override); and `native_q_trace_attribution.xml` (1 pass
checking all 14 rank traces). Each Q extraction returns only 96 bytes of bounded
counts/ranges. The render ranges contain three D2D uploads matching the actual
point/index/color extents, no CUDA D2H. Each rank registers/releases three GL
buffers and maps/unmaps them six times. No UVM D2H is recorded. This does not
claim zero image readback or quantify native IceT image traffic yet.

HPC-X launcher wrappers choose their executable by the invoked basename;
resolving `mpiexec` to `env.sh`, or to the `orterun` binary without the package
wrapper, fails before the solver starts. The affected validation driver now
preserves the launcher name. Failed receipts remain in the same directory.
Nsight 2025.6.1 with OpenGL interception crashes inside its injection library's
`dlclose` while importing Python 3.14. CUDA/NVTX/MPI-only captures succeed;
the failed OpenGL run is retained, not used as a rendering rejection or an
image-transfer measurement. The earlier C++ analytic OpenGL traces remain
separate component evidence.

### R5 Resident Trajectory Candidate

The existing RK45 driver now has a resident-output branch. Accepted points,
colors/integrating vectors, per-particle connection indices and concatenation
remain on CUDA. It reads only the current 16/32 continuation states per round
(at most 2 KiB), plus bounded reductions. It does not accumulate those host
states into trajectory geometry. The compatible compact-output branch remains.
The common owner is now named `DeviceGeometryOwner` with a generic cell count
and explicit triangle/line arity; the underlying FP64/display separation is unchanged.

`resident_streamlines_component.xml` records 12 passes for NP=1/2 x/y/z,
actual TGV and constant fields, both/forward directions. The double-rank y
memcheck has two zero-error reports under `out/insitu_is8_r5_resident_lines_20261006`.
Its independently sampled vector maximum difference is `3.677613769070831e-16`;
the source check is unchanged. These are synthetic-field component checks.

The six-product native candidate also adds GPU slice coordinates/triangulation,
mean-vector coloring, empty-rank products and line rendering to both entries.
Its first full-product test produced all requested images, then correctly failed
because finalization still downloaded/wrote the 41-component 3-D statistics
array. Strict resident routes now skip that export; GPU statistics accumulation
and explicit native checkpoint state are unchanged. Compatible exports remain.
The corrected native all-product run passes eight cases (both entries, NP=1
and NP=2 x/y/z, pinned transport). Per-frame volume downloads and final full
statistics exports are zero. `native_all_corrected.xml` is the receipt; the
original failed output is retained separately.

The resident component Nsight pair contains two 1024-byte continuation reads
per rank, bounded pack/count reductions of at most eight bytes, device-only
concatenation and no UVM download. `component_trace.xml` closes this attribution,
not the complete native transfer audit.

### R6 Numerical And Display Gate

`out/insitu_is8_r6_20261006/numerical_v3.xml` records 16 passes: both entries,
both private face transports and NP=1/NP=2 x/y/z. A read-only CUDA oracle
independently interpolates each product's source halo, checks the contour
threshold/slice plane and constant-field trajectory, and checks display casts.
Only two scalar error values are downloaded per product when the explicit
diagnostic is enabled. The native rendering route never uses this oracle as
host geometry. Maximum product field difference is `1.5543122344752192e-15`;
display conversion difference is zero. Same-phase CPU/GPU authoritative flow
and cache difference is at most `1.9895196601282805e-13`; common physical
point/regional statistics differ by at most `1.1102230246251565e-15`.
Rendering preserves the GPU authoritative checkpoint fields exactly.

CPU statistics retain seam copies whereas GPU statistics pack unique periodic
nodes and leave unused storage zero. The comparison checks the explicit
CPU/GPU storage-role/native flags, identical clocks/partitions, the unique
32-cubed physical nodes and decoded regional FP64 statistics. It separately
checks CPU extra seam values against their corresponding global nodes and GPU
unused extras against zero. Raw storage-role bits and unused seam slots are
not physical-field errors. The earlier comparison-harness failures are retained.

`images.xml` compares all six native images across the sixteen configurations.
Colored geometry coverage differs by at most one pixel (including the
partition crossing); this is an image seam check, not a field precision check.
`analytic_pixels.xml` records three component modes, each on both physical
GPUs, with analytic triangle projection at most one pixel, front/back depth
occlusion and exact reference colors, CPU-access refusal and buffer reuse.
Neither proof relies on downloading product geometry. The default has not changed.

### R7 Transfer, Safety And Resource Gate

`out/insitu_is8_r7_20261006/memory.xml` records 16 passing native cases,
both entries and private transports, NP=1/NP=2 x/y/z. All 28 rank memcheck
reports contain zero errors. Initial frame zero has four products without
mean coverage; steps one and two have all six. The narrow existing UCX API
probe suppression remains in device-aware tests, not a kernel-access suppression.

`native_audits_v2.xml` records four matched external resource observations,
eight explicit host/device budget refusals and four actual native Nsight
captures. Sampled additional host RSS is at most 1108443136 bytes/node;
additional device memory at most 463736832 bytes/GPU. Minimum device free
memory exceeds 18 GB. The 20 ms observer does not prove interception of every
third-party instantaneous peak. Native budget checks remain enabled.

`transfer_ledger.xml` checks all eight rank captures. No compact geometry or
trajectory read ranges and no UVM D2H are present. Each rank's two-frame face
payload is 1488384 bytes per direction: pinned transport has that D2H and H2D;
device-aware has zero face D2H/H2D and matching peer payloads. Continuation
state reads total 28672 bytes/rank across 16 rounds, each 1024 or 2048 bytes,
not accepted trajectory arrays. Counts/ranges are bounded scalar metadata.

Render copies are D2D only, exactly matching actual display buffers: 498624
bytes on rank zero and 233760 on rank one. Empty rank-local line products
perform no uploads. Registrations equal releases and maps equal unmaps;
growth is accounted separately from reuse. The optional private pixel observer
records actual GL RGB/RGBA/depth reads without Nsight's failing Python/OpenGL
injection. Two frames/six products read 109440000 raw pixel bytes on rank zero,
63360000 on rank one. This includes the extra screenshot render, not geometry.
RGB screenshot reads also occur on the empty-geometry rank.

Native MPI events inside the render range show 24 Isend calls/rank (1318160
and 2639344 bytes) and 24 Irecv calls/rank (46081056 bytes each). These are
recorded API buffer sizes, including receive capacity and protocol metadata,
not a claim that every reserved receive byte traversed the wire. Do not sum
send and receive capacities to estimate image bandwidth. Geometry residency
therefore does not mean zero host transfer or GPU-only image composition.

R7 passes within the approved local budget. R8 exact continuation and failure
gates are running; R9/R10/R11 remain pending. The first R8 harness configuration
incorrectly requested keep=3; the solver correctly rejected it. It now uses
the existing test-only checkpoint protection marker, retaining production
keep=1/2 semantics. A run whose script changed between save and resume was
correctly rejected by the configuration signature. Frozen-script checks are
required; that failed run is not a numerical-equivalence failure.

### R8 Exact Continuation And Failure Gate

The sixteen exact continuation cases in `out/insitu_is8_r8_20261006/frozen_pipeline.xml`
all pass: both entries/transports and NP=1/NP=2 x/y/z, continuous twelve
completed steps versus checkpoint-five plus seven steps. Authoritative q,
caches, point/regional statistics, control clocks/identities and original
JPEG/EPS images match exactly. The source checkpoint is unchanged. Largest
individual case is 103279878 bytes, below the approved 256 MiB bound.

That receipt's later fault-wrapper test failed to inject its requested draw
error: `runpy` returns a copied namespace, so changing that dictionary did not
change a function's globals. It was a test fault, not a successful failure
gate. `lifecycle_v2.xml` contains the corrected six initialization/draw/image
publication passes. Both entries abort on initialization/draw exceptions;
approved EIO during EPS staging removes the partial pair, records the same
missing-frame receipt on all ranks and leaves later images and state exact.
Encoding/publication now have separate timing entries; no geometry file is
invented in the missing-frame receipt.

`interrupted_v3.xml` records six additional passes: both entries and interruption
before statistics, render control or COMPLETE. Partial bundles cannot resume;
the prior COMPLETE bundle and LATEST survive, cross-pipeline identity is
explicitly rejected, and resumed q/caches/statistics/control/images match the
healthy reference exactly. Source resources remain immutable. A previous
harness expected the generic override rejection rather than the more specific
cross-pipeline error; its retained failure is not a bypass of the guard.

R8 passes. R9 optional-build/compatible regressions are next; R10/R11 remain
pending, including the 256-cubed profile and the default switch.

### R9 Optional Builds And Compatibility Gate

`out/insitu_is8_r9_20261006` records successful root-CMake CPU and CUDA builds
with Catalyst disabled, and an AIR5 CUDA build without device products. The CPU
binary links no CUDA, VTK, ParaView or Catalyst libraries; the CUDA binary links
no VTK, ParaView or Catalyst libraries. Both disabled builds pass NP=2 four-step
versus two-plus-two exact state/statistics continuation. `config.xml` records
87 parser/collective passes; `unbuilt.xml` records four explicit unavailable-entry
refusals without fallback; `compatible.xml` records three real compact-entry
product/continuation checks. `air5.xml` records the existing bounded CPU/GPU
AIR5 volume-statistics regression, not an AIR5 resident-rendering admission.

Root-CMake staging installs the strict pipeline only when its optional rendering
target is enabled, together with its EGL and image-publication helpers. The staged
binary and staged pipeline pass a one-frame 32-cubed standard-device run. The
dependency remains the independent build tree, not a completed ParaView install.
The staged executable requires explicit runtime library discovery, for example:

```sh
export LD_LIBRARY_PATH=/home/dell/workspace/astr_dependencies/build/paraview-6.1.1-astr-device-gcc13/lib:/home/dell/workspace/astr_dependencies/install/catalyst-2.1.0-gcc13/lib:${LD_LIBRARY_PATH}
```

This is a local evidence command, not a portable deployment path or a standalone
binary claim. A deployment must supply its own matching dependency prefix.
R9 passes; R10 bounded 256-cubed admission and matched timing are next. The device
default remains unchanged until R10 and the R11 selection checks pass.

### R10 Bounded 256-Cubed Preflight And Attribution

The isolated `tgv256_demo` branch now supplies Q=0 and instantaneous streamlines
to both strict entries, with GPU speed coloring in [0,1] and 1280x960 JPEG/EPS.
The 32-cubed threshold, signed-u colors, seeds and diagnostic products remain
unchanged. The current binary passes sixteen repeated 32-cubed numerical gates,
two selected exact continuations and two selected y/z memchecks: twenty passes
in `insitu_is8_r10_32_regression_20261006/pytest_v2.xml`. Its initial temporary
parent-directory failure never started a solver and is retained separately.

`insitu_is8_r10_256_preflight_20261006/report.json` records all four two-step
preflights and three separate CUDA/NVTX/MPI traces. Small physical diagnostics
match the off baseline exactly. Strict products remain device-resident; images
are nonblank, correctly framed and speed-colored. External 20 ms sampling shows
strict extra node RSS at most 921567232 bytes, extra memory at most 2250944512
bytes per GPU and minimum free memory above 12 GB. Native phase budgets pass.
The copied input grid file is an input, not an intermediate field output.

`transfer_v2.xml` independently checks all six rank traces. The compatible
entry reads 77631488/75665408 bytes of compact geometry on the two ranks over
two frames. Both strict entries read zero geometry and no UVM D2H pages;
their render uploads are 27255552/26859264 D2D bytes. Each rank reads 12288
bytes of particle continuation state in six rounds, each 2048 bytes, plus
bounded scalar metadata. GPU buffer registrations/releases and maps/unmaps
balance. Images/IceT composition are still host-visible, not zero total D2H.
The first read-only checker incorrectly omitted the four-byte CUDA Fortran
finite-state flags from its scalar whitelist; the retained failed receipt
was corrected using the source and capture, not by allowing geometry reads.

The initial five-round invocation was interrupted by SIGTERM during its
preflight wrapper, after successful solver finalization. The sender is not
identified; no numerical failure appears in those logs. That incomplete
directory is retained, and is not included in timing results. The runner now
records signal interruption and every completed preflight. A fresh five-round
matrix subsequently completed under `insitu_is8_r10_256_matrix_v2_20261006`.
The interrupted invocation is excluded from the results below.

### R10 Five-Round Matched Matrix Complete

`report.json` records twenty complete 100-step runs, rotated four-case order,
no profiler during formal runs, and all requested 200 JPEG/200 EPS files per
rendering run. All matched kinetic-energy/enstrophy/dissipation differences are
zero. No checkpoint, volume, slice or VTK geometry output is present.

| Entry | Complete-window min/median/max, seconds | Sample SD | Pure RK median |
|---|---|---:|---:|
| Off | 60.368274 / 60.580561 / 60.617399 | 0.100664 | 56.371491 |
| Compatible | 257.208569 / 257.746235 / 264.008394 | 2.867249 | 56.784510 |
| Standard device | 85.644225 / 85.719774 / 85.815857 | 0.072039 | 56.387421 |
| Direct device | 85.608615 / 85.666557 / 85.994778 | 0.160788 | 56.412208 |

Complete windows exclude startup and include lazy first-frame setup and image
publication. Nested stages cannot be summed. Standard/direct add 25.139213/
25.085996 seconds over off, versus compatible's 197.165674. Compatible/standard
window ratio is 3.0068, not a CPU/GPU solver acceleration. Their two strict
medians differ by only 0.053217 seconds; no reliable winner is asserted.
All native resource budgets pass; standard/direct extra node RSS peaks are
815087616/808374272 bytes and per-GPU extra device peak is 2036707328 bytes.
Device free memory stays above 12 GB. The largest directory is 1560568974
bytes, and the group 23961800971 bytes before the preserved executable copy.
The preceding 20 ms external preflight observations independently pass.

Raw values, stages, environment, input/script/binary identity and resources
are retained in the report. Its frozen binary `benchmarked_astr` matches SHA256
`882e1ac5b4ef40cdd5941d5151383c2eaa8e7537ddec2fcca15884786ede3728`.
R10 closes only the approved 256-cubed two-product demonstration.

### R11 Default Selection And Bounded Admission Complete

`insitu_is8_r11_20261006/config_launcher.xml` has 117 passes; three unrelated
native-observer checks were deselected, not counted as passes. `native_v2.xml`
has four passes: default standard save to explicit standard exact resume,
explicit direct exact resume, and NP=1/2 compatible product regressions.
`unbuilt.xml` has six passes: explicit standard/direct and omitted pipeline,
each at NP=1/2, fail without a compiled strict entry and without fallback.
CPU/GPU Catalyst-OFF and AIR5 CUDA/Catalyst non-device root builds also pass.

Omitted `rendering_pipeline` now selects standard-device only when device
processing and rendering are enabled. Host/disabled defaults remain compatible;
optional dependency switches and all in-situ work remain default OFF.
The final binary differs from the benchmark only in tested parser/default
selection, not compute kernels or rendering scripts. Performance provenance
stays attached to the frozen pre-default binary.

The independent NP=2 pinned launcher passes four steps with frames 2/4,
ten JPEG/EPS pairs, no VTP, zero reported geometry host bytes and finalized
rank receipts. Its 78518807-byte directory and native phase resources fit
the 256 MiB, 4 GiB host, 2 GiB/GPU and 1 GiB free bounds. The existing small
checkpoint fixture is retained for restart, unlike the 256-cubed timing run.
No validation module is imported. Receipts are in `standalone_default`.

IS8-R0-R11 are complete within the scope in
`ASTR_INSITU_IS8_RESIDENT_ACCEPTANCE.md`. Image composition, bounded metadata,
particle continuation and explicitly pinned faces remain permitted host traffic.
Next are PF independent clocks and AP variable cadence; no CURVE, walls, AIR5,
other hardware, remote work or Git operation is admitted by this completion.
