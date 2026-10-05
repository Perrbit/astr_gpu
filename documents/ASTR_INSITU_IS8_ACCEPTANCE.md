# IS8-A GPU Resident Postprocessing Acceptance

Date: 2026-10-06. Scope: approved IS8-A0-A9, not all future IS8 extensions.
The device product implementation is opt-in; the host compatibility path remains.
No remote job, Git commit, branch operation or system configuration change was
performed during this goal.

## Capability Boundary

| Item | Accepted scope |
|---|---|
| Solver | Internally generated 32^3 periodic Cartesian TGV, FP64, 643e/643e, five conserved variables |
| Partition | NP=1; NP=2 x/y/z slabs, one rank per local GPU |
| Products | z=pi/4 index velocity slice, Q_rs=0.25 surface, instantaneous/Reynolds/Favre velocity streamlines |
| Diagnostic product | Constant physical velocity (1,0,0) on the postprocessing copy, not the solver |
| Sampling | Read-only completed RK step; unchanged authoritative state, caches and statistics |
| Integrator | Approved FP64 Cash-Karp RK45, length units, 0.1h/0.01h/0.5h, h=2*pi/32, error target 1e-8 |
| Trajectory | Sixteen approved seeds, both directions, pi per direction, MPI continuation, no periodic wrapping |
| Restart | Same computation backend/grid/topology, exact continuation; explicit transport-only override in either direction |
| Outputs | Paired JPEG/EPS, independent VTK geometry with field/time/terminal metadata, fixed camera and u color range [-1,1] |

An identified sub-minimum terminal request retains VTK's last accepted vertex,
raw status and remaining arc. The classifier requires the unchanged VTK error
sentinel, status 3, a positive remainder strictly below the approved minimum,
and a request equal to that remainder. Other failures remain fatal. No final
point extrapolation, smaller minimum or looser analytic threshold was used.

The Q definition is unchanged: $Q_{\mathrm{rs}}=-\mathrm{tr}[(\nabla u)^2]/2$, with the approved explicit
sixth-order physical velocity gradient. Slice, contour and streamline fields
are checked by independent interpolation at their final geometric coordinates.
This is numerical/product acceptance, not a turbulence or SBLI physical gate.

CURVE, walls, AIR5, arbitrary slice planes, larger grids, NP>2, render
repartition, asynchronous execution and adaptive product frequency are not
admitted by this device path. Their existing host capabilities are separate.

## Implementation And Dependencies

`insitu_products_gpu.cuf` orchestrates the native device products;
`insitu_device_velocity.cuf` owns private canonical nodes, halos and fixed face
slots; `insitu_sample_gpu.cuf` computes gradients/Q. Device mean velocities
come from `insitu_statistics_gpu.cuf`, with existing weights and coverage.
`insitu_device_products.cu` uses forced CUDA slice/Flying Edges worklets and
the private RK45/MPI driver in `insitu_device_streamlines.h`. Borrowed solver
arrays must be current-device CUDA allocations without a host mirror. Coordinates
are reused for a stable mesh/partition identity; transient arrays survive through
all consumers and are released at finalize or rebuilt at restart.

The accepted dependency route is a private ParaView 6.1.1 source/build providing
required CUDA/FP64/MPI Viskores component libraries, linked with the preserved
EGL/Catalyst rendering installation. It is **not** a full new ParaView install.
The root optional build does not replace the old host installation or require
unrelated GUI, flow-RK4, clipping or material-interface modules.
Reproducible coordinate, FP64/MPI, accepted-length and optional CUDA/glibc
compatibility patches are under `scripts/insitu/patches/`. The SDK is unchanged.
Root CPU+Catalyst, CUDA+Catalyst with device ON/OFF, and AIR5+Catalyst device OFF
builds passed. Default device OFF keeps the original dependency requirements.

Frozen local environment: NVHPC 26.1, CUDA 13.1.80, GCC 13, Catalyst 2.1.0,
ParaView 6.1.1, matching Python 3.14, HPC-X 2.25.1/Open MPI, two RTX 4000 Ada
devices, driver 595.91.07, kernel 7.0.0-34-generic. Both GPU IOMMU domains are
identity under user-configured `iommu.passthrough=1`. Earlier DMA-FQ failures
remain recorded; simultaneous kernel/domain changes do not isolate one cause.

## Two Independent Transports

`processing_backend='device'` requires an explicit, collectively consistent
`postprocess_transport='device-aware'` or `'pinned'`. This does not inherit
`ASTR_GPU_HALO_TRANSPORT`; solver halo transport stayed pinned in these tests.
Missing, inconsistent, unsupported or unavailable choices abort, with no fallback.
Private communicator/tag ownership isolates the postprocessing exchanges.

| Mode | Actual NP=2 evidence |
|---|---|
| device-aware | Device face buffers passed to MPI; Nsight records matching local P2P copies, no face DTOH/HTOD |
| pinned | GPU packs faces, fixed registered host slots stage face DTOH/HTOD, host MPI communicates, GPU unpacks |

At NP=2 x, each rank retains 219024 bytes of device face slots; pinned mode also
retains 219024 host bytes, while device-aware mode retains zero host face bytes.
Both observed frames retain allocation generation 1. Across two instantaneous
supplies and two mean-supply calls, face payload per rank/per direction is
998064 bytes: two 34848-byte, eight 109512-byte and one 52272-byte messages.
These are face payloads, not a partitioned full-volume download. NP=1 uses no
host face slots or inter-rank field transfer.

Four read-only SQLite gates on actual NP=2 all-product captures establish:

- CUDA snapshot/means, slice/Flying Edges, RK45 and velocity coloring actually ran.
- Sampling and mean-supply regions have no unified-memory DTOH migration.
- Geometry extraction has only 32 bytes of scalar DTOH per rank across two frames.
- Trace/compaction regions have 8-byte count DTOH per dispatch and 1024/2048-byte
  explicit continuation-state HTOD, with no unified-memory DTOH there.
- Library managed workspace HTOD is retained in the accounting: rank0/rank1
  trace regions total 1376256/589824 bytes, not exclusively tiny metadata.
- Unified-memory DTOH totals 4325376/3145728 bytes for rank0/rank1, entirely
  within final compact geometry/trace read regions. Page granularity is not
  the same as the logical compact-output byte count.

Checkpoint and final statistical-state downloads occur outside those product
regions. EGL consumes host-accessible compact geometry, not the three-dimensional
solver field. Therefore neither transport means zero total host traffic.
Local P2P evidence does not certify remote NVLink/network protocols or prove
another MPI installation avoids internal staging. No NCCL path was introduced.

## Numerical, Safety And Lifecycle Evidence

Receipts below are relative to `tests/gpu_validation/out/insitu_is8_transport_20261005/`.
They are separate gates, not a sum of unique tests or a production certificate.

| Receipt | Result |
|---|---|
| `config_and_timing.xml` | 84 passed: 75 configuration/collective checks and 9 timing/template checks |
| `standalone_config.xml`, `prebind_final.xml` | 15 standalone configuration checks; 1 formal independent device-prebind check |
| `compact_trace_final.xml`, `terminal_regression.xml` | 12 distributed synthetic trajectory cases; 4 unchanged-RK45/reference/terminal gates |
| `products_layout.xml` | Final layout, 16 passed: 8 native products/independent field checks, 6 x/y/z restart/override cases, 2 native memcheck cases |
| `products_final.xml` | 20 passed before presentation-only label fixes; also includes 4 actual host/device observed-budget rejections |
| `native_transfer_attribution.xml` | 4 passed: actual two-backend, two-rank transfer/operator attribution |
| `products_postlink.xml`, `products_postlink_restart.xml` | After link-only regeneration: 4 NP=2 product/memcheck gates and 2 x-slab restart/override gates |
| `host_restart_final.xml`, `air5_host_restart_final.xml` | Final layout, device-OFF TGV host restart: 2 passed; AIR5 mean-wall host restart: 1 passed |

Maximum independently evaluated instantaneous/Q/Reynolds/Favre geometry-field
difference is 2.1649348980190553e-15, below 2e-10. The constant-field distributed
endpoint/straightness difference is 1.7763568394002505e-15; seams are continuous.
Same-backend state/cache/statistics/control, JPEG pixels and VTK bytes match
continuous and checkpoint-resumed cases. Unapproved transport switches reject;
transport-only override preserves statistics and clocks and changes only the
transport identity. Device control uses ASTRIR02; old host control remains ASTRIR01.
Source batches remain unchanged. Initially uncovered mean products are absent,
not replaced with instantaneous velocities.

Four rank memcheck logs from native NP=2 y all-product/zero-coverage cases have
zero errors. Fixed slots, failure paths and components have separate clean
records. No sanitizer suppression or numerical tolerance adjustment was used.
Images are independently decoded, geometry fields/coordinates read back, and
native GPU UUID/lifecycle recorded. Long mean-field legend titles exposed a
rendered color-bar loss despite proxy visibility; fixed bar length/thickness,
short display titles and a colored-pixel gate now check the actual output.
Camera, field names, colors and color ranges remain unchanged.

## Resource And Timing Contract

The accepted limits remain extra host 4 GiB/node, extra device 2 GiB/physical
GPU, device reserve at least 1 GiB and each case directory at most 256 MiB.
Native phase-boundary checks and independent 20 ms process-tree RSS/NVML
observations supplement controlled-allocation admission. Neither catches every
transient third-party peak or guarantees recovery from an arbitrary OOM.
Actual observed-budget violations abort; no reduced quality or backend fallback.

Matched short-window receipts use 32^3, dt=1e-3, four complete steps, NP=1/2 x,
images at steps 2/4, mean coverage only at step4. Ten product frames are retained.
Render-off keeps the same statistics and checkpoint schedule; it is not a pure
RK timing. Initialization and finalize are separate; lazy first-frame pipeline
setup remains in the window. Each stage is the maximum of per-rank accumulated
local times, without new phase barriers. Nested phases and rank maxima must not
be summed. The old host path uses its original accepted-length counter while
the new device path uses the approved correction, so streamline vertex counts
can differ. Wall differences cannot be attributed solely to volume transfers.
Final isolated receipt:
`tests/gpu_validation/out/insitu_is8_final_20261006/report.json`.
The table reports accumulated seconds, including both frames where applicable.

| NP | Render-off window | Old host window | Pinned device window | Device-aware window |
|---|---:|---:|---:|---:|
| 1 | 0.789619 | 3.526173 | 3.254824 | 3.289197 |
| 2 x | 1.129778 | 4.478137 | 4.006239 | 3.951360 |

| Stage, NP=2 x | Pinned device | Device-aware |
|---|---:|---:|
| Snapshot/halo/gradient sample, inclusive | 0.004667 | 0.008970 |
| Mean supply, inclusive | 0.003243 | 0.003716 |
| Slice/Q extraction, inclusive | 0.057104 | 0.057722 |
| Slice/Q compact read | 0.017213 | 0.018026 |
| Product consumer, inclusive | 2.830788 | 2.784974 |
| Render, inclusive | 0.171634 | 0.161091 |
| Image encoding | 0.265733 | 0.228931 |
| Geometry writing, inclusive | 0.080037 | 0.077740 |

Device face exchange alone is not the dominant cost in this small short window.
First-frame Catalyst/view setup, compact product handling and publication remain
substantial; consumer time contains these subphases. Different per-stage maxima
cannot be subtracted to recover an exclusive cost. This is one matched local
observation, not repeated-run significance or a production scaling result.

Across the final device groups, external peak differences against render-off
were <=1226477568 host bytes/node and <=451153920 bytes/physical GPU.
Minimum observed device free was 18708037632 bytes. The largest device case
directory was 93120948 bytes; every group stayed below its 256 MiB case limit.
Native phase checks also passed. Third-party initialization contributes to
these differences and is not a measurement of only the fixed face buffers.

The native Nsight captures in `insitu_is8_native_trace_20261006/` predate
presentation fixes (black legend text and bounded legend layout). Sampling,
extraction and transfer source/compiled objects are unchanged. The final root
build performed link-only regeneration of GPU executables, without recompiling
the 115 native objects, and changed the executable hash. Its two-backend NP=2
product and zero-coverage memcheck gates pass; final state/statistics match the
previous accepted binary exactly. Two x-slab restart/override gates also pass.
The timing/trace executable hash is retained separately from the current
postlink executable: these are not new measurements of the regenerated binary.
Operator/transfer evidence is reused for unchanged code, not claimed as a new
capture of the current executable or final font/layout.
`final_dependency_manifest.json` in the receipt directory retains the immutable
dependency manifest identity and current executable/pipeline/script hashes.

The initial mean-frame fixture failure and the interrupted concurrent timing
run are excluded, not relabeled as success. Both remain in their original output
directories. New final receipts do not overwrite old measurements.

Final standalone cases are `insitu_is8_standalone_final_{aware,pinned}_20261006/`.
Both execute NP=2, frames2/4 with covered means only at4, and match the final
controlled run's authoritative state/statistics datasets exactly. They use no
validation imports. The earlier same-settings standalone runs are retained.

## Reproduction And Next Steps

Build/configure details and independent launch commands are in
`scripts/insitu/patches/README.md` and `scripts/insitu/README.md`.
Native product tests are `test_insitu_device_products.py`; actual captures and
matched resource/timing controls use `run_insitu_device_acceptance.py`.
`test_insitu_device_native_trace.py` reads immutable per-rank SQLite captures.
Do not run a timing pair concurrently with another GPU validation.

This closes the approved bounded IS8-A path only. The next planned work is
PF0-PF3 independent fixed product clocks, followed by AP0.1-AP3.2 variable
frequency. CURVE/wall/AIR5 device products and asynchronous/independent analysis
resources retain separate decision entries and do not inherit this admission.
