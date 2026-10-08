# ParaView Streamline Coordinate Precision

## Device Display Buffer Capacity Reuse

`paraview-6.1.1-device-buffer-capacity.patch` applies after the device-rendering
and allocation-preflight patches in the private ParaView 6.1.1 source. It changes
only `vtkOpenGLBufferObject::UploadDevice`: a growth allocation reserves 25%
headroom, subject to the existing allocation callback and `GLsizeiptr` bounds.
Copies, point/cell/index counts and draw lengths still use the exact payload.
Ordinary host uploads, extraction, interpolation and RK45 are unchanged.

Apply with `--dry-run --fuzz=0`, then without `--dry-run`, and rebuild the
`RenderingOpenGL2` target. Use `--dry-run --reverse --fuzz=0` to check an already
patched tree. Never replace a rendering library while an application uses it.
Retain the preceding library and its hash for comparisons and rollback.

The original periodic CURVE 256-cubed 100-step image run hit its 6 GiB additional
device budget at step 98. CUDA allocation tracking, pinned/device-aware controls
and a Python GC control did not explain or remove the frame-to-frame growth.
Capacity reuse removed growth in the 12-frame actual-field diagnostic, with all
48 JPEG/EPS files byte-identical to the unmodified rendering library. This
associates the growth with repeated exact-size graphics-buffer reallocation;
it does not establish a general driver leak. Subsequent actual periodic CURVE
256-cubed 100-step OFF/ON checks pass with stable frame-after device memory;
their numerical/safety/transfer and observed-resource receipts are in plan
section 5.5. The old y-wavy 100-step OFF case fails numerically at step 29 before
rendering is enabled; the capacity patch does not repair that numerical failure.
The user approved nondimensional Tw=1 for this round's y-wavy scale fixtures.
Fresh 64/128-cubed numerical/exact-restart gates and 64/128/256-cubed memcheck
pass; the fresh 256-cubed numerical/exact-restart and four entry/transport
observations also pass. Five clean timing rounds per combination and the selected
standard-device/device-aware 100-step OFF/ON complete with all 200 JPEG/200 EPS
files and the unchanged 6/16/2 GiB budgets. Frame-after memory increases by only
2 MiB first-to-last per rank, remaining bounded rather than byte-constant.
The two-mapping scale matrix is closed only for these approved local fixtures,
default 16 seeds and NP=2 x; this is not a generic thermal-wall/filter fix.
Historical Tw=273.15 evidence is retained and not reused as new-input timing.

The root-built graphics interop probe checks growing uploads and pixel output;
the standard-device probe also draws fixed/growing meshes through both entries.
These component controls do not replace actual-field restart, memory-safety,
transfer, external peak-resource or complete production-window gates. Current
evidence is under `build_insitu_air5_device_render/curve_graphics_lifecycle_20261007/`;
the production closure is recorded in the in-situ plan, section 5.5.

## CURVE Scale Allocation Preflight

`paraview-6.1.1-allocation-preflight.patch` applies to the private device
dependency after the existing device-rendering patches. It adds an optional
Viskores allocation callback, routes host-launched Thrust scratch through that
allocator, and checks OpenGL buffer growth through ASTR's exported callback.
No locator, contour, interpolation or RK45 mathematics changes. Apply without
fuzz, then rebuild `viskores_cont RenderingOpenGL2` and the root-built ASTR
device products. Never replace libraries while a job uses them.

The callback admits each requested byte count against the decrease in CUDA
device-wide free memory since the native observer baseline and the configured
free-memory reserve. Thus retained arrays, deferred frees and earlier frames
remain counted. Other processes' allocation growth is conservatively counted
too. The job's current NVML allocation increment is checked independently so
another process freeing memory does not increase the job's configured budget.
The general scale route requires one MPI rank per physical GPU. The bounded
M12 exception below uses separate atomic cross-process reservations. Guarded Viskores
allocations use the synchronous memory allocator, not CUDA asynchronous pools.
The existing native NVML/RSS checks and external 20 ms measurements remain
independent. The guard does not promise to intercept every graphics-driver
texture or arbitrary third-party allocation.

The root-CMake `insitu_allocation_guard_probe` checks large-array rejection
before malloc, actual Thrust scratch interception, unchanged sort output and
scratch rejection. `test_insitu_allocation_budget.py` checks overflow-safe
admission arithmetic separately. Larger CURVE scale admission requires the
patched dependency and an active native observer; a missing patch cannot
silently fall back to an unguarded larger run.

### Bounded Shared-GPU Allocation Lifecycle

`paraview-6.1.1-shared-allocation-lifecycle.patch` applies after the existing
preflight and device-rendering patches, without fuzz. Rebuild `viskores_cont`
and `RenderingOpenGL2` while no application uses these private libraries.
Both exported lifecycle version functions must report version 1; a shared-GPU
run rejects an older dependency instead of disabling its guard.

Only the approved M12 64x32x24-interval, NP=4, standard-device/pinned fixture
can use this exception: two ranks per physical GPU, at most 2 GiB additional
device memory per GPU and 4 GiB host memory per node, with at least 1 GiB
device free. A robust process-shared mutex in an MPI shared-memory window
serializes request admission per physical GPU. Successful CUDA/OpenGL
allocations retain their reservation until actual release; deferred CUDA frees
do not return it when merely queued. Failed requests return their reservation.
The ledger also checks arithmetic overflow and unmatched releases.

Admission conservatively counts outstanding reservations in addition to the
observed device-wide/NVML allocation increase. It can therefore refuse a
request earlier than an exact allocator-only budget would. Native RSS/NVML
and external 20 ms observations remain independent; arbitrary textures and
unhooked third-party allocations are not guaranteed to be intercepted.
Retained dependency buffers at finalization may leave nonzero reservations,
which are reported rather than automatically labeled a leak. The component
probe checks actual immediate/deferred CUDA release and concurrent refusal
before allocation. This does not admit general shared-GPU CURVE or production
workloads and is not a multi-GPU scaling benchmark.

The user approved this local ParaView 6.1.1 dependency patch on 2026-09-29.
It changes four `vtkPoints` allocations to double storage in the serial
streamline thread output, merged output, and parallel tail creation/reception,
plus the parallel per-segment integration seed array from float to double.
RK45, interpolation, step sizes and ASTR fields are unchanged. Streamline point
coordinates require twice the storage of the original float coordinates; resource
measurements must be repeated for the complete patched pipeline.

Apply to a clean ParaView 6.1.1 source tree with GNU patch, without fuzz:

```bash
patch --dry-run --fuzz=0 -p1 -d "$PV_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-streamline-fp64.patch"
patch --fuzz=0 -p1 -d "$PV_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-streamline-fp64.patch"
cmake --build "$PV_BUILD" --target FiltersFlowPaths FiltersParallelFlowPaths -j 2
cmake --install "$PV_BUILD/VTK/Filters/FlowPaths"
cmake --install "$PV_BUILD/VTK/Filters/ParallelFlowPaths"
```

`PV_BUILD` must already be configured with the intended install prefix and the
same compiler/MPI/ABI as the Catalyst installation. The two install commands
replace only the corresponding module installations, not a system ParaView.
Do not install while another process is using those libraries. For an already
patched tree, `patch --dry-run --reverse --fuzz=0 ...` checks patch presence.

Validate with the ASTR paired driver using `--streamlines --extracts` and the
matching Catalyst implementation. The constant-field oracle must pass 2e-10
endpoint/straightness error and MPI crossing. Double file storage alone does not
prove intermediate integration or communication remained double precision.
The original failed run is retained as `insitu_streamlines_20260929`.

This is a local dependency deviation, not an upstream VTK release guarantee.
Reassess the patch rather than force-applying it to later ParaView versions.

## IS8-A Private Device Dependency

The bounded native route is now accepted; historical component observations
below explain its development. The required partial-build libraries, root
integration recipe and latest gate are at the end of this document.

The separate CUDA/Viskores dependency route was approved on 2026-10-05.
It does not replace the validated host-processing/EGL installation. Copy the
source to a new directory, retain the coordinate patch above, and apply
`paraview-6.1.1-viskores-fp64-mpi.patch` with `--fuzz=0`. This patch adds an
opt-in build option because VTK otherwise forces Viskores default floating
types to FP32 and disables its MPI support, regardless of cache settings.
The unselected option preserves the original dependency configuration.

`../build_device_paraview.sh` requires explicit `PV_DEVICE_SOURCE`,
`PV_DEVICE_BUILD`, `PV_DEVICE_PREFIX`, `CATALYST_DIR`, `MPI_ROOT`, and
`CUDA_ROOT`. It configures CUDA, FP64, MPI, the VTKmFilters plugin, Python,
and EGL without X. Set `CONFIGURE_ONLY=1` for configuration alone. The
default parallel build is two jobs; no system installation is modified.
Set `PV_DEVICE_PYTHON` to the intended interpreter before configuration.
For a bounded compute-only dependency build, set `PV_DEVICE_TARGETS` to an
explicit space-separated list of CMake targets and `PV_DEVICE_INSTALL=0`.
Partial builds never attempt to install missing modules. The default remains
a full build and installation. The approved IS8 path does not require unrelated
GUI, clipping, or material-interface filters.
`PV_DEVICE_PYTHON_INCLUDE` and `PV_DEVICE_PYTHON_LIBRARY` can explicitly pin its
matching development headers and library when upgrading an existing build cache.
Do not confuse generated feature macros with evidence of device execution.

CUDA 13.1 and newer glibc can disagree on the exception specifications of
`rsqrt`/`rsqrtf`. The optional `cuda-13.1-glibc-rsqrt.patch` changes only
these two declarations in a private header copy. Never apply it inside the
installed CUDA SDK. `../prepare_cuda_glibc_compat.sh CUDA_ROOT NEW_DIRECTORY`
creates a private copy of `cuda_runtime.h` and `crt/`, applies the patch,
and prints the include flag for `PV_DEVICE_CUDA_FLAGS`. Copying the `crt/`
include chain is necessary: `crt/common_functions.h` includes the math
header relative to its own directory, so overlaying only the math header
or pre-including it does not override the original declaration.
Only use this compatibility patch for an environment reproducing the conflict.
[NVIDIA developer discussion](https://forums.developer.nvidia.com/t/fedora-43-and-nvcc-cuda13-1-error-exception-specification-is-incompatible-rsqrt-rsqrtf/354510)
records the same declaration mismatch. This is not a new math approximation.

ASTR's root CMake has the default-off `ASTR_BUILD_INSITU_DEVICE_PROBES`
option. With Catalyst, tests, CUDA C++ compiler/host compiler, and `VTK_DIR`
explicitly configured, build `insitu_device_compiler_probe` and
`insitu_device_rk45_probe`. The latter tests the private FP64 Cash-Karp
adaptation against the original VTK CPU integrator for constant/rotational
tangents, both integration directions, domain exit, and stagnation. It also
checks nonfinite rejection. This does not yet verify mesh interpolation,
MPI particle continuation, mean fields, or any complete visualization product.

With the separately built `Viskores_DIR`, root CMake also exposes
`insitu_viskores_rk45_probe`. It generates a 33-node-per-axis synthetic field
in a CUDA allocation, borrows that allocation without ownership transfer, and
forces CUDA interpolation/RK45 dispatch. Only 16 particle results per field
are read on the host; the input buffer must never acquire a host mirror.
`test_insitu_viskores_device.py` requires `ASTR_INSITU_DEVICE_MPI_PREFIX` to
identify the matching Open MPI installation. Its single-rank launcher uses
`ob1`/`pt2pt`: it does not exercise device halo exchange or MPI RMA. This avoids
loading unrelated UCX RMA components before an application CUDA context exists.
The corresponding memcheck is a component check, not a CUDA-aware MPI gate.

The affine analytic/interpolated comparison is made at the same accepted arc
length. Interpolation rounding can turn a near-zero RK45 error estimate into
exact zero, activating VTK's existing exact-zero branch and choosing a different
step. That step difference is reported, not hidden or used to compare points at
different arc lengths. The unchanged adaptive integrator has a separate original
VTK differential test; the complete cross-rank endpoint gate is still pending.

The accepted-length correction was approved on 2026-10-05. Apply
`paraview-6.1.1-streamline-accepted-length.patch` with `--fuzz=0` to the separate
device dependency, retaining the coordinate patch. It changes only the upper
trajectory counter to `abs(stepTaken)`, not the integrator, bounds or tolerance.
Do not apply this patch to the old validated installation. The root-CMake
`ASTR_INSITU_PATCHED_STREAMTRACER_SOURCE` option can compile the patched
translation unit into a private reference executable without installing it.
`insitu_streamtracer_length_probe` is the original failing control;
`insitu_streamtracer_length_fixed_probe` verifies corrected termination.
`insitu_device_trace_probe` checks actual-step accumulation and exact small-state
resume, including a simulated ownership change. It is not yet an MPI gate.

`insitu_viskores_trace_probe` checks bounded distributed interpolation and
continuation on a synthetic uniform grid at NP=1/2. Its MPI payload contains
host particle states only, not device fields. The component launcher disables
HCOLL/UCC and `opal_cuda_support` to avoid unrelated MPI CUDA classification
errors under memcheck; CUDA worklets remain mandatory. Never copy that MPI
setting to the solver's device-buffer halo launcher.

On the pre-reboot local kernel/driver/IOMMU configuration, the MPI-independent
`insitu_device_peer_probe` fails actual buffer verification even though CUDA
peer-copy calls report success. Two cards had `DMA-FQ` IOMMU domains. Do not
use CUDA IPC field tests to pass IS8 until this environment is corrected and
the direct-copy and larger MPI payload controls pass. The private CUDA Fortran
`insitu_device_velocity_probe` is a separate root-CMake target. Its NP=1 result
passes; disabling IPC passes NP=2 only as a diagnostic, not as device-transfer
acceptance. Details and failed receipts are in
`documents/ASTR_INSITU_IS8_COMPONENT_PROGRESS.md`.

The user subsequently approved an explicit pinned-host face transport alongside
device-buffer MPI, without requiring an IOMMU change. Only halo/shared-node
faces may be staged; full-volume host mirrors and silent fallback remain banned.
The fixed registered pinned slots and device-buffer route are now implemented
in the private sampler. After reboot, kernel 7.0.0-34-generic and identity IOMMU
domains pass bidirectional peer-copy, larger CUDA-aware MPI, and NP=1/2 x/y/z
private sampler controls. The kernel and domain mode both changed, so the earlier
failure is not isolated to one cause. Each transport has separate correctness,
budget and memcheck evidence; neither yet has full product admission. Details
are in `documents/ASTR_INSITU_IS8_COMPONENT_PROGRESS.md`.

For the frozen uniform TGV contour, root CMake can compile the upstream
`viskores/filter/contour/ContourFlyingEdges.cxx` translation unit directly via
`ASTR_INSITU_FLYING_EDGES_SOURCE`. This uses the original Flying Edges algorithm
and forced CUDA worklets, linked with the private build's cont, worklet,
filter_core and filter_vector_analysis libraries. It avoids building unrelated
filters in the unified contour library. This is not a replacement algorithm or
permission to enable a CPU fallback. Geometry admission still requires actual
operator, field, partition, transfer and memory checks.

## Accepted Partial Build And Root Integration

The complete receipt is `documents/ASTR_INSITU_IS8_ACCEPTANCE.md`. Keep the
previous EGL/Catalyst install intact. Apply coordinate, FP64/MPI and
accepted-length patches to a separate ParaView 6.1.1 source copy, with
`--dry-run --fuzz=0` followed by the matching non-dry command. All patches are
version-specific. The accepted-length reference is compiled privately; no
replacement of the old VTK library is necessary.

Provide the explicit source/build/prefix/compiler/MPI/CUDA/Catalyst variables
documented above, plus a compatible Python interpreter/development library.
Then build only the required component targets:

```bash
PV_DEVICE_TARGETS='viskores_cont viskores_worklet viskores_filter_core viskores_filter_vector_analysis' \
PV_DEVICE_INSTALL=0 BUILD_JOBS=2 bash scripts/insitu/build_device_paraview.sh
```

The partial component build does not produce a full ParaView installation.
Use its generated Viskores package directory and the preserved Catalyst
implementation at runtime. ASTR compiles the upstream Flying Edges translation
unit directly, avoiding unrelated unified-contour filters. The root build
forces CUDA execution and FP64; it does not use Viskores's RK4/Euler flow library.

With `ASTR_ROOT`, `ASTR_BUILD`, `NVHPC_ROOT`, `CUDA_ROOT`, `MPI_ROOT`,
`HDF5_ROOT`, `CATALYST_DIR` and `PV_DEVICE_SOURCE/PV_DEVICE_BUILD` explicitly
set to compatible installations:

```bash
cmake -S "$ASTR_ROOT" -B "$ASTR_BUILD" \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_Fortran_COMPILER="$NVHPC_ROOT/compilers/bin/nvfortran" \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DMPI_Fortran_COMPILER="$MPI_ROOT/bin/mpif90" \
  -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc" \
  -DHDF5_ROOT="$HDF5_ROOT" -Dcatalyst_DIR="$CATALYST_DIR" \
  -DASTR_WITH_CUDA=ON -DASTR_WITH_CATALYST=ON -DASTR_WITH_INSITU_DEVICE=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER=g++-13 \
  -DViskores_DIR="$PV_DEVICE_BUILD/lib/cmake/paraview-6.1/vtk/viskores" \
  -DASTR_INSITU_FLYING_EDGES_SOURCE="$PV_DEVICE_SOURCE/VTK/ThirdParty/viskores/vtkviskores/viskores/viskores/filter/contour/ContourFlyingEdges.cxx"
cmake --build "$ASTR_BUILD" --target astr --parallel 2
```

Only for the demonstrated CUDA/glibc declaration conflict, append the private
overlay include flag to `CMAKE_CUDA_FLAGS` and `PV_DEVICE_CUDA_FLAGS`. Do not
patch the SDK. Default `ASTR_WITH_INSITU_DEVICE=OFF` remains buildable without
Viskores. `ASTR_BUILD_INSITU_DEVICE_PROBES=ON`, `VTK_DIR` and optional private
streamtracer source are test-only requirements, not native-runtime requirements.
Freeze generated CUDA/MPI/FP64 macros, library versions/hashes, actual root
link command and GPU mapping alongside validation results. A cache flag alone
does not verify device execution.

## IS8-R CUDA Rendering Candidate

`paraview-6.1.1-device-rendering.patch` is an unadmitted candidate for the
separate ParaView 6.1.1 source, not the preserved IS8-A installation. It adds
opt-in `ASTR_VTK_DEVICE_RENDERING`, a reused CUDA/GL buffer copy, and a bounded
device-array path inside the standard vtkOpenGLPolyDataMapper. This is not the
dedicated mapper backend. It currently requires contiguous FP32 points,
device UInt32 homogeneous triangle or line indices and device UInt8 RGBA values named
`_astr_display_rgba`; unsupported layouts/render modes throw.

Apply without fuzz to the separate source, using a dry-run first:

```bash
patch --dry-run --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-device-rendering.patch"
patch --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-device-rendering.patch"
patch --dry-run --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-device-core-dependencies.patch"
patch --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-device-core-dependencies.patch"
PV_DEVICE_RENDERING=ON PV_DEVICE_TARGETS='RenderingOpenGL2 AcceleratorsVTKmCore AcceleratorsVTKmDataModel IOCatalystConduit' \
  PV_DEVICE_INSTALL=0 BUILD_JOBS=2 bash scripts/insitu/build_device_paraview.sh
```

The script defaults `PV_DEVICE_RENDERING=OFF`; existing compute-only recipes
do not implicitly enable this candidate. Never mix the candidate VTK core
with the older Viskores ABI. Root CMake's separate default-off
`ASTR_BUILD_INSITU_RENDER_PROBES` exposes `insitu_standard_device_probe` only
with CUDA, Catalyst and the existing device probes enabled. Point `VTK_DIR`
and `Viskores_DIR` to the same private build and use its real loaded libraries.
The probe uses ordinary vtkmDataArray/vtkPolyData/vtkActor and the standard
mapper, plus analytic pixel checks. It is not a Catalyst admission receipt.
Launch this FP64/MPI probe with the matching MPI runner, even at NP=1. The
core-dependencies patch keeps control/worklet libraries on the VTK array
interface and links the filter umbrella only to the filter module (Fides
retains its clean-grid dependency). It avoids building unrelated CUDA
filters for an array-only target; the option-off dependency graph is preserved.

`ASTR_VTK_STRICT_DEVICE_ACCESS=1` is an internal dependency guard: CPU value,
tuple and geometry access throws; contiguous device access borrows a read
pointer, not a write pointer that could invalidate the producer's read tokens.
It is not a user-facing solver option or evidence that a ParaView filter runs
on CUDA. Unsupported filters must still be excluded or rejected before use.
The strict component uses device allocations rather than managed-memory
fallback. Actual copies, geometry residency, resource reuse and release still
require trace/safety verification. The probe alone is not admission; the later
R0-R11 native gates below now provide the bounded independent evidence.

The R5 common owner retains both FP64 geometry and GPU-generated FP32 display
arrays for slice triangles and accepted trajectory line cells. Native strict
routes use `device_render_pipeline.py`; they never call a geometry writer or
export the final full 3-D statistics array. Statistics accumulation and explicit
native checkpoint state remain available. The compatible route keeps the old
geometry/statistics exports. Native R6-R11 numerical, safety, transfer,
continuation, resource and selection gates have since completed, as recorded below.

`paraview-6.1.1-pixel-audit.patch` is an optional diagnostic on that same source.
It wraps actual OpenGL render-window and IceT color/depth reads, leaves their
arguments unchanged and records sizes/formats/pack destinations only when
`ASTR_VTK_PIXEL_AUDIT=1`. It is not needed for ordinary rendering. Apply with
the same dry-run/`--fuzz=0` discipline, then rebuild `RenderingOpenGL2` and
`RemotingViews`. Native CUDA/NVTX/MPI traces do not measure GL pixel readback;
the audit supplies a separate pixel ledger and must not be called a zero-D2H
proof. Unknown formats or PBO destinations need additional attribution before
acceptance. Do not enable diagnostic output during matched performance runs.

The strict homogeneous Conduit converter retains device UInt32 connectivity
with implicit fixed-stride cell offsets. It does not convert to a host Id64
vector or use the general Viskores-to-VTK cell conversion. Only matching
triangle/3 or line/2 arities are accepted. The standard mapper retains its
attribute VBOs when each frame supplies a new external array wrapper; GPU
capacity growth re-registers the affected buffer, not every frame by default.

For the actual private Catalyst Python runtime, build the wrappers explicitly:

```bash
PV_DEVICE_RENDERING=ON \
  PV_DEVICE_TARGETS='catalyst-paraview pvpython paraview_all_python_modules' \
  PV_DEVICE_INSTALL=0 BUILD_JOBS=8 bash scripts/insitu/build_device_paraview.sh
"$PV_DEVICE_BUILD/bin/pvpython" --no-mpi --force-offscreen-rendering \
  -c 'from paraview import simple; print(simple.GetParaViewVersion())'
```

`pvpython` alone does not build the Python wrappers. Do not instantiate a raw
`vtkPVRenderView` without an active ParaView session as an import test. The
partial build is a build-tree runtime, not a complete installable prefix;
global `all`/install additionally requests unrelated Viskores filters. The
array-only helper deliberately sets `PARAVIEW_USE_VISKORES=OFF` while enabling
CUDA Core/DataModel modules explicitly. The dependency patch also preserves
the CUDA compiler's user flags before `MODIFY_CUDA_FLAGS` appends its options,
including any separately generated glibc compatibility include overlay.

Root CMake's default-off `ASTR_WITH_INSITU_DEVICE_RENDERING` requires
`ASTR_WITH_INSITU_DEVICE`. Select `ParaView_DIR`, `VTK_DIR` and `Viskores_DIR`
from this same private build. It links the narrow control bridge to
ParaView's view/preset module and exports a local actor lookup for the direct
pipeline. CUDA/GL details remain in `src_gpu/`; no device handles enter Fortran.
Both resident entries pass the bounded 32-cubed six-product numerical, display,
memory, transfer and same-entry exact-continuation gates. Their script is
`../device_render_pipeline.py`; compatible uses the existing `tgv_pipeline.py`.
The 256-cubed Q=0/instantaneous-streamline preflight and four-entry five-round
100-step comparison also pass. R11 configuration, launcher, default/exact
continuation and unavailable-entry refusal gates pass. Enabled device rendering
now defaults to standard-device; direct and compatible are explicit choices.
Host processing remains compatible, and optional dependency switches and
in-situ work remain default OFF. Admission is still only the approved TGV scope.
Latest receipts and the retained benchmark binary identity are in
`documents/ASTR_INSITU_IS8_RESIDENT_ACCEPTANCE.md`.

Native Python plus Nsight OpenGL injection failed in the tool's unload path;
successful CUDA/NVTX/MPI traces and the separately documented optional pixel
observer must not be described as a successful native Nsight OpenGL trace.
Do not patch the preserved baseline installation.
