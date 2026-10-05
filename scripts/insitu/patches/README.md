# ParaView Streamline Coordinate Precision

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
