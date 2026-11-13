# Multi-backend qualification and fluid integration (MB0--MB3)

This is a toolchain probe, not an alternative TGV solver. Build the
`astr_backend_mb0` target through the repository root CMake project with
`ASTR_BUILD_GPU_BACKEND_PROBES=ON` and `ASTR_WITH_CUDA=OFF`. The latter disables
CUDA Fortran; it does not disable the probe's CUDA C++ translation units.

The Fortran main uses ISO_C_BINDING, MPI scalar reductions, and Fortran HDF5.
`qualification.cu` launches kernels calling non-inline device functions defined
only in `qualification_device.cu`. CUDA separable compilation and device symbol
resolution are required. The GNU and NVHPC main-program startup conventions are
handled by the target, without introducing a C++ replacement main.

The test uses 1031 and 65537 values, a dyadic transform, a periodic sixth-order
derivative and a 512-thread shared-memory reduction. Two non-default streams
exchange an event dependency before derivative/reduction and pinned-host copies.
The transform and unchanged input are exact checks; derivative and reduction
use scaled error <=2e-10 against independent Fortran calculations. One HDF5
summary is written, closed, reopened and compared exactly. Explicitly provide
its output path as the executable's first argument. All MPI ranks must see the
same node's devices in this probe; rank i uses device i. Multi-node device
assignment belongs to the production runtime, not this test.

## Native CUDA

Use the NVHPC SDK's `compilers/bin/nvcc` driver alongside its matching
Fortran/MPI/HDF5 environment. On the present local system the raw
`cuda/13.1/bin/nvcc` fails compiler identification against glibc's rsqrt
declarations; the installed SDK compiler driver used by existing ASTR builds
handles that environment. No system header or SDK was modified.

The probe is launched with the matching MPI launcher, including for NP=1.
The local pinned-host test explicitly selects Open MPI `pml=ob1`,
`btl=self,vader,tcp`, `osc=pt2pt`, `coll=^cuda,hcoll,ucc` and
`opal_cuda_support=0`. This does not test GPU-aware MPI. Without the OSC setting,
UCX's MPI initialization calls cuCtxSetFlags on an invalid CUDA context under
Compute Sanitizer before the probe's first kernel; keep that observation
separate from the kernel checks.

## DTK GPUfusion

`build_gpufusion.sh` takes seven explicit arguments: work directory, installed
DTK root, MPI root, explicit HDF5 Fortran wrapper, CMake executable, DCU target (such as gfx906),
and GNU compiler root. The latter explicitly sets OMPI_CC/CXX/FC and the host
runtime search path; it does not silently change the compiler behind a wrapper.
The wrapper must expose the requested Fortran high-level libraries: the private
CMake-built HDF5 uses `bin/h5hlfc`, whereas the earlier public Autotools-built
parallel installation exposed them through `bin/h5pfc`. No wrapper fallback or
discovery loop is used.
It redirects HOME, temporary files and caches into the supplied work directory,
loads those specific vendor environments, configures the root project and
builds the complete Fortran host `astr`, `astr_backend_mb0`, the MB1 state
probe and `astr_backend_fluid`/`astr_backend_rhs`. It selects the gpufusion C++
runtime and enables BUILD_TESTING for the explicitly requested terminal snapshot;
use_gpu remains a runtime input. The script does not execute a device program or call
Slurm. `gpufusion.cmake` selects the installed CUDA-12 GPUfusion frontend,
uses shared runtime libraries and uses the vendor frontend for final linking.

Installed DTK 26.04 behavior must be measured; the local DTK 26.10 manual and
the previous DTK 22.10.1 probe are not substitutes. No device run is permitted
on a login node. Scheduler submission and queries require separate user
authorization under the project's platform rules.

## MB1 State Foundation

`astr_device_state` builds `src_gpu/device_state.cu` and the matching
ISO_C_BINDING module without `.cuf` dependencies. It owns only the FP64 q,
qrhs and qsave buffers at this stage: q has bounds -hm:im+hm in each direction;
qrhs and qsave have only 0:im physical-node bounds. Components are contiguous
planes in Fortran order. No primitive, geometry, filter, chemical or statistics
workspace is allocated by state creation. `fluid_prepare`, `fluid_rhs_prepare`
and `fluid_filter_prepare` allocate only their selected consumers' workspaces.

`astr_backend_mb1` checks unequal 7/5/3 interval counts with hm=3/5, signed-zero
preservation, all components and halo indices, physical q-to-RHS mapping,
qsave isolation, unchanged q and owner invalidation on destruction. A separate
CUDA consumer reads borrowed device views; the Fortran test never dereferences
device addresses. Its expected data are constructed independently in Fortran,
and comparisons use the FP64 bit patterns, not a floating-point tolerance.
NP=2 checks per-rank contexts, not halo communication. The subsequent fluid
checks exercise both dependency-based execution and explicit per-launch debug
synchronization. This does not complete the main solver's backend integration.

The ordinary `astr` executable still uses its existing CPU/CUDA Fortran paths.
Those remain the default when ASTR_GPU_CPP_BACKEND=none. The new C++ runtime
and its current boundary are described below; no probe result substitutes for
ordinary-main execution.

## Private HDF5 For The Modern GNU Host

The existing GNU7 HDF5 modules cannot be assumed compatible with GNU12.
`build_private_hdf5.sh` accepts work directory, GNU root, MPI root, CMake and an
explicit official HDF5 1.14.6 source archive. It installs parallel shared C,
Fortran and high-level libraries into the work directory, disables static
libraries, command-line tools, tests, examples, SZIP and automatic external downloads, and uses the existing public
zlib. It never changes a public installation or submits a scheduler task.
The GNU12/public Open MPI4.1.5/private HDF5 combination still requires its own
compiler, library-resolution and runtime evidence; the GNU7 MB0 job does not
qualify a different MPI/HDF5 stack. The new combination ran MB0 NP=2 and MB1
NP=1/2 in job 127800728. MB1 passed its bit checks but exposed missing 512-thread
launch bounds in test consumers; the bounds have been corrected, with DCU
runtime confirmation in job 127808864: the bit checks passed without the
256/512-thread launch warning.

## MB2 Fluid Components

`fluid_explicit` provides the temperature-based five-variable conversions, D6
gradients with the full nine-entry metric projection, three directional CFL
maxima, standard central convection, Sutherland nodal viscous flux/divergence,
explicit tenth-order filtering and RK3 save/update. The baseline filter uses
an optional full-five-component ping-pong buffer, not a scalar workspace.
All device kernels state their 512-thread launch bound; CUDA C++ contraction
is disabled with `--fmad=false`. The shared reductions do not assume warp32.

`device_halo` supplies **local periodic axes only**. Endpoint averaging and
halo-only refresh are separate operations; transverse indices are physical.
`fluid_refresh_primitives` reconstructs physical nodes plus face halos and
never consumes undefined double/triple-outside corners. `fluid_step.F90`
composes the original filtered RK stages through an explicit exchange callback,
so no host field copy is needed inside a stage. The ordinary runtime now supplies
the distributed pinned-host callback described below.

`astr_backend_fluid` checks a modified-Fourier-symbol manufactured gradient,
the original Fortran `cfl_spectrum` function and frozen conversion states.
`astr_backend_rhs` reads frozen physical geometry, pre-RHS and primitive
snapshots, then checks central convection, full viscous RHS and RK updates.
Its stage-one case also checks initial filtering and a four-full-step coupled
trajectory against frozen checkpoints at steps 2/4. Both use debug=0/1.
The halo test independently constructs the original endpoint-average/final
refresh contract in Fortran and compares every bit, including unchanged corners.

Run `run_fluid_components.py` with explicit `--binary`, `--rhs-binary`,
`--mpiexec`, `--reference` and a new `--output` directory. The reference is the
frozen `evidence/standard_tgv_explicit10` directory. Python only launches the
Fortran probes and records their errors; it does not implement a second flow
solver. The input constants match the frozen reference: Pr=0.72, Sutherland
constant=110.3 K, reference temperature=273.15 K, Re=1600, Ma=0.1, dt=1e-3.
Using 110.4 K in the first test was a fixture mismatch, corrected without
changing the device operator or tolerance; the failed evidence is retained.

The initial component tests preceded ordinary-main wiring. Its subsequent
integration is described below. Component or test-driver results alone do not
complete the MB0--MB3 goal.

`run_mb2_gnu12.slurm` ran the second DCU component batch as job 127808864
(COMPLETED, 0:0, 13 seconds). All 342 scalar comparisons were present;
max scaled operator error was 2.4867315238236154e-14 and the four-step
trajectory error was 1.1102229888203125e-15. This is not distributed TGV
communication. One PMIx dstore NOT-FOUND diagnostic in the batch log remains
unexplained despite successful application exits; no runtime workaround was
added. New scheduler operations still require their own scoped authorization.

## Ordinary ASTR C++ Runtime

Select ASTR_GPU_CPP_BACKEND=cuda or gpufusion through the root CMake project,
with ASTR_WITH_CUDA=OFF. These select one C++ implementation, not an input
case; use_gpu remains an input-file choice. The shared numerical library is
defined in src_gpu/CMakeLists.txt even when qualification probes are disabled.
For GPUfusion, provide the explicit toolchain and gfx target. The build helper
uses plain selected GNU C/C++ compilers plus MPI_C_COMPILER so FindMPI exports
header paths to CUDA units, not only implicitly to an MPI C wrapper.

fluid_runtime.F90 owns the opaque context and calls existing Fortran topology,
input/run controls and statout. It retains the first-stage filtered statistics
phase and the original TGV normalizations. State is uploaded once; only scalars
are downloaded during ordinary time advancement. BUILD_TESTING enables an
explicit final-q snapshot when ASTR_VALIDATION_RHS_STEP selects the terminal
step. This is a test readback, not a production checkpoint implementation.

The implemented flow path is FP64, Cartesian periodic five-variable TGV,
standard 643e convection/nodal viscosity, explicit10/full workspace and RK3.
ASTR_GPU_SYNC_MODE supports dependency and explicit. CPU use_gpu=f still takes
the original Fortran path. C++ crashfix, mean fields, production field/restart
publishers and in-situ providers are not connected; no stale CPU proxy is used.

run_cpp_tgv_gate.py runs the ordinary executable and compares fields, scalar
outputs and timestep/CFL records with independent frozen CPU/CUDA Fortran
binaries. Default 16-cubed/4-step evidence can be reused. --fresh-reference is
required for changed --grid/--steps/--dt; references are then generated by the
frozen binaries, not by a mutable CPU build. The 128-cubed,20-step matrix uses
dt=2e-4 and automatic initial_dt=2e-4,max_dt=1e-3,growth=1.1,CFL=0.5.
Checkpoints are disabled for the new GPU main; reference runs write only their
final checkpoint. A --sanitizer option instruments the C++ run, not the frozen
reference. All integration is performed by ASTR, not Python.

## Pinned Halo Component

halo_transport creates fixed device/pinned send/receive slots once, a private
duplicated MPI communicator when distributed axes exist, and fixed tags 0/1.
Main-thread MPI owns requests; no communication thread is created. Local axes
use explicit local periodic operations. The caller selects endpoint averaging
or halo-only refresh; halo-only packets contain exactly hm layers, and only
reconciliation packets add endpoints. Requests complete before slots are reused.

astr_backend_halo and run_halo_components.py test NP=1 plus NP=2 x/y/z, hm=3/5
and both sync modes. The independent Fortran reference uses array sections and
the original ascending endpoint packet convention. Every physical/halo/corner
bit is compared after each averaging/refresh axis. q device addresses stay
stable. All 28 configuration checks and two-rank memcheck pass locally.
The ordinary runtime now uses this transport with the actual Fortran neighbors
and topology. run_cpp_multi_tgv_gate.py checks the real main at NP=1 and NP=2
x/y/z, including each RK stage's q/primitive coherence and all 15 exchanges per
stage (filter, reconciliation, final halo refresh and viscous-flux exchange).
Solver-owned origin/layout records locate each field within the independent
frozen global reference. All hm layers and reconciled endpoints are checked
bitwise. Observation copies have a separate six-component Fortran oracle,
including signed zero, and the final halo observation is also checked against
an independent full-q download.

The local eight-run matrix passed 315 rank-exchange observations, with maximum
terminal scaled field error 1.2666454347843286e-15. Trace on/off and explicit/
dependency terminal physical fields are bitwise equal. The actual NP=2 x main
also passed memcheck and initcheck on both ranks with no errors or leaks.
These observations require BUILD_TESTING plus explicit selected validation steps;
normal advancement does not download full fields. --include-quad adds NP=4
2x2x1 trace/plain checks for actual four-device environments; it does not enable
rank oversubscription. The same source passed the actual DCU ordinary-main
NP=1, NP=2 x/y/z and NP=4 2x2x1 matrix in job 127827004. All ten runs were
present, using one rank per device; 495 rank-exchange observations passed.
Maximum terminal scaled field error was 1.1102230246251565e-15 and stage/cache
error 1.3253106061101023e-15. All five observation/sync isolation pairs were
bitwise equal. See the migration progress document, section 7.2, for raw files.
Scheduler submission/query still needs its own user authorization.

run_cpp_multi_gnu12.slurm runs the prepared finite case manifest without Python
or a host flow integrator. Its MPI stdin is /dev/null: inheriting the manifest
stdin consumed the remaining entries in the first batch (127826437), which
only ran NP=1 despite a successful job exit. The original record is preserved;
only the corrected full batch counts as multi-device evidence. PMIx dstore
NOT-FOUND diagnostics remain recorded separately from successful application
exits and numerical comparisons; no MPI workaround was added.

The ordinary single-rank GPUfusion main ran as job 127818756: 128 cubed,
20 fixed steps (dt=2e-4) and 20 automatic steps (target CFL=0.5,
initial_dt=2e-4,max_dt=1e-3,growth=1.1). Production field/checkpoint I/O was off;
only explicit test terminal snapshots were copied. The downloaded fields,
scalars and timestep/CFL logs passed the frozen-reference comparison, with
max field error 7.721471032007265e-15. See compare_cpp_main_results.py and
the migration progress document for paths and per-variable errors. The full
26-second batch duration includes initialization and I/O, not pure RK timing.
