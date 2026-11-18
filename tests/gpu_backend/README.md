# Multi-backend qualification and fluid integration (MB0--MB8)

## Current Ordinary Build Entry

This directory contains both component probes and gates for the ordinary
Fortran `astr` program backed by the shared C++ device library. The phase
entries below record successive increments, not one current capability list;
later results supersede earlier "not connected" notes within their stated
scope. See the [migration progress](../../documents/ASTR_MULTI_BACKEND_MIGRATION_PROGRESS.md),
especially10.169--10.175, for the latest corrections, DCU receipts and ordinary
test-disabled local build/continuation evidence.
No code-generation or alternative Python CFD solver is involved.

Configure from the repository root in a new build directory. Select a matching
Fortran/MPI, C++/CUDA and HDF5 Fortran/HL environment; the variables below must
name those installed compilers/wrapper, not defaults selected by this example:

```bash
cmake -S . -B build_cpp_cuda \
  -DCMAKE_Fortran_COMPILER="$FC_WRAPPER" \
  -DCMAKE_CXX_COMPILER="$HOST_CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DHDF5_Fortran_COMPILER_EXECUTABLE="$HDF5_WRAPPER" \
  -DASTR_GPU_CPP_BACKEND=cuda -DASTR_WITH_CUDA=OFF \
  -DASTR_NUMERICAL_PRECISION=fp64 \
  -DCHEMISTRY=OFF -DASTR_WITH_AIR5_CHEMISTRY=OFF \
  -DASTR_WITH_CATALYST=OFF -DASTR_BUILD_GPU_BACKEND_PROBES=OFF \
  -DBUILD_TESTING=OFF
cmake --build build_cpp_cuda --target astr --parallel 4
```

`build_cpp_cuda/bin/astr` still reads the normal named case/run/output files;
`use_gpu=t` selects device advancement and `use_gpu=f` selects CPU advancement
in the same executable. `ASTR_GPU_CPP_BACKEND=none` remains the default.
`ASTR_WITH_CUDA=ON` selects the original CUDA Fortran implementation and cannot
be combined with `cuda` or `gpufusion`. Turning CUDA Fortran off therefore does
not turn the new CUDA C++ implementation off. Qualification probes are optional;
enable `BUILD_TESTING` only when the requested test observations need it.

For DCU, load the selected DTK CUDA/GPUfusion environment and use the same root
project with `ASTR_GPU_CPP_BACKEND=gpufusion`,
`CMAKE_TOOLCHAIN_FILE=tests/gpu_backend/gpufusion.cmake` and an explicit
`CMAKE_CUDA_FLAGS=--offload-arch=<DCU target>`. Select GNU/MPI/HDF5 built with
compatible ABIs, including an explicit `MPI_C_COMPILER` when plain GNU C/C++
compilers are used. Do not pass NVIDIA compute capabilities as the DCU target.
`build_gpufusion.sh` below is the separate probe-enabled qualification build,
not a requirement to build probes into every ordinary program.

Current MB6/MB7 coverage is nonreacting five-variable flow in the documented
format/boundary/precision subsets; it is not the full old-backend capability
list. Native FP32 uses a separate build and corresponding existing precision
gates. New-backend AIR5 is MB8; DCU Catalyst/ParaView rendering is MB9. Native
NVIDIA device products additionally require `ASTR_WITH_CATALYST`,
`ASTR_WITH_INSITU_DEVICE` and `ASTR_WITH_INSITU_DEVICE_RENDERING`, with matching
Catalyst/ParaView dependencies as detailed in the consumer sections below.
These build commands do not submit a scheduler task or establish production
physics/performance acceptance.

## MB8 AIR5 Boundary Target Components

`astr_backend_air5_boundary_targets` tests resident fixed left/right states,
normal-shock extrapolation/pressure branches, and the local j-fastest profile
inlet. Build this optional target through root CMake. The
`run_air5_boundary_targets.py` driver takes a frozen directory from
`run_air5_boundary_components.py`, the candidate, MPI launcher, Memcheck and a
new output directory. It reuses the original CPU/CUDA Fortran pressure outputs,
checks copied/untouched q words and carry exactly, and compares pressure
construction with the original component error budgets.

The local matrix is NP=1, NP=2 x/y/z, and NP=4 (2x2x1), explicitly sharing two
GPUs for four ranks. It includes explicit target updates and host-input lifetime
checks. MPI supplies topology and scalar observations here, not halo traffic.
These q-only kernels do not clear carry or refresh primitives/halo; those belong
to the original complete boundary scheduler, which is not yet connected for
ordinary C++ AIR5. Component success does not imply NSCBC, G2, production or DCU
acceptance. See progress sections 10.191--10.192 for evidence and pending
independent numerical decisions.

`run_air5_hbl_boundary_components.py` additionally uses
`astr_backend_air5_boundary` for original HBL outflow and similarity-top
mathematics, real face writes, incident target selection, dynamic-top ghost
copying and the separate carry-reset masks. Original outflow blend FMA is
explicit; global contraction remains off. The driver can explicitly reuse a
previous frozen run with `--reference-run`, checking input and original source
identity before reuse. Failure statuses and untouched q are checked; differing
failed-helper scratch arrays are retained and separately reported, not passed
as equivalent state data. This is not a complete characteristic RHS or ordinary
AIR5 boundary scheduler. See progress 10.193.

`astr_backend_air5_characteristic` and `run_air5_characteristic_components.py`
freeze the shared Fortran characteristic operators at identical thermodynamic
inputs and compare six eleven-component outputs and five validity flags.
Complex-step differentiation of the EOS flux, uniform zero and outgoing-only
limits are separate checks. The corresponding 461-input local receipt is in
progress 10.194.

The distinct `astr_backend_air5_characteristic_field` gate starts from q and
observes the original pre-limiter boundary assembly. Its original CPU/CUDA
Fortran reference comparison currently fails; the C++ field matrix did not run.
Reuse those frozen records when diagnosing rather than rerunning them without
a numerical decision. `run_air5_characteristic_fields.py --reference-run ...
--diagnose-rows ...` explicitly instruments selected original rows and requires
unchanged RHS words; it is diagnosis only. A constant target supplier and a
count-only limiter handoff isolate normal assembly, not a substitute full-flow
solver. Near-sonic classification and all-incoming cancellation need human
review; fixed-temperature V-T approval does not waive these failures. The
independent uniform Fortran ABI/memory check is not this field acceptance gate.

`astr_backend_air5_boundary_stage` and `run_air5_boundary_stages.py` exercise
the Fortran/MPI state-boundary coordinators with frozen original GPU kernels
and original apply order. Eleven representative configurations cover HBL
similarity/incident selection/dynamic ghost, cache/compensation/debug controls,
and fixed/normal extrapolation/pressure-state outlets. They are manufactured
state fixtures, not physical shock or complete HBL cases. The local NP=1/2 x/y/z
and NP=4 (2x2x1, explicitly sharing two GPUs) receipts are in progress 10.195.

`air5_hbl_boundary_apply` returns a globally identical nonzero physics status
and failed-stage number before later faces or carry reset are submitted. The
ordinary caller must treat this as fatal, not continue the failed flow. Device
API/MPI runtime errors abort directly. `air5_postshock_boundary_apply` retains
the original normal-pressure global release and does not clear carry. Neither
coordinator rebuilds primitives or exchanges halo. The fixture explicitly
refreshes physical primitive caches after success and checks unchanged q.

Four injected failure scenarios check stage release across all ranks; full
q/carry arrays, copied targets, wall temperature/outlet pressure, and physical
cache values are compared. Read-only alias checks do not reconcile carry: its
initial interior values intentionally differ between ranks. These tests use
host scalar MPI only and do not resolve the known characteristic reference
failure, AIR5 compensated shared-node exchange, ordinary CFD/G2 or DCU gates.

`astr_backend_air5_compensation_exchange` tests the explicit per-axis AIR5
compensation component: q11/carry11 face packets update carry, existing q
exchange averages shared nodes, and TwoSum normalization follows. A final
halo-only sweep precedes physical/communication-halo primitive reconstruction.
The independent CPU source oracle is test-only; production mean/normalization
runs on the GPU and reuses transport slots. `run_air5_compensation_exchange.py`
covers pinned/device MPI, NP=1/2 x/y/z and NP=4 xy with two-GPU test sharing.
The fixture has physical x/y and periodic z. PMPI checks inspect real pointer
kinds, signatures, and live-neighbor payload words. MPI_PROC_NULL payload is
not transferred/staged and is not read as defined data by the checker.

The separate `probe_air5_periodic_compensation.py` retains a known original
all-periodic single-rank fast-controller discrepancy: its three q-only kernels
leave opposite carry at a duplicated physical node. This has not been repaired.
Passing explicit component phases is not approval to silently replace that
controller or claim ordinary AIR5/G2/DCU acceptance. See progress 10.196--10.197.

## MB0 Probe Entry

The initial MB0 probe is not an alternative TGV solver. Build the
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

The 2026-11-16 DCU regression receipts are
`results/dcu_mb67_20261116/summary.json` (job 128170436) and
`results/dcu_mb67_extended_20261116/summary.json` (job 128172334).
They cover bounded FP64/FP32 compute, MPI, restart and numerical archives,
not DCU rendering or production/performance acceptance. The host-only
`astr_backend_compare field` retains the FP64 `2e-10` gate; explicit
`field32` requires actual four-byte references/snapshots and reports D10
errors without applying an FP64 limit. Both modes reject malformed or
nonfinite field data. No automatic precision conversion or backend fallback
is selected.

`run_cpp_selected_gate.py --precision fp32` likewise records native
selected-time errors, while exact restart and observation-isolation checks
remain bytewise. Its expanded fluid-only fixture is explicitly selected
with `--expanded --without-thermal-gradients`; the existing shared FP64-only
admission for the newer thermal diagnostics is unchanged. With no precision
option the original FP64 limits still apply. Native local evidence and the
retained thermal-admission failure are indexed by
`results/multi_backend_mb7_native_selected_20261116/summary.json`.

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
probe, `astr_backend_fluid`/`astr_backend_rhs` and the later component targets
listed in the script. It selects the gpufusion C++
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
ISO_C_BINDING module without `.cuf` dependencies. It owns only build-kind q,
qrhs and qsave buffers at this stage: q has bounds -hm:im+hm in each direction;
qrhs and qsave have only 0:im physical-node bounds. Components are contiguous
planes in Fortran order. No primitive, geometry, filter, chemical or statistics
workspace is allocated by state creation. `fluid_prepare`, `fluid_rhs_prepare`
and `fluid_filter_prepare` allocate only their selected consumers' workspaces.
The FP64 default remains the qualified ordinary flow path. Later native FP32
state/halo component evidence is described below; it does not qualify FP32 flow.

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
explicit tenth-order filtering and RK3 save/update. The filter supports a
single-component 3-D workspace (default) or full-five-component ping-pong.
All device kernels state their 512-thread launch bound; CUDA C++ contraction
is disabled with `--fmad=false`. The shared reductions do not assume warp32.

`device_halo` supplies **local periodic axes only**. Endpoint averaging and
halo-only refresh are separate operations; transverse indices are physical.
`fluid_refresh_primitives` reconstructs physical nodes plus face halos and
never consumes undefined double/triple-outside corners. `fluid_step.F90`
composes the original filtered RK stages through an explicit exchange callback,
so no host field copy is needed inside a stage. The ordinary runtime now supplies
the distributed pinned/device callbacks described below.

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

The initial path is FP64 static periodic five-variable TGV. Native increments
now include representative low-order physical TGV/channel orchestration with
standard/KEEP 643e/643c, nodal/ME6 viscosity, explicit10/compact10 and RK3;
bc31 plus the existing RTI gravity source have separate main evidence below. See
Ordinary Physical-Boundary Main below for actual tested configurations. These
increments do not extend the earlier DCU execution evidence automatically.
ASTR_GPU_FILTER_WORKSPACE selects scalar (default, one 3-D work field) or full
(five work components). Scalar/full keep the same per-component x/y/z formula
and terminal physical bits in the native eight-case matrix. The allocated
workspace is reported, and scalar uses exactly one fifth of the full bytes.
ASTR_GPU_SYNC_MODE supports dependency and explicit. CPU use_gpu=f still takes
the original Fortran path. C++ crashfix and solver-sample means remain missing;
native velocity/selected-time statistics and device in-situ consumers are
described below. The initial MB4 output/restart bridge is described below;
it explicitly publishes the device state, not a stale CPU proxy.

run_cpp_tgv_gate.py runs the ordinary executable and compares fields, scalar
outputs and timestep/CFL records with independent frozen CPU/CUDA Fortran
binaries. Default 16-cubed/4-step evidence can be reused. --fresh-reference is
required for changed --grid/--steps/--dt; references are then generated by the
frozen binaries, not by a mutable CPU build. The 128-cubed,20-step matrix uses
dt=2e-4 and automatic initial_dt=2e-4,max_dt=1e-3,growth=1.1,CFL=0.5.
Checkpoints are disabled for the new GPU main; reference runs write only their
final checkpoint. A --sanitizer option instruments the C++ run, not the frozen
reference. All integration is performed by ASTR, not Python.

## Halo Component And Split Exchange

halo_transport creates fixed send/receive slots once, a private
duplicated MPI communicator when distributed axes exist, and fixed tags 0/1.
Main-thread MPI owns requests; no communication thread is created. Local axes
use explicit local periodic operations. The caller selects endpoint averaging
or halo-only refresh; halo-only packets contain exactly hm layers, and only
reconciliation packets add endpoints. ASTR_GPU_HALO_TRANSPORT=pinned (default)
uses four pinned and four device slots; device uses only the four device slots
and passes their actual pointers to GPU-aware MPI. There is no automatic fallback.
begin posts real nonblocking requests without Waitall; end drives progress on
the caller thread and queues the unpack. The next begin completes preceding
unpack work before MPI can overwrite reused receive slots. In-flight destroy,
nested begin and mismatched/duplicate end are tested. Stress-x exchange can run
while the independent convection kernel executes; other exchanges retain their
existing ordering. This is not a claim of complete communication hiding.

astr_backend_halo and run_halo_components.py test NP=1 plus NP=2 x/y/z, hm=3/5
and both sync modes. The independent Fortran reference uses array sections and
the original ascending endpoint packet convention. Every physical/halo/corner
bit is compared after each averaging/refresh axis. q device addresses stay
stable. All 28 configuration checks and two-rank memcheck pass locally.
The ordinary runtime now uses this transport with the actual Fortran neighbors
and topology. run_cpp_multi_tgv_gate.py checks the real main at NP=1 and NP=2
x/y/z, including each RK stage's q/primitive coherence and 15 full/19 scalar exchanges per
stage (filter, reconciliation, final halo refresh and viscous-flux exchange).
--workspace scalar|full selects storage; --transport pinned|device selects MPI
buffers in both the main driver and the halo component driver. Native device
tests use UCX PML and CUDA awareness, with an explicitly supplied UCX module
directory and UCX_MEMTYPE_CACHE=n on this installation. Without that setting,
the static-CUDA application's device buffers were misclassified into CMA and
failed with process_vm_readv Bad address. The failed records are retained; this
configuration is not silently applied by ASTR and is not evidence for DCU MPI.
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

## MB4 Checkpoints And Numerical Archives (In Progress)

The ordinary C++ main now publishes q and, for exact checkpoints, the stored
primitive caches through the existing HDF5 protocol. Restoration uploads the
restored conservative/cached state and geometry; exact restoration does not
replace caches with reconstructed values. Lean checkpoint writes skip primitive
readback, preserving the existing physical-q-only role and resource contract.
All transfers occur at explicit output/restore events, not every RK stage.

Completed-step volume and JSONL instantaneous numerical-slice archives consume
an explicit host publication of primitives. The host mirror is invalidated at
RK entry and reused by products at the same complete state; download counters
include these transfers. Derivatives/packing use the existing private host
archive routines. This does not complete device in-situ/rendering providers,
mean fields or adaptive dense-output monitoring.
The retired &slices input is not re-enabled: numerical planes use product_catalog.

run_cpp_restart_gate.py takes --binary, --reference, --mpiexec, --output and
--ranks 1|2; --sync dependency|explicit and --mode exact|nonexact|all select the
representative path, while --memcheck instruments actual MPI solver processes.
--workspace default checks the actual baseline default and its control identity.
It checks 4 continuous steps against restoration from the same trajectory's
step-2 checkpoint, all exact dataset words, lean payloads, frozen-reference
physical fields and output-on/off physical-q isolation.
--automatic-reference points at the existing cpu_automatic/old_gpu_automatic
reference pair from the frozen binaries and requires --mode exact. It checks
restored dt history and complete-step records without comparing different
physical times to the fixed-dt reference. The NP=2 x automatic test passed;
non-exact automatic restart starts a new controller history and needs its own
reference rather than a forced exact-trajectory assertion.

run_cpp_archive_gate.py takes the same four paths and --ranks 1|2|4 (NP=4 uses
2x2x1). It generates
frozen CPU/CUDA Fortran references, then compares two complete-step volumes
and 27 JSONL slice frames including derived quantities. --analyze-only reuses
the generated artifacts without advancing flow again. Both local NP=1 and
NP=2 x passed, but these bounded tests do not close the actual DCU restart
matrix. Native four-way CPU/old-GPU/
new-C++ non-exact transfers and NP=1/2 repartition have since passed.
run_cpp_restart_gate.py --restore-source selects an external step-2 batch; --backend selects which
ordinary frozen/new ASTR reads it. --keep 1|2 checks generated-batch retention.
--automatic --mode nonexact compares the new controller history against
independent frozen CPU/GPU restorations from the same batch, rather than a
continuous trajectory ending at another physical time. --transport device
also exercises exact restoration with actual device-buffer exchanges.
--no-test-observations runs restoration without test snapshots and is used
with the ordinary BUILD_TESTING=OFF binary. It does not claim the omitted
output-on/off snapshot comparison.

astr_backend_compare is a CPU-only C++17/HDF5 test utility for compute images
without NumPy/h5py. `field REFERENCE_H5 RUN_DIRECTORY` checks all rank layouts
and five physical components of the bounded step-4 TGV snapshots at scaled
error <=2e-10. `exact STATE_A STATE_B` compares every root numeric dataset's
shape/type/words in the current checkpoint protocol. It never advances a flow
or runs as a production checkpoint check. Local positive cases match the Python
analysis; a perturbed physical field and a one-ULP checkpoint change are both
rejected. All MPI benchmark invocations use /dev/null stdin so a launcher cannot
consume a surrounding case manifest.

## Private ROCm MPI And Performance Work

build_rocm_mpi.sh builds pinned UCX1.18.1/OpenMPI4.1.5 versions below an explicit
work root, with GNU12 and the supplied DTK/ROCt paths. It requires HAVE_ROCM and
HAVE_HIP before compilation and records library/configuration evidence. Public
SDK installations are not modified. This login-node build is not a GPU-aware
communication test; running or querying DCU Slurm jobs needs separate approval.

run_cpp_performance.py disables large output and test field observations,
retains per-step scalar statistics, and measures five five-step windows after
five warmup steps. It takes the maximum accumulated rank time per window, then
reports median/best seconds per step and strong-scaling metrics. --old-binary
compares the independent frozen CUDA Fortran binary on the same machine;
the common pinned baseline requires --sync explicit because the old pinned
implementation does not admit dependency sync. C++ dependency/device results
are separate configurations, not interchangeable old/new code speedups.
ASTR_GPU_RK_TIMING=1 produces completed-RK timings excluding scalar-statistics
work; ASTR_CPP_HALO_TIMING adds host send-ready/post/Waitall/unpack-enqueue
durations, sent bytes and application staging bytes. Host intervals are not
nonoverlapping device-time components, and zero application staging bytes do
not prove the MPI library avoids its own host staging.

prepare_cpp_mb45_batch.py prepares inputs only; --wire-followup reuses completed
component evidence and retains 14 representative runs plus 10 timing runs.
--archive-followup instead prepares only four 16-cubed, four-step runs:
device NP=2 x and pinned NP=4 xy, each with archive output on/off. It does not
repeat completed communication components, restart cases or performance windows.
run_cpp_mb45_rocm.slurm runs the manifests with stdin closed and checks actual
frame counts for io_archive using Bash, not merely the solver exit code. Submission
and even scheduler queries still require separate, scoped user approval.
analyze_cpp_mb45_results.py reads downloaded outputs without running a solver;
archive datasets and product states also require run_cpp_archive_gate.py --analyze-only
on the analysis machine; compute nodes need no Python interpreter.

DCU jobs 127898190 and 127905611 provide actual pinned/device NP=1/2/4 data.
The latter completed checkpoints and five-window timings, but all nine numerical
slice products were disabled. GNU CPU-only, max_steps=0 reproduced successful
allocation followed by a false agreement flag. index_slice_provider now returns
the collective flag from a logical function instead of aliasing an input/output
actual. Nine initial slices match the frozen NVHPC CPU to scaled error 6.67e-16;
native NP=2 volumes/27 slices and output isolation pass. Actual DCU slice output
passed NP=2 in job127915636, including all 27 frames and output isolation. That
job stopped afterward because the old test script required python3 on compute
nodes; the Bash frame-count check removes that dependency. Job127917357 then
exposed inherited NP<=2 derived-volume admission before any RK step. Only the
C++ GPU host-publication path now drops that rank-count restriction, retaining
case/grid/operator/local-stencil checks and the old CPU/CUDA Fortran restrictions.
The revised candidate builds and native NP=2 archive comparisons pass. After
separate approval, job127919260 completed NP=4 output: two volumes and 27 slice
frames match frozen CPU/CUDA Fortran, all nine products remain active, and
terminal physical-q words equal the saved same-topology output-off run.
MB4 is closed within the first-stage periodic FP64 TGV scope. The final reports
are archive_127915636_np2/report.json and archive_127919260_np4/report.json below
results/multi_backend_mb45_20261113; prior failed evidence is retained.

checkpoint_logical_probe checks supported metadata flags: fixed-width int32
0/1 writes and integer-to-logical decoding (including existing NVHPC -1 true).
No floating-point state, record width or retired format changes. Actual exact
restoration checks all 15 datasets; a real checkpoint-switch change remains an
expected rejection. Private UCX may log a NULL pointer-info diagnostic around
startup/shutdown or empty HDF5 selections. Representative measured RK windows
contain none; no external-library patch or log suppression is applied.

## 32-GiB DCU Scaling Campaign

prepare_cpp_scaling32.py prepares ten configurations with three independent
runs each: 384-cubed strong scaling, two pinned comparisons, and fixed-local-load
weak scaling. Every run uses five warmup steps and ten measured steps. The
rectangular-grid preparation and the 5+10 accounting were checked using a small
native NP=2 run with no field/checkpoint output. Analyze downloaded results using
--output CAMPAIGN --analyze-only; the report preserves the three raw samples,
median/best/maximum and same-backend strong/weak scaling efficiencies.

run_cpp_scaling32_rocm.slurm uses Bash only on compute nodes. Device/HDF5,
four-rank halo and small frozen-reference checks precede timing. No GPU runtime
was exercised on a login node. The initial 8-CPU, whole-node-memory request was
rejected at submission by kshdnormal02's memory-per-CPU policy; no job was created.
An authorized policy query found 32 CPUs/node and 3569 MiB/CPU. The user then
approved 32 CPUs and an explicit 110 GiB request; the resource directives have
been updated accordingly, with at most four MPI ranks and OMP_NUM_THREADS=1.
Submission state is recorded separately; see documents/ASTR_DCU32_TGV_SCALING_PLAN.md.

## MB6 Compact Line Component

Root target astr_backend_compact_line builds the ordinary Fortran/C ABI and
CUDA C++ line solver without CUDA Fortran. It reuses CPU constant-factor
preprocessing but solves RHS data entirely on device using the existing
two-level PCR, cyclic rank-two correction and reconstruction. A plan borrows
the parent device/stream, owns reusable buffers, and exposes matrix views;
destroying a parent with outstanding plans is rejected without losing its handle.
Explicit host transfers are component input/observations, not a production
per-kernel host mirror. The local-line component does not implement rank coupling.

Build via `cmake --build build_backend_cpp_native --target astr_backend_compact_line`;
run `build_backend_cpp_native/bin/astr_backend_compact_line`. CPP_COMPACT_PASS
reports 252 cases spanning lengths 2..513, tile widths 4/128, cyclic/noncyclic
matrices, RHS counts 1/5/11 and both sync modes. Checks use manufactured solutions,
full-matrix residual <=1e-12 and independent pivoted LU scaled difference <=2e-10.
RHS bits, sync-mode bits and borrowed addresses are checked separately. Eleven
RHS columns are a linear-algebra test, not AIR5 flow support.

Native maximum scaled difference/residual are 6.665e-15/8.292e-16. Memcheck,
initcheck and shared-memory racecheck report no errors; memcheck reports no leaks.
The root ordinary astr build and eight existing explicit TGV cases also pass.
Evidence: results/multi_backend_mb6_20261113/native. GNU12/DTK26.04 compilation
and dependency preflight pass; actual DCU execution still needs scoped approval.
No distributed or main-program capability is claimed from the local component alone.

## MB6 Distributed And Main Compact Paths

astr_backend_compact_distributed exercises the global interface PCR, cyclic
correction and shared-node restoration against independent global LU and a
manufactured solution. run_compact_distributed.py supplies the launcher and
explicit pinned/device-aware selection; MPI/UCX environment is supplied by the
caller. --compare DIR requires all solved/restored binary records to match the
other transport bitwise. Native NP=1/2 x/y/z pass 216 cases per rank and mode;
max scaled solution error/residual are 3.632e-15/8.905e-16. Device mode reports
no application host staging and allocates no pinned slots. A four-rank line
uses at least nine intervals so each local PCR core has two unknowns.

compact_operator.cu/.F90 connects the ordinary device fields to derivative and
filter plans without full-field host packing. Compact filtering solves each
captured line batch before overwriting that disjoint line/component, requiring
no additional full qwork field. Scalar/full select one/five components per batch.
ASTR_GPU_COMPACT_TRANSPORT=pinned|device-aware and
ASTR_GPU_COMPACT_SYNC=explicit|dependency retain their independent selections;
the latter defaults to explicit. ASTR_GPU_COMPACT_BATCH_RHS defaults to 65535.
General halo transport and sync remain independent. Plans are destroyed before
the parent context. Setup uses CPU constant factors, not CPU RHS solves.

run_compact_tgv_gate.py runs matched immutable CPU/CUDA Fortran references for
compact_explicit, explicit_compact and compact_compact. Native 16-cubed four-step
TGV covers NP=1/2 x/y/z, representative independent filters, both transport modes,
scalar/full and sync choices. Short-window stage fields/caches and convective/full
RHS satisfy scaled difference <=2e-10. Representative NP=2 main memcheck reports
zero errors/leaks. run_cpp_restart_gate.py --compact-form FORM
--compact-reference DIR checks the same ordinary checkpoint protocol: NP=2 exact
continuation preserves all 15 datasets bitwise, non-exact continuation and
output isolation pass. A native single-rank automatic non-exact restore is
also compared with both frozen implementations at matching step/time.
Native batch_rhs=127 covers nonzero line/component offsets and a partial final
batch. The compact primitive-only archive check compares two volume frames and
27 numerical slices against both frozen implementations and preserves output-off
physical q bits; it does not admit compact postprocessing derivatives.

prepare_compact_mb6_batch.py prepares inputs and native-run reference states
for DCU; it never submits a job. run_compact_mb6_rocm.sh is a compute-allocation
body, not a submission command: it checks 12 distributed configurations and
18 ordinary four-step runs including representative restart/primitive archives.
DTK26.04/GNU12/gfx906 build, device link and runtime-library preflight pass.
Actual DCU compact execution still requires separate authorization.
KEEP/ME6, low precision and MB7 remain incomplete.

## Native Periodic KEEP And ME6

keep_operator.cu implements the original arithmetic-average pair flux and the
existing 294-term,85-basis compact expansion. The constant tables are uploaded
once; actual RHS data and the global compact solves remain on device.
astr_backend_keep checks rotated x/y/z manufactured fields at 32/64/128 nodes,
sixth-order convergence, conservative sums and kinetic/pressure exchange.

The first combined compact-filter run exposed NVHPC host-code writes 28 bytes
before a keep_split_plan object. astr_backend_keep_host_guard reproduces the
exact pattern with preceding/following sentinels; an optional output filename
records the full table. The approved CPU fix replaces default construction with
complete member initialization and intent(inout). Before/after NVHPC and fixed
GNU12 tables match bitwise (294 terms,85 bases); no flux or weight was changed.
Matched corrected CPU/CUDA Fortran references are frozen under
results/multi_backend_mb6_20261113/reference_fixed_keep, with their source delta
and build provenance. The original 0d0f367 archive remains unchanged.

me6opti_operator.cu/.F90 implements the existing mapped G2 operator and its
private extended halos: y includes x ghosts, then z includes x/y ghosts; no
shared-node averaging is added. The ordinary halo backend chooses pinned/device
transport, independently of compact interface transport. Fixed buffers and two
derived mu/kappa fields are allocated only for enabled ME6. Unused nodal stress/
heat-flux fields are not allocated in the periodic integrated-ME6 path.
astr_backend_me6 compares same-state orthogonal/oblique mapped cases, variable
Sutherland properties and a checkerboard field; density RHS and sync bits are
checked. It does not claim physical-boundary admission from prescribed patch halos.

The matched 16-cubed/4-step native KEEP+ME6 compact matrix passes NP=1/2 x/y/z;
max scaled field/cache and RHS differences are 4.197e-14 and 1.570e-12.
Representative device/full/dependency and pinned/scalar/explicit q bits agree.
ME6 allocation/application-traffic records check the expected reusable buffers
and zero host staging in device mode, not MPI's internal mechanism or peak usage.
NP=2 y exact continuation preserves all 15 HDF5 datasets bitwise and ten memcheck
logs report zero errors/leaks across output and continuation choices.

run_compact_tgv_gate.py accepts --convection-form and --viscous-form, as well as
--reference-only for matched frozen-implementation runs. The restart driver
also accepts these choices and an explicit representative --topology.
Physical walls, open boundaries, CURVE, low precision, new in-situ consumers and
actual DCU KEEP/ME6 execution remain incomplete; do not substitute native TGV
evidence for the full MB6/MB7 goal.
The same maintained source passes DTK26.04/GNU12/gfx906 compilation/device link
and runtime-library preflight for astr, astr_backend_keep and astr_backend_me6.
No login-node GPU execution or scheduler operation is implied by those records.

## First Static CURVE Integration

run_compact_tgv_gate.py --grid 32 --mapping periodic uses the existing amplitude
0.15 periodic mesh generator, with --checkpoint to compare the 13 global geometry
fields and complete-step q publication. Native KEEP-compact+ME6+compact10 passes
NP=1/2 x/y/z against freshly frozen CPU/CUDA Fortran references. Maximum terminal,
stage/cache, RHS and geometry differences are 2.421e-14,3.148e-14,3.927e-13 and
7.772e-16. These are short-window implementation checks, not long-time physics.
The restart driver accepts the same geometry choices; NP=2 y 2+2 exact recovery
preserves all 15 datasets without the external input grid, and output-on/off
physical q bits agree. Rank-specific geometry extras are checked in same-topology
restart, not numerically compared against NP=1 partition metadata.
TGV scalar diagnostics retain the CPU computational-node average, not a newly
introduced volume average. Physical-boundary CURVE and new device in-situ
consumers remain incomplete.

## Physical-End Halo Components

The transport prepare ABI now requires periodic(3). A size-one physical axis
does not wrap; a size-one periodic axis retains its local periodic operation.
Distributed MPI_PROC_NULL sides neither average endpoints nor overwrite ghosts;
they are excluded from application copy/sent-byte accounting. All callers use
the explicit interface, without a compatibility wrapper. Ordinary face halos
remain distinct from ME6's extended edge exchange.

run_halo_components.py --layout physical covers native NP=1/2 x/y/z, hm=3/5
and both sync modes. Pinned and device MPI each pass 36 complete layouts, each
with six exact full-array phase comparisons and begin/end lifetime checks.
Representative pinned NP=2 initcheck/memcheck report zero errors and no leaks.
Periodic component checks and the eight-case ordinary standard TGV matrix still
pass. The updated ABI now compiles with DTK26.04/GNU12; physical-end execution on
DCU is still pending. This component alone does not implement wall physics/RHS.

## Basic Stationary bc41 Component

boundary_state.cu/.F90 provides the existing low-order stationary isothermal
no-slip state command, without modifying RHS. It does not cover
blowing, high-order wall closure or chemistry. prepare_wall41_reference.py freezes
real three-stage 41x/y/z snapshots from corrected CPU/CUDA Fortran executables.
astr_backend_wall41 poisons only the requested face(s), restores their reference
state to <=2e-10, and checks all other physical/halo words unchanged. Its 108
native checks have maximum scaled state error 3.578e-16. The initial unused
face-RHS-zeroing helper was removed after inspecting the complete CPU stage
chain: it was not the CPU low-order wall contract. The state test now requires
all RHS words to remain unchanged.
Memcheck reports no errors/leaks; initcheck reports no errors. This is a component, not complete main-program
wall integration: physical-end operator/filter and RK wiring into the ordinary
runtime remains unfinished, and that runtime still requires periodic TGV.

## Low-Order Physical-End Gradient Component

fluid_gradient_basic_boundary explicitly receives low/high physical-end flags;
it retains the existing diff6ec 2/2/4/6-order end-to-interior closure and full
metric projection. MPI interfaces remain centered. The periodic entry retains
its dedicated kernel; neither API allocates temporary fields or calls CPU math.

prepare_wall41_reference.py --gradients freezes actual primitive, geometry and
gradient inputs from the corrected reference pair. astr_backend_boundary_gradient
passes 72 real three-stage x/y/z wall checks, with/without NaN physical ghosts,
at maximum scaled error 5.654e-16; all primitive input bits remain unchanged.
Another 256 quadratic analytic checks cover all 64 end combinations, hm=3/5,
both sync modes and full oblique varying metrics, with maximum absolute error
1.213e-15. No-end results also match the periodic entry bitwise. Memcheck and
initcheck report zero errors, with no leaked device allocation. This is not yet
main-program wall/filter/RK integration or a DCU execution result.

## KEEP Basic Physical-End Accumulation

fluid_keep_accumulate_basic accepts existing boundary types and the active body
box. It adds to, rather than clears, RHS; outside-box values are untouched.
One/two points from a physical end retain the CPU ordinary-flux second/fourth
order transition, with KEEP two-point sixth-order flux further inside. No new
wall RHS or boundary formula is inferred from that transition.

astr_backend_keep passes 256 physical-end combinations against the CPU volume
operator with NaN physical ghosts, oblique varying metrics, seeded nonzero RHS
and exact outside-box preservation. Maximum scaled error is 2.241e-16. Its prior
periodic convergence, conservation and kinetic/pressure-exchange checks still
pass. Memcheck/initcheck report zero errors and no device leak. Main-program
physical-boundary advancement and DCU execution are not covered by this probe.

## bc41 Four-Step Device Stage Chain

prepare_wall41_reference.py --steps 4 --gradients produces actual CPU/old-GPU
three-stage snapshots over four steps. astr_backend_wall41_step ROOT 4 runs
the device fluid_step chain, with physical-side q ghosts poisoned by NaN,
stationary wall state, nonperiodic fixed-slot transport, low-order gradients,
KEEP and nodal viscosity. Run the probe through the matching mpiexec, even at
NP=1, so the SDK MPI startup prefix is selected correctly.

Native x/y/z walls, both references and both sync modes pass 144 stages. The
maximum scaled q/primitive/RHS errors are 1.253e-15/1.230e-15/1.695e-15.
The probe compares prepared q/primitive caches, convective/full RHS and every
post-RK state without replacing the evolving device field. Memcheck reports
zero errors/leaks; initcheck passes after limiting observation to defined nodes
and a periodic face. The initial full-allocation primitive readback included
undefined physical ghosts; that failed observation log is retained. Production
ghost storage was not initialized or silently substituted to suppress the report.

The stage hooks apply wall state before shared/halo exchange and rebuild only
valid primitives afterwards. Nodal divergence retains CPU per-direction active
ranges, not KEEP's common active box. It does not erase tangential viscous wall
RHS after diffusion: the old low-order RK contract permits that transient stage
contribution. No ordinary-runtime wall admission, channel statistics/forcing,
wall output/restart or filtering completion is implied.

run_wall41_steps.py partitions the same immutable global reference into native
NP=1/2 x/y/z cases. For each run, all three wall directions, both trajectory
references and both sync modes advance without stage reset. Pinned and actual
device-buffer MPI each pass 1008 rank-stage observations. This covers split
normal walls and tangential wall/shared-node intersections. Maximum scaled
q/primitive/RHS errors remain 1.253e-15/1.230e-15/1.695e-15. Logs and commands
are recorded in wall_steps_pinned/device; this is not DCU execution or ordinary
runtime admission. Physical-end output must not read undefined physical ghosts.

The current ordinary astr, wall-state and NP=1/2 stage-probe sources compile/link
with DTK26.04/GNU12/gfx906 on the login node. The final stage probe resolves the
selected private MPI/HDF5, GNU12 and DTK GPUfusion/galaxyhip libraries with no
missing dependency. No login-node GPU execution or DCU wall-stage numerical pass
is implied; immutable submitted/staged run binaries were not replaced.

## Channel Scalars And Uniform Body Force

prepare_channel_components.py packs existing immutable CPU/old-GPU rk1 fields,
gradients, coordinates and flowstate values; it does not recompute the statistics
or advance either solver. astr_backend_channel ROOT [SPLIT_AXIS] runs at NP=1/2.
Native NP=1 and NP=2 x/y/z pass 168 scalar observations: the original y-weighted
mass-flux integral, lower-minus-upper wall shear and spatial RMS(w), not a time
average. Maximum scaled difference is 3.197e-16. Repeated local sums are bitwise
stable. Only the physical y-coordinate scalar field is allocated on request;
normal fluid preparation does not allocate it. No x/z coordinate copies are held.

Another 168 checks compare zero/three-component body forces to momentum and
local-velocity work increments, with zero difference, density RHS unchanged and
all input q words unchanged. Memcheck/initcheck/racecheck report zero errors,
leaks and shared hazards. Final single-y-field logs are channel_scalar_y_*.log;
earlier channel_component_np*.log belong to the initial three-coordinate layout.
These new interfaces have not yet been rebuilt/executed on DTK/DCU.

prepare_wall41_reference.py --force-fixed 1e-4 and run_wall41_steps.py with the
same option exercise the actual nonzero source through the optional post-flux,
pre-RK source_operator. Existing x-homogeneity eligibility is retained. Both
transports pass 1008 native rank-stage checks each; maximum q/primitive/RHS
errors are 1.253e-15/1.331e-15/2.228e-15. The ordinary runtime remains periodic
TGV-only pending channel consumer/output integration and the explicit decision
on undefined physical padding initialization. This does not complete MB7.

## Generic Physical-End Explicit Filter

filter_axis_basic/filter_component_axis_basic retain the CPU generic
endpoint-preserving, sixth/sixth/sixth/eighth/tenth-order transition. They do
not select the separate high-order-wall filter. fluid_filter_explicit shares
the existing scalar/full x/y/z pipeline and halo protocol without a new field
or per-step allocation. The periodic entry retains its original kernel.

astr_backend_boundary_filter passes 288 polynomial/Nyquist checks with zero
error, covering x/y/z, all end combinations, both synchronization modes, all
scalar components/full storage and a minimum eight-interval direction. Physical
ghosts are NaN; sources, ghosts and other components stay bitwise unchanged.
Memcheck/initcheck report no errors or leaks.

The first coupled test caught stale pressure caches: post-filter q already
matched, but the wall state still used the preceding stage's pressure. The
original failure is wall_filter_stale_cache.log (rho error 1.798e-8). A physical
primitive refresh now precedes the wall command after filtering, matching the
CPU order; neither CPU math nor the periodic path was changed.

With the new frozen four-step filtered/forced wall reference, scalar+pinned,
full+pinned and scalar+device each pass NP=1/2 x/y/z (1008 rank stages each).
A representative full+device NP=2 y case also passes. Maximum scaled field/cache/
RHS errors are 2.505e-15/2.572e-15/2.487e-15. The 276 physical q packets, including
metadata, match bitwise across storage/transport and synchronization choices.
run_wall41_steps.py --compare-output performs these checks; check_packets can
reanalyze the immutable packets without rerunning a solver. They are explicit
test observations, not a new restart format. Representative multirank memory
checks report zero errors/leaks. DTK execution and ordinary wall-runtime
output/restart admission remain incomplete.

The physical filter's local-length check was narrowed to its actual template
requirement: eight intervals are required only when both ends are physical. A
single end may consume the valid halo toward its MPI neighbour. The mathematical
probe now has 504 zero-error cases, including a five-interval patch with hm=5;
updated memcheck/initcheck report no errors/leaks. A frozen global 10-cubed
reference passes actual NP=2 five-interval local patches in x/y/z; representative
full+device and scalar+pinned physical q also agree bitwise. Maximum short-patch
q/primitive/RHS errors are 1.997e-15/2.330e-15/5.581e-15. No CPU formula, halo
width or tolerance changed.

## Compact Physical-End Field Chain

compact_line_cpp now owns the shared compact_stencil_cpp C binding. Its
active_first/last fields restrict destination ownership only, never RHS rows or
the globally coupled matrix. Nodal divergence uses the current-axis range;
KEEP uses the common body box. Gradients still cover all physical nodes and
compact filters keep true endpoints after the global solve, as in the CPU.

run_wall41_steps.py --method compact --compact-transport pinned/device-aware
connects actual device gradients, KEEP, nodal diffusion, PCR and restoration.
Frozen CPU/CUDA Fortran references use 643c/643c and optional compact10. No-filter
NP=1/2 x/y/z passes 1008 rank stages per interface mode (max field/cache/RHS
1.253e-15/1.553e-15/1.713e-15). Scalar compact10+pinned passes the same matrix
(4.903e-14/4.908e-14/9.365e-14); representative NP=2 y full+device matches its
scalar+pinned physical q bits. Device interface staging counters are zero.

Actual first-stage RHS columns are observed, not replaced, and checked against
a whole-system residual and pivoted-LU solution. Maximum residual/LU differences
are 9.038e-16/7.879e-15. The same-state gradient probe accepts ROOT compact
[pinned|device-aware] through matching mpiexec; both modes pass 72 frozen-input
checks with max error 4.974e-18 and unchanged input bits. Its explicit mode still
passes the prior polynomial checks. Coupled gradient error is reported separately.

Representative multirank memcheck/initcheck and same-state gradient initcheck
report zero errors (and no memcheck leak). Ordinary compact periodic KEEP TGV
NP=1/2 y also passes after sharing the binding. These do not complete ordinary
wall output/restart, high-order wall/ME6/open-boundary integration or DTK execution.

## Physical-End ME6 and Nodal Wall Band

The wall stage runner accepts --viscous-form me6_opti. Matching references
are generated with prepare_wall41_reference.py using the same option.
Nodes within four global intervals of a nonperiodic boundary use the CPU
nodal viscous band; the disjoint inner box uses ME6. Stress/heat flux fields
are computed over all physical nodes. Compact nodal derivatives solve the
whole globally coupled system before the explicit wall-band commit; neither
the matrix nor its RHS is cut at the band boundary.

Four-step 16-cubed CPU/old-GPU comparisons cover x/y/z walls, NP=1/2 x/y/z,
both synchronization modes, and the existing fixed-force eligibility rule.
No-filter explicit KEEP passes pinned/device (1008 rank stages each), max
field/cache/RHS 1.253e-15/1.230e-15/1.821e-15. Scalar explicit10+pinned passes
another 1008 stages (2.505e-15/2.460e-15/2.643e-15). Compact KEEP+compact10
passes scalar+pinned and full+device (1008 stages each), maxima
4.903e-14/4.903e-14/1.033e-13. Final physical q packets are bitwise equal
across synchronization choices and paired transport/storage configurations.
Global compact residual/LU differences are at most 9.038e-16/7.879e-15;
coupled, not same-state, gradient error is at most 1.104e-12.

ME6 private edges actually communicate in y/z decompositions. Device-mode
application staging is zero. The x decomposition needs no separate ME6 edge
messages because the normal face halo already supplies x data. Representative
NP=2 z explicit memcheck and NP=2 y compact-filter initcheck report zero
errors; memcheck reports zero leakage. Ordinary periodic KEEP+ME6+compact10
NP=2 y also passes after the shared C binding change.

Evidence is under native/wall41_me6*_reference, wall_me6_explicit_*,
wall_me6_compact_filter_*, and ordinary_me6_after_wall_band. The poisoned
KEEP host guard independently preserves both sentinels, clears unused bases,
and produces the unchanged 294-term/85-basis table hash. None of these
component/phase checks admits ordinary wall-runtime checkpoint/restart or
claims DTK execution. Those, high-order/open boundaries and the remaining
MB7 consumers are still pending.

## Wavy Walls and Independent Filter Selection

prepare_wall41_reference.py --mapping wavy reuses mapped_grid x/y/z-wavy
with amplitude 0.15 and the existing fixture lengths. All Jacobians are
positive and off-diagonal metrics are nonzero. Explicit KEEP+ME6 passes
NP=1/2 x/y/z with pinned/device, max q/cache/RHS
1.331e-15/1.553e-15/1.962e-15. The independent compact10 filter also passes
scalar+pinned and full+device with bitwise-equal physical q packets; maxima
are 4.884e-14/4.881e-14/4.308e-14. The runner's --filter-method explicitly
separates main derivative and filter choices; it calls the existing kernels
and global PCR, not another solver. The same-state gradient probe passes
72 frozen-input checks and its existing 256 analytic quadratic checks.

Compact KEEP with wavy physical ends is explicitly rejected by both the
frozen CPU and current keep_geometry implementation. Its failed reference
run is retained. This is not a new-C++ numerical failure and was not bypassed;
the tested combination above uses explicit derivatives and compact filtering.
See progress section 10.17 and native/wall_me6_wavy_* evidence.

## Adiabatic No-Slip State

wall42_basic_state ports the existing CPU x/y range only. Pressure and
temperature use (4*f1-f2)/3, velocity is zero, and q is reconstructed with
the current perfect-gas constants. The z branch is not implemented in the
CPU reference and is not admitted here. wall41_basic_state retains prescribed
temperature. Neither state command writes RHS or other nodes/halos.

Use --boundary-type 42 --wall-axes x y with the reference producer and stage
runner. The state probe accepts ROOT 42 xy. It passes 72 frozen-state checks
(5.552e-16), including untouched-data/RHS bits, and memcheck reports zero
errors/leaks. Four-step NP=1/2 x/y/z KEEP+ME6 stages pass explicit no-filter
pinned/device and compact+compact10 scalar/pinned versus full/device; each
matrix checks 672 rank stages. Max q/cache/RHS is
1.432e-15/1.553e-15/1.673e-15 for explicit and
5.099e-14/6.256e-14/9.225e-14 for compact. Paired physical q packets are
bitwise equal. Compact residual/LU maxima are 7.600e-16/7.879e-15, and the
representative NP=2 y stage initcheck reports zero errors.

The original 108 bc41 state checks and wavy-wall compact-filter stage packets
still match after sharing the state kernel. See native/wall42_* and
wall41_wavy_after_adiabatic. These do not complete ordinary-main output,
restart or DCU device execution.

The current physical-end ABI, wall-band ME6, bc42 and updated probes also
compile/link under DTK26.04/GNU12/gfx906. Main/wall-step/channel/filter ldd
checks resolve the existing private MPI/HDF5 and DTK libraries. The initial
batch stopped with Error 9 in keep_probe.F90 without a compiler diagnostic;
a measured, single-target compilation with unchanged source/optimization
succeeded (74.57 s, peak RSS 976540 KiB), then the six previously unattempted
targets built successfully. The initial termination remains unexplained,
not a repaired compiler/code defect. No retry or compiler fallback was added.
The GNU CPU-only poisoned-table guard produces the unchanged NVHPC table
hash. See progress 10.19 and dtk2604/wall_me6_bc42* and
keep_probe_compile_measured.*. No new DCU GPU run or Slurm action occurred.

## Extrapolation and Geometric Symmetry

--boundary-type 50/60 now selects the existing low-order perfect-gas command.
bc50 extrapolates velocity/pressure and T in x, rho in y/z; EOS reconstructs
the other scalar. bc60 projects the extrapolated velocity with I-n*n^T,
using the normalized resident metric row, and extrapolates T except at z+,
where the existing CPU/CUDA Fortran rule extrapolates rho. Both commands
reset only the selected face RHS. Neither changes CPU math or adds normal
storage. The wall41/42 commands still preserve RHS.

Each state probe passes 108 frozen-state/footprint checks. bc60 additionally
passes 36 independent oblique-normal checks (projector matrix, sign/scale
invariance and unchanged other nodes), max state error 2.221e-16 and normal
velocity 6.004e-17. This is a component diagnostic, not a curved-flow claim.

For each boundary, KEEP+ME6 passes four-step NP=1/2 x/y/z stages with explicit
no-filter pinned/device and compact+compact10 scalar/pinned versus full/device
(1008 rank stages per matrix). Worst q/cache/RHS errors are
5.558e-14/7.188e-14/1.031e-13. Paired physical q packets are bitwise equal.
KEEP+nodal viscosity also passes bc50 explicit+full explicit10 and bc60
compact+scalar explicit10. Compact residual/LU maxima are
8.675e-16/5.719e-15. State memcheck and representative stage initcheck report
zero errors; memcheck reports zero leakage. The bc50 template initcheck also
confirms that nonsymmetric extrapolation does not read the unused metric.
See progress 10.20 and native/bc50_*/bc60_*.

These are local NVIDIA stage/component results, not ordinary-main output or
restart admission. They postdate the DTK snapshot in 10.19 and have not yet
been DTK-built/executed.

## Standard Convection Body Ownership

CPU convrsdcal6 accumulates all directions only in the common body box;
diffusion restricts each direction separately. The new C++ explicit physical
entry and compact kind=2 commit now use that common box. The original
2.400951741e-5 first-stage face error is retained in
native/standard_boundary_mask_initial_failure. Matrix rows/global coupling,
diffusion and CPU code are unchanged.

Configure through the root CMake, then build astr_backend_boundary_flux.
The explicit/compact modes use constant density/velocity, linear
pressure, identity metric/J and an analytic Euler-flux divergence. Each mode
passes 256 boundary-ownership/endpoint/synchronization checks; max scaled
errors 3.834e-16/5.112e-16 and outside-box RHS bits are unchanged. Compact
mode requires a matching NP=1 MPI launch: its full nonperiodic line systems
are solved once before varying destination boxes. This is not 64 CFD MPI
topologies. Logs are standard_body_flux_explicit/compact.log. Ordinary
periodic compact TGV NP=2 y also passes after the ownership correction.
Explicit initcheck and compact memcheck report zero errors; the latter has
zero leaked bytes.

The full standard physical-end trajectory gate is NOT closed. In the old
CPU standard path, full precedes channel forcing; the old GPU label follows
it. The measured difference is exactly that source (phase-aligned residual
below 6.662e-16). CPU pre_rhs_primitives/post_filter coverage is also limited.
Some standard/nodal boundary output/compact combinations are explicitly
rejected by the old reference. No missing file is reconstructed, skipped
automatically, or declared a pass. The proposed diagnostic-only extension
and fresh reference freeze await approval; see progress 10.21. Existing
reference files and CPU solver/logging source have not been modified.

## Private In-Situ Sampling Prerequisite

Build astr_backend_insitu_sampling through the root project. It uses the new
insitu_sampling_cpp wrapper over common C++ kernels, not CUDA Fortran device
arrays. Launch with the matching MPI launcher; arguments are topology
(e.g. 2,1,1), pinned/device-aware, and optional legacy/sixth. Omitting the
third argument selects the periodic trigonometric fixture. legacy uses a
nonperiodic quadratic field; sixth adds degree-six terms to check the existing
seven-node sixth-order physical-end closure. These are private postprocessing
derivatives, not replacements for fluid derivatives or boundary conditions.

The 32-interval fixture tests full14/Q-only/halo-only/velocity-only capture,
two synchronization modes, nonuniform rho, poisoned duplicate upper nodes,
canonical q4 bits, all actually allocated parent-buffer bits, view layouts,
parent/child lifetime, allocation lifetimes and transfer accounting. A final
rank0-only zero-density owner checks that all peers return the same failure
and release without modifying q. Its auxiliary full-array observations are
explicit test downloads, counted separately from production capture.

Capture owns its MPI communicator and fixed send/receive slots, canonicalizes
upper nodes from unique lower owners, then converts velocity. It releases q4
before allocating the private halo, retains packed velocity only on request,
and allocates diagnostics only as 1 or 14 components. Returned device views
expire at the next capture/destroy; consumers must finish before that event.
No Viskores/Catalyst/render dependency or new resource cap is introduced here.

Local NP=1/2 x/y/z tests pass periodic pinned/device-aware, physical legacy
pinned and physical sixth-order device-aware matrices. Maximum absolute
diagnostic errors are 5.108e-15, 1.990e-13 and 5.196e-13, respectively; the
velocity and valid halo values match the independent owner reference exactly.
The Q-only result is bitwise equal to the full Q slot. Device-aware capture
has zero application host-staging bytes. Capture reads only 4/8 status bytes;
payload allocation accounting is not a measurement of driver/MPI peak memory.
Representative memcheck/initcheck report zero errors, and memcheck no leaks.
Evidence and the unchanged legacy sampler reference are under
results/multi_backend_mb7_insitu_20261114; see progress 10.22.

This is a provider component, NOT ordinary-main in-situ/render admission,
CFD-trajectory validation or cumulative-statistics/restart closure. The main
C++ in-situ rejection is unchanged. The increment, including the 50/60 state
commands and standard body-flux ownership correction, also compiles/device-links
and final-links with DTK26.04/GNU12/gfx906. Sampling, boundary-flux and wall-state
targets built successfully; sampling ldd has no missing dependencies and keeps
the selected private MPI/Fortran/HDF5 stack. Vendor void-return and existing
forward-unknown-option warnings remain recorded in build_sampling_dtk.log.
No login GPU execution, Slurm action or new DCU numerical evidence is claimed;
DCU rendering remains MB9.

## Completed-Step C++ Product Consumer

insitu_index.cu/.h/.F90 supplies the existing global index-list/stencil producer;
insitu_index_runtime.F90 binds it to the ordinary Fortran host. It consumes the
current complete-step primitive caches and resident metric, not a host flow
mirror or the main solver gradient. It preserves the old seven-node derivative,
SoA selected values, interleaved coordinates and optional private MPI strips.
Static coordinates upload once on demand per parent; per-group workspaces are
released after consumption. Parent destruction also releases that static cache.

astr_backend_insitu_index checks 30 source/derivative/wall/vortex fields on an
analytic quadratic state with identity/nonorthogonal metric, NP=1/2 and both
synchronization policies. Transport arguments are pinned or device-aware.
Maximum absolute error is 1.426e-13; the complete neighborhood matches unique
owners exactly. NP=2 pinned memcheck has no errors/leaks. The eigensystem
reference is the existing CPU math component, not a separate physical model.
The final probe also rejects negative Re/temperature coefficients that cancel
into positive viscosity, matching the original material parameter contract.
These unit wall checks do not admit a wall CFD case in the ordinary C++ main.

run_cpp_insitu_gate.py runs actual ASTR executables. Required arguments are
--binary, --reference (the frozen CUDA Fortran astr), and a new --output directory.
The default 16^3/four-step test requests three index planes with 26 fields and
compares the new/legacy same-topology state and archives; --ranks 2 --axis x/y/z
selects the split. These numerical archives explicitly use host communication
strips and selected field downloads; their comparisons do not prove device MPI.
Maximum scaled field/terminal errors are 2.939e-13/1.547e-15; toggling numerical
archives leaves all authoritative state.h5 dataset bytes unchanged.

--images selects actual 32^3/four-step slice, Q=0.25 and instantaneous line16
products at steps 0/2/4. --pipeline standard-device/direct-device and --transport
pinned/device-aware select the existing renderer and private provider transport;
the solver halo stays pinned in this test. NP=1 standard/pinned, NP=2 x
standard/device-aware and direct/pinned pass with zero field/geometry readback,
and device-aware also records zero application private host staging. Images
are nonempty and were visually inspected; pixel differences are reported, not
substituted for the field comparison. --restart additionally compares 4 versus
2+2 complete steps, saved catalog, product times and authoritative bits. The
NP=2 y data and NP=2 x standard/device-aware image restart checks pass; image
restart emits only step 4, with pixelwise equal results. EPS is disabled and
the driver removes artificial test budgets/timeouts using existing controls.

Build through root CMake with ASTR_GPU_CPP_BACKEND=cuda, ASTR_WITH_CUDA=OFF,
ASTR_WITH_CATALYST=ON, ASTR_WITH_INSITU_DEVICE=ON and
ASTR_WITH_INSITU_DEVICE_RENDERING=ON. Select the same Catalyst/ParaView/Viskores
dependencies as the existing renderer. In the local GCC13 dependency setup,
explicitly set both CMAKE_CXX_COMPILER and CMAKE_CUDA_HOST_COMPILER to g++-13
before compiler initialization; do not rely on the system default g++-15.
The new CUDA final link explicitly retains the required Python embedding
library as the existing render probes do. See build_backend_cpp_insitu_gcc13
and configure_cpp_insitu_gcc13.log/build_cpp_insitu_gcc13.log in the evidence
directory; failed build logs remain separate.

The supported new consumer sources are instantaneous nonreacting FP64, the
Reynolds/Favre velocity-time extension and the selected-time subset below.
Solver-sample sources remain unimplemented rather than reading stale CPU data.
Cartesian one-dimensional temporal consumers are described below.
Cropped-volume streamlines, new wall CFD product
matrices and read-only previews are not claimed yet. The CURVE increment below
adds physical-coordinate streamlines without changing these other limits.
The common Index/vector/time-moment code now has the DTK compile/link evidence
recorded below, but its new device numerical matrix has not been run. DCU
rendering is still MB9. This is not CPP-P performance work, a new physical validation, or
closure of MB6/MB7. See progress 10.23 and
results/multi_backend_mb7_insitu_20261114/index_* and images_cpp_* evidence.

## Selected Vectors And Physical CURVE Products

astr_backend_insitu_vectors checks the selected Index packet, not velocity
reconstructed from parent q. It uses a five-column source with a nonfinite
unused pressure column, explicitly selects all three velocity columns and
checks that a nonfinite selected column is collectively rejected. Both the
selected source and parent q stay bitwise unchanged. The borrowed input view
may be strided; it must not alias outputs retired by this capture.

sample_capture_coordinates adds a separate physical-position halo to the same
sample/communicator/packet slots. Unlike the velocity halo, global coordinate
ends never wrap. Only internal MPI faces exchange coordinates; true exterior
padding stays zero as in the frozen CUDA Fortran provider. XYZ exchanges
extend transverse regions in order. No extrapolation or periodic translation
was introduced. No q snapshot or duplicate packed vector is allocated for a
halo-only selected-vector request. Owned bytes, capture count and transfer
counters are checked in the component probe.

NP=1/2 x/y/z, both synchronization policies and coupled-periodic/y-wavy mappings
give exact selected velocity and physical-position arrays over repeated
captures. NP=2 pinned memcheck/initcheck report zero errors (memcheck zero
leaks). Default CUDA-aware MPI numerical tests also match, but sanitizer
interoperability is NOT closed: initcheck flags receive buffers written by
UCX CUDA IPC in both the new provider and the frozen old provider. An explicit
diagnostic UCX_TLS=self,sm,cuda_copy run removes these initcheck errors; this
is a test configuration, not a production fallback or automatic switch.
Protocol logs show cuda_ipc zero-copy writes in the default test. Cross-process
shadow tracking is a plausible explanation, not a vendor-confirmed verdict.
Aware memcheck also reports MPI/UCX CUDA driver API probe errors and the default
IPC run retains allocations; preserve these failed logs rather than declaring
aware memory safety passed. No receive memset or suppression was added.

run_cpp_insitu_gate.py --mapping periodic uses an actual 32^3 static nonorthogonal
TGV. Numeric archives use the 21 fields already admitted by the shared CURVE
source; the Cartesian-only thermal/vortex field admission is unchanged.
NP=2 x fields/terminal scaled errors are 1.4164e-14/1.9880e-15, geometry zero;
observation on/off leaves authoritative dataset bytes equal. Actual NP=1/2
slice/Q/physical streamlines produce frames 0/2/4. Standard+pinned and
direct+device-aware are covered; slice/Q pixels match the frozen backend,
whereas TGV streamline mean absolute pixel differences reach 0.0914/255.
These are display comparisons, not an analytic trajectory gate. Direct/aware
4 versus 2+2 restores q and output clocks exactly and emits only frame 4;
all three continuous/restarted images are pixelwise equal. Field/geometry
readback is zero; pinned private halo staging remains separately reported.

--automatic selects the existing CFL=.5, initial_dt=.001, max_dt=.002,
growth=1.1 control and, for images, a physical-time interval of .002. The
driver checks actual completed-step logs, CFL, receipt times and restart
history; it does not assume fixed time spacing from step labels.
Evidence is results/multi_backend_mb7_curve_products_20261114; see progress
10.24 for the actual automatic-clock result and the remaining acceptance gaps.
The new vector target is listed in build_gpufusion.sh, but this is not DTK
compile/run evidence. DCU rendering remains MB9 and no remote task was run.

## Device Time-Moment Component

insitu_moments.cu/.h/.F90 implements the existing FP64 endpoint-weighted
Reynolds/Favre recurrence. The shared Fortran clipped_trapezoid supplies its
two endpoint weights; this is not an unweighted sample average or an integral
of interpolated products. It consumes parent q directly on the owned box,
without reading undefined ghost nodes or changing q. The caller supplies the
owned extent, including a nonperiodic physical upper node when appropriate.

Persistent storage is 30 FP64 values/node and one status integer. The optional
spatial consumer adds one static FP64 node measure/node and six device scalars;
spatial MPI normalization stays with Fortran. An observation reads one status
integer, and a spatial request reads six scalars, not the flow/state arrays.
The 41-component result is allocated only for an explicit host output and
released immediately. The existing 34-component packed state writes metadata
on host and transfers only the contiguous 30-component device state. These
explicit pack/restore/output transfers are separately counted.

astr_backend_insitu_moments takes a topology, for example 1,1,1 or 1,2,1. The
root-CMake target links the unchanged CPU insitu_velocity_statistics and
insitu_time_integral as a reference. NP=1 and NP=2 x/y/z each cover both sync
policies, half-open/physical-upper owned boxes, six nonuniform sample times,
a clipped window, zero coverage and an analytic constant-velocity case.
Each rank completes 72 valid samples and 12 restore branches. The same input
replayed from the third sample yields bitwise-identical 34-component state.
State/result/spatial maximum scaled differences are respectively
8.2140e-17/8.2150e-17/1.7307e-16; near-zero RMS is gated by its square and the
largest raw RMS difference, 6.853e-17, is reported separately.

The probe also checks parent lifetime, state view lifetime, input q bits
(including unused NaN halo), output allocation release, no implicit array
transfer, pre-update time/restore rejection and nonpositive source density.
A failed device update is discarded, not retried or rolled back.
Final NP=2 y memcheck has zero errors/leaks and NP=2 z initcheck zero errors.
No MPI payload is exchanged by the temporal component itself; these tests
do not constitute a new device-aware transport or global statistics gate.

An initial push launch used 152 registers/thread and failed with CUDA 701 at
512 threads/block. Matching the old launch_bounds(512) contract reduces it to
128 registers and makes the unchanged arithmetic executable. Resource dumps
and the initial failed log are retained; this is not a performance result.

Evidence: results/multi_backend_mb7_moments_20261114, including frozen old
CPU/device mathematical sources and SHA256SUMS. This component increment
preceded the ordinary-main wiring described below. The target is listed in
build_gpufusion.sh but has not been DTK-built/run. See progress 10.25.

## Ordinary Time Statistics And Mean Products

insitu_statistics_runtime.F90 connects the existing completed-step session,
regional reduction, statistics.h5 and final export to C++ moments. CPU and
CUDA Fortran providers are unchanged. Static measures come from the existing
uniform-cell, channel trapezoidal or CURVE Gauss-volume definitions; they
upload once when spatial reduction is requested. Fortran retains the MPI
reduction and regional time statistics. No new environment switch, state
format, checkpoint sidecar, per-step budget query or implicit CPU flow mirror
was introduced. Device identity queries support the existing explicit
per-physical-device allocation budget; tests use the existing unlimited form.

The Index provider borrows owned moments directly for Reynolds density/velocity
and Favre velocity. Mean center values, including periodic/MPI upper nodes,
come from the completed unique-owner neighborhood rather than an out-of-range
half-open moment index. The existing stencil/metric computes mean gradients,
Q_rs and curl. Q of mean velocity is not the time average of instantaneous Q.
Pressure/temperature/Favre-density means remain unavailable; a bad request is
isolated to its product, not silently replaced by an instantaneous field.

run_cpp_statistics_gate.py requires --binary, --cpu-reference, --gpu-reference
and a fresh --output; --ranks/--axis select topology, --means adds two mean data
slices and one intentionally unavailable pressure mean. --mapping periodic
uses a static nonorthogonal 32^3 grid (otherwise 16^3); --automatic enables
CFL=.5/growth=1.1. --memcheck instruments only the actual C++ statistics run.
The test compares 12 complete steps versus 5+7, on/off authoritative bits,
34-component point states, regional state, the 41-component derived export and
mean slice fields. RMS is gated by its square and reported in raw form too.
The different CPU/native storage role and producer geometry signature are
not mistaken for numerical field differences. Same-backend exact comparison
still includes every authoritative/statistics dataset and output value.
New segment creation step/clock must match that segment's first frame; only
geometry values/indices, not creation identities, equal the older segment.

Actual NP=1 and NP=2 y/z periodic runs pass; NP=2 x static CURVE with automatic
steps also passes. Largest terminal/statistics scaled error is 5.551e-15 and
mean derivative/data error 2.024e-14. Statistic/observation switches do not
change authoritative bits. State, statistics, final export and step history
are bitwise equal after exact continuation; only the scheduled step-9 mean
data frame is emitted after restoring step 5. Real NP=2 z mean-data Memcheck
has zero errors; its invocation does not include full leak checking.

run_cpp_mean_images_gate.py requires --binary, --reference (old CUDA Fortran)
and --output; it accepts --ranks/--axis/--mapping/--pipeline/--transport.
It renders Reynolds/Favre slices and Favre line16 at covered steps 1/3 over
four full steps, plus a 2+2 continuation. Actual NP=1 standard/pinned and
static CURVE NP=2 x direct/device-aware pass. Field/geometry readback is zero;
aware private host staging is zero, while the main solver halo stays pinned.
Q/state and statistic output are unchanged by rendering. The two mean slices
match the old backend pixelwise; streamline mean absolute pixel differences
reach .07661/255 and are reported, not an analytic trajectory gate. Restored
images are pixelwise identical to the continuous run. EPS is disabled and
actual CURVE slice/line images were inspected.

See progress 10.26 and results/multi_backend_mb7_moments_20261114/main_* and
images_* reports. The selected-time/dynamic lifecycle increment is below.
These do not close solver-sample statistics, wall ordinary-main consumers,
preview/cropped volume, or the new DTK matrix. DCU rendering remains MB9.

## Sparse Selected-Time Device Component

The selected_* C ABI in insitu_moments.cu/.h/.F90 preserves the existing
insitu_sample_moments layout: 3+3*fields plus twelve optional velocity central
moments, at most thirty components. This is NOT the layout of full velocity
time moments. It allocates only the supplied node support and fields. Static
indices upload once; indexed observations decode the requested q fields on
device and read back one status integer. Unused energy/coefficients are not
read or rejected. An external packet may be interleaved or SoA, with unique
one-based target positions, and remains producer-owned/non-aliasing until the
synchronized call returns. This existing wall-source route has no covariance.

Profile measures/results allocate only on request. The existing 128-thread
terms reduction downloads terms*ny plus status, never point states. Explicit
checkpoint observation/restore copies the existing raw component matrix;
host group metadata/coverage still belongs to the original orchestrator.
Empty supports retain one zero row and perform no update/transfer.

Build astr_backend_insitu_selected through root CMake and pass a topology,
e.g. 1,1,1 or 1,2,1. Native NP=1 and NP=2 x/y/z cover both sync policies,
the nine field/covariance layouts, and representative zero-coverage, reset and
constant cases. Each rank completes 144 indexed observations, 24 restored
branches and 18 external packet calls. State/profile maximum scaled errors
are 2.2182e-16/1.8776e-16; external packet error is zero. Restored raw state is
bitwise equal after subsequent observations. Source q, including unused NaN
halo, is unchanged. Counters check array bytes, status/explicit transfer,
index reuse, optional buffers and profile measure reuse.

Native-only selected_alloc_fault.cu links into this probe with --wrap, NOT
into production. A deliberately failed second profile allocation demonstrated
a partial-workspace cleanup defect; the fix frees only partial allocations,
returns the error and leaves statistics unchanged. No production retry or
numerical rollback was added. This is ownership failure injection, not proof
of recovery from arbitrary real GPU OOM/context failure. Final NP=2 y full
Memcheck has zero errors/leaks; NP=2 z Initcheck has zero errors, including
the injected cleanup case. The full-velocity component also passes after
relinking the shared library.

Evidence: results/multi_backend_mb7_selected_20261114/reference and
selected_cleanup_* logs; the pre-fix injected failure is retained separately.
See progress 10.27. This component alone has no MPI payload and does not prove
distributed selected statistics or dynamic catalogs; the ordinary-main
integration and its evidence are below. Neither proves physical statistical
convergence or DCU execution. The DTK build
helper includes the target; remote compile and execution evidence are separate.

The current root astr and Index/vector/full-moments/selected probes also build
and device-link on DTK26.04/GNU12/gfx906 in an isolated ~/weiph/ work directory.
Main/probe ldd has no missing libraries and resolves the original private
GNU12 MPI/HDF5 and DTK runtime. Uploaded core source digests match local files.
The GNU build exposed the Index probe's implied-do variable declaration order;
moving the declaration before the constant initializer preserves the field
list and passes the native numerical recheck. The initial failure and later
four-target build logs are retained in the selected evidence directory.
This updates older not-built notes, not device execution or DCU rendering:
no Slurm operation or login-node GPU test was performed. See progress 10.27.

## Selected-Time Ordinary Main Consumers

insitu_selected_device.F90 connects the C++ sparse states to the original shared
group/coverage/reconcile and version-2 serializer. Indexed updates remain on
device; profile reductions return only terms*ny plus status. The Index reader
supports selected Reynolds/Favre primitives, the original private gradients/Q
and fixed-j fluctuations. Missing sparse support is not replaced by instantaneous
data. Optional workspace accounting queries actual owned bytes. No full-domain
statistics array or CPU numerical fallback is introduced.

Run run_cpp_selected_gate.py with --binary, --cpu-reference, --gpu-reference,
--output and optionally --expanded --ranks 2 --axis x/y/z --memcheck. The
16-cubed twelve-step matrix compares immutable CPU/old-GPU references, independently
integrates actual raw endpoint samples and projects station profiles. Maximum
field error is 8.5266e-14; independent time/profile errors are below 1.0e-15.
Same-backend 5+7 continuation preserves raw statistic words, field arrays and
profile CSV bytes. Final NP=2 z actual-main Memcheck has zero errors; no full
leak-check was selected for that run.

run_cpp_selected_lifecycle.py uses the existing catalog_replay.so test publisher
to add, disable and re-enable a sparse pressure/temperature support. The two
actual coverage intervals exclude the inactive gap (duration .0055, segments=2),
agree with independent integration to 5.9375e-16, and survive 9+3 exact continuation.
The watcher test does not alter the numerical solver or add production retries.

run_cpp_mean_images_gate.py --statistics-source selected-time checks actual
Reynolds velocity/Favre temperature/fixed-j pressure-fluctuation images. The
NP=2 x direct-device+pinned outputs match the old GPU pixelwise; rendering does
not alter the flow or accumulated state, and restored images are pixelwise equal.
The non-rendering comparison uses equivalent numerical products so it retains
the same sparse supports. Device-resident rendering reads back neither fields
nor geometry. An additional periodic-CURVE NP=2 y standard-device+device-aware
velocity-time slice/Favre-line check retains exact continuation; its old/new line
pixel difference is reported, not treated as an analytic trajectory check.

Evidence is in results/multi_backend_mb7_selected_main_20261114; see progress
10.28 for the named final reports and retained invalid test configurations.
The original selected-time contract does not admit streamlines or Q surfaces;
velocity-time streamlines are a different provider. Selected wall external
sampling and solver-samples are not connected. This increment
has not yet been rebuilt/run on DTK; the preceding DTK snapshot covers only
the earlier component/source set. DCU third-party rendering remains MB9.

## One-Dimensional Temporal Consumers

After connecting the actual Index/selected sources, the temporary C++ rejection
of time_space and probe_spectrum was removed. The shared temporal histories,
analysis, publisher, serializer and host chart implementation are unchanged.
The existing FP64 nonreacting Cartesian restriction still applies. No resampling
or substitute GPU renderer is added.

run_cpp_temporal_gate.py uses 16-cubed TGV with twelve complete steps and immutable
CPU/old-GPU references. NP=2 x covers instantaneous data and independent SciPy
Welch; NP=2 y --selected adds allowed R/F mean/fluctuation histories; NP=2 z
--automatic --selected checks true varying timestamps, observation isolation and
5+7 exact continuation. It compares raw histories/means, checkpoint state and
frequency CSV bytes, not just the presence of output files. The largest reference
history scaled difference is 5.6844e-14. Automatic-step raw histories remain
available while equal-spacing Welch is explicitly unavailable (waiting status,
empty numerical spectrum, zero valid segments). The actual combined automatic
run has zero Memcheck errors on both ranks; see the retained initial test-status
mismatch and corrected read-only analysis in progress 10.29.

Every-step sampling and three-step publication are distinct: the seventeen-node
three-field instantaneous line downloads 5304 bytes over thirteen samples; the
two-field point downloads 208 bytes. These figures exclude private exchange,
status scalars, metadata and explicit checkpoints. Only these selected 1D arrays
reach the existing host history/FFT/chart code; a full flow volume is not copied
for temporal processing.

--images selects the matching Catalyst-enabled executable and the original
host-matplotlib chart consumer. NP=2 x emits five pressure time-space charts and
two PSD charts, JPEG only. New/old images and same-backend restored images are
pixelwise equal; state/statistics exact continuation and image on/off isolation
are checked. The existing chart displays the current archive block, not the
entire accumulated time series; a final single-sample block appears as one row.
Short-window spectra are implementation checks, not physical turbulent spectra.

Evidence: results/multi_backend_mb7_selected_main_20261114/temporal_*; final
reports are temporal_fixed_np2_x, temporal_selected_velocity_np2_y,
temporal_auto_selected_np2_z and temporal_images_np2_x. This new integration is
not covered by the preceding DTK build snapshot and has no new DCU execution or
rendering evidence. Physical-boundary ordinary-main, wall external statistics,
solver-samples and the remaining MB6/MB7 targets are still separate work.

## Approved Physical-Exterior Initialization And RHS Phase

fluid_initialize_exterior is called only during the C++ startup after initial q
upload. It defines physical-exterior q/rho/velocity/pressure/temperature and q's
multi-axis corners as positive zero. It does not touch physical nodes, valid MPI
face halos, geometry or CPU arrays. File restoration does not call it again.
astr_backend_exterior_initialization checks all 64 physical-end configurations
in both sync modes using nonzero markers and exact word comparisons. All 128
cases pass; full Memcheck reports zero errors/leaks and Initcheck zero errors.
A separate NP=2 y ordinary TGV checkpoint/restart check preserves all state words.

The user also approved a diagnostic-only CPU/old-GPU extension: keep existing
full tags, add nonreacting post_source after source/boundary RHS assembly and
before stage limiting, and provide requested primitive/filter observations
independent of format. No ordinary production output or numerical ordering is
changed. The stage probe uses post_source for standard convection, avoiding the
old CPU/GPU full-tag phase mismatch. KEEP's full remains at the matching phase.

Updated immutable CPU/old-GPU binaries and source/build records are under
results/multi_backend_mb7_physical_approved_20261114/reference_diagnostics.
At matching bc60-y wavy standard643e/ME6 fixed-force conditions, 122 pre-existing
physical field/cache/RHS snapshots remain bitwise unchanged. Fresh four-step
references give standard bc41-y explicit/filter/force and bc60-x/y/z wavy-ME6
stage checks on NP=1/2 x/y/z: RHS maximum scaled errors 1.8874e-15/2.0503e-15;
gradient maximum 2.0356e-14. All 1344 rank stages pass and sync variants preserve
the final physical q words.

See progress 10.30 and the named reports in that evidence directory. These
approvals unblock ordinary physical-boundary integration; the first actual
main-program increment is below. No new DCU execution or rendering is claimed.

## Ordinary Physical-Boundary Main

fluid_runtime now dispatches the existing low-order state/filter/gradient and
standard/KEEP convection plus nodal/ME6 operators, rather than only running
component drivers. Supported flow statistics/source orchestration is currently
nonreacting FP64 3D nondimensional TGV/channel. The boundary library has existing
41/42/50/60 commands (42 has no z implementation); measured ordinary-main scope
is narrower and is listed in progress 10.31. High-order/SBP, blowing, sponge,
IBM, NSCBC, other source families and legacy mean fields are not silently
replaced by another algorithm when they are not connected.

Channel scalar reduction uses the original nodal y integral, wall shear and w
square definitions, then the original Fortran chanfoce and serialized driver
history. Only a static y coordinate component uploads once; ordinary reductions
return scalars, not flow fields. Initial unused physical exterior storage is
defined as approved; exact restoration still preserves the file's complete
state and cache phase.

run_cpp_physical_gate.py takes --binary, --reference (a matched frozen fixture
directory), --mpiexec and --output; select --ranks/--axis/--wall-axis and the
existing --workspace, --transport, --compact-transport controls. It copies
immutable initial/grid inputs, compares twelve same-phase q/cache/post_source
observations per rank, scalar output and terminal checkpoint, checks checkpoint
on/off isolation, and verifies 2+2 same-backend exact restoration of all state
and control bytes. The run is actual ASTR advancement; Python supplies no RHS.

The 16-cubed four-step evidence covers standard643e bc41-y explicit10 on NP=1
and NP=2 x/y/z; KEEP643c/ME6 bc42-y compact10 on NP=1 and NP=2 y, including
full-filter/device halo and device PCR; and standard643e/ME6 bc60-y wavy on
NP=2 x. Maximum canonical RHS error is 9.811e-14; state/control and restored
scalar values are bitwise equal within each exact run. Final bc41 explicit
and bc42 compact NP=2 y pinned Memcheck runs report zero errors on both ranks.
The following increments add physical volume/slice archives, a bounded wall
consumer subset and bc60-x/z ordinary-main cases. Other boundary combinations,
adaptive dt and feedback/frozen drivers still need their own joint evidence.

Root integration exposed missing nodal-flux allocation for physical ME6 and
the need to use the existing compact wall-band commit rather than full-domain
commit. A separate NVHPC same-module BIND(C) wrapper call generated a NULL
handle/missing integer arguments; GDB/registers and before/after disassembly
are preserved. The affected route now calls the existing C ABI directly; no
compiler-specific switch, retry or numerical fallback was added.

See results/multi_backend_mb7_physical_approved_20261114/main_* reports and
main_control_bytes.json, and progress 10.31 for retained failed/debug records.
This is not physical statistical convergence, CPP-P timing, DTK execution or
MB6/MB7 completion. Source/statistics work still missing from the full goal is
not removed by this bounded ordinary-main increment.

## Physical Volume And Numerical Slice Archives

run_cpp_archive_gate.py accepts --physical-fixture and --wall-axis to copy the
matching frozen CPU/old-GPU inputs. --reference points to the immutable binary
directory, not the fixture. --axis selects the NP=2 decomposition. Each actual
four-step solver run writes two six-component primitive volumes and 27 HDF5
numerical slices: i/j/k at 0/8/16, steps 0/2/4. Default slices contain the six
primitives plus Q, divergence and density-gradient magnitude. The original
read-grid/channel volume-derived admission is not expanded by this test.

The comparison checks float datasets, integer metadata and checkpoint-output
isolation against independent frozen CPU/old GPU. Field volumes use their
existing event cache; Index numerical slices download only selected 2D values.
Exact expected transfer counts replace the obsolete requirement for shared
volume/slice downloads: on NP=2 y, two volume events total 2659392 bytes and
27 nine-field slices total 561816 result bytes, excluding MPI/status/geometry.

Evidence is results/multi_backend_mb7_physical_archives_20261114:
standard41_final_np2_y (explicit10 scalar), keep42_np2_y_full (compact KEEP/ME6),
standard60_wavy_np2_x (wavy ME6 without filtering). Maximum archive scaled
differences are 3.6923e-14, 6.5077e-13 and 1.8190e-14, respectively. All are
below 2e-10. Main-program bc60-x and bc60-z on NP=2 z also pass same-phase
state/cache/RHS, 2+2 exact state/control/scalar restoration and output isolation;
see main_standard60_wavy_x_np2_z and main_standard60_wavy_z_np2_z in the
physical_approved evidence root. Multi-physical-axis intersections remain separate.

## Wall Selected Moments, Profiles And Images

run_cpp_wall_consumers.py accepts --binary, --reference-binaries, --fixture,
--mpiexec, --output, --ranks and --axis. First --freeze-only produces independent
CPU/old-GPU wall outputs, checks the pair and protects the reference directory.
Subsequent runs use --reference-results; they never overwrite frozen results.
The fixture is Cartesian bc41-y, 16-cubed intervals, dt=1e-5, four steps and
the clipped statistics window [0.5dt,3.5dt]. It is a short implementation test,
not a turbulent statistically stationary case.

The new native bridge scatters the retained device Index packet into the
existing sparse moments; no full wall flow array downloads for accumulation.
The selected reader retrieves rho/T and the stored molecular wall means by
identity, rather than recomputing viscosity or shear from mean primitives.
Original group coverage, endpoint interpolation, empty ranks, true z weights,
CSV/HDF5 consumers and serialized histories are retained. Workspace accounting
includes both rho and T in every molecular wall packet, plus requested velocity.

Eight numerical products cover raw six-field walls, full/partial/point spans,
viscosity-only/shear-only/thermal-only profiles and a Reynolds wall slice.
Independent endpoint integration differs by at most 3.3307e-16; frozen solver
fields by 1.9965e-15. NP=1, NP=2 y with an empty wall rank and NP=2 z with real
device private exchange pass 2+2 exact state/statistics/control restoration,
CSV/result equality and observation on/off state/clock isolation.

--images additionally requires --renderer-reference pointing to the immutable
Catalyst-enabled old GPU executable. It compares five instantaneous/mean wall
shear images at 800x600, JPEG only, fixed [0,3e-9] color range. Existing direct
device rendering receipts show no field/geometry host readback; new/old images
and same-backend restored images are pixelwise equal. This does not enlarge
the original product contract to CURVE wall normals or wall-unit diagnostics.
The shared legacy rendering template slightly clips the bottom x-axis label;
the current numerical integration does not change that unrelated layout.

Sampler counters in wall_transfer_analysis.json record zero flow D2H. NP=2 z
private exchange totals 1811040 bytes, with 120/80 status bytes returned and
10240/16384 index bytes uploaded across the five sample times. Final requested
product results and checkpoints still transfer under their separate contracts.

--memcheck with --axis y --mpi-pml ob1 has zero errors on both ranks. Its wall
private exchange is local (zero network bytes), so it is not device-MPI memory
proof. The retained --axis z UCX Memcheck attempt exits 99: its first driver
error is cuCtxSetFlags during MPI_Init before ASTR kernels; 18883/32931 errors
include UCX context/pointer queries and exceed the print limit. Actual z-device
numerical/restart tests pass without the tool, but that tool gap is not hidden
by suppression, SDK changes or automatic backend replacement.

Evidence: results/multi_backend_mb7_wall_consumers_20261114/frozen_wall,
main_np1, main_np2_y, main_np2_z, main_np2_y_ob1_memcheck, images_np2_y and the
failed main_np2_z_memcheck. See progress 10.32--10.33 for provenance and scope.
No new DTK execution, performance, Git write or remote job operation is claimed.

## Original Ducros Sensor Components

prepare_sensor_components.py freezes existing CPU/old-GPU pressure, gradients,
raw sensor and byte markers for three actual RK stages of 16-cubed periodic
Shu--Osher (543e, reconstruction 3, dt=1e-5, no viscosity/filter/characteristics).
It requires --reference (immutable bin directory), --template (original input.tgv
and controller directory), --mpiexec and a fresh --output. Only explicit existing
sensor diagnostics are enabled; formal output/checkpoints are disabled. The
successful directory and its pair report are made read-only before consumer
development. Peak sensor is about 0.7 with 4046 active nodes, not an empty mask.

shock_sensor.cu/.F90/.h retains the original divergence-squared Ducros factor,
pressure curvature, 1e-30 regularizer, physical-end index clamping, halo-only
ssf exchange and -hm+1:hm expansion followed by strict > threshold. It does not
enable the H1 compression-only candidate. The FP64 raw field and physical byte
mask are allocated only when requested; ordinary centered runs do not allocate
them. The mask currently has no extended compact halo. Its host download API
is a validation observation, not per-kernel production host computation.

Run the root target astr_backend_shock_sensor via run_sensor_components.py:
--binary, --reference (the frozen packet directory), --mpiexec, --output;
select --ranks 1/2, --axis x/y/z, --transport pinned/device and optionally
--sanitizer memcheck/initcheck. Every rank checks 12 frozen combinations and
ten analytic combinations across both synchronization modes. The polynomial
pressure test uses a separate closed-form curvature, with compression/expansion
symmetry, pure rotation, constant pressure and strict zero-threshold behavior.
An all-nonzero gradient tensor has trace 15 and curl (2,-4,2), so an independent
75/83 factor separately checks the nonzero-vorticity denominator.
Physical-exterior input pressure/raw halos are explicitly NaN-poisoned only
in the probe. Exchanging raw halos must preserve all physical raw words.

Evidence: results/multi_backend_mb7_sensor_20261114. native_np1, native_np2_x/y/z
and native_np2_z_device preserve the initial eight-analytic-case records.
The final tensor_np1, tensor_np2_x_memcheck, tensor_np2_y_initcheck and
tensor_np2_z_device cover ten analytic cases with zero raw differences and
exact byte masks. Pinned NP=2 x Memcheck and NP=2 y Initcheck have zero errors;
Memcheck also reports zero device leaks. periodic_main_unchanged checks ordinary
NP=2 y scalar TGV exact continuation and output isolation after the optional
state extension. Preparation failures remain separate from the immutable valid
frozen_shuosher_no_budget_cap reference. This is sensor-component evidence,
not an integrated C++ MP7/MP7-LD or compact expanded-mask delivery.

The current DTK26.04/GNU12/gfx906 snapshot is
~/weiph/mb67_sensor_build_20261114.iLrwIU: ordinary astr and the build helper's
complete target list compile/device-link/final-link, including the new sensor
and exterior probes and current physical/selected consumers. This supersedes
older build-only snapshots for these source increments, not their missing DCU
device execution. Catalyst remains off. The helper now explicitly sets the
matching private MPIEXEC_EXECUTABLE rather than accepting CMake's public launcher.
See progress 10.35 and the sensor evidence root's dtk2604 logs. No Slurm query,
submission, login-node GPU execution or Git write is part of this build check.

## MP7, MP7-LD And Selective Roe

prepare_upwind_components.py freezes actual 16-cubed CPU/old-GPU inputs/RHS for
periodic physical-space MP7, selective-Roe MP7 and a stretched 743e/643e flat
plate. It takes --reference (immutable binary directory), --template, --mpiexec
and --output. Original ASTR geometry dumps provide the Jacobian/metrics. The
controlled plate compression activates 2567/4913 nodes; it is an operator
fixture, not a physical shock/boundary-layer validation. References are protected
only after independent CPU/old-GPU comparisons.

upwind_operator.cu/.F90/.h retains cached-temperature Steger-Warming splitting,
MP7/MP7-LD coefficients and limiter rules, low-order physical-end closure and
selective Roe characteristic projection. One five-component interface buffer
is allocated on demand and reused between directions. No full-volume split
flux pair is allocated; no CPU numerical RHS is used. Compact expanded masks
now have the separate byte-halo/consumer component checks below. High-order
boundary overrides and crashfix critical nodes remain unconnected.

Build astr_backend_upwind, then use run_upwind_components.py with --binary,
--reference (packet directory), --mpiexec, --output, --ranks 1/2, --axis x/y/z,
--transport pinned/device and optional --sanitizer memcheck/initcheck. Each rank
checks 72 combinations including native-gradient -> raw sensor -> halo-only
exchange -> expanded physical mask -> all three convection directions. Actual
q and primitive face exchanges are exercised; immutable physical q words and
workspace addresses are checked. Maximum RHS error is 2.22045e-16 and gradient
error 6.51331e-15. Actual x/y pinned and z device decompositions pass.

prepare_upwind_steps.py reuses frozen full RHS/post-update snapshots; it does
not run a Python integrator. Supply its directory via --steps-reference to add
36 prepared ASTR RK stages per rank, including real nodal viscous exchange and
SSP-RK3. The optional sensor hook finishes before stress exchange starts on the
shared transport. Full RHS and updated q errors are at most 2.22045e-16 and
3.55272e-16. Pinned x Memcheck and y Initcheck report zero errors on both ranks;
Memcheck also reports zero device leaks. This still supplies frozen boundary
states and does not claim an ordinary C++ profile-inflow/SBLI boundary chain.

run_cpp_upwind_gate.py checks ordinary MP7 TGV. --freeze-only first creates
immutable CPU/old-GPU runs for --variant inviscid or me6_filter; later give
--reference-results and choose ranks/axis/transport. The four-step dt=1e-3
checks compare stage q/cache/RHS, terminal fields and scalars, checkpoint on/off
isolation and 2+2 exact state/control/scalar restoration. Actual NP=1/inviscid,
NP=2 z/device/inviscid and NP=2 x/y/pinned/ME6+explicit10 pass; maximum canonical
RHS error is 6.67911e-13, terminal q 1.33214e-15. Ordinary NP=2 x ME6/filter
Memcheck reports zero errors on both ranks (no full leak-check). Existing central TGV and bc41-y
channel representative exact continuation checks still pass after the shared
stage-interface change.

## Profile Inflow And Legacy Upper Farfield Components

Build astr_backend_profile_boundary. prepare_profile_boundary_reference.py
freezes the original CPU/old-GPU bc11 profile and actual bc51 upper-y states;
run_profile_boundary_gate.py takes --binary, --reference, --mpiexec, --output,
optional --ranks 1/2, --axis x/y/z and --sanitizer memcheck/initcheck.
The new device target contains only the inlet plane, allocated on demand.
Complete-state and normal-Mach pressure modes are checked with the original
fields and independent oblique sub/supersonic and exactly sonic targets.
Only the selected boundary face may change or have its RHS reset.
The bc51 command reproduces the existing second-order upper extrapolation;
it is not a characteristic farfield. NP=1 and NP=2 x/y/z pass, including y
Memcheck and z Initcheck, with maximum field error 6.66134e-16. These are
component checks; ordinary static-profile plate coverage is recorded below,
not general SBLI or NSCBC admission. The probe now also checks original x-upper
bc21: extrapolated supersonic states, prescribed-pressure subsonic states and
the exactly sonic branch, with the original boundary-temperature rule.

run_shock_mask_faces_gate.py takes --mpifort, --mpiexec and --output. It
mechanically extracts exchange_shock_mask_faces from src/parallel.F90 and
checks all three axes at NP=1/2, including local periodic wrap, physical null
neighbors and distributed periodic/nonperiodic neighbors. The approved fix
swaps the two local interior send planes only for an undecomposed periodic
axis. CPU and CUDA Fortran use this same helper. The pre-fix test has 84 wrong
bytes at NP=1; both post-fix rank counts have zero. This does not yet deliver
the new C++ compact expanded-mask transport by itself. The subsequent GPU
transport/consumer evidence is recorded below.

prepare_expanded_mask_reference.py separately freezes the corrected original
CPU/CUDA-Fortran 743e/643c MP7-LD plate at NP=1 and NP=2 z. It takes
--reference, --mpiexec and --output, compares the three actual RK phases and
decomposition changes, then protects successful evidence read-only. The current
reference_fixed and compact_flat_frozen_valid records have maximum CPU/old-GPU
cache error 7.77157e-16 and RHS error 1.92819e-17. This is the corrected source
for the next C++ compact-mask transport increment, not proof that increment
already exists.

## Expanded Byte-Mask Transport And Interface Consumer

Build astr_backend_shock_mask_halo and astr_backend_upwind. The on-demand
shock_faces buffer has two one-layer planes per axis, not a second 3D mask.
astr_device_shock_mask_exchange reuses idle fixed HaloTransport slots with
MPI_BYTE. Pinned mode transfers only face bytes; device mode passes actual
device pointers. Undecomposed periodic axes wrap on device, physical ends copy
the physical marker, and no shared-node averaging occurs. expand invalidates
the old face layer. The upwind operator reads face values only after a completed
exchange, otherwise retaining the original nonexpanded rule.

run_shock_mask_halo_gate.py takes --binary, --mpiexec, --output and optional
--ranks 1/2, --axis x/y/z, --transport pinned/device, --sanitizer memcheck/initcheck.
Eight cases per rank cover periodic and mixed physical/periodic layouts, both
sync modes and changed masks. Physical masks, all six faces and every interface
flag are exact. The test-only CUDA observation calls the same private device
predicate as the actual upwind kernel; it is not a production transfer path.
At NP=1, 924 interface choices differ from the nonexpanded rule, ensuring this
test does not pass merely because the mask happens to be uniform at the seam.

pack_expanded_upwind_reference.py reuses the immutable corrected original
743e/643c results, without another advance. Feed its output to
run_upwind_components.py --compact-expanded. This checks 12 frozen-gradient
convection combinations per rank; it deliberately does not substitute explicit
gradients or viscosity into a compact prepared RK step. NP=1, NP=2 x/y pinned
and z/device have maximum RHS error 2.64546e-17. These component checks do not
by themselves prove the complete compact RK chain; that subsequent ordinary-main
coverage is recorded below. Neither is MB7 completion or a DCU runtime result.

Evidence: results/multi_backend_mb7_upwind_20261114, frozen_all/frozen_steps,
native_*, prepared_* and main_tgv_* reports. New source is not covered by the
preceding DTK snapshot. See progress 10.36--10.37 for remaining unimplemented
and unverified paths; ordinary-main selector availability is not broader
numerical, physical, CURVE, low-precision or DCU evidence.

## Ordinary Static-Profile Plate

The native FP64 main now connects x-inlet bc11, x-outlet bc21, lower bc41 and
upper bc50/51 with the original face ordering. Inlet targets are static planes;
dynamic inflow, wall blowing, NSCBC, high-order overrides, positivity and crashfix
remain separate pending consumers. For 743e/643c, the real compact gradient,
sensor byte halos, convection, compact viscous divergence and optional compact
filter run in the ordinary RK loop, not a frozen-gradient probe.

Build astr_backend_flat_statistics and use run_flat_statistics_gate.py with
--binary, --mpiexec, --output, optional --ranks 1/2, --axis x/y/z and tools.
Analytic checks use linear velocity/temperature on a stretched Cartesian mesh:
volume-weighted x momentum, wall shear, heat, area and unique-node spanwise RMS.
NP=1/2 x/y/z match the analytic integrals exactly for this binary-exact fixture.
Repeated calls preserve words and geometry addresses; x Memcheck and y Initcheck
have zero errors, with zero x device leaks. Main initialization uploads actual
CPU cell volumes and quadrilateral areas once, not full 3D coordinates. Derived
weights are rebuilt on restore and are not added to the checkpoint format.

run_cpp_plate_gate.py takes --binary, --reference (frozen original binary root),
--mpiexec, --output, --variant mp7_51/mp7ld_compact/mp7ld_compact_filter.
First use --freeze-only; then --reference-results with ranks/axis/transport,
--workspace scalar/full and --sync explicit/dependency. --advance-only is an
explicit no-formal-output diagnostic scope, not a restart check or a fallback.
The default full gate checks stage fields/caches/RHS, terminal fields and scalar
sequences, checkpoint on/off isolation and exact 2+2 continuation.

The approved static lreadgrid/ninit=3 743e/643c no-blowing bc11/21/41/50 case
now uses the existing read-profile Cartesian proof and resource inventory.
No new on-disk fields or legacy-format support were added. The copied resources
are geometry.h5, input.txt, grid.h5, inlet.prof and flowini3d.h5.
run_original_plate_restart.py checks the same extension on the original CPU
and CUDA Fortran binaries using --reference and --reference-results.

Evidence: results/multi_backend_mb7_plate_20261114, summary.json and
reference_output_numerics_unchanged.json. The 16-cubed four-step standard MP7
bc51 plate, compact MP7-LD plate and compact-filter plate have maximum phase
errors 2.66454e-15, 3.99681e-15 and 9.22487e-14 respectively. Representative
NP=1/2 x/y/z, pinned/device, scalar/full and both sync policies pass; each full
gate preserves exact state/control/scalars across restore and q/clock words
across output toggles. Original CPU/old-GPU compact/filter 2+2 checks also pass.
These bounded numerical checks are not a physical BL/SBLI validation, general
CURVE/NSCBC coverage, low-precision admission or a DCU device result.

## Static-Plate Consumer Products And Streamline Step Recovery

run_cpp_archive_gate.py --profile-fixture accepts a frozen four-step cpu/old_gpu
plate result root. The compact NP=2 y run compares two primitive field volumes
and 27 numerical slice frames (including Q/divergence/density gradients), with
maximum scaled error 7.39519e-14. Scheduled volume publication downloads its
existing primitive caches; numerical slices download only selected 2D results.
This is not a claim that images download the volume.

run_cpp_plate_products.py takes --binary, --reference-binaries, --fixture,
--mpiexec and --output. --freeze-only first protects the original numeric
consumer reference. Later provide --reference-results and ranks/axis; --images
uses --pipeline standard-device/direct-device with --renderer-reference or
explicit --renderer-results. --completed-run explicitly reuses an unchanged
completed full case; it is not a fallback or automatic retry. Only the new
working input copy is made writable, never its immutable source.

The gate re-integrates instantaneous wall samples over the clipped time window,
checks selected wall means/profiles against CPU/old-GPU products, observation
isolation, exact state/statistics/control continuation and numerical product
bytes. Actual strict-device images include instantaneous/mean sections, wall
shear, a positive-Q surface and normalized-domain layered streamlines. Receipts
must have nonempty geometry and zero image field/geometry readback; pinned
private stencil traffic is still allowed. EPS stays off. The current bounded
plate has 14 JPEG frames per image run, not a turbulent-physics validation.

After user approval, the shared postprocessing RK45 now recovers from an
accepted minimum step and suggests the configured upper bound for exactly
zero error. Cash-Karp coefficients/error estimates, requested limits/seeds,
termination classification and CFD RK3/CFL remain unchanged. This differs from
the two original VTK step-selection choices; do not require old-policy
streamline pixels to match the completed new trajectories.

Build astr_backend_trace_minimum for eight FP64 constant/circular forward/back
analytic traces and astr_backend_rk45_gpu for CUDA status/endpoint/floor-growth
checks. The original policy binary/header/log are immutable in rk45_original.
With the same minimum=1e-5, maximum=.02, tolerance=1e-8 and target arc=2,
old floor-start traces stopped at arc .1 after 10000 steps; new ones complete
in 101 steps, circle error <=7.09433e-14. FP64 and existing FP32 CPU/GPU
component checks pass; this is not native FP32 C++ flow-port delivery.

Evidence: results/multi_backend_mb7_plate_products_20261114. Final image
closures are rk45_recovery_np2_x_direct and rk45_recovery_np1_standard:
all 16 particles reach length at steps 0/2/4, with no step-budget truncation;
same-build exact restore pixels are equal. rk45_policy_isolation.json checks
unchanged CFD and statistics dataset words across the policy revision.
Three-dimensional y-axis ticks remain crowded in the shared display preset;
presentation polish was not accepted or modified. The optional old VTK probe
build lacked CUDA::cudart during generation; it was not repaired or counted
as passed. See progress 10.42--10.43 for exact scope and pending MB7 work.

The later actual Hermite7 isothermal-wall main chain is recorded in progress
10.61 and results/multi_backend_mb7_high_wall_20261115. It covers 643e/643e,
standard viscosity, unfiltered bc41 in x/y/z, representative NP=1/2 transports,
forcing, output isolation and exact restore; this is not delivery of high-order
open/SBP, adiabatic/slip or ME6 wall assembly. Local-f6 filtered evidence is
tracked separately from that unfiltered baseline.

Progress 10.62 adds the actual explicit10/local-f6 bc41 chain using the same
root. `prepare_high_wall_reference.py --filter` freezes original CPU/old-GPU
trajectories first; `--wall-axis x|y|z` rotates the weak boundary fixture, and
`--force-fixed` retains the named driving input. `run_cpp_high_wall_gate.py
--workspace scalar|full` checks all RK phases, operator RHS, terminal fields,
scalar clocks, output isolation and 2+2 exact restore against those references.
Six filtered main cases have maximum scaled stage difference 4.50931e-15;
the y NP=2 full-workspace main has clean Memcheck on both ranks. Scalar and
full y NP=2 checkpoint datasets agree bitwise across the sampled transports.
`astr_backend_boundary_filter wall41` checks 252 polynomial/source-immutability
cases (maximum 4.20740e-16); its unchanged default checks 504 basic polynomial
and Nyquist cases. The polynomial check is independent of the closure-table
copy; actual non-polynomial main references establish coefficient equivalence.
These results do not establish adiabatic/slip high-wall assembly, high-order
open/SBP, high-wall ME6 or high-wall compact-filter joint validation, nor DCU
device execution. No additional full-volume workspace is allocated.

Progress 10.63 supersedes the adiabatic/slip missing-main status for the
existing uniform Cartesian Hermite7 subset: bc42 x/y and bc421 y, standard
643e/643e viscosity, with and without explicit10/local-f6. Shared high-wall
state/RHS algebra is in high_wall_math.cuh; internal API names now describe
both wall families, without legacy API wrappers. The wall RHS consumes the
same-stage interior rates, with the original pressure-Neumann endpoint rows
and slip tangential mass/kinetic terms. No new volume buffer is allocated.

`astr_backend_high_wall_state REFERENCE 42|421` checks current authoritative-q
state projection against immutable CPU/old-GPU trajectories, unchanged density,
off-face q/RHS, and the normal Neumann constraints. Use the existing mpiexec
launcher even for one rank. `astr_backend_boundary_filter wall42` checks the
adiabatic filter polynomial/source identities. `run_high_wall_constant_gate.py
--wall-type 42|421 --filter` adds independent one-step constant states; the
slip case retains nonzero tangential velocities (0.1,0.02).

Eight ordinary main cases at results/multi_backend_mb7_high_wall_20261115
have maximum scaled stage difference 1.06457e-14 and final difference
8.86185e-15, with output isolation and 2+2 bitwise exact restore. Sampled
NP=2 pinned Memcheck runs for both 42 and 421 have zero errors on both ranks.
The first 42 main failed at launch (CUDA 701, 154 registers/thread with a
512-thread wall RHS block); only this new launch was changed to 256 threads.
Its failed directory is retained. Current source/binary/resource receipts are
in candidate_6c5f6ab and adiabatic_summary.json. Native bc41 and low-order
compact KEEP/ME6 LDC representative main results remain within their gates.
This does not establish high-order open/SBP, high-wall ME6, high-wall compact
filter joint coverage, new high-wall in-situ images or DCU device execution.

Progress 10.64 adds the existing FP64 KEEP SBP63/SAT weak-wall chain, separate
from strong Hermite7 projection: sixth-order interior, third-order six-row y
closures, diagonal H, gradient lift and inviscid/viscous isothermal SAT. The
normal table uses global y indices across rank interfaces. Ordinary bc41 state
projection is skipped; existing stress/heat storage and halo traffic are reused.
Current evidence is uniform Cartesian, two y bc41 walls, x/z periodic,
643e/643e standard viscosity, no filter. The test-only steady MMS source is
not yet connected and is explicitly rejected when requested on the new main.

`prepare_sbp63_reference.py` freezes original CPU/old GPU trajectories, tensor
operator checks and RK budgets. `--perturbed-wall` only changes the private
fixture to excite nonzero wall velocity/temperature SAT terms. Its first
independent check exposed a test-oracle transport mismatch (110.4 K versus
the actual default Sutherland 110.3 K); only that oracle was corrected, not
solver transport. Failed data remain in reference_perturbed. Existing base
data were reanalyzed with the corrected oracle without overwriting them.

Use `pack_high_wall_gradients.py` then `astr_backend_boundary_gradient ROOT
sbp63` for same-state gradient, source-immutability and exterior-NaN checks.
`run_cpp_high_wall_gate.py --wall-mode keep_sbp63` checks actual RK stages,
operator RHS, fields/scalars, output isolation and 2+2 exact restore. Evidence
is results/multi_backend_mb7_sbp63_20261115: six actual main runs, NP=1/2 x/y/z
representatives with pinned/device MPI, maximum scaled stage/RHS difference
3.30696e-15, final difference 1.11023e-15, and clean NP=2 y Memcheck on both
ranks. Original perturbed-reference tensor checks are within 7.54121e-13.
This is not new long-window physical, stretched-SBP, steady-MMS, in-situ-image
or DCU-device validation. SBP wall nodes need not equal the strong wall state.

Progress 10.65 connects the existing x+/y+ layer sponge. This is the original
post-RK six-neighbor conservative-variable smoothing, not a continuous RHS
relaxation term. Each active direction first refreshes all ranks' fixed hm q
halo without shared-node averaging; x precedes y. Static coefficients and one
max-region packed five-variable work buffer are allocated once. No sponge
volume buffer is allocated when off, and no full field is read back to advance.
The post-update callback now runs sponge, then existing strong-wall projection,
then physical primitive reconstruction/check. CPU/old-GPU sponge math is unchanged.

Build the original CUDA Fortran `sponge_reference_probe` through root CMake.
`run_sponge_components.py --freeze-only` calls it for NP=1/2 x/y/z and freezes
both CPU/old-GPU arrays; ordinary mode compares `astr_backend_sponge` against
those fields with pinned/device MPI, both sync modes and constant identities.
Eight invocations (15 rank executions, 90 mode/sync checks) pass with maximum
scaled difference 2.96060e-16; pinned NP=2 y full Memcheck/leak-check is clean.
The comprehensive old MP-LD driver hit a compiler signal 11; its temporary
changes were removed rather than changing solver math. The small original
sponge driver compiles and executes only the affected functions.

`run_cpp_open_shock_gate.py --sponge-width N --checkpoint` checks actual x+
stages/output/restore; `--transverse 16` permits the representative y/z splits
with hm=5. Frozen old-GPU references use `--freeze-only` without formal output,
because that frozen output entry rejects this case. Five actual new main runs
have maximum updated-state difference 1.15135e-15, RHS 1.00614e-16, statistic
4.79829e-13, with observation/output isolation and 2+2 exact continuation.
`run_cpp_channel_sponge_gate.py` adds a private y+ channel fixture against its
actual CPU reference; old GPU channel does not apply this sponge. The NP=2 y
device run has maximum stage difference 2.18715e-15 and exact output/restore
checks. References and failed setup directories remain distinct in
results/multi_backend_mb7_sponge_20261115; summary.json records their scope.

No new checkpoint file/history is needed: coefficients derive from unchanged
input/geometry, work is overwritten. Exact evidence requires the same build,
configuration and topology, not arbitrary changed widths. Other directions and
circle are unimplemented; full M12/SBLI, active sponge/high-wall/KEEP/compact/
main-filter joint cases, new in-situ products and DCU execution remain unverified.
Off-mode SBP63/Hermite42 representative main trajectories and exact restore pass.

## Classified NSCBC Outlet RHS Component

astr_backend_nscbc_outlet exercises the new resident x+ bc22 RHS command in
boundary_state.cu. It preserves the existing low-order characteristic matrix,
metric correction, local-normal wave classification, pressure relaxation and
backflow target/rates. It adds negative divergence to RHS without changing q,
allocating a volume or copying fields to the host. This command alone is not
ordinary-main NSCBC admission or a full boundary/filter stage implementation.

run_nscbc_outlet_components.py --cpu-build BUILD --gpu-build BUILD --output OUT
--mpiexec MPI freezes actual original routines using their current objects.
Use --reference OUT --binary BUILD/bin/astr_backend_nscbc_outlet --output NEW
to compare the new component. Select --ranks 1/2 and --axis x/y/z consistently
for freeze and comparison. --sanitizer memcheck/initcheck records per-rank logs.
The references include source/binary hashes; freeze directories become read-only.

Twenty states per layout cover sub/supersonic outflow/backflow, grazing/sonic
limits, a pressure target ratio, strong tangential speed, gradients and full
metric data. Two synchronization modes give 40 checks per rank. The uniform
Cartesian reference also has an independent acoustic/entropy decomposition,
without either implementation's characteristic matrix. Physical transverse
edges are not part of this first matrix: it uses periodic/interior y/z types.
The test uploads sliced global states, not a real distributed halo exchange.

Evidence: results/multi_backend_mb7_nscbc_x_20261114. Native NP=1 and NP=2 x/y/z
have maximum component-scaled difference 1.87878e-13 against CPU/old GPU.
NP=2 x memcheck has zero errors/leaks; NP=2 y initcheck has zero errors.
Nonowners/interior RHS and q are unchanged. Full outlet filtering, top/corner
assembly, main time advancement/restart and DCU checks remain outstanding.
The generic CPU bc12 formula is not the approved, repaired old-GPU bc12 policy;
do not use that CPU routine as a matching inlet oracle or silently change it.

## NSCBC Outlet Plane Filter And Halo Lifecycle

astr_backend_nscbc_filter checks the existing six-order transverse outlet
filter against directly linked original CPU/old-GPU routines. Freeze with
run_nscbc_filter_components.py --cpu-build BUILD --gpu-build BUILD --output OUT
--mpiexec MPI, then provide --reference OUT --binary BUILD/bin/astr_backend_nscbc_filter
--output NEW. Select --ranks 1/2, --axis x/y/z and --transport pinned/device.
The existing low-order physical closures and imposed lower-wall value stay
unchanged; high-order boundary filters are not part of this command yet.

Only an outlet owner prepares a five-component 2D FP64 scratch plane, with
40*(jm+1)*(km+1) bytes. It is fully overwritten on each call, reused and freed
with State; no volume qwork or checkpoint state is added. Caller-owned actual
q halo exchange precedes/follows filtering, followed by primitive refresh.
The probe poisons exchanged axial halos, uses the real communicator/transport,
checks nonoutlet words, and checks physical/valid-halo thermodynamic and
velocity caches. Bind before MPI_Init, initialize unused exterior caches and
use fluid_refresh_primitives_boundary, not just physical-node primitives
followed by an uninitialized full-cache download.

Evidence: results/multi_backend_mb7_nscbc_filter_20261114. corrected_* runs
cover NP=1/2 x/y/z, both sync modes and representative pinned/device transports,
with maximum error 4.44090e-16. Pinned x memcheck has zero errors/leaks;
pinned y initcheck has zero errors. Device y/z numerical checks pass, but
device+Sanitizer is NOT a clean tool result: initcheck does not support the
selected CUDA IPC allocations, and device memcheck still exits 99 on UCX
Fabric/VMM capability queries plus IPC mapping records. Keep those logs; do
not suppress API errors or initialize receive storage to manufacture a pass.
See migration progress 10.45 for the observed callsites and tool limitation.
Ordinary-main NSCBC staging, restart and new DCU evidence remain pending.

## Repaired NSCBC Inlet And Ordinary X-Open Main

astr_backend_nscbc_inlet builds the shared nscbc_outlet_probe.F90 with
TEST_INLET. run_nscbc_inlet_components.py first uses --gpu-build BUILD to
freeze the actual repaired old CUDA kernel; later use --reference OUT and
--binary BUILD/bin/astr_backend_nscbc_inlet. The unrepaired generic CPU bc12
is not an equivalent oracle. An independent uniform-state acoustic/entropy
projection checks incoming-only selection without the characteristic matrix.
NP=1/2 x/y/z give 48 checks per rank, maximum scaled difference 1.02040e-14;
x memcheck and y initcheck are clean. Both q and nonselected RHS stay unchanged.

The ordinary main now connects the existing inviscid MP7 x-NSCBC combination
12/22/1/1/1/1, including outlet filtering, real halo reconciliation/refresh,
primitive caches, once-only boundary RHS and RK. Other characteristic/wall/top
and viscous/compact combinations remain pending. No bulk host RHS fallback
or alternate boundary formula is introduced. Use imported exchange C ABI
inside the module, not its own BIND(C) wrapper (the recorded NVHPC issue).

run_cpp_open_shock_gate.py --binary OLD --freeze-only --output OUT --mpiexec MPI
freezes the actual ordinary old-GPU reference. Then provide the new --binary
and --reference-results OUT, selecting ranks/axis and pinned/device transport.
The current 64x8x8 intervals allow NP=2 x, not y/z with the fixed five halos.
The comparison uses full post_source RHS and post_update q at matching stages,
not pre_rhs observations taken on opposite sides of the outlet filter.
--checkpoint adds current-format output isolation and exact 2+2 continuation;
the static inlet and overwritten outlet plane add no authoritative sidecar.

Evidence: results/multi_backend_mb7_nscbc_inlet_20261114 and
results/multi_backend_mb7_open_shock_20261114. Four steps at dt=1e-5 use the
existing Ma=3 stationary shock and classified bc12/bc22. Native NP=1 and NP=2
x have full RHS difference <=6.24501e-17, stage q <=6.11180e-16, and unchanged
clock/observation state. Pinned main memcheck has zero errors. Final checkpoint
closures (NP=1 pinned/explicit, NP=2 device/dependency) preserve state/control/
scalar words across exact restore and q/clock across separate output/observation
toggles. This is not a long-time physical shock validation or new DCU evidence;
full top/corner/viscous NSCBC and the remaining MB6/MB7 goal still need work.

## Upper-Y NSCBC RHS And Normal-Mach Reduction

astr_backend_nscbc_top compares compatibility (0), target relaxation (1,
ordinary or existing shock targets) and nonreflecting (2) with actual CPU
farfield_nscbc and old CUDA production kernels. Target relaxation and
source-balanced nonreflecting are distinct; GRCBC complete-RHS and viscous
source projection are not implemented by this component. keep_corner=0 is
tested; the nonzero ownership branch still needs actual corner validation.

Run run_nscbc_top_components.py --cpu-build BUILD --gpu-build BUILD --output OUT
--mpiexec MPI to freeze, then --reference OUT --binary BUILD/bin/astr_backend_nscbc_top
--output NEW, with matching --ranks 1/2 and --axis x/y/z. The original RHS
reference uses lfilter=t to isolate it from transverse filtering. The default
16x16x16 intervals have two x-end layouts, four policy/target selections and
12 states. Both sync modes against both references give 384 checks per rank.
Uniform cases also use an independent acoustic/entropy decomposition.

Shock targets are an explicitly uploaded, derived conservative 2D plane,
constructed from the existing geometric/postshock helper during reference
preparation; there is no new volume coordinate copy or public field id.
The target plane is allocated only on upload. A separate optional eight-byte
work value supports the physical top-face normal-Mach2 reduction; the result
is a scalar readback plus the existing status check, not a field readback.
The caller owns top-rank selection, MPI_MAX and the pre-filter sampling phase.

Use --reduction-only --plane-intervals 32 --ranks 2 --axis y to check the
independent 32x16x32 fixture, with the unique Mach2=4 peak in the last top
node across three 512-thread blocks and the lower rank contributing zero.
producer_consumer_np2_x/y feed actual GPU/MPI reduction values into the RHS.

Evidence: results/multi_backend_mb7_nscbc_top_20261114. NP=1/2 x/y/z maximum
scaled difference is 5.26571e-12; uniform independent absolute error is
5.29483e-12 (unchanged 2e-10 gate). x memcheck has zero errors/leaks, y
initcheck has zero errors, and the multi-block analytic maximum is exact.
Full q and unselected RHS are unchanged. The component's state slicing is not
a claim of halo communication; the Mach2 MPI reduction is actual.

The ordinary main still does not admit bc52. Top x/z filters, stage cache
refresh, viscous snapshot/projection, joint corners and actual main/restart
remain pending. The existing x-NSCBC checkpoint closure was rechecked after
the private State extension, not counted as a top-boundary closure. The DTK
build list includes this component but no new DCU result is claimed.

## Upper-Y X/Z Transverse Filter Sequence

Build astr_backend_nscbc_top_filter and use run_nscbc_filter_components.py --top.
The freeze mode directly links original CPU spafilter6exp and old GPU x/z
kernels; the comparison mode uses --reference/--binary, ranks/axis and
pinned/device options as for the outlet filter. Six input fields generate
twelve x/z phase records. Same-backend z input physical words must equal its
computed x output; cross-backend computed input differences use the unchanged
2e-10 gate, not a false bitwise-input requirement after floating-point work.

The device chain uploads only its initial input, poisons exchanged halos,
filters x, performs real halo refresh, then filters z without another reference
q upload. Primitive caches and valid axial halos are checked afterward;
non-top physical words are unchanged. A single optional FP64 top scratch
plane is reused by both axes and freed with State, not a volume qwork.
NP=1/2 x/y/z, both sync modes and representative pinned/device paths have
maximum error 6.46732e-16; x memcheck and y initcheck are clean. Evidence is
results/multi_backend_mb7_top_filter_20261114/reference_phased and cpp_*.

This is NOT the old complete bc52 RHS order. Original CPU diagnostics exposed
an additional defect: its second transverse filter is followed by RHS before
halo refresh. run_nscbc_top_halo_phase.py --cpu-build BUILD --output OUT --mpiexec MPI
calls actual unchanged CPU functions. NP=1/2 z q and physical caches are
identical, but original RHS differs by 1.33514e-3; refresh/recompute gives zero
partition difference. See cpu_halo_phase_cache and migration progress 10.50.
CPU/old-GPU repair awaits explicit approval. Do not port that stale-halo order,
change the reference or claim bc52 ordinary-main/restart closure meanwhile.

## Nonreflecting Viscous Increment Snapshot/Projection

Build astr_backend_nscbc_top_viscous and select --viscous-source in
run_nscbc_top_components.py. This mode directly freezes the original CPU
capture/apply and old GPU production kernels. It adds a manufactured
zero-mass source vector after the actual nonreflecting convective boundary
RHS, then checks only removal of its incoming characteristic projection.
It is not a physical viscous-flux calculation or a main-program closure.
Uniform states additionally use independent acoustic/entropy decomposition.

The C ABI nscbc_top_prediff_capture saves one optional FP64 top-face RHS
plane; nscbc_top_viscous_source consumes it once after viscous assembly,
without changing q or downloading a field. Missing/consumed snapshots fail;
valid captures can overwrite the derived workspace. The future main caller
must preserve the same q/cache phase and complete the operation before
full-step publication. No checkpoint authority/sidecar is added.

Evidence: results/multi_backend_mb7_top_viscous_20261114. NP=1/2 x/y/z,
both sync modes and both original references give 96 checks per rank,
maximum scaled difference 1.60757e-12, independent absolute 5.30687e-12.
Zero source leaves RHS words unchanged; nonowners and non-top nodes retain
their intended source increment; x memcheck/y initcheck are clean.

The complete-box gate is prepared in run_cpp_nonreflecting_box_gate.py.
Its original CPU/old-GPU run is NOT a complete frozen reference: old GPU
enstrophy/dissipation are NaN, while CPU values and both physical fields
are finite. --analyze-reference reuses completed runs without advancement
and preserves rejection on nonfinite values. See rejected_reference.json
and migration progress 10.52. The old GPU gradient-selector repair awaits
approval separately from N7; do not waive the scalar gate or port that
selector omission. No new C++ main/restart result is claimed for this box.

## Low-Order Slip-Wall States And Main Consumers

Build astr_backend_slip_wall through the root CMake. Freeze the actual CPU
and CUDA Fortran functions before comparing the new device implementation:

```sh
python3 tests/gpu_backend/run_slip_wall_components.py \
  --cpu-build build_cpu_probe --gpu-build build_gpu_probe \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/slip_reference --ranks 2 --axis y
python3 tests/gpu_backend/run_slip_wall_components.py \
  --reference results/slip_reference \
  --binary build_backend_cpp_native/bin/astr_backend_slip_wall \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/slip_candidate --ranks 2 --axis y --sanitizer memcheck
```

The reference covers 24 states: Cartesian/tilted/varying normals, codes
411/421, both sides and two temperatures. Each new rank compares 96 cases
across two sync modes and two references; untouched q/cache words and a
nonzero RHS sentinel are checked. The original GPU kernel also clears its
selected RHS, unlike the CPU state function, so only its q/cache output is
the state oracle. The C++ state command preserves RHS, like wall41/42.

411 uses a private lower-wall byte mask uploaded once at initialization
and rebuilt from restored static geometry. No full coordinate/normal
volume or checkpoint sidecar is added. Both slip families use normalized
resident metric row two, not global-v suppression. Blowing and high-order
branches remain unconnected.

prepare_wall41_reference.py accepts --boundary-type 411/421 --wall-axes y;
--xslip selects the 411 split. A split inside periodic x is not a valid
mixed-wall trajectory fixture: its duplicated endpoints have different
conditions. Use uniform lower-slip for periodic main tests; mixed-mask
equality and both branches are component checks, not a nonperiodic main
claim. The ordinary main gate remains run_cpp_physical_gate.py.

Evidence: results/multi_backend_mb7_slip_wall_20261114, progress 10.53.
NP=1/2 x/y/z components pass, max scaled 4.33109e-16, memcheck/initcheck
clean. Filtered KEEP411 main passes NP=1/2 x/y with both workspace and
halo modes sampled. Cartesian KEEP421+ME6+compact derivative/filter passes
NP=2 y with device halo/PCR transport. Output isolation and 2+2 exact
restart are checked for each successful main configuration.

The standard/ME6 y-wavy 421 main passes only NP=1. NP=2 y fails already
in the first convective RHS. check_geometry_face_halos.py compares saved
internal face halos with the published owner state; both this new run and
an unchanged original CPU NP=2 run have max metric difference ~0.600.
geom.F90 exchanges before shared-node averaging, with no final refresh.
See progress 10.54/N9: the common CPU/GPU repair awaits human approval.
Do not waive this gate, call the multi-rank wavy case passed, or apply a
C++-only repair. No new DCU results are provided by the build-list entry.

## RTI Fixed States And Density-Weighted Gravity

The new fixed_rti_y command ports only the original y-face bc31 states:
lower rho/p=2/1, upper=1/2.5, zero velocity and the existing EOS. It is a
state-only command and preserves RHS. fluid_rti_source adds rho*J to y
momentum and rho*v*J to total energy, preserving the original +y unit
gravity normalization and CPU multiplication order. It is not the
channel's density-independent force. No new arrays or checkpoint history
are needed; no source calculation is delegated back to the host.
RTI retains the original generic conservative-variable extrema output,
without applying TGV-specific reference-velocity normalization.

Build astr_backend_rti. run_rti_components.py has the same freeze/compare
CLI as run_slip_wall_components.py. It links the actual original fixbc,
src_rti and CUDA Fortran production kernels, freezing six states per rank
before the new commands are compared. Two thermodynamic parameter sets,
spatial rho/J variation and positive/negative/zero v cover 24 checks per
new rank across both sync modes and both references. The nonzero boundary
RHS sentinel is separate from the old GPU kernel's zero-face behavior.
NP=1/2 y, max scaled/independent absolute errors 5.55112e-17; memcheck and
initcheck are clean.

run_cpp_rti_gate.py uses the original RTI initialization, three-dimensional
extrusion and generated 0.25x1x0.25 box at 16^3, dt=1e-6, four complete
steps. The migration probes choose KEEP643e unfiltered or KEEP643c with
compact10; they do not claim the example's original 543c default as a
tested format. Use --freeze-only first with --reference pointing to the
immutable CPU/old-GPU bin directory's parent. For example:

```sh
python3 tests/gpu_backend/run_cpp_rti_gate.py \
  --binary build_backend_cpp_native/bin/astr \
  --reference results/multi_backend_mb7_plate_20261114/reference_output \
  --reference-results results/multi_backend_mb7_rti_20261114/main_compact_filter_reference \
  --template 'examples/Rayleigh–Taylor-Instability/datin' \
  --output results/rti_compact_candidate --variant compact-filter \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --ranks 2 --axis y --transport device --sync dependency
```

Evidence: results/multi_backend_mb7_rti_20261114, progress 10.55. Explicit
NP=1/2 x/y/z and compact-filter NP=2 y check actual stages, fields, extrema,
output on/off isolation and 2+2 same-build exact restore. Compact max field
scaled difference is 4.07374e-14; extrema max is 4.82772e-13. An initial
command wrongly selected the explicit reference for a compact run; its
failed directory is retained. Reference variant identity is now checked
before MPI startup, rather than comparing different discretizations.
These are bounded numerical/runtime checks, not long-time RTI physics,
2D C++ support, new in-situ product admission or new DCU execution evidence.

## LDC Lid And Two Physical Axes

ldc_top_lid_y ports the shipped userdefine.F90 upper-y LDC rule, not a
general Fortran UDF executor. It sets u/T=1 and v/w=0, extrapolates pressure
from two inward nodes and reconstructs density/energy with the existing
EOS. The state-only command leaves RHS untouched and allocates no arrays.
The main applies x walls, lower y and then the lid, preserving the CPU's
lid-owned side-wall intersections. It does not regularize the idealized
corner velocity discontinuity. Generic extrema, not TGV normalization,
remain the LDC flowstate output.

prepare_ldc_reference.py freezes actual original CPU/old-GPU four-step
results using the existing 16^3 three-dimensional weak-perturbation fixture.
It checks every-stage q/cache/RHS, terminal q, extrema, complete-step clocks
and independent lid/wall targets before making the reference read-only.
--variant explicit-filter uses KEEP643e/nodal viscosity/explicit10;
compact-me6-filter uses KEEP643c/ME6/643c gradients/compact10. Both use the
1x1x1 box, Ma=.3, Re=3000, dt=1e-5, four complete steps, bc41/41/41/0/1/1.
This fixture is not the completely resting ldcavityini initial state.

Reuse run_cpp_physical_gate.py with either frozen directory as --reference;
its cpu_y/gpu_y labels refer to the fixture's lid normal, not a restriction
to one physical axis. Work inputs are copied without inherited read-only
file permissions and only the new datin directory is made writable. The
original frozen reference is never modified.

```sh
python3 tests/gpu_backend/run_cpp_physical_gate.py \
  --binary build_backend_cpp_native/bin/astr \
  --reference results/multi_backend_mb7_ldc_20261114/compact_me6_reference \
  --output results/ldc_candidate \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --ranks 2 --axis y --transport device --compact-transport device-aware --workspace full
python3 tests/gpu_backend/check_cpp_ldc_walls.py results/ldc_candidate/continuous \
  --report results/ldc_candidate/walls.json
```

check_cpp_ldc_walls.py checks all 12 saved RK states by global coordinates,
including lid/side intersections and replicas on MPI partitions. Evidence:
results/multi_backend_mb7_ldc_20261114, migration progress 10.56. Explicit
NP=1/2 x/y/z and compact-ME6 NP=2 y pass, sampled across transport/storage
and synchronization modes. Max field scaled error is 6.51024e-14, extrema
4.90824e-13; wall u/T absolute error <=4.44090e-16. Each successful case
checks output isolation and 2+2 exact continuation. Pinned NP=2 y memcheck
is clean. No general UDF, 2D, curved LDC, long-time physics, LDC rendering
or new DCU numerical coverage follows from these results.

## High-Order Wall Reference Preparation Only

prepare_high_wall_reference.py reuses the existing HB1 initial-field
preparation and input conversion, then runs only the immutable original
CPU/old-GPU binaries. It selects actual hermite7, standard643e/643e,
viscosity on, filtering/crashfix/means off, fixed force zero, 16^3 and four
steps at dt=1e-5. --wall-type 41/42/421 freezes one original family, with
stage q/cache/RHS, gradients, terminal state, scalar and clock records.
It does not run the imported file's pytest matrix or implement a new
C++ high-order kernel. Output buffer budgets are not artificially capped.

References: results/multi_backend_mb7_high_wall_20261114, progress 10.57.
CPU/old-GPU maximum scaled stage differences are 1.72432e-15, 1.76595e-15
and 2.23173e-15 respectively; complete-step clocks agree. KEEP+Hermite7
remains unimplemented in the original solver and is not made admissible
here. The existing KEEP/SBP path is distinct. New C++ high-wall state,
gradient/flux rows, RHS/RK constraint, local-f6 filtering and exact restart
still require their own actual implementation and evidence.

The first new state-only prerequisite is wall41_evolved_density_state:
it retains q density and imposes zero velocity and Tw, without derivatives,
RHS writes or workspace. Build astr_backend_high_wall_state and pass the
frozen reference41 directory. The probe reads actual initialized/pre_rhs
records, changes face primitive caches to 77 before the call, and verifies
that q, not those caches, is the authoritative density. Both sync modes
and both original references give four comparisons, max scaled 2.17691e-16;
rho/velocity/Tw identities are exact, density and off-face q bits plus the
nonzero RHS sentinel are unchanged. Memcheck/leaks are clean.

Only NP=1 y/Tw=1 is covered. This does not enable the C++ high-wall main,
claim x/z validation or implement 42/421, wall derivatives/RHS, RK
projection, local-f6 or restart. The current DCU build snapshot through
LDC predates this state component; do not attribute it that build evidence.

fluid_gradient_high_wall is the next component prerequisite. It corrects
the designated normal axis's first/last three rows using exactly the
24 boundary_derivative6_weights entries, then retains the full metric
projection. It uses the existing gradient fields plus a 192-byte constant
table, not a new volume buffer. Velocity/temperature inputs are unchanged.

pack_high_wall_gradients.py packs actual first-step three-stage data from
reference41 without advancing either original solver. Run the existing
astr_backend_boundary_gradient with the packed directory and hermite7
(a diagnostic selection, not new production admission). Original explicit
and compact diagnostic selections are unchanged. The packed output names
remain cpu_y/gpu_y. Both references, sync modes and physical-exterior
normal/NaN inputs give 24 comparisons, max scaled 5.66214e-15; independent
quadratic projections give 40 checks, max absolute 7.86698e-15. Full tensor
manufacture checks projection algebra, not real CURVE geometry. Memcheck
and leaks are clean. The current executable's explicit selection also
reuses immutable low-order references to check the unchanged base path.

Evidence: gradient41_memcheck, basic_gradient_unchanged and summary.json
under the high-wall result root, progress 10.59. There is still no new
C++ high-wall main, 42/421 actual-gradient input validation, cross-rank
high-wall stage, wall flux/RHS/RK/local-f6 or high-wall exact-restart gate.
The DCU frozen snapshot through LDC does not include this later component.

After adding both components, the current ordinary main was rebuilt and a
16^3/four-step NP=2 y low-order LDC check retained max stage difference
3.32357e-15 plus output isolation and 2+2 exact continuation. This checks
the unchanged production selection, not a high-order main admission.

## DTK Build Snapshot Through LDC

results/multi_backend_mb7_ldc_20261114/dtk_build records the isolated
~/weiph/mb67_boundary_build_20261114.W6i1Ke build. DTK26.04 GPUfusion/gfx906,
GNU12.2.0, private OpenMPI4.1.5/HDF5 1.14.6 and CMake3.20.4 produced
ordinary astr plus all 36 requested probes. Final MB0 dependency paths
resolved. This is build/link evidence only: no GPU application was run on
the login node and no Slurm command was executed. It predates the later
high-wall state and gradient components, and does not replace DCU numerical,
communication or restart checks. See migration progress 10.60.

The default build helper links the entire probe list, causing repeated
whole-library device LTO. For a scoped increment, invoke root CMake with
the actually affected --target list instead of rebuilding every old probe.
Do not interpret a live linker as a stalled simulation. After verified
remote completion, one stale local SSH observer was closed; no remote
build/job was terminated. The original channel's exit status was not
observed, so report.json preserves the independently verified target and
dependency evidence rather than inventing a zero command status.

## Low-Order Bottom-Wall Forcing

The C++ main now reuses the common Fortran prescribed wall velocity for
current hash, fixed-phase modal and legacy rank-local random forcing.
Only a scalar bottom-wall plane is uploaded; CFD state reconstruction and
normal projection remain on the device. The plane is allocated once on
the owning ranks, and observation/output do not draw random numbers.
The active implementation is lower bc41, not a high-order or slip-wall
forcing implementation.

prepare_wall_forcing_reference.py freezes original CPU/old-GPU four-step
channel or static-profile plate trajectories. Match the MPI topology for
legacy_random: a different decomposition is a different RNG realization.
run_cpp_wall_forcing_gate.py checks stage q/cache/RHS, scalar/clock data,
observation and output isolation, and optionally 2+2 exact continuation
including the existing control.bin RNG payload. Formal output uses the
existing static-profile resource and Cartesian-geometry checks, not a
new checkpoint format. Build astr_backend_wall_forcing for the independent
signed-speed/unit-normal projection and off-face/RHS footprint check.

Migration progress 10.66 and
results/multi_backend_mb7_wall_forcing_20261115/summary.json record seven
current-build runs, NP=1/2 x/y/z, pinned/device and explicit/dependency
representatives. Max stage scaled difference is 2.66454e-15; the normal
projection probe gives 1.37065e-16. Pinned NP=2 y and the standalone probe
have clean Memcheck/leak-check logs. The device-MPI sanitizer attempt has
UCX/MPI driver-probe errors and IPC mapping leaks, retained separately;
successful native device-MPI numerics do not resolve that tool result.
These short tests do not certify full M12/SBLI, active forcing with
high-order walls or in-situ images, cross-compiler stochastic identity,
or DCU execution.

## Time-Series Inlet Cache And Continuation

The new C++ main supports the existing turbinf=intp inlet through
src_gpu/inflow_timeseries.F90. The inlet communicator reads and uploads
one five-variable 2D slot at a time; four slots remain on device. The
boundary kernel evaluates the original cubic prescription on the fixed
profile, including complete-state/mach-pressure choices. It does not
download CFD fields or delegate advancement to the CPU.

prepare_dynamic_inflow_reference.py freezes matching original CPU/old-GPU
16^3 four-step fixtures, including a genuine source-window rollover.
It accepts --ranks 2 --axis x|y|z or --automatic. The initial dt=6e-6
setup did not reach the pre-RK rollover threshold and is retained, not
counted as rollover evidence; the accepted fixed fixture uses dt=8e-6.

run_cpp_dynamic_inflow_gate.py checks actual stages, scalar/clock data,
observation/output isolation, and --checkpoint adds 2+2 continuation of
state, existing role-8 inflow.h5, controls and scalars. --restart-mode
nonexact uses the established reset of the time-step controller, so
same-step trajectory comparisons use fixed dt. --prefix-binary selects
an explicit old-GPU producer for the tested non-exact migration path;
it is not an exact cross-build or CPU-cache conversion claim.

Build astr_backend_dynamic_inflow for eight independent cubic-polynomial,
slot-order, sync and sub/supersonic pressure-selection checks, including
slot transfer bits and off-inlet q/RHS footprints. Migration progress
10.67 and results/multi_backend_mb7_dynamic_inflow_20261115/summary.json
separate the initial and final build hashes, matrix, failed test setups,
same-backend exact continuation, and old-GPU non-exact migration.

Representative fixed NP=1/2 x/y/z and automatic NP=1 stage differences
are at most 2.66451e-15; the independent probe gives 5.22458e-17. The
probe and final pinned NP=2 y Memcheck/leak-check logs are clean. The
test does not certify full physical inflow statistics, active-inlet
filter/positivity/crashfix/high-order-open-boundary or image products,
real multi-rank CURVE, CPU-format inlet-cache migration, or DCU execution.

## Ordered Crashfix Prerequisite

crashfix_reference_probe (original CUDA Fortran build) calls unchanged CPU
and old-GPU production repair functions. prepare_crashfix_components.py
freezes NP=1 and NP=2 x/y/z records: four CPU/old-GPU nearest-neighbour
cases, plus old-GPU-only distant/remote donor and overflow recovery.
The latter are not presented as CPU-equivalent algorithms. Original
nearest fields match; expanded GPU cases match known donors within
2.22045e-16.

Build astr_backend_crashfix through root CMake and launch with the matching
reference/np1 or reference/np2_x|y|z directory. It tests both sync modes,
state/cache and history against both references, repeat-call bit isolation
and unchanged RHS/RK storage. Ordering is the existing in-place k/j/i and
kk/jj/ii, not a simultaneous average. Remaining invalid nodes are reported
as unresolved; an API success code is not a full-recovery claim.

Progress 10.68 and results/multi_backend_mb7_crashfix_20261115/summary.json
record 56 local cases with max scaled difference 0, byte-exact masks,
repeat q/history, clean final Memcheck/leak-check, Initcheck and Racecheck.
The first Initcheck failure exposed four C-ABI padding bytes in the
88-byte report. One initialization-time report memset fixes its full
transfer without changing field values; failed logs are retained.

The ordinary C++ main still rejects active crashfix. Expanded search,
remote donor propagation and shared-node repair are now covered below;
critical-mask MP flux, runtime activation and checkpoint integration remain
to be implemented and checked. No repair arrays are allocated until requested. The inactive
dynamic-inlet main/restart check is not an active-repair CFD gate. No DCU
execution is claimed.

## Expanded Recovery And Remote Donors

crashfix_runtime.F90 coordinates the original repair sequence through the
C ABI: ordered pass, global invalid-state reduction, halo-only q exchange,
expanded device pass, and global progress. It does not average shared nodes
before invalid replicas are repaired. Bound clipping, shell traversal,
online mean and the original one-point donor selection retain their meanings.
No-progress returns a nonphysical status, not a successful replacement state
or an arbitrary retry loop.

run_crashfix_components.py --mode expanded accepts the frozen reference
directory and the new probe. --layouts selects representative entries from
its manifest; --transport pinned|device chooses the existing backend.
--sanitizer memcheck|initcheck|racecheck is explicit; use pinned for the
established clean memory-tool environment. --mode ordered retains the
earlier whole-array gate. --mode no-donor checks collective rejection of
an all-NaN state without inventing q values. There are no added run/disk caps.

Progress 10.69 and expanded_summary.json under the crashfix result root
record 96 pinned rank/cases and 56 device rank/cases, max scaled difference
2.22045e-16 across q/cache including halo phase, exact mask bytes and repeat
isolation. Rounds, expanded-node totals, maximum radius and single-donor
counts match original logs. The remote-only donor cases exercise x/y/z
search radii 17/9/1 and 2601 recovered nodes; they are actual communication
checks, not a re-labelled local-neighbour test. The one-point branch has
count 0 in both implementations, so it has not been independently activated.

Final Memcheck/leak-check, Initcheck and Racecheck logs are clean. The
ordinary repair-enabled main is still not connected: stage input/filter lifecycle, critical-mask flux
consumption and checkpoint/mode transitions remain. The inactive random
wall-forcing main and exact restart check does not replace those gates.

## Reconciled Recovery And N10 Decision

prepare_crashfix_components.py --reconciled freezes the original GPU
shared-node repair lifecycle, not the CPU single-pass algorithm. The
non-fluid fixture is excluded because the ordinary C++ immersed-boundary
path is not connected. Run run_crashfix_components.py --mode reconciled
with that reference. Progress 10.70 records 82 pinned and 48 device rank/cases,
max scaled q/cache difference 2.22045e-16, matching full halo/cache phase,
exact masks/repeat isolation, strict fluid rho/p/T positivity, and clean
Memcheck/leak-check, Initcheck and Racecheck. Stage boundary/filter calls
and the ordinary repair-enabled main remain unconnected.

The original critical-flux consumer has a separate confirmed MPI defect
(N10, progress 10.71). A manufactured persistent mark on one shared-node
replica survives the original reconciliation without being synchronized.
At the same active interior interface, both eight-node q/primitive stencils
match bitwise, but critical choices are 0/1 and momentum/energy fluxes differ
by 9.32381e-6/6.52667e-6. critical_shared_interior_audit contains the raw
paired record and original executable. CPU source has the same unconditional
rank-edge rule; a CPU runtime pair has not yet been run. No numerical fix
or production-failure attribution is claimed.

prepare_crashfix_components.py --flux freezes six original physical-end
critical-flux fixtures. astr_backend_critical_flux is currently a failing
consumer test: the C++ critical-node flux branch is still missing. Its
initial sparse-mask difference is 8.70829e-4, preserved in critical_before.
Do not count that target as numerically passing or port the MPI defect
without the pending user decision. Proposed OR merge, halo refresh,
physical-end-only clipping, aligned CPU/old-GPU/C++ correction and exact
restart identity update are awaiting approval.

## Legacy Compact Production Statistics

`astr_backend_production_statistics` compares the new FP64 C++ device
component with six frozen captures from the original CUDA Fortran
`compact_statistics_probe`. Capture is optional; the original independent
nonuniform-z, warped-segment, physical-time trapezoid and tilted-wall tests
still execute unchanged. The C++ plan owns 12 plane moments/endpoints and
3 lower-wall moments/endpoints; it reads existing primitive/gradient views.
Its one-time geometry adapter uploads segment lengths and lower-wall frames,
not three additional full coordinate volumes. No per-sample field download.

Run `run_production_statistics_components.py` with `--binary`, `--reference`,
`--mpiexec`, `--sanitizer` and a fresh `--output`. It checks both sync policies,
reset/repeat and component history restore bits, unchanged q/RHS/RK-save,
primitive/gradient input bits, and all measure/sum/endpoint fields. The current
28 NP1 and 56 NP2 rank/cases have max scaled difference 1.35868e-16.
NP2 uses the same fixture independently on two devices; it is not spatial
partition or MPI reduction validation. Memcheck/leak-check, Initcheck and
Racecheck are clean. Evidence is under
`results/multi_backend_mb7_legacy_statistics_20261115/components`.

`prepare_legacy_statistics_reference.py` freezes the existing admitted
743e/643c profile plate's four-step fixed/automatic CPU and old-GPU runs,
sampling-off isolation and 2+2 exact restore. CPU44 and compact GPU12 statistics
are different protocols, not interchangeable arrays. Sampling refreshes the
original velocity/temperature periodic face halos: compare all physical
fields and full conservative halos bitwise, record derived halo refreshes,
and retain full state/statistics equality for same-enabled exact restart.
See `reference_main_compact_phase/reference.json`; earlier rejected fixtures
remain preserved, not accepted or silently overwritten.

Progress 10.73 now connects ordinary C++ `lavg` to the existing compact plate
statistics protocol. `run_cpp_legacy_statistics_gate.py` checks actual
743e/643c profile-plate stages/state, canonical z-integrated 32-channel output,
window clocks, sampling/publication isolation and same-topology 2+2 exact
state/statistics/control restore. Native NP1/2 x/y/z, fixed/automatic steps,
step/time sampling and representative pinned/device MPI pass; max statistic
scaled error is 3.05312e-16. Use `--sample-clock time` with the separate
`reference_time` reference. `--old-gpu-migration` checks an NP1 old-GPU batch
with non-exact restore; the automatic fixture explicitly seeds initial_dt
from source dt_next to match observation clocks, not a new restart policy.
The C++ build's CPU mean path also matches frozen CPU state/statistics with
zero differences in the representative fixed/automatic cases.

Full-main NP2-y Memcheck is clean with zero device leaks. An initial NP1
Initcheck failed in the old test-only exchange snapshot, which read undefined
stress outside true physical ends. Its report/binary remain preserved. The
snapshot ABI now requires physical low/high flags; undefined exterior and
corners are output padding, while valid MPI/periodic halos remain observed.
No CFD array initialization or numerical operator was changed. The 24-case
`astr_backend_snapshot_bounds` probe leaves excluded source storage unwritten
and verifies values/signed zero/padding exactly; Initcheck is clean. The final
full-main Initcheck, numerical comparison and restart checks also pass.
Affected RHS/wall/sensor probes were rebuilt, not claimed as fully rerun.

See `lifecycle_summary.json` and each `main_*/report.json` for this first
main integration. Progress 10.74 adds the following bounded extensions:

* `prepare_legacy_statistics_reference.py --case repartition --mode fixed`
  freezes the existing admitted six-step Cartesian 543e/643e filtered plate.
  `run_cpp_legacy_statistics_repartition.py` checks NP1 step2 -> NP2 x step4
  -> NP2 z step6, nonzero inherited density moments and same-z exact recovery
  from step5. Both pinned/device MPI pass. Canonical sums add inherited
  cumulative moments, never measures or current endpoints twice.
* `run_cpp_legacy_statistics_windows.py` reuses the existing read-only,
  path-scoped input.run replay and `run_adaptive_mean_statistics_gate` math.
  NP1/NP2-y check actual cadence changes, disable at4 through5, enable at6,
  and another cadence change at9. Step9 publishes two samples ending at8;
  step10 has three samples, coverage8.221814601000001e-6. Restoring the
  statistics-disabled step5 batch reproduces complete fields/statistics/
  controls bitwise. Old-GPU compact output difference <=3.97772e-15;
  independent frozen CPU normalized mean/stress difference <=4.88434e-15.

`extended_lifecycle_summary.json` records these extensions; all frozen
references remain preserved. Wider joint boundaries/products, nonseparable
CURVE, DCU and FP32/mixed statistics still need evidence. N7--N10 remain
pending decisions; this does not complete the full MB6/MB7 goal.

## MP2 Mixed Viscous Storage

`prepare_mixed_viscous_reference.py` freezes matching actual CPU/old-GPU
16-cubed TGV states (643e/643e, no filter, dt=1e-4, four steps), including
the existing viscous_flux candidate. Its old-mode FP64 comparison uses the
established MP2 field-atol1e-9/statistics-atol1e-10, zero relative tolerances.
`pack_mixed_viscous_components.py` only subsets immutable records by domain.

`astr_backend_mixed_viscous` and `run_mixed_viscous_components.py` test typed
stress/viscous-energy-flux storage, same-state old-GPU gradients/RHS/RK,
independent constitutive tensors, exact final FP32 stores, halo/shared-node/
padding/signed-zero bits, both exchange entry styles and both sync modes.
Energy flux contains conduction AND stress work; it uses unrounded FP64
stress locals before its final FP32 store. The manufactured states explicitly
distinguish the incorrect early-rounded-stress alternative. NP1/NP2 x/y/z
pinned and representative x/z device MPI pass, max scaled difference
2.84218e-14. Component Memcheck/leak-check, Initcheck and Racecheck are clean.
Only these nine arrays halve in logical bytes; MPI buffers/payloads remain
FP64. This is not half the application peak memory or a speedup claim.

The ordinary C++ runtime accepts existing precision_options selections
`gpu_precision_mode='mixed_workspace', gpu_mixed_candidate='viscous_flux'`
for its connected standard explicit-viscosity pipeline. FP64 remains the
default; the later MP1 flux increment is described below. Derivative and strict
native FP32 are not connected; characteristic_flux has the later component and
bounded periodic-main increments below.
The old derivative candidate is DIAGNOSTIC ONLY: the old solver independently
reconstructs FP64 gradients. Do not change the C++ shared solver gradient to
FP32 and claim equivalence to that candidate.

`run_cpp_mixed_viscous_gate.py` runs actual four-step FP64/mixed main paths
with checkpoint/archive writes disabled. NP1, NP2-z device and NP2-y pinned
pass; mixed vs old-GPU scaled error<=6.66134e-16, mixed-vs-FP64 max absolute
field difference1.42109e-13, scalar-file difference0. Main Memcheck/leak-check
and Initcheck are clean. The later bounded physical extension is described below;
DCU and production physics are not qualified.

The state C ABI transfers raw byte pointers. Fortran uses the existing FP64
upload/download names and explicit *_f32 names without conversion. GNU15
syntax-check rejected the initial assumed-size generic overload for rank4;
the final explicit entries compile. This is not DCU GNU7 runtime evidence.
The initial RK probe also saved q before uploading Jacobian; its failure
source/binary are retained, and only the test ordering was corrected.

Existing output still rejects mixed_workspace checkpoints. LP1 asks whether
to encode the candidate in an unused current control.bin workspace flag,
preserve default FP64 records, and allow only same-mode exact continuation.
No new file/derived-flux checkpoint, cross-mode migration or implicit approval.
Evidence: `results/multi_backend_mb7_mixed_viscous_20261115`; see progress10.75.

Progress10.76 adds `--filter explicit10`, `--automatic`, `--workspace full|scalar`
and `--fixture plate` to the same reference/main runners. The plate reuses
prepare_flat: 16-cubed intervals, MP7 543e/643e, bc11/21/41/51, periodic z,
four fixed steps dt=1e-6. It is an interface/stage fixture, not a physical
boundary-layer validation. Main comparisons now check all three RK phases,
primitive caches, convective/full RHS and complete-state CFL as well as q,
scalars and exact same-mode clocks. Existing unchanged runs are reanalyzed,
not rerun solely for adding assertions.

Four added groups cover TGV full-filter NP2-x device, TGV scalar-filter automatic
NP2-y pinned Memcheck, unfiltered plate NP2-y device, and scalar-filter plate
NP2-x pinned Memcheck. Including the four original groups, matching-old-mode
field error<=2.43497e-15, phase error<=6.73559e-14, candidate-vs-FP64 absolute
field difference<=2.27374e-13. Both added memory runs have zero errors/leaks.
All filters remain FP64. The attempted `--filter compact10` old mixed reference
was rejected by its existing native-precision requirement before advancement;
there is no qualified mixed compact comparison, and the old gate was not changed.
`summary.json` and `phase_reanalysis.json` in that evidence root are generated
from the actual logs/reports by its retained summarize.py. No mixed-mode output
contract or checkpoint/restart capability was added.

## MP1 Physical-Space MP7 Flux Storage

`upwind_prepare_mixed_flux_cpp` and the C ABI allocate FP32 interface storage
only when explicitly requested. The existing default prepare remains FP64.
The current subset is periodic physical-space MP7, 543e/recon3, not WENO7,
MP7-LD, characteristics or physical boundaries. Preserve both original roundings:
round reconstructed F+ to float, promote it and add FP64 F-, then round the
sum to float. Difference loads promote each stored interface to FP64; q,
primitives, geometry, RHS, RK and MPI remain FP64. This does not select the
independent viscous_flux candidate at the same time.

`prepare_mixed_viscous_reference.py` and `run_cpp_mixed_viscous_gate.py` now
accept `--candidate flux`; their default remains viscous_flux. MP1 uses its
original field-atol2e-6/statistics-atol5e-11, rtol0, not the MP2/MP3 values.
`pack_mixed_flux_components.py` copies existing original CPU/old-GPU phases
into the ordinary upwind probe format; `--step 3` isolates the later phase.
`run_upwind_components.py --mixed-flux --steps-reference PATH` selects that
component subset. It also checks FP32 padding NaN bits and workspace reuse.
NP1/NP2-x/y pinned and NP2-z device, plus later-step NP1, pass same-state
RHS/RK gates: max RHS5.82077e-11, state6.66114e-16. X Memcheck has zero
errors/leaks and y Initcheck zero errors. The original default FP64 three
families and the previous mixed viscous plate/filter consumer still pass.

The ordinary four-step NP1 trajectory is NOT qualified by the current script:
its energy RHS at zero-based step3/RK2 differs by2.3283064365386963e-10, above the copied
FP64 phase gate2e-10. Final same-mode field error is only1.98952e-13 absolute;
candidate-vs-FP64 field8.22559e-8 and statistics2.29001e-11 meet the original
MP1 field/statistic gates. LP2 asks to distinguish strict same-state components
from mixed-precision trajectories; the script's phase gate has NOT been relaxed.
No further mixed main MPI matrix, checkpoint qualification, DCU run or
performance claim follows from these records. LP1 is separately pending.

Evidence: `results/multi_backend_mb7_mixed_flux_20261115`, especially
trajectory_diagnosis.json, component reports, and snapshot_hook_isolation.json.
The inviscid full-RHS observation label was added only in the testing build;
61 existing FP64 state/cache/RHS snapshots match bitwise before/after it.
See progress10.77 for recorded early test-orchestration failures and exact scope.

`astr_backend_mp1_replay` plus `run_mp1_replay.py` isolate the later RHS
discrepancy without advancing CFD or introducing another flux implementation.
The probe calls the existing public device API on two frozen q/primitive
inputs with the same geometry and exports actual FP32 interfaces and FP64 RHS.
At zero-based step3/RK2, point(3,5,10), five surrounding energy interfaces are
bitwise equal; only the z-right interface changes from bb3772a2 to bb3772a1,
exactly one FP32 ULP=2.3283064365386963e-10. Both replayed point RHS values
exactly match their actual trajectories. The whole discrepancy at that point
is therefore the same operator's response to the changed input, not a
same-state implementation difference. The diagnostic does not change LP2
status or qualify the ordinary mixed trajectory.

Recorded replay: `results/multi_backend_mb7_mixed_flux_20261115/replay_step3_rk2_exec`.
Each downloaded RHS is exactly reconstructed from its three actual interface
arrays; q input bytes are unchanged. `summary.json` retains the main gate as
not accepted. The first replay directory records a test-copy execute-permission
error; copy2 fixed only that test setup. No original CFD run was repeated.

## MP3 Final Characteristic-Path Flux Storage

`upwind_prepare_mixed_characteristic_cpp` selects the existing MP3 contract:
all physical/characteristic reconstruction and inverse Roe transformation stay
FP64, with one final FP32 physical-flux store on BOTH marked and unmarked
interfaces. It is not MP1's early F+ rounding. The private context distinguishes
the two FP32 storage contracts without allocating a second interface field.
This first increment is a component contract; its later ordinary-main
connection is described below, not inferred from these component tests.

Run the existing upwind runner with `--characteristic-storage` and the frozen
FP64 `frozen_all` inputs. It checks the original FP64 RHS against CPU/old GPU,
then compares each candidate store to an exact float conversion of the completed
new FP64 interface result. The expected promoted-interface negative divergence
equals the actual candidate RHS exactly; raw sensor words and integer masks
equal the same-state FP64 context, and q/padding bytes are preserved. Test-only
cloning/transfers are not a production per-stage copy design. These properties
do not substitute for a new-MP3 versus old-MP3 coupled trajectory comparison.

NP1 and NP2-x/y pinned plus NP2-z device pass: 168 state/stage/sync/gradient/rank
pairs and7,919,280 exact float stores, including marked/unmarked interfaces.
The matching original FP64 RHS error<=2.22045e-16; x Memcheck has zero
errors/leaks, y Initcheck zero errors. The FP64 three-family, MP1 late-stage
and ordinary mixed-viscous plate/filter consumers remain within their existing
gates. Only the selected current C++ interface storage halves, not peak memory
or network payloads.

`prepare_characteristic_flux_reference.py` additionally freezes original CPU,
old FP64 and old MP3 periodic Shu-Osher interface results: 16-cubed intervals,
Mach3, two dt1e-5 steps, no filter/viscosity/checkpoint. The old candidate's
field/statistic differences1.11101e-10/5.21681e-11 satisfy the frozen MP3-A
absolute2e-6 gates. Initial sensor raw words are exact, all six stage masks
agree; later differing-trajectory raw values are NOT claimed bitwise equal.
Flowstate records maxq1..5 at t1e-5; the terminal field is t2e-5. The initial
one-step attempt lacked the CPU's first scalar publication, so only the later
two-step directory is a completed reference; no CPU output behavior was changed.

Evidence: `results/multi_backend_mb7_mixed_characteristic_20261115/summary.json`,
component reports and reference_candidate_two_steps. See progress10.79.
No restart/DCU or production-performance qualification follows from this
component increment; its later main/output evidence is separate. No LP1/LP2
approval or threshold change follows from these records.

## MP3 Periodic Main And Basic Archives

The ordinary runtime now connects existing `shuosher` initialization and
`gpu_precision_mode='mixed_workspace', gpu_mixed_candidate='characteristic_flux'`
for periodic inviscid unfiltered MP7 543e/recon3 with characteristic decomposition.
It calls the verified MP3 prepare API, not MP1 rounding or a CPU RHS. Authoritative
fields and MPI remain FP64. Shu-Osher publishes general maxq1..5 and does not
evaluate unused TGV normalization. All older statistics branches are unchanged.

`run_cpp_characteristic_gate.py` compares real two-step16-cubed Mach3/dt1e-5
main runs with the frozen original modes. FP64 keeps scaled2e-10 stage gates;
MP3 uses its existing field/statistic absolute2e-6 gates, with zero relative
tolerance, records trajectory RHS differences, and requires exact integer masks.
It does not modify the separate MP1 driver or LP2's pending decision.

Evidence: `results/multi_backend_mb7_characteristic_main_20261115` (progress10.80).
NP1 pinned/explicit, NP2-x device/dependency, NP2-y pinned/explicit Memcheck,
and NP2-z pinned/dependency Initcheck pass. Matching-old-MP3 terminal q/primitive
absolute errors<=8.88179e-16/3.55272e-15; all six matched-mode RHS stages are
numerically equal. MP3 versus current FP64 q/primitive/statistic differences are
1.11101e-10/3.70517e-10/5.00009e-11. Initial same-state raw sensor words agree;
all integer masks agree. Memory tools are clean, with full device leak-check
only on the Memcheck main run.

`--archives` enables per-step basic volumes and current JSONL i8/j0/k16 data
slices, not retired &slices. Completed-step1/2 FP64 field values match primitive
reconstruction from same-phase q within scaled4.20994e-16; geometry/index and
volume vector/component bits match. The catalog initial frame is included only
in inventory checks here. Turning intermediate observations and archives off
preserves every rank's full q snapshot bytes, statistic values and clocks.
The first z attempt rejected a retired slices configuration before RK; only
np2_z_catalog_archives_initcheck is the completed archive run. Reanalysis adds
type/status checks without re-running CFD. A four-step FP64 MP7-LD/643c plate
also checks the sensor hook at physical ends with matching masks and clean
Memcheck, separately from the new periodic MP3 run.

This is bounded numerical/consumer evidence, not physical or performance closure.
Sensor observations execute only in a testing build with an explicit request.
Mixed checkpoints and restore remain unchanged/unqualified;
LP1's original viscous-flux scope is not MP3 restart approval. The subsequent
physical-end increment is below; DCU, native FP32, derivative storage and the
remaining full goal are still open.

## MP3 Viscous Profile Plate

`prepare_characteristic_hbl_reference.py` freezes actual original CPU/old-FP64/
old-MP3 results before the new runtime connection. It reuses existing grid and
similarity-profile generators, not a replacement flow integrator. The controlled
MP3-HBL1 fixture uses 192x192x8 intervals, Mach5, Re1.83052e6, Tref226.65,
Tw/Tref5.191440547760865, existing oblique-shock initial/profile data, MP7 543e
characteristic reconstruction and 643e nodal viscosity. Boundary types are
11/21/41/51/1/1; filtering, sponge, crashfix, means and checkpoints are off.
Two complete fixed-dt1e-5 steps are a numerical fixture, not a steady SBLI result.

`run_cpp_characteristic_hbl_gate.py` accepts `--binary`, `--reference`, `--mpiexec`,
`--output`, NP1/2 x/y, pinned/device, explicit/dependency and optional sanitizer.
It compares a real new-FP64 run, new-MP3 run and observation-disabled new-MP3
run. Only the initial sensor and final completed-step2 basic volume are observed;
no per-stage full-field traces. The existing sensor-dump interface now also
works in the C++ testing build, using physical nodes and safe axis snapshots.
Absent explicit observation requests, that hook does not allocate or copy.

Evidence is `results/multi_backend_mb7_characteristic_hbl_20261115/summary.json`
(progress10.81). NP1 pinned/explicit, NP2-x device/dependency and NP2-y
pinned/explicit pass the original absolute5e-7 field/reconstructed-q,
1e-12 scalar and initial-sensor gates. Against new FP64, MP3 max primitive/
reconstructed-q/scalar differences are2.04378e-8/2.41418e-9/1.30009e-16.
Matching original MP3 primitive/scalar maxima are9.41469e-14/4.71179e-13.
Initial raw sensor port difference<=6.92829e-16 and masks agree. All physical
fields remain FP64, finite with positive density/pressure/temperature;
max C_upper is0.0262005624. Full Memcheck on NP2-y reports zero errors/leaks.
The new binary also repeats the periodic NP2-x device gate successfully.

The strict reanalysis reads the actual MPI dump layouts and applies the original
1e-12 sensor gate. Reconstructed q uses runtime1/(1.4-1), not literal2.5;
the immutable earlier reference is not rewritten. Observation-on/off preserves
published six-field bytes, scalar values and clocks, not a claim of full-q exact
restart. Cross-topology fields have differences<=4.44090e-16 in FP64 and
3.38814e-21 in MP3; they are not all bitwise equal. NP2-z would leave only four
local intervals below the existing hm5 transport minimum and is not qualified.
No new filter, restart, DCU, steady-physics or performance claim follows.

## MP2 Diagnostic Derivatives Without Changing Solver Precision

The original `derivative` candidate stores independently computed diagnostic
velocity/temperature gradients in FP32 and promotes them for FP64 statistics;
its viscous solver independently retains FP64 arithmetic. The C++ port therefore
does not turn the shared solver gradients into float. New optional diagnostic
views20/21 hold9+3 physical-node components, allocated only on explicit prepare.
`fluid_prepare_diagnostic_gradient` reuses their addresses;
`fluid_diagnostic_gradient` uses the existing FP64 stencil/metric evaluation and
final FP32 stores. The original solver views8/9, q, RHS and MPI remain FP64.
TGV scalar reductions read promoted diagnostic values. No per-kernel host copies.

Ordinary runtime first connects the existing viscous periodic TGV 643e/643e,
standard convection/viscosity subset with no characteristic decomposition or
compact filter. Use the existing precision options `gpu_precision_mode=
'mixed_workspace', gpu_mixed_candidate='derivative'`. Explicit10 filtering and
automatic steps retain their normal contracts; FP64 remains the default.
Other physical-end, mean/wall consumers and mixed restart are not connected by
this increment. It is not a new lower-precision filter or native FP32 solver.

`prepare_mixed_viscous_reference.py --candidate derivative` freezes actual CPU/
old-FP64/old-candidate runs before porting, using the original derivative gates:
TGV fields exactly equal, scalar absolute1e-8, relative0. The reference has16³
intervals, Mach.1/Re1600, four complete steps, fixed dt1e-4 or auto growth1.1.
Flowstate has original step1..3 samples; the terminal field is complete step4.
No checkpoint/archive or CPU publication phase is changed.

`astr_backend_mixed_derivative` and `run_mixed_derivative_components.py` reuse
the frozen packets. NP1/NP2 x/y/z, both sync policies and three stages check
2,954,736 exact final-float conversions over84 rank/mode/stage/sync cases;
matching rounded original diagnostic values have max difference0. Solver
gradient words, full q/halo words and RHS words remain unchanged; unused
diagnostics are absent and repeated prepare preserves addresses. NP2-y full
Memcheck reports zero errors/leaks; NP2-z Initcheck zero errors. These local
frozen-input components do not claim actual halo communication by themselves.

The ordinary runner `run_cpp_mixed_viscous_gate.py --candidate derivative`
also covers NP1 pinned/explicit, NP2-x device/dependency, NP2-y pinned/dependency
with auto/explicit10 scalar workspace, and NP2-z pinned/explicit Memcheck.
New FP64 and candidate complete q buffers, including halo, are bitwise equal
per rank; all matched-original stage errors<=6.10068e-14. Candidate-vs-FP64
scalar max is4.09330e-9, matching-old-candidate scalar max4.37983e-14.
`--observation-isolation` keeps only the final test snapshot and checks that
intermediate observations do not change q words, scalar values or clocks.

The first auto/filter attempt failed because global test assembly overwrote
shared endpoints in nondeterministic MPI log order. Its actual per-rank fields
were already equal; saved reanalysis identifies the roughly4.49e-17 overlap
selection discrepancy. The runner now chooses rank copies in a fixed sorted
order, without averaging values, modifying the solver or relaxing gates.
Only np2_y_auto_filter_rank_order is the completed joint test.

This restores numerical storage/consumer semantics, not a memory/performance
benefit: at16³ NP1 diagnostics add235824 logical bytes while471648 bytes of
solver gradients remain necessary. No net-memory or speedup claim. The later
main build also preserves the original derivative requirement of diffterm=t
and lchardecomp=f. Recorded binary hashes distinguish that admission-only edit;
the component's subsequent removal of one unused local declaration changes no
numerical code/assertions and does not replace its actual test receipt.

Evidence: `results/multi_backend_mb7_mixed_derivative_20261115/summary.json`,
the immutable reference_datin/reference_auto_filter, components_storage_identity,
four main reports and the MPI-order diagnosis. The initial reference path typo
and unsupported-main RED run remain recorded. The shared gradient store change
is also checked by a new MP3-HBL1 NP2-y run, within the original HBL1 gates.
See progress10.82; LP1/LP2, N7--N10, broader consumers and DCU remain open.

## MP2 Diagnostic Wall And Mean Consumers

Progress10.83 extends `derivative` to the existing static profile plate with
bc11/21/41/51, MP7 543e physical reconstruction and standard643e viscosity.
The wall scalar and production-wall kernels load diagnostic F32 values as
double; solver gradients remain F64. The completed-step mean callback refreshes
the separate diagnostics after its existing velocity/temperature halo refresh,
so wall moments never consume an earlier stage's diagnostic workspace.
Plane moments, temporal integration, output protocols and FP64 default stay
unchanged. This does not admit NSCBC/high-order/CURVE/time-series combinations.

`prepare_mixed_viscous_reference.py --fixture plate --candidate derivative
--means` freezes actual CPU/old-GPU modes with standalone legacy mean writes,
no checkpoints. The fixture is16-cubed intervals, Mach2/Re1000, four fixed
dt1e-6 or auto-growth1.1 steps, unfiltered or explicit10. Its profile and
piecewise initial field are controlled numerical inputs, not a steady physical
boundary-layer validation. CPU44 and compact GPU32 payloads are distinct.

`flat_statistics_probe`/`run_flat_statistics_gate.py --diagnostic` intentionally
inject FP32 diagnostic du/dy7.25 and dT/dy.375 while the F64 solver views retain
3 and.5. This makes wrong view selection observable: before the change the
analytic scalar discrepancy was.010625. The updated component additionally
checks span-integrated wall endpoints and their constant-in-time trapezoidal
integral, q isolation, repeated values and reused geometry. F64 NP1 and mixed
NP2-y/z analytic errors<=2.77556e-17; mixed y Memcheck has zero errors/leaks,
mixed z Initcheck zero errors. These injected local partitions are not an MPI
transport or physical trajectory test.

The ordinary runner adds explicit `--means --archives`; only the named plate
fixture uses these options. NP1 pinned/explicit, NP2-x device/dependency,
NP2-y pinned/dependency auto+scalar filter (full main Memcheck), and NP2-z
device/explicit auto+full filter pass. Each configuration runs actual FP64,
candidate and archive/intermediate-observation-disabled candidate modes.
Candidate and FP64 complete q buffers including halos are bitwise equal;
matching-original stage/cache scaled error<=6.04991e-15. Candidate scalar
difference<=9.92000e-12 versus FP64 fits the existing absolute1e-8 gate.
Matching-old compact mean/endpoints scaled error<=5.27356e-16 and sample/
coverage metadata agree; mean files are bitwise equal across observer switches.

`check_plate_basic_archives.py` inspects completed-step2/4 basic six-field F64
volumes and JSONL i8/j0/k16 numerical sections. Each slice has0/2/4 frames;
initial0 is inventory-only, while steps2/4 compare against same-phase q/EOS
within scaled2.22045e-16. Geometry/index identity, F64 types, volume vector/
component words and absence of duplicate slice vector/coordinates are checked.
The first archive test omitted its required empty output root and stopped before
RK; the helper now prepares that directory. No production directory guard,
fallback or file format was changed. These are numerical archives, not images.

Evidence: `results/multi_backend_mb7_derivative_wall_20261115/summary.json`,
reference/reference_means/reference_auto_filter_means, the analytic components,
np1_means_archives_prepared and three NP2 main reports. The GPUfusion build
script lists the related mixed/production-statistics probes but has not been
run on DCU for this increment. Mixed checkpoint/recovery remains unchanged;
LP1's pending viscous candidate scope is not derivative/MP3 restart approval.
No net-memory or speedup claim; full MB6/MB7 remains open.

## Native FP32 State And Halo Foundation

Progress10.84 uses the existing `ASTR_NUMERICAL_PRECISION=fp32` macro/type in
the actual C++ state owner and transport. q/RHS/save and default Fortran copy
bindings follow the build kind; explicit `_f32` workspace calls remain. The
shared header supplies `numerical_real` to loads/stores, local periodic means,
fixed send/receive slots and packet arithmetic. C MPI uses MPI_FLOAT for FP32,
MPI_DOUBLE for FP64. FP64 builds with mixed workspace views still communicate
double, preserving their existing contract. No per-exchange allocation or
conversion of authoritative full fields was added.

For this isolated FP32 increment configure the root project with
`ASTR_GPU_CPP_BACKEND=none`, `ASTR_BUILD_GPU_BACKEND_PROBES=ON`,
`ASTR_NUMERICAL_PRECISION=fp32`, `ASTR_WITH_CUDA=OFF` and optional consumers off;
build `astr_backend_mb1 astr_backend_halo`, not a new CPP flow executable.
At this foundation increment the FP32 CPP-main restriction remained; the later
ordinary periodic TGV integration is documented under progress10.116 below. The
primitive/RHS arithmetic still required double; progress10.85 below separately
types and qualifies the periodic explicit components. No physical-time/CFL/
statistics, compact, output/restart or rendering claim follows from the state
and halo tests. The minimal VortexReal
gradient declaration fix only makes the shared library compile; the index
producer's remaining double operations are not qualified FP32 diagnostics.

`run_native_foundation.py --bin-directory ... --precision fp32|fp64 --mpiexec
... --sanitizer ... --output ...` runs actual CUDA/MPI components. MB1 uses
distinct integers exactly representable in FP32 and signed zero; borrowed-view
reverse and physical-node copying are not a flow integrator. Each precision
checks6 state rank/halo/sync cases plus36 halo rank/topology/halo/sync cases,
with six full-buffer comparisons per halo case. NP1/NP2, x/y/z, periodic and
physical ends, pinned and true device MPI, shared averaging/halo-only updates,
fixed addresses and begin/end lifecycle are covered.

The test-only `halo_wire_probe.c` delegates MPI calls through PMPI and records
actual types/counts/bytes, not guessed array storage. Initially FP32 numeric
halo checks passed while MPI_DOUBLE was still used: x-periodic sent161280
bytes/rank with32 bad-type observations. After the change, matching FP32/FP64
tests send80640/161280 (x periodic),100800/201600 (y device periodic),57600/
115200 (z physical pinned),40320/80640 (x physical device), bad types0. These
are cumulative component payloads, not production per-step traffic or speedup.
Both state Memchecks have zero errors/leaks and physical-z Initchecks zero
errors; an additional FP32 halo Memcheck checks the resized actual packet path
with zero errors/leaks. The observer is not linked into ordinary astr.

A proposed assumed-size generic copy interface compiled under NVHPC but failed
GNU15.2 rank matching. It was discarded; direct build-kind bindings pass GNU15.2
FP32/FP64 multidimensional syntax checks, with old generic modules retained for
the failure record. This does not claim a GNU12/DTK device run. The test precision
module library needs PIC for PIE linking and does not refactor the main's module
ownership. The isolated component CMake architecture warnings were not repaired
as an unrelated policy change or used in any performance result.

Evidence: `results/multi_backend_mb7_native_foundation_20261115/summary.json`,
fp32/fp64 reports, state_before/halo_wire_before, compiler checks and
fp32_halo_memcheck. Current FP64 ordinary mixed-viscous TGV and derivative
plate/means/archives also pass their original gates; they are explicitly separate
from native component claims. The remaining native capabilities are pending,
as are LP1/LP2, N7--N10 and DCU validation.

## Native FP32 Periodic Explicit Components

Progress10.85 adds build-kind primitive/conservative conversion, six-point-pair
centered gradients with full metrics, convection, standard viscous flux and
divergence, explicit10 scalar/full filtering and RK3. The matching C ABI and
Fortran fluid_step coefficients use the same kind. This removes the former
fluid_prepare eight-byte restriction, not the ordinary CPP-main CMake guard.

Build astr_backend_native_explicit32 in the isolated FP32 component build.
prepare_native_explicit_reference.py freezes four-step16^3 original CPU/CUDA
Fortran FP32 TGV runs; pack_native_explicit_reference.py partitions their
actual records without integrating a flow. run_native_explicit_components.py
checks same-state operators and four full GPU steps through fluid_step for
NP1/NP2 x/y/z, scalar/full filter storage and pinned/actual-device MPI. Numeric
buffers are four-byte; full q words agree between synchronization modes and
physical q words agree across matching decompositions. Report-only native
D10 differences do not use the FP64 2e-10 gate: largest same-state RHS scaled
difference1.04904e-5 and four-step q2.55009e-6. Physical density/internal energy
remain finite and positive. Exact dyadic conversion, affine metric gradients
and constant-state RHS/filter/RK preservation are independent checks.

Evidence: results/multi_backend_mb7_native_explicit_20261115/summary.json,
reference/packets and final_np1_scalar, final_np2_x_device_full,
final_np2_y_memcheck_scalar, final_np2_z_initcheck_scalar. The two sanitizer
cases have zero errors. Ten selected periodic kernels' SASS contains no double
arithmetic; unported kernels still do, so this is not a whole-binary claim.
FP64 manufactured fluid and ordinary automatic-step filtered mixed-viscous
TGV pass their existing gates. Native CFL/statistics/physical time, ordinary
main, compact/other formats, physical boundaries, output/restart/consumers and
DCU remain to be qualified. No memory-peak or performance claim follows.

## Approved Mixed Storage Restart And MP1 Trajectory Gates

Progress10.86 implements LP1 only for the FP64 C++ main's existing
mixed_workspace/viscous_flux candidate. control.bin contract slot7 bit2 records
that choice; bit0(full filter), bit1(MP positivity) and default FP64 values are
unchanged. The existing comparison rejects a cross-mode restore. No extra
file, stress/flux checkpoint dataset or new compatibility path is introduced.
Other mixed candidates and old CUDA Fortran restart admission are not opened.

run_cpp_mixed_restart_gate.py uses four continuous steps versus2+2, output-off
isolation and both directions of rejected FP64/mixed exact switching. Local
NP1 fixed TGV, NP2-y automatic filtered TGV (restored Memcheck), and NP2-x
filtered profile plate with actual device MPI/full filter storage pass. Full
q/cache state and control history match exactly; archived control/insitu
metadata also match in the independent evidence analysis. The first plate
attempt omitted its required empty output root; only test preparation changed.

LP2 changes no numerical producer: only the MP1 mixed trajectory RHS uses
explicit report-only comparisons, with per-variable errors and raw snapshots.
Default compare remains2e-10, including same-state components and other modes.
The existing MP1 absolute field2e-6/statistics5e-11 gates remain. Read-only
original NP1 reanalysis and actual NP2-x device/dependency plus NP2-y
pinned/explicit (Memcheck) retain the observed2.3283064e-10 RHS difference;
they do not relabel it as zero or silently change the old failure artifacts.

Evidence: results/multi_backend_mb7_low_precision_decisions_20261115.
The separate old test_m12_integration_contract source-text suite still has
three failures and one error predating this change; it was not repaired or
reported passing. N7--N10 are approved but still await implementation/recheck.

## Approved N7/N8/N9 Halo And Gradient Repairs

Progress10.87 now supersedes the pending status for these three defects only.
CPU filtered bc52 refreshes halos after the second transverse filter before
RHS; the matching old CUDA Fortran exchange moves before RHS. The shared
geometry initializer refreshes Jacobian/metrics after all shared averages.
Old CUDA Fortran diagnostic gradients choose a unique physical axis from
periodicity, not equality of the end boundary codes. Existing stencils remain.

run_nscbc_top_halo_phase.py against the completed Fortran-linked reference
build records zero extra-refresh and NP1/NP2-z RHS differences. Do not use
the C++ build as its CPU-build: that helper expects the Fortran link driver.
run_filtered_top_reference.py uses actual CPU/old-GPU16^3 plate stages at
NP1 and GPU NP2-y/z. pre_rhs observations are now after the filtered boundary
on both implementations; the old observer's mismatched-phase failure is kept.
The reordering changes no complete q buffer/scalar/clock words in12 snapshots.

gradient_axis_reference_probe.F90 links to the original completed CUDA Fortran
build through compile_reference, with compile/link commands retained in the
evidence report. Affine complete-metric gradients are exact for x11/21,
y41/52 and z50/50, with NaN physical-exterior donors. The immutable pre-repair
gradient source fails this same y-axis probe. The real nonreflecting41/52
box now has finite statistics satisfying its original CPU/GPU gate.

N9 y-wavy421 standard643c/ME6/compact10 runs pass in the new C++ main at
NP2-y device/PCR device-aware/scalar and NP2-x pinned/PCR pinned/full,
including2+2 exact restore/output isolation and the x Memcheck. Actual CPU
NP2-y and C++ NP2-y saved geometry face halos match owners exactly; the13
physical geometry fields remain bitwise unchanged from matching old runs.

Nonreacting five-variable control slot7 bit3 identifies this halo revision.
check_repair_restart_revision.py rejects a real pre-revision exact request
and checks explicitly non-exact fixed-step continuation with its saved output
schedule preserved. It adds no file or compatibility mode. LP1 current flags
are8/12 for scalar storage; both same-mode resumes still pass. No AIR5 identity
or new C++ ordinary bc52 capability is claimed; N10 remains approved/pending.
Evidence: results/multi_backend_mb7_approved_repairs_20261115/summary.json.

## N10 Critical History Ownership And Actual Flux

Progress10.88 implements the approved nonreacting five-variable history
protocol: OR all physical shared/periodic endpoint replicas first, then
refresh the fixed-hm single-face halos without averaging. CPU and old CUDA
Fortran use the existing byte-face exchange; C++ reuses its prepared pinned
or true device-MPI slots and handles local periodic copies directly on device.
No q transfer or per-call allocation is introduced by this protocol.

Build astr_backend_critical_history and astr_backend_critical_shared_flux
through the root CMake. run_critical_history_components.py takes --binary,
--mpiexec, --transport pinned/device and a new --output directory. It checks
independently enumerated owner truth on a rectangular grid, all valid hm
faces, idempotence, physical-exterior/corner isolation and unchanged full-q
words. Its five NP1/2/4 cases include all three split directions. NP4 uses
local-rank modulo available devices in this test only; it is not a scaling
benchmark or a relaxation of ordinary-main GPU binding.

run_critical_shared_flux.py takes --binary, --mpiexec, --transport,
--references flux_after_final, --packets the immutable reconciled_reference
directory and --output. It compares actual MP7-LD interface values to both
CPU and old GPU, independently for five components, shared-endpoint and
last-interior-halo seeds, x/y/z and both synchronization strategies. Matching
state/history is required before evaluating the flux; it is not a Python
implementation of the solver. Original reference probes live in
tests/gpu_validation/crashfix_reference_probe.cuf.

Evidence: results/multi_backend_mb7_critical_history_20261115/summary.json.
CPU/oldGPU/C++ history matrices and both C++ actual-flux transports pass.
Pinned marker NP4 and pinned actual-flux NP2 memcheck are clean. The actual
device-flux memcheck reaches numerical PASS but exits99 on UCX Fabric/VMM
API probes and IPC mapping leak records; it is not a clean tool result.
Keep that log alongside numerical device results and the clean marker-only
device initcheck; do not suppress the API errors or claim full device-tool
coverage from pinned tests.

Nonreacting checkpoint contract bit4 now records this history revision,
independent of bit3 geometry/top revision and bit2 mixed viscous storage.
Current scalar FP64/mixed flags are24/28. Same-mode ordinary zero-history
TGV restarts remain word-exact, the previous revision exact request is
rejected and explicit nonexact restore passes. Original nonzero bitmap I/O
has component evidence, but C++ ordinary crashfix RK/hot-switch/nonzero
checkpoint lifecycle is still unconnected and its guard remains.

## Ordinary C++ Crashfix Lifecycle Increment

Progress10.89 supersedes the preceding ordinary-main pending status for the
qualified FP64 subset. Filter recovery, boundary/shared-input checks and
post-update recovery now run before ordinary primitive kernels can latch an
invalid state. The existing recovery algorithms, thresholds (via thermal),
mask protocol and byte-packed checkpoint format are reused. No failure flag
is cleared and no full field is sent to a CPU repair routine.

run_cpp_crashfix_gate.py takes --binary, --reference (the repaired original
executable), --template (the immutable TGV datin), --mpiexec and --output.
Select --form central/keep/mp7ld, --fixture tgv/plate, --filter,
--ranks/--axis, --transport, --sync and --workspace. The existing MP7-LD
plate fixture is743e/643c, bc11/21/41/50, no global filter. --reference-cpu
explicitly selects the CPU reference where the original GPU main does not
admit periodic743e; it is not an automatic fallback.

prepare_crashfix_main_probes.py --cpp-build BUILD --old-build BUILD --output
NEW relinks actual completed root-CMake objects with crashfix_main_probe.F90.
The local NVHPC-only preparation changes the entry point, not solver objects.
It injects one negative-density initial device node after normal flowinit;
the normal steploop owns repair, RK, publication and restore. Pass those two
probes to the runner with --seeded-probes. Their nonzero histories are checked
through actual complete-step checkpoints and2+2 word-exact continuation.
--disable-restore checks an explicit nonexact change to repair-off and another
nonexact lean continuation; effective marks must remain unchanged. It does
not request exact restore from a role11 file.

Evidence: results/multi_backend_mb7_crashfix_main_20261115/summary.json.
Four-step representative prepared-state/terminal/scalar scaled differences
are at most2.81828e-14/1.77330e-14/4.37428e-14. Eighteen state datasets,
full-q words, output controls, clock and scalars are exact on same-mode exact
continuation. Pinned representative restored runs have clean memcheck.
The final ordinary executable also preserves LP1's24/28 identities and
same-mode restart/isolation. NativeFP32/mixed crashfix, actual input.run hot
reload with nonzero history, broader physical/NSCBC couplings and this
increment's DCU execution remain unverified or unconnected as stated in
progress10.89. The earlier device/UCX memcheck failure is not reclassified.

## Runtime Repair Control And Ordinary Upper-Y NSCBC

Progress10.90 checks the actual input.run parser, polling, validator and apply
chain against a live C++ device state. The existing crashfix_main_probe has
a test-only ASTR_VALIDATION_CRASHFIX_HOT_CONTROL=1 mode, selected by
run_cpp_crashfix_hot_control.py --binary PROBE --template TGV_DATIN
--mpiexec MPI --output NEW. It starts disabled/unallocated, activates real
repair on a negative-density node, then checks off/on preservation of all q
words and history. NP1/2, pinned/device and pinned memcheck pass. Poll
positions1/2/3 are synthetic and no RK step advances in this component;
ordinary advancement and checkpoint evidence remain separate in10.89.

Progress10.91 connects the existing upper-y52 components to the ordinary
FP64 main. Compatibility/incoming-only self-filter stages now refresh shared
nodes and halos after each transverse direction. A real input repair also
updates the Mach reduction before assembly; no sticky failure is cleared.
The characteristic RHS follows the prepared q, and the original conditional
nonreflecting viscous projection follows diffusion. Shock reference planes
use the original CPU scalar generator on initialization/restore, not CPU
field advancement. GRCBC complete assembly is still unconnected.

run_cpp_filtered_top_gate.py --binary ASTR --reference N7_REAL_REFERENCE
--mpiexec MPI --output NEW compares the actual repaired CPU/oldGPU phases,
terminal fields and scalars. Select --policy compatibility/incoming_only/
sbli_shock, --ranks/--axis, --transport and --sync; --memcheck requests full
leak checking on pinned. For new reference policies,
run_filtered_top_reference.py takes --reference FROZEN_BIN_ROOT, --mpiexec,
--output and the same --policy. The shock fixture uses Ma2, x=5,36degrees;
it tests static target supply and short stages, not a physical SBLI solution.
These ordinary MP7 plate runs do not claim checkpoint admission.

run_cpp_nonreflecting_box_gate.py now selects --convection-form standard/keep
and --viscous-form standard/me6_opti. First freeze matching CPU/oldGPU with
--freeze-only, then pass --reference-results. The existing --checkpoint
contract is exercised for KEEP and ME6: word-exact state, controls, full q,
clock/scalars and output isolation, with clean pinned memcheck/leak results.
The standard box remains an advancement/diagnostic case without formal
checkpoint admission. No new file version or checkpoint compatibility mode
was introduced.

Evidence: results/multi_backend_mb7_top_main_20261115/summary.json and
results/multi_backend_mb7_crashfix_hot_20261115/final_control/report.json.
Prepared-state/cache, full RHS, terminal and scalar maxima are respectively
3.73959e-15,2.44281e-13,1.40444e-15 and2.51123e-11. All are within the
unchanged per-variable2e-10 gate. These are local short numerical checks;
GRCBC, joint22/52 intersections, high-order open stages, the remaining
optional-capability coverage and new DCU execution are still pending.

## Static MP7 Top Checkpoint Registration And N11

Progress10.92 extends the existing unforced static-profile resource path to
543e/643e,bc11/21/41/52; formats and restore meanings are unchanged.
run_cpp_filtered_top_gate.py now offers --checkpoint, --backend cpp/cpu/
cuda-fortran (original modes are tested at NP1), and --source-prefix for an
additional same-topology external step2 nonexact read. It checks actual
continuous/2+2 state words, all controls, full q, scalars and dt/time;
nonexact physical fields use the original scaled gate. Active target changes
must fail specifically at boundary/driver identity, not an unrelated setting.
Static shock parameters now use the same existing seven-word driver record
for every active nonreacting bc52 shock target, rather than only the old
joint-SBLI selector. No new file version, override or compatibility read was
introduced. Evidence: results/multi_backend_mb7_top_restart_20261115.

run_filtered_top_reference.py also prepares the existing joint22/52 fixture
with --joint-family keep/me6/mp7ld_compact and explicit --mpi-axes. The first
KEEP/nonreflecting freeze has been REJECTED at the scalar gate; do not use it
as a complete reference. Statistics use the old GPU's pre-outlet-filter q,
whereas the subsequent RK state matches CPU. Test-only statistics_input/
statistics_primitives snapshots in gpu_runtime.cuf are guarded by the existing
RHS observer. They preserve all12 pre_rhs/post_update files and printed old
GPU scalars byte/value exactly. The independent original cell-average mass
calculation reproduces both phases. Python does not integrate CFD.

N11's production fix was explicitly approved and is implemented: prepare the
existing outlet filter/halos before statistics and reuse them once in RK stage1,
with no formula, CPU, RK or random-history change. The rejected pair remains
rejected. New accepted KEEP/ME6 reference matrices are separate, not a change
to scalar tolerances. The initial diagnosis is in progress10.93 and
results/multi_backend_mb7_joint_open_reference_20261115/statistics_diagnosis.json.

run_nscbc_statistics_phase.py --before OLD --after NEW --template FIXTURE
--mpiexec MPI --output NEW checks all actual prepared/RHS/update files byte
for byte before/after the phase correction and compares new GPU scalars to
the actual CPU. --family keep/me6, --ranks/--axis, --filter and --legacy-random
select the affected paths. --legacy-random --restart additionally checks the
unchanged active RNG control words and new 2+2 exact continuation. No Python
time integrator or alternate boundary is used. Progress10.94 and
results/multi_backend_mb7_n11_statistics_20261115/summary.json index the runs.

## Native Diagnostic Arithmetic

astr_backend_native_diagnostics builds against the common root-selected
precision. run_native_diagnostics.py --binary PROBE --mpiexec MPI --output
NEW --precision fp32/fp64 supports NP1/2, x/y/z and optional pinned memcheck
or initcheck. It tests dimensional/nondimensional directional CFL maxima,
the distinct maximum nodal sum, deterministic global tie keys, local profile
selection, output-buffer widths, exact dyadic constant TGV/channel/plate
integrals and q-word isolation. An invalid temperature must retain the
existing latched spectrum failure. Both explicit/dependency launch modes
are tested; there is no new numerical tolerance or fallback.

All diagnostic arithmetic, reduction storage and C/Fortran scalar arrays
follow the build kind; native FP32 does not retain double reduction arrays.
Only scalar/profile results are read back. Channel/plate constant checks
qualify these components, not their full physical main/closure/mean consumers.
Progress10.95 and results/multi_backend_mb7_native_diagnostics_20261115/summary.json
retain the initial ABI build rejection, actual runs and unchanged FP64 main
checkpoint/field/statistics results. Ordinary FP32 time control, compact
operators, temporal means, output/restart and DCU evidence still require work.

## Joint Profile Outlet And Upper-Y Nonreflection

Progress10.96 connects the original low-order KEEP/ME6 profile selector to
ordinary C++ bc11/22/41/52 advancement. It reuses the existing outlet filter
(including its preserved imposed bottom row), completes shared/halo and
primitive preparation before statistics, and adds both face contributions
before the original incoming-y projection of the assembled x+/y+ corner RHS.
The corner API allocates no field. It does not extend the original viscous
source policy, GRCBC or the distinct early-outlet-RHS MP7 stage sequence.

run_cpp_filtered_top_gate.py and run_filtered_top_reference.py now accept
--joint-family keep/me6 with --policy nonreflecting. --compact selects the
existing 643c/643c path; --filter off/explicit10/compact10 selects the existing
global filter. The C++ runner additionally selects --workspace scalar/full
and the actual separate compact transport/sync alongside the ordinary halo
configuration. --checkpoint checks the already registered same-mode output
contract, without introducing a new checkpoint format. Original modes and
new C++ modes remain distinct actual executables in the evidence.

prepare_wall_forcing_reference.py --joint-family keep/me6 freezes the
original same-topology forcing reference before run_cpp_wall_forcing_gate.py
is used. prepare_legacy_statistics_reference.py --joint-family keep/me6
prepares the existing mean protocol for the actual joint layout; the generic
run_cpp_legacy_statistics_gate.py is reused unchanged. Rank-dependent random
forcing is not compared as if the topology had no physical effect.

Evidence results/multi_backend_mb7_joint_top_main_20261115/summary.json also
includes the original run_cpp_plate_products.py numeric wall/selected-mean
consumer and run_cpp_archive_gate.py three-dimensional fields plus27 numeric
slice frames. No Catalyst or image renderer ran in this increment. The
existing private postprocessing derivatives are not the solver's low-order
physical boundary closure. These are small implementation/coupling checks,
not physical SBLI validation or new DCU execution. A separate admitted
MP7-LD incoming-only reference was initially frozen for the subsequent stage
port; the ordinary-main evidence is now recorded in the next section.

## Staged MP7-LD Outlet And Approved Mean Contract

Progress10.98--10.100 connects 743e/643c,bc11/22/41/52 incoming_only to
the ordinary C++ main. It imposes the outlet/lower-wall intersection before
the outlet filter/RHS, then completes the full wall and upper face. The
existing `qrhs` array retains boundary RHS across gradients,
sensor and stress preparation; upwind volume accumulation must not clear it.
There is no new snapshot, temporary backend option or persistent stage state.
The initially tested two-plane implementation is retained only as evidence;
540 matching stage files compare byte for byte with the simpler implementation.

run_cpp_filtered_top_gate.py --joint-family mp7ld_compact --policy incoming_only
uses the existing mixed explicit/compact discretization. The --compact flag
instead selects 643c/643c in the other fixtures, not the presence of any compact
operator. Actual scheme names are retained in each input.case. The MP7 runner
also compares all12 boundary-RHS phases, normalizing only the original CPU's
positive-divergence versus GPU's negative-divergence snapshot convention.

The first C++ failure was a missing primitive-halo refresh after upper-face
x filtering. The already approved N7 correction additionally had to be applied
to the old GPU's staged helper before accepting fresh references. CPU arithmetic
is unchanged. Neither an older passing reference nor the early prototype is
relabeled as a current, numerically identical program.

The user explicitly approved the same-topology MP7-LD incoming_only mean and
checkpoint extension. prepare_legacy_statistics_reference.py --joint-family
mp7ld_compact now freezes fixed/automatic four-step mean histories before the
existing run_cpp_legacy_statistics_gate.py is run. CPU44/GPU12 remain separate
protocols. Formulas, formats, original2e-10 gates and the existing adaptive-output
restriction are unchanged. Adaptive output is not automatic timestepping.

prepare_wall_forcing_reference.py supports the same joint-family selector;
run_cpp_wall_forcing_gate.py reuses the actual same-topology RNG reference.
The existing plate numeric products and archive runners are used unchanged.
Evidence is results/multi_backend_mb7_staged_outlet_20261115/summary.json,
including failure records, phase_diagnosis.json, actual binaries, memcheck,
publication/observer isolation and exact/nonexact continuation. This increment
does not qualify rendering, wider MP7 combinations, GRCBC or new DCU runs.

## Incident GRCBC Characteristic Components

`src_gpu/nscbc_incident_math.cuh` contains the native device translation of
the original incoming-rate, three-point metric-rest normal derivative,
entropy/shear compatibility and complete-RHS projection. It retains Fortran
column-major matrices and uses `astr::numerical_real`. Only FP64 is qualified
here. The subsequent inviscid ordinary-main connection is described below;
the test-only Fortran audit closure is not used as production algebra.

Build through the root CMake configuration:

```sh
cmake --build build_backend_reference64_repairs --target nscbc_incident_cpu_probe nscbc_incident_gpu_probe -j6
cmake -S . -B build_backend_cpp_native
cmake --build build_backend_cpp_native --target astr_backend_incident_math -j6
python3 tests/gpu_backend/run_incident_math_components.py \
  --cpu-probe build_backend_reference64_repairs/bin/nscbc_incident_cpu_probe \
  --old-gpu-probe build_backend_reference64_repairs/bin/nscbc_incident_gpu_probe \
  --cpp-probe build_backend_cpp_native/bin/astr_backend_incident_math \
  --sanitizer /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer \
  --output results/incident_components_new
```

The output directory must be new. Optional output from the original probe
freezes the CPU inputs, independent Euler bases and expected results without
changing its existing assertions. Nine normal-speed states cover incoming
counts 1/4/0/5/1/0/4/1/1, including sonic and near-zero cases. The C++ test
also checks incoming/outgoing projection, late sources, idempotence, reference
preservation, and the existing zero-spacing/Jacobian rejection contract.
`report.json` records actual binary/source hashes and tool results.

Evidence: `results/multi_backend_mb7_incident_math_20261115/reproducible_gate/report.json`.
Maximum component absolute difference is 9.09495e-13; projection-invariant
error is 5.00223e-12, both below 2e-10; Memcheck reports zero errors/leaks.
This does not resolve the CPU corner ownership defect in KEEP LA3-14 or
qualify main-program GRCBC, MPI/restart, CURVE, native FP32 or DCU execution.

## Inviscid Planar Incident Main

Progress10.102 connects the shared device algebra to the ordinary C++ main.
Initialization/restoration uploads only a derived upper-y plane of exterior
q(5) and physical normal spacing, using original geometry/reference selection.
No additional whole-grid coordinate mirror or checkpoint dataset is needed.
The raw top RHS retains the original normal waves and transverse terms;
the complete incoming projection runs after convection, viscosity and sources.
The existing inlet/top ownership and boundary statuses 1/2/4 are retained.

The original staged preparation is reused, including the inviscid bc421 wall
after the outlet. Explicit/compact KEEP retain the resident boundary RHS when
adding volume terms; other compact callers still clear RHS by default. There
is no boundary-RHS snapshot or additional runtime backend option.

`run_cpp_incident_gate.py` freezes original CPU and CUDA Fortran NP1 references
before the C++ comparison. It uses the existing 16-cubed Ma2/beta40 fixture,
four steps at dt=1e-5, static profile inlet, no main filter/crashfix/means.
Use `--cpu`, `--old-gpu`, `--binary`, `--mpiexec`, `--output` to freeze; further
representatives reuse `--reference`. `--scheme`/`--diffusion` select 643e/643c,
`--ranks 2 --axis x/y/z` selects decomposition, and `--transport device` also
explicitly selects compact `device-aware`. `--memcheck` instruments restoration.
`--entry-x 2` checks the existing constant-reference case.

Eight main configurations and two unaffected original-mode representatives
are recorded in `results/multi_backend_mb7_incident_math_20261115/main_summary.json`.
The main test checks stages, physical admissibility, complete-q/halo output
isolation, exact state/control/scalar/clock restoration and an independent
first-stage characteristic projector on NP1. No Python CFD solver is used.
Both NP2 restoration Memcheck representatives report zero errors/leaks.

These are spanwise-uniform inviscid windows, not general three-dimensional
viscous GRCBC, turbulent SBLI, high-order open boundaries or new DCU evidence.
The known CPU x+/y+ missing spanwise contribution and v3 restoration identity
remain pending separate approval under KEEP LA3-14; neither is changed here.

## H1 High-Order Open-Boundary Math

`src_gpu/high_open_math.cuh` translates production mathematical dependencies
from `boundary_high_order_body.inc`: matched-seventh-moment D6 rows, physical
interface classification, FD numerical fluxes with incoming dissipation,
local F6 filtering, eight-node normal metric-rest terms and the seven-point
external-state incoming rate. Matrices/normal lines remain Fortran ordered.
The external-front branch retains the original two-state difference; the
smooth branch retains the original Neumann weights. No test-only inlet,
sensor-blending, stage-budget or positive-filter candidate is promoted.

Build and compare through the root configurations:

```sh
cmake --build build_backend_reference64_repairs --target high_order_boundary_cpu_probe high_order_boundary_gpu_probe -j6
cmake -S . -B build_backend_cpp_native
cmake --build build_backend_cpp_native --target astr_backend_high_open_math -j6
python3 tests/gpu_backend/run_high_open_math_components.py \
  --cpu-probe build_backend_reference64_repairs/bin/high_order_boundary_cpu_probe \
  --old-gpu-probe build_backend_reference64_repairs/bin/high_order_boundary_gpu_probe \
  --cpp-probe build_backend_cpp_native/bin/astr_backend_high_open_math \
  --sanitizer /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer \
  --output results/high_open_components_new
```

The original probe's optional packet preserves its existing assertions.
Ten scalar/state examples, 72 exact row/direction classifications, constant
preservation and variable-viscosity stretched-coordinate D6 composition are
checked. The C++ GPU performs both differentiations and flux construction;
Python only freezes/runs/checks evidence. Absolute component differences stay
below 2.57433e-15, free-stream metric-rest residual below 9.97466e-16, and
the composite pointwise scaled difference below 1.40297e-12. On 32/64/128
intervals the observed orders are 5.74500/5.87623, above the original 5.5 gate.

Evidence: `results/multi_backend_mb7_high_open_math_20261115/reproducible_gate/report.json`.
Memcheck reports zero errors/leaks. Only FP64 is qualified. The ordinary main
does not consume this header yet: actual H1 near-boundary flux/gradient/source
assembly, wall/corner stage coupling, main-program reference, MPI and restart
still require integration. This does not close H1 stability or DCU qualification.

## Frozen H1 Gradient, Sensor And Convection

`upwind_high_order_axis_cpp` reuses the FP64 interface buffer and original
sensor/critical/Roe/Steger rules. Smooth noncritical physical-end interfaces
use H1 biased FD fluxes. Open upper nodes included in caller bounds use the
eight-node normal flux derivative; transverse axes use interface differences.
Ordinary/H1 kernels are separate template instantiations, not a new runtime
mode. `fluid_gradient_high_open` initializes the basic gradient once, then
retains the corrections accumulated in each physical direction. The old
single-wall entry reuses the same correction kernel.

Reproduce with root-built original and component executables:

```sh
python3 tests/gpu_backend/prepare_high_open_flux_reference.py \
  --cpu build_backend_reference64_repairs/bin/astr \
  --old-gpu build_backend_reference64_repairs/bin/astr \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/h1_flux_raw_new
python3 tests/gpu_backend/pack_high_open_full_reference.py \
  --reference results/h1_flux_raw_new --output results/h1_flux_full_new
cmake --build build_backend_cpp_native --target astr_backend_upwind -j6
python3 tests/gpu_backend/run_upwind_components.py \
  --binary build_backend_cpp_native/bin/astr_backend_upwind \
  --reference results/h1_flux_full_new/shock \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/h1_flux_cpp_new --high-open --ranks 2 --axis y --sanitizer memcheck
```

The original raw `conv` snapshot is not defined identically on the prescribed
inlet/lower wall: old GPU values are zeroed later, including temporary nonfinite
entries in the shock fixture. The packaging step retains raw evidence, checks
finite/matching active nodes and the complete original RHS, and derives the
explicit post-prescribed-zero convection phase. It does not modify old solver
code or introduce a production NaN clamp. Keep this distinct from the complete
characteristic RHS, which is not yet connected to the C++ ordinary H1 main.

Evidence `results/multi_backend_mb7_high_open_flux_20261115/summary.json` covers
entropy/shock NP1 and shock NP2 x/y/z, pinned/device transport, both synchronization
choices, frozen/native gradients, exact sensor markers and temperature gradients.
The final full-convection maximum scaled difference is 1.29698e-17; velocity and
temperature gradients are below 8.76337e-15 and 3.50535e-14. NP2-y Memcheck is
clean on both ranks. Ordinary MP7 components and the affected nonuniform x-wall
main/restart are rechecked separately. H1 stage preparation, complete characteristics,
viscous coupling, ordinary main/restart/products and DCU evidence remain pending.

## Ordinary Production-Definition H1 Main

Progress10.105 connects the original H1 definition, not unpromoted H1 candidates.
The new C++ complete boundary applies the x+ incoming correction before the upper-y
seven-point external-rate projection, including their intersection. H1 pressure/
slip Neumann states are algebraic, distinct from the evolved-density channel wall.
The existing local outlet F6 uses high-order end rows; top transverse filtering stays
off. Standard nodal viscosity applies high rows in every physical direction, over
the original full physical domain, before complete characteristics/prescribed zeros.
Active H1 ME6 wall-band assembly is not connected. Explicit test-candidate requests
are not silently interpreted as production H1 in testing builds.

The static upper-y plane has 42 doubles per node: seven primitive exterior states,
the last exterior q, hn and crossing flag. It is derived once at initialization or
restoration from existing host geometry/configuration; non-top ranks prepare only
the status resource. No new whole-grid coordinate field or checkpoint sidecar is
required. Scalar status/legacy Mach observations do not download whole flow fields.

```sh
python3 tests/gpu_backend/run_cpp_high_open_gate.py \
  --cpu build_backend_reference64_repairs/bin/astr \
  --old-gpu build_backend_reference64_repairs/bin/astr --fixture viscous \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/h1_main_reference_new
python3 tests/gpu_backend/run_cpp_high_open_gate.py \
  --binary build_backend_cpp_native/bin/astr --reference results/h1_main_reference_new \
  --fixture viscous --ranks 2 --axis y --memcheck \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec \
  --output results/h1_main_cpp_new
```

`--fixture entropy/shock/viscous` reuses existing numerical fixtures. `--automatic`
uses target .5, initial_dt=1e-5, max_dt=2e-5, growth=1.1. When `--nonexact-source`
is supplied, compare against a same-build nonexact restart using the same current
controller input, not a continuous automatic trajectory at a different physical
time. Exact continuation preserves the saved controller and is checked separately.

Evidence `results/multi_backend_mb7_high_open_main_20261116/summary.json` covers
seven local FP64 four-step representatives, explicit/device and pinned transport,
12-stage fields/RHS, physical admissibility, full-q/halo output isolation, 15 exact
state datasets and controls/scalars/clocks, plus CPU-to-C++ nonexact recovery.
The maximum scaled terminal difference is 5.74389e-15; complete RHS is below
8.78204e-18. Two-rank restoration Memcheck is clean. The first invalid continuous/
nonexact time comparison is retained and explained, not relabeled as passing.

No physical/long-window H1 stability, new rendering, stochastic H1 composition,
mixed H1 qualification or DCU execution is claimed. Main-mean/checkpoint admission
is not expanded by runs with mean statistics disabled. Low-order GRCBC LA3-14
CPU repair remains a separate pending approval; no original CPU math changes here.

## H1 Forcing And Numeric Consumers

Progress10.106 uses the unchanged progress10.105 C++ binary. The previously
approved incoming-only MP7-LD/643c main-mean contract is rechecked with the
existing frozen CPU/old-GPU references: NP2-x device/dependency, four steps,
fixed/automatic controllers and physical-time sampling. The 32 serialized GPU
statistics channels differ by at most 4.44090e-16 scaled; state/statistics/control,
scalar and clock exact continuation and sampling/output isolation pass. This is
not an expansion of H1 main-mean admission.

`prepare_wall_forcing_reference.py --joint-family high_open` selects existing H1
inputs and original legacy-random forcing without changing its algorithm or
stage call count. Use the existing `run_cpp_wall_forcing_gate.py --checkpoint`
at the reference topology. NP2-x four-step evidence covers the full stages,
output isolation, exact state/control/RNG and scalar history. The prescribed
external shock lies outside this forcing fixture, so it does not test concurrent
shock crossing. Rank-dependent RNG is not compared across topologies.

The existing `run_cpp_plate_products.py` freezes numeric CPU/old-GPU references
first and then checks the H1 viscous fixture with NP2-y. Eight wall and selected
time-statistics products cover density, temperature, shear, wall heat and viscosity.
Their maximum scaled difference is 5.55111e-15; independent temporal/geometric
integration differs by at most 4.44090e-16. Product switches preserve state/clocks,
and 2+2 exact continuation preserves state/statistics/control and product values.
No new consumer algorithm or main-mean registration is introduced.

Evidence is under `results/multi_backend_mb7_high_open_consumers_20261116/`:
`incoming_mean_current/report.json`, `rng_reference_np2x/reference.json`,
`rng_cpp_np2x/report.json`, `product_reference/report.json` and
`product_cpp_np2y/report.json`. The current build has Catalyst disabled; these
numeric outputs are not image or renderer qualification. H1 mixed workspaces,
broader geometry/partitions, main-mean combinations and DCU remain pending.

## H1 Archives, Rendering And Launch Resources

Progress10.107 uses the existing archive runner with the H1 viscous profile
fixture: two completed-step volumes and 27 nine-field slices, including device
Q/divergence/density-gradient values, pass original CPU/old-GPU comparisons.
The maximum scaled difference is 5.78501e-13. Exact physical-q output isolation
holds; selected slices download 561816 result bytes, not a volume per slice.

`run_cpp_plate_products.py --images --image-preset high-open` uses the existing
five rendering products on the 1 x 1 x .125 fixture. The original `plate` preset
is unchanged. The H1 preset uses Q=1e-4, the existing 16 SBLI seeds/RK45 controls,
arc .75 and a fixed signed wall-shear range [-.001,.001]. It runs the frozen
renderer first unless explicit immutable `--renderer-results` are supplied.
Progress10.108 covers NP2-x/direct-device and NP1/standard-device, 14 JPEG frames
per full run, EPS off, numeric and pixel comparisons, full-state output isolation
and exact restart. Both ranks retain streamline segments across x=.5; all 16
particles reach the requested arc. Renderer receipts report zero field/geometry
downloads, but the explicitly pinned private streamline halo still transfers data.

The first Catalyst development build failed with error 701 even without in-situ.
Memcheck identified two 512-thread high-order correction launches whose kernels
lacked matching launch bounds: the gradient and viscous-flux correction used 144
and 138 registers/thread with inherited ptxas -O0. Matching launch declarations
now constrain both to 128 without changing formulas, thread counts or build flags.
Uninstrumented numerical/restart checks and pinned Memcheck pass on final builds.

`renderer_state` compares the 11 physical fields and phase/layout metadata with
the original 2e-10 criterion. Legacy H1 extras contain nonfinite primitive caches
outside the physical domain, not at physical nodes or internal MPI halo nodes;
these undefined caches are counted separately, never overwritten. Exact same-build
restart and output isolation still compare the entire serialized state bitwise.
The failed whole-state numerical comparison and its spatial diagnosis are retained.

Evidence is in `results/multi_backend_mb7_high_open_consumers_20261116/`, including
`archive_np2z/report.json`, `launch_bound_complete_no_insitu/report.json`,
`images_cpp_np2x_direct_crossing/report.json`, `images_cpp_np1_standard/report.json`
and `final_native_np2y_pinned/report.json`. `final_native_np2y` retains a distinct
device-MPI Memcheck failure in UCX initialization context probes; the solver ended
normally and recovered state/control/scalars/clocks match exactly, but that tool
combination is not marked clean. No API-error suppression or public-library change
was introduced. This is not H1 long-window physical validation, mixed-mode/main-mean
admission, DCU rendering or completion of the full MB6/MB7 goal.

## Native Compact Precision Components

Progress10.109 types local/interface PCR, cyclic correction, factor/stencil
structures, host preprocessing, device/pinned storage and actual MPI packets by
`astr_numerical_real` / `c_device_real_kind`. Global rank coupling, formulas,
tile ownership and synchronization are unchanged. This component increment did
not admit ordinary FP32 C++ main; progress10.116 later integrates periodic TGV.
KEEP only follows the borrowed compact buffer's element
type; its existing FP64 arithmetic/weight interfaces are not a native FP32 claim.

The existing local/distributed compact targets and `astr_backend_boundary_flux`
are now available in either component precision build. FP64 retains its 2e-10
field and 1e-12 residual gates. FP32 follows the existing D10 report-only error
policy, plus finite results, zero RHS, unchanged input words, exact synchronization
and transport isolation. `real_to_word` performs same-kind bit checks; a fixed
int64 transfer length is not used for FP32 arrays. `halo_wire_probe.c` is linked
only to the distributed probe to observe real Isend/Irecv types/counts/bytes.

`run_compact_distributed.py --precision fp32` reads four-byte payloads;
the default remains `fp64`. Select the MPI transport environment explicitly,
then use `--transport pinned` or `device-aware`; `--compare` checks same-precision
transport words. Optional `--sanitizer memcheck` is a local CUDA test selection,
not a production fallback or a DTK tool. Choose representative topologies rather
than instrumenting the entire mode matrix.

`astr_backend_boundary_flux compact` checks 256 destination ownership cases,
constant velocity/affine temperature gradients and affine stress/heat divergence.
Its periodic identity-filter case checks scalar/full layouts exactly; it is not
qualification of the actual compact10 coefficient set or coupled FP32 CFD.
These controlled algebraic inputs are not proposed as a physical initial state.

Evidence `results/multi_backend_mb7_native_compact_20261116/summary.json` includes
252 local cases/precision and 1512 distributed rank cases/precision/transport.
FP32 local/global maximum scaled differences are 4.71118e-6 / 2.98876e-6,
and residuals are 1.00075e-6 / 8.18409e-7; the numbers are reported without
an invented precision cap. Every actual packet type matches the build kind;
corresponding FP32 payloads halve FP64 bytes without changing message counts.
Local, NP2-y pinned and packed-field FP32 Memcheck logs have zero errors/leaks.
Selected PCR/field objects contain no observed FP64 arithmetic opcodes.

The changed default FP64 ordinary solver is separately checked using frozen
KEEP/ME6 compact references: NP2-x, scalar workspace, real device halo/private
PCR, dependency synchronization and exact/nonexact continuation. Maximum final
scaled difference is 2.23246e-14; state/control/scalars/clocks and output isolation
retain their existing exact contract. No new native FP32 main/output/restart,
actual compact10 field coupling, DCU execution or performance claim follows.

## Native KEEP Components

Progress10.110 types the existing KEEP weights, gamma C ABI, monomials,
two-point fluxes, explicit kernels and compact RHS/commit arithmetic by build
precision. The 294 terms/85 bases and formulas/order are unchanged. Original
CPU sources are not edited. The host guard passes in both kinds, and all table
integers/weights agree exactly across precision. Ordinary FP32 main was blocked
at this component step; the later standard-TGV integration does not qualify all
KEEP or other optional combinations.

`astr_backend_keep` is now an MPI-launched NP1 component probe. It binds its
device before MPI initialization, then checks periodic explicit KEEP, compact
KEEP with `MPI_COMM_SELF`, and low-order boundary accumulation. Use a matching
component precision build and `mpiexec --bind-to none -n 1`; direct execution
without the required relocated MPI environment is not a supported invocation.

The compact reference uses original CPU monomials/accumulation and independent
whole-matrix LU, at 32/64/128 unique periodic nodes in each direction. A private
restored endpoint is present but not counted as an extra source node. Both a
127-column truncated batch and the 3060-column full batch retain an initial RHS
of 7. Same-batch synchronization results are bitwise equal; different summation
orders across batch sizes are not required to be bitwise equal. Constant inputs
produce exactly zero explicit/compact RHS. The 256 physical-end cases retain
boundary-owned words and do not consume NaN-marked exterior values.

FP64 retains the original comparison, conservative/kinetic identities and
sixth-order gates. FP32 follows D10: types, finite values, exact properties and
raw numerical differences, without an invented error cap. Maximum scaled
FP32 explicit/compact/boundary differences are 3.61354e-7 / 5.42414e-7 /
1.20267e-7. GPU and original CPU FP32 analytic errors both increase in some
components at 128 points; the recorded orders do not establish sixth-order
convergence there. This is not a physical accuracy or long-window CFD claim.

Evidence is `results/multi_backend_mb7_native_keep_20261116/summary.json`.
The complete FP32 probe Memcheck has zero errors/leaks; four explicit/compact
consumer kernels have no observed FP64 arithmetic. The SBP63 function still has
a DADD and remains outside strict native qualification. Default FP64 NP2-y
KEEP/ME6 compact main/restart passes separately against immutable references,
maximum final scaled difference 2.48579e-14, exact/nonexact continuation error
zero. At this step FP32 main, nonlinear multi-rank KEEP CFD, actual compact10 coupling,
ME6/SBP63/other boundaries and consumers, and new DCU evidence remain pending.

## Native ME6 Components

Progress10.111 types the ME6 coefficient tables, mapped viscous operator,
Sutherland properties, private device/pinned slots and C/Fortran arguments by
the existing build precision. Actual private MPI packets select MPI_FLOAT or
MPI_DOUBLE accordingly. The original CPU operator, formulas, coefficient order,
y-then-z corner completion, directional thread blocks and synchronization are
unchanged. This does not admit the ordinary FP32 C++ main.

`astr_backend_me6` without arguments retains eight prescribed-patch checks
(four fields/maps, two synchronization modes). The periodic MPI mode is:

```bash
python3 tests/gpu_backend/run_native_me6.py \
  --binary build_backend_cpp_fp32_components/bin/astr_backend_me6 \
  --mpiexec /path/to/mpiexec --output /new/evidence/directory \
  --precision fp32 --transport pinned
```

The runner covers NP1 and NP2 x/y/z slabs. Use `--transport device-aware` and
`--compare` with the same-precision pinned directory to compare transport words;
`--cases np2y --sanitizer memcheck` selects a representative CUDA tool run.
`--analyze-only` reads completed records from the unchanged tested executable
without launching MPI. There are no automatic retries or production fallbacks.

The 16-interval manufactured field has exact public face halos and poisoned
multidirection exterior corners. ME6 must restore every private corner word,
retain the nonzero density RHS seed, produce finite fields, and preserve words
across synchronization, transports and decompositions. Rank zero creates unique
periodic data once, avoiding unrelated host trigonometric differences among
duplicated nodes. The non-diagonal map is affine, not general CURVE validation.

The linked PMPI observer checks real Isend/Irecv types, counts and bytes. ME6's
private extension only exchanges y/z; zero private x-slab traffic does not mean
the solver's public x halo is unnecessary. Both precisions have the same message
counts. FP32 private buffers and logical payloads are half FP64, not a measured
halving of total peak memory or physical interconnect traffic.

Evidence `results/multi_backend_mb7_native_me6_20261116/summary.json` reports:
prescribed-patch maxima FP32 3.15195e-8, FP64 5.86503e-17; periodic maxima
FP32 7.45059e-8, FP64 1.17962e-16. FP32 follows D10 reporting without a new
error cap; FP64 retains 2e-10. Three FP32 Memcheck logs (patch and two pinned
MPI ranks) show zero errors/leaks. Five actual ME6 kernel functions have no
observed FP64 arithmetic/conversion opcodes.

The default FP64 main separately passes NP2-z KEEP/ME6 compact derivative/filter,
scalar workspace, device halo and device-aware PCR, four steps versus 2+2 exact
and nonexact continuation. Maximum final scaled difference to immutable CPU/old
GPU references is 2.72329e-14; exact state has 15 identical datasets, and output
isolation preserves physical q words. FP32 coupled CFD, physical wall bands,
general CURVE, actual compact10 FP32 coupling, other boundaries/consumers and new
DCU execution remain separate pending work. No performance claim follows.

## Actual Native Compact10 Field Coupling

Progress10.112 adds `astr_backend_compact_filter` in either precision component
build. It uses the existing alpha=0.49 compact10 coefficients, real public q halo
exchange and globally coupled line PCR. This is a focused characterization test;
production solver code and the ordinary FP32 main admission are unchanged.

```bash
python3 tests/gpu_backend/run_native_compact_filter.py \
  --binary build_backend_cpp_fp32_components/bin/astr_backend_compact_filter \
  --mpiexec /path/to/mpiexec --output /new/evidence/directory \
  --precision fp32 --transport pinned
```

Default cases are 16-interval periodic NP1 and NP2 x/y/z slabs. The probe filters
all three directions using five-component line workspaces, one-component
workspaces, and one-component workspaces truncated to 137 RHS columns, in both
synchronization modes. This is the compact solver's line workspace, not a claim
that a full q ping-pong buffer is allocated. A nonuniform combination of Fourier
modes includes the Nyquist mode; a second all-zero input must remain exactly zero.
All modes must give identical words within the same topology/transport.

Reference one uses the original CPU `compact_filter_rhs_global` with independent
full cyclic-matrix LU after each direction. Reference two uses the analytic
Fourier transfer function for the complete three-axis filter. Same-kind
decomposition differences are reported, not required to be bitwise zero: global
PCR changes floating-point reduction order. Use `--transport device-aware`
with `--compare` pointing to matching pinned evidence for transport word checks.
`--cases np2z --sanitizer memcheck` instruments the representative CUDA MPI case.

Evidence `results/multi_backend_mb7_native_compact10_20261116/summary.json` has
252 rank/axis/sample/layout/sync observations per precision/transport. Maximum
CPU-RHS/LU differences are FP64 2.63899e-14, FP32 1.45136e-5; analytic Fourier
differences are 2.61753e-14 and 1.29421e-5. NP2-versus-NP1 differences are
1.67351e-14 and 9.67569e-6. FP64 retains 2e-10; FP32 follows D10 raw reporting,
without a new accuracy cap or a claim of FP64-equivalent accuracy. Both transport
outputs match wordwise; two FP32 pinned MPI Memcheck logs show zero errors/leaks.
Physical endpoints, nonperiodic FP32 CFD/main/restart and new DCU execution remain
pending. Periodic ordinary-main integration follows in10.116. This test does not
establish long-time filtering accuracy or performance.

## Native Shock Sensor

Progress10.113 types the existing `shock_sensor` raw arithmetic and threshold
C/Fortran arguments by build precision. It retains divergence squared, vorticity,
pressure curvature, 1e-30 regularization, physical-end index clamping, halo-only
exchange, axial -hm+1:hm expansion and strict `maximum > threshold`. There is
no compression-only H1 candidate, changed detector formula or additional clamp.

`astr_backend_shock_sensor` is available in either precision component build.
The original `--reference` option of `run_sensor_components.py` still reads
frozen FP64 inputs. Native FP32 selects independent analytic fixtures explicitly:

```bash
python3 tests/gpu_backend/run_sensor_components.py \
  --binary build_backend_cpp_fp32_components/bin/astr_backend_shock_sensor \
  --mpiexec /path/to/mpiexec --output /new/evidence/directory \
  --precision fp32 --analytic-only --ranks 2 --axis y --transport pinned
```

The probe checks positive/negative divergence, pure rotation, constant pressure,
a dense nonzero tensor, and six exact threshold tests (equality and immediately
adjacent native representable thresholds, two sync modes). It reads initialized
native sensor storage directly, rather than changing the separate FP64 diagnostic
snapshot protocol. `observations.rank*.bin` records native values and byte masks;
`--compare` checks matching topology/precision observations from another transport.
No frozen FP64 payload is decoded as FP32 or presented as a native CPU reference.

Evidence `results/multi_backend_mb7_native_sensor_20261116/summary.json` covers
FP32 NP1/NP2 x/y/z pinned, NP2-y device and NP2-z pinned Memcheck. Polynomial
reference differences are zero in this fixture; physical raw sensor and masks
are wordwise equal across decompositions, sync and the matching transports.
Both Memcheck logs have zero errors/leaks. The two actual FP32 raw/expand kernels
have no observed FP64 arithmetic or conversion opcodes. This is not a physical
accuracy or FP32 MP7 main/restart claim.

FP64 separately checks both frozen original solvers' three-stage fields/masks
at NP2-z using pinned/device, with zero raw error and exact masks. The updated
ordinary FP64 main passes a four-step/2+2 MP7-LD 743e/643c plate with joint
bc11/22/41/52 incoming_only, actual compact10, full line workspace and NP2-y
device/dependency. Stage/final maxima are 1.55444e-11/3.13279e-14; output isolation,
15 exact state datasets, controls/clocks/scalars and nonexact continuation retain
their approved gates. MP7 native reconstruction, expanded-mask face consumers,
complete FP32 scheduling/output and new DCU execution remain pending.

Scope clarification: existing `keep_sbp63_runtime` explicitly requires FP64.
The earlier observed double SBP helper is not a defect in that supported subset;
strict native FP32 SBP63 would be a new capability, not required to preserve the
existing SBP63 support. This work does not modify or extend it.

## Native MP7 Operator Components

Progress10.114 types the existing Steger-Warming splitting, Roe matrices,
MP7/MP7-LD reconstruction, interface arithmetic and C/Fortran arguments by build
precision. Native interface storage now uses `sizeof(astr::numerical_real)`.
The MP1 split F+ rounding and MP3 final physical-flux store remain unchanged in
their supported FP64-state modes. No format coefficients, threshold, sensor,
boundary formula, synchronization or reconstruction order was revised.

`prepare_upwind_components.py` retains its FP64 `--reference` mode. Explicit
`--native-binary /path/to/original/astr --precision fp32` selects the established
original CPU/CUDA Fortran executable via case input and records native four-byte
q, primitives, geometry, gradients, sensor and RHS before the new consumer runs.
The gradient reader consumes the temperature tensor and optional wall-normal
payloads using their existing header flags. It does not reinterpret FP64 files
as FP32 or downcast a frozen FP64 trajectory.

Use the matching component precision build and packet directory:

```bash
python3 tests/gpu_backend/run_upwind_components.py \
  --binary build_backend_cpp_fp32_components/bin/astr_backend_upwind \
  --reference /path/to/frozen/native32/packets --precision fp32 \
  --mpiexec /path/to/mpiexec --output /new/evidence/directory \
  --ranks 2 --axis y --transport pinned
```

The established 72 operator cases/rank cover MP7 physical reconstruction,
selective Roe reconstruction, low-order physical-end MP7-LD, both original
sources, three frozen stages, two gradient sources and two sync modes. Raw
`observations.rank*.bin` stores typed RHS and byte masks; `--compare` checks
matching topology/precision records from another transport. Same-sync RHS words,
unchanged physical q, allocation reuse and exact sensor masks are checked.
FP64 retains its original 2e-10 gate; native FP32 reports differences under D10.

Evidence `results/multi_backend_mb7_native_upwind_20261116/summary.json` includes
FP32 NP1/NP2 x/y/z pinned, NP2-y device and NP2-z pinned Memcheck. Maximum
scaled RHS differences by component are 1.18744e-8, 2.23518e-8, 1.11759e-8,
1.11759e-8 and 1.19210e-7. Native-gradient maximum difference is 2.86105e-6.
Same-source fields/masks agree wordwise across these decompositions and matching
transports. Both Memcheck logs have zero errors/leaks; these are same-state
operator checks, not native FP32 main/trajectory/restart or physical validation.

Four actual interface/difference kernels have no observed FP64 arithmetic
opcodes. Generic mixed-width `load_value`/`store_value` still compile FP64
conversion branches for eight-byte views; the native interface/state views here
are four bytes. Do not describe the whole object as containing no FP64
instructions, and do not remove shared mixed-storage support to satisfy that
incorrect static criterion.

FP64 ordinary operators and MP1/MP3 storage components pass separately against
their immutable sources. Updated FP64 MP7-LD 743e/643c, joint incoming_only
bc11/22/41/52, compact10/full, NP2-y device/dependency passes four steps and 2+2
exact/nonexact restore: stage/final maxima 1.55444e-11/3.13279e-14, fifteen exact
state datasets, controls/clocks/scalars and output isolation. Native FP32 expanded
mask faces, high-order open-boundary consumers, coupled RK/main/output/restart,
and new DCU execution remain pending.

## Native Marker Faces And Boundary States

Progress10.115 types boundary-state parameters, private planes, inlet buffers
and the top-normal-Mach maximum by the build kind. The five existing NSCBC
classification tolerances use `64*epsilon(numerical_real)`, matching original
CPU `epsilon(real_kind)` rather than retaining hard-coded FP64 epsilon in FP32.
FP64 thresholds/formulas are unchanged. A nonnegative FP32 maximum uses 32-bit
atomicMax; FP64 retains its existing CAS loop. Shared high-wall helper algebra
is compile-time typed, preserving existing FP64 callers and coefficients.

`astr_backend_shock_mask_halo` and `run_shock_mask_halo_gate.py --precision fp32`
now check raw-sensor native width, all six byte faces and actual interface-marker
consumption. Physical markers remain unchanged, local periodic/physical ends and
changed second marker patterns are exercised. This does not create a different
MPI protocol: marker payloads remain bytes; raw sensors use the existing build-kind
halo. FP32 NP1, NP2 x/z pinned and NP2-y device pass; NP2-z Memcheck has two
zero-error/zero-leak logs.

`astr_backend_native_boundary` provides focused independent polynomial checks:

```bash
python3 tests/gpu_backend/run_native_boundary.py \
  --binary build_backend_cpp_fp32_components/bin/astr_backend_native_boundary \
  --mpiexec /path/to/mpiexec --output /new/evidence/directory \
  --precision fp32 --ranks 2 --axis y
```

The fixture uses gamma=2, Mach=2, quadratic normal pressure/density, constant
temperature/velocity and a non-diagonal affine metric. These are controlled
algebraic inputs, not proposed physical run defaults. It checks known Neumann
values, no-slip states and geometric normal projection for bc41/50/60 in x/y/z,
and bc42 in its existing x/y support. The initial probe mistakenly requested
z-bc42 and was correctly rejected; failed records remain, the test scope was
corrected, and production admission was not extended. Each NP1 run has 44
state cases; NP2 x/y has 36/rank and NP2-z 38/rank.

Physical q/primitives/inner energy remain finite and positive, untouched regions
and RHS footprints retain exact words, and synchronization/decomposition q and
primitive fields agree wordwise. A varying oblique normal velocity separately
tests real local top Mach² reduction and native MPI_MAX. FP32 NP1/NP2 x/y/z and
FP64 NP1/NP2-y pass. Polynomial-state maximum difference is zero in this fixture;
Mach² differences are zero FP32 and 1.89198e-16 FP64. FP32 differences follow
D10 reporting, not a new accuracy cap. NP2-y FP32 state Memcheck is clean.

Evidence `results/multi_backend_mb7_native_boundary_20261116/summary.json`
includes four clean Memcheck logs, four tested native kernel functions without
FP64 arithmetic/conversion, data/footprint checks and preserved failures. Other
compiled boundary functions with FP64 conversions are recorded separately and
are not declared native-qualified merely because their arguments were typed.

The default FP64 ordinary main passes actual compact10/full, MP7-LD 743e/643c,
joint incoming_only bc11/22/41/52, NP2-y device/dependency: stage/final maxima
1.55444e-11/3.13279e-14, exact 15 datasets/control/clock/scalars and nonexact
restore. H1 viscous NP2-z device/dependency separately passes 4 versus 2+2 steps,
stage/final maxima 6.30095e-15/5.74389e-15 and exact restore/output isolation.
Native FP32 NSCBC RHS/target lifecycle, profile/dynamic forcing consumers,
high-order operators and complete main/statistics/output/restart remain pending;
these state components do not admit the ordinary FP32 C++ solver or new DCU runs.

## Ordinary Native FP32 Periodic Main

Progress10.116 removes the whole-main FP64 CMake build restriction and integrates
the ordinary C++ executable in FP32. Use the root CMake with the same qualified
MPI/HDF5/Fortran toolchain selections as the FP64 build, changing these options:

```bash
cmake -S . -B build_cpp_native32 \
  -DASTR_GPU_CPP_BACKEND=cuda -DASTR_WITH_CUDA=OFF \
  -DASTR_NUMERICAL_PRECISION=fp32 -DBUILD_TESTING=ON
cmake --build build_cpp_native32 --target astr -j 8
```

FP64 remains the default; `use_gpu` remains a case-input selection. This build
choice does not qualify every optional numerical method, boundary or consumer.
The current evidence is ordinary 16³-interval periodic standard TGV, four full
steps, Mach0.1/Re1600, fixed dt=1e-4 or auto target0.5/initial1e-4/max1e-3/growth1.1.

`run_cpp_native_tgv.py` prepares inputs and reads native snapshots, CFL, scalar
and complete-step logs; only ASTR advances. Select `--filter off|explicit10|compact10`,
`--derivative explicit|compact`, NP1/NP2 x/y/z, workspace, synchronization and
transport explicitly. `--reference` must be matching fixed-dt original native32
data. `prepare_native_explicit_reference.py --derivative compact` now also freezes
the actual original CPU/CUDA Fortran compact reference, without downcasting FP64.
`--quiet-compare` removes intermediate validation observations while retaining
one terminal snapshot, and requires full q/clock words to be unchanged. Formal
volume/slice/checkpoint output and rendering are disabled in these runs.

Evidence `results/multi_backend_mb7_native_main_20261116/summary.json` covers:
NP1 explicit/no-filter; NP2-x explicit10/full/device/dependency; NP2-y auto/pinned;
NP2-z compact derivative/compact10/scalar/device; and NP2-y explicit10/pinned
Memcheck. All physical density/internal energy is finite and positive; two
Memcheck logs have zero errors/leaks, and all observation comparisons preserve
full q/clock words. The auto window is growth-limited and does not reach the CFL
upper bound. It is not a CFL-bound stability or long-window physical test.

Under the existing D10 report-only FP32 accuracy policy, maximum final q/RHS
scaled differences to the matching original CPU/old GPU are respectively:
6.80024e-7/2.12997e-5 (NP1 explicit), 9.53675e-7/3.74020e-5 (NP2-x explicit10),
and 2.09108e-5/9.09294e-4 (NP2-z compact). Per-variable and scalar values are
retained, not compared with the FP64 2e-10 gate. The existing native compact FP32
long-window drift concern is not resolved or promoted to production accuracy.

Runtime `use_gpu=f` in the same executable matches a fresh CPU-only root build
from the same source/toolchain wordwise, including clocks. The older CUDA
Fortran CPU build is a different comparison and retains raw D10 differences.
The initial driver incorrectly demanded an absent fp32 banner; actual native
four-byte field records establish precision. Failed collector/driver evidence
is preserved, and completed moments records were reused rather than rerun.

The integration types numerical runtime arrays, MPI scalars, byte counts,
production means, sponge, crashfix and numerical sampling/moments/index buffers
by the build kind. Performance clocks and the explicitly fixed-FP64 test-only
snapshot interface stay separate. This is not a claim that all newly typed
optional modules have native FP32 numerical/lifecycle qualification. FP64
representatives for those six component groups, ordinary joint NSCBC/filter,
production mean sampling/exact restore and repair retain their original gates.

The repair probe's old history oracle predated approved N10 shared-node OR plus
halo refresh. The updated oracle independently gathers physical markers and
constructs expected shared/face values; q/cache references and tolerances remain
unchanged. Archived-old/current comparisons and failure logs are retained.

At10.116, the remaining physical-end filter used FP64 pointers/arithmetic.
Progress10.117 below closes that component dependency and adds bounded native
output/restore evidence; it does not qualify all nonperiodic consumers.
At that stage SBP63 and ordinary crashfix retained FP64 admissions. The native
repair component/TGV increment at10.136 below removes the latter precision
condition. SBP63 stays unchanged; the bounded FP64 viscous-flux repair increment
follows at10.138, while other mixed repair remains restricted. No CPU fallback
was added.

### Native Physical-End Filter and Ordinary Output Lifecycle

Progress10.117 uses `filter_basic_end`'s actual numerical kind for all closed
rows and its direct source pointer. The pre-fix native32 probe reproduced a
nonfinite read from NaN-poisoned physical exterior storage. Coefficients,
ownership, axis order and scalar/full storage algorithms are unchanged.
`astr_backend_boundary_filter` is now a root-CMake target in both kinds, linked
to `astr_backend_precision`. It checks three axes, lower/upper physical ends,
two local extents, explicit/dependency synchronization, five-component and
single-component workspace. Constant/Nyquist cases are exact; degree-five
polynomial errors are reported for native32 and retain2e-10 for FP64. Source
and exterior words, other scalar components, workspace/sync equality are exact.

Evidence `results/multi_backend_mb7_native_filter_end_20261116/summary.json`:

- Native32/native64:756 cases each, polynomial scaled maxima2.37887e-7/0;
  native32 Memcheck has zero errors/leaks.
- Unchanged FP64 high-wall41/42:252 cases each, maxima4.20740e-16/4.43099e-16.
- Ordinary16³ periodic TGV, four complete steps: NP2-x explicit643e/explicit10,
  scalar/pinned/dependency, exact2+2 restart and numerical archives; NP2-z
  compact643c/compact10, full/device/dependency, fixed and automatic exact2+2;
  NP1 explicit/full/pinned, nonexact restart and archives. All retain finite
  positive physical states and complete q/clock observation isolation.
- Exact runs match15 state datasets, full q, `control.bin`, `insitu_control.bin`
  and scalar words. Nonexact output actually omits primitive caches, redundant
  halo and exact geometry; this short fixed-step continuation has zero q/scalar
  differences, not a universal nonexact word-equality guarantee.
- Volume events at2/4; i/j/k=8 numerical products at0/2/4, restored products at4
  only. Exact volume/slices match continuously produced data. Density equals
  authoritative q words; same-phase slice/volume primitive fields are identical,
  Q/divergence raw native32 differences are retained (max4.17233e-7/2.98038e-7).
  Spatial payload/coordinates are FP32; existing clock/reference metadata is
  still FP64. The volume producer exports primitives to host at output events;
  this is not a no-flow-field-download or rendering qualification.

The lineage reader previously required recorded D2H bytes to be zero or equal
to selected payload bytes. C++ primitive export legitimately differs (664848
versus393040 bytes in NP2-x), preventing archive-enabled restart. Only that
diagnostic assumption is removed: counts must be nonnegative and payload layout
is independently checked against HDF5 metadata. No format or numerical-state
change. Four transfer-count tests and16 existing invalid-parent/layout tests
pass with matching GNU/HPC-X runtime. Early MPI-launcher mismatch and analysis
errors are retained, not used as acceptance evidence.

`run_cpp_native_tgv.py` adds `--restart exact|nonexact`, `--archives`, and
`--analyze-only` to inspect completed runs without advancing or rewriting inputs.
The first compact evidence directory includes `auto` in its historical name but
has `automatic=false`; the separate `automatic_compact` run actually uses auto.
Do not infer automatic control from directory names.

Native physical-boundary main/RK lifecycle, other native32 optional statistics,
high-order kernels/general geometry and new DCU execution still remain. The
larger MB6/MB7 goal is active. No Git or platform job operation was performed.

### Ordinary Native Nonperiodic Plate and RTI

Progress10.118 evidence:
`results/multi_backend_mb7_native_plate_20261116/summary.json`.
The plate/RTI drivers now accept `--precision fp32` and an explicit
`--native-reference` when freezing the original CPU/CUDA Fortran executable.
References are actual native32 execution, not quantized FP64. Default FP64
paths retain2e-10; native32 per-variable differences follow D10 report-only.

The16³, four-step plate matrix uses the existing static Cartesian profile and
flowini3d, Mach2/Re1000/Tw1, no chemistry/repair/positivity/mean statistics:

- MP7-LD743e/643e,bc11/21/41/50,no filter: NP2-x device fixed and NP2-z pinned
  auto advancement. They have no formal checkpoint qualification yet.
- Physical-space MP7543e/643e,bc11/21/41/51,explicit10: NP2-y scalar/explicit/
  pinned and NP1 full/dependency, output isolation and exact2+2 continuation.
  Checkpoint fields are native32;15 state datasets, full q, control/insitu
  control, clocks and scalar words match. Two filtered-plate Memcheck logs
  show zero errors; the earlier invocation did not enable leak checks.
- Native raw final-q/RHS scaled maxima are4.76837e-7/1.61527e-8 (MP7-LD
  fixed),5.72205e-7/1.49012e-8 (MP7-LD auto),2.06308e-6/1.00006e-5
  (MP7 with filtering). Physical rho/p/T are positive and finite; pre-RHS
  cache density equals authoritative q words, with other cache relations
  independently reported using FP64 diagnostics and source-kind constants.

Original GPU rejects the attempted MP7-LD643e+bc50+explicit10 combination;
it was not bypassed. Static read-grid ninit3 MP7-LD checkpoint currently has
an explicit FP64/compact-only admission; native32 expansion was requested
from the user and remains unapproved. Do not substitute advancement for
restart evidence or the bc51 filter reference for the rejected combination.

The RTI gravity kernel now uses the numerical kind, retaining its formula
and multiplication order. The shared RTI probe is typed through
`astr_backend_precision`; FP32 CPU/oldGPU capture and new C++ comparison
at NP2-y cover24 cases/rank, exact untouched state/equation/cache footprints
and repeated/sync words. FP32 raw maxima2.98024e-8; default FP64 to archived
references5.55112e-17. Native32 Memcheck has two zero-error/zero-leak logs.
Only the selected RTI source SASS is qualified as having no FP64 arithmetic
or conversion, not all high-order kernels in the source file.

Ordinary native32 RTI+KEEP643e with bc1/1/31/31/1/1, no viscosity/filter,
dt1e-6 and NP2-y device/dependency completes4 vs2+2 steps, exact15 state
datasets/full q/control/insitu control/scalars and output isolation. Raw
q/RHS maxima1.56827e-12/3.05176e-5 are both retained: a small short-window
q difference does not prove small RHS error. Default FP64 ordinary RTI is
also rechecked against its archived CPU/oldGPU fields and original2e-10
gate. These are implementation/lifecycle checks, not developed RTI physics.

The larger goal still includes native optional consumers, NSCBC/high-order
arithmetic, geometry and fresh DCU execution. No Git or remote operations.

### Native Mean Statistics and Inherited History

Progress10.119 closes representative native32 evidence for the production-mean
interfaces typed in10.116. No production formula, default or public format was
changed in this increment. Root CMake builds
`astr_backend_production_statistics` with `astr_backend_precision` in both kinds.
The old CUDA Fortran observation program is typed through the same kind and
linked against immutable original native32 objects, not a quantized64 result.

Evidence `results/multi_backend_mb7_native_statistics_20261116/summary.json`:

- Component28 cases/rank,798 scalar channel observations/rank: nonuniform/warped
  span measure,12 raw moments, nonuniform trapezoidal time integration, oblique
  wall pressure/traction/heat flux, reset/repeat/restore and untouched state.
  Native32 per-channel max2.18696e-7; the older whole-buffer normalization gives
  3.82383e-8 and is not substituted for it. NP1 Memcheck zero errors/leaks.
  NP2 components repeat the same input on separate GPUs, not spatial MPI.
- Default FP64 per-channel max3.95255e-16 against existing frozen data, with
  the original2e-10 gate. Native32 values remain D10 report-only.
- Ordinary16³543e/643e,explicit10,bc11/21/41/51,ninit0 static-profile plate:
  NP2-z pinned fixed dt1e-6/step sampling; NP1 pinned and NP2-y device automatic
  dt/time sampling. Four complete steps, no chemistry/repair/positivity.
  Fixed sampling has5 samples including0, duration3.999999989900971e-6;
  time sampling has3 samples at0/2/4, duration4.6410004870267585e-6.
- Sampling and intermediate-output isolation preserve physical state/full q
  halo and clocks. Existing periodic velocity/temperature cache refreshes are
  separately accounted for. Exact2+2 restores15 state datasets,74 v4 or37 v3
  statistic datasets, full q, controls/archives/insitu control and scalars.
  Spatial statistical payloads are actually FP32. The NP1 automatic test also
  performs old-GPU nonexact migration with nonzero cumulative history.
- Actual device-MPI6-step inheritance: NP1→NP2-x→NP2-z; nonzero inherited
  integrals and sample/clock identity survive, followed by an exact same-z tail.
  Final state versus continuous z is equal; raw statistic max3.41061e-13.
  This does not extend the existing x/z repartition admission to y.

CPU44 and GPU12 raw plane moments plus32 auxiliary storage channels are not
identical protocols. Their defined common12 moments are independently
span-integrated and normalized using source-kind coordinates and actual time
weight; offline arithmetic is FP64. Mean density/pressure, Favre velocity/T and
six Favre covariances have raw native32 maxima1.22175e-6 fixed/1.34185e-6 auto.
No clipping, FP64 cap or statistical-convergence claim is added. The same
analysis on immutable default64 data gives1.11023e-15 under its2e-10 gate.

The first NP2-y driver incorrectly compared v3/v4 storage version as a clock.
Only that oracle was corrected: both layouts were already validated and
canonicalized, and all actual sample identities/time records match exactly.
Failure evidence remains. Two immutable four-step reference descriptions also
say "six" in prose; their steps/input/logs are4, with a separate correction file.
Reference precision is now checked from actual HDF5 dtype, not optional JSON
report fields. No old-format compatibility or production fallback was added.

Representative wrappers now take `--precision fp32`; component cases may be
selected explicitly with `--cases`, and original-reference preparation takes
`--native-reference`/`--steps`. Native runtime window switching, broader geometry,
other consumers/high-order/NSCBC arithmetic and DCU remain pending. The separate
static-read MP7-LD precision/output question remains unapproved. No Git/remote.

### Native Low-Order x-NSCBC

Progress10.121 evidence:
`results/multi_backend_mb7_native_nscbc_20261116/summary.json`.
The existing x-inlet/outlet probe and runners take `--precision fp32` and use
actual kind/width. References link original native32 objects, not quantized64.
The generic CPU bc12 is not the approved repaired-inlet oracle: use actual
repaired old-GPU results and independent uniform characteristic projection.
Outlet references include both original CPU and old GPU. No CPU repair or
production formula/default change is made here.

Native32 cases replace unrepresentable1±1e-9 offsets by128*epsilon(kind) and
record actual velocities. The boundary's64epsilon rule is unchanged. FP64
inputs remain unchanged for immutable references. NP2-x outlet has40 observations
and400 per-variable records/rank; NP2-y inlet48/240. Conservative q and nonface
RHS remain unchanged; explicit/dependency words match. Uniform and nonuniform
all-outgoing states ignore changed external targets exactly (4 outlet/8 inlet
checks). This tests target independence, not zero floating derivative residual.
Native outlet Memcheck has two zero-error/zero-leak logs. Default64 outlet/inlet
scaled maxima1.87878e-13/1.02029e-14 satisfy the existing2e-10 gate.

Native32 inlet max5.66412e-6; outlet max1.38373e-4 is retained, concentrated in
nonuniform-metric/high-tangential-speed case20 energy RHS. The uniform subset
max is5.48363e-6. FP64 agreement is consistent with cancellation sensitivity,
not complete attribution or a native32 production-accuracy endorsement.
Independent FP64 uniform formulas use rounded primitive inputs; native field
generation/arithmetic rounding stays in reported residuals. D10 adds no FP64 cap,
and no new local FP64/mixed path or expression reorder is introduced.

Ordinary MP7 physical-space OpenShock (543e/643e,RK3,bc12/22 with periodic
transverse directions,no viscosity/filter/sensor/repair/mean,dt1e-5,four steps):
NP2-x device/dependency at64×8×8 and NP2-z pinned/explicit at64×16×16 use
separately matching actual old-GPU NP1 references. Final q maxima are
2.47256e-7/2.47249e-7; complete RHS after the source stage about2.42144e-8, scalar max1.19208e-7,
clocks equal. Observations/output do not change q/clocks. Exact2+2 restores15
state datasets, full q, controls/archives/insitu control and scalars, with real
FP32 storage. Transverse main Memcheck has two zero-error/zero-leak logs.

The initial64×8×8 NP2-z test was rejected before RK: local width4<halo5.
It remains a failed fixture, not a clean memory test. Only an explicitly chosen
larger transverse grid with a matching reference qualifies; no automatic resizing,
retry or guard bypass was added. These results do not coverbc52, viscous joint
HBL/GRCBC, developed SBLI or all native consumers. Other MB6/MB7 work and fresh
DCU execution remain required; no Git/remote operations.

### Native Precision Scope and Sponge

Progress10.120, evidence
`results/multi_backend_mb7_native_sponge_20261116/summary.json`.
Actual original native32 execution rejects Hermite7 before computation; shared
configuration also restricts H1, SBP63 and ordinary ME6 to FP64. Real-kind algebra
or direct component evidence is not ordinary-main admission. Those existing
FP64 capabilities and their consumers/DCU stay in the full goal; no new native32
high-order method is authorized by this migration. An attempted local high-wall
type trial was removed bytewise from production source and the ordinary main
rebuilt. Only unqualified trial/failure artifacts remain under
`results/multi_backend_mb7_native_high_wall_20261116`; no runtime fallback added.

The original CPU/GPU sponge production functions already use the numerical kind.
Only their observation/comparison programs needed native typing; root CMake
links `astr_backend_sponge` with `astr_backend_precision`. New wrapper option
`--precision fp32` reports actual width and each of five component differences.

Eight native32 component runs cover x+/y+/overlap, NP1/NP2 x/y/z, pinned and
device MPI, two synchronization modes, and NP2-y pinned Memcheck. They contain
90 rank-cases; new-to-CPU maximum per-variable scaled difference9.53675e-8.
The original CPU/oldGPU pair is separately captured, not mislabelled as another
direct comparison to the new backend. Complete sync words match; two memory
logs show zero errors/leaks. Native32 constant residual4.7683716e-7 is recorded,
not clipped or claimed exactly zero; the linear five-array fixture is not a
thermodynamic flow state. Default64 NP2-y maximum per-component difference is
about2.21491e-16 with zero constant residual under the existing2e-10 gate.

An ordinary native32 low-order bc41 channel numerical fixture (16³,643e/643e,
no filter/feedback/mean, y+ sponge thickness4, dt1e-5, four steps) runs NP2-y
device/dependency. The reference is explicitly original CPU only: the old GPU
channel main does not invoke this sponge, although its actual functions are
covered above. New q/RHS maxima9.44012e-7/7.67061e-7 are D10 raw reports.
Output-off and2+2 continuation preserve full q, clocks,15 state datasets,
controls/archives/insitu control and scalars exactly, with actual FP32 storage.

No production formula/default or original precision guard was changed. Remaining
native work follows admitted low-order NSCBC/inlet/source/window consumers,
not an unrequested native32 high-order extension. Other FP64 MB6/MB7 integration
and fresh DCU evidence remain required. No Git/remote operation was performed.

## Native Upper-Y NSCBC And Joint Viscous Advancement

Progress10.122 and results/multi_backend_mb7_native_top_20261116/summary.json
extend the existing top observation harness to its actual numerical kind.
Both astr_backend_nscbc_top and astr_backend_nscbc_top_viscous link
astr_backend_precision. Use run_nscbc_top_components.py --precision fp32
with native32 original objects for freezing, then --reference/--binary for
the new backend. Byte counts, packets and MPI use the source kind. FP64
retains2e-10; D10 native32 reports differences without an invented cap.

NP2-y checks owner/nonowner, two x-end layouts, four existing policies and
12 states, plus a distinct viscous-source capture/consume matrix. Repeated
reference replay and explicit/dependency RHS words must agree, q/interior
writes are protected, and all observed RHS must be finite. Viscous snapshots
reject missing/double consumption; zero source preserves RHS words. A64-plane
multi-block owner Mach2 manufactured case has zero error. The normal and
viscous native32 Memcheck runs each have two clean, leak-free rank logs.

New native32 differences to CPU are at most2.68221e-7, but to the unchanged
old GPU reach1.29469e-3 (normal) and3.48699e-4 (viscous). The high-tangential
constant state is retained. A controlled result-only default-FMA rebuild
reproduces every old-GPU RHS word; disabling only that module's GPU FMA makes
the high-tangential cases agree exactly with CPU. All other objects remain
frozen. This diagnoses contraction sensitivity, not production accuracy or
a reason to overwrite the original reference. CPU native32 constant-state
residuals are still nonzero. Production compiler options/formulas are unchanged.
FP64 maxima5.26571e-12/1.41761e-12 retain the original gate.

run_cpp_plate_gate.py --variant mp7ld_compact_nscbc selects the existing
MP7-LD743e/643c,bc11/22/41/52 incoming_only numerical fixture. Its native32
--advance-only matrix runs NP2-y device/dependency fixed and NP2-z
pinned/explicit automatic timesteps, with matching actual CPU/old-GPU
references. --quiet-compare checks full final q/halo and clock/scalar words
against final-only observation. It does not change the boundary's transverse
filters or its staged RHS ordering. Actual terminal differences are at most
9.42497e-7 fixed and5.65498e-7 automatic; all24 updated rank states are finite
with positive density/internal energy. The automatic run has clean per-rank
Memcheck logs. No checkpoint, mean, rendering or physical-SBLI claim follows:
static ninit3 native32 checkpoint admission still awaits its separate approval.
The component keep_corner=0 fixture also does not close LA3-14. Other MB6/MB7
native consumers and fresh DCU evidence remain outstanding.

## Native Mean Windows And Disable/Write Handover

Progress10.123, results/multi_backend_mb7_native_windows_20261116/summary.json:
run_cpp_legacy_statistics_windows.py accepts --precision fp32 and can freeze
an actual dual-runtime original executable with --freeze-only. It reuses the
existing designated-input read replay, not a new production hook. Native clocks
decode32-bit words and check interval addition in the source kind. FP64 keeps
the original2e-10 gate; native32 reports field/channel/normalized differences.

The existing16-cubed stretched bc11/21/41/51,543e/643e,explicit10 plate runs
ten automatic steps. CPU/old-GPU references first pass their own continuous/
step5-restored comparisons. New native32 NP1/NP2-y check clock changes at3,
disable4--5, re-enable6, cadence change9 and publication9/10 with samples
6/8 and6/8/10. Disabled step5 has no statistics payload; restoring there starts
the new window correctly. State, statistics and all three controls are wordwise
equal. Native32 field/channel maxima3.69271e-6/3.02531e-6 are retained.

--handover instead disables at the existing step4 sample/write event. Native32
NP2-z and FP64 NP2-y publish the old0/2/4 window first, including the newly
selected dt_next, then disable. Final/step5 checkpoints contain no statistics.
State/control/time restore remains exact. The existing FP64 ordinary windows
also remain below2e-10. No production arithmetic, format or resource budget
changed; this is not statistical stationarity or full MB7 qualification.

## Native Periodic KEEP Ordinary Main

Progress10.124, results/multi_backend_mb6_native_keep_main_20261116/summary.json:
prepare_native_explicit_reference.py and run_cpp_native_tgv.py both accept
--convection keep (default standard). The latter checks the frozen input's
actual convection selection before running. These extend the existing native
KEEP component evidence to ordinary four-step TGV and lifecycle consumers.

Actual CPU/old-GPU native32 references cover643e/643c and filter off/on.
Representative new runs use NP1 and NP2 x/y/z, pinned/device, both sync modes,
explicit10/compact10, exact2+2 and one lean nonexact2+2. The fixed terminal
maxima are6.80024e-7 unfiltered,9.53675e-7 explicit10,2.03158e-5 compact10.
The latter's RHS difference7.81455e-4 and the original CPU/old-GPU terminal
difference2.17608e-5 remain visible; D10 is not a production-accuracy claim.
One automatic compact KEEP run checks controller/restore only, without a
matching automatic original trajectory reference. Six explicit-filter
Memcheck rank logs (continuous/prefix/resumed) are clean and leak-free.

The filtered explicit NP2-y case also writes real native32 volume/gradient/
vorticity/Q data and i/j/k index products; final restored data are exact.
Derived archives are still restricted by the existing lfilter=true TGV
admission: the rejected unfiltered request is retained, not bypassed. No
Catalyst/rendering ran. Archive schedule words must agree, while the final
six int64 values record each actual segment's distinct checkpoint/parent
provenance; segment sizes are checked separately. Inactive archive controls
still compare as complete byte strings. No production formula/default changed;
physical-boundary native KEEP, other consumers and new DCU remain pending.

## MP Face Budget Component

`prepare_mp_positivity_reference.py --build <CUDA-Fortran-root-build>
--output <new-evidence-directory>` relinks the root-built production objects
with a test-only caller of the actual MP device functions. It records the
current CUDA Fortran implementation and hashes; it does not certify a defective
original as correct. Use a fresh output directory and preserve frozen evidence.

Build `astr_backend_mp_positivity_math` through root CMake, then pass the
frozen `input.bin` and `reference.bin`. The FP64 probe requires exact base-state
admissibility flags, face-coefficient differences <=2e-10, finite coefficients
in [0,1], and strictly admissible returned states for every admissible base.
Known rho/energy/momentum limits are also checked analytically when freezing.

Progress10.132 and `results/multi_backend_mb7_mp_positivity_20261116/summary.json`
record the approved roundoff-interior repair: 526 cases, 411 active budgets,
five defective original returns eliminated, C++/repaired-CUDA coefficient
difference7.21645e-16, and clean Memcheck including leak checking. Ordinary
CUDA Fortran NP1/NP2 x/y/z four-step plate runs preserve all stage/scalar/clock
bits before/after in the tested smooth window, whose limiter count is zero.
That initial component alone did not qualify main-program/MPI integration.
The subsequent local integration evidence is described below; default off is unchanged.

## MP Complete-Stage Integration

`mp_positivity_operator.{cu,F90,h}` reuses the existing high-interface operator,
stores three directions of high-minus-low flux in optional field22, and theta
in optional field23. Prepare/begin/build operate on device state; the final
source/boundary callback applies budgets to the complete RHS before RK.
Shared-node coefficients use minima, not averaging. Only face packets use the
selected pinned/device transport. Scratch is recomputed, not a new history file.

`run_cpp_mp_positivity_gate.py` compares all12 actual RK stages of the frozen
16³ MP7-LD/643c plate. Use `--analyze-only` to repair analysis without rerunning
an unchanged executable. Main halo/MP and compact-PCR transport/sync selectors
are independent: `--transport device --compact-transport device-aware` selects
both; merely choosing device does not change the private compact default.

`prepare_mp_positivity_stages.py` relinks a test-only observation entry onto
the root CUDA Fortran objects. Its ordinary trajectory must remain bytewise
equal to the already frozen legacy trajectory. `run_mp_positivity_stages.py`
then checks original faces and a separately labelled 2^40 anti-diffusive replay
with an algebraically unchanged complete low-order baseline. NP1 and NP2 x/y/z each cover
explicit/dependency execution and pinned/device packets. In the active replay,
old diagnostic counts3825/4080/4080/4050 include duplicate shared rank nodes;
these are not unique global-cell counts. Corrected RHS numerical difference is0.

The active RHS comparison exposed large-cancellation sensitivity to contraction.
SASS and independent scalar replay identify the original sequence as two FMA;
the C++ correction states this sequence explicitly, without global fast-math
changes or tolerance relaxation. Original failed data remain in the evidence.

Progress10.133 and `integration_summary.json` distinguish numerical success,
clean pinned Memcheck, and a failed CUDA-aware sanitizer run. The unchanged,
MP-free halo binary reproduces CUDA API reports and three UCX IPC mapping
leaks; device-mode sanitizer is not certified clean. No suppression, library
patch or automatic backend substitution was used. Actual device numerical
matrices still pass. Formal MP continuation awaits numerical-revision approval;
other joint boundaries, native precision and DCU MP are not promoted here.

The GPUfusion build entry now includes both MP math and complete-stage
targets. Progress10.139 records a fresh DTK26.04 gfx906 build of these targets,
the ordinary main and crashfix for FP64/FP32; all eight dependency listings
resolve. This is login-node build evidence only, not DCU MP/RK execution or
formal MP continuation qualification. See
`results/mb67_compile_latest_20261116/summary.json` for the exact source snapshot.

## Adaptive Complete-Step Output Collector

`astr_device_fluid_adaptive_energy_sum` reads authoritative q on each rank's
0:upper-1 nodes, not the ordinary scalar-statistics 1:upper convention. It
returns sum(momentum squared / density) and the invalid-node flag; Fortran
retains MPI, normalization, event/window scheduling and ASTROC08 history.
The two 256-thread reductions reuse the existing small statistics buffers.
No full-field download is introduced by the monitor. Existing periodic
Cartesian 16/32-cubed TGV admission and the default disabled setting remain.

Build `astr_backend_adaptive_energy` through root CMake for the closed-form
polynomial, invalid-input, empty-sum and q-word checks. The GPUfusion build
entry includes this target, but this new collector has not been DTK-built/run.
`run_cpp_adaptive_output_gate.py --original <actual-old-binary>` freezes native
CPU/CUDA-Fortran references; `--binary <cpp-binary> --reference <directory>`
checks monitor values/clocks, actual six-field frames, 2+2 exact state/control/
frame continuation, and full-q/halo output/monitor isolation. `--precision fp32`
uses actual four-byte state and D10 raw-error reporting, not the FP64 bound.
`--analyze-only` reuses completed C++ runs; it does not rerun the solver.
Progress10.140 and its evidence summary distinguish the FP64 gates, native
FP32 rate-error amplification, pinned memory checks and not-yet-tested DCU
or image-product integration. No new backend/precision restart protocol is used.

## Native FP32 CURVE Main And Frozen-State Gradients

`prepare_native_curve_reference.py` freezes actual native-FP32 CPU and original
CUDA-Fortran runs of the existing FP5 32-cubed periodic/y-wavy fixtures. It keeps
their dt=2e-5, four steps,643e/643e and scalar explicit10 filter. Do not substitute
quantized FP64 states. `run_cpp_native_curve_gate.py` runs the ordinary C++ main,
checks geometry kinds, native field/cache words, complete-step clocks, 2+2 exact
continuation and full-q/halo observation/output isolation. `--analyze-only` reads
existing results without executing CFD; `--reuse-continuous` explicitly reuses
an already completed primary calculation while executing the planned lifecycle
cases, rather than inferring success from directory existence.

Build `astr_backend_native_curve_gradient` through root CMake in the native32
configuration. `run_native_curve_gradients.py` packs each frozen backend's actual
primitive fields and all nine metrics. The probe tests existing boundary-aware
device derivatives under both synchronization modes, constant and computational
coordinate affine fields, physical-exterior poison and unchanged input words.
`check_geometry_face_halos.py` preserves the HDF5 dtype and additionally reports
internal-halo and shared-node owner-word agreement; existing FP64 numeric checks
are unchanged. `test_native_curve_samples.py` distinguishes harmless decimal
timestamp formatting from different native times or sampling steps.

Progress10.141 records six actual main cases,40 frozen-state packets,240 physical
RK snapshots and12 same-phase field frames. FP32 differences follow D10 without
a new ceiling. The largest frozen temperature-gradient difference is reproduced
from the captured FP32 values by fused versus separate multiply-add; production
compiler settings stay unchanged. This does not qualify long-time/physical FP32
accuracy, new compact-CURVE combinations, image products or DCU execution. The
GPUfusion build list includes the probe target, but target registration is not a
DTK build/run receipt for this increment.

### Actual Native-FP32 Images

`run_cpp_insitu_gate.py --precision fp32 --images --restart` now drives actual
native-FP32 original/C++ programs through the existing slice, Q_rs=0.25 and
instantaneous-streamline products. Select a matching native32 Catalyst/ParaView
directory with `ASTR_OUTPUT_INSITU_BACKEND` before launch; `--precision` does not
convert a FP64 third-party library. The default remains FP64 with its existing
bound. Native32 comparisons report raw differences without the FP64 ceiling,
while native types, finite/positive physical state, exact observation isolation,
same-build state/control continuation and identical resumed pixels remain checks.

`--mapping periodic` and `--mapping y-wavy` reuse the existing32-cubed fixtures;
native CURVE uses dt=2e-5. Progress10.142 records the two actual device pipelines,
NP1/NP2 x/y, fixed steps and one acoustic-CFL automatic-step/time-product case.
All45 primary JPEGs and15 resumed JPEGs were checked, with zero recorded render
field/geometry download and no EPS. These counters exclude checkpoint transfers.
The evidence includes exact binaries and source/build snapshots; this is not a
performance comparison, a native mean/wall-image qualification or a DCU run.

Progress10.143 subsequently adds actual native32 mean images through
`run_cpp_mean_images_gate.py --precision fp32`. Its Cartesian velocity-time,
Cartesian selected-time and periodic-CURVE velocity-time cases use the same
frozen programs/dependency, with explicit physical-window settings. Product
times are checked against completed-step logs, not step times nominal dt.
State/statistics rendering isolation and same-build2+2 continuation remain
word-exact; mean-image continuation remains pixel-exact. Native raw deviations
are reported separately, without a FP64 accuracy claim or DCU qualification.

`compare_statistics(..., precision='fp32')` decodes each regional integer word
as signed32 bits before viewing it as FP32. The default FP64 decoder and2e-10
bound remain. `test_statistics_archive_precision.py` checks both signed formats,
report-only native differences, rejected nonfinite regional values, false kind
markers and the retained FP64 error bound. This is test-reader logic only, not a
change to checkpoint serialization. Current legacy wall capture/device-product
providers still need C++ integration; the separate thermal/wall FP64 admission
is not relaxed by the mean-image results.

## Approved Static-Profile And MP Checkpoint Changes

Progress10.145 fixes only the existing `wall_curve + reynolds + wall-time`
parser distinction. `prepare_legacy_wall_reference.py` freezes actual CPU and
CUDA-Fortran32-cubed y-wavy wall curves and wall-time results in both native
kinds; the original CSV schema includes step/time/global_i as well as xyz/value.
CSV-only references use the existing no-render startup configuration. They do
not qualify the still-unimplemented C++ legacy device wall provider/history.

Progress10.146 adds MP budget revision bit5 to existing control word7 while
retaining MP mode bit1. In scalar, nonmixed examples off remains24, previous
flux is26 and repaired flux is58. Exact restore rejects the previous budget;
explicit nonexact may update revision bits, but cannot use this to toggle MP
mode. No new file, deprecated-format reader or production checksum is added.
The existing static, nonrandom, read-grid profile definition now admits native32
output/restart without widening the separate native high-order/thermal gates.

`run_profile_checkpoint_gate.py` exercises actual4/2+2 runs, native volume and
three index slices, complete-step cache words, controls and RK output isolation.
The optional previous-budget test uses an actual frozen old-budget executable,
not a patched checkpoint header. `--nonexact-probe` checks full/light sources
and five-variable light output. Input resources remain shared, original files;
do not count them as converted FP32 numerical payloads or promise half-size I/O.
Archive schedules are exact; the final48-byte per-segment receipts intentionally
differ for a new output segment. The RK snapshots are test-only full downloads,
not a performance or transfer-free production run. Progress10.146 records10
bounded local cases, not all precision/topology/statistics combinations.

`astr_backend_compare exact` now recursively compares group/dataset identities
and all nested dataset words. `test_field_compare_nested.py` checks matching
FP32/FP64 histories and changed/missing/retyped/extra-empty nested objects.
This fixes the post-run tool failure in task128215915; its seven actual CFD
results were reviewed independently without resubmission. It does not insert
HDF5 traversal into production checkpoint writes. Progress10.147 preserves the
original script failure separately from accepted numerical results and records
the frozen DCU programs; it is not latest-source MP/native-output qualification.

### MP Plus NSCBC/H1 Boundary Coupling

The same `run_cpp_mp_positivity_gate.py` accepts frozen MP-on incoming-only
and production-H1 plate references. Compact-PCR selectors are checked only
when the fixture actually selects643c; H1 fixtures additionally require both
the original H1 activation marker and the C++ production-H1 marker.

Progress10.134 and `joint_boundary_summary.json` record16-cubed, four-step,
NP1/NP2 x/y/z comparisons. The first H1 run failed because the new MP branch
excluded upper physical nodes from its body box but omitted the legacy
post-body boundary convection assembly. The MP operator now reuses the
existing high-order difference kernel on the upper x/y union, including its
normal derivative and excluding prescribed lower faces. Each direction uses
its current interface cache; the upper intersection is not duplicated.

The failed reference/comparison and repaired comparison are separate. Maximum
stage-field scaled error is5.77086e-15, RHS5.83301e-17; the pinned NP2-y H1
Memcheck has two clean rank logs including leak checks. All actual four-step
trajectories here have zero limited-node counts. Active-stage replay, device
sanitizer, restart identity and DCU remain distinct gates, not inferred from
these smooth boundary windows. No physical boundary formula was changed.

### Native-FP32 MP Checks

`prepare_mp_positivity_reference.py --precision fp32 --normal-range` builds
the actual CUDA Fortran budget caller with the selected root-build modules.
The approved normal-range family has269 cases,262 admissible bases and207
active budgets. The earlier270-case run remains failed: its positive
subnormal-density control is rejected by the old FTZ classifier and accepted
by the new C++ classifier. It is recorded separately, not normalized away or
reported as cross-compiler equality. No production FTZ policy was changed.

Build `astr_backend_mp_positivity_math` and `astr_backend_mp_positivity_stage`
through root CMake with `ASTR_NUMERICAL_PRECISION=fp32` and
`ASTR_BUILD_GPU_BACKEND_PROBES=ON`. Their packet scalars/arrays must be4-byte
native values. The original stage exporter uses2^20 anti-diffusive stress
for FP32 and retains2^40 for FP64; these are operator replays, not flow cases.
`run_mp_positivity_stages.py --precision fp32` reports coefficient and RHS
differences without the FP64 limit, while retaining finite/admissible states,
unchanged input-q words, actual RK updates and genuine limiter activation.

`run_cpp_mp_positivity_gate.py --precision fp32` compares frozen same-topology
MP-on main-program references and records both maximum absolute and component
scaled errors. It does not require FP32 limiter activity counts to reproduce
another compiler exactly; the actual counts and equality flag are reported.
The tested incoming-only plate counts are zero, so active-stage evidence
remains separate. Progress10.135 and `native32_summary.json` cover NP1/NP2
x/y/z, both packet transports, ordinary/active pinned Memcheck and affected
FP64 probe checks. They do not qualify native H1, formal MP restart, DCU MP
or a device-mode sanitizer result.

### Native-FP32 Crashfix

Progress10.136 and `results/multi_backend_mb7_native_crashfix_20261116/summary.json`
record the native repair increment. Production kernels, C reports and Fortran
callbacks already used the build kind; the ordinary admission no longer
requires64-bit storage. Existing five-variable/format/nonmixed restrictions
and repair defaults remain. This does not qualify all physical boundaries.

`prepare_crashfix_reference.py` relinks root legacy objects with the observation
entry. `prepare_crashfix_components.py --precision fp32` freezes actual old
CPU/GPU component fields. The large finite donor uses `huge` of the build kind,
not a converted FP64 overflow. `run_crashfix_components.py --precision fp32`
checks typed packet length, finite/positive states, history and unchanged RHS/RK
origin; numerical differences are reported under D10. Use the full expanded or
reconciled API for references captured after history coordination. A raw local
ordered call is a different observation phase, not evidence of faulty shared
markers; the first such failed comparison remains recorded.

`run_cpp_crashfix_gate.py --precision fp32 --seeded-probes` uses real steploop,
repair, checkpoints and restoration, with only one deliberately invalid initial
node. Representative TGV tests cover central/explicit10 and KEEP643c/compact10,
fixed/automatic time control, output isolation, exact2+2 continuation and
nonzero-history disable/restore. The ordinary production executable also reads
an old-GPU native nonexact checkpoint without losing history.
`run_cpp_crashfix_hot_control.py --precision fp32` checks actual input polling
and off/on/off/on lifecycle using native-width bit comparisons. None of these
is natural-flow crash or long-window physical validation. Native MP7-LD plate,
other complex-boundary combinations and new DCU evidence remain separate.

### Native Plate Repair Without Archives

`run_cpp_crashfix_plate_gate.py` uses the existing16-cubed incoming-only
MP7-LD/643c fixture and the seeded old/C++ test entries. Formal checkpoints,
volumes and slices are disabled. It compares all12 RK states, primitive caches
and conv/full RHS; physical rho/internal energy/p/T remain finite and positive.
FP32 errors follow D10 without the FP64 cap. NP1 and NP2 x/y/z include pinned,
device and an automatic-clock representative; the compact transport remains
independently pinned. This is same-topology coupling, not partition invariance
of the rank-local injected defect or physical instability validation.

The test-only history prefix enables a final int8 marker capture. Each entry
checks full device-q words before and after that readback in the same run,
without an extra quiet CFD trajectory. New/old history packets must match
exactly. `--analyze-only` can reuse completed fields without advancing; a first
NumPy-integer JSON failure and its repaired analysis remain separate from the
later observer-v2 runs. Progress10.137 and the native-crashfix-plate summary
record the actual errors and clean pinned rank logs. Native static-profile
formal output/restart, H1 FP32 and DCU remain distinct scopes.

### Mixed Viscous Storage With Repair

Progress10.138 adds only existing FP64-state/FP32-`viscous_flux` storage to the
ordinary repair admission. Native-FP32 mixed, derivative/interface/characteristic
mixed repair remain excluded. Defaults, physical floors, donor traversal and
history ownership do not change.

`run_cpp_crashfix_gate.py --mixed-candidate viscous_flux --freeze-only
--advance-only` freezes the old GPU compute trajectory without claiming old
mixed checkpoint support. The initial formal-output attempt is retained as an
admission failure before RK. `--frozen-reference <directory>` explicitly selects
those raw-stage references, with matching input/kind/mode/topology/binary
identity. It is not an automatic fallback for a missing checkpoint.

The new C++ path separately tests its existing formal output and exact2+2
continuation contract against the raw old terminal field. NP1 fixed and NP2-x
automatic/device representatives preserve nonzero history and all18 state
datasets/control/scalar/clock words. An additional pinned compute check runs
Memcheck through actual startup repair. The old candidate first retains its
existing1e-9 field/1e-10 scalar FP64 comparison; new same-mode operators/RK keep
2e-10. No CPP-P optimization, new old-GPU I/O support, physical stability claim
or new DCU result is implied.

### Legacy Wall Device Components

The root target `astr_backend_insitu_wall` builds in both native precisions.
`run_legacy_wall_components.py` takes `--binary64`, `--binary32`,
`--reference`, `--output` and the matching `--mpiexec`. The reference is the
immutable four-case CPU/old-GPU directory produced by
`prepare_legacy_wall_reference.py`, not newly generated C++ values.

The component implements legacy conservative endpoint ownership, three-layer
wall quantities, the original eight-node Cartesian closure, and curved-wall
metric/stress/thermal projection. A static wall geometry strip is uploaded
once; only explicit host consumers download fields or selected six-value
lines. Synthetic periodic upper duplicates are deliberately invalid until
private endpoint completion. NP1/NP2 x/y/z representatives exercise pinned
and actual device MPI; both synchronization modes check parent-q isolation,
failed-capture invalidation and empty-wall lifecycle. Actual complete-step4
pressure/shear/heat CSVs are compared per quantity and side.

Progress10.148 records the 16 component runs. FP64 uses the approved2e-10
scaled bound; nativeFP32 reports raw D10 differences without that ceiling.
Component checks alone do not prove the ordinary catalog wall consumer:
producer dispatch, existing30/34-state lifecycle and actual new-NVIDIA
product/restart checks are recorded separately below. The newer FP64-only thermal
diagnostic admission is unchanged. No new DCU or performance result is claimed.

### Legacy Wall-Time Ordinary Main

Progress10.149 adds `insitu_wall_runtime_cpp` and the actual ordinary-main
consumer. `run_legacy_wall_runtime.py` checks twelve legacy curves, the existing
34-component wall-time checkpoint group, exact2+2 state/statistics/control and
product clocks, plus statistics-off authority isolation. `--freeze-mpi-reference`
explicitly generates same-topology old CPU/GPU references; NP1 can reuse the
immutable reference-v2 directory. NativeFP32 follows D10, not the FP64 ceiling.

`--images --implementation <matching Catalyst directory>` also produces
instantaneous pressure and wall-time mean shear JPEGs using the existing host
Matplotlib curve backend. Each plot's33 values/range match its numerical curve;
same-backend restored pixels must match. The global line is reduced before
plotting and only rank0 publishes chart receipts; all ranks still participate
in the device sample/private MPI and report their selected-curve contribution.
This is not a new 3D GPU renderer. `--transport` selects private postprocessing
MPI; the solver halo remains pinned in this matrix.

`--automatic` requires explicitly frozen matching references. Its test helper
opts into variable-dt CFL checks paired with actual complete-RK time_end and
dt_used; the0.5 CFL ceiling is not removed. `--reuse-completed-full` is explicit
analysis-only reuse of completed references/full after fixing a test reader,
with the frozen executable checked; it starts only unexecuted prefix/tail/quiet
runs and is never an automatic retry or missing-file fallback.

`run_legacy_wall_migration.py` separately checks old-GPU -> C++ explicit
nonexact restoration of existing wall-time history. It does not claim exact
cross-implementation continuation. `--memcheck` exercises ordinary restore,
sampling, curves and checkpoint serialization under Compute Sanitizer.
Eight-node Cartesian closures still have component rather than ordinary-main
evidence here; physical-midspan and other wall-surface consumers, fresh DCU
execution and large/physical validation remain separate tasks.

### High-Wall Consumer Reference Preparation

`prepare_high_wall_consumers.py` reuses the existing16-cubed Hermite7
fixtures and freezes an original dual CPU/CUDA-Fortran executable before
new-C++ wall-consumer changes. It checks existing complete-step fields and
the actual `ASTRIW01` wall snapshots rather than deriving another reference
from the new C++ implementation.

Progress10.150 records the stopped first attempt: CPU advancement finishes,
but the declared nine-int32 snapshot header contains only seven elements.
The root Fortran target `astr_wall_header_probe` independently demonstrates
the NVHPC mixed-kind constructor/SHAPE length error, including the analogous
eleven-versus-nine `ASTRWS01` summary header. It is a diagnostic reproducer,
not a passing numerical test. The shared writers await user approval; the
reader does not guess the shorter corrupt layout. Existing HDF5, CSV and
JPEG evidence is separate. No new high-wall consumer result is claimed.

### SBLI Physical-Midspan Wall Curves

Progress10.151 adds `prepare_sbli_wall_reference.py`,
`run_sbli_wall_components.py` and `run_cpp_sbli_wall_gate.py`.
The existing FP64 stationary Cartesian lower-wall product retains the
seven-node derivative, normalization and nonlinear evaluation on each
bracketing physical-z layer before interpolation. Independent manufactured
layers and checkpoint-based derivatives distinguish this from legacy
three/eight-layer wall sampling. Only a private wall strip and one-dimensional
curves are used; chart images use the existing host Matplotlib backend.

NP1 pinned and NP2 private-device ordinary runs check original CPU/old-GPU
fields, curves, observation isolation and exact same-build2+2 restoration;
NP2 also checks actual JPEG pixels and device-MPI transfer receipts. These
four-step fixtures are interface/numerical evidence, not developed SBLI.
The fixed reference reaches C_upper about0.500048, not a strict0.5 bound.
No FP32, blowing, wall means or general CURVE admission is added here.

Single-rank ordinary Memcheck and pinned component Memcheck/Initcheck pass.
The additional NP2 device ordinary Memcheck does not: CUDA API context/handle
errors begin inside UCX during MPI_Init and yield exit99 after advancement.
Keep its original logs; do not claim a clean multi-rank memory check, silently
suppress these errors or substitute a non-instrumented numerical run.

Progress10.152 adds `astr_backend_mpi_cuda_context`, a pure-MPI diagnostic
with no ASTR device library, allocation or kernel calls. Build it from the
root configuration, then run under the same MPI transport and sanitizer as
the failing case. It isolates MPI initialization/host-buffer memory-type
queries from the CFD/observer. A successful integer reduction is not a GPU
memory-safety result. A no-CUDA ob1 control may explicitly use the tool's
`--require-cuda-init no`; do not add that setting to actual CFD runs.

The local UCX control reproduces API errors with both default and explicit
API reporting; the installed2025.4 tool already defaults to explicit.
Runtime loader evidence, not standalone plugin ldd, identifies matching
HPC-X2.25.1 UCX1.20.0 libraries. Do not suppress or reinterpret all of the
full-main reports from this narrower probe. Separately, actual NP2 pinned
ordinary sampling, exact restoration and observation isolation pass, with
zero Memcheck errors on both ranks. Device-MPI functional evidence remains
separate from that memory check.

### Existing M12 Sparse Monitor

Progress10.153 adds `run_cpp_m12_probes.py` and `astr_backend_sparse_probe`.
Freeze the original CPU/old-GPU reference with `--freeze-only`, then supply
that immutable directory via `--reference` to the C++ gate. The fixture
materializes the original M12 UDF extrusion as a read grid and repeated initial
fields; it does not change the physical coordinates or edit the source case.
It uses64x32x24 intervals, NP2-x, original random blowing, joint compatibility
NSCBC,743e/643e and four dt=.02 steps. An initial fixture missing the UDF and
the first C++ outlet-dispatch rejection remain recorded as failed attempts.

The actual sparse product checks all72 normalized primitive probes per step
against old GPU and independently against current checkpoint q. It also checks
per-rank byte/phase receipts,15 state datasets under observation isolation and
exact2+2 continuation, restored controls/RNG, scheduling and resumed CSV words.
`--memcheck` instruments the actual full ordinary run. No full field is staged
for probes; formal checkpoint transfer is a separate explicit event.

The component checks generic reference scales, duplicate/absent indices,
invalid-index rejection, invalid density then successful capture, parent-word
isolation/lifetime and byte counters, in both synchronization modes. FP64
analytic comparisons retain2e-10; nativeFP32 reports raw D10 error instead.
ComponentFP32 evidence does not admit a full FP32 M12 random-blowing archive.
The C++ `m12_wall` provider is separately implemented/tested in Progress10.154
below. M12 production rendering evidence follows separately in Progress10.155;
the read-only preview entry remains incomplete, not implied by this72-point CSV.

### Existing M12 Spanwise-Averaged Wall Monitor

Use `run_cpp_m12_probes.py --wall` when freezing the original reference and
checking the combined C++ consumer. `--transport pinned|device` explicitly
selects the solver halo, not a wall-observer MPI backend. The wall observer
borrows completed q/halo and metric views and uploads static geometric weights
once. It downloads only Fortran(3,nx): span, Cf and normalized heat into the
solid. Existing CSV output retains unique global-x ownership and field names.

This is the original three-layer wall-normal closure plus the physical-end
2/4/6-order derivative of blowing velocity in x, followed by weighted spanwise
averaging of the nonlinear wall quantities. Do not replace it with physical-
midspan interpolation, a seven-node normal derivative or the other legacy
wall-time provider. The independent checker uses the actual110.3 K default,
not110.4 K. Keep the initial failed test-constant record separate from v2.

`astr_backend_profile_wall` checks analytic shear including blowing, heat
normalization/sign, nonuniform periodic quadrature, physical-end/internal
stencils, rejected weights/density, later successful capture, parent-bit
isolation/lifetime and allocation/transfer counts. FP64 NP1/NP2 components
and pinned ordinary NP2 Memcheck pass; device ordinary numerical/restart
evidence is separate from the unresolved UCX sanitizer-report issue. Native
FP32 components report raw D10 errors; no full FP32 M12 admission is inferred.
The combined four-step gate checks both products,15 state datasets, controls/
RNG, scheduling and resumed CSV words. It is not a developed physical run or
evidence for the M12 device-rendering/preview entry points by itself.

### M12/Boundary-Layer Runtime-Catalog Images

`run_cpp_m12_images.py --freeze-only` uses a current original CUDA Fortran
Catalyst binary, the immutable M12 numerical fixture and matching implementation
library to freeze actual GPU images, numerical slices and primary-field checks.
Then supply that directory with `--reference` for the new C++ gate. Both runs
use the original `m12_diagnostics` session envelope and explicit runtime JSONL,
not the retired static renderer. `--profile boundary_layer` checks that existing
standard-device/pinned envelope separately. Read-only preview is a separate
entry; the TGV increment below and the later physical M12 increment have
separate frozen sources and evidence.

The six existing example products include index/physical-plane images, Q=.001,
instantaneous layered64 lines, numerical fields/Q and a pressure probe. Only
short-test schedules and camera framing differ from the shipped example. The
original camera put the upstream Q surface outside the viewport; retained v2
is not the final framing reference. v3 keeps extraction bounds, threshold,
seeds and integration settings unchanged and aligns original/new views.

Check actual JPEGs/receipts, primary fields, numerical slices plus coordinates,
independent complete-step q-to-plane values, observation isolation and exact
same-build2+2 state/control/scheduling/data/pixel continuation. Cross-backend
pixel differences are reported, not declared word-exact. A laminar long-domain
line bundle at800x600 is not a developed-turbulence or production-quality claim.
`--pipeline`, `--solver-transport` and `--postprocess-transport` distinguish
rendering, core halo and private products. Zero image field/geometry readback
does not imply zero pinned-halo staging or zero explicit checkpoint/slice I/O.

Progress10.155 records three successful numerical/image matrices and the
separate historical text-contract result:16 passed,2 skipped,5 failed. The
five source-string assertions concern earlier crashfix/MP/restart/main-loop
shapes, not files changed by this consumer wiring; do not silently update them
or claim that this entire old suite passes. No new DCU/rendering performance
or joint Compute Sanitizer result is inferred from this local four-step gate.

### Read-Only TGV Preview With Saved Means

`run_cpp_preview_gate.py` consumes an existing full FP64 periodic32³ TGV
checkpoint; it never generates a new CFD source. Supply `--freeze-only` with
the original CUDA Fortran binary, then `--reference` with the C++ binary.
The separate output directory must be empty. Default source step is4;
`--step`, `--ranks`, and `--axis` identify another already-admitted source,
not permission to widen the preview capability. `--means` requests four
views: two instantaneous cameras and saved Reynolds/Favre velocities.

The CPP restore path skips solver/RNG/boundary/driver initialization. It
loads the actual saved state and, only if requested, statistics. Private
sampling completes duplicate endpoints by owner copy and exchanges halos;
mean sources are zero-based columns6--8/9--11 of the30-component state.
No statistics are accumulated. Camera-only changes reuse integration;
changing the velocity source performs a separate integration.

The BUILD_TESTING audit explicitly downloads q/cache/metric arrays and
the exact34-component statistics pack before/after preview. Require word
identity, zero advances, unchanged source-file contents/size/mtime, no new
checkpoints/flow fields/EPS, geometry-host bytes0 and actual JPEGs. These
test-only audit transfers are not a production zero-copy claim. NP1 saved
means and NP2-x instantaneous standard-device/pinned previews match their
original pixels and pass Memcheck; progress10.156 indexes the evidence.
Progress10.157 adds actual NP2 saved-mean pictures and physical M12 preview,
not native FP32 preview or M12 mean statistics.

The shared direct mapper rejects clipping planes, while the current preview
script adds six display clipping planes. The attempted direct preview is
therefore retained as a failure, not converted to standard-device success.
Both original/new standard previews also log the pre-existing empty
vtkVectorText diagnostic; do not describe all renderer diagnostics as clean.

Root target `astr_backend_owned_vector` independently checks periodic unique
owners and all±3 halo entries with different exact Reynolds/Favre analytic
columns. Run with topology and explicit private transport, e.g. `2,1,1 pinned`.
NP1 and NP2 x/y/z, both synchronization modes, parent/statistics word
isolation and device no-host-staging counters are checked. NP2-x pinned
Memcheck has zero errors/leaks; NP2-z pinned Initcheck has zero errors.
Use the matching MPI launcher and set UCX/device support for device-aware
mode, as in the existing sampling components; do not suppress UCX API
errors to declare a device-MPI sanitizer pass.

### Physical M12 Preview And Saved Coordinate Supply

`run_cpp_preview_gate.py --profile m12 --ranks 2 --step 10000` uses the
existing450x130x96 NP2-x source, not another simulation. The test explicitly
converts its saved positional input offline; the solver still accepts named
input only. No runtime parser compatibility or source mutation is introduced.
Reynolds/Favre M12 preview is not enabled. The original64/256-layered seeds,
RK45 parameters and800x600 standard-device/pinned framing are unchanged.

The new sampling entry uploads packed physical coordinates once, retains
that buffer and extends a separate private halo across internal MPI faces.
Periodic coordinates do not wrap. Each rank uploads68922768 bytes for this
source; setup H2D is separate from the existing MPI-staging counters. No
whole-field observation or post-extraction geometry download is added to
normal preview. The optional audit itself downloads fields; report it as
test-only observation rather than including it in a zero-transfer claim.

`astr_backend_insitu_vectors` now also compares this saved-coordinate upload
to the existing borrowed-device source, with two smooth physical mappings,
two synchronization modes, repeated captures and exact private halo/packed
coordinate words. NP1 and NP2 x/y/z pass; NP2-x pinned Memcheck reports zero
errors/leaks, NP2-z pinned Initcheck reports zero errors. These component
results are not a large-M12 joint memory-tool run.

Actual M12 preview matches both original JPEGs; all512 logged trajectory
records per rank also match exactly (`trajectory_comparison.json`). Rank
logs replicate global trajectories: do not sum them as independent particles.
Source content/mtime/size and restored fields are unchanged, the process RNG
audit has zero mismatches, and no CFD advances/new statistics/checkpoints are
performed. Existing empty vtkVectorText diagnostics occur in both backends.

For the qualified32-cubed historical TGV mean source, explicitly select
`--legacy-source --window .003 .004 --ranks 2 --means`. The offline converter
does not change its saved window. Actual NP2 pictures match original pixels,
statistics/RNG/fields are word-isolated, and both ranks pass Memcheck. The
earlier16-cubed rejected source remains a failed record, not a reference.

### Current DTK Build Evidence (Progress10.158)

The isolated `results/dcu_mb67_consumers_build_20261117/` record captures
1901 current source/build/example files, their identities, source archive,
remote scripts/logs, both CMake caches, all20 executable hashes and dynamic
dependencies. The remote directory is `~/weiph/mb67_consumers_build_20261117`.
Both ordinary FP64/FP32 programs use DTK26.04 GPUfusion, GNU12 and gfx906,
with CUDA Fortran/Catalyst/AIR5 disabled. FP64 sampling/vector/owned means/
index/moments/selected/wall/profile/SBLI/probe/MP targets and the extra FP32
MP/profile/probe targets have linked; all20 dependency lists resolve.

This is build/link/dependency evidence only. No GPU program ran on the login
node and no scheduler command was executed. Do not label the uploaded source
or the prepared4-case input package as a submitted or passed numerical run.
The proposed follow-up, including explicit task/query authorization scope,
is in `NEXT_VALIDATION_PROPOSAL.md` next to `build_receipt.json`.

The qualified private MPI is ROCm-aware through UCX but CUDA-disabled. Do not
reuse NVIDIA launchers' `opal_cuda_support=1` or `backend=cuda` assertions;
there is no automatic transport substitution. Existing SDK void-return,
forward-unknown and login rocm_smi warnings remain recorded, not suppressed.
The snapshot excludes `chem/`; it includes tracked legacy example mechanism
files, notwithstanding the initial manifest's overbroad scope wording.

### Approved Wall Session And Batch Follow-up (Progress10.159)

The approved fixed-size ASTRIW01/ASTRWS01 headers retain9/11 int32
members. Actual native FP64/FP32 CPU summaries have217892/113332 bytes,
finite statistics and unchanged per-precision reference state. Broken earlier
files are evidence only; no corrupt-layout reader was added.

`run_cpp_high_wall_consumers.py` freezes the corrected original backend and
compares16-cubed Cartesian Hermite7 channel walls with the new C++ backend.
NP1/NP2-y, four steps,2+2 exact continuation, observation isolation, twelve
instantaneous/mean wall CSV products and per-rank Memcheck were exercised.
This is not a production flow or an image test. The formal channel_walls
session extension retains the existing nonreacting provider constraints.

`run_mb67_consumer_batch.py` is a compute-node task body with no scheduler
calls. Its input manifest names four existing static-profile families; the
portable compiled comparator handles HDF5 rather than Python h5py. Tests
cover finite physical states, exact schedule payload, native precision and
the private ROCm-aware/CUDA-disabled MPI environment. Offline NVIDIA checks
validate the harness, not DTK device execution.

Approved DCU task128241288 failed before GPU execution because its launcher
requested validation_v2 instead of the archive's validation directory.
The path is corrected and all four inputs/reference paths were preflighted.
Another submission needs separate authorization. Frozen remote binaries do
not contain the later local Hermite7 formal-session admission change.

`run_cpp_high_wall_raw.py` exercises the separate ASTRIW01 test consumer
with the same original sixteen-cubed Hermite7 inputs. Explicitly set the
existing test transport to pinned or device-aware; no automatic substitution
is performed. NP1/NP2-y pinned and NP2-x device-aware compare four completed
steps, header/clock/coordinate/owner words, four wall fields and full-state
observation isolation. Pinned cases also have per-rank Memcheck evidence.
The packet downloads four derived columns, not forty conservative layers:
its field-readback bytes are one tenth of the old eight-node CPU-side oracle.
This does not measure full-step speedup or claim native-FP32 packet support.
Progress10.160 records executable identities, immutable inputs and receipts.

Progress10.161 adds raw packets for the existing32-cubed y-wavy curve wall
on NP2-z and16-cubed Hermite7 adiabatic bc42/421 representatives. Geometry,
owners and clocks match the original words; field comparisons and sampling
isolation pass. Only the curved pinned case additionally runs Memcheck.
These raw samples do not broaden the formal channel_walls configuration.

`run_cpp_h1_mean_consumers.py` prepares the existing H1/random-wall main
statistics lifecycle. The first original-CPU run is rejected before time
integration by the shared joint_mean output condition. Its failure remains
recorded; no original-GPU/new-C++ statistics comparison exists yet. Approval
to broaden H1 main-statistics/output admission is separate from already
validated selected-time statistics and is currently pending.

Progress10.162 records a separate current-wall GNU12/DTK build, not a
replacement for the reviewed compute batch. Both root-CMake FP64/FP32
ordinary programs link and resolve their dynamic dependencies. The CPU-only
`astr_wall_header_probe` writes exact9/11-int32 headers (36/44 bytes), with
all members decoded independently. This does not test GPU execution, native
FP32 floating packet payloads or DCU rendering. No scheduler operation or
login-node GPU run occurred; source identities and caches are retained in
`results/dcu_mb67_wall_build_20261117/`.

`run_cpp_adaptive_images.py` connects the existing32-cubed Re=1 output
scheduling fixture to dynamic Q/streamline/slice images. FP64 NP2-x,
standard-device/pinned uses the original thresholds and fixed dt=.001 for
twelve steps; the output cadence, not dt, is adaptive. Progress10.163 records
the actual frame inventories,26 rank receipts,5+7 exact continuation and
monitor/product state isolation. EPS stays off; private halo/control traffic
is separate from zero image-field/geometry readback. Old/new streamline
pixel differences are reported, not asserted zero. Analysis-only reuse is
explicit and does not rerun the completed CFD; window phase2 means finished,
not active, and a restored product with no due event needs no image directory.

Progress10.164 supersedes the pending H1 main-statistics and second-task
authorization notes above. The approved viscous FP64 H1 admission reuses
CPU44/GPU12 math and formats; CPU persists44 moments plus14 endpoints,
not44 total file fields. NP2-x four-step comparisons, original/new GPU
statistics, per-backend2+2 and RNG/control histories pass. Sampling preserves
physical fields and full conservative halos; legacy derived-cache halo
refresh is reported separately. C++ per-rank Memcheck has zero errors/leaks.

Authorized DCU task128249156 completes29 component invocations and18 main
runs across four input families in91 seconds (not a performance benchmark).
The frozen b7039c2a/26ed3718 candidates remain distinct from later H1/wall
source changes. Pinned/device, native precisions, MP stages, output isolation,
exact state/control/volume/slices and representative lean continuation have
recorded results. Main statistics are off in these four families; no H1
statistics or DCU rendering claim follows. Reports, logs and representative
HDF states are indexed in `validated_128249156.json`; the first failed job
record is preserved and no other tasks were queried or modified.

`run_cpp_h1_mixed.py` probes the existing16-cubed H1 viscous-storage path,
not a new precision algorithm. The FP64 control passes, but the old mixed
reference fails: its SP x/y physical-end divergence omits the H1 high-order
branch present in FP64, changing closure order rather than only storage.
Progress10.165 records the exact source pair, frozen programs and phase data.
Mixed continuation is stopped pending approval to correct the reference;
do not downgrade C++, loosen limits or call this combination qualified.

`prepare_sensor_replay.py` packages existing FP64 snapshots without advancing
ASTR. The ten-int32 sensor packet uses entries5:7/8:10 for global low/high
physical flags; the original all-zero header remains periodic. The probe
derives per-rank endpoints and periodic axes from these fields. Progress10.166
records bc50 xphysical frozen-input replay: raw values/masks match word for
word with pinned/device MPI, and pinned per-rank Memcheck is clean. A complete
periodic Shu--Osher packet also preserves its previous exact results.

This does not qualify the MP3 xphysical trajectory. Its FP64 fields/RHS pass,
but raw sensor values near zero boundary divergence differ after1e-15 gradient
residuals enter the existing1e-30 regularizer. The gradient replay reproduces
the observed difference; all six stage masks remain identical. A proposed
same-input/trajectory distinction awaits approval, independently of the H1
SP-closure correction. No sensor formula, threshold or production setting was
changed, and mixed xphysical advancement has not been run on C++.

`run_cpp_native_inflow.py` exercises the original native-FP32 CPU/CUDA
Fortran and new C++ inlet consumers on the existing16-cubed NP2-y profile
fixture. Four steps, checkpoint-off isolation and2+2 continuation preserve
the state/inlet/control/scalar/clock words within each backend. Fixed and
automatic stepping both roll the four-slot inlet window; the old/new GPU
metadata and26 inlet fields also match exactly. Stage fields/RHS and scalar
differences are D10 reports, not FP64 acceptance ceilings. Native authority
and inlet HDF payloads are checked as FP32. Fixed C++ runs include per-rank
Memcheck; automatic runs do not repeat it. Progress10.167 records the evidence.

Use a BUILD_TESTING=ON ordinary C++ program for stage comparisons. The first
selected rendering build had tests disabled: it advanced four steps and
wrote checkpoints but lacked test-only stage snapshots. That record remains;
the corrected run explicitly reused the completed original-backend results
and only executed the outstanding C++ cases. No production kernel, interval,
interpolation or boundary formula was changed. This is not native-FP32 random
wall output, inlet statistics/rendering, or DTK execution evidence.

`run_cpp_initial_families.py --stage reference` freezes original CPU/GPU
four-step inputs and defined stages for existing HIT, extruded vortex and
MP7 Sod initializers. `--stage candidate --reference DIR` exercises the
ordinary C++ program, keeps q/halo/clock words unchanged with observations
off, and optionally checks each rank with Memcheck. The16-cubed cases use
NP2-z/y/x respectively. HIT reuses the deterministic existing velocity-file
generator; this is not a turbulent DNS validation. The C++ change only adds
the missing three names to its existing input condition. Progress10.168
records scaled errors and actual memory-tool results.

Formal checkpoints/archives remain off: their shared original contract does
not yet register these cases, and the HIT velocity resource must be connected
before self-contained continuation. That extension awaits approval. The
separate original CPU `turbstats` report contains NaN Kolmogorov lengths;
inspection finds an unassigned div and uninitialized rhoe accumulation.
Neither this report nor its defective logic is used as a porting reference,
and it has not been repaired or copied. Standard flowstate and the actual
q/cache/RHS comparisons are independent evidence, not proof of that report.

Progress10.169 supersedes these pending decisions after user approval.
Original H1 stored-SP x/y rows now use the existing eight-node high-order
closure with FP32 storage retained. `run_cpp_h1_mixed.py` passes same-mode
stages, LP1 identity and2+2; the original corrected SP branches also have
two-rank Memcheck evidence. Other high-wall mixed combinations are not implied.

For the approved MP3 bc50 subset, explicitly select
`run_cpp_characteristic_gate.py --xphysical-sensor-trajectory`. Same-input
raw checks remain strict; trajectory raw values are reported and masks must
match exactly. FP64 field/RHS and mature mixed gates remain unchanged.
Actual mixed advancement, isolation and Memcheck pass. MP3 checkpoints were
not added. `run_cpp_initial_families.py --checkpoint` now checks original
CPU/GPU and C++4 versus2+2; HIT tails omit local velocity.h5 and read its
byte-identical frozen resource. CPU turbstats now assigns div and initializes
rhoE; finite output matches an independent saved-q reduction and exact
restart rows. Old invalid reports remain, not used as oracles. These local
FP64 cases do not qualify native-FP32, DCU, repartitioning or new rendering.

`run_mb67_closeout_batch.py` is a standard-library compute-allocation task
body, with no scheduler calls. The10 prepared16-cubed families cover only
new consumers; random H1 uses the same GNU executable's CPU field because
intrinsic RNG streams differ across compilers. DCU task128272750 completes
the sensor component and seven main families, then fails before the quad
quiet run: it attempted to edit an unselected one-line input.insitu. Quiet
now changes that file only for the explicitly active wall-CSV provider.
The exact dormant single-line case has a focused test; no compatibility
parser, retry or fallback was added. Three unfinished families need another
separately authorized allocation; completed evidence is retained.

Progress10.170 also records two actual DTK512-versus256 launch diagnostics.
The plane statistics and high-wall filter kernels now declare their existing
512-thread launches. Affected NVIDIA numerical/lifecycle checks pass; new
DTK build/runtime confirmation remains outstanding. Previous numerical and
exact-continuation results do not erase these diagnostics or qualify memory
tools. See `results/dcu_mb67_closeout_20261117/validated_128272750.json`.

Progress10.171 records the separately approved follow-up128277354: all three
remaining families (11 short runs) complete on four allocated DCUs with the
unchanged frozen binaries. Four-rank incoming-only/MP and two-rank adaptive
output match references within 1.333e-15 and restore all15 state datasets
exactly. The adaptive case also compares all eight actual datasets in each of
three frames and their exact resumed counterparts. MP3 x-physical final fields
match within 8.882e-16 absolute; integer masks match in12 packets, while the
0.0204674 raw-sensor trajectory difference is explicitly reported, not called
same-state equivalence. This does not test the separate launch-declaration
candidate, extend MP3 restart, render on DCU, or measure performance. See
`results/dcu_mb67_closeout_20261117/validated_128277354.json`.

Progress10.172 separately closes the two512-thread launch declarations on DCU:
approved task128277778 runs only H1 main statistics and Hermite7 wall consumers
(nine short runs including the same-GNU CPU reference), with the new FP64 binary.
No former launch-parameter/bounds messages remain in those run logs. Field,
common12 time-mean and actual wallCSV errors are at most7.328e-15,7.078e-16,
and5.006e-15; observation isolation and exact2+2 continuation pass. The distinct
FP32 build is not new FP32 execution evidence. No DCU rendering or performance
claim follows. See `results/dcu_mb67_launch_bounds_20261117/validated_128277778.json`.

## MB8 AIR5 component progression

The new FP64 AIR5 device path is under construction; it is not yet an ordinary
ASTR reacting-flow runtime. Progress10.177--10.184 separates property/source,
ROS-2/G1, chemical half-step state, frozen-acoustic CFL/RK, and spatial flux
components from full CFD integration, G2, MPI, output/restart and DCU delivery.
The approved fixed-T/Tv V-T operator comparison does not erase the retained
near-equilibrium full-q instantaneous source differences.

The following local targets build through the root CMake project:

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_transport astr_backend_air5_spatial -j 8
```

`run_air5_transport_components.py` freezes original CPU and old-GPU scalar
arithmetic for the 11-component Jacobian-weighted SSP-RK3 and CFL checks.
`run_air5_spatial_components.py` uses analytic manufactured gradients and the
frozen property executables as local diffusion-flux oracles:

```bash
python tests/gpu_backend/run_air5_spatial_components.py \
  --output results/air5_spatial_new_run \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_spatial \
  --reference results/multi_backend_mb8_properties_20261117_v3 \
  --memcheck
```

Each output directory must be new. The recorded spatial test covers eight
504-node configurations, physical-end versus interface stencils, layered
energy-channel allocation, field isolation, failure commits, and periodic
16/32/64 derivative convergence. It does not execute flux divergence, positivity
limiting, MPI halo exchange or a complete reacting timestep. AIR5 remains FP64;
building the shared library with native FP32 does not admit FP32 chemistry.

`astr_backend_air5_flux_math` covers bounded species mass/moment projection,
the original energy admissibility and roundoff budgets, and symmetric face
trials. `astr_backend_air5_faces` exercises actual context-owned 11-component
centered/MP7/LLF and diffusion face candidates. These targets do not yet assemble
the limited field RHS or exchange AIR5 private faces through MPI.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_flux_math astr_backend_air5_faces -j 8
python tests/gpu_backend/run_air5_flux_components.py \
  --output results/air5_flux_new_run \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_flux_math \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --properties-reference results/multi_backend_mb8_properties_20261117_v3 \
  --memcheck
python tests/gpu_backend/run_air5_faces_components.py \
  --output results/air5_faces_new_run \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_faces \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --properties-reference results/multi_backend_mb8_properties_20261117_v3 \
  --memcheck
```

GPU reference wrappers are copied byte-for-byte to `.cuf`, matching their
original device source mode; CPU wrappers retain `.F90`. On this NVHPC26.1
installation, the same device wrapper named `.F90` plus `-cuda` takes an extra
`fort2ex` pass: it misgenerated the admissibility reference and ICEd on a module
array spectral-radius expression. `.cuf` removes both problems without changing
the original functions, O2, or tolerances. The failed wrappers and diagnostic
predicate-splitting attempts remain evidence, not production corrections.
Progress10.183 supersedes the former request to approve predicate splitting;
that splitting is not adopted. Representative passing artifacts are
`results/multi_backend_mb8_flux_constraints_20261118_cuf_v2/` and
`results/multi_backend_mb8_faces_20261118_v6/`. The latter also tests48 inviscid,
sensor-off cases without allocating gradient or viscous workspaces; it does not
claim inviscid shock-sensor gradient supply is integrated.

The minimal source-mode reproduction uses frozen original functions, not the
earlier discarded predicate rewrites:

```bash
python tests/gpu_backend/probe_air5_admissibility_codegen.py \
  --reference results/multi_backend_mb8_flux_constraints_20261117_v3 \
  --output results/air5_source_mode_new_run \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran
```

## AIR5 Owned-Face MPI Component

`astr_device_air5_symmetric_exchange(handle,axis)` uses axis1..3 and the
context's existing topology/backend. It exchanges the two owned12-component
face planes, not an hm-wide field, and never averages endpoints. The ordinary
11-component low-order budget uses the existing halo-only transport separately.
The MPI-thread call is serialized with ordinary halo phases; device mode uses
the existing device slots, pinned mode allocates four reusable host slots once.
No new runtime option or communication rank is introduced.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_face_exchange -j 4
python tests/gpu_backend/run_air5_face_exchange.py \
  --binary build_backend_cpp_native/bin/astr_backend_air5_face_exchange \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun \
  --output results/air5_owned_faces_new_run \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer
```

The runner tests NP1 and NP2 x/y/z, periodic and physical ends, both transports,
both synchronization modes and two repeated plane stages. It compares every
face/budget word to independent rank-encoded ownership, checks unchanged q,
fixed slot pointers, actual PMPI datatype/tags/pointer kinds and application
copy bytes. `--memcheck` deliberately tests pinned MPI under ob1 with CUDA-aware
disabled. It does not silently rerun a failed device test as pinned: the earlier
raw device Memcheck is retained in
`results/multi_backend_mb8_face_exchange_20261118/memcheck.log` and is **not
clean** (UCX Fabric/VMM API probes and IPC mapping leak records). No reports are
suppressed. Exact-value communication evidence is
`results/multi_backend_mb8_face_exchange_20261118_v2/report.json`.

This gate is independent of the shared limiter's status201 issue in progress
10.185. Ordinary AIR5 integration, private viscous27/33-channel exchange,
filtering, complete G2/restart and DCU AIR5 are still pending; do not label this
synthetic communication check a reacting-flow or physics validation.

## AIR5 Viscous-Flux Halo Component

`astr_device_air5_flux_transport_prepare` accepts the existing separate,
packed and packed-overlap choices (0/1/2); it follows the prepared diffusive
fields'27 or33 channels and is independent of the existing pinned/device MPI
backend. Separate reuses ordinary six-component halo slots; packed modes own
fixed per-distributed-axis slots. Packed-overlap waits on an explicit field-ready
event and prepackages the next distributed axis while MPI is posted, with main
thread MPI progress and explicit pack completion. It does not create extra
communication ranks, change formulas, average nodes or touch outer corners.
The MPI completion metric includes the prepack/progress interval in this mode;
it is not an isolated network latency or a performance result.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_flux_exchange -j 4
python tests/gpu_backend/run_air5_flux_exchange.py \
  --binary build_backend_cpp_native/bin/astr_backend_air5_flux_exchange \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun \
  --devices 2 --output results/air5_viscous_halo_new_run \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer
```

The component matrix uses NP1, NP2 z and NP4(2x2x1), periodic/physical ends,
both MPI backends, all three packet modes, and optional carried-energy storage.
Debug synchronization alternates by the recorded deterministic rule rather than
exhaustively multiplying the matrix. NP4 explicitly shares two local GPUs;
this is not a scalability benchmark. PMPI checks actual types/pointers/counts,
every outgoing packet word and the original Fortran channel/coordinate order.
This **test-only** packet readback must not be described as production data
transfer. Full field words, NaN-payload corners, shared/physical nodes and q are
checked separately. Two stages use different data to detect stale-slot reuse.
The optional Memcheck is pinned/ob1 only, covering four ranks and the actual
two-direction prepack path without suppressing device-MPI tool reports.

Evidence: `results/multi_backend_mb8_flux_halo_20261118_v2/report.json`,
per-topology logs, `memcheck.log` and `fortran_abi.log`. These APIs are still
unconnected to ordinary AIR5 scheduling. The shared projection issue in
progress10.185 remains pending; no complete coupled-flow or physics gate is
claimed by this synthetic packet/field test.

## AIR5 Conservative-Filter Math Component

`chemistry_filter_math.cuh` ports the existing strict energy/pressure expansion
predicates, represented high-minus-carry pressure, finite-representation mass,
face bounds, commit boxes, thermal limiter and periodic/physical high-face
formulas. It uses the existing single-mass projection, not the pending shared
mass/moment projection in progress10.185. Contraction/flushing must remain
disabled for these IEEE transforms. Original CPU/GPU math-module flags and
Fortran source modes are retained; only the test launcher is copied to `.cuf`.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_filter_math -j 4
python tests/gpu_backend/run_air5_filter_components.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_filter_math \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --output results/air5_filter_math_new_run \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer
```

This reuses existing T3 fixtures and the original absolute2e-10 component budget.
Predicate/status checks are exact; independent rational energy/pressure signs,
mass intervals, the periodic tenth-binomial stencil and physical `-D5^T D5`
matrix provide checks beyond implementation agreement. The thermal tests use
common frozen CPU mass-closed candidates. They are not a full native field
pipeline. Original near-cone zero-flux freezes skip actual update arithmetic;
do not require those unchanged states to pass a box for a hypothetical commit.
They must still be legal and their exchange exactly zero.

Evidence is `results/multi_backend_mb8_filter_math_20261118_v2/`.
`--memcheck` covers the thermal rows; the additional `basic_memcheck.log`
covers the rational/represented-pressure expansion rows with the same candidate
and `basic.bin`. Both are clean, including the128-slot represented expansion.
No scalar/full field commit, filter-strip MPI, shared-alias propagation, complete
CFD, DCU or physical gate is claimed. The original uniform Cartesian
643e/643e conservative-filter geometry restriction is not removed by this math
component.

## AIR5 Resident Conservative-Filter Candidate

`chemistry_filter.cu` and the transport filter APIs implement the original
strip/frozen-face/commit/unique-alias/model-check sequence on q/carry, with1/11
component workspaces. MPI-thread orchestration uses the prepared pinned/device
backend and globally releases physics status before committing an axis. No
physical-step rollback is introduced. Primitives are not refreshed by this API;
the subsequent ordinary spatial phase must refresh them from the filtered q.
The original uniform Cartesian643e/643e precondition remains the caller's
geometry contract; the candidate is not admitted to ordinary AIR5 yet.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_filter_field -j 4
python tests/gpu_backend/run_air5_filter_fields.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_filter_field \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpif90 \
  --mpiexec /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun \
  --output results/air5_filter_fields_new_run \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer
```

This command currently **stops at the original CPU repeated-vibration failure**;
do not treat its nonzero exit as a completed matrix. In the retained evidence
`results/multi_backend_mb8_filter_fields_20261118/`, smooth and trace-step pass
before that failure. Independent pressure-floor/carry-only and memory checks
are explicitly retained under `independent_after_vibration/`, not substituted
for the failed case. The four passing fixtures total26 new field runs, NP1/2
x/y/z and representative NP4(2x2x1) on two explicitly shared test GPUs. They
include both transports and word-equal scalar/full and shared aliases.

The original CPU/GPU and new C++ all reproduce physics status3 on vibration.
Exact high-minus-carry energy margins first become negative after the fourth
directional commit, before the next direction rejects the base. Existing commit
boxes protect pressure, not the two lower-energy inequalities. The minimal
numerical revision is awaiting approval; no clipping, relaxed tolerance or
fallback is applied. Full ordinary reacting flow, physics, performance, restart
and DCU are not claimed. See progress10.189 for exact snapshots, margins and
separate precommit-release/Fortran ABI evidence.

## AIR5 Legacy Trace Diagnosis

`probe_air5_legacy_trace.py` freezes the original GPU tenth-filter scalar and
legacy repair kernel plus the current CPU species-repair function. It creates
the same positive dyadic candidate for both repairs; it is not a full legacy
filter or CFD run. Before NR3, under theta1/limited=false, CPU reconstructed
`low+(filtered-low)` and erases two normal-FP64 positive traces, while GPU
copies them unchanged. The approved NR3 CPU copy repair is checked with
`--expect-repaired`; the C++ legacy path is not covered by this probe.

```bash
python tests/gpu_backend/probe_air5_legacy_trace.py \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --output results/air5_legacy_trace_new_run \
  --expect-repaired
```

Evidence: `results/multi_backend_mb8_legacy_trace_diagnosis_20261118/`.
The original two positive inputs8.470329472543003e-22 and
7.703719777548943e-34 become CPU zero, GPU unchanged, both status0.
This was cancellation, not FTZ/underflow. Progress10.203 and
`results/multi_backend_mb8_nr3_20261118/` record the fixed three trace cases
and unchanged limited case. No tolerance was relaxed.

## AIR5 Wall And Pressure-State Boundary Components

`chemistry_boundary.cu/.F90/.h` provides lower-y isothermal wall and upper-x
pressure-state outlet kernels using the original point equations in
`chemistry_boundary_math.cuh`. They write q on successful sites only; the caller
owns physical-face selection, global status release, final carry reset and
halo/primitive rebuild. They are not a full boundary scheduler or NSCBC RHS,
and are not admitted to ordinary AIR5 yet.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_boundary -j 4
python tests/gpu_backend/run_air5_boundary_components.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_boundary \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --properties-reference results/multi_backend_mb8_properties_20261117_v3 \
  --output results/air5_boundary_states_new_run \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer
```

Evidence: `results/multi_backend_mb8_boundary_states_20261118/` contains the
first1104 input comparison and raw zero-error/leak Memcheck including real
ghost writes. `mixed_composition_report.json` adds two unexecuted inputs using
the same frozen binaries; the current script includes them by default(1106).
The maximum original-property budget ratios are0.001142/0.000276 versus CPU/
old GPU. Independent wall T/Ev, outlet pressure and288 word-exact supersonic
copies are checked. Failed field sites/non-target words and all carry remain
unchanged; no implicit carry reset is hidden in a single boundary kernel.
`fortran_abi_v2.log` records actual Fortran calls; the first failed ABI log is
retained as an incorrect test halo index, not a numerical formula correction.
Full boundary ownership/lifecycle, characteristic RHS, MPI, coupled flow,
restart, physics and DCU remain pending.

## AIR5 Batched Primitive Cache Refresh

`astr_device_air5_primitives_regions` accepts inclusive bounds in Fortran
`(3,nregions)` layout, clears the physics status once and preserves failures
across all regions. It launches the existing conversion kernel without changing
q, carry or chemistry history. Failed nodes do not write caches; successful
nodes may already have written them when another node fails. This is not an
atomic cache transaction. The caller must check/release the phase before using
its caches in subsequent flux work. Exterior multi-axis corners remain skipped.
The existing single-region interface and mathematics are unchanged.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_primitives_regions astr_backend_air5_compensation_exchange astr_backend_air5_convection_abi -j 4
python tests/gpu_backend/run_air5_primitives_regions.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_primitives_regions \
  --input results/multi_backend_mb8_compensation_exchange_20261118_v2/input.bin \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer \
  --output results/air5_primitives_regions_new_run
```

The retained local evidence is
`results/multi_backend_mb8_primitives_regions_20261118/`: four debug/preserve
contexts, five regions, sixteen injected-failure checks, word-exact cache
comparison against separately checked calls, immutable q/carry/history,
zero-error/leak Memcheck, and actual two-region Fortran binding calls.
`run_air5_compensation_exchange.py` now uses this API for the final physical
and real halo cache rebuild. Its repeated NP1/2 x/y/z/4 matrix with both
transports is retained in
`results/multi_backend_mb8_compensation_regions_20261118/`:88 rank contexts,
880 phase observations,112 send signatures,64 live payload checks and maximum
property budget ratio0.00033437305212240026. Pinned two-rank Memcheck is clean;
the earlier device-MPI tool reports remain separate. Neither component gate
admits ordinary reacting flow, fixes the old periodic fast controller, or
closes G2, statistics/restart or DCU AIR5.

## AIR5 Chemistry Region Phase And Global Reports

`chemistry_device_coupling::air5_chemistry_apply_regions` performs one begin,
the solver-owned regions, and the existing global failure/report reductions.
It keeps separate/packed reductions and quiet sampling distinct. A returned
physics failure forbids the caller's subsequent boundary/halo/RK work; API/MPI
failures abort the communicator. Reports are not valid statistics unless their
`collected` flag is true. The routine does not allocate device storage, repair
failed states, exchange solution halos or execute a complete Strang/RK step.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_chemistry_phase -j 4
python tests/gpu_backend/run_air5_chemistry_phases.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_chemistry_phase \
  --input results/multi_backend_mb8_compensation_exchange_20261118_v2/input.bin \
  --mpirun /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer \
  --output results/air5_chemistry_phases_new_run
```

Evidence `results/multi_backend_mb8_chemistry_phases_20261118/` includes99
rank-cases and22 injected-failure observations on NP1/2 x/y/z/4. Each compares
the device fields word-for-word against direct begin/region/report calls in an
independent context; MPI_Allgather plus host extrema/sums supplies the global
reference. Packed and quiet variants also preserve the same initial-state
trajectory word-for-word. Both ranks pass Memcheck across all nine cases.
This host-metadata MPI test does not claim device-buffer MPI, complete G2,
ordinary AIR5 admission, physics, restart, DCU or performance results.

## AIR5 Main Time-Statistics Device Component

`chemistry_statistics.cu/.F90/.h` retains the original main-time-statistics
SUM44 and ENDPOINT14, not the separate in-situ central-moment groups. Prepare
explicitly allocates the two physical-node buffers and one private failure word;
disabled statistics allocate neither. Existing views/uploads/downloads expose
the same44+14 raw channels without another packing buffer. The caller supplies
current primitives/physical velocity gradients and owns the sample clock.
Failed nodes do not write either buffer; other nodes may have committed.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_mean -j 4
python tests/gpu_backend/run_air5_mean_components.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_mean \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --properties-reference results/multi_backend_mb8_properties_20261117_v3 \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer \
  --output results/air5_mean_new_run
```

`results/multi_backend_mb8_mean_components_20261118_v4/` freezes original CPU
transport/moment functions and the original GPU sampling kernel. CPU/GPU
reference agreement is checked before the new implementation under the original
`test_meanflow_air5_transport` per-field1e-14 spatial-scale budget. Six valid
unequal-time samples and two injected failures cover state codes, non-commit
sites, optional allocation, exact reset and ten unchanged solver field groups.
Four explicit/dependency-sync and continuous/raw-array-restore outputs are
word-identical; Memcheck reports zero errors/leaks. Worst new/CPU and new/oldGPU
budget ratios are0.005551115123125783 and0.003469446951953614.
This is raw58-channel restoration only, not sample-clock, checkpoint-file,
MPI/repartition, ordinary AIR5, in-situ Tv/species moments, G2, physics or DCU
closure. The new field IDs37/38 append to the context; existing IDs are stable.

## AIR5 Main Statistics Clock And Current HDF5 Files

`chemistry_statistics_runtime.F90` contains the ordinary-runtime module
`chemistry_mean_statistics_cpp`. It keeps the original sample identity,
duplicate-step suppression, explicit sampling/window controls and time coverage.
It writes/reads the existing role6 SUM44+ENDPOINT14 file and9-word metadata via
`checkpoint_state_io`; no new format or deprecated checkpoint reader is added.
Sampling assumes current primitives/physical velocity gradients. Release retires
only its optional statistic arrays/status after consumers have finished.

```bash
cmake --build build_backend_cpp_native --target astr_backend_air5_mean_runtime -j 4
python tests/gpu_backend/run_air5_mean_runtime.py \
  --candidate build_backend_cpp_native/bin/astr_backend_air5_mean_runtime \
  --compiler /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpif90 \
  --hdf5-root /opt/hdf5-1.14.6 \
  --properties-reference results/multi_backend_mb8_properties_20261117_v3 \
  --mpirun /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun \
  --memcheck /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer \
  --output results/air5_mean_runtime_new_run
```

Evidence `results/multi_backend_mb8_mean_runtime_20261118/` freezes the complete
original GPU mean-statistics module and exercises actual HDF5 clock restoration
on NP1/2 x/y/z/4. It covers empty/seed/end/reset states, disabled and duplicate
samples, context recreation, same-backend word-equal files, and inheritance from
the current original-GPU statistics file under the original1e-14 field budget.
There are130 functional HDF5 file-pair comparisons; maximum budget ratio is
0.008326672684688674. NP2-x Memcheck has zero errors/leaks and its seven files
are also word-equal to the continuous result. All MPI here is host-side
metadata/HDF5, not device-buffer transport.
The ordinary C++ AIR5 run loop is tested separately by
`run_cpp_air5_output.py --mean-statistics`. That test covers complete-step
sampling, scheduled output, same-topology checkpoint bundles and observation
isolation for the specified-top HBL case (see migration progress 10.219).
The component test does not cover in-situ mean consumers, CFD/G2, physics or DCU.

## AIR5 Captured Projection Geometry Diagnosis

```bash
python tests/gpu_backend/diagnose_air5_projection_geometry.py \
  --capture results/multi_backend_mb8_convection_fields_20261118_v4/point_trace \
  --output results/air5_projection_geometry_new_run
```

This CPU-only diagnostic treats each captured FP64 word as an exact rational.
It compares greedy moment extrema on a fixed-mass box section against80
independent single-free-variable vertices. The retained result is
`results/multi_backend_mb8_projection_geometry_20261118_v4/report.json`.
Both captured strict-equality problems are infeasible, but physical-box
extremal witnesses rounded to FP64 satisfy the original final return budgets;
their moment residual ratios are0.6472324727 and0.6493969230. The original
intermediate increment budget puts the two inputs on opposite sides of its
threshold. This does not implement a new limiter, change any producer, repair
status201, or validate the witness's coupled energy/RK update. Proposed
revisions remain subject to `ASTR_MB8_NUMERICAL_REPAIR_PROPOSAL.md` approval.
