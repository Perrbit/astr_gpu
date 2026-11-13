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

The implemented flow path is FP64, Cartesian periodic five-variable TGV,
standard 643e convection/nodal viscosity, explicit10 and RK3.
ASTR_GPU_FILTER_WORKSPACE selects scalar (default, one 3-D work field) or full
(five work components). Scalar/full keep the same per-component x/y/z formula
and terminal physical bits in the native eight-case matrix. The allocated
workspace is reported, and scalar uses exactly one fifth of the full bytes.
ASTR_GPU_SYNC_MODE supports dependency and explicit. CPU use_gpu=f still takes
the original Fortran path. C++ crashfix, mean fields and device in-situ providers
are not connected. The initial MB4 output/restart bridge is described below;
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
