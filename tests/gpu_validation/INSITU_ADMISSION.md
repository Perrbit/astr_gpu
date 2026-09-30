# Catalyst Admission Checks

This is an optional dependency check, not an in-situ flow sampling or rendering
interface. `astr run` does not invoke Catalyst. No input flow settings, RK stages,
boundary conditions, numerical kernels, or checkpoint formats are changed.

## Build

Use the root CMake project. `ASTR_WITH_CATALYST=OFF` is the default and does not
discover Catalyst or enable C++ for this feature. Runtime `usegpu` is unchanged.

For the optional backend check, provide a Catalyst 2.1 package and a ParaView
Catalyst implementation built with the same MPI as ASTR. Ensure that the HDF5
wrapper can find its compiler through `PATH` before configuring.

```bash
cmake -S . -B build_insitu \
  -DASTR_WITH_CATALYST=ON -DBUILD_TESTING=OFF \
  -Dcatalyst_DIR="$CATALYST_PREFIX/lib/cmake/catalyst-2.1" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DHDF5_Fortran_COMPILER_EXECUTABLE="$HDF5_PREFIX/bin/h5hlfc"
cmake --build build_insitu --target astr -j 8
```

Add `-DASTR_WITH_CUDA=ON` to build the CUDA-capable binary. Use a C++ runtime
compatible with the Fortran link driver, not merely any available C++ compiler.
The locally tested combination is NVHPC 26.1, GCC/G++ 13.4, HPC-X 2.25.1,
HDF5 1.14.6, Catalyst 2.1.0, and ParaView 6.1.1. This is not an automatic
compatibility guarantee for other installations.

## Configuration And Invocation

Use a separate file, for example `insitu.nml`:

```fortran
&insitu
  implementation_path='/absolute/path/to/paraview/lib/catalyst'
/
```

`implementation_path` is a directory containing `libcatalyst-paraview.so`, not
the Catalyst API library directory. Every rank reads the configuration and must
agree on the filename and parsed value. LF and CRLF are accepted. Records of
4096 characters or more are rejected. Configuration filenames and backend paths
should not exceed 1024 characters. This admission interface currently targets
Linux shared libraries.

```bash
mpirun -np 2 build_insitu/bin/astr insitu-check insitu.nml
```

Unset `CATALYST_IMPLEMENTATION_PREFER_ENV`: overriding the selected backend with
environment variables is rejected. Missing files, parse errors, inconsistent
rank configurations, failed dynamic loading, and backend lifecycle failures
abort collectively. A build with the option disabled rejects the command.

The command checks initialization, implementation identity, and finalization.
It does not call `catalyst_execute`, create a mesh, select a render device, or
prove EGL rendering works. The separate analytic rendering probe described in
the project plan covers that dependency test; it is not an ASTR field test.

## Regression Tests

```bash
export ASTR_INSITU_TEST_BINARY="$PWD/build_insitu/bin/astr"
export ASTR_INSITU_TEST_DISABLED_BINARY="$PWD/build_without_catalyst/bin/astr"
export ASTR_INSITU_TEST_BACKEND="$PARAVIEW_PREFIX/lib/catalyst"
export ASTR_INSITU_TEST_MPI="mpirun"
python -m pytest -q tests/gpu_validation/test_insitu_admission.py
```

The MPI launcher and optional flags can be specified in `ASTR_INSITU_TEST_MPI`.
Local tests used `--mca coll_hcoll_enable 0` because the workstation has no HCA;
this is not a required production MPI setting. Tests have a 60-second timeout.
An optional `insitu_runtime_probe` target (`BUILD_TESTING=ON`) exercises the
same production module without building the fluid solver.

## Physical Device Identity Probe

`insitu_device_identity.py` matches visible CUDA UUIDs to CUDA-capable EGL
devices. It rejects absent or ambiguous matches. Its launcher is only for local
Open MPI dependency probes, not a replacement for ASTR device binding. A future
solver adapter must use the device already bound by ASTR. No MIG, multi-node,
or non-NVIDIA support is claimed by these tests.

The external dependency probe and Python pipeline are in `$DEPS/probes`; their
rendering callback queries the actual current EGL display device UUID. The local
matrix tests native visibility, reversed visibility on two ranks, and visibility
restricted to the second physical GPU. It requires two local CUDA GPUs.

```bash
python -m pytest -q tests/gpu_validation/test_insitu_device_identity.py
python tests/gpu_validation/run_insitu_identity_probe.py \
  --probe "$DEPS/build/backend-probe/catalyst_backend_probe" \
  --pipeline "$DEPS/probes/backend_pipeline.py" \
  --backend "$PARAVIEW_PREFIX/lib/catalyst" \
  --mpiexec "$MPIEXEC" \
  --output tests/gpu_validation/out/insitu_identity_new
```

The output directory must be new. Each invocation is bounded to 120 seconds,
16^3 analytic nodes and one 640 x 480 image. The runner stops on failure and
records selected/actual UUID, PCI address, cell counts and RSS at the rendering
callback. JPEG and raster-embedded EPS are generated. RSS is the whole-process
historical peak at that callback, not measured additional solver memory or VRAM.

## Explicit Software Rendering

The same external probe accepts `ASTR_PROBE_RENDER_BACKEND=osmesa`. The runner
selects this explicitly and verifies the actual OSMesa window, llvmpipe renderer,
and loaded library directory. It never silently substitutes software for EGL.
The tested user-local runtime consists of Ubuntu `libosmesa6` 25.1.7-1ubuntu3 and
`libllvm20` 1:20.1.8-2ubuntu8, extracted without installation or system changes.

```bash
python tests/gpu_validation/run_insitu_identity_probe.py \
  --probe "$DEPS/build/backend-probe/catalyst_backend_probe" \
  --pipeline "$DEPS/probes/backend_pipeline.py" \
  --backend "$PARAVIEW_PREFIX/lib/catalyst" \
  --mpiexec "$MPIEXEC" \
  --render-backend osmesa \
  --software-library-dir "$DEPS/install/osmesa-25.1.7/usr/lib/x86_64-linux-gnu" \
  --output tests/gpu_validation/out/insitu_osmesa_new
```

This matrix uses NP=1/2, hides CUDA devices, unsets `DISPLAY`, and limits llvmpipe
to two threads per rank. The dependency probe's resource snapshots require
`pynvml` and the local NVIDIA driver even for this software-rendering test; this
does not impose NVML on ASTR production builds. CPU-only-host deployment remains
untested. Runtime dependencies are checked before launching MPI.

Per-process NVML snapshots before rendering and after saving the image exclude
other applications and deduplicate compute/graphics records. These are not
continuous VRAM peaks. The RSS sample is likewise the process historical peak
at the callback, not a finalize-inclusive or solver-relative budget.

Remaining gates: integration with ASTR's bound CUDA device,
same-completed-step read-only field sampling, output
schedules, resource limits, derived fields, and distributed visualization
validation. The identity probes do not change `astr insitu-check` into a
rendering command or certify a production memory budget.

## Bounded Completed-Step Sampling

With `BUILD_TESTING=ON`, `ASTR_INSITU_SAMPLE_PREFIX` enables a separate validation
snapshot after the completed RK update. This is independent of Catalyst and is
not a production scheduling option. It supports five-variable TGV only, with
local extents no greater than 32, periodic boundaries and the 643e stencil.
GPU data is downloaded into private buffers, not solver host mirrors.
Host-side sixth-order gradients, Q_rs, divergence and curl use a private velocity
halo; this is not GPU-accelerated diagnostics or rendering.

```bash
python tests/gpu_validation/run_insitu_sample_validation.py \
  --cpu build_insitu_check/bin/astr --gpu build_insitu_gpu/bin/astr \
  --mpiexec "$MPIEXEC" \
  --output tests/gpu_validation/out/insitu_sample_new
```

Both executables must be built with `BUILD_TESTING=ON`. The runner prepares 32^3
TGV cases, compares NP=1/2 CPU/GPU samples at absolute tolerance 2e-10, checks
primitive/conservative algebra and analytic coordinates, and requires existing
field output to remain bitwise unchanged with sampling enabled. `maxstep=2`
executes three updates under the existing inclusive loop semantics. Each sample
is labelled with the completed update number and its actual end time. Shared
interface nodes remain in raw snapshots (`ASTRIS01`). A separate diagnostic copy
(`ASTRIC01`) uses half-open node ownership, copying the positive neighbor's low
face onto the high face, with periodic wrapping. Five conservative components
are exchanged on a private communicator in x/y/z order, then primitives are
rebuilt. No averaging or solver-state modification is allowed. The derivative
file uses `ASTRID01`; raw and canonical readers reject each other's signatures.
Private velocity halo exchange uses existing `dataswap`, which advances its MPI
tag counter but does not overwrite solver fields or gradient caches.

The runner checks canonical conservative values against unique owned raw values
bitwise, all 14 derived quantities against an independent global stencil, and a
nonzero-divergence manufactured field against its analytic discrete response.
`--topology 1,2,1` or `--topology 1,1,2` selects another NP=2 slab;
`--skip-single` avoids repeating NP=1. Use a fresh output directory for each run.
NP=1 and all three NP=2 slab directions passed on 2026-09-29. Maximum CPU/GPU
absolute differences were 2.274e-13 for sampled fields and 1.387e-14 for derived
fields; the 2e-10 acceptance threshold was unchanged. Three unit tests cover
format separation and Q_rs rather than the second invariant at nonzero divergence.
Multi-axis decompositions, physical boundaries, CURVE, AIR5, production statistics
ownership and Catalyst ghost metadata remain outside this bounded gate.
The production build
(`BUILD_TESTING=OFF`) contains neither sampling module nor its main-loop call.

## Real TGV Bridge Lifecycle

Add `--render-backend-dir "$PARAVIEW_PREFIX/lib/catalyst"` to the sampling runner
to render three completed steps of GPU sampling-on cases. The GPU executable
must also enable `ASTR_WITH_CATALYST`. The runner creates the output directory
before UUID selection and uses the existing EGL identity launcher. CPU and GPU
sampling-off controls remain unchanged. No production task is touched.

The test-only C++ adapter receives private Fortran arrays directly, using explicit
coordinates and structured, nonoverlapping cells. Shared interface nodes remain
necessary for connectivity; there is no node-statistics ghost contract yet.
The Python pipeline checks the actual VTK coordinates and 25 point fields exactly
against canonical/derived files written at the same phase. Those files are only
an independent validation oracle, not the Catalyst input transport.

The one-frame presets are Q_rs=0.25 and z=pi/4, colored by u over [-1,1].
NP=1 and NP=2 x-slab passed: 32768 total mesh cells, 8768 Q triangles and 2048
slice triangles; original solver field outputs remained bitwise unchanged.
Artifacts: `out/insitu_mesh_v2_20260929/np{1,2}_gpu_on/outdat/`, containing
`q_surface.jpeg/.eps`, `velocity_slice.jpeg/.eps` and per-rank mesh/UUID records.
EPS embeds the JPEG raster; it is not vector geometry. Images were inspected.

That directory records the earlier single-frame gate. The current adapter keeps
one backend and private communicator across frames, copies input into its own
fixed-size persistent buffers, and finalizes at normal `steploop` exit. It rejects
shape changes. No caller-owned temporary array is retained across callbacks.
This uses synchronous host-side diagnostics plus GPU rendering, not zero-copy
GPU analysis. The adapter remains excluded from production builds.

Three-frame NP=1/2 validation passed at t=0.001, 0.002, 0.003, including exact
current-frame VTK field checks and bitwise unchanged solver outputs. Evidence:
`out/insitu_multiframe_20260929/summary.json`. Current artifact names include
`.step00000001` through `.step00000003`; per-rank `lifecycle_rank*.json` must
report frames [1,2,3] and finalized=true. This also rejects per-frame backend
reinitialization or missing finalize. Temporary geometry/view proxies are deleted
after each frame. Three frames do not establish long-run memory stability.
Budgets, statistics, restart, streamlines, extract files and non-TGV cases remain
pending. D2 now approves coalescing crossed time targets into one actual-state
frame, recording their count, and default-off optional initial/final sampling.
No production defaults have been introduced.

## Bounded Schedule Gate

`insitu_schedule_probe`, built through the root CMake, exercises a solver-free
Fortran scheduling module: step/time modes, nonuniform times, crossed targets,
optional initial/final samples, duplicate suppression and invalid clock rejection.
The module is currently test-build-only. Target times use origin + index*period,
not repeated addition; comparisons do not use an early-trigger tolerance.

The real TGV runner accepts mutually exclusive `--render-step-interval 2` or
`--render-time-interval 0.0004` together with `--render-backend-dir`. The step
test passed NP=1/2, rendering only step 2 without forcing a final frame. The time
test passed NP=2, rendering actual times 0.001/0.002/0.003 with crossed counts
2/3/2. Records are under `out/insitu_schedule_steps_20260929/` and
`out/insitu_schedule_time_20260929/`. Field/derivative gates and bitwise on/off
solver output comparisons passed. Initial/final switches subsequently gained
real-loop coverage below. Neither test changes dt.

Scheduling currently controls rendering only: validation snapshots still capture
every completed step. There is no production input, restart recovery, statistics
weighting or memory-budget claim. Tests start from step/time zero, not checkpoints.

## Real Initial And Final Sampling

The runner accepts independent `--render-initial` and `--render-final` flags,
both default off. Initial snapshots have step/time/dt=0 before the first RK step.
Final rendering uses a saved private copy of the last completed-step sample,
not re-read arrays or the loop clock after controller/user hooks. An already
rendered final state is not emitted twice. The backend finalizes after this check.

Passed local 32^3 matrices: NP=1/2 steps=2 with both flags (frames [0,2,3]);
NP=2 time interval 0.0004 with both flags ([0,1,2,3], final deduplicated);
NP=2 steps=10 with only final enabled ([3], delayed backend initialization).
Evidence directories: `out/insitu_endpoints_steps_20260929/`,
`out/insitu_endpoints_time_20260929/`, `out/insitu_final_only_20260929/`.
Checks include initial-state global stencil, all current-frame VTK fields,
CPU/GPU differences <=2e-10 and bitwise unchanged original solver outputs.

Per-frame JSON now records current VmRSS, historical peak RSS at that callback,
and persistent bridge/final-snapshot payload bytes. RSS includes solver/runtime
and validation-oracle memory; it is neither GPU memory nor incremental renderer
cost. Current and peak RSS are sampled at different instants. This bounded test
does not certify long-run memory stability or a production resource limit.

## On-Demand Capture

With `--on-demand` and a rendering backend, only GPU sampling-on cases enable
`ASTR_INSITU_TEST_ON_DEMAND=1`. CPU reference cases still capture every step.
Without final sampling, the scheduler is previewed before allocating/downloading
fields or exchanging diagnostic halos. Skipped steps advance only the schedule
clock; due events are committed by actual capture/rendering, not consumed twice.
With final sampling enabled, all-step capture is deliberately retained so normal
loop termination can use the last complete-state snapshot without rereading it.

Tests: NP=1/2 steps=2 capture only step 2; NP=2 time interval 0.0015 captures
steps 2,3; NP=2 steps=10 plus final keeps three snapshots and renders only step 3.
Evidence is under `out/insitu_on_demand_{steps,time,final}_20260929/`.
This option requires the bounded rendering pipeline, does not alter production
inputs, and provides no performance or total-memory-limit claim.

## Window Integral Foundation

Build `insitu_time_integral_probe` through the root CMake. The solver-independent
`clipped_trapezoid` routine returns interval increments and covered duration for
arbitrary precomputed integrands, with no extrapolation. Its exact-value tests
cover nonuniform times, clipped constant/linear integrands, separately interpolated
second moments, empty overlap, large finite values, overflow rejection, and
duplicate/backward/nonfinite input rejection. The probe passed on 2026-09-29.
The routine is not yet linked into solver statistics. Covariance definitions,
stable accumulated variance, saved state and paired restart remain separate gates.

## Pointwise Velocity Statistics Foundation

The user has approved separate Reynolds/Favre velocity covariance outputs and a
distinct positive density-weighted stress (mean density times Favre covariance).
`insitu_velocity_statistics_probe` tests the corresponding FP64 pointwise core,
using clipped endpoint weights and stable weighted central moments rather than
subtracting large raw second moments. Exact-value tests cover varying density,
cross covariance, clipping, no coverage, invalid-sample state preservation and
variance one under a 2^40 mean translation. The root-CMake probe passed.
This is not a GPU kernel, real-flow statistics gate or paired-checkpoint test.
Spatial reductions and the two spatial RMS definitions remain separate work.

The subsequent `insitu_spatial_statistics_probe` passed with NP=1/2. It verifies
unique periodic nodal volume (including zero-weight duplicated endpoints), local
variance mean then square root = 1 versus regional-signal temporal RMS = 0 for
opposite-phase fluctuations. Empty-contribution ranks are supported; a negative
weight on one rank and globally zero measure fail collectively. This is an MPI
foundation test, not real-flow/GPU statistics integration or CURVE validation.

## Statistics And Schedule Record Continuation

The root-CMake `insitu_velocity_statistics_probe` now round-trips a little-endian
stream state containing the previous sample and weighted central accumulators.
Continuation agrees exactly with uninterrupted accumulation. Batch, step, time,
window mismatches and truncated records are rejected without replacing state.
`insitu_schedule_probe` round-trips both step/time modes, checks restart-point
deduplication and subsequent crossings, rejects identity/configuration/truncation
errors, and preserves terminal state. Both probes passed on 2026-09-29.
These are record-level tests, not immutable-batch publication, checksum/MPI
agreement, solver field restart or IS2 completion evidence.

## Real TGV Statistics Reference

`run_insitu_sample_validation.py --statistics` activates the test-only window
`[0.0005,0.0025]` and captures initial plus all three completed states. The
Fortran pointwise core accumulates canonical samples on the host for both solver
backends. It also computes unique-volume mean pointwise Reynolds variance and
the temporal Reynolds variance of the spatially averaged velocity signal.
Rendering cadence cannot suppress these statistics samples.

Approved D7 compares means/covariances/density stresses with absolute `2e-10`,
compares squared RMS with that bound and separately reports raw RMS differences.
The independent two-pass oracle constructs quadrature weights from sample times,
without reusing the incremental Fortran implementation. NP=1/2 passed in
`out/insitu_statistics_fields_20260929/`; maximum oracle difference was
`4.16334e-16`, CPU/GPU statistics difference `8.88178e-16`, raw RMS difference
`4.54728e-16`. Existing output was bitwise unchanged by sampling/statistics.
This does not verify device-resident accumulation or actual paired field restart.

The NP=2 `--statistics --on-demand --render-step-interval 2` EGL run also passed
in `out/insitu_statistics_sparse_render_20260929/`. It renders steps 0 and 2 but
retains all four statistical samples. All six rank/frame statistics files match
the no-render group byte for byte. The sampling helper unit tests passed (3 tests).

## Device-Resident Pointwise Statistics Candidate

`insitu_statistics_gpu.cuf` accumulates FP64 central moments directly from `q_d`
on half-open owned nodes. State persists on device between completed samples.
No canonical host field is uploaded. Test outputs `sample.device_statistics.*`
contain the 41 device-computed pointwise fields, but their regional metadata is
still copied from the host reference; it is not a GPU reduction result.
All results are downloaded each sample for this gate, not as a production policy.

NP=1/2 passed in `out/insitu_device_statistics_fixed_20260929/`: device/host
pointwise results matched exactly and independent-reference error was
`4.16334e-16` (raw RMS error `5.69750e-17`). The initial resource-exhausted kernel
launch was fixed with `launch_bounds(512)`; the block shape remains `(512,1,1)`.

NP=2 memcheck passed with zero errors per rank in
`out/insitu_device_statistics_memcheck_hostmpi_20260929/`. Its MPI overrides were
`--mca pml ob1 --mca osc pt2pt --mca btl self,vader,tcp --mca coll_hcoll_enable 0
--mca coll_ucc_enable 0 --mca opal_cuda_support 0`, with pinned halo. Statistics
were also checked against the independent reference after the sanitized run.
Earlier default-UCX and ob1-only attempts failed on MPI CUDA context/pointer
probe API calls. Those logs are retained, not suppressed or counted as passes.
This gate does not certify CUDA-aware MPI, device checkpoint restore, production
resource admission or production output cadence.

## Device State Serialization And Continuation

`--statistics --statistics-roundtrip` writes a per-rank `ASTRGS01` state record
after step 1, releases the device statistics arrays, restores them from file and
continues the real solver. Records bind batch, step/time, sample time, window,
global/local dimensions, rank coordinates/offsets and topology to all 30 state
components. Validation precedes replacing device state. Wrong batch, step, time,
window, topology and truncated records are rejected.

NP=1/2 passed in `out/insitu_device_state_roundtrip_20260929/`. All nine device
statistics outputs were byte-identical to the uninterrupted reference. The
NP=2 run with extra malformed-record probes passed memcheck with zero errors
per rank using the host-only MPI overrides above, and its six outputs were
also byte-identical. Evidence: `out/insitu_device_state_memcheck_20260929/`.
This is an in-process serialization/reallocation gate, not a separate-process
flow restart or immutable paired-batch publication. File checksums and complete
batch validation remain an outer-layer requirement, not provided by this record.

## Immutable Batch Protocol Foundation

Build `insitu_checkpoint_batch_probe` via root CMake. Run with NP=1/2 and a new
nonexistent directory below an existing test-output parent. The native module
binds explicit identity, step/time, topology, configuration and expected leaf
filenames to file sizes and CRC-64/ECMA-182. All ranks must report readiness.
A closed pending manifest is hard-linked to a non-overwritable completion name;
no completion manifest means no valid restart batch.

NP=1/2 passed (including CRC check vector `123456789` = `6C40DF5F0B497347`),
missing files/manifest, one unready rank, wrong configuration, repeated publish
and same-length corruption rejection. Evidence directories:
`out/insitu_batch_np1_20260929/`, `out/insitu_batch_np2_20260929/`, and
`out/insitu_batch_np2_final_20260929/` after explicit MPI error checks.
The probe deliberately leaves corrupted payloads at exit. These are not restart
inputs. This is not solver-flow/statistics pairing or a new-process restart test.
No new runtime dependency is required; POSIX filesystem calls supply publication.
CRC is integrity checking, not authentication. Power-loss durability and
cross-filesystem publication are not claimed. Writers must close and stop
modifying all batch payloads before collective publication.

`copy_batch_file` copies original checkpoint bytes to a new destination and
compares destination size/CRC with the source fingerprint; it refuses overwrite.
NP=2 copy/no-overwrite plus the batch rejection suite passed in
`out/insitu_batch_copy_np2_20260929/`. Actual solver integration is pending the
CPU exact-state-sidecar decision: legacy CPU restart rebuilds q from primitive
HDF5 fields, whereas GPU restart already restores an exact q record.

## Independent-Process GPU Paired Restart

The user selected GPU-only pairing for now; no CPU exact-state extension was made.
Run `run_insitu_paired_restart.py --gpu <astr> --mpiexec <mpiexec> --output <new-dir>`.
It prepares 32^3 periodic TGV with 643e/RK3, FP64, explicit synchronization and
pinned halo. At checkpoint step 2, `ASTR_INSITU_TEST_BATCH_PREFIX` publishes the
original HDF5/auxiliary, per-rank exact q and host/device statistics files.
`ASTR_INSITU_TEST_RESTORE_BATCH` validates/stages the explicitly chosen batch
before legacy initialization can rewrite checkpoint files. Destinations must be
empty. The legacy GPU generation check remains in effect. No boundary/RK/output
phase is moved, and restart does not push the checkpoint sample a second time.

NP=1/2 passed in `out/insitu_paired_restart_guarded_20260929/`: steps 3/4 fields
exactly match continuous execution and both statistics products match byte for
byte. NP=2 missing-rank and one-byte-corruption cases fail before sampling.
Source-batch file hashes are unchanged. Non-paired CPU/GPU NP=1/2 output
invariance passed in `out/insitu_pair_default_off_20260929/`; both builds pass.
The paired gate rejects external grids, nonperiodic/non-TGV cases, other global
dimensions, sequential output and legacy averaging. It remains test-only.
With `--render-backend-dir <lib/catalyst> --extracts`, the separate run
`out/insitu_paired_extracts_retry_20260929/` passed NP=1/2 with real EGL:
continuous frames 0/2/4, resumed frame 4, and identical step-4 image pixels.
Slice/Q partitioned geometry is written and read back from files inside the
callback, checking time/step, 25 finite point fields, and slice/Q constraints.
Independent offline readback of all 16 products also passed with
`check_insitu_extracts.py --paired-output <paired-output> --report <new-json>`
under a fresh `pvpython` process. Use the matching HPC-X `mpiexec -np 1` launcher:
direct singleton startup on this installation needs MPI installation-prefix
configuration. Evidence: `offline_readback.json` in the same output directory.
Streamlines, formal runtime configuration and allocation admission remain open.

Approved local D8 limits apply only to 32^3 NP=1/2: additional postprocessing
device memory <=2 GiB per physical GPU, additional host memory <=4 GiB per node,
and device free memory >=1 GiB. Use matched on/off measurements before allocation
admission and third-party peak observations. Exceeding a limit must stop, not
reduce resolution or switch backends. These are not production defaults;
sampled third-party usage does not establish an absolute hard allocation limit.

`run_insitu_sample_validation.py --observe-resources --statistics
--render-step-interval 2 --render-final --render-backend-dir <lib/catalyst>`
adds process-tree RSS and per-physical-GPU NVML sampling to the matched GPU
on/off cases. `out/insitu_resource_observation_retry_20260929/` passed NP=1/2
alongside CPU/GPU field and on/off invariance gates. Additional sampled peak
differences: host 439/936 MiB, GPU 132--152 MiB; observed free device memory >17 GiB.
This covers statistics plus three EGL slice/Q frames, not streamlines or extract
writers. RSS sums may overcount shared pages; the 20 ms sleep excludes query
overhead. Short spikes can be missed. Allocation admission and injected
over-budget rejection still require separate tests.

## Allocation Ledger Foundation

Build `insitu_resource_budget_probe` through the root CMake project and run
`<build>/bin/insitu_resource_budget_probe`. CPU and CUDA build variants pass
limit/headroom equality, over-limit rejection without mutation, duplicate slots,
release accounting, active-reservation reconfiguration rejection and int64 size
overflow checks. `src/insitu_resource_budget.F90` has no Catalyst/CUDA dependency.
It is not yet connected to actual allocations. Node/GPU aggregation across
ranks, allocation-failure rollback and runtime admission must be tested at their
real call sites; per-rank ledgers alone do not enforce a physical GPU quota.

The GPU statistics first-allocation and restore-allocation paths now accept the
test-only `ASTR_INSITU_TEST_DEVICE_BUDGET='<limit> <headroom>'` in bytes. A shared
node communicator gathers PCI identities, demand and free memory; all ranks on
the same local non-MIG device count toward one allocation demand. The guard
covers state/output/error-mask device buffers only, not all postprocessing.

`run_insitu_device_budget.py --gpu <astr> --mpiexec <mpiexec> --output <new-dir>`
passed four cases in `out/insitu_device_budget_20260929/`: NP=1 admission,
NP=2 separate-device admission at 12 MiB/device, rejection of the same ranks
sharing one GPU at that limit, and an impossible 1 TiB headroom rejection.
These altered limits are fault-injection values, not new defaults. No statistics
output is produced for the rejected cases. With `--device-budget`, the paired
restart driver passed NP=1/2 exact fields/statistics and malformed-batch checks
in `out/insitu_paired_budget_20260929/`. Host/node accounting, other buffers,
third-party resources and formal runtime configuration remain open.

## Streamline Gate: Original Failure and Patched Verification

`run_insitu_paired_restart.py --streamlines --extracts --render-backend-dir ...`
adds the approved RK45/length-unit step settings, instantaneous TGV traces and
an independent constant-field crossing oracle. The first run
`out/insitu_streamlines_20260929/` failed at NP=1 initial frame with endpoint/
straightness error 1.0192611599180168e-7 against 2e-10. The saved points are
float, despite double seeds. VTK serial output points and parallel tail points
are created at default single precision. A local dependency patch requires the
pending approval; no tolerance relaxation or post-hoc coordinate correction is
accepted. The constant oracle integrates forward pi from x=pi/2 to 3pi/2 to
isolate MPI crossing; actual TGV uses both directions with pi allocated to each.
Exterior termination, mean traces and successful NP=2 crossing remain open.

The user subsequently approved the local dependency patch in
`scripts/insitu/patches/`. It also fixes parallel per-segment float seeds.
An empty-partition ColorBy collective mismatch was diagnosed with both rank
stacks; the pipeline now sets a fixed map without auto-ranging, rebuilding the
approved preset range each frame. Run `insitu_streamlines_double_seeds_20260929`
passed NP=1/2 paired restart including all four products, exact step-4 pixels,
fields and statistics. Constant-field endpoint/straightness max error was
1.7763568394002505e-15. Partition segments are checked by seed identity and
connected endpoints, not required to be a single VTK cell. Fresh-process
`check_insitu_extracts.py --streamlines` read all 32 products successfully.
Mean traces and exterior termination remain unverified.

The same run used `--host-budget --device-budget`. The new host envelope sums
explicit statistics arrays across a shared-memory node. The separate
`insitu_host_device_budget_20260929` rejects two rank-local requests of 21071864
bytes against a 32 MiB node quota (aggregate 42143728), while each alone fits.
The host envelope excludes sampler/bridge/diagnostic buffers, implicit compiler
temporaries and third-party storage. It does not replace full-process sampling
or complete resource admission.

## Mean Streamline Bridge

`run_insitu_paired_restart.py --mean-streamlines --streamlines --extracts
--host-budget --device-budget --render-backend-dir <backend> ...` passed NP=1/2
in `out/insitu_mean_streamlines_20260929/`. Native host Reynolds/Favre means and
coverage are passed through the bridge; serialized statistics are read only by
the test oracle, never as the rendering data source. GPU statistics remain an
independent numerical comparison path rather than the direct rendering source.
Initial zero coverage omits mean traces. Steps 2/4 each produce Reynolds and
Favre streamlines, colored by their own mean u with the approved fixed range.
All six products have identical continuous/resumed step-4 pixels; fields and
statistics remain exact. `check_insitu_extracts.py --streamlines
--mean-streamlines` independently read 44 geometry files and checked finite mean
fields and clipped coverage duration. The new temporary/retained mean buffers
still need full resource admission and whole-pipeline remeasurement. Exterior
termination and formal runtime integration remain open.

## Exterior and No Conventional Field I/O

Run `pvbatch --symmetric inspect_insitu_exterior.py --output <new-dir>` with the
matching MPI launcher. NP=1/2 passed in `insitu_exterior_np{1,2}_assertions_20260929`:
16 constant-field backward trajectories stop with OUT_OF_DOMAIN at x=0, maximum
endpoint error 5.551115123125783e-16, finite coordinates and no reversal/wrapping.
Assertions also check seed completeness and straightness. This is a bounded
constant-field test, not a general nonlinear boundary-event accuracy guarantee.
Plain pvbatch only executes the top-level script on root and is not suitable
for this explicitly collective probe. Use explicit NumPy operations rather than
global max/any/size names, which programmable-source execution can replace.

The paired driver has a separate continuous gate:
`--continuous-no-field-io <prior-paired-output>`. In
`out/insitu_no_field_io_fixed_20260929`, NP=1/2 use the existing no-field-I/O
benchmark switch and checkpoint interval 1000 (four completed steps). No HDF5
or restart files are created; six image/geometry products still appear. Flow
and statistics snapshots are byte-identical to the field-I/O reference and
image pixels match exactly. This mode does not test restart, and does retain
test-only oracle snapshots. It is not a claim of eliminating all full-field
diagnostic files or of completing the production interface. The first failed
run retained checkpoint interval 2; changing test inputs, not solver checkpoint
semantics, resolved that failure.

Adding `--no-oracle-io` sets `ASTR_INSITU_TEST_ORACLE_IO=0`. In
`out/insitu_no_oracle_io_20260929/`, NP=1/2 emit no sample*.bin, HDF5 or restart
files. All six image products have exact reference pixels and all extracted
geometry files match reference bytes. Only small schedule/lifecycle records
remain as test infrastructure; arrays reach Catalyst through memory, not files.

Adding `--observe-resources` creates matched off cases and samples the full
pipeline. `out/insitu_full_pipeline_resources_20260929/` passed with host peak
differences about 533/1299 MiB and device differences 336 MiB (NP=1), 356/288 MiB
(NP=2); all observed free device memory exceeds 17 GiB. This includes mean and
instantaneous traces, slice/Q, geometry writing and readback. It remains sampled
observation, not proof that implicit/third-party allocations obey a hard limit.
The uniform buffer budget and formal configuration are still incomplete.

## Run Configuration Foundation

`src/insitu_run_config.F90` defines default-disabled options and a strict
`&insitu_run` parser. It is not yet called by the solver and does not change
`insitu-check` or the test environment interface. Build the
`insitu_run_config_probe` target, then set `ASTR_INSITU_CONFIG_PROBE` to that
executable when running `test_insitu_run_config.py`: all 12 cases passed,
including CRLF, exclusive schedules and invalid/truncated input rejection.
`src/insitu_config_collective.F90` now compares parsed values across MPI ranks
using typed broadcasts and a collective verdict. Any rank's parse failure is
reported consistently, and failure leaves the caller's options unchanged.
Formatting differences alone are accepted. Build `insitu_config_collective_probe`
and set `ASTR_INSITU_COLLECTIVE_PROBE` and `ASTR_INSITU_MPIEXEC` alongside the
serial probe environment variable to run all 22 tests. Ten NP=2 cases cover
equivalent files, differing option types, one-rank parse failure and a missing
rank-local file. CPU root-CMake build passed. Neither configuration interface
is connected to the solver lifecycle yet; that integration must use native
options directly, not translate them into test environment flags.

## Native Field Module Extraction

`src/insitu_fields.F90` now owns private field capture, periodic ownership
canonicalization and explicit sixth-order velocity diagnostics. It and the CUDA
download module build independently of BUILD_TESTING and Catalyst. The existing
validation driver calls these interfaces; test files and manufactured fields
remain in the validation layer. The interfaces do not enable runtime sampling
by themselves or certify nonperiodic/curvilinear cases.

`insitu_fields_extraction_20260929` passed 32^3 NP=1/2 CPU/GPU sampling, on/off
bitwise solver-field invariance, independent statistics, manufactured gradients
and global-stencil comparisons. Maximum CPU/GPU field and derived differences
were 1.9895196601282805e-13 and 1.3575073068777237e-14. CPU/CUDA builds and five
source-without-tests CMake configuration checks passed. No rendering regression
was rerun in this extraction gate; the formal lifecycle and resource coverage
remain incomplete.

## Native Statistics Lifecycle

The coordinator is now `src/insitu_session.F90`; the old validation-module name
is superseded. Set `ASTR_INSITU_CONFIG` to an explicit `&insitu_run` file to use
the native statistics lifecycle. The output directory must already exist
(`outdat` is created by normal ASTR setup). Example for the bounded TGV gate:

```fortran
&insitu_run
 enabled=t, statistics=t, render=f,
 statistics_window=0.0005,0.0025, output_directory='outdat',
 host_budget_bytes=4294967296, device_budget_bytes=2147483648,
 device_reserve_bytes=1073741824
/
```

The budgets here are approved local validation values, not production defaults.
All completed steps accumulate; finalization emits the last statistics snapshot
in ASTRST01 format only. No raw-field test snapshots are needed. Native GPU
paired restart and native GPU/EGL rendering are now connected as described below.
The configuration is not converted into test environment variables, and
BUILD_TESTING=OFF ignores those variables entirely. Native GPU statistics still
coexist with host statistics; device spatial reduction is pending.

`run_insitu_formal_statistics.py` compares native output against a prior immutable
`run_insitu_sample_validation.py --statistics` run. With `--production`, it also
checks no-config default-off while stale test environment variables are present.
The test-enabled native path passed eight CPU/GPU NP=1/2 on/off cases in
`insitu_native_statistics_20260929`, with exact final statistics and solver fields.
Legacy sampling and paired-restart regressions passed in
`insitu_session_legacy_20260929` and `insitu_session_pair_20260929`.

Independent CPU/CUDA root builds with BUILD_TESTING=OFF and
ASTR_WITH_CATALYST=OFF passed. `insitu_native_no_testing_20260929` passed 12
NP=1/2 CPU/GPU no-config/disabled/enabled runs, including stale test variables.
Final statistics are byte-identical and solver fields bitwise identical to the
reference. Three NP=2 GPU fault tests also passed: insufficient host budget,
missing device budget, and an explicitly unsupported formal-render request.
These gates do not certify the remaining formal rendering or paired restart.

## Native GPU Paired Restart

Formal statistics now accepts optional `batch_prefix` and `restore_batch` paths
in the same namelist. They require enabled statistics; restore additionally
requires `lrestart=t`. The admitted state remains GPU 32^3 periodic TGV,
643e/RK3 and nonsequence HDF5. CPU exact paired restart remains deferred.
When in-situ is disabled, normal checkpoint/restart defaults are unchanged.
An enabled native statistics restart must explicitly select a paired batch.

`run_insitu_paired_restart.py --native` exercises a BUILD_TESTING=OFF,
Catalyst-disabled executable, without test sampling files. Native tests run
through completed step 5 so the checkpoint written before RK step 4 is present.
Step-4 host/device statistics records and exact q payloads, step-5 final
statistics, and HDF5 fields must match continuous execution exactly. Only the
wall-clock checkpoint generation identifier is excluded from the q comparison.
NP=1/2 passed in `insitu_native_pair_complete_20260929`; immutable source hashes
and missing/corrupt batch rejection also passed. Combined native-render restart
evidence is recorded below.

`insitu_native_pair_faults_20260929` repeated the native NP=1/2 equivalence gate
and additionally rejected mismatched statistics windows and MPI topology before
sampling. Configuration parser/collective tests now total 27 passing cases.

## Native Current-Context Device Mapping

`src/insitu_device_map.cpp` maps the existing CUDA context's device UUID to a
unique CUDA-capable EGL device. It does not create/switch a CUDA context, infer
the EGL ordinal from local rank, or fall back to software. It is compiled only
with CUDA and Catalyst enabled and requires EGL development headers. Runtime
driver/EGL symbols are loaded dynamically. Other builds have no new EGL dependency.

Build `insitu_device_map_probe` through root CMake and run
`run_insitu_native_device_map.py --probe <probe> --mpiexec <launcher> --output <new-dir>`.
`insitu_native_device_map_20260929` passed NP=1/2, reversed visible devices,
two ranks sharing one visible GPU, unchanged CUDA binding, and no-context
rejection. Mapping agrees with independent CUDA/EGL enumeration. The root build
and five configuration-decoupling tests passed. This is a local Linux NVIDIA
full-device gate, not MIG support or verification of the actual VTK rendering
context. The following native-render gate additionally checks actual contexts.

## Native TGV Rendering

The bridge is now `src/insitu_mesh_adapter.cpp`, built without test-directory
sources. `scripts/insitu/tgv_pipeline.py` receives output path and CUDA UUID via
Catalyst arguments and only reads in-memory mesh data. It needs neither a test
binding wrapper nor sample/schedule files. The approved camera, color range,
slice/Q and instantaneous/mean/constant-diagnostic streamline presets remain
unchanged. Actual VTK EGL contexts are checked against solver UUID on every
product. The callback itself writes JPEG and raster EPS plus VTK geometry.

`run_insitu_native_render.py` passed NP=1/2 against the existing immutable
no-oracle reference. BUILD_TESTING=OFF tests passed steps and time scheduling in
`insitu_native_render_no_testing_20260929` and `insitu_native_render_time_20260929`.
All products are pixel-identical/geometry-byte-identical, with no conventional
field files, test snapshots or schedule text. Fresh pvpython offline readback
passed 32 products using `check_insitu_extracts.py --continuous-only --streamlines
--mean-streamlines`. Mean-streamline visual inspection passed. Seven identity/
sampling unit tests and five configuration-decoupling tests passed.

See `scripts/insitu/README.md` for runtime options and dependencies. Device
spatial reduction and full resource accounting remain outstanding.

### Native Rendering With Paired Restart

`insitu_native_render_pair_20260929/summary.json` passed with BUILD_TESTING=OFF
at NP=1/2. Continuous frames are steps 0/2/4; restoration from checkpoint 2
produces only frame 4. Step-4 pixels and geometry bytes match exactly, as do
checkpoint q payloads, host/device statistics state, HDF fields and final step-5
statistics. Wall-clock checkpoint generation identifiers are excluded from q
state comparison. Missing/corrupt batches, changed statistics windows and wrong
MPI topology are rejected before sampling. Source batches remain unchanged.

An independent pvpython process read all 44 extracted products successfully;
`insitu_native_render_pair_20260929/offline.json` records coordinate/field/time,
slice/Q and constant-field cross-partition streamline checks at the approved
2e-10 tolerance. This is the bounded native GPU restart gate, not CPU exact
restart, arbitrary-grid rendering or full resource certification.

The 2026-09-30 configuration guard additionally requires the saved schedule
presence to match native `render` in both directions. Render-enabled batches
cannot silently become statistics-only restarts, and statistics-only batches
cannot invent a fresh render schedule. Existing schedule restoration rejects
interval changes. `insitu_native_render_pair_guards_20260930` passed NP=1/2
equivalence plus render-disable/interval rejection; the corresponding
`insitu_native_statistics_pair_guards_20260930` passed NP=1/2 equivalence and
render-enable rejection before attempting to load a renderer.

### Device Spatial Reduction

`insitu_statistics_gpu.cuf` now reduces current velocity and the three Reynolds
variance diagonal entries over half-open owned nodes using a fixed FP64 shared
memory tree. Six sums cross to the host for a seven-value MPI reduction including
physical volume. The six-double buffer is included in device admission and
allocated/released with statistics, including restore. No floating-point atomic
sum is used. Uniform Cartesian TGV weights are unchanged; CPU retains host sums.

`insitu_device_spatial_checked_20260930` passed NP=1/2 independent discrete
statistics, CPU/GPU, on/off solver fields and device-state roundtrip checks.
Times/windows remain exact. Volume uses the existing independent-reference
2e-10 tolerance: host summation and count-times-cell-volume differ by roundoff
(the initial exact metadata assertion differed by 6.37e-12).
`insitu_device_spatial_pair_20260930` passed native NP=1/2 exact same-backend
paired continuation and rejection gates. `insitu_device_spatial_memcheck_20260930`
ran NP=2 Compute Sanitizer memcheck with zero errors on both ranks.
Host point statistics and full device output copies still serve field checking
and rendering. Full native resource enforcement remains outstanding.

### Native Render Resource Observation After Device Reduction

`run_insitu_native_render.py --observe-resources` now measures a matched disabled
run before the enabled run. Both `insitu_native_spatial_render_resources_20260930`
(steps) and `insitu_native_spatial_time_resources_20260930` (time) passed NP=1/2
pixel/geometry equivalence without conventional fields or test snapshots.
Across these runs the additional sampled host peak difference stayed below
548 MiB / 1323 MiB for NP=1/2. Additional device peak differences were about
336 MiB / 356 and 288 MiB; observed free memory stayed above 17 GiB.
This covers dependency loading and synchronous rendering during the monitored
process lifetime, but cannot guarantee capture of every short-lived peak.
The test monitor stops the process group on observed overbudget usage; this
does not replace native third-party resource enforcement. Bridge C ABI entry
points now catch C++ exceptions and abort MPI with a stage-specific diagnostic.
Actual allocation-failure injection has not yet been validated.

### Bridge Allocation Failure Injection

`insitu_allocation_fault` is an EXCLUDE_FROM_ALL shared target available only
with BUILD_TESTING and Catalyst. It is not linked into ASTR or installed.
The runner preloads it only into solver processes, refusing the bridge coordinate
vector size on a selected rank rather than consuming real system memory.

`insitu_bridge_allocation_fault_checked_20260930` passed NP=1 (rank 0,
862488 bytes) and NP=2 (rank 1, 444312 bytes). Both logs identify the injected
allocation and `ASTR INSITU BRIDGE ERROR: execute: std::bad_alloc`, followed by
MPI abort without timeout or frame products. This validates this C++ allocation
failure path, not arbitrary third-party OOM recovery. The first run lacked the
required RK timing flag for no-field-IO and was rejected before rendering; it
is retained as a failed harness run, not an exception-handler pass.

### Disabled Build Regression After Integration

On 2026-09-30, `build_insitu_native_check` rebuilt successfully with CUDA,
Catalyst and BUILD_TESTING all OFF. The NP=1 32^3 TGV in
`insitu_disabled_cpu_20260930` completed without an in-situ configuration or
in-situ outputs. All eight HDF datasets (`nstep`, `time`, `ro`, `p`, `t`,
`u1`, `u2`, `u3`) exactly match the existing CPU-off reference in
`insitu_device_spatial_checked_20260930/np1_cpu_off`.

Seven selected `test_cmake_production_decoupling.py` checks passed: four
CUDA/AIR5 combinations with Catalyst OFF, test-only MPI trace rejection, and
two CUDA OFF/ON configurations with Catalyst ON and BUILD_TESTING OFF. The
source fixture has no tests directory. Native bridge rules contain no test
source paths, fault/probe targets are absent, and installed pipeline entries
are present. The latter are configuration checks, not fresh full builds or
installation runs for every combination.

### Current Integrated GPU Restart Gate

`insitu_spatial_render_pair_final_20260930` repeats the formal render-paired
gate with device spatial reduction and bridge exception handling included.
NP=1/2 q payloads, host/device state, final statistics and HDF fields match
exactly; restored step-4 geometry bytes and image pixels also match.
Missing/corrupt batches, windows, topology, rendering enablement and interval
rejection passed. Independent pvpython readback passed all 44 products
(`offline.json`). This supersedes pre-reduction integrated restart evidence.
Native resource observation remains pending policy approval and implementation.

### Approved Native Resource Observer

The user approved the phase-boundary policy on 2026-09-30. Native GPU rendering
now records node RSS and current physical GPU job memory before statistics
allocation, after initialization, before/after frames and renderer finalization,
and after session-buffer release. NVML is dynamically loaded; unavailable data
is rejected. Rank PIDs are gathered per node, compute/graphics entries are
deduplicated, and a common device baseline is used for ranks sharing a GPU.
The observer aborts MPI on an observed increment/reserve violation.

`insitu_native_observer_20260930` passed NP=1/2 split GPUs and
`insitu_native_observer_shared_time_20260930` passed time scheduling with shared
GPU execution, including external monitoring and unchanged pixels/geometry.
`insitu_native_observer_limits_20260930` passed real observed host/device/shared
device limit refusal and impossible device-reserve refusal. These tests lower
limits rather than exhausting system memory; they require an observer error,
not merely any nonzero return. `insitu_observer_render_pair_20260930` passed
NP=1/2 exact continuation, existing refusal gates and 44-product offline readback.
Resource CSV stage sequences passed for both continuous and resumed processes.
CPU disabled build and seven configuration-decoupling tests passed again.

See `scripts/insitu/README.md` for the log contract, NVML header override and
runtime dependencies. Phase observations are not an allocator interception
mechanism; external sampling remains independent complementary evidence.

### Optional Device Statistics Download

The full 41-component output argument of `accumulate_statistics_gpu` is optional.
Native calls omit it; the independent test oracle explicitly requests it.
`ASTR_INSITU_GPU_STATS` reports cumulative sample calls and completed full-output
copies. `insitu_no_unused_stats_copy_20260930` passed NP=1/2 native rendering,
resource checks and unchanged geometry/pixels with five samples and zero such
copies per rank. At 32^3 this removes 10.25 MiB of aggregate D2H payload per sample;
it is a byte-count reduction, not a measured performance speedup.
`insitu_optional_stats_oracle_20260930` passed NP=2 CPU/GPU, independent reference,
solver on/off and state roundtrip with the explicit output argument retained.
`insitu_optional_stats_pair_20260930` passed NP=1/2 native exact continuation and
refusal gates; the driver checks six/three samples for continuous/resumed runs
and zero full statistics output downloads. Checkpoint state transfers are not
included in that counter and remain necessary. Native 11-component flow sampling
and host point accumulation still run each step; this is not yet a device-only
statistics path or output-only flow transfer.

### Device-Owned Native Statistics

The subsequent native GPU implementation supersedes the intermediate transfer
counts above. Pointwise accumulation stays on device; host point states and
per-step flow samples are absent. Rendering downloads flow and mean fields only
when a frame is due. A covered final output downloads 41 fields once. Invalid
flags, regional reductions and checkpoint state transfers remain; this is not
zero-copy. The legacy independent test oracle is unchanged in purpose.

- `insitu_device_owned_statistics_20260930`: formal CPU/GPU comparison uses the
  approved D7 absolute tolerance and squared-RMS criterion; CPU baseline is exact.
- `insitu_device_owned_render_20260930`: NP=1/2, five samples per rank, three flow
  downloads and one final statistics download; images/geometry unchanged.
- `insitu_device_owned_pair_20260930`: exact paired continuation and refusal
  gates; `offline.json` independently reads 44 geometry products.
- `insitu_device_owned_legacy_pair_20260930`: `--legacy-pair-reference` restores
  immutable `ASTRPS01` batches at NP=1/2. Final statistics, HDF fields, pixels and
  geometry match the new reader/writer path; source-batch hashes are unchanged.
- `insitu_restore_memcheck_20260930`: NP=2 `compute-sanitizer --tool memcheck
  --error-exitcode 99` with `ASTR_INSITU_TEST_STATISTICS_ROUNDTRIP=1`; both rank
  logs report zero errors. MPI uses ob1/self,vader,tcp with CUDA support disabled
  in MPI for this pinned-host test, avoiding unrelated MPI registration paths.

New native GPU records use `ASTRPS02`, omitting redundant host point records but
retaining regional and device states and rendering schedule. Legacy `ASTRPS01`
remains readable. Restoring device state rebuilds derived statistics without
changing the saved accumulated state. Native resource logs now also observe
`statistics_exported`, before session buffers are released. CPU exact paired
restart remains deferred; these gates do not validate AIR5 or CURVE processing.

The current `insitu_device_owned_time_20260930` run repeats native NP=1/2 with
physical-time scheduling and matched off/on external resource sampling. Frames
remain 0/2/4, with no duplicate final frame or conventional full-field files.
Images and geometry match the step-scheduled reference. Observed additional node
RSS peaks are about 0.501/1.247 GiB; per-GPU peak differences are at most 0.348
GiB. These are sampled peak differences, not bounds on unobserved allocations.

`insitu_device_owned_final_statistics_20260930` repeats the 12 formal CPU/GPU
NP=1/2 unset/disabled/enabled runs on the rebuilt current executables. Solver
fields remain bitwise identical to their references; GPU statistics differ by
zero from the independent test output. Missing budget and unpaired-restart
refusals pass. Seven selected CMake source-copy configuration checks also pass
with the compiler and Catalyst directory explicitly set; an earlier invocation
without those opt-in variables skipped tests and is not acceptance evidence.

### Approved CPU Exact Paired Continuation

After explicit approval on 2026-09-30, CPU pairs preserve the complete pre-filter
q array in `ASTRCQ01` sidecars. They include duplicate nodes and halos. Metadata
checks include topology, rank, step/time, timestep, schemes and physical settings.
Only an explicitly selected paired restore replaces HDF-reconstructed q and
rebuilds primitives. Ordinary CPU restart and the legacy HDF format are unchanged.
Snapshot memory is included in the host admission envelope. CPU and GPU batch
identities differ; cross-backend continuation is not supported.

`run_insitu_cpu_paired_restart.py` passed in `insitu_cpu_pair_final_20260930`
(CPU-only build) and `insitu_cpu_pair_cuda_build_20260930` (CUDA build with
usegpu=f). NP=1/2 step-4 q and accumulated state plus step-5 statistics output
are byte-identical, and all eight HDF datasets are bitwise identical. Missing,
corrupt, topology, window and timestep mismatches are rejected. The source batch
hashes remain unchanged. An earlier parameter-refusal test edited the wrong
input file and stopped at its own assertion; the final driver uses the existing
controller timestep setter and both final suites pass.

`insitu_cpu_extension_gpu_pair_20260930` passes the unchanged GPU exact paired
gate. `insitu_cpu_extension_statistics_20260930` passes the 12 formal off/on
CPU/GPU configurations, including default-off behavior. This is bounded TGV
statistics restart coverage, not CPU EGL rendering or general checkpoint support.
