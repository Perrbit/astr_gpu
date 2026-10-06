# Native In-Situ TGV Preset

### Matched 256^3 Device Visualization Timing

`benchmark_tgv256_visualization.py` runs the separately approved render-only
demonstration: 256^3 periodic Cartesian TGV, FP64, 643e/643e, tenth-order
explicit filtering with scalar workspace, two GPUs and topology 2x1x1.
The explicit `products='tgv256_demo'` profile emits only Q_rs=0 surfaces and
instantaneous streamlines, both colored by speed in [0,1], at 1280x960.
The existing sixteen seeds and RK45 rules are retained with h=2*pi/256.
This profile requires device processing without accumulated statistics. It
does not extend the standard 32^3 profiles, walls, AIR5, CURVE or restart gates.
Q=0 is the strain/rotation balance surface, not a positive-Q vortex-core threshold.

```bash
python3 scripts/insitu/benchmark_tgv256_visualization.py \
  --output /new/empty/tgv256-comparison \
  --executable /path/to/device-enabled/astr \
  --mpiexec /path/to/matching/mpiexec \
  --library /path/to/paraview/lib/catalyst --transport device-aware
```

The default sequence is a two-step check, visualization on for 100 completed
steps, then visualization off for 100 steps. Both start from the analytic
initial condition, dt=1e-4, and use identical solver/output inputs. Every on
step publishes two JPEG/EPS pairs; no initial/final extra frame, checkpoint,
volume, slice or VTK geometry file is written. Small native diagnostics remain
enabled at the same frequency in both cases. Encoding movies is not performed.

The report compares the maximum rank-local completed-window duration, excluding
solver initialization but including first-frame lazy rendering setup and image
publication. It separately records pure RK, per-stage and process-launch time.
Nested stage times must not be added. This is one matched pair, not a repeated
performance distribution. Native phase resource checks remain enabled on the
visualization side: at most 6 GiB additional device memory per physical GPU,
16 GiB additional host memory per node, and at least 2 GiB free device memory.
Exceeding the budget aborts, without fallback or reduced image size.

`--smoke-only` stops after the two-step check. `--skip-smoke` is only for the
same executable and pipeline after that check has already passed. The output
directory must be fresh; failures retain logs and a failed report.

Current IS8 device admission and independent launch examples are in the final
section below. Unless marked as device processing, earlier derivative/field
sections describe the default `processing_backend='host'` compatibility path.
Its volume downloads are not used by the new device entry.

This is the bounded IS3 integration preset, not a production-scale resource
certification. It currently admits GPU, internally generated Cartesian,
periodic five-variable TGV with explicit sixth-order derivatives and local
extents at most 32. CPU statistics remain available without rendering.

Build from root CMake with `ASTR_WITH_CUDA=ON` and `ASTR_WITH_CATALYST=ON`.
`BUILD_TESTING` is not required by the native interface. The Catalyst API and
ParaView implementation must use compatible compiler/MPI libraries. CUDA render
builds also need EGL development headers.
GPU render builds additionally need `nvml.h` (override discovery with
`ASTR_NVML_INCLUDE_DIR` if necessary). NVML is loaded dynamically from
`libnvidia-ml.so.1` only when native GPU rendering starts. A missing runtime or
unavailable per-process memory data is an error, not a zero-memory reading.
The local approved ParaView 6.1.1
precision patch and rebuild procedure are in [patches/README.md](patches/README.md).
The ParaView Python runtime needs NumPy and Pillow. CMake installation places
the four Python pipeline/helper files under `share/astr/insitu`.

Set `ASTR_INSITU_CONFIG` to a namelist file. Without it, native in-situ work is
disabled. The output directory must exist; normal ASTR startup creates `outdat`.
Use actual absolute paths for the implementation and Python preset:

```fortran
&insitu_run
 enabled=t, statistics=t, render=t,
 statistics_window=0.0005,0.0035, output_directory='outdat',
 schedule_mode='steps', step_interval=2, initial_frame=t, final_frame=t,
 host_budget_bytes=4294967296, device_budget_bytes=2147483648,
 device_reserve_bytes=1073741824,
 implementation_path='/path/to/paraview/lib/catalyst',
 pipeline_file='/path/to/share/astr/insitu/tgv_pipeline.py'
/
```

These budgets apply only to approved local 32^3 validation, not larger jobs.
For time scheduling replace the schedule line by `schedule_mode='time',
time_interval=0.002, initial_frame=t, final_frame=t` and omit `step_interval`.
Frames carry actual complete-RK state time, not interpolated target times.
Statistics accumulate every completed step independently of frame cadence.

### Image Publication Failures

The native 32^3 preset completes collective geometry extraction and rendering
before capturing an in-memory screenshot. Only rank zero encodes and publishes
the JPEG/EPS pair. The pair is staged in exclusive `.partial` files and published
without replacing existing files. Encoding and collective capture errors are
fatal, not treated as file errors.

Only `EACCES`, `ENOSPC` and `EIO` during image staging/publication are recoverable.
On these errors the writer removes its own partial pair; all ranks agree on
the errno and phase, and each writes an identical
`missing.<product>.step<step>.rank<rank>.json` record containing the product,
completed step, physical time, reason and retained VTK geometry name. The frame
receipt marks that image as missing. There is no replacement picture or retry;
later frames, statistics and checkpoints continue normally. A checkpoint does
not certify that every image was published.

Failure to clean up or record the missing image is fatal. In particular, a full
filesystem that also prevents the missing-image journal from being written
cannot continue safely. Geometry writer errors, render/capture/encoding,
statistics, numerical errors and unknown failures remain fatal. This is not
crash-atomic publication of two files, nor recovery from a failed MPI collective.
The separate 256^3 demonstration preset retains its existing fatal policy.

The bounded IS4 matrix also checks fatal nonfinite private samples, missing
fields, statistics-file creation errors and MPI error returns at NP=2/4.
The C++ bridge now checks its MPI operation return codes explicitly. Three
process-interruption points before statistics/render-control/COMPLETE creation
leave an unsealed candidate that cannot restart. The previous complete batch
stays unchanged; explicitly restoring it reproduces state, statistics and
remaining images/geometry exactly. A resealed cross-step statistics or render
control member is rejected even when file checksums are valid. These checks do
not establish MPI fault tolerance, production capacity, or wall/AIR5/CURVE
rendering. The later IS5 wall sections document its separately approved
definitions and bounded implementation; the IS4 evidence alone does not admit them.

### Optional GPU Diagnostic Candidate

`derivative_backend='cpu'` is the default. Explicitly selecting
`derivative_backend='gpu'` admits only internally generated 32^3 periodic TGV,
GPU solver, NP=1/2 or NP=4 with topology 2x2x1, and explicit `643e` derivatives. Other configurations are
rejected; wall, AIR5, CURVE and the 256^3 demonstration are not admitted by
this candidate. The existing FP64 private-output derivative kernel supplies
all nine velocity gradients, Q_rs, divergence and curl, using a private halo
and the canonical completed-step velocity. Solver arrays are not overwritten.

The local numerical gate compared all fourteen diagnostic fields at complete
steps 0/2/4. Q_rs max absolute difference was 1.11e-16 at NP=1/2; enabling this
backend preserved the solver state bitwise. Compute Sanitizer reported zero
errors for all three tested rank processes. NP=2 real EGL products and
same-backend exact render continuation also passed. Changing the backend on
restart requires the existing explicit output override; it is not silently
accepted. These are bounded correctness gates, not production performance
certification.

With the default `products='all'`, the implementation downloads eleven basic fields, uploads three
canonical velocity components to its private GPU buffer, then downloads all
fourteen derived fields. GPU workspace is released after each diagnostic
frame; no solver-stage derivative cache is borrowed.

### Product-Specific Fields

The optional GPU candidate also accepts `products='q_surface'`,
`products='streamlines'` or `products='q_streamlines'`. Set
`derivative_backend='gpu'` alongside this option. Scope and budgets remain the
32^3 periodic Cartesian TGV NP=1/2 or NP=4 2x2x1 gate above. These profiles are explicit
subsets; omitting the option preserves all products and mean-streamline behavior.
The existing step/time render clock applies to the selected product union;
statistics and checkpoint clocks are unchanged. This is not independent
per-product scheduling.

| Profile | Fields supplied to Catalyst | Preset products |
| --- | --- | --- |
| `all` | Eleven basic and fourteen diagnostic fields; means when covered | Original slice, Q, instantaneous/diagnostic and available mean streamlines |
| `q_surface` | `u,v,w,Q_rs` | Q_rs=0.25 surface, colored by u |
| `streamlines` | `u,v,w` | Instantaneous streamlines and explicitly named constant-field crossing diagnostic |
| `q_streamlines` | `u,v,w,Q_rs` | Q surface and the two streamline products |
| `velocity_slice` | Plane-local `u,v,w` and coordinates | One explicitly selected Cartesian index plane |

All selected profiles preserve the original camera, color range, Q definition
and RK45 integration. Compact profiles do not export mean fields; explicitly
enabled device statistics still accumulate and retain exact restart state.
Both the preset and diagnostic capture reject missing required fields instead
of creating zero-filled stand-ins. Custom pipelines must accept a third Catalyst
argument identifying the profile, after output directory and device UUID.

Canonical velocity is still reconstructed on the host from four downloaded
conservative components (density and three momenta), using the original endpoint
ownership. Q profiles upload three canonical velocities to private GPU storage
and download only Q; streamline-only frames perform no derivative calculation
or velocity upload. Requested field D2H payload is therefore five or
four FP64 components per local 3-D node, versus twenty-five in the full GPU
diagnostic path or eleven in the default CPU-diagnostic path. Bridge copies are
seven/six components including coordinates, versus twenty-eight without means.
These counters exclude private-halo communication, compiler temporaries and
third-party internal transfers. They are not total PCIe traffic or a speedup.

Only due frames capture fields. The bridge retains owned host buffers through
synchronous Catalyst execution; it does not borrow solver pointers or claim
zero-copy operation. Profile changes on restart require the existing explicit
output override. GPU-resident geometry/rendering, static-coordinate reuse
and larger/nonperiodic admission remain future work.

### Lightweight Index-Plane Channel

Within the same 32^3 periodic TGV GPU NP=1/2 or NP=4 2x2x1 scope, select:

```fortran
 products='velocity_slice', derivative_backend='gpu',
 slice_axis='z', slice_index=4,
```

Add these entries to the existing `&insitu_run` namelist, retaining its explicit
resource budgets and step/time schedule. The axis is `x`, `y` or `z`; the global
node index is zero-based, 0 through 31. Defaults `z,4` give z=pi/4. This is an
exact mesh-node plane, not an interpolated or tilted physical-space slice.
The periodic upper endpoint at index 32 is not a second selectable copy of zero.
Nondefault slice settings are rejected for other profiles.

The GPU packs only four plane-local conservative components. Host-side endpoint
reconciliation and division preserve the canonical complete-step velocity.
No full 3-D sample, derivative kernel or velocity re-upload is used. Only the
plane coordinates and `u,v,w` enter Catalyst, as an explicit quad mesh. On a
normal partition interface only the upper-side rank supplies the plane;
tangential shared points are retained while cells are disjoint. Ranks without
the plane provide an empty domain but enter every Catalyst/MPI collective.
ParaView can drop zero-length field arrays on these empty domains; only empty
slice domains are exempt from the missing-field check.

`ASTR_INSITU_SLICE` logs plane node count, four-component field payload bytes,
capture and inclusive rendering time. The bridge reports six FP64 components
per plane node, including coordinates. Owned quad connectivity is built once
per lifecycle and is not included in the per-frame floating-field copy counter.
For NP=2 x decomposition and the default z plane, both ranks together request
35,904 field bytes per frame, versus 3,258,288 for the default full 3-D
eleven-component sample. This is a 98.90% reduction in requested field payload,
not a measured PCIe total or whole-step speedup.

The existing TGV preset writes JPEG/EPS and VTK geometry for `velocity_slice`.
Statistics and checkpoint semantics remain independent. Axis/index changes on
restart require explicit output override, just like other render settings.
The channel does not implement wall diagnostics, CURVE/AIR5, mean products,
independent cadence or a 2-D replacement for the 3-D streamline product.

The synchronous preset exports a z=pi/4 velocity slice, Q_rs=0.25 surface,
instantaneous streamlines, and Reynolds/Favre mean streamlines after statistical
coverage exists. The separately named `crossing_streamlines` uses a constant
velocity diagnostic copy to verify MPI crossing; it is not the physical flow.
All use the approved fixed camera/color ranges and RK45 settings. JPEG and
image-embedded EPS accompany parallel VTK PolyData geometry (`.pvtp`/`.vtp`)
with point fields, `simulation_time` and `complete_step` metadata. EPS is raster,
not vectorized streamline geometry; the VTK files retain the spatial geometry.

The adapter matches the current solver CUDA UUID to EGL and the pipeline checks
the actual rendered EGL context UUID. No rank-to-EGL ordinal assumption, Python
binding launcher, test snapshots or test scheduling files are required. There is
no silent CPU renderer fallback. Unset `CATALYST_IMPLEMENTATION_PREFER_ENV`.
MIG and other device backends are not admitted by this local identity gate.

Optional `batch_prefix='outdat/paired'` creates immutable paired batches at
existing solver checkpoints. To restore, select `restore_batch='/path/to/batch'`
and set `lrestart=t` in the flow input. Native GPU statistics and combined
formal-render restart passed the bounded NP=1/2 TGV gate. CPU exact paired
statistics restart also passes NP=1/2, with rendering disabled. Device spatial reduction passed bounded TGV numerical
and restart gates. Native GPU pointwise statistics remain on device, without a
per-step host point-state mirror or 11-component flow download. Flow samples are
downloaded only for scheduled render frames; six mean-velocity components and
coverage are exported for mean streamlines. The final 41-component statistics
output is downloaded once when the window has coverage. Per-sample invalid-value
flags and regional reduction results still cross to the host. Checkpoint state
transfers also remain necessary; this is not a zero-copy path.

New native GPU paired statistics use `ASTRPS02`, retaining device state and the
regional signal state without redundant host point records. The reader also
accepts `ASTRPS01`; bounded NP=1/2 legacy-batch restarts passed. The independent
test oracle retains explicit full-output downloads and host accumulation.
CPU batches use `insitu_cpu_q.rankNNNNNNNN.bin` (`ASTRCQ01`), preserving the
pre-filter conservative state including duplicate nodes and halos. This sidecar
is created only for an explicitly configured paired batch. Restoring such a
batch replaces the HDF-reconstructed conservative state and reconstructs primitive
variables before stepping. Legacy HDF format and ordinary restart remain unchanged.
CPU/GPU batch identities are distinct and cannot be interchanged. This support
is limited to the admitted 32^3 periodic TGV, explicit 643e/RK3, no legacy averaging;
it is not a general checkpoint conversion facility.

A paired restart must preserve whether rendering is enabled and its schedule
configuration. Changing render enablement or the step/time schedule is rejected;
the saved statistical window and MPI topology must also match.

## Completed-Step Native Checkpoints

The separate new output path can now use this GPU TGV preset. Set both
`ASTR_OUTPUT_CONFIG=datin/input.output` and `ASTR_INSITU_CONFIG=datin/input.insitu`.
Use `output.restore_directory` to select a new checkpoint, keep legacy
`lrestart=f`, and omit `batch_prefix`/`restore_batch`. Mixing old pairs with new
checkpoints is rejected. See [../output/README.md](../output/README.md) for
independent checkpoint/volume/slice configuration; all native checkpoint
members and their shared resources must be retained together.

Native checkpoints include the active renderer's saved schedule and content
identities in `insitu_control.bin`. Unchanged settings resume progress, clear
only the job-termination flag and avoid duplicate restored frames. Final
rendering occurs before final checkpoint sealing. A no-advance restart exports
saved statistics without initializing a renderer or producing another image.
Only explicit `restart_output='override'` permits enabling/disabling rendering
or changing its cadence/initial/final settings. Changed render origin starts
at the restore point; a requested new initial image does not resample statistics.
Statistical window/configuration remain a separate immutable history contract.

The entry script and implementation plugin are fingerprinted, not every Python
helper, transitive library or driver. Exact image continuation requires the
unchanged dependency environment used by the local tests. Plan 10.36 records
33 passed GPU TGV NP=1/2 same-topology short checks, including steps/time,
12 versus 5+7, render-only, no-advance, override and corrupt-state rejection.
JPEG pixels, VTK geometry and saved flow/statistics match continuous execution.
This does not extend rendering to CPU, CURVE, AIR5 or changed decompositions.
Repartition is rejected when either saved or current configuration enables
rendering, including an override that disables previously enabled rendering.
The old paired rules above remain unchanged.

Resource observation begins before native restoration allocates statistics or
postprocessing buffers. Due render frames still download the full local
physical flow sample; private CPU halo/gradient and geometry work remain.
Native render-only finalization reads the live completed-step state instead
of retaining the previous frame's full sample. This is not an all-GPU analysis
path and does not resolve the separate 256-cubed long-sequence issue below.

## Resource Observation

### Separate 256 Cubed Demonstration

`run_tgv256_demo.py --output <new-directory> --steps 2` prepares the two-step
resource check. After it passes, use `--steps 100 --encode` for the approved
demonstration. This local runner uses two GPUs, topology 2,1,1, dt=1e-4,
statistics disabled, no conventional field/checkpoint files, and a frame after
each complete step (1..100). It writes JPEG/EPS Q_rs=0 surfaces and instantaneous
streamlines colored by speed in [0,1], then 20 fps H.264 MP4 files. Q=0 is the
rotation/strain balance surface, not a positive-Q vortex-core threshold.

The separate `tgv256_demo.py` preset leaves the IS3 acceptance preset unchanged.
Its resource limits are 6 GiB additional device memory per physical GPU, 16 GiB
additional host memory per node, and 2 GiB device reserve. Larger local extents
are admitted only for formal render-only runs, up to 256; statistics and paired
restart retain their previous size gates. This is not a production-scale IS3
certification. Rendering includes CPU-side geometry and streamline work even
though EGL uses the solver's bound GPU.

The initial display trial stalled in `ColorBy` automatic range rescaling when
different MPI ranks took different collective paths. The preset now uses the
acceptance pipeline's explicit scalar-coloring and fixed-range calls. The final
two-step test `tgv256_render_smoke_noio_20260930` passes with no field files.
Checkpoint cadence is 1000, outside this run, independently of frame cadence 1.
The runner also writes the native `datin/input.output` with checkpoint, volume
and slices disabled. As of 2026-10-04, the benchmark no-field-I/O flag overrides
these three native products even when configured enabled and due. CPU/GPU NP=2
regressions cover that override. It does not disable in-situ images or statistics
and does not broaden the benchmark's admitted physical cases.

The attempted 100-step run was managed by local user service
`astr-tgv256-100-render-20260930.service`, with output under
`tests/gpu_validation/out/tgv256_render100_20260930`. Only a successful final
`summary.json` and completed MP4 files indicate completion; service startup is
not the acceptance result. The runner uses local dependency paths, which must
be edited for another machine. FFmpeg is supplied through the isolated
`astr_dependencies/python-video` imageio-ffmpeg installation, not the system.

Outcome: the service failed after 48 completed frames when the observed node
host increment reached 16.052 GiB, above the approved 16 GiB limit. Both image
sequences contain 48 frames, but there are no MP4 files or successful summary.
GPU increments also grew to 5.703/4.584 GiB. The source of that historical growth
is not yet established. Both TGV scripts now reuse their filters, views,
representations and color objects; finalization deletes only objects they own.
Matched frozen-field tests retain identical JPEG pixels and VTK geometry, but
proxy-count stability alone does not establish the cause of the old failure.
Do not raise budgets or describe the two-step smoke as proof of long-sequence
stability. Plan section 5.1 records the bounded tests and remaining gates.

`demo_rankN.json` records per-frame proxy counts, geometry sizes, extraction,
render, screenshot/JPEG and EPS times. Native logs separately report transfer
synchronization/download, ownership reconciliation, halo, derivatives,
bridge copies and Catalyst execution. Frame capture, derivative and render
times are inclusive: do not add their nested substage measurements twice.
Download includes CUDA Fortran section-copy handling; screenshot timing includes
capture and JPEG encoding, not an independently measured encoder-only cost.

The frozen-field harness is
`tests/gpu_validation/insitu_pipeline_lifecycle_probe.py`, run with the matching
MPI launcher and `pvbatch`. `--structured-grid --full-fields` tests explicit
coordinates and 25 fields without advancing ASTR. This is a resource probe,
not a numerical or physical validation. Its device observations cover the whole
selected GPU, so run it without concurrent GPU workloads. Phase-boundary
sampling cannot bound all transient peaks.

The runner records incomplete/corrupt rank receipts on failure and never marks
that run successful. With `--encode`, it decodes both MP4s, checks frame count,
20 fps, duration and dimensions, and reports first/last-frame RGB differences
from their source JPEGs. H.264 is lossy; those differences are reported rather
than presented as bitwise image equivalence.

### Bounded bc41 Wall Products

The optional `products='channel_walls'` profile supplies both physical y walls,
not a 3-D flow mirror. Use `derivative_backend='cpu'` and
`render=t`; rendering still uses the solver's GPU-matched EGL context. The
candidate admits internally generated dimensionless Cartesian channels up to
32 cells per direction, NP=1/2, bc `[1,1,41,41,1,1]`, explicit `643e` derivatives
and at least two local interior layers next to each wall. The actual acceptance
case is 16^3, four complete steps, dt=1e-3, with NP=1 and all NP=2 slab directions.
The code limit of 32 is not a completed larger-mesh validation.

The bridge receives `wall_pressure`, `wall_shear_x`, `wall_heat_into_gas` and
`wall_normal_y`. Normals point into the fluid; shear is signed in +x, and heat
is positive from the wall into the gas. Sutherland viscosity and the existing
explicit boundary closure are evaluated from the completed conservative state.
The GPU packs only five conservative components on three planes per wall;
host diagnostics follow that compact download. This is not GPU-resident wall
diagnostic computation. Periodic x/z duplicate nodes are reconciled in private
buffers; no solver state is changed. The surface cells are disjoint between
ranks and never connect the two walls.

The preset writes one JPEG/EPS/VTK product per wall scalar. Colors are fixed
at pressure [0,12], signed shear [-0.02,0.02] and heat [-0.2,0.2]; these bounds
do not clip stored field values. The camera uses the globally reduced physical
bounds. Image/geometry continuation from completed-step checkpoints has been
checked at the same backend/topology. Optional complete-step velocity statistics
also pass CPU/GPU and exact restart checks on the stretched channel. Upper-wall
nodes are retained and spatial weights use actual y coordinates; GPU pointwise
accumulation stays on device. CPU statistics require rendering disabled.
With `statistics=t`, pressure, signed shear and heat also acquire time means,
variances and RMS. Their window uses the same clipped endpoint weights as the
velocity statistics. AIR5, CURVE, arbitrary rank counts and statistics/render
repartition are not admitted by this profile.

Example selection inside an otherwise complete `&insitu_run` configuration:

```fortran
products='channel_walls', derivative_backend='cpu', statistics=f, render=t,
```

Bounded tests are `test_insitu_channel_walls.py` and
`test_insitu_channel_wall_render.py`. The former also exercises CPU diagnostics
and Compute Sanitizer; the latter checks actual EGL products, surface area,
same-phase fields, resource limits and exact continuation.

### Bounded AIR5 Bottom Wall

Use `products='air5_walls', derivative_backend='cpu', statistics=f, render=t`
for the internally generated dimensional Cartesian noncatalytic AIR5 HBL,
bc `[11,50,41,51,1,1]`, explicit 643e, <=32 cells/axis and NP=1/2.
Actual acceptance is 16^3, four coupled steps, NP=1 and all NP=2 slabs.
The GPU downloads only three wall-adjacent planes of 11 conserved components;
existing host AIR5 EOS/transport routines supply 18 SI fields. The native EGL
context remains mapped to the solver GPU. Ten JPEG/EPS/VTK presets show pressure,
T/Tv, five species fractions, signed +x shear and heat positive into the gas.
Per-field units and completed-step identity are retained in geometry metadata.
The empty upper y rank participates, and exact restart preserves images/state.
This is not GPU-resident diagnostics or general 3-D AIR5 visualization. Those
paths and CURVE remain unadmitted. Wall-only cumulative statistics are available
as described below. See
`test_insitu_air5_walls.py` and `test_insitu_air5_wall_render.py`; render tests
explicitly use the approved 256 MiB directory cap, other driver defaults stay
at 64 MiB. Fixed visualization ranges do not clip stored field values.

### Wall Scalar Statistics And Restart

Set `statistics=t` with an explicit increasing `statistics_window` and either
`products='channel_walls'` or `products='air5_walls'`. Sampling occurs at each
completed step, including an initial endpoint, independently of image cadence.
CPU statistics require `render=f`; GPU may combine wall statistics and rendering.
These are per-position Reynolds means, variances and RMS, not Favre scalar
statistics. Channel velocity/Favre/covariance/density-stress statistics remain
available in addition to its three wall scalars. AIR5 defaults to the 18
wall fields; volumetric statistics require `air5_volume_statistics=t`.
Images depict instantaneous fields by default. `wall_mean_render=t` adds
the bounded mean-wall products described below.

GPU wall diagnostics use the compact three-plane D2H path, then upload the
selected compact wall scalars to resident FP64 accumulators. For a single
16^3 AIR5 rank this is 76,296 bytes D2H and 41,616 bytes H2D per endpoint.
The empty upper y rank transfers neither. Accumulator downloads occur at
checkpoint/final export, not as full flow downloads at every sample.

Raw restart state is part of `statistics.h5`: AIR5 uses 204 components at the
file root; channel appends 34 components in group `wall_statistics` beside
existing volumetric velocity state. Only physical wall planes contain data;
the current checkpoint container is sparse in a full 3-D layout, not a compact
on-disk surface format. Metadata retains field/profile/backend identity,
sample count, window and coverage clock. Same-backend/topology continuation
is exact; repartition is refused. No extra loose restart files are added.
AIR5 root metadata has 12 integers, including volume and separation selection
at entries 11 and 12. These selections must match on restart.

Final scalar export is `sample.wall_statistics.stepXXXXXXXX.rankXXXXXXXX.bin`:

| Part | Little-Endian Layout |
|---|---|
| Magic | 8 bytes, `ASTRWS01` |
| Header | 11 int32: version=1, completed step, rank, nx, nz, number of local walls, global i/j/k starts, field count, profile=5/6 |
| Clocks | 4 FP64: previous sample time, window start/end, covered duration |
| Count | 1 int64, endpoint sample count |
| Coordinates | `(nx,nz,walls,3)` FP64, Fortran order |
| Moments | `(nx,nz,walls,3*fields)` FP64, blocks mean / variance / RMS |
| Ownership | `(nx,nz,walls)` int32, 0 or 1 |

Field order is pressure/shear/heat for channel and the 18-field AIR5 order in
`test_insitu_air5_wall_render.py::FIELDS` for AIR5. An empty rank has zero array
payload but retains clocks/counts. Zero covered duration exports zero moments
with duration zero; it does not represent a physical time average.

Bounded checks are in `test_insitu_wall_scalar_statistics.py`: NP=1 and NP=2
x/y/z, independent frozen-endpoint integration, exact checkpoint continuation,
solver isolation, joint AIR5 rendering/restart, empty ranks, memcheck and
corrupt/budget rejection. AIR5 cross-backend means use fixed input-derived
scales, variances/RMS squared use squared scales (<=2e-10), with SI/raw-RMS
differences reported. Each sampler's own checkpoint phase retains absolute
<=2e-10; exact restart is not tolerance-based. These short hot-gas tests do not
establish turbulent convergence or production SBLI validity.

### Optional AIR5 Volume Statistics

Set `air5_volume_statistics=t` with enabled AIR5 wall statistics. It retains
T/Tv/five species means, variances and RMS, plus the existing 41 velocity
Reynolds/Favre/density-stress results. Set `air5_volume_reduction=t` separately
for whole-domain geometric Reynolds velocity diagnostics. Both flags default
false. Region averaging is not a claim of statistical homogeneity.

The bounded Cartesian admission is the same dimensional, noncatalytic HBL
fixture (<=32, NP=1/2, tested 16^3). Open x/y physical upper nodes are owned
once; periodic z duplicates are excluded. Tensor trapezoid weights come from
actual coordinates. Device state remains resident; 8 FP64 reduction values
(64 bytes) cross D2H per sample. Restart/final export downloads state arrays.

`statistics.h5/air5_volume_statistics` contains 136 packed components and
44 integer metadata entries. Four groups reuse the 34-component state:
velocity with physical density; `(T,Tv,Y_N2)`, `(Y_O2,Y_N,Y_O)` and `(Y_NO,0,0)`
with unit density. The last two dummy scalars are not exported physical fields.
Metadata retains step/time/window/backend/count/coverage/volume, reduction
selection and the 34-component regional signal. AIR5 root wall metadata also
retains the volume-selection flag. Selection mismatches are rejected on restore.

`sample.air5_statistics.stepXXXXXXXX.rankXXXXXXXX.bin`, little-endian:

| Part | Layout |
|---|---|
| Magic | 8 bytes, `ASTRAS01` |
| Header | 10 int32: version=1, step, rank, nx/ny/nz owned counts, global i/j/k starts, regional-reduction flag |
| Clocks | 5 FP64: previous time, window start/end, covered duration, global physical volume |
| Count | int64 endpoint sample count |
| Regional State | 34 FP64; inactive when reduction flag is zero |
| Coordinates | `(nx,ny,nz,3)` FP64, Fortran order |
| Measures | `(nx,ny,nz)` FP64 physical nodal volume |
| Moments | `(nx,ny,nz,62)` FP64: existing 41 velocity results, 7 scalar means, 7 variances, 7 RMS |

Scalar order is T, Tv, Y_N2, Y_O2, Y_N, Y_O, Y_NO. Optional root-only
`sample.air5_volume_rms.stepXXXXXXXX.csv` names both local temporal-variance
volume mean/RMS and temporal variance/RMS of the regional mean velocity signal.
Metadata records whole physical domain, Reynolds average, window, duration,
step and volume. Neither class establishes turbulent convergence.

`test_insitu_air5_volume_statistics.py` verifies independent 0.65/0.35 clipped
endpoint weights, unique physical volume, cache phase, both RMS classes,
exact continuation, isolation, selection mismatch and corrupt state rejection,
and two-rank memcheck. Volume statistics do not replace wall diagnostics.

### Optional Mean-Wall Images

`wall_mean_render=t` requires enabled wall statistics and GPU/EGL rendering.
It defaults false and adds `mean_` versions of the existing three channel or
ten AIR5 image presets once covered duration is positive. Cameras and fixed
color scales are unchanged. It does not add volumetric mean-field images.

The wall geometry contains `mean_`, `variance_`, and `rms_` arrays for each
of the three channel or eighteen AIR5 statistical fields, plus
`statistics_duration`, `statistics_window_start`, and `statistics_window_end`.
Window, duration, Reynolds averaging identity and units also appear in the
geometry/JSON metadata. Variance units are squared. Mean products reuse these
FP64 moments, not re-integrated exported snapshots. A private copy completes
shared x and periodic-z endpoints; upper-y empty ranks remain valid empty
participants. JPEG/EPS/VTK products are named `mean_<preset>.stepXXXXXXXX.*`.
No covered duration means no mean image, not an artificial zero mean.

### Optional Bottom-Wall Separation Record

`wall_separation=t` requires enabled AIR5 wall statistics and defaults false.
For the current Cartesian bottom wall, the tangent is +x. At each completed
sample, signed shear is geometrically averaged using actual spanwise-z nodal
lengths and unique ownership. No x-periodic wrap or equal-node weighting is
introduced. The root writes `sample.wall_separation.stepXXXXXXXX.csv`, with
profile rows and eligible crossing rows, step/time, coordinates, pair ID,
classification and status; comments retain reference inlet speed, span length
and number of complete pairs.

Between nonzero opposite-sign neighbors, + to - is separation and - to + is
reattachment; the zero is linearly interpolated. An exactly zero run, including
a single exact zero, stays an interval and interrupts unique pairing. Unpaired
crossings are not complete bubbles. The interpolation uses scaled magnitudes
to avoid overflowing the shear sum. Nonfinite data or nonmonotone x is an error.

The present zero-inflow hot-gas fixture is labeled
`not_applicable_no_positive_inflow`; it supplies sampling, geometry, zero-value
and restart evidence, not a physical no-separation finding. The root algorithm
is validated by independent synthetic sequences. No positive-inflow production
SBLI or statistical stationarity is claimed.

### Bounded Static Curvilinear Products (IS6)

The local candidate uses 32^3, FP64, explicit 643e, NP=1 or NP=2 x/y/z.
Use `tests/gpu_validation/generate_curvilinear_tgv_grid.py` with amplitude 0.15
and `periodic` for periodic TGV, or `y-wavy` for the existing TGV/bc41
manufactured wall diagnostic. The latter has x/z periodic, two y walls and
physical mapping x=xi, y=eta+0.15*sin(xi)*sin(zeta), z=zeta. It is not a
developed channel-flow physics benchmark.

Periodic products retain all nine inverse metrics in physical gradients,
Q and divergence. GPU private derivatives remain opt-in. Curved walls require
`products='channel_walls', derivative_backend='cpu'`; their GPU solver exports
only three compact wall layers for the CPU diagnostic. Optional wall statistics
and `wall_mean_render=t` use the same completed-step samples and checkpoint.
The inward normal comes from geometry, and the shear tangent is the normalized
projection of global +x onto the wall tangent plane. `wall_normal_y` now stores
the y component, not an integer side marker. Heat is positive into the gas.
The existing second-order three-layer wall closure is retained; tangential
derivatives use the periodic sixth-order stencil. Do not reuse the old
Cartesian +/-1 assertion for this candidate.

Each trilinear hex has positive Gauss Jacobians and a 2x2x2 Gauss volume,
equally lumped to its eight nodes. MPI cell ownership is unique; shared and
periodic endpoint masses are assembled on the node owner, including local
periodic wrap when a direction has just one rank. The static device weights
are uploaded once and reused. Bilinear wall quads use 2x2 Gauss areas with
equal four-corner mass. Root log `ASTR_INSITU_CURVE_WALL wall=... area=...
area_means=...` reports instantaneous area-weighted pressure, signed tangent
shear and heat, not a homogeneous-direction or time average. Pointwise wall
time moments remain pointwise.

The original grid is frozen as `resources/grid.h5`; an exact same-topology
restart needs the COMPLETE batch and its resource tree, not the original
input path. Geometry metadata retains physical coordinates and complete-step
identity; nondimensional TGV/wall fields explicitly say `dimensionless`.
The constant physical-velocity streamline oracle is postprocessing-only and
checks crossing x=pi with <=2e-10 endpoint/straightness error. Actual TGV
streamlines are geometry/field/restart checks, not exact trajectories.

Reproduce the bounded gates from the repository root after CPU/CUDA Catalyst
builds (`insitu_geometry_probe` is an explicit CPU test target):

```bash
cmake --build build_insitu_check --target astr insitu_geometry_probe -j 4
cmake --build build_insitu_gpu --target astr -j 4
python3 -m pytest -q tests/gpu_validation/test_insitu_geometry.py \
  tests/gpu_validation/test_insitu_curve_derivatives.py \
  tests/gpu_validation/test_insitu_curve_statistics.py \
  tests/gpu_validation/test_insitu_curve_walls.py -x
```

The test executable/MPI/Catalyst paths can be overridden through
`ASTR_OUTPUT_CPU_EXE`, `ASTR_OUTPUT_RUNTIME_EXE`, `ASTR_OUTPUT_MPIEXEC`,
`ASTR_CATALYST_LIBRARY` and `ASTR_INSITU_GEOMETRY_PROBE`.
Fluid tests are 32^3 and four steps; 16/32/64 refinements are pure geometry
quadrature, not fluid simulations. Per-case directory cap is 256 MiB, local
host/device budgets are 4 GiB/node and 2 GiB/physical GPU with >=1 GiB free.
These are acceptance limits, not production defaults. CURVE NP=4, arbitrary
wall geometry or chemistry, moving grids and render repartition remain denied.

### Observation Contract

Native rendering records `resources.rankNNNNNNNN.csv`. Its baseline precedes
sample/statistics allocation. Node RSS is the sum of participating ranks' RSS
(shared pages may therefore be counted more than once). Device memory sums this
job's rank PIDs on the current physical GPU, deduplicating compute/graphics
records per PID; ranks sharing a GPU use one common baseline. Other jobs are
excluded from the job increment but affect available device memory.

Checks run after renderer initialization, before/after every frame, before/after
renderer finalization, after final statistics export, and after session buffers
are released. Logs include
observed totals, increments and peaks. Exceeding the explicit node/device budget
or losing the configured device reserve reports an error and aborts MPI. There
is no automatic fallback or reduction in output quality. A failure after a frame
may leave its products on disk; it does not indicate successful session completion.

These phase-boundary observations supplement controlled allocation admission.
Independent external sampling is retained in validation; neither mechanism
captures every transient third-party allocation or guarantees arbitrary OOM
recovery. Missing observations and pre-existing resource logs are rejected.

## IS7 Bounded First-Version Receipt

The capability matrix, dependency conditions, phase boundaries and measured
times are in `documents/ASTR_INSITU_IS7_ACCEPTANCE.md`. The matched local
32^3 TGV NP=1/2 receipt is `tests/gpu_validation/out/insitu_is7_final_20261005/report.json`;
25 affected regression gates passed in `insitu_is7_regression_20261005.xml`.
This closes IS7's bounded record, not arbitrary production capacity or IS8.

`ASTR_INSITU_TIMING=1` records local stage durations without new phase barriers.
The completed window includes four RK steps, requested products and unchanged
native checkpoint schedules. Aggregate by summing each rank's own durations,
then taking the maximum rank. Do not sum nested timers or phase maxima.
First-frame Catalyst/view initialization is inside that window; the separate
solver initialization excludes pre-MPI device binding and MPI initialization.
Launcher wall time includes both and the external sampler when selected.
Compute/pack/sync and diagnostic D2H are measured separately; flow section copies,
screenshots and VTK writers retain inclusive labels. Timed/untimed and paired
restart field/statistics/image checks passed; the observer is not an asynchronous
execution or performance optimization mode.

An independent launcher and frozen inputs, without validation-module imports:

```bash
python3 scripts/insitu/start_tgv_acceptance.py --output /new/empty/case \
  --executable /path/to/cuda-catalyst/astr --mpiexec /path/to/mpiexec \
  --library /path/to/paraview/lib/catalyst --np 2
```

Use `--prepare-only` to inspect generated files; `--off` changes only the new
in-situ enable flag. Existing directories are never overwritten. Templates are
under `presets/tgv32/`: 32^3, dt=1e-3, four completed steps (controller's inclusive
zero-based maxstep=3), statistics window 0.003..0.004, frames 2/4, all products,
checkpoint interval 1/keep 2, local host/device/free budgets 4/2/1 GiB.
Keep the positional primary/controller input layout, including header and blank
lines. The launcher prepares separate empty output roots. The independent NP=2
four-step launch passed with all ten product frames and exact authority against
the matched on case; receipt: `insitu_is7_standalone_final_20261005/`.
These are acceptance presets, not production defaults. Configure runtime libraries
matching the build first; ParaView 6.1.1 needs the recorded FP64 streamline patch.

The native launcher uses phase-boundary resource checks. Independent external
20 ms process-tree/NVML sampling and per-case 256 MiB admission remain in the
validation driver. Neither mechanism catches all transient third-party OOMs.
No remote job or new production-scale benchmark was launched for IS7.

## IS8 Bounded Device Products

`ASTR_WITH_INSITU_DEVICE=ON` additionally requires root CUDA/Catalyst builds
and the private FP64/CUDA/MPI Viskores components. It defaults OFF. Build steps,
patches and the preserved EGL installation are described in
[patches/README.md](patches/README.md). The complete local receipt is
`documents/ASTR_INSITU_IS8_ACCEPTANCE.md`.

Inside `&insitu_run`, explicitly add:

```fortran
 derivative_backend='gpu',processing_backend='device',products='all',
 postprocess_transport='pinned',
```

Alternatively explicitly select `'device-aware'`. The selected transport is
collectively checked and independent of the solver's halo transport. There is
no default device transport and no fallback. Use the existing positive resource
budgets, rendering/library/pipeline settings and native output configuration.
Device processing requires rendering. Missing/unsupported configuration fails
before products are accepted.

Admission is internally generated 32³ periodic Cartesian FP64 TGV,
643e/643e, NP=1 or NP=2 x/y/z. The existing frozen slice/Q/instantaneous and
Reynolds/Favre streamline preset is retained. Samples, shared nodes, halos,
gradients/Q and interpolated product extraction stay on the GPU. Only declared
face buffers (pinned mode), final compact geometry and small state need the
host. EGL rendering is on the selected GPU, while image/VTK encoding and
publication are host work. Library workspaces may migrate managed pages;
this is not zero total DTOH. Checkpoint/statistics exports are separate.

For a fresh empty local test directory, with matching runtime libraries loaded:

```bash
python3 scripts/insitu/start_tgv_acceptance.py --output /new/empty/device-case \
  --executable /path/to/device-enabled/astr --mpiexec /path/to/matching/mpiexec \
  --library /path/to/paraview/lib/catalyst --np 2 \
  --processing-backend device --postprocess-transport pinned
```

Change only `--postprocess-transport` to `device-aware` for the independently
qualified device-buffer mode. This frozen local launcher supplies tested Open
MPI ob1/TCP or UCX CUDA settings, keeps solver halos pinned, and performs four
completed steps with ten scheduled product frames. Other MPI vendors/transports
or hardware must be requalified, not assumed to match these settings.
`--prepare-only` generates files without starting the solver; a preexisting
directory is refused. The launcher imports no validation module. Both modes'
actual NP=2 launch, products and final state/statistics match the controlled run.

Checkpoint continuation retains the saved transport identity. Changing only
that choice needs `restart_output='override'` in the native output configuration;
statistics and output clocks are preserved while transient slots are rebuilt.
No CUDA pointers, requests or pinned handles are serialized. Same-backend
same-topology continuation is exact; render repartition remains rejected.
Terminal streamline metadata retain raw VTK status and untravelled arc for
the explicitly approved sub-minimum remainder condition. Other failures abort.
The endpoint is never extrapolated.

`ASTR_INSITU_TIMING=1` also records device sample/means, snapshot/halo/Q,
slice/contour extraction, compact read, each streamline kind and inclusive
consumer time. `ASTR_INSITU_FACE_USAGE` reports fixed capacities, allocation
generation and cumulative D2H/MPI/H2D/local/sync times. MPI time includes error
consensus; consumer time includes Catalyst/rendering. These nested durations
are not additive. Matched short-window timing and actual per-object Nsight
captures are recorded in the acceptance document, not production speed claims.

CURVE, walls, AIR5, larger grids, NP>2, asynchronous execution and per-product
variable clocks remain separate work; the prior host support is unchanged.
