# Native In-Situ TGV Preset

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
the three Python pipeline files under `share/astr/insitu`.

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
The benchmark no-field-I/O flag alone did not suppress due GPU checkpoints in
the earlier trial; this demonstration does not claim that behavior is repaired.

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
GPU increments also grew to 5.703/4.584 GiB. The source of growth is not yet
identified. Do not raise budgets or describe the two-step smoke as proof of
long-sequence stability. Plan section 5.1 now prioritizes lifecycle diagnosis,
stage timing and I/O-contract checks before a fresh 100-step run and IS4 expansion.

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
