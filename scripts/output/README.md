# Completed-Step Checkpoint Export

This candidate tool reads new ASTR schema-2 checkpoint bundles with role 2
(five conservative variables plus six cached primitive fields) or role 7
(AIR5 q11, carry11 and twelve cached fields), and their shared geometry.
It does not read legacy checkpoints, advance
the solver, reconstruct the EOS, apply boundary conditions, or accumulate
statistics. Ordinary solver output has no Catalyst dependency.

Requirements for offline export: Python, NumPy and h5py. For graphical or
scripted readback, use a ParaView build with **XDMF3** support. The local
Catalyst-rendering-only edition lacks file readers and is not sufficient.
The bounded readback gate passed with ParaView 6.0.1 `Xdmf3ReaderS`; select
that reader rather than the older XDMF2 reader. The latter failed on the
two-dimensional vector path and is not certified here.

The root CMake installation includes this README, all three offline scripts,
and the basic/derived TGV and SI AIR5 derived configuration examples.
These seven files form the `OutputTools` component. To install just these files
from an already configured build, without installing the solver or examples:

```bash
cmake --install /path/to/build --prefix /path/to/install --component OutputTools
```

They normally reside in `share/astr/output`; a configured
`CMAKE_INSTALL_DATADIR` can change that directory. Keep the scripts together:
`combine_series.py` imports `repair_series.py` from its own directory.
Run them with Python using their installed absolute paths. They need no CUDA,
Catalyst or live solver process. CPU and GPU root-build install rules have
passed isolated installation and off-checkout CLI import checks (plan 10.46).

```bash
python3 scripts/output/export_checkpoint.py \
  /path/to/run/checkpoints/step000000000012 /path/to/new-export \
  --units dimensionless --buffer-bytes 1048576 \
  --slice i:8 --slice j:0 --slice k:8
```

Use `--slices-only` to omit the volume. Repeated `--slice` selects fixed global
node indices, including endpoints; duplicate indices are removed and out-of-grid
indices are rejected before creating output. The output directory must be new
and outside the source run. Keep the source immutable during export and copy
its shared resources when moving it. Validation rejects incomplete batches,
CRC-64 mismatches, symlink/hardlink members, unsupported roles/phase/types,
external HDF5 links/storage, and nonfinite exported physical fields.

`--units dimensionless|si` is explicit, checked against the frozen primary
input's `nondimen` flag, and performs no conversion. SI labeling additionally
requires the original dimensional case to use SI units. Dimensionless data
retain their original normalization; no unspecified reference scale is applied.
The source input remains the normalization reference.

Products are `volume.h5`/`volume.xdmf` and, when requested,
`slices.h5`/`slices.xdmf`. Each has FP64 nodal density, velocity_x/y/z,
pressure and temperature, plus a materialized velocity vector and XYZ
coordinates. Velocity is also stored as three scalars for direct Python use;
the vector adds three scalar components on disk but no additional source-field
read or simultaneous full-volume buffer. HDF5 axis order is k,j,i (i fastest),
with vector components last. A fixed-index slice drops its selected axis and
preserves the relative order of the remaining axes. Coordinates remain 3D even
for warped two-dimensional surfaces.

AIR5 role 7 adds cached vibrational temperature and five mass fractions named
`mass_fraction_N2`, `mass_fraction_O2`, `mass_fraction_N`, `mass_fraction_O`,
`mass_fraction_NO`, in the native AIR5 model order. Mass fractions remain
dimensionless even when `units=si`; temperatures are in kelvin. All twelve
fields are copied from checkpoint caches, not recomputed from q or q-carry.
The tool does not export carry or a restart representation, infer a new
chemical mechanism, normalize fractions, or clip trace values. AIR5 export
requires dimensional SI input and valid role-7 version/compensation metadata.

```bash
python3 scripts/output/export_checkpoint.py \
  /path/to/air5-run/checkpoints/step000000000012 /path/to/new-air5-export \
  --units si --slices-only --slice j:0 --slice k:8
```

Offline XDMF uses a temporal collection with the actual completed-step time,
not a target schedule time. Each offline-export directory contains one time;
the separate native interface below provides live schedules and indexes.
`COMPLETE.json` is published only after data/XML and completion metadata close;
a failed export may leave partial files without that marker. Products are
independent copies, so subsequent checkpoint retention does not remove them.
Exports are for postprocessing, not restart.

The report records field-read bytes, file bytes, the configured tile capacity
and peak controlled FP64-array/finite-mask bytes. CRC validation streams all
bundle members with a separate 64 KiB buffer. Slice-only field extraction reads
only selected planes, **but integrity validation still scans complete source
files**. This does not prove a zero-full-file-read path or a bounded HDF5/
ParaView memory peak. No GPU is used by this offline tool. The native gates
below, not this export, provide evidence for selected-plane GPU transfers and
independent live schedules.

```python
from paraview.simple import Xdmf3ReaderS
reader = Xdmf3ReaderS(FileName=['/path/to/new-export/volume.xdmf'])
reader.UpdatePipeline()
```

Targeted checks: `test_checkpoint_export.py` uses an asymmetric 9x7x5 grid;
`run_checkpoint_export_validation.py` checks real saved CPU/GPU fields, slice
values, vector components, coordinates and time in ParaView without rendering.
The readback references are restricted to 2 MiB and each real test directory to
64 MiB. This is a local correctness gate, not an I/O-performance claim.

AIR5 evidence: `tests/gpu_validation/out/or5_air5_export_20261001.xml`
(25 asymmetric-grid checks) and `out/or5_air5_paraview_20261001/summary.json`
(four real CPU/GPU HBL/SBLI checkpoints, ParaView 6.0.1). The latter uses
`run_checkpoint_export_validation.py --units si`; all twelve cached fields,
coordinates, vectors, slice-only products and time match the source exactly.

## Optional Native Archives

The new runtime admits basic live volume/slice output for registered RK3
checkpoint cases: periodic TGV, bc41 channel, static/dynamic CURVE flatplate,
and fixed dimensional AIR5 HBL/SBLI, on CPU or GPU. Existing case/boundary/
model admission checks still apply; this is not arbitrary-case support.
The first delivery has passed its bounded local acceptance matrix (redesign
plan 10.77-10.78, user-approved scope A), not arbitrary production-scale
qualification. Derivative fields have the separate bounded admission below.
The offline exporter above remains a separate interface.
Ordinary native output requires MPI/HDF5, not Catalyst or ParaView.

`&output device_reserve_bytes` is an optional nonnegative int64 resource
setting, separate from `device_budget_bytes` (controlled workspace limit).
For basic GPU field packing it defaults to zero, with no free-memory queries.
A positive value checks `cudaMemGetInfo` before packing allocation and after
release, and requires the planned allocation to leave that many bytes free.
Each rank records `ASTR_OUTPUT_DEVICE_RESERVE` measurements. Query failure or
insufficient free memory aborts collectively before publishing that frame;
there is no resolution/backend fallback. An empty segment ledger or `.tmp`
directory can remain and is not a published frame.
Derived allocation retains its existing minimum 1 GiB reserve; a larger
explicit setting increases that minimum. CPU output performs no CUDA query.
The option neither changes the checkpoint schema nor output schedules and
can be changed on restore without `restart_output='override'`. It covers
GPU frame packing, not all checkpoint allocations or the independently
configured `&insitu_run` rendering/statistics budgets. These are device-wide
point measurements, not job-owned capacity or an intra-stage/OOM guarantee.
Local basic-output acceptance uses `device_reserve_bytes=1073741824` for
16-cubed TGV NP=1/2 and registered NP=2 channel x, static CURVE x,
dynamic CURVE z, AIR5 HBL z and AIR5 SBLI x. This is not a production
capacity recommendation; plan 10.59 records the added seven short checks.

The shared writer and offline index tools support explicit derivative selections.
`ASTR_DERIVED_FRAME_1` records add
fourteen selection slots to the basic FRAME declaration, matching the HDF5
`derived_layout` attribute. Scalar datasets declare definitions, physical
coordinate space and units. Gradient names use velocity component first and
physical derivative axis second; `velocity_gradient_yx` is du_y/dx. Q is the
full-strain rotation/strain `Q_rs`, not the second principal invariant.
The repair/parent-chain tools reject changing layouts and inconsistent field
declarations. Real native derivatives admit 16-cubed-cell, five-variable,
dimensionless FP64 cases with viscosity and 643e differentiation, NP=1/2:
generated periodic TGV; bc41 channel with an unsplit wall-normal direction;
registered static/dynamic CURVE flatplate with an unsplit wall-normal direction.
TGV/channel require 643e convection and filtering; flatplate uses 543e convection.
The nonperiodic matrix covers NP=1 and NP=2 x/z for channel/static CURVE, and
NP=2 z for dynamic inflow (plan 10.60). Fixed AIR5 derivatives separately admit
generated 16-cubed SI HBL NP=1/2 z and SBLI NP=1/2 x, with eleven conservative
variables, five species, two temperatures, 643e/643e, viscosity/filtering and
bc=[11,50,41,51,1,1]. The wall-normal direction remains complete. Gradients,
vorticity and divergence use s^-1; Q_rs uses s^-2. Tests use the fixed dimensional
scales in plan 10.61, not a dimensionless tolerance applied to raw SI values.
Other sizes or unregistered combinations remain rejected. GPU rendering and repartition have
independent admission rules.
`export_checkpoint.py` still exports cached basic fields only.

Within this gate, select `velocity_gradient=t` (nine components),
`vorticity=t` (three components), or `qcriterion=t` (Q_rs and divergence)
independently in `&volume` and `&slices`. Selection appends fields to `fields='basic'`;
all three together produce twenty nonreacting or twenty-six AIR5 scalar fields.
See `input.output.tgv.derived.example` and `input.output.air5.derived.example`;
the basic example stays unchanged.
The private complete-step velocity snapshot receives fresh communication halos.
Interior differences are sixth-order centered; physical-boundary closures match
the solver's second-order one-sided, second-order centered and fourth-order
centered formulas. Grid metrics map computational to physical derivatives.
Stale solver gradients are not exported, including at nonperiodic faces.
Only selected field tiles/planes cross from GPU to host. The private device
velocity neighbourhood and halo communication are still needed, so slice-only
derivatives are not a zero-copy or single-layer-storage path. Controlled budgets
include private velocity, full reusable communication buffers and packing;
the GPU allocation guard additionally preserves 1 GiB of free device memory.
Changing selection on restart requires `restart_output='override'`; combining
segments with different derivative layouts is rejected.

`out/or5_real_derived_runtime_20261001.xml` records 21 passed real short gates:
CPU/GPU NP=1 and NP=2 x/y/z, independent same-phase discrete references,
exact 12 versus 5+7 continuation, output-switch invariance, actual HDF/ParaView
readback, layout/parent handling and rejection. Reported final-state reference
and CPU/GPU maxima are 4.00e-15 and 1.01e-14. This is not analytic truncation-error,
turbulence-physics or production-scale I/O validation.

Plan 10.45 adds joint lifecycle checks within this same admission. CPU/GPU
NP=2 x/z use variable timesteps, independent physical-time schedules and
Q/divergence output, with exact 12 versus 5+7 continuation and actual sequence
readback. Their slice-only GPU frames download 110976 selected-field bytes
for two frames, three 17-by-17 planes and eight columns. All-fourteen-field
NP=2 z copies also survive relocating the complete resource tree, same-root
continuation, `keep=1` rotation, stopped-segment index repair and parent-chain
readback. The selected source checkpoint remains protected even with `keep=1`;
this is not a promise of exactly one directory while a source is protected.
Changing the checkpoint interval requires explicit `restart_output='override'`,
just as changing the field schedules does. This override does not change q,
cached fields or accumulated statistics. These are bounded correctness gates,
not production I/O timings or wider derivative/repartition admission.

`input.output.tgv.example` is a **16x16x16-cell short-test** configuration,
not a production output-frequency recommendation. Copy it to `datin/input.output`,
prepare `outdat/new` as a new directory and launch the normal
`astr run datin/input.tgv` command. Runtime `usegpu` still selects CPU or GPU.
Completed-step output is now the only normal field/checkpoint entry. The default
configuration file is mandatory; `ASTR_OUTPUT_CONFIG` is an optional path
override, not an activation switch. Missing/invalid files fail without fallback.
Legacy controller `lwsequ/lwslic` flags are ignored with a notice. `feqchkpt`
still controls controller reload/CFL checks, not the new checkpoint schedule.
Set all three product `enabled` values false to run without these files;
provide the required valid configuration/budgets even for that choice.
Legacy checkpoints, q sidecars and old paired restart batches are not accepted.
Keep `lrestart=f` and select the new complete directory via `restore_directory`.

Volume and slices have independent `mode='steps'|'time'` schedules. Set only
the corresponding `interval_steps` or `interval_time`, not both. Initial/final
frames are independently selectable for each; the final checkpoint is mandatory
when checkpointing is enabled. Actual complete-step time is stored, not the
requested schedule threshold. Restore through `restore_directory` into a new
output directory, or reuse its own validated checkpoint root as below.
Unchanged schedules retain their origin/history and do
not duplicate the restored point. Changed schedules require
`restart_output='override'`; only changed products reset their schedule origin
to the restored clock. This does not alter numerical or statistics settings.

Native products have the layout:

```text
outdat/new/
  resources/geometry.h5, input.txt         # only when checkpointing is enabled
  checkpoints/step.../state.h5, archives.bin, ...
  fields/resources/data.h5                 # shared volume coordinates only
  fields/segment00000000/series.frames, series.xdmf
  fields/segment00000000/step.../data.h5, data.xdmf, FRAME, ...
  slices/resources/data.h5                 # shared selected-plane coordinates only
  slices/segment00000000/series.frames, series.xdmf
  slices/segment00000000/step.../data.h5, data.xdmf, FRAME, ...
```

Each frame is sealed and published after HDF5/XML/metadata close. All selected
planes at that clock share one HDF5 file with named plane groups. Fields and
slices are retained independently of checkpoint `keep=1|2`. With checkpointing
disabled, slice-only output creates no volume, checkpoint or shared checkpoint
geometry. On GPU
only selected field nodes are downloaded, without a full-volume/halo
copy or extra velocity-vector download. Coordinates use existing host geometry.
The log reports actual field/download bytes and controlled tile-array bytes;
these are not total process/HDF5 memory peaks.

Native frames no longer repeat their coordinates. Each product owns one
`resources/data.h5`, containing only coordinates and geometry metadata, not
flow fields, Jacobians, metrics or halos. Slice-only resources contain only
the selected planes, not the global volume. Every frame's RESOURCES record
binds the coordinate file by byte count/CRC-64; XDMF refers explicitly to it,
without an HDF5 external link. Move the whole product directory, including
resources, to preserve these relative references. Geometry is frozen per new
run; it is not reused across separate output roots. Resource verification
still scans the coordinate file and may itself cost I/O time.

`data.xdmf` reads one frame. Open segment00000000/series.xdmf with Xdmf3ReaderS
for all published times in that segment, including grouped slices. The series
uses actual completed-step times, not requested trigger thresholds. Both frame
files and product resources must remain present. Native CPU/GPU bounded tests
read every time in ParaView 6.0.1. Creating an index does not reread fields or
download another GPU array; metadata rewriting still grows with series length.

AIR5 live output uses twelve cached fields in the model order given above.
For 17^3 physical nodes, the basic-field GPU download is 471,648 bytes per
volume or 83,232 per three 17x17 planes; materializing the velocity vector
does not add another download. AIR5 carry and conservative variables remain
checkpoint data, not visualization fields. mean44 retains its original RK
sampling definition, independently of complete-step field-output times.
See plan 10.27 for the nonperiodic/CURVE/AIR5 short-test matrix. It does not
certify long physical convergence or every boundary/model combination.

### AIR5 GPU Conservation Diagnostic Continuation

For registered GPU AIR5 cases, `ASTR_AIR5_C4_CONSERVATION=on` is now compatible
with native checkpoints. CPU has no matching diagnostic and explicitly rejects
that choice. Do not change the switch across a continuation. Each GPU AIR5
checkpoint includes a sealed 168-byte `air5_conservation.bin`: complete-step
clock, activation/baseline flags, first/last sample IDs, count and eleven FP64
baseline integrals. Disabled or not-yet-initialized histories are canonical
zeros, not uninitialized saved memory.

This preserves the existing diagnostic: integrate stored q (not q-carry),
using cell-corner q*Jacobian averages. The baseline is taken on first entry to
the transport substep, and phase-1 records are after transport but before the
second chemistry half-step. The file is not a complete-coupled-step physical
conservation statement, and an open boundary can change the totals. Only the
checkpoint's state clock is the full completed-step clock.

The native path uses fixed-order local GPU reductions followed by the existing
MPI sum, so same-topology fresh processes reproduce the diagnostic exactly.
Its controlled device scratch is
`11*8*min(4096,ceil(local_cells/64))` bytes and counts against the shared output
device budget, together with selected statistics and frame packing. Each
diagnostic evaluation downloads 88 bytes, not a flow array. This is not a
cross-topology bitwise guarantee or a performance improvement claim. The previous
opt-in/legacy reduction selection is historical; the mandatory native output
entry now owns the registered continuation state and conservation history.

Each continuation creates an exclusive
`air5_conservation_from_step############.dat` in the output root. A resumed
file echoes phase 0 using the persisted original baseline, then writes only
new phase-1 samples. It never overwrites the earlier diagnostic. An existing
file for the same restore step, including same-root step-zero restart,
causes a clear failure; choose a fresh output root for that branch.
The 16-cubed HBL/SBLI, NP=1/2, scalar/full gates in plan 10.35 are short
continuation checks, not long-time or production AIR5 acceptance.

Indexes are updated after frame publication through complete temporary files
and same-directory replacements. Each replacement is individually atomic,
not a two-file transaction and not a power-loss durability guarantee. Failure
can leave an index behind its complete frames; automatic repair is unfinished.
Automatic cross-segment publication and the registered case-family derivatives
are now covered by the later sections below. Additional Catalyst case/topology
combinations are not admitted. Registered GPU TGV render continuation is
described below. The separate explicit
parent-chain builder below is available for stopped native segments.
The candidate binary schedule format is not a long-term compatibility promise.
See redesign plan 10.24-10.27 for the initial runtime/resource/sequence gates
and 10.64-10.65 for dimensional derivatives and automatic parent indexes.

### Bounded TGV Repartitioned Continuation

The native restart gate supports CPU-to-CPU and GPU-to-GPU NP=1<->2 x/y/z slabs,
and all six ordered NP=2 slab-direction changes, only for generated periodic
five-variable TGV (ninit=0, 643e/643e, filtering and diffusion enabled, no legacy
averages). Same-topology continuation remains exact; the approved 16-cubed,
dt=1e-3, continuous-12 versus 5+7 repartition gate uses 2e-10 for q, required
caches and statistics, with identical sample identity, duration and schedules.
Global physical-node records and pointwise statistics are redistributed; halo
values are rebuilt under the target partition rather than replayed from old
rank supplements. Regional/statistical histories are not reset.

This TGV gate does not admit walls, CURVE, AIR5, other numerical settings,
larger rank counts or backend migration. The separately bounded channel and
static-CURVE exceptions are described below. Repartition also rejects either saved or current
render enablement; disabling rendering with override cannot bypass the saved
renderer's unvalidated partition history. Keep the source checkpoint and its
shared resources immutable. See plan 10.38 for the matrix and readback evidence.

### Bounded Fixed-Force Channel Repartitioned Continuation

Plan 10.49 separately admits generated 16-cubed, nondimensional bc41 channel
with ninit=3, bctype=[1,1,41,41,1,1], 643e/643e, filtering and diffusion enabled,
ASTR_CHANNEL_FORCE_MODE=fixed and ASTR_CHANNEL_FORCE_FIXED=1.d-4. Within the
same backend it supports NP=1<->2 x/z slabs and NP=2 x<->z; both saved and
current y extents must span the whole domain. CPU registered mean44 histories
are migrated; compact or in-situ accumulated statistics are not admitted.
Rendering, feedback/frozen forcing, other walls, CURVE, AIR5 and backend
migration remain outside this gate.

CPU/GPU now refresh all physical primitive caches from filtered q before
boundary treatment and spatial operators on nonreacting five-variable explicit
paths. The frozen-field 12 versus 5+7 matrix uses the existing 2e-10 tolerance;
same-topology continuation stays exact. This changes the formerly stale-cache
numerical baseline: generate donors with the matching new executable rather
than reuse an old binary's checkpoints for exact continuation. This is a
short numerical gate, not long-time turbulent-channel validation.

### Bounded Static-CURVE x/z Repartitioned Continuation

Plans 10.52 and 10.54 separately admit the static profile flatplate configuration:
16-cubed, nondimensional bl/prof, ninit=0, bctype=[11,21,41,51,1,1],
543e/643e, MP7 physical-space reconstruction, no turbulence model, filtering
and diffusion enabled, no accumulated statistics or rendering. Same-backend
NP=1<->2 x/z and NP=2 x<->z are supported; both source and target y extents
must span the whole domain. The tested grid is the extruded warp_x=0.08/warp_y=0.04 mapping
with dt=1e-5, not an arbitrary three-dimensionally varying metric field.

Global physical q and required caches are read into the target partition;
communication halos are rebuilt without averaging nodes or recalculating the
saved caches. Shared coordinates, Jacobian and nine metric components must
be finite and agree with freshly initialized target geometry within 2e-10.
Only this cross-topology CURVE geometry check uses a tolerance; original
TGV/channel geometry checks and same-topology exact restore are unchanged.

CPU scalar/GPU full continuous-12 versus 5+7 checks have maximum x/x-z state
error 2.5457045834135963e-18 and zero physical geometry error. Target-layout
shared nodes, interior x and periodic-z face halos are checked separately; undefined
physical-face exterior/corner halos are not treated as authoritative data.
Same-topology q/caches/rank supplements and geometry remain exactly equal.
Original grid/profile files can be absent when matching shared resources are
carried with the checkpoint. ParaView actually reads the resumed field and
slice series. GPU face-cache refresh now follows CPU neighbor guards: a
MPI_PROC_NULL face is not converted from zero conservative ghost values.
All tested state rank supplements remain finite; no clipping is introduced.
Nonperiodic y repartition, dynamic inflow, AIR5, derivatives,
rendering and backend migration remain outside this
gate. Fixed AIR5 HBL periodic-z repartition passed its separate scale-aware
gate in plans 10.55-10.56; see below, not the CURVE evidence.
These CURVE records enforced the controlled packing budget, not a native
free-memory reserve. The later optional guard is documented above; they do
not count as reserve-enabled CURVE acceptance.

The later statistics gate in plan 10.67 admits CPU mean44 and GPU compact
histories for the same 16-cubed static CURVE case. GPU metadata version 2
retains the original local partition layout and adds an `inherited/` group
inside `statistics.h5`, not another checkpoint file. Root `q0002:q0013`
and `q0015:q0017` contain partition-local increments. Their global total is
the sum over the root partition axis plus the first inherited plane, once.
Do not sum copies of the inherited baseline across ranks. Root `q0001` and
`q0014` are geometric measures; sum their partition contributions without
adding the inherited geometry copies. Terminal planes are zero padding;
wall moments occupy j=0 only. Both groups carry the same current sampling
identity, not separate sample counts to add.

On repartition, old local increments join the inherited history and new local
increments start from zero; total sample identity is preserved. Same-topology
restore preserves both arrays exactly. A second exact restart after migration
is checked in both x/z directions. Nonzero migrated history cannot be written
to legacy binary sidecars. The host baseline is replicated and budgeted; this
bounded 16-cubed implementation is not a production-size capacity claim.
CPU/GPU cumulative errors are at most 2.28e-13/7.11e-15. Native first/last
frame reserve checks, controlled buffers and two-rank memcheck gates passed.
The original geometry-only evidence above retains its original scope. Dynamic
inflow adds the separate bounded gate below; AIR5 history migration and rendering
repartition remain unregistered.

### Bounded Dynamic-CURVE Inflow And History Repartition

Plans 10.68-10.69 admit the same 16-cubed extruded geometry with dt=6e-6,
twelve frozen nonpolynomial temporal source frames, four active cached frames,
543e/643e, viscosity/filter, FP64 and explicit synchronization. CPU/GPU separately
cover NP=1<->2 x/z and NP=2 x<->z. Solver y stays complete; the inlet file uses
physical y/z/dummy axes, so only its z coordinate is repartitioned.

Cached physical values are read directly for the destination partition. Past
frames are not reinterpolated, and accumulated statistics are neither reset nor
resampled. Time windows, slot identities, sampling and schedules are preserved.
Tests actually advance the frame cursor and include another exact restart after
both directional migrations. Frozen resources suffice without the original inlet
directory. Same-topology payloads remain exact; cross-topology state/history
errors stay below 2e-10 and inlet cache differences are zero.

This admission is NP<=2 only. Larger or different dynamic configurations, y
repartition, cross-backend migration and rendering repartition remain unavailable.
The replicated host history baseline is bounded-test evidence, not an arbitrary
production-grid memory guarantee. No additional checkpoint file is introduced.

### Bounded AIR5 HBL Periodic-z Repartition

Plans 10.55-10.56 admit the fixed 16-cubed-cell dimensional HBL short gate,
not SBLI: generated Cartesian grid, ninit=0, bc=11/50/41/51/1/1, 643e/643e,
recon_schem=5, physical-space reconstruction, filter/viscosity enabled and FP64
explicit synchronization. It uses coupled sources, compensation on,
symmetric_species convection, layered diffusion and the characteristic top.
The original gate excluded accumulated histories. Plans 10.70-10.71 additionally
admit mean44 and GPU global conservation history; formal in-situ statistics,
derivatives and rendering repartition remain outside this gate. Source and target
x/y stay complete; only same-backend NP=1<->2
periodic-z repartition is admitted. Tests use CPU scalar and GPU full storage.

The tested domain is 0.08/0.01/0.002 m, initial rho=0.05 kg/m^3, T=3000 K,
N2/O2=0.767/0.233 and dt=1e-10 s. q, carry, their extended-precision represented
value q-carry and caches are compared separately against frozen physical scales.
Maximum normalized state/defined-halo differences are 2.585e-25/2.585e-26;
all thirteen geometry fields match exactly. Pure restore preserves all 34
physical components bitwise before the first advance; same-layout continuation
is bitwise exact. This is a matched numerical restart gate, not independent
chemistry physics, arbitrary-profile, long-time or performance validation.

`ASTR_CHECKPOINT_TEST_RESTORE_PROBE=1` is a reserved local validation switch.
It requires this bounded HBL restore and agreement across ranks, writes
`outdat/restore_probe.h5` before any advance and, when enabled, writes
`outdat/restore_statistics_probe.h5` and `outdat/restore_conservation_probe.bin`
after restoring those histories. It leaves normal checkpoint
schedules unchanged. The file is a standalone diagnostic state, not a published
restart batch. Without the explicit switch there is no additional output.
Do not use it in production launch scripts.

Mean44 remains a global physical-node HDF5 array, not a partition-integrated
history. Its actual accumulated values are redistributed directly, with counts
and first/last sample metadata unchanged. No historical samples are regenerated.
The global conservation baseline is restored once per rank from the same saved
scalar state, never multiplied by the destination rank count or recomputed.
The continuation phase-0 row echoes that baseline, not a new sample.

Pure restore preserves all 34 state/cache/carry and 44 mean fields bitwise, plus
the enabled conservation scalar history. Same-topology continuation remains
bitwise. Cross-layout mean differences use fixed pre-run dimensional scales
times actual sample count; conservation totals use conserved scales times the
fixed domain volume. Both normalized gates are 2e-10. These are numerical
restart checks, not claims about long-time statistical convergence.

Seventeen main gates, eight affected real-case continuations and thirteen
provider/padding checks pass. Per-test disk/controlled host maxima are
65596431/9803688 bytes, below 64 MiB; MPI/HDF5/compiler transient peaks are not
included. This AIR5 gate did not enable the later optional basic-archive
reserve guard (plan 10.57) and is not its acceptance evidence.
Pre/post-run free-memory observations are not a per-frame OOM guarantee.

### Bounded AIR5 SBLI x Repartition

Plans 10.72-10.73 additionally admit fixed dimensional 16-cubed SBLI,
same-backend NP=1<->2 x only, with y/z complete. The HBL domain, physical
scales, boundary inventory, FP64, explicit synchronization, filter/viscosity,
coupled sources, compensation and limiter settings above are retained.
SBLI uses recon_schem=3, inlet speed 4*a_ref, incident angle 25 degrees,
top intersection x=0.035 m and the frozen initial field/Tv profile.

The checkpoint restores all 34 physical state/cache/carry fields and actual
mean44 moments bitwise before advancing. It preserves the GPU conservation
baseline and sampling metadata, without resampling or clearing history.
Same-topology and another exact restart after migration are bitwise;
cross-layout fields and moments satisfy the fixed-scale 2e-10 gates.
CPU/GPU y/z repartition fails closed. This does not admit arbitrary SBLI,
rendering repartition, cross-backend migration or long-time physical validation.
The reserved restore probe described above also accepts this bounded SBLI gate.

### Optional Native TGV Rendering And Continuation

Provide `datin/input.output` and set
`ASTR_INSITU_CONFIG=datin/input.insitu` to use the existing GPU EGL TGV preset
with completed-step native checkpoints. See `../insitu/README.md` for the
renderer build and configuration. Create the renderer output directory first.
Do not set legacy `lrestart=t`, `batch_prefix` or `restore_batch`; select only
`output.restore_directory` for the new checkpoint entry. Statistics settings
must still match the saved history; output override does not change that rule.

All native checkpoints now require a sealed `insitu_control.bin`, even when
rendering is off. It is 144 bytes inactive and 313 bytes with rendering, clock,
configuration/content identities and saved schedule. Missing members, clock
changes, invalid schedule history and any binary tail are errors. Earlier
candidate checkpoints missing this member are not accepted. Ordinary output
does not need or load Catalyst to write/read the inactive record.

An unchanged render schedule resumes progress and never repeats its saved
frame. Final rendering completes before sealing the final checkpoint, so the
same checkpoint can either continue stepping or finish without a duplicate
image. Changed render enablement, cadence or initial/final settings require
`restart_output='override'`. Only the changed render origin resets to the
restored point. An explicitly changed initial frame renders that point without
counting another statistics sample. Entry-script and implementation-library
content changes also require override; these two fingerprints do not cover
all helper modules, library dependencies or driver changes.

This admission is local GPU Cartesian periodic TGV, same topology/executable
and unchanged dependency environment. It is not CPU, CURVE/AIR5 or repartitioned
rendering certification. The 16-cubed NP=1/2 gate (plan 10.36) compares q,
caches and statistics exactly, as well as JPEG pixels and VTK pieces. Rendering
still downloads all eleven local physical flow components on due frames
(432344 bytes/rank at NP=1, 228888 at NP=2 x), then uses private CPU diagnostics
and geometry work. It is not the plane-only file transfer path or zero-copy
postprocessing. Render-only native runs no longer retain a full previous-frame
snapshot for final rendering. Short observed resource peaks are not a
long-sequence or production-scale memory guarantee.

Plan 10.74 additionally checks simultaneous checkpoint, basic volume, three
index slices and formal velocity statistics, with GPU rendering enabled.
CPU-only and GPU NP=1/2 complete 12 versus 5+7 steps exactly; disabling archives
and rendering leaves flow/statistics unchanged. Use the existing output examples
and independent `&volume` / `&slices` schedules alongside `input.insitu`.
Keep the renderer output directory distinct from the native output root.
The joint gate retains the same topology and does not expand renderer admission.
Scope A closes the first delivery with GPU Cartesian TGV rendering only.
CPU/CURVE/AIR5 rendering and all render-enabled repartition remain rejected
and require separate future implementation and acceptance goals.

Checkpoint publication now rejects a stale `.LATEST.tmp`, or a `LATEST` that
is a symbolic link, hard link or directory, before renaming the candidate.
Candidates and earlier complete batches remain available for diagnosis on
failure; do not automatically delete temporary or unregistered directories.
`LATEST` changes only after the new batch is sealed. A pointer-update failure
leaves the previous entry; a later retirement failure leaves the newly
published entry. An old batch whose COMPLETE was removed is not restorable.

Common state checks during checkpoint writing report `batch` and
`last_complete`. Before the first successful new save after restart, the
latter identifies the validated restore source. Context is cleared on success.
Process loss or errors bypassing these checks cannot promise this diagnostic.
Bounded real CPU/GPU TGV failure/recovery and archive regressions are documented
in plan 10.28. Test-only EIO injection is not a disk-full, power-loss or target
filesystem durability gate; no new fsync guarantee is provided.

### Same-root continuation

An existing output root is reusable only when `restore_directory` names a
validated checkpoint in that root's `checkpoints` directory. Each enabled
product creates a new `segment########`; old segments remain unchanged. Its
shared coordinates are read-only verified against current geometry, in bounded
tiles, without packing/downloading GPU flow fields. Missing plane geometry,
layout/unit/type mismatches, aliases and incomplete directories are rejected.
Changed slice positions that require new geometry need a new output root.

Read each segment's `series.xdmf` independently, or explicitly build the separate
parent-chain index below. Existing checkpoint step-name collisions are rejected,
so this is not unrestricted forking from historical checkpoints. A later
product failure can leave empty new-segment metadata in an earlier product;
the implementation does not claim automatic rollback. Old frames/resources
are not rewritten. See plan 10.29 for four bounded physical combinations and
actual ParaView readback, not a production-scale lifecycle claim.

### Offline single-segment index repair

For a stopped native segment, validate sealed frames without writing:

```bash
python3 scripts/output/repair_series.py /path/to/run/fields/segment00000000
```

After confirming that **no solver or other writer uses the segment**, explicitly
publish reconstructed indexes:

```bash
python3 scripts/output/repair_series.py /path/to/run/fields/segment00000000 \
  --publish-stopped-segment --catalog-bytes 1048576
```

The same command supports grouped slices and cached AIR5 fields. It reconstructs
only `series.frames` and `series.xdmf`, from the sealed frame manifests, native
HDF5 metadata/layouts and shared-coordinate identities. It does not follow
arbitrary old XML references, advance the solver, or modify flow/checkpoint
files, `LATEST`, sampling or restart controls. It does not join different segments.
Complete frame directories with missing/bad markers, CRC/layout/unit mismatch,
nonmonotonic clocks, unsafe links and oversized catalogs are rejected. Partial
`step############.tmp` directories and original native index temporaries are
preserved but not indexed. An empty segment is not given an empty time index.

CRC validation **scans complete HDF5 files**, even though no flow arrays are
loaded: `crc_scan_bytes` and `field_array_read_bytes` are distinct report fields.
`--catalog-bytes` caps frame-name/list/sorting storage, not total Python/HDF5
RSS. One frame's XML is built at a time; native plane count is limited to 768.
This is an integrity/layout check, not new physical validation or a low-I/O claim.

Two exclusive `.repair.tmp` files are closed before their sequential replacements.
Each replacement is atomic, but the pair is not a transaction; a second failure
can leave a newer ledger with older XML. Source frames remain unchanged. Inspect
repair temporary files before retry; the tool refuses to remove or reuse them
automatically. It cannot prove absence of a concurrent writer or guarantee
power-loss durability. See plan 10.30 for 44 bounded checks, including actual
ParaView readback of immutable accepted TGV/AIR5 artifact copies.

### Explicit parent-chain indexes

New `ASTROA02` archive histories save each enabled product's current segment
number and SEGMENT byte count/CRC-64 in the sealed checkpoint. At restore,
the registered parent must match that receipt. A new `ASTR_OUTPUT_SEGMENT_2`
record stores the restore clock, source checkpoint MANIFEST fingerprint,
explicit relative parent path/fingerprint and frozen input fingerprint.
Every new frame's sealed RESOURCES binds both SEGMENT and shared coordinates.
Products disabled in the source have no field parent; no earlier history is
inferred. These candidate binary histories do not read ASTROA01 histories.
State/statistics exactness and exact schedule comparison are unchanged;
segment receipts intentionally differ between uninterrupted and resumed runs.

For **stopped, immutable source segments**, inspect the explicit chain:

```bash
python3 scripts/output/combine_series.py /path/to/run/fields/segment00000001
```

To create a separate index directory, without replacing any solver index:

```bash
python3 scripts/output/combine_series.py /path/to/run/fields/segment00000001 \
  --output /path/to/new-combined-fields --catalog-bytes 1048576
```

The new directory must not exist and must be outside every source product
directory. It contains `lineage.frames` and `lineage.xdmf`; open the latter
with Xdmf3ReaderS. The same command handles grouped slices and AIR5 caches.
Only the registered parent chain is followed. At each edge, parent frames
after the child's restored step are excluded; retained parent times must not
exceed the restored time. Other branches are not scanned or included.
Duplicate clocks, changing field/plane layouts or units, cycles, missing or
changed parents, aliases, incomplete frames and exceeded catalog budgets
are rejected. Old frames/branches remain independently readable.

SEGMENT-1 data remain supported by single-segment repair, but cannot supply a
combined parent chain: neither directory numbers nor nearby times establish
parenthood. A combined index requires the whole chain's product resources,
segments and input records, **not retained checkpoint files**. Relative edges
and XDMF references survive moving their common containing tree together.
Moving only the new segment or independently moving an external parent root
breaks those references and is not automatically repaired.

All source CRCs are validated by streaming full files; no flow arrays are
loaded. The catalog cap includes retained chain metadata and one frame-name
catalog, not total Python/HDF5/ParaView RSS. Publication closes both exclusive
temporary indexes before two sequential renames inside the fresh directory.
On failure, partial diagnostics remain; no source files are removed or changed.
This is not a two-file transaction, live-writer repair or fsync/power-loss
guarantee. See plan 10.34 for CPU/GPU and CURVE/AIR5 local readback evidence.

Plan 10.62 adds a combined lifecycle check for registered 16-cubed dynamic
CURVE inflow (z), AIR5 HBL (z) and AIR5 SBLI (x), CPU/GPU each NP=2. A relocated
five-step source resumes to step twelve, explicitly changing checkpoint
scheduling to every step and keep=1. The protected source at step five and the
latest checkpoint at step twelve both remain: keep=1 does not override restore
source protection. Historical fields and shared resources remain unchanged.
State, accumulated statistics, inflow history or AIR5 compensation and GPU
conservation history match the corresponding uninterrupted run exactly.
After another directory move, stopped-segment index repair and explicit
parent-chain ParaView readback pass. The six matched refusal checks reject a
schedule change without `restart_output='override'`, without publishing a new
complete checkpoint or changing an existing one. These are same-topology basic
field checks, not AIR5 derivative, repartition or live-writer repair acceptance.

### Native Automatic Parent Index

Each enabled native product now publishes a per-segment `series.xdmf` and,
when layouts match, a `lineage.xdmf` referencing the explicit parent chain and
current frames. Ancestors are truncated at each declared restore point, not
at arbitrary directory order. The companion `lineage.frames` uses
`ASTR_NATIVE_LINEAGE_1`: each frame has a clock row, a data-file reference and
a geometry-file reference. This is distinct from the offline combined ledger.
The solver uses native Fortran/C/HDF5 APIs; Python is not a runtime dependency.

Startup validates ancestor seals, CRCs, input/segment fingerprints and typed
HDF5 inventories. File CRC checks scan payload bytes but do not load field
arrays. Subsequent frame publication streams cached ancestor references and
current clock rows, without repeated ancestor CRC scans. Paths are relative,
XML-escaped and reject aliases, controls and HDF-reference delimiters.
The bounded catalog allows 128 segments and 8192 frames per ancestor segment;
paths are at most 1200 characters. Exceeding a bound is an error, not truncation.

An explicit legal override can change fields or planes. A valid incompatible
layout reports `ASTR_OUTPUT_LINEAGE unavailable`, keeps all parent edges and
per-segment indexes, and allows continuation without a unified index. It does
not intersect field sets, fabricate missing fields or delete history. Missing,
corrupt or invalid sources remain errors. Keep the full relative source tree
when moving an archive; the index does not copy source fields or geometry.
Metadata replacements are separate atomic file operations, not a two-file
transaction. On failure, inspect staging; indexes are not checkpoint authority.
