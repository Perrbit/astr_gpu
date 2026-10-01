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

## Candidate Native Archives

The new runtime admits basic live volume/slice output for registered RK3
checkpoint cases: periodic TGV, bc41 channel, static/dynamic CURVE flatplate,
and fixed dimensional AIR5 HBL/SBLI, on CPU or GPU. Existing case/boundary/
model admission checks still apply; this is not arbitrary-case support.
These are bounded local acceptance paths, not a completed production-output
redesign. Derivative fields have the separate narrow TGV admission below.
The offline exporter above remains a separate interface.
Ordinary native output requires MPI/HDF5, not Catalyst or ParaView.

The shared writer and offline index tools support explicit derivative selections.
`ASTR_DERIVED_FRAME_1` records add
fourteen selection slots to the basic FRAME declaration, matching the HDF5
`derived_layout` attribute. Scalar datasets declare definitions, physical
coordinate space and units. Gradient names use velocity component first and
physical derivative axis second; `velocity_gradient_yx` is du_y/dx. Q is the
full-strain rotation/strain `Q_rs`, not the second principal invariant.
The repair/parent-chain tools reject changing layouts and inconsistent field
declarations. Real native derivatives are admitted only for generated periodic
Cartesian 16-cubed-cell TGV, five variables, dimensionless input, 643e/643e,
viscosity/filter enabled, FP64 and NP=1/2 (plan 10.44). Other cases and sizes
are rejected, even when their basic fields are admitted. GPU rendering and
repartition have independent admission rules.
`export_checkpoint.py` still exports cached basic fields only.

Within this gate, select `velocity_gradient=t` (nine components),
`vorticity=t` (three components), or `qcriterion=t` (Q_rs and divergence)
independently in `&volume` and `&slices`. Selection appends fields to `fields='basic'`;
all three together produce twenty scalar fields. See
`input.output.tgv.derived.example`; the basic example stays unchanged.
The private complete-step velocity snapshot receives fresh periodic halos and
sixth-order physical derivatives; stale solver gradients are not exported.
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

`input.output.tgv.example` is a **16x16x16-cell short-test** configuration,
not a production output-frequency recommendation. Disable the legacy field
sequence/slice flags in the controller, prepare `outdat/new` as a new directory,
set `ASTR_OUTPUT_CONFIG=datin/input.output`, and launch the normal
`astr run datin/input.tgv` command. Runtime `usegpu` still selects CPU or GPU.
Without `ASTR_OUTPUT_CONFIG`, this path does not change legacy output.

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
slices are retained independently of checkpoint `keep=1|2`. Slice-only output
does not create a volume, checkpoint or shared checkpoint geometry. On GPU
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
cross-topology bitwise guarantee or a performance improvement claim. Without
`ASTR_OUTPUT_CONFIG`, the legacy atomic reduction and file behavior are unchanged.

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
Automatic cross-segment publication, other case families, derived quantities
and other Catalyst restart combinations remain pending. Registered GPU TGV
render continuation is described below. The separate explicit
parent-chain builder below is available for stopped native segments.
The candidate binary schedule format is not a long-term compatibility promise.
See redesign plan 10.24 for runtime gates, 10.25 for shared coordinates, and
10.26 for the single-segment sequence gate, and 10.27 for the additional cases.

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

This does not admit walls, CURVE, AIR5, other numerical settings, larger rank
counts or backend migration. Repartition also rejects either saved or current
render enablement; disabling rendering with override cannot bypass the saved
renderer's unvalidated partition history. Keep the source checkpoint and its
shared resources immutable. See plan 10.38 for the matrix and readback evidence.

### Optional Native TGV Rendering And Continuation

Set both `ASTR_OUTPUT_CONFIG=datin/input.output` and
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
