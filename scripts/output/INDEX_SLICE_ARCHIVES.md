# Fixed-Index Numerical Slices

Numerical slices save native node values, not screenshots or restart states.
They work without Catalyst. A fixed global i/j/k index identifies the original
grid surface; on CURVE grids this need not be a constant physical x/y/z plane.

## Configuration

`input.output` retains checkpoint, volume and shared settings. The `&output`
group accepts `product_catalog='input.products.jsonl'` and
`product_poll_steps=10` (both are defaults). Only rank 0 observes this file,
at completed-step boundaries. The old `&slices` group cannot define enabled
products, positions or diagnostics. An empty/disabled placeholder is accepted
but creates no product.

Copy the TGV or AIR5 JSONL example to the run directory, adjusting global
indices and ranges to the actual grid. The first line is exactly:

```json
{"catalog_schema":"astr-insitu-products-v1"}
```

Each following line is one independent product. Example for a 16-interval box:

```json
{"id":"midplane","kind":"slice","enable_request":1,"source":"instantaneous","fields":["velocity_x","pressure","Q_rs","density_gradient_magnitude"],"sampling":{"mode":"grid_index","axis":"k","index":8,"index_ranges":{"i":[0,16],"j":[0,16]}},"output":{"channel":"data","format":"hdf5-xdmf"},"schedule":{"mode":"steps","interval":4}}
```

Ranges are inclusive integer global node indices, with no interpolation or
stride. To save a whole plane, select the full ranges of its other two axes.
Schedules accept either `mode="steps"` with an integer interval, or
`mode="time"` with a positive time interval. Output occurs at actual completed
steps, not at interpolated target times. A new definition emits a first frame
at its first accepted completed step. Subsequent cadence changes remain
anchored to the last successful output. There is no implicit final frame.
Existing adaptive event/window IDs may be listed in `events`/`windows`, with
an explicit smaller `dense_interval`; this does not start a new monitor.

The file describes the whole desired list. Omitting an existing ID stops it;
header only means an empty list. To publish edits atomically:

```bash
python scripts/insitu/product_catalog.py publish candidate.jsonl input.products.jsonl
```

An unreadable file or bad header keeps the effective list. An identifiable bad
product disables only that ID. Other independent records continue. Publication
does not imply acceptance: read `products.status.json` under the output root.
Correct a configuration error and republish. For a resource/I/O-disabled
product, explicitly increase its `enable_request`, for example:

```bash
python scripts/insitu/product_catalog.py enable input.products.jsonl midplane
```

## Fields And Sources

Instantaneous fields are `density`, `velocity_x/y/z`, `pressure`, `temperature`.
AIR5 also provides `vibrational_temperature` and
`mass_fraction_N2/O2/N/O/NO` in the existing model order. There is no duplicated
three-component velocity dataset. Offline readers can assemble one from three
requested scalars.

Diagnostics are `velocity_gradient_xx/yx/zx/xy/yy/zy/xz/yz/zz`, `Q_rs`,
`velocity_divergence`, `vorticity_x/y/z`, and `density_gradient_magnitude`.
In the gradient names, the first suffix is velocity component and the second
is physical derivative direction. Gradients use an independent seven-node
explicit sixth-order operator, with one-sided/biased physical-boundary stencils
and the existing static grid metrics. They do not change solver derivatives.
MPI stencil neighborhoods are private and do not average solver state or halos.
Only selected two-dimensional field values are downloaded for file output;
private communication-strip staging is counted separately.

For `source="reynolds"` or `"favre"`, only already enabled, covered means are
available. Zero coverage waits; missing fields are not replaced with zero or
instantaneous values. When needed, select `statistics_source` explicitly:

| Provider | Available Fields | Actual Measure |
|---|---|---|
| `velocity-time` | Reynolds density; Reynolds/Favre velocity and velocity-derived diagnostics | Existing clipped time integrals and density-time integrals |
| `solver-samples` | Reynolds density, pressure and velocity; Favre velocity and temperature; their supported diagnostics | Existing equal-weight solver samples |

There is no accumulated Favre density, Reynolds temperature or AIR5 Tv/species
mean in these providers. M12 GPU spanwise-reduced statistics are not arbitrary
three-dimensional pointwise means and are not replicated as such. A requested
mean diagnostic is computed from the selected mean field: `Q(mean velocity)`,
not `mean(Q)`. This channel never starts, resets or extends an accumulator.

Fields preserve their source precision and current solver unit convention.
The HDF5 type, field units, raw product definition, completed-step clock,
statistical measure and available freestream/reference values are stored.
No implicit SI conversion, pressure/temperature ratio or schlieren contrast
mapping is performed. Those are offline choices.

## Files And Continuation

Each ID/version has self-contained continuation segments:

```text
slice_products/<id>/version000000000001/segmentXXXXXX/
  definition.json
  geometry.h5
  step000000000004.h5
  step000000000008.h5
  series.xdmf
```

The static 2D XYZ geometry is written once per segment. Each successful time
has its own HDF5 file, without compression. HDF5 array axes are the remaining
global axes in reverse order; coordinates have a final XYZ dimension. XDMF
references are relative to this directory, which can be moved independently.
A frame is indexed only after its HDF5 write, close and publication succeed.
There is no checksum or write-after-read pass.
`definition.json` records the segment's initial definition. Each frame also
stores its then-effective full record, including any cadence-only updates;
earlier frames are not rewritten when cadence changes.

New checkpoints contain the effective product definitions, versions, clocks
and consumed enable requests in `insitu_control.bin` (`ASTRIR07`). Old control
formats are not accepted. Restore starts new self-contained segments and does
not repeat the saved successful frame. The checkpoint's effective definitions
win over disk configuration until that file is republished after restore.
Earlier segments are not implicitly copied or referenced by the new segment.

Recoverable allocation/path/write errors suspend that product, not CFD or
other products. No timed retry is made. An MPI failure, invalid CUDA context or
partially created collective HDF5 object is not claimed recoverable.

## Offline Reading

The solver needs no Python. Offline tools require Python, NumPy and h5py;
replotting additionally uses matplotlib/scienceplots. JPEG is the default;
EPS is only emitted with `--eps`.

```bash
python scripts/insitu/slice_archive.py list /path/to/segmentXXXXXX
python scripts/insitu/slice_archive.py plot /path/to/step000000000008.h5 \
  --field pressure --output /path/to/pressure --limits 1 2
python scripts/insitu/slice_archive.py profile /path/to/step000000000008.h5 \
  --field velocity_x --dimension 0 --index 0 --output /path/to/profile.csv
python scripts/insitu/slice_archive.py join /path/to/first-segment /path/to/next-segment \
  --output /path/to/combined.xdmf
```

Plot axes default to the two varying computational directions' corresponding
physical coordinate components. Use `--axes x y`, etc., to choose an explicit
physical projection of a curved surface. This is a projection, not a new
constant-height slice. Profile dimension/index refer to the stored 2D array.
The tools do not invent normal gradients or missing data from a 2D archive.
`join` takes explicitly ordered segments with matching slice definitions and
increasing, nonoverlapping output times. It writes an external-reference index,
not a self-contained directory, and does not modify or copy source frames.

ParaView should open `series.xdmf` using its XDMF3 reader. A rendering-only
Catalyst installation may lack this reader. Local numerical/I/O tests and
measured costs are recorded in
`documents/ASTR_INDEX_SLICE_ARCHIVE_VALIDATION_REPORT.md`; these short gates are not
production physical validation or a claim that every solver configuration has
been tested.
