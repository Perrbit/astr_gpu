# Runtime Postprocessing Products

This interface is still under X2 integration. The complete capability and
validation record is in `documents/ASTR_INSITU_RUNTIME_PRODUCT_CATALOG_PROGRESS.md`.
Wall node and original SBLI physical-midspan curves support the subsets below.
GPU-produced index/physical surfaces, Q and streamlines also have an explicit
compatible consumer. Native VTK geometry export is available through that explicit
consumer. Explicit host processing also has bounded TGV/mean/wall checks;
general allocation-failure isolation and final X2 delivery are not yet complete.
Do not infer release readiness
from this example.

`input.products.tgv.images.example.jsonl` contains explicit definitions for the
tested 32-cubed, FP64 GPU TGV setup on `[0, 2*pi]^3`. It is not a production
resolution or cadence recommendation. Configure Catalyst, the rendering pipeline,
budgets and `postprocess_transport` in the base in-situ file before starting.
Use `standard-device` or `direct-device` for resident geometry, or explicitly
select the compatible consumer below; select `pinned` or `device-aware`
explicitly. A requested unavailable device transport does not fall back to host
payloads. Mean products need the existing `velocity-time` accumulation enabled at
startup; the product catalog never starts or resets it.

## Publish The Complete List

The first JSON line identifies the format. Each following line is one product.
The file is a complete target list, not a patch. Keep all still-needed entries,
including numerical archive or probe entries, when adding an image. Use one
header when combining examples. Publish a finished candidate using
`product_catalog.py publish CANDIDATE FORMAL`; do not edit the watched file in
place. `product_catalog.py enable FORMAL ID` increments an explicit retry request.
The saved effective catalog, not an unconsumed disk edit, is authoritative after
restart.

Native production does not require a common `schedule_mode`, `step_interval`,
`time_interval`, `initial_frame` or `final_frame` in the base namelist. Previously
supplied values and the old `product_ids` clock arrays are ignored there; they
cannot create a second product list or change continuation. The existing small
control envelope retains an inert canonical clock for its unchanged layout.
Independent `insitu-preview` still parses its own original configuration.

## Definitions And Reuse

- An index image is selected by global node indices. A physical plane is cut
  using its unit-normal direction and physical coordinates.
- Q uses the existing `Q_rs` definition. A zero threshold is not a vortex-core
  definition. Valid empty geometry is reported as empty, not a failed product.
- Streamlines use the selected instantaneous, Reynolds or Favre velocity for
  both integration and coloring. They are spatial streamlines, not particle
  paths through a changing flow.
- RK45 steps and `maximum_length` are lengths in the case coordinates.
  `maximum_steps` is an integrator stop, not the CFD run length. No periodic
  wrapping is introduced. The accepted last point and subminimum remaining
  length retain the existing termination semantics.
- The current streamline color producer supports velocity components,
  magnitude and the existing `u_Uinf`/`w_Uinf` normalizations. It does not invent
  mean temperature, pressure or species accumulators.
- Streamline `bounds` clip display geometry; they do not move the existing
  seeds or add an integration stop surface. New clipping vertices interpolate
  product fields along the accepted line segments.
- Source, seeds, RK45 settings and direction identify an integration. Cameras,
  widths, colors and clipping reuse it within the same frozen complete step.
  Instantaneous results are never reused for a later step.

The example uses only JPEG. Each product has its own definition version, output
segment and cadence. A pure cadence change retains its definition version and
anchors the next output to the last successful output. Rendering images does
not enable three-dimensional archives or checkpoints.

## Current Numerical Evidence

The local bounded checks cover FP64 TGV NP=1/2, both device bridges, existing
16/256 seed layouts, private halo input checks, an existing y-wavy fixture,
CPU/GPU same-state scalar oracles, dynamic controls and 12 versus 5+7 restart.
Validation-only geometry downloads are explicitly flagged and are not evidence
of production zero readback. These are software and numerical checks, not a
long-time turbulence or SBLI physical validation.

## Explicit Compatible Consumer

In the current device-rendering build, select `processing_backend='device'`,
`derivative_backend='gpu'`, `rendering_pipeline='compatible'` and the same
`device_render_pipeline.py`. The solver still needs its existing GPU/EGL binding.
This does not enable a CPU solver renderer or the separate host-processing
producer path.

The compatible consumer reuses the native GPU extraction, then explicitly
downloads only final FP64 coordinates, connectivity and the selected quantity.
Ordinary VTK mapping consumes that host geometry. It does not download a 3D
flow snapshot or silently replace an unavailable device bridge. The two resident
pipelines retain their no-geometry-readback contract.

Within a frozen extraction group, identical native geometry/quantity owners share
one host packet across cameras. The packet is borrowed synchronously and retired
after its group; no previous-frame packet is required. Receipts separate
`geometry_d2h_bytes` (actual download for this consumer) from
`geometry_host_bytes` (the bound packet size). A reused camera has zero additional
download, not zero host data. Private halo staging and source-field transfers
remain separate counters.

Bounded evidence covers 32-cubed FP64 TGV, NP=2, all four image kinds, actual
native coordinates/scalars within the existing threshold, empty geometry,
instantaneous/Reynolds/Favre products, 12 versus 5+7 restart, safe controlled-host
allocation refusal, image publication failure and explicit-request recovery.
Pinned and device-aware private inputs produce identical results. These checks
do not admit arbitrary host processing, all legacy presets or production scales.

## Native Geometry Files

`input.products.tgv.geometry.example.jsonl` gives four complete definitions for
the same 32-cubed TGV: an index surface, a physical plane, a Q surface and
streamlines. Select `processing_backend='device'`, `derivative_backend='gpu'`
and `rendering_pipeline='compatible'` in the device-rendering build. These
records use `fields`, not a coloring `field`, and have no `display` section.
They reuse existing extraction, interpolation and RK45 settings; exporting
geometry alone does not create a view or an image.

Each frame has one `.vtp` part per rank and a `.pvtp` entry point under
`geometry_products/ID/version.../segment...`. Open the `.pvtp` in ParaView.
Parts retain the solver's native coordinate/field precision and Int64
connectivity. Surface fields are the explicitly selected canonical quantities.
Streamlines currently export selected velocity components, plus existing
`accepted_length`, `termination_status` and `untravelled_arc_length` point data;
`SeedIds` and signed `IntegrationDirection` are cell data. Segments sharing a
seed/direction retain their identity across rank pieces. They are not stitched
into one new global polyline.

Frame metadata records the complete step, time, source, definition version,
unit system, field units and the existing twelve reference values. Available
mean-source coverage is retained. `definition.json` is stored once per segment,
not repeated in every frame. Geometry files are diagnostics, not checkpoints.

All ranks close their pieces before rank zero publishes the manifest. A safe
piece or manifest publication failure removes only files owned by that failed
frame, disables that instance and leaves other products/CFD running. It requires
a new `enable_request` before retrying. There is no CRC or write-then-read check.
The two resident pipelines reject this explicit export request rather than
silently downloading geometry.

Bounded device-producer checks cover FP64 TGV NP=1/2, native scalar references, trajectory reuse,
12 versus 5+7 exact continuation, publication failures and dual-rank memcheck.
The existing y-wavy index surface and AIR5 wall plane also retain their native
coordinates/fields. This does not admit arbitrary curved Q/streamline or AIR5
trajectory combinations.

## Explicit Host Processing

`input.insitu.tgv.host.example` contains the corresponding bounded startup
settings. Replace its implementation/tool paths, create `outdat/render`, select
it with `ASTR_INSITU_CONFIG`, and publish the desired TGV JSONL definitions.
It omits the obsolete base clock; every native product declares its own schedule.

For the current bounded GPU-solver/EGL setup, select
`processing_backend='host'`, `derivative_backend='cpu'`,
`rendering_pipeline='compatible'`, `products='all'` and the same
`device_render_pipeline.py`. Leave `postprocess_transport` unset: that setting
belongs to the device-processing private transport. Use the same explicit
image/geometry JSONL definitions. The base in-situ file does not supply product
parameters. The catalog path belongs to `&output product_catalog`, whose default
is `input.products.jsonl` in the case working directory.

This route samples the selected source fields, reuses the existing CPU sixth-order
diagnostics and VTK cutter/contour/parallel RK45 filters, then uses the compatible
consumer. Volume products can download selected three-dimensional fields;
formal host statistics retain their separately accounted existing sampling.
It is not a fallback for either resident pipeline and does not enable a CPU-only
solver's native EGL renderer. Do not request GPU diagnostics from an existing
host-resident mean source: that upload/derivative combination is not implemented.

Within a completed-step group, images, geometry exports and display-only changes
share the same VTK extraction/trajectory. Native fields and connectivity remain
in solver precision and Int64 respectively; the host route does not prepare
the device route's FP32 display arrays. Its triangulation and vertex ordering
are not promised to match the GPU extractor. Same-backend continuation is checked
separately from CPU/GPU source-value comparisons.

Host streamline geometry preserves VTK's original `IntegrationTime` point data
and `ReasonForTermination` cell data, with global `SeedIds` and signed
`IntegrationDirection`. `IntegrationTime` is the steady-field integration
parameter, not the CFD output time and not accepted arc length. SI sources use
seconds; dimensionless sources use their case normalization. It is deliberately
not relabeled as the device integrator's `accepted_length` or its termination
enum. Every geometry frame identifies `processing_backend`.

Receipts record host processing and selected source-field download separately
from CPU-generated geometry. Private MPI staging is logged in
`ASTR_RUNTIME_TRANSFER`; formal statistical sampling has its existing accounting.
No geometry device download means the geometry was built on the CPU, not that
the host route had zero D2H traffic.

Bounded evidence covers FP64 32-cubed TGV NP=1/2, all four image/geometry kinds,
three velocity sources, exact 12 versus 5+7 continuation, same-phase scalar
oracles, shared trajectories, publication failure, explicit-request recovery
and dual-rank memcheck. Existing channel wall charts use
`products='channel_walls'` with the host/CPU/compatible settings; instantaneous
and wall-time means, a y-wavy wall and an AIR5 pressure representative were
checked. CSV-only runs retain their existing source selection without requiring
Catalyst. These are bounded integration checks, not new production-scale or
physical validation claims.

## Wall CSV Curves

`input.products.channel.walls.example.jsonl` targets the existing 16-cubed bc41
channel fixture. The base in-situ options must be enabled, with explicit memory
budgets and `products='channel_walls'`; rendering may remain disabled. Enable the
existing wall statistics before startup to use the last record, or remove that
record. The catalog does not create an accumulator. CPU and GPU CSV extraction
do not require Catalyst.

`spanwise_index` is a global k node, not a physical-height selector or a spanwise
average. The current producer supports j-normal walls only. SBLI physical-midspan
interpolation uses the separate explicit mode below; no node definition is replaced.

The CSV contains global i, physical x/y/z and the selected quantity for each
included node. Physical `bounds` select nodes; no new ROI endpoint interpolation
is introduced. The existing inward wall normal and projected global +x tangent
set the signed shear. Heat is positive into the gas. `skin_friction` uses
`2*tau/(rho_inf*U_inf^2)` and `p_pinf` uses `p_wall/p_inf`, with required finite
nonzero reference values. Current numerical evidence covers all five quantities
on the bounded channel fixture.

`source='reynolds', statistics_source='wall-time'` consumes the existing clipped,
time-weighted wall scalar mean, not a mean computed from the current velocity.
Favre wall scalars are not accumulated and are rejected. Each CSV records the
requested window, actual covered duration and source sample count separately.

Within a completed step and source, the existing wall producer runs once.
Quantities sharing a side and k index reuse the same one-dimensional packet.
GPU extraction currently computes the existing wall fields on the device, then
downloads only the selected line (coordinates and three existing quantities).
Formal statistics and their transfers are accounted separately. This is not
evidence that every wall source allocation can already be recovered locally.
Without a device-processing base configuration, these GPU CSV exchanges use
the existing pinned staging transport. An explicitly requested device-aware
transport is checked and is not silently replaced.

For native dynamic GPU wall products, a safely agreed instantaneous wall-workspace
budget/allocation refusal disables dependent instances and releases that capture;
other sources and plane images continue. Retry requires a new `enable_request`.
This does not change formal statistics failure handling or promise recovery from
MPI failure, a damaged CUDA context or an incomplete shared-transport initialization.

## Wall Charts

`input.products.channel.wall-charts.example.jsonl` contains four complete chart
definitions. Use a GPU solver, an enabled Catalyst device-processing base
configuration and either explicit device pipeline or `compatible` consumer in
the current device-rendering build; keep `processing_backend='device'` and the
same `device_render_pipeline.py`. It still uses the GPU wall field producer and
does not admit a CPU solver renderer or silently select a different producer.
Mean charts additionally need the existing wall-time statistics enabled before
startup. CPU wall CSV remains available.

Wall charts reuse the existing small-host Matplotlib encoder within the Catalyst
session. This is the existing one-dimensional host-data exception, not a CPU
fallback for three-dimensional device geometry. Install `InsituTools`, including
`sbli_products.py`, and provide Matplotlib, scienceplots and their LaTeX runtime
in the embedded Python environment. Three-dimensional products still use the
selected ParaView device pipeline.

Set `display.y_range` and `display.line_width`, not a 3D camera. The physical x
bounds set the x-axis range. Image size, JPEG quality and DPI are explicit. JPEG
is required; EPS is optional per instance and remains off when not requested.
Charts do not implicitly write CSV. Add a separate CSV record when needed; it
can share the selected line but has its own clock and output history.

ROI selection preserves original node adjacency: a gap in selected indices
breaks the plotted line, without inserting physical ROI intersection points.
An empty selection is a valid empty chart. Receipts identify the host chart
backend, source precision, quantity range, line payload, connectivity flags and
actual wall-time provider. Private MPI staging is not included in the selected
line download counter.

Current evidence covers channel charts, an existing y-wavy wall fixture and
noncatalytic AIR5 HBL pressure with an empty wall rank, plus hot changes, isolated
image publication failure and GPU exact restart. These short checks are not
production SBLI validation. The inner dynamic control format is `ASTRRC04`, outer
`ASTRIR07`, without migration of earlier experimental control versions.

## SBLI Physical-Midspan Curves

`input.products.sbli.midspan.example.jsonl` supplies one CSV and three independent
charts for the existing bounded HCW/SBLI fixture. Select `sampling.mode='wall_midspan'`,
`axis='j'`, `side='lower'`, physical bounds and an instantaneous source; do not give
`spanwise_index`. This samples the physical midpoint `(z_min+z_max)/2`, not a
spanwise average or a chosen node. The existing producer evaluates wall quantities
on the two bracketing spanwise layers and then interpolates those quantities.

Supported fields are `p_pinf`, `skin_friction` and `wall_heat_into_wall_normalized`.
Their existing normalizations are respectively `p_wall/p_inf`,
`2*mu*du/dy/(rho_inf*U_inf^2)` and
`kappa*dT/dy/(rho_inf*U_inf^3)` on the lower wall. The last quantity is positive
into the wall, unlike the node-line `wall_heat_into_gas` field. No time-mean
accumulator is introduced for this source.

The unchanged producer requires an admitted native-precision MP7-LD SBLI
x-slab, increasing separable Cartesian axes and a stationary impermeable wall.
It is not a general CURVE or blowing-wall stress implementation. The dynamic
records share one capture and reduction per complete step, whether CSV or chart.
The current kernel still uses its existing private two-dimensional workspace;
only the small wall packet is downloaded, and no plane image is implicitly emitted.
NP=1/2 and both device bridges have bounded numerical/continuation evidence;
this is not production physical validation or evidence for arbitrary SBLI grids.

## M12 Short-Fixture Products

`input.products.m12.short.example.jsonl` is the complete six-instance list used
for the 64 x 32 x 24 interval, NP=2 x-slab local integration check. Build the root
CMake with the original M12 UDF and use the existing M12 input resources; this
file does not replace the flow, grid, boundary or wall-blowing input. Select an
enabled `products='m12_diagnostics'` device-processing base configuration with
GPU derivatives, explicit budgets, the selected device pipeline and
`device_render_pipeline.py`. The native path no longer requires legacy
`product_ids` or the old fixed probe map. Product identity, fields and clocks
come from this JSONL, not an implicitly emitted legacy profile.

The example explicitly requests an index section, physical plane, Q=0.001
surface, instantaneous layered64 streamlines, a numerical section and pressure
probe. Its selected indices refer to the short fixture. Cameras, bounds and
ranges are local diagnostic choices, not production framing or transition
criteria. Replace all indices and explicitly select suitable display parameters
before using a different grid; the original full grid is not covered by this
local validation.

The bounded M12 preparation/inspection tool and platform acceptance tool now use
this six-instance example directly. The full-grid platform tool instead requires
an explicitly reviewed full-grid JSONL and never rescales these fixture indices.
Its paired images-off control retains only numerical records from that list.
An initial complete state is eligible for the first frame; using an extremely
large interval is not a way to turn a product off. Remove the image records.

Native base configurations also ignore legacy slice coordinates, seed layout,
preview-view file and mean-display switches. They retain fixed compatibility
envelope values for the admitted source family; the product geometry and source
remain exclusively in JSONL. The standalone `insitu-preview` command still uses
its original explicit configuration and is not canonicalized this way.

The current evidence uses dt=0.02 and ten complete steps with original
`legacy_random` wall blowing. Both device pipelines preserve the authoritative
state; same-topology CPU/GPU physical fields and numerical diagnostics meet the
existing absolute threshold, and same-backend 10 versus 5+5 restoration preserves
the state and image pixels. This is software integration, not turbulent or
statistically stationary validation. Private postprocessing transport was pinned;
no M12 device-aware production result is claimed here.
