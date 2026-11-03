# Runtime Postprocessing Products

## Explicit Unlimited Accounting For Catalog Statistics

For `products='catalog_stats'`, `host_budget_bytes=-1` and
`device_budget_bytes=-1` explicitly disable those artificial quotas. Set
`device_reserve_bytes=0` for no artificial free-memory headroom. This reuses the
existing nonallocating unlimited sentinel; it does not allocate a sentinel-sized
buffer. Monitoring and real allocation/representability errors remain active.
Finite positive budgets are still enforced when explicitly requested.

The same `-1` values are already supported by the separate `&output` I/O group.
Its `buffer_bytes` controls a streaming chunk, not total allowed output/memory,
and need not be expanded to an artificial maximum. Numerical tolerances,
integration length/step counts, field validity and GPU-residency contracts are
not resource quotas and are unchanged. Existing production defaults and frozen
historical inputs are not rewritten by the current goal's unlimited policy.

The local acceptance helper's `unrestricted=True` requests no external memory
increment or wall-clock quota, records resource use, and writes unlimited I/O
budgets. Pass explicit unlimited `input.insitu` fields as well; the helper does
not silently replace a caller's declared in-situ quota. Python resource reports
use null for unset increment/time limits. Four-rank shared-GPU accounting remains
enabled without using the old 2/4-GiB or one-GiB-reserve admission thresholds.

## Wall Shear Direction Glyphs

For a fixed-j lower-wall image, `field: "wall_shear_vector"` requests the two
complete tangent shear components. Choose `source: instantaneous`, or
`source: reynolds` with `statistics_source: selected-time`. The latter displays
the norm and direction of the averaged vector, not the mean instantaneous norm.
This is not a wall-velocity streamline or a separation-line diagnostic.

Explicit `extraction: {"stride":[8,4],"length":0.6}` specifies global i/k index
intervals anchored at index zero and a fixed shaft length in native coordinate
units. Sampling remains on actual grid nodes; the display interval is not a
change to the CFD grid. Arrow shafts follow the signed physical tangent vector.
Color represents its native norm. Exactly zero vectors are omitted without an
epsilon threshold, including a valid empty geometry for an entirely zero field.
Tiny nonzero vectors are retained; their directions may be noise-sensitive.
No statistical convergence or separation is inferred from an arrow map.

Shared nodes contribute once and periodic terminal aliases do not add arrows.
The source coordinate frame still includes the selected periodic closure.
Arrow point tiles can contain one local row/column, including an ROI ending
exactly on an MPI interface. Original triangle surfaces retain their existing
cell requirements. Explicit display stretching affects geometry only; the
arrowhead is compensated for that linear transform so that it remains readable.
It does not alter the physical shaft direction/length, shear or statistics.

Both strict device pipelines are supported in the bounded FP64 nonreacting
Cartesian lower-wall gate; host glyph construction is not implemented. Geometry
stays on the device. Six coordinate-bound scalars (48 bytes for a nonempty FP64
rank tile) are read back for the coordinate frame, in addition to small selection
control reductions. These are not flow-field or final-geometry downloads.
Per-frame receipts record glyph count, stride, native length and metadata scope.

`input.products.sbli.wall-vectors.example.jsonl` is the small GRCBC example, with
z display x20 and JPEG only. Its stride, length, camera and color range are
explicit example choices. It uses the same `catalog_stats` configuration and
budgets as the wall-footprint example. Changing display parameters does not reset
matching mean history; exact continuation uses the saved typed product definition.

## Two-Dimensional Complete Wall Footprints

Fixed-j `grid_index` slices at j=0 can display instantaneous complete wall shear
and Fourier heat flow, or their nodewise Reynolds selected-time means. The mean
is accumulated from each instantaneous molecular quantity: it is not viscosity
at the mean temperature multiplied by a mean velocity gradient. No spanwise
reduction is performed. Native wall-shear signs are retained; positive
`wall_heat_into_wall` denotes heat entering the solid.

Mean wall packets support density, temperature, `wall_shear_x`, `wall_shear_z`,
`wall_heat_into_wall` and `dynamic_viscosity`. Primitive pressure means remain
available as a separate primitive slice; arbitrary mixed primitive-gradient/wall
mean packets are not implemented. This path requires the admitted FP64
nonreacting Cartesian lower wall. The first actual footprint matrix uses bc41;
it does not verify every wall family or introduce wall glyphs/shear lines.

`input.products.sbli.wall-footprints.example.jsonl` contains two JPEG-only means
for the bounded 64x48x16 GRCBC fixture, with explicitly annotated z display x20.
Its image scale, camera and color limits are examples, not production defaults.
Both strict device pipelines consume the field/geometry without downloading
them. Explicit HDF5 data slices remain a separate, opt-in two-dimensional archive.

For GPU statistics without images, `products='catalog_stats',statistics=t,render=f`
may explicitly use `processing_backend='device',derivative_backend='gpu',`
`postprocess_transport='device-aware'`. This declares the independent transport
to the existing pre-MPI CUDA binder and does not require Catalyst. A plain host
configuration remains usable for CPU statistics. Currently, the GPU wall
accumulator itself uses aware private exchanges even when image strips use
pinned staging; these are distinct paths, not an implicit solver-halo change.
No new sanitizer suppression or silent transport fallback is introduced.

## Fixed-j Local Time-Mean Fluctuations

`fluctuation: true` selects the completed-step instantaneous primitive minus its
local accumulated mean, not a gradient or an RMS. It defaults to false. Use
`source: reynolds` or `source: favre` with `statistics_source: selected-time`;
these identify the mean reference. Only fixed-j `grid_index` slices are currently
admitted. Velocity components support either reference; pressure and temperature
support Reynolds fluctuations. A fluctuation of speed is not implicitly defined.
Fields retain their native dimensional/nondimensional units; no wall-unit or
amplitude normalization is introduced. Actual coverage is reported, and a mean
from the future end of the requested window is never substituted.

`input.products.tgv.fluctuations.example.jsonl` is a bounded 32-cubed TGV example,
not a turbulent flat-plate production preset. Numerical slices can instead use
`fields` and `output: {"channel":"data","format":"hdf5-xdmf"}` with no display.
For images, use `products='catalog_stats'`, `statistics=t`, `render=t`,
`processing_backend='device'`, `derivative_backend='gpu'`, an explicit private
transport and `standard-device` or `direct-device`. Declare the common statistics
window and existing host/device/reserve budgets. With `render=f`, the numerical
mean/fluctuation channel continues without Catalyst. Image products themselves
still require rendering to be enabled; they are not automatically converted to
data products. The existing renderer and selected histories are reused.

The first bounded gate covers CPU/GPU numeric TGV and GRCBC SBLI with NP=1/2/4,
same-topology exact numeric continuation, and TGV device images through both
strict pipelines. Images report the actual selected-time coverage and label
Reynolds/Favre fluctuations. Device fields and geometry are not read back for
these images. This is not statistical-convergence or production-layout approval.

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
payloads. Existing mean images need `velocity-time` accumulation enabled at
startup; the product catalog does not start or reset that dense provider.
The new `selected-time` provider is instead catalog-driven and currently serves
numerical statistics through `products='catalog_stats'`, not mean images.

## Selected-Time Station Profiles

`input.products.tgv.station.example.jsonl` selects the global i=8 station of
the 16-interval periodic TGV fixture. Use `products='catalog_stats'`,
`statistics=t`, `render=f` and an explicit `statistics_window` in the base
namelist. These are numerical-only FP64, nonreacting Cartesian products;
this example does not admit CURVE, AIR5 or arbitrary production configurations.

- `kind='station_profile'`, `output.channel='data'`, `output.format='csv'`.
- `sampling.mode='station_span'` averages the specified k interval using
  physical spanwise nodal length weights, including nonuniform spacing.
- `sampling.mode='station_line'` selects exactly one k node and retains
  pointwise temporal statistics along j. Separate lines need separate products.
- Both use `axis='i'`, a global station `index` and inclusive j/k
  `index_ranges`. The CSV records the actual physical x, y and span coordinates;
  it does not interpolate to a requested physical station.
- `source='reynolds'` or `'favre'` requires
  `statistics_source='selected-time'`. Each product has its own output clock;
  sampling still uses every eligible completed step, independent of that clock.

Request primitive names in `fields`, not a display `field`. Density-only and
pressure/temperature-only requests do not allocate velocity covariance state.
Requesting any velocity component selects the full three-component velocity
bundle: means, RMS and six covariance components in xx,xy,yy,xz,yz,zz order.
`rho_time_mean` is always the ordinary time mean of density, including in a
Favre profile. An explicit Favre-density request is unsupported, since it
would need an additional density moment.

Each local temporal mean/covariance is accumulated before the spatial
combination. Reynolds quantities use physical length weights. Favre means
combine `rho_time_mean * local_favre_mean`; Favre covariances combine local
density-weighted temporal covariances and divide by the spanwise mean density.
RMS is the square root of the combined diagonal covariance. Spatial variation
of the local temporal means is not added to that covariance. CSV
`rho_covariance_f_*` columns contain the positive-sign density-weighted
covariance, not its negative in a mean momentum equation.

Periodic terminal aliases and MPI shared nodes contribute only once. A full
periodic j profile repeats the first value at the physical terminal coordinate
for plotting. CPU/GPU use the same projection; on GPU, selected moments and
the spanwise reduction remain on device. Only a compact one-dimensional
weighted packet and one status integer are downloaded per reduction. Explicit
diagnostic slice archives and checkpoint staging are separate transfers.

Files are under `outdat/new/profile_products/ID/version.../segment.../`:
the effective `definition.json` and independent `frame.stepNNN.csv` files.
Metadata includes units, spatial measure, requested and actual statistical
coverage, duration and sampling-segment count. Windows may contain gaps;
duration, not just the envelope endpoints, describes the sampled coverage.
Exact same-backend/same-topology checkpoints preserve the selected moments
and previous endpoint in the existing statistics component. A failed product
is disabled independently; any remaining consumer can keep shared history.

### Traditional Wall-Unit Profiles

This TB1 extension has completed the bounded procedural/read-Cartesian M12
CPU/GPU topology matrices; the small GRCBC SBLI combination is covered below,
separately from production physics. The read-Cartesian M12
wall41/random-forcing checkpoint closure is described below.
`input.products.wall.units.example.jsonl` supplies the bounded 64x32x24
example: an explicit `wall_profile` with `wall_reference={}` and two
`station_profile` consumers. Each station adds `"wall_units":"wall-reference"`
to name that live wall product. It must contain the requested i station,
use exactly the same k range, and pair `station_span` with `wall_span` or
`station_line` with `wall_line`. No inlet scale, nearest station or alternate
reference product is silently substituted.

The original primitive means, RMS and covariance columns remain in the CSV.
New columns use the same statistical wall scale described below:
`y_plus=(y-y_wall)/viscous_length`, velocity and RMS divided by `u_tau`,
covariance divided by `u_tau^2`, and `rho_covariance_f_*` divided by
`rho_wall*u_tau^2`. The latter uses mean **wall** density, not each profile
node's density. Covariance signs are retained; no minus sign is introduced.
Pressure and temperature retain their original means and units; this option
does not invent pressure/temperature wall scales. A scalar-only profile can
therefore add y+ without allocating unused velocity covariance.

The wall reference is evaluated from current accumulated quantities, not the
last published CSV, so its output cadence may differ from the station's.
Combination requires identical positive-weight sampling history, not just
equal start/end times, duration or number of segments. The registry assigns
the same integer identity only to updates with the same history prefix,
completed step/time and exact quadrature weights. This is not a hash or a
growing sample log; identity comparisons are meaningful within that registry,
not between arbitrary independent jobs. A missing target, mismatched sampling
history or unavailable friction scale leaves normalized values empty and
reports the reason. Raw profiles and other products continue.

Matching requests reuse their original field histories. Selected moments and
large spatial reductions remain on GPU; normalization consumes only the
approved one-dimensional station and wall results. The additional wall
projection is a separate one-dimensional download and is included in timing,
not described as zero transfer. Saved effective definitions retain the target
ID after exact continuation. Selected-time statistics now store an explicit
v2 component with the coverage identity/counter; v1 selected-time components
do not contain that identity and are not accepted as exact v2 continuation.
No new sidecar or top-level checkpoint file is introduced, and other statistics
providers keep their existing formats.

### Thickness And Edge References

`input.products.wall.thickness.example.jsonl` adds the optional `thickness`
object to a Favre `station_profile`. It requires `velocity_x`, `pressure` and
`temperature` fields and a measured profile starting at the lower wall j=0:

```json
"thickness": {"edge_band": [24, 28], "explicit_upper": 21.234}
```

The integers are global j indices within that profile, strictly above the wall.
The upper limit is an optional **native physical y coordinate**, not a grid
index or distance from the wall. Omitting it requests only the delta99 integral
family; supplying it additionally requests a separately named explicit-limit
family. The numbers above belong only to the bounded M12 fixture. Choose an
external plateau away from shocks and sponge layers for a real case. A positive
mean velocity alone does not prove that a band is external.

Edge rho, pressure and temperature are physical-y trapezoid length averages of
their Reynolds means. Edge velocity is
`U_e=sum(w*rho_time_mean*u_Favre)/sum(w*rho_time_mean)`. A single-node band uses
that node directly. Edge viscosity is the case transport law evaluated at the
Reynolds edge temperature, `mu(T_e)`, not the mean instantaneous viscosity.
This local reference does not change the fixed freestream Cf/Cq denominators.

With `f=rho_time_mean*u_Favre/(rho_e*U_e)`, displacement and momentum thickness
integrands are `1-f` and `f*(1-u_Favre/U_e)`. Integrals use physical y, a wall
origin, trapezoids and a linearly reconstructed partial last interval. They
are second-order diagnostics, not sixth-order because the solver is high order.
Delta99 is the inner edge of the >=0.99 U_e component connected to the whole
declared edge band, found by linear interpolation. An internal overshoot is not
selected in its place; the number of above-threshold components is recorded.

The same CSV retains all raw mean/RMS/stress columns and adds
`thickness_integrals_available,cumulative_displacement,cumulative_momentum`.
Comment metadata `# thickness_NAME=VALUE` stores available edge rho/U/T/p/mu,
delta99, displacement99, momentum99, H99 and Re_theta99, plus the explicitly
named limit/integrals/H/Re_theta when requested. `H=displacement/momentum` and
`Re_theta=rho_e*U_e*momentum/mu_e`. Thicknesses and cumulative integrals use
native length; H and Re_theta are dimensionless. The availability header names
its ordered flags. Undefined quantities have no numeric metadata entry; invalid
cumulative cells are empty. A missing crossing or out-of-range explicit upper
limit leaves other defined results and the original profile intact. No
extrapolation, velocity clipping, reference maximum or empirical floor is used.

The current acceptance scope is FP64 nonreacting axis-aligned Cartesian lower walls
41/42/411/421, with +y pointing into fluid. Bounded M12 wall41 CPU/GPU seven-topology
formula and exact same-backend continuation checks have passed for generated
and read-Cartesian grids; the other wall families and production statistical
convergence still need their own checks. The small Cartesian GRCBC SBLI
combination below also passes its quantitative gate. Thickness uses the existing selected history and
one-dimensional result, with no new three-dimensional/device state or checkpoint
sidecar. Its options are broadcast, versioned and restored from the existing
effective JSON definition. The extra one-dimensional workspace uses the existing
resource budget; `station_thickness_diagnostic` timing includes its budget
collectives as well as the local calculation.

### Read-Cartesian Random-Wall Checkpoints

The original random wall-forcing mode now has an explicitly validated FP64
read-Cartesian M12 combination. The bounded gate uses 64x32x24, 743e MP7-LD,
643e viscous derivatives, no filter, boundaries 11/21/41/50/1/1, `ninit=2`,
`turbinf=prof`, `ASTR_WALL_BLOWING_MODE=legacy_random`, and ten steps at dt=.02.
It is an I/O/continuation gate, not a physical or statistical convergence claim.

The two-dimensional extruded input still freezes `grid.2d`. For `lreadgrid=t`,
the actual three-dimensional input is frozen as `resources/grid.h5`, together
with the original inlet profile, `wallbs.dat` and initial field. Restore uses
these resources even if the newly prepared working directory has none of its
original grid/profile/forcing/initial-field copies. Exact state/metric and
statistics restoration keep their existing files and options; use
`exact_restart=t` for the bitwise continuation contract. Default output settings
and ordinary non-exact restart behavior are not changed by this increment.

Before accepting this new combination, a one-time initialization check uses
the physical coordinates already present on the host: x/y/z must be
axis-separable, strictly increasing and agree across MPI shared axis nodes.
This check precedes the first RK step and checkpoint publication and is rebuilt
at each startup. Its small axis workspace uses the configured host budget;
`ASTR_OUTPUT_PROFILE_GRID` records bytes and `checkpoint_profile_cartesian_proof`
records initialization time. It neither downloads geometry nor adds a checksum
or checkpoint sidecar. It does not admit rotated or curved random-wall grids.

The RNG draw algorithm, rank-local order, first-call state, runtime signature
and topology checks are unchanged. They remain in the existing
`control.bin` ASTROC09/ASTRWR02 payload. CPU/GPU NP1, NP2 x/y/z and three NP4
mixed decompositions pass exact same-backend 5+5 continuation for full state,
statistics, control/RNG bytes and matching continuation CSVs. Statistics can
also be disabled without changing the authoritative field or preventing this
checkpoint path. Repartitioning, cross-backend exact RNG continuation, FP32,
read-grid NSCBC 22/52, compact-viscous combinations and other wall families are
not newly verified by this matrix.

### Small GRCBC SBLI Quantitative Gate

`input.products.sbli.statistics.example.jsonl` is for the existing 64x48x16
Cartesian incident-shock fixture, not a production default. It uses Ma=2,
Re=296000, reference temperature 148.9 K, Tw/T_ref=.8, a 32.6-degree shock,
743e MP7-LD convection, 643e viscous derivatives, no filter, and boundaries
11/22/41/52/1/1 with the complete-RHS GRCBC upper boundary. The field is
initialized with the oblique jump. The bounded gate is twelve steps at
dt=2.5e-4, statistics over [0.000125,0.002875], all products emitted every step.
It validates interfaces and numerical definitions, not developed SBLI or
statistically converged turbulence.

Wall transport/time means, fixed-freestream Cf/Cq, statistical Stanton number,
wall/grid units, Reynolds/Favre velocity moments, pressure/temperature means
and the two thickness families pass CPU/GPU NP1, NP2 x/y/z and NP4 xy/xz/yz.
The example's Tr=1.72 is an explicit test reference, not an automatic recovery
temperature model. Its station i=24 is at x=3.125; its external band is j=40..44
and its optional upper integration coordinate is y=.6. Choose these values
again for a different case rather than reusing the fixture numbers blindly.

GPU **selected wall statistics currently require CUDA-aware MPI for their
private stencil exchange**, independently of the solver halo backend. The
solver may remain pinned. A launcher that explicitly disables CUDA buffers
cannot provide these wall means; no host fallback is silently selected. The
bounded regression enables the MPI device-buffer capability and UCX while
keeping `ASTR_GPU_HALO_TRANSPORT=pinned`. With that capability disabled, only
the wall histories fail, dependent wall-unit cells are empty, and other valid
products and the main solver continue. Record the actual launcher/backend
configuration when reproducing the case.

Both new profile CSV writers now use an explicit three-digit exponent,
including for near-zero covariance. A value such as `1.843...E-110` remains
finite and readable by ordinary CSV tools; it is not replaced with zero.
This is a serialization correction, not a change to the moment algorithm.
The real-flow checks compare every native frame for portable finite values,
and compare RMS through variance while reporting the raw RMS difference.

## Complete Molecular Wall Quantities

The FP64, nonreacting Cartesian lower-wall path now has distinct fields
`wall_shear_x`, `wall_shear_z`, `wall_heat_into_wall`, `dynamic_viscosity` and
`temperature_gradient_x/y/z`. The thermal derivatives use the same private
seven-node sixth-order operator and physical metric projection as velocity
gradients, not saved RK gradients. This increment does not admit new AIR5,
CURVE or FP32 diagnostics. The tested real case is the existing 64x32x24,
ten-step M12 interface fixture; this is not turbulent/production acceptance.
Procedural Cartesian grids keep their existing admission. Read grids can use
these new fields with active `catalog_stats` after a one-time proof of finite,
globally consistent, axis-separable Cartesian coordinates and strictly monotone
physical axes. The complete lower-wall stress/heat path currently requires
inward normal +y; viscosity alone has no wall-orientation restriction.
Rotated/curvilinear grids are not admitted by this proof. An ordinary read-grid
request without the selected-statistics proof still needs separate admission.
Dimensional transport has operator checks, not this increment's real-flow validation.

Wall x/z shear uses the complete molecular Newtonian stress projected onto
the physical lower wall, including `mu*(du/dy+dv/dx)` and
`mu*(dw/dy+dv/dz)`. Heat is Fourier conduction `kappa*dT/dy`, positive into
the solid, not the older `wall_heat_into_gas` convention. Blowing enthalpy,
kinetic energy and viscous work are not part of this field. Material properties
use the case's current perfect-gas transport parameters and solver units.
The default nondimensional Sutherland reference currently uses 110.3 K;
the dimensional air formula uses its existing 110.4 K constant. No material
law or solver boundary is changed by these diagnostics.
In nondimensional output, `dynamic_viscosity` means
`mu_phys/(rho_ref*U_ref*L_ref) = (mu_phys/mu_ref)/Re`, including the `1/Re`
factor. It is not just `mu_phys/mu_ref`. Dimensional output uses Pa s.
The nondimensional conduction coefficient is `(mu_effective/Pr)/const5`,
with `const5=(gamma-1)*Mach^2`; dimensional conduction uses `cp*mu/Pr`.

Instantaneous numerical wall fields use an ordinary `slice`, `grid_index`,
`axis='j'`, `index=0` record. The wall fields select a physical bottom wall
with bc41/42/411/421; only bc41 has this increment's real-flow matrix.
Molecular properties do not include an unrequested turbulence closure.
Images using these new fields have not yet received a dedicated rendering gate.

For temporal wall curves use `kind='wall_profile'`, `source='reynolds'`,
`statistics_source='selected-time'` and CSV output. The
`input.products.wall.statistics.example.jsonl` indices refer to the bounded
M12 fixture, not a production mesh. Enable `products='catalog_stats'`,
`statistics=t`, `render=f` and an explicit `statistics_window` in the base
namelist. `wall_span` selects physical spanwise-length averaging;
`wall_line` selects exactly one k index without that average. Both select
global j=0 and explicit inclusive i/k ranges. Fields are density, temperature
and the four molecular quantities above; density is an implicit ordinary
time mean. Wall profiles are Reynolds averages of already computed
instantaneous quantities, not Favre averages or fluxes of primitive means.

The effective viscosity, complete stress and heat are computed first, then
accumulated every eligible completed step. Product CSV clocks are independent.
Periodic terminal selections map to unique physical nodes; a wrapped range is
sampled as its canonical pieces, without expanding across the unrequested gap.
Time endpoints, gaps, coverage, failures and exact same-topology continuation
reuse the selected-statistics registry and existing statistics checkpoint.

On GPU the stencil packets, molecular quantities and cumulative states stay
on device. Private stencil communication currently uses CUDA-aware MPI;
there is no application-side host payload fallback if it is unavailable.
The solver halo backend is a separate setting. This does not claim that MPI
internally performs no staging. Only status integers and final one-dimensional
weighted curve results are downloaded. Explicit diagnostic HDF5 sections and
checkpoint staging are separate, requested I/O paths.
Unused device coordinate packing is skipped for wall statistics.

CSV files are in `outdat/new/wall_profile_products/ID/version.../segment...`.
They preserve actual x/y/span, units, requested and actual time coverage,
duration, segment count and the full-stress/conduction definitions.

### Wall Reference Scales And Coefficients

An optional `wall_reference` object on `wall_profile` enables local statistical
wall scales and fixed-free-stream coefficients. The example
`input.products.wall.references.example.jsonl` uses the same bounded M12
indices as the raw wall example; its recovery inputs are examples, not
recommended production values.

```json
"wall_reference": {}
"wall_reference": {"recovery_temperature": 26.92}
"wall_reference": {"recovery_factor": 0.9}
```

These are alternative object values, not three entries in one product.
An absent object preserves the original raw CSV. An empty object adds
`u_tau`, `nu_wall`, `viscous_length`, signed `Cf` and `Cq_inf`, but no Stanton
column. The calculation uses the time/span-averaged wall shear vector,
density and effective viscosity: `u_tau=sqrt(|mean(tau_wall)|/mean(rho_wall))`,
`nu_wall=mean(mu_effective)/mean(rho_wall)` and
`viscous_length=nu_wall/u_tau`. It does not average instantaneous friction
velocities or instantaneous shear magnitudes.

`Cf=2*mean(tau_x)/(rho_inf*U_inf^2)` and
`Cq_inf=mean(q_into_wall)/(rho_inf*U_inf^3)` use the fixed case reference
state, with `U_inf` the magnitude of its velocity vector and x the Cartesian
streamwise direction. They do not substitute a local edge state.
Recovery temperature must be positive and in the case's native temperature
units. Alternatively, an explicit nonnegative recovery factor sets
`Tr=T_inf*[1+r*(gamma-1)*Mach_inf^2/2]`. Zero is permitted; no empirical
factor or layer classification is inferred. The two inputs are mutually
exclusive. Only with one of them does the table add
`St_stat=mean(q_into_wall)/[rho_inf*U_inf*cp_inf*(Tr-mean(Tw))]`.
This is a ratio of compatible means, not `mean(St_instantaneous)`.
In nondimensional cases `cp_inf=1/[(gamma-1)*Mach_inf^2]`; dimensional
cases use the case's cp in native units.

The reference request automatically accumulates temperature, both shear
components, conduction and effective viscosity together, with density
implicit, even if the explicit field list is shorter. Those raw means remain
in the CSV. Matching requests share the existing histories; changing only a
recovery input changes the product version, not the sampled field history.
The saved effective definition restores these inputs on exact continuation.
Unavailable derived values are empty CSV cells with explicit
`wall_scale_available`, `cf_available`, `cq_available` and, when requested,
`stanton_available` flags. Exact zero shear or recovery-wall temperature
difference is not replaced by a floor. Other quantities and products continue.

The same table also contains traditional grid wall units, using the row's
statistical `viscous_length` rather than an instantaneous or inlet scale:

- `y_first_plus`: distance between physical j=0 and the first off-wall node.
- `dx_minus_plus`, `dx_plus_plus`: adjacent physical cell lengths in decreasing
  and increasing global i, respectively. Nonperiodic missing neighbors are
  empty cells; periodic endpoints use the corresponding actual terminal cell.
- `dz_min_plus`, `dz_max_plus`: minimum and maximum physical cell lengths
  contained in the selected k span. A `wall_line` instead uses the cell(s)
  incident on its node, including the periodic seam when applicable.

Lengths are positive physical distances; the nondimensional columns have unit
one. They are not nominal uniform-grid intervals or a complete-domain y+
coordinate. If the common wall scale or an individual ratio is unrepresentable,
the corresponding cells remain empty. Geometry comes from small one-dimensional
host coordinate axes; these columns add no device field or geometry downloads.
Reference-table examples enable these columns with no additional switch.

The profile interface above has its own matching-history and validation scope;
the scalar/grid tables alone do not enable it. Boundary-layer integral product
output still needs integration.

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

## Coordinate Axes And Display Scale

Dynamic slice, Q-surface and streamline images now show native physical
coordinate axes by default, with gridlines off. Native SI coordinates use m;
dimensionless coordinates use 1. The coordinate annotation is independent of
the color-field normalization. See `input.products.tgv.coordinates.example.jsonl`.

The normal coordinate axis is hidden for an axis-aligned plane. VTK's special
2D text placement assumes world X/Y axes, so it is used only for XY planes;
XZ/YZ planes use the existing three-dimensional camera followers with the
normal axis hidden. This prevents Z titles/ticks from overlapping the image.
It does not change the source geometry, coordinate ranges or display scale.

An image's optional `display.coordinates` object accepts:

```json
"coordinates": {
  "visible": true,
  "reference_length": 2.0,
  "origin": [1.0, 0.0, 0.0],
  "display_scale": [1.0, 2.0, 1.0]
}
```

Omit `reference_length` for native ticks. Supplying a positive fixed length
changes ticks to `(coordinate-origin)/reference_length` without changing the
data or geometry; `origin` requires that length. Display scale is a positive
three-component actor transform, not a data filter. Its default is `[1,1,1]`.
Nonunit factors are printed on the image, and ticks retain their native or
explicitly normalized meaning. Camera position/focal point and up direction
follow the display map, but `parallel_scale` remains the explicitly supplied
display framing height; choose it to fit the exaggerated view.

Axes use actual product bounds. An axis-aligned plane shows its two varying
physical axes. An oblique plane retains a physical x/y/z bounding box; no local
plane-coordinate basis is implied. A logical grid-index surface is treated as
physically planar only when its corresponding physical coordinate is constant.
Very short projected axes put their named numerical ranges in the image note
instead of overlapping labels. This is text layout, not data resampling.

The native and normalized examples have identical geometry. Field values,
gradient definitions, streamlines and numerical archives are not transformed.
An unrepresentable display transformation disables only that optional product;
other products and the main simulation continue.

Formal simulation images come only from the dynamic product catalog; the old
fixed-preset production entry remains closed. Read-only streamline preview now
also has native-coordinate annotations. These use the existing physical display
window when one is explicit, otherwise the current globally reduced geometry
bounds. Axis-aligned planes retain their physical plane position and hide the
normal axis. Oblique planes retain global Cartesian axes, not an inferred local
basis. An empty rank does not hide a globally nonempty product; a globally empty
product hides its annotations. Only scalar counts and bounding boxes are
reduced; no geometry arrays are downloaded for the annotation. Existing camera
and clipping settings remain unchanged. The configurable display object above
still belongs to dynamic catalog products, not to a new preview option.

No-render VTK component checks and real 32-cubed TGV read-only previews pass for
NP1/2 standard-device. Two candidates change camera/window/line width while
reusing trajectories; source files and GPU flow/statistics remain unchanged.
The direct-device mapper does not support the preview's display clipping planes
and is not newly admitted here; direct-device dynamic-catalog images are a
separate path. Production-resolution text layout and new FP32/AIR5/CURVE
annotation acceptance are not claimed by these bounded checks.

First device-image checks use 32-cubed FP64 TGV, NP1/2 standard-device, NP2
direct-device and NP4 `2x2x1` standard-device with pinned private transport.
The NP4 shared-device fixture uses explicit configuration budgets and normal
resource monitoring. It does not admit arbitrary new topologies or cases.

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

## Instantaneous And Mean Wall-Shear Lines

`input.products.sbli.wall-shear-lines.example.jsonl` selects
`kind="slice",field="wall_shear_lines"` on fixed global `j=0`.
Here `slice` identifies the two-dimensional wall support; the extractor traces
curves tangent to the wall shear, not a velocity streamline on a no-slip wall.

The implementation uses FP64, five-variable nonreacting states on increasing,
separable Cartesian x/z axes with the bottom-wall inward normal along +y.
It inherits the existing molecular-wall inputs for 41/42/411/421; this new
product's real-flow matrix specifically verifies the small bc41 GRCBC fixture.
Neither CURVE, AIR5 nor native FP32 is admitted by this extension.
Select `standard-device` or `direct-device`, not host/compatible processing.

Required `extraction` keys:

| Key | Meaning |
| --- | --- |
| `seeds` | Ordered global zero-based `[i,k]` node pairs, inside the sampling ranges. Order defines stable seed IDs. No automatic stride or seed placement. |
| `integrator` | `rk45`, using the existing Cash–Karp coefficients with wall-specific strict error acceptance. |
| `direction` | `forward`, `backward` or `both` along the selected shear vector. Both directions share the same seed order. |
| `initial_step`, `minimum_step`, `maximum_step` | Positive physical arc-length increments in the input coordinate units, with minimum <= initial <= maximum. |
| `maximum_error` | Positive local integration error control; this is not a spatial interpolation accuracy claim. |
| `maximum_length` | Requested physical arc length per direction, not a display-space distance. |
| `maximum_steps` | Positive representable accepted-step count. It is a numerical termination setting, not a resource/time quota. |

Select `source="instantaneous"`, or `source="reynolds"` with
`statistics_source="selected-time"`. The latter averages the instantaneous
complete viscous wall-stress components at each wall node. It does not use
mean viscosity times a mean gradient, average vector norms, or spanwise-reduced
stress. Components are bilinearly interpolated on actual x/z cells and only
then normalized for arc-length integration. Color is the norm of that displayed
interpolated vector. Physical integration remains unchanged by display scaling.

No empirical near-zero stress threshold is used. Exact seed/accepted-point zero,
trial-stage zero, accuracy-limited termination, accepted-step limit, requested
length, physical exit and sampling-region exit are distinguished. Failed trial
points are not committed. The short final remainder must also pass the error
test. There is no periodic wrap and no extrapolated exact boundary endpoint.
Insufficient local interpolation support fails the product instead of reporting
a false physical exit. The real fixture uses three private support layers.

The ordinary fluid tracer's min-step policy is unchanged. Wall arrays and line
geometry stay on the GPU, with only seed/bounds/current-particle control packets
read back. Pinned private MPI staging is reported separately, as for other
native products. Empty wall ranks participate; source/geometry bulk-download
counters remain zero on both device pipelines. Terminal seeds without accepted
line cells are metadata, not fabricated line segments.

Camera/line-width changes reuse the same frozen-frame integration. Changing the
seed list or integration settings reintegrates. Geometry is regenerated for each
new complete step. The effective JSON seed definition is restored through the
existing catalog checkpoint record, without another sidecar.
Selected-time history is independent of image cadence: a continuing consumer
preserves it, while actual gaps after removing all consumers remain excluded.

JPEG receipts contain `wall_trace`, including termination counts and the small
final particle packet. Traffic counters describe this product's integration;
reused views report zero new integration/control reads. This metadata is not a
host copy of the accepted trajectory. Formal numerical archives and checkpoint
statistics retain their separate I/O contracts.

The shipped example's 64x48x16 indices, three seeds, length 2, color range and
z display factor 20 are the local fixture settings. Change them explicitly for
another case; they are not automatic production defaults. Evidence and exact
scope are in `ASTR_INSITU_TURBULENT_BOUNDARY_LAYER_PRODUCTS_PROGRESS.md`.
