# KEEP With Existing Low-Order Physical Boundaries

The deterministic FP64, nonreacting five-variable examples use a 16 x 16 x 16
interval mesh, four RK3 updates, fixed dt=1e-5, and no main filter or crashfix.
They demonstrate the existing boundary integration, not resolved channel or
flat-plate or cavity physics. Time and field values are solver-dimensionless.

| Example | Boundaries x-/x+/y-/y+/z-/z+ | Setup |
| --- | --- | --- |
| `wall` | 1/1/41/41/1/1 | Existing internally generated stretched channel grid, Re=3000, Ma=0.3, Tw=1, constant x forcing=1e-4; smooth prescribed initial field |
| `open` | 11/22/41/52/1/1 | Flat stretched tensor grid, dimensions 10 x 1 x 0.25, Re=1000, Ma=0.3, Tw=1; static profile inlet, nonreflecting top, weak 3D perturbation; no blowing or sponge |
| `ldc` | 41/41/41/0/1/1 | Unit cube with periodic z, Re=3000, Ma=0.3; stationary isothermal x walls and lower y wall, moving upper y lid u=1 and T=1; smooth prescribed initial field |
| `inflow` | 11/21/1/1/1/1 | File-defined perturbed freestream in a 2*pi x pi x 2*pi box; Re=3000, Ma=0.3, static inlet rho=T=u=1 and v=w=0, original pressure outlet with pout=1; periodic y/z |
| `plate` | 11/21/41/51/1/1 | Same stretched profile grid and weak perturbation as `open`, but the original pressure outlet and conventional farfield top; no NSCBC RHS |
| `incident` | 11/22/421/52/1/1 | Inviscid 1 x 1 x 0.125 box, Ma=2, uniform initial gas, static profile, GRCBC exterior oblique-shock reference at x=0.25 with beta=40 degrees; explicit KEEP only |
| `incident-compact` | 11/22/421/52/1/1 | Same inviscid incident fixture with globally coupled compact KEEP and 643c auxiliary selection |

Viscous examples use reference temperature 273.15 K and the existing Sutherland law. The
channel initial field is prescribed by computational index; it is not a fully
developed turbulent channel profile. The open example is not a shock case.
The `inflow` example reads its field with ninit=3 and uses the `uniform` label,
not a Taylor--Green initializer. Its static inlet uses the freestream target
constructor; it does not need an inlet.prof file.

## Prepare And Run

Build `astr` through the repository root CMakeLists.txt. Preparation requires
Python, NumPy and h5py, but does not import the tests or run a CFD calculation.
Set the paths below to your own root build, MPI launcher and new run directory.

```sh
ROOT=/absolute/path/to/astr_gpu
EXE="$ROOT/build_gpu/bin/astr"
MPIEXEC=/absolute/path/to/mpiexec
RUN=/absolute/path/to/new_keep_open_run
python "$ROOT/examples/KEEP_Low_Order_Boundaries/prepare.py" \
  --case open --backend gpu --output "$RUN"
cd "$RUN"
export ASTR_FORCE_MPI_TOPOLOGY=1,2,1
export ASTR_GPU_SYNC_MODE=explicit
export ASTR_GPU_HALO_TRANSPORT=pinned
"$MPIEXEC" -np 2 "$EXE" run datin/input.case > run.log 2>&1
```

Select `--case wall` for the channel or `--case ldc` for the cavity example.
Select `--case inflow` for the single-x static inlet/outlet example.
Select `--case plate` for the profile/pressure-outlet/conventional-farfield example.
Add `--viscosity me6_opti` to prepare the fixed nodal-end-band/ME6 variant;
the default remains `--viscosity standard`.
LDC uses the supplied default `userdefine` module, not an arbitrary custom UDF.
The top lid owns both top-side intersections: u=1, v=w=0, T=1 also at those
corners. `twall(1:3)` controls the three stationary faces; the supplied lid UDF
fixes its own T=1 rather than reading `twall(4)`. The geometry is axis-aligned
and tensor-product, including the validated orthogonal stretching option.
Select `--backend cpu` and your
FP64 CPU executable for CPU calculation. The generated `use_gpu` input controls
execution; preparation does not change how the executable is built.

## Independent Numerical Options

`--case y-farfield` supplies the separately verified lower-wall/upper-NSCBC box:
16 intervals, `1,1,41,52,1,1`, periodic x/z and the y-wavy mapping of amplitude
0.15. It uses existing explicit `643e/643e`, `nonreflecting`, no main filter,
four steps of 1e-5 and Tw=1. KEEP with standard nodal viscosity or ME6, and
original central convection with ME6, have CPU/GPU, MPI and exact-restart
checks. Its `flowtype='tgv'` is the source-free perfect-gas driver; the supplied
weak nonuniform field is not the classical Taylor--Green initial condition.
Reference Mach is 0.3, while its velocity amplitudes are much smaller than the
reference velocity. The separate profile-inlet/pressure-outlet `21/52` candidate
has a diagnosed CPU stale-halo failure and is not accepted by this example.

`--case incident` and `--case incident-compact` use reference temperature 288 K and
`diffterm=f`; the Reynolds
input is unused. It checks a nonzero top-boundary response over four steps, not
a resolved shock or a viscous SBLI. Its lower 421 wall is entirely slip/adiabatic
(`xslip=-1`), and the inlet owns the inlet/top intersection. The existing outlet
transverse filter remains active even though the main filter is off. Do not add
viscous ME6, dynamic inlet, wall forcing or the main filter to this recipe:
those combinations have not been integrated for this incident setup. The constant-state
and response matrix has NP=1/2 x/y/z/4 xy and exact-restart evidence. A separate
instrumented test uses extra CUDA stack space; normal runs do not configure it.
The compact example retains the original third/fourth-order physical end rows
and global MPI coupling, not independently closed rank subdomains. Both E/C
convection and independent E/C auxiliary selections have representative checks;
the auxiliary selection does not imply a viscous-operator test with diffusion off.

Edit the generated `input.case` before a new run:

- Add `--convection standard --viscosity me6_opti` when preparing an example
  to use original central convection with ME6 instead of KEEP. The independent
  `open` example has a four-step NP=2 y dynamic-inlet/modal-wall check with
  that selection; this does not select an upwind scheme or validate long-time physics.
- `conschm='643e'` or `'643c'`: explicit or globally coupled compact KEEP.
- `difschm='643e'` or `'643c'`: original nodal viscosity, independent of convection.
- `lfilter=t, filter_scheme='explicit10'` or `'compact10'`: existing main filter.
- `wall`, `ldc`, `inflow`, `open` and `plate`: `viscous_form='me6_opti'` selects the fixed low-order nodal
  physical-end band and internal ME6 operator. `difschm='643e'` or `'643c'`
  chooses that band's nodal gradient/divergence; the internal ME6 formula is
  unchanged. The LDC band is the union of the x/y physical-end bands, counted
  once at their intersections. `open` uses the same union for physical
  x/y ends and retains the existing profile/outlet/top intersection ownership.
  These are fixed nodal end bands, not pure half-point boundary closures.
  Other mixed open boundaries and `intx`/custom inlets are not implied.
- `plate`: the existing zero-normal-derivative top is selected by changing
  bctype(4) from 51 to 50 before a new run. It uses the original second-order
  extrapolation of rho/u/v/w/p and EOS reconstruction of T, not an adjacent-node copy.

For the recorded upwind/ME6 state-profile subset, prepare `plate` with
`--convection standard --viscosity me6_opti`. Before starting, select in
`input.case` either `conschm='543e', recon_schem=3, lchardecomp=f` (MP5 physical
reconstruction) or `conschm='743e', recon_schem=5, lchardecomp=t` (MP7-LD).
`difschm='643c'` uses the globally coupled nodal auxiliary/end-band operator;
the internal ME6 coefficients do not change. The progress file records MP5
without filtering or with explicit10/full, and MP7 with compact10/scalar,
including dynamic-inlet/modal-wall representatives. This recipe does not
extend the separate `open` NSCBC modes or imply shock stability.
  With ME6, original central
  convection (`convection_form='standard'`, `conschm='643e'` or `'643c'`) also
  has representative CPU/GPU evidence. This does not include upwind/ME6 or
  nonorthogonal physical-boundary geometry.
- `inflow` and `open`: `profile_inflow_mode='complete_state'` imposes the full
  inlet target. `profile_inflow_mode='mach_pressure'` instead fixes velocity and
  temperature with interior pressure extrapolation for target-normal subsonic
  inflow, while target-normal supersonic inflow uses the complete target.
  Set this in `boundary_options`; it does not select an NSCBC inlet.
- `open` only: top mode `compatibility`, `incoming_only` or `nonreflecting` in
  `boundary_options`. For `incoming_only`, explicitly provide
  `nscbc_farfield_rho=1, nscbc_farfield_u=1, nscbc_farfield_v=0,
  nscbc_farfield_w=0, nscbc_farfield_t=1` in that same group.

Explicit KEEP uses second/fourth-order rows near physical walls. Compact KEEP
uses the original third/fourth-order coupled endpoint rows. MPI interfaces do
not use independent physical-end closures. These are not globally sixth-order
boundary schemes, and no Hermite7 or new SBP closure is selected here.

The existing coarse compact-KEEP wall window without filtering or with
compact10 developed negative wall pressure. That unresolved numerical result
is not fixed or hidden by these four-step examples; do not infer long-time
stability or select a filter solely from a successful short run.

## Dynamic Inlet And Wall Forcing

For `open` and `plate`, add `--inlet-mode intp` to generate 12 native HDF5
fluctuation planes at intervals of 1e-5. Their nonpolynomial temporal modulation
checks window advancement; this short prescribed sequence is not a production
turbulent inlet or a periodic replay service. Other examples retain their
original static inlet behavior.

Optional `--blowing` recipes use amplitude 1e-3, beta=2, xa/xb/xc=2/5/8 and
two spanwise modes:

- `modal`: existing `wall_blowing_mode='current'`, two temporal modes and
  `wallbs_phase.dat` with 15 fixed phases.
- `counter`: existing `current` mode with zero temporal modes and the original
  global-coordinate deterministic perturbation.
- `legacy_random`: existing rank-local intrinsic RNG, preserving its seed and
  draw order. Its physical forcing changes with rank count/decomposition.
- `off`: no wall-forcing input files, the default.

For example, prepare a new open-boundary run with ME6, dynamic inlet and modal
wall forcing:

```sh
python "$ROOT/examples/KEEP_Low_Order_Boundaries/prepare.py" \
  --case open --backend gpu --viscosity me6_opti \
  --inlet-mode intp --blowing modal --output "$RUN"
```

This read-tensor-grid combination has a native four-step CPU/GPU NP=2 y
comparison. The detailed representative matrix is in progress LA3-5; it does
not establish all option combinations, nonorthogonal geometry or long-time
physical stability.

## Existing HBL Extrusion

`--extruded` for `open`/`plate` uses the repository's existing
`user_define_module/examples/userdefine_hbl.F90`. Its original grid producer
reads x/y from `grid.2d` and generates a periodic span of length 90; that length
and its mathematics are unchanged. The default empty UDF build cannot replace
it. With the NVHPC-compatible MPI/HDF5 deployment environment already loaded,
select that source explicitly through root CMake:

```sh
cmake -S "$ROOT" -B "$ROOT/build_keep_hbl_gpu" \
  -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=ON \
  -DASTR_NUMERICAL_PRECISION=fp64 -DASTR_WITH_CATALYST=OFF \
  -DASTR_USERDEFINE_SOURCE="$ROOT/user_define_module/examples/userdefine_hbl.F90"
cmake --build "$ROOT/build_keep_hbl_gpu" --target astr --parallel
```

For a CPU build use a different build directory and `ASTR_WITH_CUDA=OFF`.
Preparation, not the build, still selects runtime `use_gpu`:

```sh
python "$ROOT/examples/KEEP_Low_Order_Boundaries/prepare.py" \
  --case open --backend gpu --viscosity me6_opti --extruded \
  --inlet-mode intp --blowing modal --output "$RUN"
```

This writes the original 2D coordinate input, a native `flowini2d.h5` and the
selected inlet/wall inputs. The ASTR reader initializes w=0 for ninit=2; later
three-dimensional perturbations evolve normally and full checkpoints preserve
that evolved state. Progress LA3-6 records the actual CPU/GPU run, MPI/filter
matrix and native ninit=0/2/3 representatives. It is not arbitrary-UDF support.

## Checkpoints

The example keeps complete-step exact checkpoints at steps 2 and 4, and writes
no volume or slice sequence. To continue step 2 to 4, copy the same generated
inputs into a fresh run directory, create its empty output directory, and set
`restore_directory` in `input.output` to the absolute step-2 checkpoint path.
Keep the same numerical options, precision and MPI topology for exact restart.
The inlet cache is stored in `inflow.h5` when active. Wall parameters and used
phases are frozen with the new batch; they need not be copied into a restore
directory. The original random mode also restores per-rank RNG records and
requires the same topology/compiler runtime. Earlier batches without frozen
wall inputs are not newly certified as exact-restart sources.
Extruded runs freeze the original `grid.2d`, not just the generated 3D mesh;
read-3D-grid runs retain `grid.h5` even with a 2D initial field. Mesh resources
are selected independently of ninit. Earlier batches missing `grid.2d` are not
repaired by reconstructing it from a generated grid.
For an intentional method change explicitly choose `exact_restart=f`; that
does not promise a bitwise-equivalent trajectory.

Current evidence and remaining physical validation are recorded in
`documents/ASTR_GPU_KEEP_INTEGRATION_PROGRESS.md` in the source repository.
