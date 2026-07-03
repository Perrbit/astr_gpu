# ASTR Source Architecture Memory

This note records the current understanding of `src/` for the CUDA Fortran port. It intentionally excludes `src_gpu/` except where the CPU/GPU boundary matters.

## Top-Level Run Path

`src/astr.F90` is the only main program. For `run`, the execution order is:

```text
mpiinitial
statement
listcmd
getcmd
readinput
mpisizedis
parapp
parallelini
refcal
fileini
infodisp
allocommarray
ibprocess
gridgen
solvrinit
geomcal
spongelayerini
flowinit
steploop
mpistop
```

GPU hooks in `src/` should remain narrow facade calls around `refcal`, allocation, flow initialization, and time integration. CUDA arrays and kernels belong outside `src/`.

## Global State

- `commvar.F90` owns scalar runtime state: local/global sizes, start/end active ranges, physics constants, schemes, flags, time-step counters, I/O frequencies, homogeneous direction flags, and runtime `use_gpu`.
- `commarray.F90` owns field arrays:
  - halo-backed arrays: `x`, `q`, `rho`, `vel`, `prs`, `tmp`, `spc`, `dxi`, `nodestat`, `lsolid`, `crinod`
  - active-domain arrays: `qrhs`, `dvel`, `dtmp`, `dspc`, `vor`
  - optional turbulence arrays: `tke`, `omg`, `miut`, `dtke`, `domg`
- Many modules mutate these globals directly. The CPU update order is therefore part of the interface, even when no explicit arguments show it.

## Domain Decomposition And Halos

- `parallel.F90` owns MPI setup, rank geometry, local ranges `is:ie`, `js:je`, `ks:ke`, boundary scheme selectors `npdci/npdcj/npdck`, reductions, gather/scatter helpers, and halo exchanges.
- `parallelini` sets active ranges and periodic/non-periodic neighbor ranks from `lihomo/ljhomo/lkhomo`.
- `qswap` is the reference halo and periodic-plane update for flow variables. For single-rank homogeneous directions it:
  - fills negative and positive halos from opposite interior sides,
  - averages duplicate periodic planes, for example `q(0)=0.5*(q(0)+q(im))`,
  - sets the opposite plane equal to the averaged plane,
  - refreshes primitive variables on halo/periodic slabs via `q2fvar`.
- GPU single-rank periodic handling must match `qswap`, not only copy one endpoint to the other.

## Numerical Setup

- `solver:refcal` determines `ndims`, `numq`, nondimensional constants, reference states, and thermodynamic constants.
- `comsolver:solvrinit` initializes derivative and filter objects:
  - central schemes use `derivative:fds%central`;
  - suffix `e` selects explicit central differences;
  - suffix `c` selects compact central differences;
  - central convection and diffusion schemes are expected to match.
- `filter.F90` contains compact filter setup and explicit filter coefficients.
- `derivative.F90` contains compact and explicit finite-difference operators. `diff6ec` is the CPU reference for explicit sixth-order central derivatives.

## Time Integration Contract

`mainloop:steploop` calls `crashcheck`, `time_integration_rk`, checkpoint/controller updates, CFL reporting, user end-of-step hooks, then increments `nstep` and `time`.

Inside `mainloop:time_integration_rk`, each RK substep follows this CPU order:

```text
if lfilter: filterq
set lreport
qrhs = 0
if immersed-boundary: immbody; qswap
boucon
qswap
gradcal
if first RK substep:
  qsave = q * jacob
  rkfirst
rhscal
RK update:
  q = (a*qsave + b*q*jacob + c*qrhs*deltat) / jacob
spongefilter
updatefvar
```

This order is critical for validation. `rkfirst` writes monitoring/statistics/checkpoint outputs at the same state that CPU uses.

## RHS Contract

`solver:rhscal` is the reference RHS assembly:

```text
convrsd... accumulates +conv into qrhs
qrhs = -qrhs
if diffterm: diffrsdcal6 adds +diff
flow-specific/user/chemistry sources are added afterward
```

For central schemes, `convrsdcal6` constructs contravariant fluxes using `dxi`, `jacob`, primitive variables, and `q`, then differentiates with `fds%central`. It writes active ranges `is:ie`, `js:je`, `ks:ke`.

`diffrsdcal6` depends on `gradcal` outputs (`dvel`, `dtmp`, `dspc`) and builds stress/heat/species fluxes before differentiating them with the same finite-difference abstraction.

## Flow Variable Conversion

`fludyna.F90` is the reference for:

- equation of state via `thermal`,
- conservative update from primitives via `updateq` / `fvar2q`,
- primitive update from conservatives via `updatefvar` / `q2fvar`,
- nondimensional pressure/temperature/energy constants.

GPU conversion kernels must match these formulas for the active first-stage case before extending to species, chemistry, or turbulence.

## Initialization, Geometry, And I/O

- `readwrite.F90` reads runtime input and controller files, writes checkpoints, flowfield HDF5 files, slices, meanflow, and monitoring files. `use_gpu` is read and broadcast here as runtime state.
- `gridgeneration.F90` creates analytic grids when `lreadgrid=f`; for TGV it generates the cube grid.
- `geom.F90` computes geometry metrics: `jacob`, `dxi`, node/cell state, immersed-boundary geometry, and solid-grid relations.
- `initialisation.F90` selects flow initialization by `flowtype`; `tgvini` is the first-stage reference.
- `hdf5io.F90`, `mpiio.F90`, `tecio.F90`, `vtkio.F90`, and `stlaio.F90` are I/O backends and should not be mixed into numerical kernels.

## Diagnostics And Validation

- `statistic.F90` computes monitoring quantities such as TGV kinetic energy, enstrophy, and dissipation.
- TGV `flowstate.dat` columns are produced through `statcal` and `statout`, not by post-processing HDF5 fields.
- `enstophycal` and `diss_rate_cal` depend on `dvel`; therefore statistics are meaningful only if `gradcal` has run on the state being reported.
- GPU statistical validation should first reuse CPU `statcal/statout` after a correct GPU-to-CPU field copy and CPU `gradcal`, then move diagnostics to GPU only after L1/L2 numerical agreement.

## GPU-Porting Implications For `src/`

- Keep `src/` as the CPU reference and orchestration layer.
- CPU-side GPU calls should go through `gpu_runtime` only.
- Do not import GPU implementation modules into CPU reference files.
- Do not duplicate CPU formulas casually; when GPU results differ, compare against the precise CPU module and update order above.
- Validate with both field diffs and native statistics, because statistics expose missing `gradcal`, `qswap`, or state-order mismatches quickly.
- Current first-stage TGV GPU loop keeps full flow variables resident on device across time steps. Native `flowstate.dat` is written from GPU reductions.
- Current project scope intentionally excludes GPU porting for file output, checkpoint, and HDF5 `flowfield` writing. Full-field D2H at those CPU-owned output boundaries is acceptable and must not be counted against GPU-resident compute-loop status.
