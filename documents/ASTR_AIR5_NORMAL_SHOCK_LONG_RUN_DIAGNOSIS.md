# Air5 Captured Normal-Shock Long-Run Diagnosis

Date: 2026-09-23. Branch: `feature/gpu_dev` working tree.
Status: approved MP shortcut correction passes targeted regression. The fresh
matched interval was stopped at checkpoint 60; a new checkpoint-40 single-step
replay already crosses the same transverse threshold. Equal-duration smaller
steps suppress the first local growth, but a longer bounded replay shows
renewed growth and an intermediate-stage crossing without an intervening
restart. Flux diagnostics distinguish high-order pressure response from the
bulk-state sensitivity induced by trace-species positivity constraints.
Long-time physical acceptance remains open; no further algorithm was changed.

## Reproduction

The completed run used `GRID=32,6,6`, `NP=1`, `TOPOLOGY=1,1,1`,
`dt=8e-8 s`, `MAXSTEP=200`, `diffterm=f`, and `lfilter=f`.
The inclusive loop performs 201 updates, reaching `t=1.608e-5 s`.
It starts with the Mach-8 frozen Rankine--Hugoniot step at `x=0.01 m`
in a `0.02 m` domain. The source is coupled five-species/two-temperature
chemistry and V-T relaxation. No thresholds or production algorithms were
changed during the original diagnostic replay. Subsequent user-approved
pressure-outlet and MP corrections are recorded separately below.

Evidence roots under `tests/gpu_validation/out/`:

- `air5_c5_captured_normal_shock_20260922_long/`: original input, log,
  binary phase snapshots, checkpoints, and failed `short_contract.txt`.
- `air5_c5_captured_normal_shock_20260923_diagnosis/`: `replay.py`, isolated
  CPU/GPU one-step replay cases, geometry dumps, `cpu_gpu_replay.txt`,
  `cpu_gpu_same_phase.txt`, and `strict_physical_diagnostic.txt`.

The checkpoint at step 200 is at `t=1.6e-5 s`; the final binary phase
snapshot is after the subsequent complete update. They are not the same phase.

## Observations

| Quantity | Measured result |
|---|---:|
| Final reported CFL | 0.5848651 |
| Minimum density, short checker | 0.0337861005 kg/m3 |
| Minimum species density, short checker | 2.82692639e-8 kg/m3 |
| Minimum temperature, short checker | 450.276070 K |
| Species mass closure, short checker | 2.58127e-14 kg/m3 |
| Component-scaled transverse extrusion discrepancy | 1.87842e-2 |
| Maximum absolute transverse velocity, final full state | 0.515476 m/s |
| Pressure-jump location, final snapshot | 0.0159375 m |
| Final outlet pressure on the reference sampling line | 295851.39 Pa |
| Final outlet Mach number on that line | 0.680515 |
| Final outlet incoming acoustic speed, u-c | -422.634 m/s |

The extrusion discrepancy already uses each component's scale
`max(1, max(abs(q_component)))`. It is not an unscaled total-energy error and
must not be described as a 1.88% error in the complete physical state.
The initial and first post-chemistry snapshots have exactly zero transverse
variation. The first RK update introduces roundoff-sized variation; by the
end of step 0 the maximum transverse speed is about `4e-14 m/s`.
Subsequent growth is real in the numerical solution.

The strict steady-profile diagnostic also fails: downstream mass, momentum,
and total-energy flux errors are approximately 71.4%, 8.03%, and 42.8%.
Only four comparison nodes remain after excluding three shock-layer cells.
These discrepancies quantify failure to approach the chosen stationary
reference; they do not by themselves identify a chemistry-source error.
No physical-validation pass or downstream SBLI promotion is justified.

## CPU/GPU Replay

Both backends read copies of the same unmodified step-200 checkpoint and
perform one update with the original physical parameters. Nine comparable
phase snapshots pass the existing elementwise criterion
`abs(delta_q) <= 2e-9 + 2e-10*abs(q_reference)`.
Maximum absolute difference is `4.39584e-7` in dimensional conservative
variables; maximum component-scaled difference is `3.96873e-12`.
CPU and GPU both increase maximum transverse speed from `0.401165` to
`0.515476 m/s`. This local replay supports a shared numerical evolution at
the failing state, but does not establish full-trajectory CPU/GPU equivalence.

The initial replay report, including `pre_chemistry`, fails and is retained.
Its large difference is confined to the last x plane. GPU setup prepares
the open boundary before that snapshot, whereas CPU retains the checkpoint
boundary at that phase. Interior difference is at most `6.98492e-10` and the
first post-chemistry/boundary snapshot passes. The comparable-phase report
therefore excludes this known phase mismatch explicitly; it does not relax
the numerical tolerances or replace the failed report.

CPU/GPU geometry dumps agree. The Jacobian range is
`6.944444444444426e-9 .. 6.944444444444475e-9`, and diagonal metrics differ
from the Cartesian constants only at roundoff level. This rules out a
large geometric scaling error in this replay, not amplification of tiny
geometric perturbations by the nonlinear transport scheme.

## Boundary And Acceptance Mismatch

`src/chemistry_boundary.F90:apply_air5_postshock_boundary` and
`src_gpu/chemistry_boundary_gpu.cuf:air5_postshock_boundary_kernel` copy
all 11 components from `im-1` into `im:im+hm` for `air5normalshock`.
They do not impose a pressure target. The downstream flow is subsonic,
so one acoustic characteristic enters through the outlet. The current
extrapolation does not supply the independent back pressure used to select
the desired stationary shock and relaxation solution.

The independent `Air5PostShockReference` at a downstream distance of `0.01 m`
gives `p=402491.55 Pa`, `u=405.13447 m/s`, `T=4168.95493 K`, and
`Tv=4167.52848 K`. The pressure mismatch, incoming acoustic characteristic,
and downstream shock migration establish that the current run is not the
stationary reference problem. Outlet treatment is a concrete incompatibility
to resolve before another long physical acceptance run. It is not yet proven
to be the sole cause of transverse disturbance growth.

The flow-through diagnostic has a second problem: it uses the frozen speed
and the *current* shortened shock-to-outlet distance. As the shock moves
outward, this reports more flow-throughs despite loss of the intended domain.
For the originally prescribed post-shock length, integrating `dx/u_ref(x)`
gives about `2.32911e-5 s`, versus `1.54871e-5 s` using frozen speed.
The completed run is only about 0.69 of that reference transit time.
Neither elapsed transit time nor pressure alignment alone proves stationarity.

## Approved Change And Implementation

The user approved the pressure-outlet change on 2026-09-23. CPU and GPU now
implement an opt-in Cartesian x-max air5 pressure outlet. The state file may
contain `# outlet_pressure_pa=<SI pressure>`; absence preserves extrapolation.
Fresh initialization and restart both read this metadata. The generator's
`--outlet-reference-length 0.01` option derives the target from the independent
reference and leaves both frozen jump rows unchanged.

For an outward subsonic interior state, the frozen mixture sound speed is
`c^2=(1+R_mix/cv_tr_mix)*p/rho`. With `dp=p_target-p`, the implemented
linear acoustic correction is `rho_b=rho+dp/c^2` and
`u_b=u-dp/(rho*c)`. Composition, transverse velocity, and vibrational
temperature retain the interior primitive values. The existing air5
conversion routines reconstruct temperature and all eleven conservative
components, including formation and vibrational energy. This is a local
linear characteristic pressure closure, not the full multidimensional NSCBC
system. Supersonic outflow copies the interior state. Nonphysical target
pressure, inflow, nonpositive corrected density, and a correction crossing
the outward subsonic regime fail explicitly; there is no clipping.

The implementation is in `src/chemistry_boundary_state.F90`,
`src/chemistry_boundary.F90`, and `src_gpu/chemistry_boundary_gpu.cuf`.
Both runtimes reduce failure status across ranks. The default extrapolating
case remains available; the long-run driver selects the new pressure outlet
by default and uses `OUTLET_REFERENCE_LENGTH=0` only for legacy reproduction.
Its restart contract also pins the state-file content, so an existing
extrapolating run cannot silently resume with a changed pressure boundary.

The physical checker now integrates the reference transit time over the fixed
prescribed relaxation length. It additionally requires two temporally ordered
HDF5 checkpoints at matching phases, whose largest field-scaled primitive
change per reference transit is at most `1e-3`. The final checkpoint must be
within one checkpoint interval of the evaluated final time. This is a bounded
stationarity gate, not a replacement for grid or time-step convergence.
The long driver defaults to `MAXSTEP=600` (601 updates, `48.08 us`),
approximately two reference transit times; duration alone cannot yield a pass.

Completed bounded checks for the approved implementation:

- Root CMake builds `astr` and `air5_pressure_outlet_probe`; the latter
  passes pressure-increase/decrease, outgoing-variable, supersonic-copy,
  negative/NaN pressure, and flow-reversal checks.
- NP=1 and NP=2 x on `16x6x6`, plus diffusion-enabled NP=2 y on `8x12x6`,
  pass one coupled update at `dt=1e-8 s`. Across these runs, CPU/GPU
  maximum absolute/scaled differences are `1.16415e-9 / 1.90890e-14`.
  Maximum pressure-target relative error is `4.33855e-16`.
- Three-step continuous GPU versus checkpoint restart comparison passes with
  maximum scaled difference `6.63213e-14`.
- The pressure-enabled, diffusion-enabled NP=1 Compute Sanitizer run reports
  zero memory errors and zero leaked bytes; its pressure and state checks pass.
- The 27 affected Python tests pass, including reference-transit integration,
  stale-checkpoint rejection, opt-in pressure metadata, and a deliberately
  corrupted physical outlet that the pressure checker detects.
- Re-evaluating the old long-run data with the revised checker still fails:
  reference-transit count `0.690392`, stationarity metric `6.08675`
  versus the `1e-3` threshold. This result is a checker regression test,
  not a run of the new pressure boundary.

Evidence uses `tests/gpu_validation/out/air5_pressure_outlet_*` directories
dated `20260923`. The old failed long-run evidence is retained unchanged.
The corrected-boundary continuation below was stopped at its first observed
failed checkpoint gate; it did not complete the planned long run.

## Pressure-Outlet Long-Run Follow-Up

Evidence: `tests/gpu_validation/out/air5_pressure_outlet_long_20260923/`.
A fresh run, not a continuation of the legacy extrapolating case, used the
same `32x6x6`, NP=1, `dt=8e-8 s`, inviscid/unfiltered configuration and the
approved pressure target. Root CMake rebuilt `astr` before launch. The planned
limit was `MAXSTEP=600`; monitoring stopped the solver at checkpoint step 60,
after 60 complete updates (`t=4.8e-6 s`), on the first observed failed
transverse-extrusion gate. The signal-15 exit is intentional, not a solver crash.

The checkpoint monitor reconstructs all conservative components from HDF5
`rho/u/T/Tv/Ys` with the fixed air5 thermodynamics and uses the existing
component scale `max(1,max(abs(q_component)))`. It archives each inspected
checkpoint. At step 60, direct binary restart snapshots reproduce the same
error, excluding checkpoint reconstruction as the explanation for failure.

| Checkpoint step | Time (us) | Scaled transverse error | Maximum transverse speed (m/s) | Pressure-jump x (m) |
|---:|---:|---:|---:|---:|
| 20 | 1.6 | 1.33530e-11 | 8.41232e-11 | 0.0103125 |
| 40 | 3.2 | 9.97347e-12 | 1.45380e-10 | 0.0109375 |
| 60 | 4.8 | 4.11426e-9 | 9.38642e-8 | 0.0115625 |

At the failed checkpoint, the pressure-target relative error is `4.33855e-16`,
minimum density is `0.03459897 kg/m3`, minimum temperature is `492.771 K`,
minimum species density is `2.23297e-8 kg/m3`, and species mass closure is
`5.55112e-17 kg/m3`. The original `2e-11` extrusion tolerance is unchanged.
The transverse velocity is still small in absolute terms; this is a failure
of the strict planar-invariance gate, not evidence of a macroscopic flow crash.
Only `0.20609` reference transit times have elapsed. No downstream steady-profile
accuracy, grid convergence, or stationarity acceptance is claimed.

During setup the monitor was corrected to close HDF5 before waiting, avoiding
a long-held read lock. No checkpoint-write or HDF5 error occurred. This local
diagnostic correction did not change the solver or case inputs.

### Same-State Replay

`replay.py` runs CPU and GPU from copies of the same step-60 checkpoint with
the pressure target retained. Nine comparable phase snapshots pass
`2e-9 + 2e-10*abs(q_reference)`; maximum absolute/scaled differences are
`1.57394e-7 / 2.79131e-12`. As in the earlier replay, `pre_chemistry` is not
used for CPU/GPU boundary-phase acceptance. All three RK sensor masks agree
between backends and are uniform across y/z; their maximum raw-sensor
difference is `6.08513e-13`.

The phase metrics in `replay_phase_metrics.json` show:

- First chemistry half step: transverse error remains `4.11426e-9`.
- First RK stage: error becomes `5.13752e-9`.
- Complete RK transport: error decreases to `3.22866e-9`.
- Second chemistry half step: transverse momentum error is unchanged.

Both backends reproduce this behavior. Thus this replay localizes changes
in transverse momentum to transport, not direct chemical-source updates.
It does not establish monotonic growth per step, a GPU-only defect, or the
precise cause within the common spatial operator and positivity limiter.

`replay_half_dt.py` takes two `4e-8 s` GPU steps from the same saved state,
matching the physical duration of one `8e-8 s` replay step. The resulting
transverse error is `3.56146e-9` and maximum transverse speed is
`8.11181e-8 m/s`. The full-state maximum component-scaled difference between
these two integrations is `1.09625e-2`. This is a local time-step sensitivity
diagnostic, not a convergence rate or a fresh-start half-dt stability test.
Halving dt for this short replay does not restore the planar-invariance gate.

Remaining physical acceptance:

1. Isolate the common transport operator, sensor-selected central/LLF-MP7
   interface fluxes, and full-state positivity-limiter contributions using
   the archived states. Separate perturbation injection from amplification.
2. Determine time-step sensitivity from a matched fresh-start interval before
   launching another long acceptance run. The one-step replay is insufficient
   to claim temporal convergence or to exclude a time-step contribution.
3. After transverse stability passes, resume long-time location and stationarity
   checks, then x-grid refinement and independent relaxation-profile comparison.
4. Do not force transverse momentum to zero, project planes to their means,
   loosen the extrusion threshold, or add filtering as an unapproved repair.

The initial diagnosis made no production numerical changes. The subsequent
pressure-outlet implementation follows the user's approval. No Git operation
is included in this work.

## Read-Only Transport Diagnosis

The next investigation used the archived step-40 and step-60 checkpoints.
Temporary CPU instrumentation recorded the high-order/low-order face fluxes,
the left/right positivity factors, and the Jacobian. It did not change a flux
or a state. The instrumented step-60 replay matches the uninstrumented CPU
replay bitwise in all ten active-state snapshots (`atol=rtol=0`). The patch
is archived as `flux_instrumentation.patch` in the same evidence directory;
the temporary source instrumentation was then removed. No production
reconstruction policy has been changed.

`flux_summary_step40.json` and `flux_summary_step60.json` separate the
directional RHS contributions. At the locations of the largest transverse
momentum RHS differences, the adjacent positivity factors are exactly one.
For step 60, stage 1, the largest increment discrepancy occurs at `i=8`,
where `theta_left=theta_right=1` and the local sensor mask is false. Thus
the positivity limiter is not directly modifying the largest local RHS
contribution there. This does not exclude earlier or nonlocal limiter effects.
The lateral pressure-flux contribution is important; a transverse-momentum
variance increase alone is not a proof of growth of total acoustic energy.

The step-40 replay was initially rejected because its copied auxiliary file
still named step 60. The failed harness directory is retained. The corrected
`flux_cpu_step40_matched` case sets the auxiliary step to the checkpoint's
actual value; it does not change the checkpoint field data or bypass the
production restart consistency check.

### Confirmed MP Shortcut Scale Dependence

CPU `air5_mp5`/`air5_mp7` in `src/chemistry_solver.F90` and their GPU
counterparts in `src_gpu/chemistry_solver_gpu.cuf` use the early-return test

```fortran
(linear - center) * (linear - mp) < 1.0e-10_real64
```

The arguments are physical-space split flux components already multiplied
by the grid Jacobian and metric factors. The left side scales quadratically
with flux amplitude, whereas the fixed right side does not. Consequently,
the decision depends on variable units, component magnitude, and grid scaling.
It is not sufficient to describe this constant as a dimensionless roundoff
tolerance for the present dimensional air5 path.

`check_mp_scale.py` reconstructs the recorded step-60, stage-1, x-line
MP7 stencils. Its high-order face fluxes agree with the production records
to relative component scales below `7.55e-16`. It then evaluates the same
scalar routine on `v` and on `v/max(abs(v))`, restoring the output scale.
There are 86 scale-dependent candidate stencils on this sampled line;
8 belong to the first species, whose final face flux is overwritten by
mass closure. The other 78 affect used reconstructions, including transverse
momentum and trace-species components. This is a local algebraic diagnostic,
not a second flow solver or a revised acceptance criterion.

One concrete example is the negative split of component 3 at x face 9:

| Quantity | Value |
|---|---:|
| Maximum absolute stencil flux | 2.155979738964063e-12 |
| Shortcut product | 1.886446105618657e-24 |
| Current returned flux | -2.766275862278205e-13 |
| Normalized-then-rescaled result | 1.096851975184965e-12 |

The current shortcut accepts this oscillatory stencil without evaluating
the full MP bounds. Normalization changes the decision and even the sign
of this tiny interface flux. The 63.7% difference relative to this stencil's
amplitude is **not** a 63.7% error in the CFD solution. This proves the
scale dependence of the current implementation. It does not prove that
changing this shortcut alone cures the long-time planar-invariance failure.

This concern has a literature precedent: Mosta et al., *GRHydro: a new
open-source general-relativistic magnetohydrodynamics code for the Einstein
toolkit*, section 4.1.5, introduce a field-scale factor into the MP5 shortcut
and discuss setting its tolerance to zero near strong shocks or steep
contacts. That discussion supports investigating the criterion, not claiming
their relativistic-flow results validate this air5 case.
[Author-hosted paper](https://ccrgpages.rit.edu/~scn/papers/grhydro-et.pdf).

### Limited Negative Results

`frozen_momentum_operator.py` forms a small frozen-background x-advection
Jacobian for a transverse-momentum perturbation, using the observed mask and
positivity factors. It reproduces the recorded directional RHS to relative
scales below `4.21e-16`. The step-40/60 actual limited operators have maximum
SSPRK3 eigenmode amplification factors `0.999092/0.997287` at the current dt.
No unstable eigenmode is found in this scalar frozen subproblem. It excludes
a simple explanation based solely on that scalar x operator, not transient
nonnormal amplification, time-dependent masks, or the coupled acoustic system.
The hypothetical all-central control is not the production operator and
must not be used as evidence that the actual hybrid operator is unstable.

Geometry has no large defect in this sample: Jacobian y/z variation is
`7.44e-24`; small off-diagonal `dxi21/dxi31` entries are at most `2.38e-12`
in magnitude. These roundoff-level metric variations are possible initial
perturbation sources, but their causal role has not been isolated by an
ablation experiment.

### Approved MP Shortcut Correction

The user approved this correction on 2026-09-23. It is implemented only in
air5 CPU/GPU MP5 and MP7. It replaces the absolute
shortcut tolerance with the sign-equivalent interval test
`min(center,mp) <= linear <= max(center,mp)`. Otherwise evaluate the existing
full MP bounds. This corresponds to zero shortcut tolerance, avoids a product
underflow test, adds no tuning parameter, and retains the current MP
coefficients, SSPRK3, sensor, positivity limiter, and pressure outlet.
It does not unconditionally force first-order reconstruction.

`test_air5_mp_reconstruction.py` extracts and compiles the actual pure CPU
Fortran helpers, and checks algebraic agreement with the GPU device helpers.
Before the correction, the recorded-template and scale-homogeneity tests fail;
afterwards all four tests pass. They cover constants, affine data, smooth
quadratic extrema, the recorded oscillatory stencil, and 128 seeded random
stencils scaled from `1e-200` to `1e150`, including sign reversal. Preserving
these smooth polynomials is not a general mesh-convergence demonstration.
The eight existing shock-capturing source-contract tests and 27 normal-shock
and postshock-reference tests also pass. The root-CMake `astr` build passes.

New immutable evidence root: `out/air5_mp_interval_20260923/`.

- `replay.py` reuses the original step-60 checkpoint as a numerical diagnostic,
  not a new accepted physical initial condition. Nine CPU/GPU matching-phase
  snapshots pass the unchanged `atol=2e-9, rtol=2e-10` test; maximum scaled
  difference is `4.86610e-12`.
- `replay_algorithm_effect.json` compares corrected and original algorithms
  on that identical checkpoint. Final full-state scaled difference is
  `1.39109e-4`. GPU final extrusion is `3.228885e-9` instead of `3.228656e-9`,
  and maximum transverse speed is `8.00526e-8` instead of `7.34241e-8 m/s`.
  This is an algorithm change, not a failed CPU/GPU equivalence test or proof
  that existing transverse noise has been removed.
- `periodic_np2/` passes the five-update `64x8x8`, x-slab periodic shock-tube
  regression. All 24 CPU/GPU snapshots pass; maximum scaled difference is
  `1.01925e-13`, conservation drift `4.83459e-14`, and extrusion `9.10533e-15`.
- `fresh_control.py` runs a new initial condition with the original `32x6x6`,
  `dt=8e-8 s`, NP=1, coupled sources and approved pressure outlet. It retains
  the checkpoint extrusion gate `2e-11` and stops on failure. It passed the
  first-update contract and checkpoints 20/40, then stopped on checkpoint 60
  at `t=4.8 us`. The planned inclusive step-60 update was not completed.

### Fresh-Start Controlled Result

`fresh_control/checkpoint_monitor.jsonl` and `fresh_comparison.json` preserve
equal-time comparisons without reusing a failed checkpoint as the new initial
condition. The component scale is `max(1, max_active(abs(q_component)))`, and
each transverse plane is compared with its `y=z=0` x-line.

| Completed updates | Time (us) | Original shortcut extrusion | Corrected shortcut extrusion | Gate |
|---:|---:|---:|---:|---|
| 20 | 1.6 | `1.33530e-11` | `1.02617e-11` | pass |
| 40 | 3.2 | `9.97347e-12` | `1.76247e-11` | pass |
| 60 | 4.8 | `4.11426e-9` | `5.63028e-10` | fail, above `2e-11` |

At step 60 the discrepancy is smaller by about `7.3x`, but is still `28.15x`
the acceptance threshold. Improvement is not monotone across checkpoints.
The largest component is `q(4)=rho*w`, at zero-based global node `(14,2,4)`;
maximum transverse speed is `1.61436e-8 m/s`. Minimum density, temperature,
and species density remain `0.0346879 kg/m^3`, `494.772 K`, and
`3.42614e-8 kg/m^3`. Species mass closure is `5.55112e-17`, and outlet
pressure relative error is `7.23092e-16`. The largest logged CFL before the
stop is `0.5976890`. The pressure jump remains at `x=0.0115625 m` at this
checkpoint. The diagnostic stop is intentional, not a NaN crash or GPU fault.
The local MPI process and GPU context have terminated.

The first-update contract gives extrusion `1.12254e-14` and outlet-pressure
relative error `1.44618e-16`. This and the periodic regression establish
bounded implementation checks, not steady captured-shock physics. No new
Compute Sanitizer or performance claim is made for this correction.

### Next Diagnostic Boundary

The scale-dependence defect is corrected, but that correction alone does not
restore planar invariance over the matched interval. Continue with the
coupled transverse-momentum/pressure response and sensor-selected transport,
including thermodynamic feedback from the split source steps. The existing
stable frozen scalar x-operator does not establish stability of this coupled
system. Use archived pre-failure/failure checkpoints to isolate stage and
direction before proposing another numerical-method change. Do not silently
alter the sensor, full-state positivity limiter, chemistry tolerances, or
acceptance threshold.

Only a successful fresh-start control permits another long physical gate.
The long-time root cause remains open; the confirmed scale-dependent shortcut
defect must not be equated with a complete explanation of the instability.

## Corrected-Case Transverse Diagnosis

Evidence root: `out/air5_transverse_diagnosis_20260923/`. These are bounded
diagnostic replays of the corrected-MP fresh-run checkpoints, not extensions
of a failed production run. No numerical coefficients, source tolerances,
boundary conditions, or acceptance thresholds were changed. Input projection
and time-step variants below are diagnostic controls only.

### A Short Reproducer

From checkpoint 40 (`t=3.2 us`), one original `8e-8 s` update raises GPU
extrusion from `1.76247e-11` to `2.76304e-11`, crossing the `2e-11` gate.
The CPU replay gives `2.76286e-11`. Their nine matching-phase snapshots pass
with maximum scaled difference `1.80129e-12`. The corresponding checkpoint-60
CPU/GPU replay passes with `5.05549e-12`, but its complete step decreases
extrusion from `5.63028e-10` to about `4.91669e-10`. Thus growth is not monotone
from step to step. The checkpoint-40 replay is a short crossing reproducer;
it does not identify the first crossing in the original uninterrupted run.

### Exclusions Within These Replays

- Sensor masks are transversely identical in all three RK stages at both
  checkpoints, and the marked x interval does not change between stages
  (`8:20` at checkpoint 40; `9:21` at checkpoint 60). This excludes a mask
  mismatch or stage switch as the immediate cause in these two updates, not
  the influence of the spatial central/MP interface or earlier mask changes.
- Cached primitive density/velocity agree exactly with conservative recovery
  on active and directionally required halo nodes. Pressure and temperature
  differ only at rounding scale, at most `7.28e-16` and `5.16e-16` scaled.
  Periodic q-halo discrepancies are at most `8.38e-15` scaled in y and zero
  in z. No large stale-primitive or halo discrepancy was found; small periodic
  shared-plane averaging effects remain possible perturbation seeds.
- At checkpoint 40 the GPU chemical half steps preserve momentum exactly.
  Their pressure-increment transverse differences are `1.11e-9` and
  `1.80e-9 Pa`, versus transport pressure variation of order `1e-6 Pa`.
  Source-step feedback is therefore not the immediate dominant amplification
  in this replay. Its role in producing the background state is not excluded.

### Directional Production RHS

Temporary CPU logging recorded high/low face fluxes, positivity factors,
Jacobian, and the actual accumulated RHS after each direction. The initial
postprocessor regrouped floating-point sums and failed its reconstruction
check (`7.98e-9` scaled). That failed analysis was discarded. Recording the
actual accumulator instead gives bitwise agreement with the binary full RHS;
the telescoped directional sum differs by at most `2.22e-16` scaled.

`instrumentation_equivalence.txt` verifies ten step-60 snapshots are bitwise
identical with and without the first logger. `accumulator_equivalence.txt`
verifies ten step-40 snapshots are bitwise identical between logger versions.
The archived `instrumented_cpu_vs_head.patch` includes the prior approved MP
correction as well as temporary logging; it is not a new production patch.

At checkpoint 40, stage 1, the largest transverse `rho*w` RHS increment is
at `(11,2,4)`. Its x/y/z contributions to `dt*delta(rhs/jacob)` are
`+6.65e-13`, `+2.74e-14`, and `-1.76916e-11`. The relevant face factors are
all one. The analogous `rho*v` response is dominated by the y direction.
Stage-2/3 x faces can be limited, so this does not globally exclude the
positivity limiter.

The pressure response is evaluated by contracting the recorded conservative
RHS with the analytic derivative of the existing reduced-air5 EOS, in
`flux_metrics.py`. At stage 1 the maximum transverse differences in
`dt*dp/dt` from x/y/z are `3.42764e-6`, `6.55e-9`, and `5.64e-9 Pa`.
This localizes the dominant pressure response to streamwise transport and
the transverse-momentum response to the transverse fluxes. It does not yet
identify a faulty flux formula or prove an unstable eigenmode.

### Initial-Perturbation Controls

All controls start from the same checkpoint-40 x-line/background and use the
actual GPU ASTR time integrator for one `8e-8 s` update. Projection is applied
only to copies of the diagnostic input file, never to production states.

| Diagnostic initial state | Final extrusion | Maximum transverse speed (m/s) |
|---|---:|---:|
| Original checkpoint | `2.76304e-11` | `7.42128e-10` |
| Exact transverse extrusion, v=w=0 | `7.94998e-15` | `7.00646e-14` |
| Original thermodynamic/streamwise fields, v=w=0 | `1.91373e-11` | `5.14544e-10` |
| Extruded thermodynamic/streamwise fields, original v/w retained | `8.29870e-12` | `2.24349e-10` |

The isolated transverse-velocity perturbation decreases from its initial
`1.76247e-11` scaled amplitude. The other original fields regenerate transverse
momentum even when v/w start at zero. Together with the RHS budget, this
supports a pressure-related coupled response rather than isolated growth of
transverse-velocity advection. The unperturbed control excludes a single-step
injection of comparable magnitude, not accumulation of roundoff over many
steps. These controls are nonlinear; their results must not be treated as an
exact additive decomposition or as repaired physical solutions.

### Equal-Duration Time-Step Controls

All three start from the unmodified checkpoint 40 and end at `t=3.28 us`.

| Substeps | dt (s) | Final extrusion | Transverse pressure RMS (Pa) |
|---:|---:|---:|---:|
| 1 | `8e-8` | `2.76304e-11` | `1.09968e-7` |
| 2 | `4e-8` | `8.69839e-12` | `3.92318e-8` |
| 4 | `2e-8` | `8.69394e-12` | `3.60290e-8` |

The RMS uses deviations from each transverse-plane mean over internal x
nodes. The postprocessor centers already reference-subtracted pressures to
avoid spurious mean-subtraction residuals in an exactly uniform plane.
The smaller steps suppress this local crossing, making time discretization
a priority diagnostic. However, maximum component-scaled full-state
differences are still `1.33994e-2` between dt and dt/2, and `9.28143e-3`
between dt/2 and dt/4. Similar transverse maxima do not establish convergence
of the entire reacting shock state or long-time stability.

Next: continue matched-duration time refinement and locate the dominant
full-state discrepancy and limiting faces in streamwise transport. Then test
a candidate smaller dt from a fresh initial condition over the same physical
interval, without promoting it solely on the one-step result. Do not change
the sensor, switch reconstruction, remove chemistry, or project out transverse
motion as a production repair without a separately approved decision.

Temporary logging has been removed, CPU/GPU solver files match their
pre-diagnostic hashes, root-CMake `astr` rebuild passes, and 12 reconstruction/
shock-contract tests pass. No Git mutation, remote job operation, new memory
safety claim, or performance claim was made in this diagnostic round.

## Flux And Time-Refinement Localization

The follow-up uses the same evidence root and unchanged root-CMake executable.
`localize_refinement.py`, `pressure_flux_terms.py`, `limiter_candidates.py`, and
`verify_recorded_updates.py` postprocess actual ASTR snapshots/face records.
They do not advance an independent Python flow solver. No production source,
numerical coefficient, source tolerance, or acceptance threshold was changed.

### Bulk-State Differences And Transverse Error Are Distinct

The large full-state differences between dt and dt/2 occur around x indices
17--19, especially in streamwise momentum and vibrational energy. Excluding
the physical x endfaces does not remove them. For dt/2 versus dt/4, the maximum
is `9.28143e-3` in `rho*u` at x=17. These are differences between time-step
choices, not errors measured against an exact solution.

Summing the recorded source and transport increments for one full step and
two half steps closes the state budget to rounding accuracy. On internal x
nodes, the differences in density, momentum, and total energy arise entirely
in transport. The maximum component-scaled vibrational-energy difference of
the accumulated chemical increments is `3.05988e-4`, compared with
`1.37054e-2` for transport. This localizes the direct update difference, while
retaining the possibility of source feedback through the transported state.

At checkpoint 40, RK stage 1, the transverse pressure response in the recorded
x flux is already `3.52525e-6 Pa` for the high-order flux. The corresponding
low-order flux gives `6.42450e-7 Pa`; the limited flux gives `3.42764e-6 Pa`.
The maximum limiter correction to this response is `1.92819e-7 Pa`, and the
part attributable to transverse theta variation is `2.06072e-8 Pa`. Maxima
can occur at different nodes and must not be added as scalars.

At the high-order maximum `(12,4,1)`, the streamwise-momentum contribution to
the EOS differential is `3.61588e-6 Pa`, with density/energy and other terms
partly cancelling it. Thus the immediate pressure response is not primarily
created by a transverse fluctuation of the positivity factor. The high/low
comparison is an instantaneous recorded-flux diagnostic, not a time-evolved
first-order control or proof of an unstable coupled eigenmode.

### Why Trace Constraints Affect The Entire Shock Profile

`air5_admissible_face_ratio` checks `low_state + 6*theta*face_correction`.
The final face factor is the minimum from its two adjacent cells and applies
to all 11 conservative components. These are conservative sufficient
conditions; admissibility of the summed high-order candidate alone does not
justify bypassing the face conditions.

For the recorded checkpoint-40 update:

| RK stage | Inadmissible high-order candidate nodes | Nodes affected by limited faces | Affected nodes with admissible high-order candidates |
|---:|---:|---:|---:|
| 1 | 49 | 441 | 392 |
| 2 | 0 | 343 | 343 |
| 3 | 0 | 392 | 392 |

The first-stage inadmissible candidates are at x=15. Their minimum O and NO
partial densities are `-1.49171e-7` and `-4.60461e-8 kg/m^3`. These are
**unlimited diagnostic candidates**, not negative states accepted by ASTR.
The low-order candidates remain admissible. Turning the limiter off is not
a valid repair. At stage 1 the limiter changes `rho*u` by up to `2.31550e-2`
and vibrational energy by `2.81728e-2` in component-scaled units. This explains
why trace-species restrictions can materially affect the bulk shock profile;
it does not identify a programming error in the sufficient-condition design.

Every-step diagnostics give a more specific time-step dependence. With
`dt=1e-8 s`, O activates stage-1 limiting from the second substep, with N
joining later; the last recorded minimum factor is `0.57325`. With
`dt=5e-9 s`, only the final two of 16 updates report limiting, due to O,
with minimum factors `0.96949/0.95105`. Raising checkpoint/diagnostic frequency
changes none of the 12 compared active-node post-stage snapshots for either
run: both cadence comparisons are bitwise identical.

### Finer Equal-Duration Replays

Each row compares runs starting at checkpoint 40 and ending at `t=3.28 us`.
The scale is the maximum absolute reference value per component, floored at
one, as in the preceding diagnostic. Physical x endfaces are included unless
the interior column is specified.

| dt pair (s) | Maximum full-state scaled difference | Interior maximum |
|---|---:|---:|
| `8e-8 / 4e-8` | `1.33994e-2` | `1.33994e-2` |
| `4e-8 / 2e-8` | `9.28143e-3` | `9.28143e-3` |
| `2e-8 / 1e-8` | `6.33114e-3` | `6.33114e-3` |
| `1e-8 / 5e-9` | `2.64352e-3` | `2.64352e-3` |
| `5e-9 / 2.5e-9` | `6.56319e-5` | `3.83197e-5` |

Final transverse errors for dt=`1e-8/5e-9/2.5e-9 s` are respectively
`8.69168e-12`, `8.69008e-12`, and `8.69558e-12`. The bulk-state differences
shrink considerably once limiting becomes less active. This is evidence of
time-step sensitivity coupled to nonlinear limiting, not a measured formal
temporal order or a closed long-time physical gate. The sampled final sensor
interval also differs between coarse and fine runs (x=8:20 versus x=9:20),
although it remains transversely uniform.

The original MP analysis gives the scalar sufficient restriction
`CFL <= 1/(1+alpha)`, or `0.2` for alpha=4, and reports practical computations
at `0.4`. Consequently, `CFL<1` alone is not a monotonicity guarantee. This
scalar result is not a proof of stability or instability of ASTR's hybrid,
multi-species, two-temperature system.
[Suresh and Huynh, NASA TM-107367, equation 2.10](https://ntrs.nasa.gov/api/citations/19970010128/downloads/19970010128.pdf).

### RK And Evidence Checks

Reconstructing the sampled internal RK updates from the saved origin state,
pre-RHS state, actual RHS, Jacobian, and SSPRK3 coefficients gives maximum
component-scaled residual `3.33068e-16`. No qsave/coefficient assembly mismatch
is found in these samples. This check does not test spatial accuracy.
`recorded_update_checks.json` verifies the executable and both CPU/GPU solver
files still match the pre-investigation hashes. No rebuild or sanitizer claim
is needed for this source-unchanged diagnostic extension.

### Smaller Steps Do Not Establish A Repair

The `dt=2e-8 s` replay was extended only as a bounded diagnostic from the same
checkpoint-40 state. The reported CFL remains about 0.147. Complete-update
transverse errors are `8.69394e-12` at `t=3.28 us` and `1.80569e-11` at
`t=3.52 us`: the initial decay is followed by renewed growth.

A short continuation through a saved checkpoint initially gave an
intermediate-stage crossing at `t=3.58 us`. Its overlapping step differs from
the unrestarted trajectory by up to `1.59286e-13` in transverse momentum,
comparable to the small margin above the threshold. That continuation alone
was therefore not used as decisive evidence. A matched 19-update replay from
checkpoint 40, with **no intervening restart**, was run instead:

| Final sampled update at t=3.58 us | Transverse error |
|---|---:|
| Before chemistry/transport | `1.97612e-11` |
| After RK stage 1 | `2.02205e-11` |
| After RK stage 2 | `1.98793e-11` |
| After RK stage 3 and final chemistry | `1.99609e-11` |

The intermediate stage exceeds `2e-11`, whereas the complete update remains
just below it. This is not a completed-update failure at this final sample,
nor proof that smaller dt will always fail. It does rule out interpreting the
first short-window decay as demonstrated suppression over time. No further
extension or long-time physical acceptance was attempted.

In the final sampled update, all sensor masks remain transversely identical,
but the x interval expands from 8:22 to 8:23 at RK stage 2. The stage-1
crossing precedes this particular switch. Earlier history and spatial
central/MP interfaces remain part of the coupled diagnostic, not globally
excluded mechanisms. Evidence is in `quarter_dt40_window4/`,
`quarter_dt55_window1/`, `quarter_dt40_uninterrupted19/`, and their phase
metric JSON files. Step indices are not interchangeable between dt variants;
physical elapsed time is defined by the copied checkpoint and controller.

### Current Attribution And Next Diagnostic

Two mechanisms must remain separate:

1. The immediate transverse-pressure response is dominated by the x-directed
   high-order flux, especially its streamwise-momentum component, and feeds
   transverse momentum. CPU/GPU equivalence, recorded RK algebra, and the
   earlier primitive/halo checks do not support a GPU-only or qsave defect.
   A specific faulty formula or unstable mode of the full coupled system has
   not yet been demonstrated.
2. Trace constraints activate a shared full-state face factor and modify
   momentum/energy as well as species. Their time-step-dependent activity is
   a material contributor to profile sensitivity. The sufficient-condition
   construction is conservative, not thereby a confirmed CPU implementation
   bug; actual negative unlimited candidates prohibit simply disabling it.

Next, isolate coupled streamwise momentum/pressure/composition perturbations
using the actual ASTR spatial operator and compare their stage responses at
the saved states. Revisit the coupled operator rather than relying on the
earlier frozen scalar transverse-advection eigenvalue result. A low-order
flux control, changed characteristic reconstruction, or a less restrictive
positivity construction needs an explicit diagnostic/design decision before
implementation. Preserve all production methods and thresholds meanwhile.

## Approved Long-Time Diagnostic (2026-09-23)

The next action above is superseded by a user-approved fresh long-time
observation before further operator changes. The small transverse disturbance
may originate in floating-point roundoff, but its long-time amplification has
not been classified. The material full-state differences between time-step
variants remain a separate issue, not machine noise.

- Case: existing Mach-8 air5 pressure-outlet normal shock; intervals `32,6,6`,
  NP=1, GPU 0, `dt=8e-8 s`, 625 complete updates (`maxstep=624`), target `50 us`.
- Frozen FP64 executable, explicit synchronization, corrected MP shortcut,
  coupled ROS-2/Strang source path, inviscid transport and no filter. Fresh
  initial state, no checkpoint restart or transverse projection.
- Only in this dedicated diagnostic, the old `2e-11` extrusion threshold is a
  recorded warning, not a stop. The original short regression and physical
  acceptance drivers are unchanged. No physical pass is issued by this driver.
- NaN, accepted negative species, nonpositive density, out-of-domain T/Tv/p,
  failed solver/source integration or missing final completion stop the run.
  The solver retains its internal checks at intermediate updates; checkpoint
  postprocessing is not a replacement for them. Wall-time cap is 12 hours.
- Checkpoint statistics every five updates (`0.4 us`), with retained HDF at
  steps 5, 60, 65, 200 and 620. The `5 us` window is bracketed by `4.8/5.2 us`;
  `16 us` is sampled exactly. Final `50 us` metrics use the step-624
  `post_chemistry/rk02` conservative snapshot, not the earlier HDF checkpoint.

`run_air5_long_time_diagnostic.py:metrics` records the legacy component-scaled
extrusion, maximum and volume-RMS transverse speed divided by upstream speed,
volume-mean transverse kinetic energy divided by upstream dynamic pressure,
pressure fluctuation RMS divided by upstream pressure, and species mass
fraction transverse RMS. Pressure/species fluctuations subtract the y-z mean
at each x. RMS integrals exclude duplicate periodic endpoint planes and use
trapezoidal x weights on this uniform Cartesian grid. Maxima still include all
active nodes. Floating-point averaging itself has a small diagnostic noise floor.
Positive-state extrema, closure, pressure-jump position and outlet pressure
are also retained. Domain inventory changes are not called conservation
residuals: this open domain would need boundary-flux accounting for that claim.

Six focused metric/contract tests pass. Postprocessing the immutable corrected
MP checkpoint at `4.8 us` reproduces legacy extrusion `5.63028e-10` while
`max(sqrt(v^2+w^2))/Uinf=4.75455e-12` and transverse-speed RMS/Uinf is
`8.12345e-13`. These diagnostics motivate observing the time history, not
declaring the original discrepancy harmless.

Live evidence root: `tests/gpu_validation/out/air5_long_time_diagnostic_20260923/`.
`contract.json` freezes the executable/input provenance and runtime modes,
`statistics.jsonl` contains the time history, `gpu/run.log` contains solver
output, and `status.json` distinguishes running/stopped/diagnostic-complete.
The executable provenance is the latest no-intermediate-restart replay, not
the earlier fresh-control binary before diagnostic rebuilds. Input text is
matched to the fresh control; `grid.h5` is regenerated because `lreadgrid=f`.
Two preflight attempts stopped before launching ASTR while these provenance
roles were separated. Neither attempted a time update.

At launch on 2026-09-23 16:40 CST, the job is **running, not accepted** under
`astr-air5-longdiag-20260923-r2.service`. User-service linger is enabled;
`Restart=no` prevents automatic resumption after a numerical failure. Solver
startup confirms GPU explicit synchronization and coupled chemistry; the
initial checkpoint passes state-domain checks. Historical cost suggests
roughly 4-5 hours, but chemistry adaptivity can change that estimate.

Startup verification: the first complete update finishes with CFL `0.5976890`
and its final `post_chemistry/rk02` q snapshot is bitwise identical to the
earlier fresh-control snapshot, including the stored halo. The reconstructed
active-state extrusion is `1.97963e-14`. GPU 0 was observed at 100% utilization
with ASTR PID 2891020 and 424 MiB process memory. The first step takes about
120 seconds because the initial chemistry transient is more expensive; it
must not be mistaken for a settled per-step timing or a performance result.

### Completed 50 us Diagnostic

All 625 updates finish normally in `17660.17 s`; maximum reported CFL is
`0.597689`. No hard invalid-state/source gate fires. Among the saved samples,
minimum species density is `3.31622e-8 kg/m^3`, and maximum mass-fraction
closure error is `2.94320e-13` (the final conservative snapshot, distinct from
the normalized HDF mass-fraction representation).

| Time (us) | Transverse velocity RMS/Uinf | Pressure transverse RMS/pinf |
|---:|---:|---:|
| 4.8 | 8.12345e-13 | 7.33405e-11 |
| 16 | 6.33310e-7 | 8.38260e-5 |
| 37.6 | 4.78764e-7 | 6.54548e-5 |
| 44 | 1.16175e-6 | 1.52257e-4 |
| 50 | 1.60396e-6 | 2.01446e-4 |

Final maximum transverse speed is `0.0275830 m/s`; pressure fluctuation RMS
is `1.00723 Pa`. The legacy component-scaled extrusion ends at `2.46514e-3`
and peaks at `4.00593e-3`. It is not a relative error in the whole flowfield.
The disturbances grow, partly decay, then grow again late in the run. A
roundoff-scale seed remains plausible, but neither harmless roundoff nor
long-time saturation has been established. Status remains
`diagnostic-complete-not-physical-pass`.

### Late-State Spatial Localization

Evidence: `out/air5_late_diagnosis_20260923/diagnose.py localize` and
`spatial_localization.json`. No production solver edits are made.

At `50 us`, maximum transverse speed lies at `x=0.00125 m` (near the inflow),
and the largest plane pressure RMS is `2.51674 Pa` at `x=0.008125 m`.
The pressure-jump midpoint lies at `x=0.0128125 m`; the pressure-fluctuation
maximum is therefore upstream of the captured shock. The outlet pressure
plane itself has only rounding-scale transverse scatter. This does not
exclude boundary feedback or establish the location where disturbances were
first generated.

All three final-update sensor masks remain transversely identical and mark
x node indices 11:23. Raw sensor values differ across planes, but do not
cross the branch threshold differently in these sampled stages. Pressure
fluctuation transverse Nyquist modes carry about 16.4% of its final weighted
spectral power; the largest mode is `(kz,ky)=(-1,1)` on the six unique nodes
per periodic direction. The observed state is not dominated by a pure
two-node checkerboard pattern. No claim of transverse resolution convergence
is made on this deliberately small extruded grid.

The next bounded comparison uses the unchanged frozen executable and the same
`49.6 us` checkpoint for CPU NP=1, GPU NP=1, and GPU NP=2 x-slab (`2,1,1`).
Each performs one complete update with `dt=8e-8 s`. Numerical acceptance uses
the existing elementwise `2e-9 + 2e-10*abs(q)` at matching phases; the known
startup `pre_chemistry` boundary-phase mismatch is recorded separately.
Shared x nodes are owned by the right rank when assembling the global field.
These comparisons test backend/decomposition consistency, not recovery of
transverse symmetry or physical correctness of the original long trajectory.

### Late-State Backend And Two-GPU Results

The three one-update replays complete. Their copied checkpoint SHA256 values
are identical, and all use the immutable long-run executable. Same-phase
elementwise comparisons retain `atol=2e-9, rtol=2e-10` with no new tolerance.

| Comparison | Comparable phases | Max component-scaled difference | Max error / elementwise allowance |
|---|---:|---:|---:|
| CPU NP=1 vs GPU NP=1 | 9 | 2.35247e-12 | 0.00566212 |
| GPU NP=1 vs GPU NP=2 x-slab | 9 | 4.45721e-15 | 0.0000373299 |

All nine matching phases pass in both comparisons. CPU/GPU startup
`pre_chemistry` is not comparable at the outlet because of the previously
documented boundary preparation order; its failure is retained in the report,
not hidden by a larger tolerance. The first post-chemistry phase matches.
The two GPU startup states are bitwise equal after global assembly. Duplicate
MPI-interface states have zero difference at each `pre_rhs` and at most
`5.82077e-11` absolute difference after updates (across dimensional q fields).

CPU and GPU NP=1 both increase maximum transverse speed from `0.02026745` to
`0.02224501 m/s`, about 9.76% over this one update. Chemistry half steps do
not change this maximum; transport does. The interior plane-demeaned pressure
RMS is `1.0718316 Pa` before chemistry, `1.0702311 Pa` after its first half,
`1.0982456 Pa` after transport, and `1.0963387 Pa` after its second half.
These RMS changes are not an additive field or energy budget, but again point
to transport rather than source-only growth in this sampled interval.

This evidence does not support a GPU-only or x-slab communication defect in
the sampled late-state update. It does not exclude a shared numerical defect,
boundary/transport coupling, or earlier-history effects. All existing physical
acceptance gates remain open. Next diagnosis should target the shared coupled
transport and boundary closures using the saved state, rather than assuming
that another longer run or a looser extrusion threshold is a repair.

Logs confirm rank 0 -> GPU 0 and rank 1 -> GPU 1. Separate ASTR processes
were observed on both devices, each using approximately 422 MiB. At one
sample GPU 0 was idle while GPU 1 was at 100%; in an x-slab shock case the
reacting post-shock region is unevenly distributed. End-to-end single-update
wall times were about `28.05 s` on one GPU and `28.85 s` on two GPUs, including
startup and diagnostics, not an accepted performance benchmark. Two GPUs are
available for subsequent validation, but this small x decomposition has not
demonstrated a speedup. Transverse slabs would require more than the current
six intervals to preserve the stencil/halo requirements, so they are not a
drop-in replacement for the identical-grid comparison.

Evidence: `out/air5_late_diagnosis_20260923/{cpu1,gpu1,gpu2}/`,
`cpu1_vs_gpu1.json`, and `gpu1_vs_gpu2.json`. All three bounded processes have
exited; no new long-running or remote job was launched.

### Diagnosis Deferred By User (2026-09-24)

The user requests skipping the present transverse-velocity-error investigation
and continuing C5-6B reacting-shock development. The proposed next transport/
inflow diagnosis is therefore deferred, not executed and not marked resolved.
Existing artifacts and acceptance failures remain intact. C5-6B2 top incident
shock boundary preparation may proceed as engineering work; physical, temporal
and grid-convergence claims remain unavailable. See the dated C5-6B decision
in `ASTR_GPU_CHEMISTRY_PORTING_PLAN.md`. Hard invalid-state/source failures
and any new CPU/GPU equivalence failures still stop their affected tests.
