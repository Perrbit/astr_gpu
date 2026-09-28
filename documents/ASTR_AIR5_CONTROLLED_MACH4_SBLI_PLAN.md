# Controlled Mach 4 AIR5 SBLI

Date: 2026-09-28. User-approved self-defined case, not a reproduction of
Passiatore 2023 or a NASA experiment. No solver model changes are included.

## Physical contract

Frozen translational Mach number 4; inlet T=Tv=1500 K, pressure 20000 Pa;
mass fractions N2/O2/N/O/NO = 0.767/0.233/0/0/0. This is prescribed
nonequilibrium composition, not an equilibrium-air claim. Wall: no slip,
noncatalytic, T=Tv=3000 K. The hot-wall precursor can react without a shock.
Spanwise periodicity is used for the initial laminar extruded configuration.

The incident shock angle is 25 degrees, not the flow deflection angle.
Frozen jump: deflection 12.937449 degrees, pressure 63346.312859 Pa,
T=2177.259771 K and Tv=1500 K. Normal conserved-flux scaled residual is
4.75282e-16. Reaction and vibrational relaxation occur downstream rather
than being imposed as equilibrium across the discontinuity.

Initial domain is 30.03857 x 4.50579 x 0.750964 mm. The top target changes
at x=7.509643 mm; the geometric wall intersection is x=17.172332 mm.
Neither is a prediction of viscous separation or actual shock impingement.

## Prepared cases

| Directory | Shock | Source mode | Purpose |
| --- | --- | --- | --- |
| precursor | no | coupled | ASTR-only flat-plate development |
| flat_plate_coupled | no | coupled | hot-wall and upstream chemistry control |
| sbli_vt_only | yes | vt | frozen species with vibrational relaxation |
| sbli_coupled | yes | coupled | finite-rate chemistry and vibrational relaxation |

Generator: `tests/gpu_validation/prepare_air5_mach4_controlled_sbli.py`.
It refuses existing destinations and does not launch processes. Output:
`tests/gpu_validation/out/air5_mach4_controlled_sbli_20260928_v1/`.
Each directory contains `datin`, metadata and `environment.json`.
The JSON environment is a configuration record, not automatically loaded
by ASTR. No launch script or characteristic GPU admission opt-in is supplied.

All four templates presently contain the same quartic startup seed, not a
developed inlet. No old checkpoint is silently reused. The three comparison
cases remain held until inlet development and extraction are accepted.
Their input metadata explicitly records this preparation-only status.

## Startup settings

- 64 x 128 x 16 points (upper bounds 63,127,15), uniform Cartesian mesh.
  This is a short-gate mesh, not a resolved SBLI/DNS production grid.
- NP=2, topology 2,1,1; intended one rank per local GPU.
- FP64, explicit synchronization, compensated updates, symmetric species
  convection and layered diffusion limiters, filter disabled.
- dt=0.25 ns, maxstep=1, only a short-start template. No long-window step
  count is prescribed before local acoustic/viscous/chemical limits are checked.
- Characteristic top candidate tau=Ly/a_inf=5.79083446343305 us.
  Half/double tau sensitivity remains necessary for this incident-flow case.
- Left profile inflow 11, right outlet 50, lower wall 41, top 51 routed
  through the optional AIR5 characteristic mode, periodic z boundaries.
  Outlet 50 must not be described as a newly implemented characteristic outlet.

## Admission sequence

### 20 us continuation and scheduling estimate

The 40000-update run completed to 10 us and passed sampled first/last phase
state checks. Its rolling checkpoint is at step 39000, 9.75 us, not 10 us.
Two-update continuation from that checkpoint passed positivity, model-domain
and strict composition checks while restoring compensation. No bitwise replay
claim is made for this new short continuation.

`flowstate.dat` contains global maxima of q1--q5, not equation residuals or
wall/profile convergence. For scheduling only, relative changes between samples
1000 steps (0.25 us) apart were fitted as log(change) versus time. Fits using
4--10, 6--10 and 8--10 us give a time of about 12.7--13.2 us for q1/q2/q5
to reach a 1e-4 relative change per window, but 23.2--28.9 us for q3.
That extrapolation is sensitive to the window and to moving spatial maxima.
The 1e-4 level is illustrative, not an approved physical admission criterion.

The next run starts at the actual 9.75 us checkpoint and takes 41000 updates
at unchanged dt=0.25 ns, targeting 20 us (last update index 79999).
Service: `astr-mach4-precursor-20us-20260928`.
Evidence: `out/air5_mach4_precursor_20us_20260928/`.
Recent cost about 0.576 s/update implies about 6.6 hours for this window.
Reaching the extrapolated 23--29 us range would require roughly 53--77k
updates from 9.75 us, about 8.5--12.3 hours at the same cost; it is not a
convergence guarantee. Review again at 20 us rather than automatically extend.
Actual wall and candidate inlet profiles must still be checked before SBLI.

### Monitored continuation to 30 us

The 20 us window completed with finite sampled states and strict composition
checks. Its actual rolling checkpoint is step 79000, time 19.75 us. Late
global-maximum changes are not monotonic; the earlier 23--29 us extrapolation
is not a reliable convergence forecast. The old run has no probe/wall history.

On 2026-09-28 the user approved a monitored restart. The new executable replayed
the archived step-39000 phases from the preserved 9.75 us checkpoint with zero
scaled difference. Two-update restarts at 19.75 us with monitoring off/on have
bitwise-equal phase fields and conservative/compensation checkpoints. Evidence:
`out/air5_monitor_restart_gate_20260928/qualification.json`. The restart driver
requires this exact executable/checkpoint-bound qualification for a binary
transition; ordinary mismatched executables are still rejected.

Service `astr-mach4-precursor-30us-monitor-20260928` uses the frozen qualified
executable, NP2 x-slab, unchanged dt=0.25 ns and 41000 updates, targeting 30 us.
The output is `out/air5_mach4_precursor_30us_monitor_20260928/`. It restores
chemistry compensation and the characteristic top contract. Profiles and wall
diagnostics are sampled every 100 updates (25 ns), with rolling checkpoints
every 1000 steps and first/last phase snapshots only. The original run and its
checkpoint remain unchanged. A 24-hour timeout and no automatic retry bound
this run. Estimated duration is about seven hours, not a completion guarantee.

Review consecutive statistical windows before deciding whether to continue
development or extract inlet data. Reaching 30 us does not automatically admit
a developed inlet, turbulent statistics, or the incident-shock SBLI run.

### GPU window preparation

Both 50 ns windows completed with strict first/last-phase state checks.
The matched complete-step comparison is recorded in
`out/air5_mach4_precursor_50ns_comparison_20260928.json`: maximum scaled
conservative difference 6.21461e-11; absolute temperature, pressure and
velocity differences 2.03136e-9 K, 4.08763e-8 Pa and 2.49327e-9 m/s.
This is timestep sensitivity evidence, not a measured convergence order.

A 40000-update, 0.25 ns precursor (10 us, about one nominal flow-through)
was launched as local user service `astr-mach4-precursor-20260928-v2`.
Evidence: `out/air5_mach4_precursor_10us_20260928_v2/`.
It overwrites a rolling checkpoint every 1000 steps, keeps detailed phase
fields only at the first/last updates, and does not auto-launch SBLI.
Expected solver duration from the 50 ns window is roughly six hours, not
a completion guarantee. A 24-hour timeout and no automatic retry are used.
The first service attempt failed before time integration because it selected
system MPI; v2 explicitly inherits the tested PATH and LD_LIBRARY_PATH.
The failed attempt remains under `out/air5_mach4_precursor_10us_20260928/`.
After completion, inspect the inlet development and actual checkpoint phase
before extraction or restart; do not infer a developed inlet from elapsed time.

`run_air5_mach4_precursor_window.py` runs fail-fast GPU windows with immutable
input copies, executable hashes, first/last phase checks and optional memcheck.
The two-rank, two-update memcheck completed with two zero-error reports under
`out/air5_mach4_precursor_memcheck_20260928/` (relative to gpu_validation).
nvitop sampled 85% and 96% utilization with one solver process on each GPU.
This establishes device execution, not a performance benchmark.
The next bounded comparison targets 50 ns: 200 updates at 0.25 ns and
400 updates at 0.125 ns. Neither window establishes a developed boundary layer;
one nominal domain flow-through time is 9.65139 us.

### 2026-09-28 precursor startup result

Built `build_gpu_probe` through the root CMake project. The same executable
ran CPU NP1 and GPU NP2 x-slab using the prepared precursor. Driver:
`tests/gpu_validation/run_air5_mach4_controlled_startup.py`; evidence:
`tests/gpu_validation/out/air5_mach4_controlled_startup_20260928/result.json`.
Both completed updates 0 and 1, reaching 0.5 ns. Eighteen matched global
phase fields passed the 2e-10 scaled tolerance; maximum difference was
8.03256057748139e-16. Strict 128-epsilon composition checks passed with
nonnegative species. Physical-domain checks covered 18 CPU and 36 GPU
rank snapshots: temperature approximately 1500--3000 K and pressure
19999.999748--20000.767604 Pa. Maximum reported CFL was 0.0152886.

This run did not include Compute Sanitizer, a long development window, or
incident forcing. The nvitop snapshot missed the brief GPU execution and
does not establish sampled sustained utilization. No production admission
or developed-inlet claim follows from this result. Next: GPU memory-safety
gate and bounded precursor extension, followed by inlet-development checks.

1. Check generated input, units, exact-zero species, jump conservation and LF
   line endings. Completed: 8 focused generator tests passed.
2. Run CPU/GPU same-phase short gates with the actual wall and oblique top
   target; check positivity, composition closure, topology and memory safety.
   Existing uniform/acoustic gates do not admit this physical case automatically.
3. Develop the flat plate with ASTR. Check temporal drift and streamwise
   evolution of profiles, wall heat flux and skin friction. Extract an interior
   profile with matching coordinates, composition and both temperatures.
   Freeze extraction station and acceptance thresholds before long runs.
4. Prepare matched comparison cases from the accepted inlet. Changing top
   targets or source mode requires explicit restart lifecycle validation, not
   blindly copying a characteristic checkpoint and bypassing its contract.
5. Check temporal and spatial refinement, upstream profiles, wall pressure,
   heat flux, separation/reattachment if present, species and T/Tv. A steady
   laminar solution is not assumed in advance.

Retain existing physical-domain checks, including the 1000 K transport
diagnostic floor. If a state leaves the model domain, stop rather than clip
temperature or relax the guard. Chemistry-on/off differences must be compared
against discretization error and upstream development, not all attributed to
local shock chemistry. No physical validation or production admission is claimed.
