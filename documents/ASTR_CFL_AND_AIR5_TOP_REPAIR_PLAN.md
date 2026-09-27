# CFL Diagnostics and AIR5 Top Boundary Repair

Approved: 2026-09-26. Active goal; not an acceptance report.

## Scope

Work locally on feature/gpu_dev. Preserve FP64, explicit synchronization, fixed
time steps, the existing AIR5 model, tolerances, compensation and the old top
boundary as the default control. Do not change remote jobs or perform Git
stage/commit/push operations. No subagents or additional worktrees.

## A. Complete-Step CFL

Status: implemented; the scoped CPU/GPU diagnostic matrix passes locally.

Use `abs(dot_product(u,grad_xi)) + a*norm2(grad_xi)` in each direction.
Report three directional maxima, the maximum of the pointwise directional sum,
and the sum of directional maxima (the conservative upper bound). Tie locations
use the smallest global Fortran-linear node index, independent of MPI topology.
AIR5 sound speed uses current partial densities and the frozen translational
heat capacity, not equilibrium vibrational heat capacity or reference composition.

CPU/GPU diagnostics read the state after the complete chemistry/transport step.
The outer loop passes the time step actually used, even if the controller has
subsequently been reread. No automatic time-step adjustment is introduced.
GPU reductions return four maxima and four locations; optional y profiles return
only per-plane maxima. No flow state is changed or copied wholesale for CFL.

Runtime diagnostic controls:

- `ASTR_CFL_DIAGNOSTICS=on|off`, default on. Off is for the noninterference test.
- `ASTR_CFL_PROFILE_Y=0|1`, default 0. All ranks must agree on these controls.
- Profiles report global computational j planes; they are not automatically
  physical wall-distance profiles on arbitrary curved grids.

Acceptance, required before claiming A complete:

1. Positive/negative uniform velocities give identical spectral radii; controlled
   nonorthogonal metric, inactive direction and invalid-state probes pass.
2. Root CMake CPU/GPU builds succeed. Device arithmetic and full solver reduction
   agree with matched-state CPU/independent diagnostics to FP64 roundoff.
3. NP1 and NP2 x/y maxima and tie locations agree for matched analytical states.
4. Diagnostic on/off and old/new diagnostic paths do not alter flow or carry;
   use exact comparisons for the same executable and matched configuration.
5. New GPU kernels pass Compute Sanitizer. Diagnostic sampling is explicitly
   complete-step and cannot silently use the previous checkpoint state.

## B. Existing Boundary Time-Step Control

Status: both matched candidate-executable 50 ns runs and diagnostics complete;
physical/time-convergence acceptance is not claimed.

Run 0.125 ns x 400 from the unchanged incident seed, and compare with
0.25 ns x 200 at the same 50 ns endpoint. Check nonnegativity, 128-epsilon
sequential composition closure, thermodynamic/transport domain, wall/top
contracts, and per-component temporal differences with their global locations.
Do not infer a formal convergence order from two step sizes or different times.
If refinement does not reduce the relevant differences, stop window extension
and isolate the responsible operators; do not relax acceptance thresholds.

## C. Optional Characteristic Top Boundary

Status: explicit incoming relaxation-time candidate and source-retaining closure
A approved on 2026-09-26. Shared CPU/GPU local characteristic RHS implemented
and algebra-probed. CPU cold-start lifecycle is wired and stationary-uniform
checked; GPU full-boundary, restart and physical acceptance remain open.
The acoustic protocol and thresholds are approved in
`ASTR_AIR5_CHARACTERISTIC_ACOUSTIC_GATE.md`. Frozen-source CPU/GPU API probes
and the CPU uniform solver gate pass; the first fine-grid acoustic baseline
is running locally. Subsequent acoustic comparisons remain gated on its result.

First scope is Cartesian upper-y AIR5 two-temperature flow. Current incident
states are external targets, not unconditional replacement of all conservative
components. Preserve outgoing acoustic information; explicitly derive incoming,
tangential/zero-speed, transverse, viscous and chemical/vibrational source
treatment consistent with the actual split integrator. Do not count sources
twice or reuse the five-equation characteristic matrices for eleven fields.

Before implementation, specify ghost closure, corner precedence, stage state,
compensation reset/preservation, and any target relaxation. A new unresolved
physical choice requires human review rather than an invented coefficient.
The user has approved tau=Ly/a_infinity and half/double-time sensitivity tests
as a candidate, not as the default or a validated optimum. The dimensional
values and approval limits are recorded in `ASTR_AIR5_CHARACTERISTIC_TOP_DESIGN.md`.
Keep current all-state Dirichlet mode as an independent control; do not change
shock strength or add a startup ramp to hide a discrepancy.

Freeze quantitative gates before each experiment: uniform state, outgoing
acoustic wave, frozen oblique inflow, reacting/two-temperature source response,
then matched old/new SBLI. Include CPU/GPU, MPI, restart and memory safety.
Characteristic acceptance replaces the old exact-target top-plane checker;
it must test characteristic compatibility and reflection instead of requiring
all top variables to equal their external targets.

## Completion and Stops

Complete only after A/B and the explicitly scoped C implementation and acceptance
are finished, with commands, executable/input provenance, and bounded conclusions.
Failing scientific gates prevent downstream physical/performance claims. Pending
physical closure is reported for decision, not silently treated as passed.
The remote four-A800 P0--P3 campaign remains untouched.

## Local Evidence, 2026-09-26

Artifacts: `tests/gpu_validation/out/cfl_repair_20260926/`.
Candidate GPU executable SHA256:
`204019ba4de7d63c7e80f3d8a00d89b580ceed691e65099cf30509c83434f065`.
Root CMake CPU/GPU main builds and both `cfl_spectrum_probe` targets pass.
The probe covers reversed velocity, nonorthogonal metric rows, an inactive
direction and CPU NaN rejection; its GPU memcheck reports zero errors.

- Three-step AIR5 replays at 0.25 ns: CPU NP2 x, GPU NP1, NP2 x and NP2 y
  complete. `check_air5_cfl.py` independently checks each complete-step maximum,
  its reported location, and requested y-plane profiles. Maximum absolute
  difference is 8.014422459012849e-15. The unsnapshotted middle step is checked
  against the next checkpoint, independently of host snapshot refreshes.
- First-step directional maxima are approximately 0.002041602, 0.044888131,
  0.002570709; pointwise-sum maximum 0.049038173; upper bound 0.049500441.
- GPU diagnostic on/off: all 40 selected active-node phase files agree exactly;
  checkpoint conservative and compensation fields agree bitwise.
- Frozen old/new GPU executable comparison gives the same exact result for
  40 phase files and all 22 checkpoint fields. These are three-step checks,
  not long-time accuracy or performance evidence.
- Full solver NP1 one-step profile-enabled Compute Sanitizer memcheck passes
  with zero errors and independently checked CFL values.

The first callback version failed in host array-descriptor access before the
kernel launch. The collector interface now passes the profile length and an
explicit-shape array; the same failing replay passes after this change.

An additional all-phase CPU/GPU comparison failed for the first pre-chemistry
top-plane snapshot only. GPU `prepare_rkfirst_stats_gpu` calls
`air5_prepare_spatial_state_gpu`, which applies the incident boundary before
this snapshot. CPU records its pre-chemistry state before the boundary call
following the first chemistry half-step. For this startup, CPU still contains
the pre-incident checkpoint top state. Rank-0 interior differences are zero;
36 files at matched boundary phases pass the existing 1e-9 absolute / 1e-10
relative comparison. Their maximum absolute conservative difference is
1.1641532182693481e-10. This is not a pass for the failed first snapshot and is
not repaired or hidden by changing tolerances. Boundary lifecycle alignment
requires explicit review before C; no CPU boundary semantics were modified.

Additional full-solver matrix: six 16-cubed nonorthogonal free-stream runs
(CPU/GPU, NP1/NP2-x/NP2-y), plus twelve zero-deformation positive/reversed
free-stream runs. All flow gates pass. CFL values, maximum locations and all
requested y profiles agree exactly across backends and topologies in each
matrix. Cartesian values also match analytical spectral radii within
5e-13 relative / 1e-14 absolute. Numerical metric rounding means a nominally
uniform Cartesian maximum need not be at node zero; reported locations are
compared for the actual discrete metrics, not fabricated exact metric ties.
`freestream_cfl_matrix.json` records the checks. Grid-tool tests: 14 passed.

B reran the 0.25 ns control with the candidate executable before the
0.125 ns run, preserving executable identity for matched-window comparisons.
C is not implemented. Its EOS/source derivation and open physical choices are
in `ASTR_AIR5_CHARACTERISTIC_TOP_DESIGN.md`. Do not promote the diagnostic
matrix to a completed top-boundary or physical validation claim.

### B Results

Same seed, NP2 x-slab, unchanged strong-state top, FP64 and compensation restore;
both end at t=4.140000000000027e-6 s. No remote jobs were touched.

| Run | Updates | Driver elapsed, includes I/O | Maximum recorded new CFL upper bound |
| --- | ---: | ---: | ---: |
| 0.25 ns | 200 | 286.9167 s | 0.0508921 |
| 0.125 ns | 400 | 562.2762 s | 0.0254375 |

Both runs pass 40 sampled phase-state sequential mass-closure checks at the
existing 128-epsilon tolerance, nonnegative species and the wall/top/model-domain
checks. No claim of offline inspection of every step is made. The candidate
0.25 ns run agrees exactly with the frozen old executable at all 36 selected
matched boundary-phase files, including the final state.

| Maximum difference at the same 50 ns endpoint | Previous 0.5/0.25 ns | New 0.25/0.125 ns |
| --- | ---: | ---: |
| u, m/s | 2.1510934269 | 0.9858599960 |
| v, m/s | 1.0654765473 | 0.3445802555 |
| w, m/s | 1.4511741608e-7 | 2.6618226112e-8 |
| T, K | 8.4543075361 | 3.5358389920 |
| p, Pa | 241.5908832563 | 97.9208554859 |

The new temperature/pressure maximum is at global (21,508,0), still three
y intervals below the top. Errors decrease under time-step refinement but
remain much larger than machine noise. The old-boundary startup is not yet a
physically validated SBLI solution; neither a formal convergence order nor a
unique boundary/limiter cause is established. The comparison does not authorize
changing the physical model, extending the window, or relaxing tolerances.
Evidence: `dt025_dt0125_50ns_matched.json` plus each run's `validation_gate.json`
under the CFL repair artifact directory. The previous comparison lives under
`air5_mach4_incident_20260926/dt05_dt025_50ns_matched.json`.
