# AIR5 Characteristic Top Acoustic Gate

Status: all seven frozen GPU acoustic controls pass on the corrected executable
on 2026-09-28. Current CPU/GPU evidence is the matched short-window bridge;
the earlier full-window CPU/GPU bridge below remains historical evidence.
Parent: `ASTR_AIR5_CHARACTERISTIC_TOP_DESIGN.md`. Local execution only.
This is frozen acoustic acceptance, not reacting-flow production admission.

## Completed Corrected Matrix (2026-09-28)

Evidence: `out/air5_characteristic_20260928/zero_support_acoustic_controls/result.json`,
with `passed=true`, `complete=true` and all seven raw-case paths. Baseline and
extended artifacts were re-read and reused; only the remaining five cases ran.
All cases use executable SHA256
`72a1ef61c981def1979665d448ff81db99b8bfdb68ed1ce511fc2a75946fa393`,
and logs confirm `full_state` convection. No solver change was made in this
completion pass. SHA is validation provenance, not a solver startup message.

| Case | R_peak | R_L2 | Comparison outcome |
| --- | ---: | ---: | --- |
| Baseline | 3.9802951e-5 | 4.8358197e-5 | Pass |
| Half tau | 3.8976918e-5 | 4.7929617e-5 | Pass; L2 difference 4.2857991e-7 < 0.03 |
| Double tau | 4.0240848e-5 | 4.8583898e-5 | Pass; L2 difference 2.2570144e-7 < 0.03 |
| Extended | 3.2146554e-5 | 4.3404097e-5 | Incident peak/L2 differences 6.8043e-9 / 1.7953e-6 < 0.05 |
| Coarse | 3.9802694e-5 | 4.8357698e-5 | Pass; L2 difference 4.9912919e-10 < 0.02 |
| Half dt | 3.9837590e-5 | 4.8415702e-5 | Pass; L2 difference 5.7505272e-8 < 0.005 |
| Prescribed | 2.0011047e-5 | 3.1161709e-5 | Valid comparison control; no reflection upper-bound gate |

All applicable individual reflection gates and sampled physical-state checks
pass. Across all seven cases, 84 saved pre-RHS/post-update states retain exactly
zero N/O/NO. Maximum sequential composition closure error is
5.551115123125783e-16. These are sampled-state checks, not every-step field dumps.
The prescribed control has lower measured reflection in this test: no claim of
reflection superiority is supported. Nor do the refinement pairs establish a
formal convergence order or physical SBLI validity.

The sequential driver now accepts `--reuse-extended`; it rechecks raw evidence
through the same matrix validator and rejects a mismatched executable before
starting subsequent cases. Eighteen monitor/matrix/driver tests and six subtests
pass, including reuse and mismatch rejection. New oblique/grazing physical
criteria still require approval before those cases run.

## Fixed Baseline And Extended Pair (2026-09-28)

Evidence: `out/air5_characteristic_20260928/zero_support_acoustic_pair/`.
Both cases retain `full_state`, frozen sources, no filter, dt=5 ns and 1500
updates. Baseline R_peak=3.98029511308272e-5 and R_L2=4.835819671647331e-5
pass. Incident peak relative difference is 6.804254382153374e-9, and L2 relative
difference is 1.7952715767322047e-6, both below 0.05. Sampled N/O/NO remain
exactly zero; physical-state and composition checks pass. The complete
implementation and matched CPU/GPU short-window checks are in
`ASTR_AIR5_ACOUSTIC_DOMAIN_DIAGNOSIS.md`.

The remaining controls were subsequently completed with the same executable,
as recorded above. Older artifacts were not mixed into this corrected pass.

## Failed Requalification Before Correction (2026-09-27)

Subsequent diagnosis is recorded in `ASTR_AIR5_ACOUSTIC_DOMAIN_DIAGNOSIS.md`:
top limited transport introduces initially absent trace species; interior
full-state positivity limiting amplifies them into an acoustic-signal change.
CPU exhibits the same species constraints. A symmetric-interior control removes
the first-100-sample domain difference, but is not a completed acoustic gate.
The original failure below is retained as regression evidence; the approved
shared correction and recovered pair are recorded above.

Evidence root: `out/air5_characteristic_20260927/pre_sbli_acoustic_f3cced3/`.
All four completed solver runs use the same executable hash recorded in
`gate.json`. Frozen sources, FP64, no filter and the existing `full_state`
limiter settings are unchanged. The coupled topology gate instead uses
`symmetric_species` convection and `layered` diffusion; it is not a full-window
CPU/GPU acoustic equivalence bridge.

| Case | R_peak | R_L2 | Incident peak |
| --- | ---: | ---: | ---: |
| baseline | 7.9542881953e-5 | 7.1564396270e-5 | 0.4010870545 |
| half_tau | 7.8380149884e-5 | 7.2075947932e-5 | 0.4014257918 |
| double_tau | 1.0655599575e-4 | 8.1264293703e-5 | 0.4021695830 |
| extended | 1.3698056533e-5 | 1.1315940593e-5 | 0.4324445317 |

Baseline and tau controls pass their individual reflection and pairwise
sensitivity gates. Extended-domain relative incident errors, defined as
`abs(baseline/extended - 1)`, are **0.0725121372991665** (peak) and
**0.03899951062717699** (L2). The peak exceeds the approved 0.05 limit.
The sequence exited with status 1 at this comparison and did not launch coarse,
half_dt or prescribed. `controls/progress.json` records only the last successful
three-case prefix, not the extended-domain failure or a completed matrix.

All four sampled-state gates pass; maximum sequential composition closure is
4.440892098500626e-16. These state checks do not override the failed comparison.
The extended R values are diagnostics in the baseline observation window, not
a reflection gate for its more distant top boundary.

Initial-file values in the overlapping domains are exactly equal. In saved
first-step RK1/2/3 post-update fields, the central pulse band (zero-based
i=4:12, j=160:229, all owned k) is also exactly equal. Both incident peaks
arrive at 1.605 us. The first central-probe difference appears at update 16,
initially only 7.35e-28 in streamwise velocity; this does not by itself identify
the later amplitude-error mechanism. No boundary or limiter root cause has
been established. Preserve the failed artifacts and diagnose the first
amplification before resuming the matrix. No threshold has been relaxed.

## Historical Completed GPU Matrix

Service `astr-air5-acoustic-gpu-matrix-20260927.service` exited with status 0.
Evidence: `out/air5_characteristic_20260927/gpu_matrix/bridge.json` and
`out/air5_characteristic_20260927/gpu_matrix/controls/result.json`.
The full-window CPU/GPU maximum scaled phase-field difference is
9.083104639042587e-14; maximum scaled three-probe difference is
7.685706838225895e-13. Both meet 2e-10.

| Matrix measure | Result | Limit |
| --- | ---: | ---: |
| Baseline characteristic R_L2 | 4.83582217485009e-5 | 0.05 |
| Half tau absolute R_L2 difference | 4.2855486457319585e-7 | 0.03 |
| Double tau absolute R_L2 difference | 2.2568830304709825e-7 | 0.03 |
| Extended-domain incident peak relative difference | 6.790728201977458e-9 | 0.05 |
| Extended-domain incident L2 relative difference | 1.7952702934254106e-6 | 0.05 |
| Coarse/fine absolute R_L2 difference | 4.835293975313077e-10 | 0.02 |
| Fine dt/dt2 absolute R_L2 difference | 5.7503524940400916e-8 | 0.005 |
| Prescribed control R_L2 | 3.1161708697675965e-5 | Report only |

All applicable peak-ratio and sampled-state gates also passed in the individual
reports. The prescribed control has a smaller measured R_L2 in this experiment;
these results do not demonstrate a reflection advantage of the new boundary.
Refinement pairs do not establish formal order. Reacting sources, oblique/grazing
states, general three-dimensional disturbances, viscous coupling and restart
remain separate requirements. The execution history below records intermediate
states and superseded CPU execution, not additional pending acoustic runs.

## Purpose

### 2026-09-27 Backend Transfer Update

The user requested prioritizing CPU/GPU same-phase and MPI checks, then moving
remaining physical tests to GPU. The CPU sequence service was deliberately
stopped during the extended-domain run; that partial run is not a numerical
failure and is not accepted evidence. The three completed CPU controls remain
immutable references.

`run_air5_characteristic_backend_gate.py` advances a 16x128x16-interval,
three-step frozen inviscid pulse centered at y=0.0096 m to exercise the dynamic
top directly. It compares all 11 conserved components over 18 assembled physical
phase fields, including the top and corners, with scaled tolerance 2e-10.
GPU NP1, NP2 x/y/z and NP8 2x2x2 all pass against CPU NP1; CPU NP8 also passes
against GPU NP8. Maximum scaled discrepancy in each comparison is
2.2796579438969816e-15. Shared-interface overlap is checked during assembly.
Full-solver GPU NP1 memcheck reports zero errors. Evidence directories are
`out/air5_characteristic_20260927/backend_*`.

This gate is spanwise homogeneous and source-frozen: it does not close general
three-dimensional reacting, viscous, carry-restart or filter admission. NP8
shares two local GPUs and is not scalability evidence. An explicit
`ASTR_AIR5_TOP_GPU_VALIDATION=on` opt-in now enables the staged GPU top;
restart/filter/non-RK3 guards remain. The default prescribed top is unchanged.

The first GPU long attempt (`out/air5_characteristic_20260927/gpu_fine`) was
deliberately stopped after discovering the AIR5 GPU path did not call the CPU
monitor writer. It is incomplete, not accepted and not a numerical failure.
An opt-in `ASTR_AIR5_ACOUSTIC_PROBE=on` now writes only time, velocity,
temperature and pressure for the existing local monitor points, immediately
before the first RK RHS after frozen half-step preparation. It transfers only
point values and supplies no fabricated derivative channels. It explicitly
rejects reacting-source, filter and restart use. GPU text probes and CPU binary
monitors have separate readers with matching time/index/finite-value checks.

Probe-enabled NP1 memcheck and NP8 repeats pass without changing phase-field
errors; CPU/GPU three-point probe maximum scaled difference is
1.7171431377499546e-14, below 2e-10. Evidence: `backend_gpu_probe` and
`backend_gpu_xyz_probe`. Twenty reader, comparator and sequencing unit tests pass.

The new `run_air5_characteristic_gpu_matrix.py` service
`astr-air5-acoustic-gpu-matrix-20260927.service` runs under
`out/air5_characteristic_20260927/gpu_matrix`. First require a complete GPU
baseline's physical gates, 18 first/last phase fields and all three probe
time series against the immutable CPU baseline at scaled tolerance 2e-10.
Only then start half tau and the remaining sequential controls. `bridge.json`
records the two executable hashes and measured differences. GPU controls use
a common executable hash/backend; the comparator rejects unbridged backend
mixing. Historical CPU controls are retained, not relabeled as GPU runs.

Measure outward acoustic transmission and reflected inward waves separately
from chemical/V-T relaxation, viscous attenuation and other boundaries. A small
remaining pressure signal alone is not evidence of a nonreflecting boundary.
This gate is one part of stage C, not a substitute for reacting-source, MPI,
restart, GPU or SBLI acceptance.

## Frozen Source Control

Existing runtime source modes are coupled, chemical and vt. The uniform test
uses vt at T=Tv, where its source vanishes. A frozen acoustic wave instead
preserves composition and specific vibrational energy; its translational
temperature changes, so vt is not source-free during propagation.

Use an explicit `ASTR_AIR5_SOURCE_MODE=frozen` diagnostic mode that disables
both chemical and V-T evolution on all nodes. It must preserve q and chemical
carry exactly during the two nominal chemistry halves, while retaining required
boundary and primitive preparation. Default coupled behavior and the other
source modes remain unchanged. Log the selected mode and prohibit treating this
test as reacting-flow validation. Implement CPU/GPU identically before claiming
cross-backend coverage; a partial backend must reject unsupported selection.

## Initial State And Domain

- SI units, Cartesian box: Lx=0.08 m, Ly=0.01 m, Lz=0.002 m.
- rho0=0.05 kg/m3, T0=Tv0=1500 K, zero mean velocity.
- Mass fractions (N2,O2,N,O,NO)=(0.767,0.233,0,0,0).
- Pressure comes from the implemented AIR5 EOS, not a separately chosen gamma.
- a0=778.0891928821296 m/s from frozen AIR5 thermodynamics.
- Isothermal stationary lower wall at 1500 K, existing fixed left reservoir,
  existing right outflow, periodic z. No filters or sponge.
- Start inviscid with frozen sources to isolate acoustic convection/BC errors.
  Viscous and reacting-source gates follow separately and remain mandatory.
- Grid intervals: 16x128x8 and 16x256x8. Every node is advanced by ASTR; Python
  prepares data and analyzes it, never integrates the flow.

At x=Lx/2 create an upward isentropic pulse centered at y0=0.0075 m with
sigma_y=0.0003125 m and pressure amplitude epsilon=1e-5. Use an x envelope equal
to one on [0.02,0.06] m, smoothly tapered to zero before both x boundaries.
The center measurement's acoustic domain of dependence remains inside this
plateau throughout the proposed 7.5 us observation. Verify lateral contamination
by comparing adjacent central columns, rather than assuming the envelope alone
proves independence for a finite-difference stencil.

Let G denote the envelope times exp(-(y-y0)^2/(2 sigma_y^2)), gamma_f=1+R/cv_tr
at the fixed composition, and P=1+epsilon G. Set

\[
p=p_0 P,\quad \rho=\rho_0 P^{1/\gamma_f},\quad
v=\frac{2a_0}{\gamma_f-1}
\left[P^{(\gamma_f-1)/(2\gamma_f)}-1\right].
\]

Set u=w=0, Y=Y0, Tv=Tv0 and Ev=rho*ev(Y0,Tv0); recover T and total energy
through the existing EOS. This is an exact 1-D frozen simple-wave relation on
the central plateau. The tapered 2-D region is not claimed to be a pure 1-D wave.

## Measurement And Timing

Use x=0.04 m, y=0.00875 m (a node on both grids), any periodic z plane.
Define pressure-unit amplitudes

\[
A_+(t)=p(t)-p_0+\rho_0 a_0 v(t),\qquad
A_-(t)=p(t)-p_0-\rho_0 a_0 v(t).
\]

Measure outgoing A_plus near (yp-y0)/a0=1.6065 us and reflected A_minus near
(2Ly-y0-yp)/a0=4.8195 us, each over a +/-3 sigma_y/a0 window (about 1.205 us).
Compute peak ratio and L2 amplitude ratio using these separate windows:

\[
R_{peak}=\frac{\max_{W_r}|A_-|}{\max_{W_i}|A_+|},\qquad
R_{L2}=\left[\frac{\int_{W_r}A_-^2\,dt}{\int_{W_i}A_+^2\,dt}\right]^{1/2}.
\]

Do not call R_L2 an energy fraction; its square is the corresponding linear
acoustic flux ratio under the present common zero-mean medium assumptions.
Record actual node coordinates and sampling phase. Existing `writemon` is called
through RK-first statistics: audit its timestamp/state phase before using it.
Do not quietly mix it with complete-step records. Only p and normal velocity
are needed for these amplitudes; stored derivative fields are not part of this gate.

Proposed dt=5 ns, 1500 complete updates to 7.5 us; nominal fine-grid acoustic
Cy=0.0995954. Confirm repaired CFL diagnostics during the run; no automatic dt
change. For temporal control halve dt at the same endpoint only after the first
spatial/physical gate passes. Use explicit tau=Ly/a0=12.851997 us, not the smaller
Ly value from the existing SBLI case; also test half and double this time.

## Approved Acceptance And Sequence

1. Source-off lifecycle probe: q/carry unchanged by chemistry halves, default
   coupled mode unchanged; constant-state algebra checks retain their existing gates.
2. Fine-grid baseline tau: both R_peak and R_L2 <=0.05. No negative species,
   invalid model state or IEEE invalid; existing 128-epsilon closure gate applies.
3. Half/double tau: both ratios <=0.10 and absolute R_L2 change from baseline
   <=0.03. These are proposed engineering gates, not universal NSCBC constants.
4. At the same grid and source/transport settings run an extended-top reference
   (Ly=0.02 m with unchanged dy, explicit tau, and pulse/probe coordinates). Require
   the small-domain incident A_plus peak and L2 norm to be within 5% of this
   reference. This checks incident-wave agreement between domains; attenuation
   common to both runs is not bounded by this comparison alone.
5. Coarse/fine absolute R_L2 difference <=0.02; fine-grid dt/2 difference <=0.005.
   Neither pair alone establishes formal convergence order.
6. Run the old prescribed top as a matched control and report both signals and
   ratios, without assuming beforehand that either mode must win numerically.

Stop at the first failed gate; diagnose rather than adjust the threshold or
change the target pulse. Any subsequent change of source semantics or numerical
acceptance thresholds requires a separate decision before rerunning the gate.

## Implementation And Local Evidence

- `src/chemistry_runtime.F90` recognizes the explicit frozen diagnostic mode,
  checks MPI agreement and logs that both source families are disabled.
- `src/chemistry_kinetics.F90` and `src_gpu/chemistry_kinetics_gpu.cuf` validate
  the frozen state and return zero sources/Jacobians. Fixed/adaptive ROS-2
  returns unchanged state and carry without a zero-increment compensated add.
  Outer boundary, halo and primitive preparation are not bypassed.
- Root CMake CPU/GPU `air5_frozen_source_probe` passes for two compositions,
  T=3000 K and Tv=1500 K, nonzero signed carry, and default/explicit coupled
  equality. Device probe memcheck reports zero errors. These are local API
  checks, not full GPU boundary/restart acceptance.
- `chemistry_source_gpu_probe` covers 14 states in each of four source modes,
  including invalid composition, pressure, temperature and NaN inputs.
  CPU/GPU status mismatches are zero; existing scaled comparison gates pass.
- `out/air5_characteristic_20260926/frozen_uniform_cpu` passes the three-step
  CPU solver check under the IEEE invalid trap: maximum normalized physical
  state drift 7.073019170245721e-16, fixed threshold 2e-10.
- `run_air5_characteristic_acoustic.py` reads existing monitor records. In
  `mainloop.F90`, the first chemistry half and boundary refresh precede
  `rkfirst/writemon`; the step-start time is incremented only after the whole
  step. Frozen/no-filter interior monitor states therefore match this time.
  Derivatives are not used. Five reader tests cover binary layout, phase,
  truncation, index and finite-pressure checks.
- `check_air5_characteristic_acoustic_matrix.py` requires all seven controls:
  baseline, half/double tau, extended top, coarse grid, half dt and prescribed
  top. It re-reads raw monitors and state samples, checks configuration agreement
  and applies the approved pairwise thresholds. Nine reader/matrix tests and six
  failure-threshold subtests pass. This is checker verification, not evidence
  that the still-pending physical matrix passes.

The completed fine-grid baseline is
`tests/gpu_validation/out/air5_characteristic_20260926/acoustic_fine_service`,
managed by user service `astr-air5-acoustic-fine-20260926.service` (exit 0).
It uses 16x256x8 intervals, 1500 updates, dt=5 ns and tau=12.851997035145649 us.
Executable SHA256 is stored in its `gate.json`, not printed by the solver.
The initial `acoustic_fine_baseline` attempt was deliberately stopped before
completion to replace its one-hour foreground timeout with service supervision
and a four-hour bound. It is not a failed numerical gate or acceptance evidence.
The replacement starts from the same inputs, without a numerical change.
The baseline completed on 2026-09-26 at 23:13:29 CST in 6650.944 s wall time.
Its `result.json` reports R_peak=3.9803108659313885e-5 and
R_L2=4.835822529092618e-5, both below 0.05. Eighteen sampled states have
minimum T=1499.9999320260497 K and p=21622.239144869538 Pa; maximum sequential
composition closure error is 4.440892098500626e-16. These are sampled-state
checks, not all-step field monitoring. End-of-run IEEE underflow/inexact flags
were present, with no IEEE invalid report. The baseline alone does not prove
parameter robustness, refinement, or source-active boundary correctness.

Completed sensitivity run: `out/air5_characteristic_20260927/acoustic_half_tau`, service
`astr-air5-acoustic-half-tau-20260927.service`, tau=6.425998517572821e-6 s.
Its executable hash, domain, grid, time step and endpoint match the baseline;
only tau changes. R_peak=3.8977105762030735e-5 and R_L2=4.792967732518466e-5
pass the individual 0.10 gates. The absolute baseline R_L2 difference is
4.2854796574151767e-7, below 0.03. Maximum sequential composition closure error
is 5.551115123125783e-16. The service exited with status zero.
Double tau (`acoustic_sequence/double_tau`) also passes: R_peak=
4.024095899124515e-5, R_L2=4.8583909968975916e-5 and absolute baseline R_L2
difference=2.2568467804973515e-7. Maximum sequential closure error is
4.440892098500626e-16. The driver has advanced to `acoustic_sequence/extended`,
not yet accepted. `acoustic_sequence/progress.json` records three completed
cases, with `complete=false`.

The sequential driver `run_air5_characteristic_acoustic_sequence.py` is running
under `astr-air5-acoustic-matrix-20260927.service`. It waits for the existing
half-tau service without restarting it, re-reads its raw evidence, and checks
all available pairwise gates before launching each next case. The order is
double tau, extended top, coarse grid, half dt, prescribed control. Any solver,
state, configuration or pairwise failure stops the sequence. Each solver has
a four-hour timeout; the supervising service has a 24-hour limit. No GPU or
reacting-flow gate is automatically launched by this acoustic driver.
Reports will be under `out/air5_characteristic_20260927/acoustic_sequence`;
`progress.json` explicitly distinguishes partial from complete matrix evidence.
Fourteen monitor/matrix/sequence unit tests pass, including rejection of skipped
gates and preventing the next solver launch after a failed pair comparison.
Mocked sequence results in those unit tests are not physical simulation evidence.
All paths beginning
`out/` in this section are relative to `tests/gpu_validation/`.
