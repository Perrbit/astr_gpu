# AIR5 Acoustic Domain Mismatch Diagnosis

Diagnosis: 2026-09-27, solver baseline `f3cced3`.
Approved CPU/GPU correction and paired-gate recovery: 2026-09-28, working tree
based on that commit. The full frozen acoustic matrix subsequently passed on
the same executable; this is not SBLI admission.

## Approved Correction And Verification (2026-09-28)

The captured offending call is the top **normal** flux projection, at RK2
cdt=1.25e-9 s, beta=1, left side active and right side inactive. The exterior
NO lower bound was -3.2552083333331987e-4, with upper bound zero, despite zero
NO high/low candidates, both RK budgets and both states. Affine redistribution
produced NO flux -2.0971097442184562e-36. The full captured input is preserved
in `out/air5_characteristic_20260927/zero_support_capture_gpu/baseline/run.log`;
its projection constants are also retained in the Fortran regression probe.
Temporary solver logging has been removed.

`air5_preserve_zero_flux_support` in `src/chemistry_core.F90` is a shared pure
host/device helper. It fixes a species' lower and upper flux bounds to zero
only when its high/low candidates, left/right RK budgets and left/right states
are all exactly zero. CPU/GPU face solvers call it after constructing exterior
bounds and before the existing affine projection. No species-density threshold,
cell-state clipping, new physical model or relaxed acceptance tolerance is used.
Nonzero reconstruction candidates prevent locking even with zero face states.
Existing mass/moment checks, two-temperature energy corrections and admissible
fallback remain in force.

Verification evidence under `out/air5_characteristic_20260928/`:

- Captured-call probe failed before support preservation and passes afterward.
  CPU and GPU probes pass, including a 1e-200 contribution in each of the six
  inputs separately, unchanged bulk bounds and the original affine constraints.
  `air5_flux_gpu_probe` Compute Sanitizer Memcheck reports zero errors.
- `zero_support_fixed_gpu`: baseline/extended 100-update runs pass exact
  N/O/NO absence in 12 saved pre-RHS/post-update states per case. Their center
  outgoing acoustic signals are exactly equal over the 100 samples.
- `zero_support_fixed_cpu`: both domains pass an eight-update run, including
  the original failure stage. Each has 12 sampled zero-species states. At step 7,
  nine CPU/GPU matched phases per domain have maximum scaled difference
  7.519910620127727e-15, below 2e-10. This is not a full-window CPU bridge.
- `zero_support_coupled_topology`: three-update coupled/viscous CPU NP1 versus
  GPU NP1, NP2 x/y/z and NP8 2x2x2 passes. Maximum scaled field difference
  1.7108975961903168e-13, with exact sampled face-halo/donor equality.
- `zero_support_acoustic_pair`: both original full-state configurations reach
  1500 updates (7.5 us). Each passes 18 sampled physical-state checks and 12
  sampled exact-zero checks. Maximum sequential closure is 5.551115123125783e-16.

| Full-window measure | Baseline | Extended |
| --- | ---: | ---: |
| Incident peak | 0.43244452880561257 | 0.4324445317480752 |
| Incident L2 | 0.0003648606663484776 | 0.0003648613213736373 |
| R_peak | 3.98029511308272e-5 | 3.214655364644184e-5 |
| R_L2 | 4.835819671647331e-5 | 4.340409735530344e-5 |

The incident relative differences `abs(baseline/extended-1)` are
6.804254382153374e-9 (peak) and 1.7952715767322047e-6 (L2), both below 0.05.
The baseline reflection gate passes. Extended R values remain diagnostics in
the original observation window, not a reflection gate at the distant top.
The original paired-domain blocker is resolved without changing limiter mode.
Half/double tau, coarse grid, half dt and prescribed controls subsequently passed
on this fixed executable. Their combined report is
`out/air5_characteristic_20260928/zero_support_acoustic_controls/result.json`.
Across all seven cases the 84 sampled pre-RHS/post-update states remain exactly
zero in N/O/NO. See `ASTR_AIR5_CHARACTERISTIC_ACOUSTIC_GATE.md` for the full
table and unchanged limits. No oblique/grazing or real SBLI admission follows.

Build and tests use root CMake targets `astr`, `air5_transport_roundoff_probe`
and `air5_flux_gpu_probe`. Twenty-two related Python checks and six subtests
pass. No Git or remote operation was performed.

## Finding

The current frozen acoustic comparison fails because the two domains develop
different full-state limiting in the pulse region. The constraint that activates
is positivity of N, O and NO, initially and physically absent in this frozen
test. Tiny nonzero species first appear during the characteristic-top limited
transport operation, then propagate into the interior and trigger a common
fluid/species limiter. The resulting sound-wave difference is not merely a
machine-level field error, even though the initiating species densities are tiny.

The originally reported incident peak difference remains 7.2512%, exceeding 5%.
No acceptance tolerance, species floor, physical source or boundary equation was
changed. CPU has the same class of behavior, not an exclusively GPU defect.

## Evidence Chain

Evidence root: `tests/gpu_validation/out/air5_characteristic_20260927/`.
All new runs use the existing executable and retain frozen sources, no viscosity
or filter, and the same initial fields and dt=5 ns. Short runs are diagnostics.

1. `acoustic_domain_onset_7_gpu/baseline`: at step 7, RK2, all owned N/O/NO
   densities in `pre_rhs` and all corresponding `conv_raw` RHS values are exactly
   zero. After limited convection and characteristic-top processing, `conv`
   contains NO RHS of magnitude `2.0971097442184562e-36` at `(16,256,5)`.
   `post_update` NO there is `5.36860094519947e-35` kg/m3. Indices are zero-based.
   This is the physical top/outlet intersection, not an MPI interface.
   The RHS snapshot is in the solver's mapped-volume convention; its number is
   not directly a physical species production rate.
2. `acoustic_domain_diagnostic_gpu_v2`: at step 30, RK1, the central band
   i=4:12, j=160:229 contains N up to `7.753966696144241e-74`, O up to
   `3.251663160243172e-74`, and NO up to `2.3000956443254766e-75` kg/m3 in
   the baseline. All three remain exactly zero in the corresponding extended
   band. The baseline minimum full-state ratio in this band is zero, whereas
   the extended-domain ratio is one in all three sampled RK stages.
3. `acoustic_domain_constraints_gpu/baseline`: step-30 diagnostics identify
   only species constraints. RK1 has 9205 limited points, attributed to N/O/NO;
   density and translational/vibrational energy constraints are not activated.
4. `acoustic_domain_constraints_cpu/baseline`: the CPU completes 32 updates
   and likewise reports N/O/NO constraints (9194 limited points at step-30 RK1,
   minimum ratio zero). This establishes shared behavior, not bitwise CPU/GPU
   equivalence of this sensitive full-state trajectory.
5. `acoustic_domain_symmetric_gpu`: change only the interior convection limiter
   to the existing `symmetric_species` option. The characteristic-top algorithm
   is unchanged. Baseline and extended center-probe outgoing acoustic signals
   are exactly equal over the first 100 samples. With `full_state`, their maximum
   absolute difference is `0.002943700994106454` Pa; sample 99 is
   `0.012336095570905998` versus `0.009392394576799544` Pa. The symmetric
   control gives `0.009392394581451111` Pa in both domains. This isolates the
   amplification by common fluid/species limiting; it does not validate the
   entire 1500-update acoustic gate or cure artificial trace creation.

Acoustic signal definition: `p - p_ref + rho_ref*a_ref*v`, as in the existing
acoustic gate. Do not interpret early-signal relative error as reflected-wave
ratio. Both complete-run incident peaks arrive at 1.605 us.

## Implementation Localization

In `src_gpu/chemistry_solver_gpu.cuf`, `conv_raw` precedes the interior limiter;
`conv` follows `air5_characteristic_top_rhs_gpu`. Ordinary interior limiting
does not update this physical top node. The top path constructs normal high/low
candidates and invokes the symmetric species/energy face correction, including
the tangent directions. `air5_symmetric_face_trial_gpu` calls
`air5_close_species_flux_moment_gpu`, with relaxed exterior-side bounds at a
physical face. The CPU counterparts are in `src/chemistry_solver.F90` and
`src/chemistry_core.F90`.

With zero species states, zero target species and frozen sources, the linear
characteristic flux differential cannot itself supply those species. The
coupled face-flux projection is the remaining creation mechanism in this path.
The phase snapshots originally localized creation to top limited transport.
The subsequent captured-call regression identifies the normal projection,
as recorded in the correction section above.

## Original Correction Proposal (Subsequently Approved)

- Preserve structural zero-species fluxes in the face closure when the complete
  face candidate/budget and relevant boundary data establish absence of that
  species. Use exact support information, not a numerical concentration cutoff.
- Retain physically supplied nonzero inlet species and chemically generated
  species. Do not infer global absence from one cell or only two endpoints of
  a wider reconstruction stencil.
- Apply the same rule on CPU/GPU, retaining interface mass consistency, the
  existing gas-constant moment constraint and two-temperature energy corrections.
  An infeasible constrained projection must reject or fall back through the
  existing admissible path; never clip the updated state.
- First capture the offending top-face projection as a minimal CPU/GPU test,
  including absence preservation and trace-but-nonzero counterexamples. Then
  replay the original full-state diagnostic and acoustic domain comparison.
- The already adopted symmetric interior policy is a useful amplification
  control, not a substitute for fixing a shared zero-species invariant defect.

The user approved the shared correction after this diagnosis. Its implementation
and bounded verification are recorded above; no production run was started.

## Reproduction And Incomplete Runs

Driver: `tests/gpu_validation/diagnose_air5_acoustic_domain.py`.
Examples use `--updates 100 --sample 30`, or
`--updates 32 --sample 30 --diagnostic-period 30` for constraint logging.
Use `--limiter symmetric_species` for the controlled interior change, `--cpu`
for the CPU backend. Output must be a new directory.

`acoustic_domain_constraints_cpu/extended` was deliberately terminated with
SIGTERM after the completed CPU baseline had reproduced the shared behavior;
exit 143 is not a numerical rejection. It is not acceptance evidence.
`acoustic_domain_diagnostic_gpu` was a preparation-only failure due to a
controller marker initially sought in the input file; no solver started there.
The corrected run is `acoustic_domain_diagnostic_gpu_v2`.
