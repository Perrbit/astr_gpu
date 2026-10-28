# Upgrade physical boundaries without replacing interior schemes

On 2026-10-27 the user completed an 18-decision interview for replacing the
low-order physical-boundary closures identified in the ASTR source audit.
CPU and GPU will use the same corrected mathematical discretization, with
independent analytic or manufactured references in addition to backend parity.
The old CPU result is not an immutable numerical oracle.

The chosen target is at least sixth-order local spatial accuracy for smooth,
unlimited boundary and adjacent-interior derivatives and final spatial RHS.
Checking an interface reconstruction alone is insufficient. Shock and
admissibility limiting may reduce order locally, with activation recorded.
Stability is a separate requirement. Boundary filtering has a deliberately
different target: one filter application must be at least sixth order; this
is not a promise of sixth-order convergence under repeated time-step filtering.

Existing interior explicit, MP7/MP7-LD and compact relations, physical targets
and the initial global halo width of five are retained. Endpoint closures and
physical ghost construction remain a mixed framework selected per operator;
MPI interfaces must not acquire physical endpoint closures. Compatible penalty
or high-order dissipation terms may be researched and validated locally, but
this is not approval to add damping everywhere, introduce a sponge, restore
disabled GRCBC filtering or adopt an unexamined SBP-SAT operator.

Implementation starts with nonreacting FP64 Cartesian explicit paths, including
stretched grids, and proceeds through independently validated CURVE, compact,
native nonreacting FP32 and existing FP64 AIR5 extensions. Wall shear and heat
flux diagnostics are upgraded with the related numerical paths, without
changing their physical definitions or adding rendering features. The isolated
CPU geometry-index defect is reproduced and, once confirmed, repaired as a
separate task so its effect is not mistaken for an accuracy improvement.

The user chose representative combination tests with coverage of every
independent closure rather than a full Cartesian product. Necessary local
time-step reductions are allowed and must be reported together with cost per
physical time; production parameters are not changed implicitly. Existing
unsupported boundary types and general subsonic two-face NSCBC closure are
outside this project scope. Additional CPU numerical or physical defects still
require their own human decision.

After acceptance and rollout review, the high-order scheme replaces the default
for the covered combination. The old scheme remains historical or development
evidence, not a permanent second production backend. Checkpoints must identify
the actual boundary algorithm; changing algorithms cannot be called exact
continuation. Existing control metadata is preferred over new sidecars, with
capacity and existing meanings checked before reuse.

The subsequent execution request authorized local work on the HCW
11/22/41/52 boundary chain. The user approved HB1-A's local incoming-flux
correction, seven-point exterior derivative, same-stage source compatibility
for every incoming mode, and outlet-filter closure. They are now connected to
the native FP64 CPU/GPU 743e/643e solver, using boundary identity
`sbli-grcbc-h1`. Internal schemes and physical targets are unchanged.

The actual first-stage Navier-Stokes RHS has passed the x-refinement boundary
strip test (orders 5.63 and 5.82) and y-refinement test (5.90 and 5.91).
Same-phase NP1/NP2 x/y/z and NP4 2x2x1,
CPU/GPU exact continuation, grazing-entropy transport and representative
memcheck checks have passed. These bounded results are not general CURVE,
compact, FP32, AIR5, acoustic-reflection or developed-SBLI acceptance. The
unchanged Mach-2 angle/plateau gate and the native HCW short window have also
passed. However, full-field inspection found growing spanwise disturbances
near the top on the finest extruded Euler fixture. Same-state backend replay
and initcheck pass, while a short scaled-perturbation replay shows substantial
amplification on the nonuniform background. Production acceptance is withheld;
uniform frozen spectra do not establish nonlinear robustness. Any replacement
of the retained limited endpoint branch or new mathematical damping requires
an HB1-B proposal. The staged work and evidence are recorded in
`documents/ASTR_HIGH_ORDER_BOUNDARY_REDESIGN_PLAN.md`, sections 28-29.
