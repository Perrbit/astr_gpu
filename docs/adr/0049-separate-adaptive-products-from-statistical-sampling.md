# Separate adaptive products from statistical sampling

Accepted design, 2026-10-04; bounded implementation completed 2026-10-06.
Adaptive cadence applies only
to individually selected images, extracted geometry, volume fields and slices.
Formal statistical sampling and checkpoints retain independent fixed cadences,
so event-driven output does not silently change time statistics or recovery cost.

Combine explicit step/time windows with independent scalar events using fixed
reference scales, two thresholds and a minimum simulation-time hold. Products
use normal/dense intervals, emit once on entry, and anchor subsequent targets to
their last actual output. This deliberately favors event coverage over preserving
a fixed output grid; missed historical states are never reconstructed.

Exact continuation requires versioned monitoring and scheduling state inside
the existing native checkpoint bundle, not additional restart sidecars. Explicit
output override may reset affected adaptive state without resetting formal
statistics. The approved first gate is bounded TGV kinetic-energy monitoring;
broader indicators and production rendering require separate evidence. See
the in situ plan sections 1.5.2-1.5.3 for scope, budgets and acceptance criteria.

On 2026-10-05 the user confirmed that images, extracted geometry, native volume
fields and slices share one monitoring/event context. Reuse a same-definition,
same-phase scalar calculation and reduction once, distribute one event decision,
and retain independent product clocks. Persist shared history once and each
product's emission state separately; do not tie formal statistical sampling or
checkpoint cadence to these events.

A fresh run establishes monitoring history from the valid initialized state and
its starting simulation time, so the first due monitoring sample can form a
rate. This baseline does not force images, formal statistical accumulation or a
checkpoint. Restart uses saved history and clocks instead of reseeding them.

Shared monitoring, event and important-window definitions belong to the existing
`datin/input.output`, not a new required file or a Catalyst-only configuration.
Native and in situ products reference stable event identities while their own
configuration retains product intervals and rendering-specific options. Validate
cross-configuration references collectively before advancement. Native adaptive
file output must remain available without Catalyst.

JPEG/EPS for the same view form one image product and share a clock; extracted
VTK geometry may have its own clock. An approved recoverable image-pair publish
failure preserves the last successful identity but records a failed attempt and
waits one interval of that attempt's cadence before trying a current state.
Later cadence transitions reevaluate from that attempt anchor. Persist the
failure/next-target state for exact continuation; this is an explicit exception
to successful-emission anchoring, not a success flag or a historical-frame retry.
Encoding, geometry, numerical and MPI failures remain fatal.

The implemented gate admits internally generated 16/32-cubed periodic Cartesian
nonreacting TGV kinetic energy only. Monitoring and native archives work without
Catalyst; actual images retain existing GPU-render admission. AP01/AC01 state
is embedded in the existing control/archive/render resources, with no new
restart file. Unrelated fixed products/statistics retain their clocks under
AP-enabled selective overrides; AP-disabled fixed-only overrides retain their
previous registry-wide rule. Source, receipts, costs and deferred scope are in
`documents/ASTR_INSITU_AP_ACCEPTANCE.md`. Production thresholds and additional
indicators remain separate decisions, not defaults inferred from this short gate.
