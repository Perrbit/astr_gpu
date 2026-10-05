# Separate adaptive products from statistical sampling

Accepted design, 2026-10-04; not yet implemented. Adaptive cadence applies only
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
