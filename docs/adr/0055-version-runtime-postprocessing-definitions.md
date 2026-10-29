# Version runtime postprocessing definitions

Accepted design, 2026-10-29; implementation is not yet authorized.
Runtime product instances retain their names, but changes to their fields,
spatial regions, Q thresholds, cameras or color definitions automatically start
new definition versions and output sequences. Earlier images and data remain
unchanged, avoiding an apparently uniform time sequence that actually mixes
different observations or display scales.

This versions the observation definition, not the flow solver. It does not by
itself require new extraction or streamline integration when only presentation
changes. Cadence-only changes retain the definition version, record their
activation step and time separately, and do not reset formal statistics.
New runtime instances and accepted definition versions produce a first frame
at the first available completed-step state with the required data, then
follow their configured cadence. This does not change startup/restart frame
rules or allow an instantaneous field to stand in for an unavailable mean.
Product updates may consume already enabled statistics but never implicitly
start or reset statistical accumulation. A missing statistics dependency
prevents only the dependent product from running; enabled statistics that
have not yet acquired their required coverage are allowed to become ready.
Cadence-only changes anchor to the most recent actual emission. If the new
interval has already elapsed, emit one current completed-step state rather
than recreating missed historical frames, then anchor to that actual emission.
Existing failure-attempt waiting rules remain intact.
Checkpoint authority and restart behavior are recorded in ADR 0057; remaining
file-format and migration details are tracked in the in situ plan section 5.7.
