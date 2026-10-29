# Keep runtime slice archives self-contained

Accepted design, 2026-10-29; local implementation and bounded validation are authorized.
This does not authorize remote production jobs or Git publication.
Fixed grid-index numerical slices use the shared runtime product catalog as
their only definition entry, with independently selected fields, regions,
sources and cadences. There is no parallel legacy static slice definition.

Each product definition version and continuation segment owns its two-dimensional
coordinates and local time index, with one HDF5 data file per emitted time.
Coordinates are reused within that segment rather than per frame. Small
coordinate duplication between products is accepted so moving or removing one
product does not require another product or a run-level geometry pool.

A restart restores effective product definitions, versions and clocks but starts
a self-contained output segment. It does not copy old frame data or automatically
reference another run's history; cross-segment inspection is an explicit offline
indexing operation. This deliberately differs from the earlier automatic slice
archive parent-chain design and does not change checkpoint or volume lineage.

See `documents/ASTR_INDEX_SLICE_ARCHIVE_EXTENSION_PLAN.md`, decisions S-D01,
S-D02 and S-D13 through S-D16. No new write-after-read validation or full-field
checksum is required by this decision.
