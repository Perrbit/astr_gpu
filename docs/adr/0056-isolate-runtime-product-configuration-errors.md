# Isolate runtime product configuration errors

Accepted design, 2026-10-29; implementation is not yet authorized.
The runtime product catalog uses one independently maintained file, but an
invalid product record must not reject unrelated valid changes or terminate
the CFD advance. Report the rejected update and continue processing other
records; a rejected update is neither an active definition nor a successful
output.

An invalid replacement for an identified existing product disables that
product until a subsequent valid configuration restores it; do not silently
continue rendering its old definition. Invalid new products are skipped.
Other products and the CFD advance continue, and historical output remains.

For an otherwise valid new or revised product, a safely isolated temporary
resource shortage with healthy GPU/MPI state also disables that product until
explicit human re-enablement. Do not retry it each output period, lower image
quality or switch execution backends. This does not rewrite existing image
publication waiting rules or imply recovery from a damaged device context or
MPI failure.

Re-enable a resource-suspended product through an increased per-product
`enable_request` number. Consume and checkpoint the request, attempt a first
frame at an available completed-step state, and retain the definition version
when its extraction/display definition is unchanged. Repeated publication of
the same request or a cadence edit must not implicitly retry it. Correction
of an invalid product record follows the separate configuration-error rule.

Publish a completed candidate by replacing the active catalog in one filesystem
operation, rather than exposing intermediate in-place edits. Publication does
not require every product entry to be valid: acceptance remains per product.
An identifiable published catalog specifies the complete intended membership,
not an incremental patch list. Omitted products stop without deleting past
output; file-read failure must not be misreported as an intentionally empty
catalog.

If the file cannot be read or framed into identifiable records, report and
reject that file-level update while preserving every product's current
active/disabled state, definition version and clock. Do not stop all products
or reset formal statistics. This is distinct from disabling an identified
product whose own replacement record is invalid.

Use JSON Lines with one complete product record per line and structured JSON
decoding per record, rather than a whole-file JSON array or namelist read.
A malformed line must not consume subsequent records. The proposed format
identifier and explicit-empty syntax are detailed in the engineering schema
awaiting overall review, rather than being implemented behavior.
Reusing startup scheduling initialization must not erase unrelated products'
runtime history.
