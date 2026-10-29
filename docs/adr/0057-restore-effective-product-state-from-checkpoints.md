# Restore effective product state from checkpoints

Accepted design, 2026-10-29; implementation is not yet authorized.
Extend the existing `insitu_control.bin` with the effective runtime product
catalog, definition versions, clocks and necessary active/disabled/pending
state. Each new checkpoint captures the state actually applied when that
generation is written, including any accepted updates since startup; earlier
checkpoint generations are not rewritten.

Restore this saved product state by default. A different catalog already on
disk must not silently replace it during restart; the user can publish another
update after restoration. A published but unapplied or rejected file is not
the effective state. This preserves product-definition and scheduling
continuity without adding product restart sidecars or volume-field data.

This does not authorize changing the solver's numerical or boundary contract.
The user explicitly rejected compatibility with checkpoints lacking runtime
catalog state: no migration or legacy-read branch is to be introduced.
Every checkpoint written by the new version contains valid catalog state,
including an empty catalog when no products are requested; recording that
state must not itself enable rendering or allocate volume workspaces.
Reject missing-state old checkpoints even under non-exact restoration.

Existing old binaries and checkpoint evidence are retained, and this design
does not alter an already submitted old-binary job. The exact control layout
remains to be specified in the in situ plan section 5.7.
