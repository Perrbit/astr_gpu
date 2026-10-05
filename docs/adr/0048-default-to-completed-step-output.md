# Default to completed-step output and reject legacy checkpoint recovery

Accepted by the user on 2026-10-04. Normal solver startup always selects the
completed-step output/restart interface and reads `datin/input.output`.
`ASTR_OUTPUT_CONFIG` is only an optional path override; an unset or empty value
selects the default file. Missing or invalid configuration fails collectively,
without legacy fallback or automatic conversion of controller frequencies.

Legacy `lwsequ/lwslic` are ignored with a notice. Keep their input positions for
the fixed-order controller reader. Preserve controller reload/CFL checks and
statistics sampling. Configure checkpoint, volume and slices independently,
including explicit `enabled=false` when no such products are wanted.

Keep primary-input `lrestart=false`. Restore only a complete new checkpoint
directory through `output.restore_directory`, with its referenced resources and
history. Old field/auxiliary files, q sidecars and paired checkpoint batches are
not accepted or converted. Preserve old executables for historical restarts.

This changes the default interface, not the registered case/topology or renderer
admission. Existing launch scripts that depend on legacy files must be migrated.
The accepted first-delivery layout and bounded regression are recorded in
`documents/ASTR_OUTPUT_RESTART_REDESIGN_PLAN.md` sections 10.77-10.79.
