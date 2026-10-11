# Publish checkpoints before retiring old generations

Accepted for the output redesign on 2026-09-30. Checkpoint retention supports one or two ordinary saved generations. Even in single-generation mode, the new batch is written separately, confirmed complete, and published before the previous batch is retired; the user permits the temporary storage for both generations during replacement.

Direct overwriting of the only valid multi-file checkpoint is not the selected design. Two-generation retention similarly requires room for a third generation during an update, excluding shared resources and protected copies. Publication and durability details must match the target filesystem; this decision does not assume that a nonempty checkpoint directory can be atomically overwritten with a single rename. The default retention count remains a separate decision.

The user subsequently selected fail-stop behavior for checkpoint write failures: do not advance further, do not publish an incomplete batch, retain valid older recovery points, and report failure. This does not change the independent in situ rendering frame-failure policy.

## Filesystem Compatibility (2026-10-11)

Prefer `renameat2(RENAME_NOREPLACE)` when available. This operation requires
support from the actual filesystem, not just a successful compiler/linker check
([Linux rename(2)](https://man7.org/linux/man-pages/man2/rename.2.html)).
Only `EINVAL`, `EOPNOTSUPP`, or `ENOSYS` permit the compatibility path; other
errors remain fatal.

The compatibility path requires a writer-owned root that is not group/world
writable. It atomically reserves the previously absent destination with
`mkdirat`, checks the reserved directory identity, and renames the validated
temporary bundle over that owned empty reservation. Other cooperating ASTR
publishers cannot acquire the same name. An already existing directory, file,
or symbolic link is never an eligible reservation. The private output root
must not be renamed, deleted, or altered by unrelated processes while writing;
this is not a transaction protocol against hostile same-user mutation.

The reserved empty name has no `COMPLETE` and is not a valid checkpoint. On an
interrupted or failed publication it is deliberately left for diagnosis, along
with the temporary bundle when still present. Do not automatically reclaim a
reservation or fall back to an unguarded check-then-rename. Update `LATEST` and
retire older owned generations only after publication succeeds. This does not
change the checkpoint format, exact-restart state, or crash/power-loss durability
claims. The same helper is used by the complete field/slice frame bundles.

`run_checkpoint_publication_preflight.py` exercises the real Fortran/C helper on
the chosen output filesystem with tiny CPU/MPI payloads, both retention counts,
validation, and rejection of an existing target. Platform build and run scripts
invoke it before the flow/rendering gates.
