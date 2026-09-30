# Publish checkpoints before retiring old generations

Accepted for the output redesign on 2026-09-30. Checkpoint retention supports one or two ordinary saved generations. Even in single-generation mode, the new batch is written separately, confirmed complete, and published before the previous batch is retired; the user permits the temporary storage for both generations during replacement.

Direct overwriting of the only valid multi-file checkpoint is not the selected design. Two-generation retention similarly requires room for a third generation during an update, excluding shared resources and protected copies. Publication and durability details must match the target filesystem; this decision does not assume that a nonempty checkpoint directory can be atomically overwritten with a single rename. The default retention count remains a separate decision.

The user subsequently selected fail-stop behavior for checkpoint write failures: do not advance further, do not publish an incomplete batch, retain valid older recovery points, and report failure. This does not change the independent in situ rendering frame-failure policy.
