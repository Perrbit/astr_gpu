# Share immutable restart resources per run

Accepted for the output redesign on 2026-09-30. Static meshes, fixed profiles, and frozen inlet data are stored once per run and referenced by checkpoint bundles using declared relative locations and content identities, rather than copied into every save. Moving a checkpoint therefore requires its referenced resources as well as the checkpoint directory.

Each saved generation still owns its time-dependent boundary history, inlet position, compensation, and statistical state. Mutable input sequences require a separate version and extent contract. The decision reduces repeated static storage without promising that a checkpoint directory alone, or an arbitrary execution environment, is sufficient for exact continuation.
