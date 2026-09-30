# Support same-backend repartitioned restart before backend migration

Accepted for the output redesign on 2026-09-30. The first redesign targets restart with a different MPI rank count or decomposition within the same CPU or GPU backend, while preserving the same mesh and physical problem. CPU/GPU migration is deferred, with format compatibility reserved rather than claimed as implemented.

The unchanged-configuration exact-continuation requirement remains in force. Repartitioned continuation has a separate numerical-equivalence gate because operation and reduction order can change; its tolerances and test matrix are not yet frozen. Storage must retain enough global ownership information to redistribute the necessary continuation state, without selecting a single-file or sharded layout at this stage.
