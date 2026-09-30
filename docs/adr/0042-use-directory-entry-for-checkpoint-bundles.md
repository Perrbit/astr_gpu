# Use a directory entry for checkpoint bundles

Accepted for the output redesign on 2026-09-30. Each saved generation is selected through a checkpoint directory containing its required data files and inventory, rather than forcing all state into a single HDF5 file. This keeps member matching inside the solver and permits separation of restart-specific state from data intended for external postprocessing.

Directory organization does not itself guarantee smaller storage or compatibility with other software. File formats, shard counts, external-reader contracts, static-data dependencies, and publication protocols remain separate decisions. Exact continuation and same-backend repartitioned restart requirements remain unchanged.
