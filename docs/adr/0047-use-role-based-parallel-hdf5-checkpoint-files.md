# Use role-based parallel HDF5 checkpoint files

Accepted for the output redesign on 2026-09-30. A checkpoint directory contains a small set of HDF5 files organized by purpose plus its inventory, rather than one file per MPI rank. Ranks write their owned regions through parallel I/O without first gathering the entire field onto one process.

The datasets must carry global ownership and any partition-specific state needed for exact original-topology continuation as well as repartitioned reads. The implemented grouping and publication contract are recorded in `documents/ASTR_OUTPUT_RESTART_REDESIGN_PLAN.md` sections 10.77-10.78; small versioned control records are not represented as parallel HDF5 files. Fewer files do not by themselves imply fewer saved bytes or better I/O performance; both must be measured.
