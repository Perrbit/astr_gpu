# Use role-based parallel HDF5 checkpoint files

Accepted for the output redesign on 2026-09-30. A checkpoint directory contains a small set of HDF5 files organized by purpose plus its inventory, rather than one file per MPI rank. Ranks write their owned regions through parallel I/O without first gathering the entire field onto one process.

The datasets must carry global ownership and any partition-specific state needed for exact original-topology continuation as well as repartitioned reads. Exact grouping, chunking, and publication details remain to be designed. Fewer files do not by themselves imply fewer saved bytes or better I/O performance; both must be measured.
