# Configure in situ transport independently

Accepted design, 2026-10-05; bounded native IS8-A0-A9 implementation accepted
2026-10-06, with scope/evidence in `documents/ASTR_INSITU_IS8_ACCEPTANCE.md`.
In situ configuration explicitly
selects device-buffer MPI or pinned-host face-buffer MPI independently of the
solver's transport selection. Analysis and solver buffers have different
ownership, lifetimes and acceptance scopes, so either context may select a
different backend without changing the other.

Each selection needs its own qualification and resource accounting. There is no
silent backend switch and no full-volume host-mirror exception. This accepts the
configuration boundary; runtime qualification remains limited to the recorded
32³ periodic Cartesian FP64 TGV, NP=1/2 x/y/z, not arbitrary production jobs.

For the new GPU-resident entry, the backend must be supplied explicitly. Missing,
invalid or rank-inconsistent choices fail collectively before flow advancement;
there is no implicit inheritance, automatic selection or default backend. This
requirement does not change the existing CPU or host-compatible entry.

On restart with the same execution backend, mesh and MPI topology, a transport
change requires explicit `restart_output='override'`. Requalify the selected
transport and rebuild transient buffers while preserving statistics, product
clocks and monitoring/event history when transport is the only changed setting.
Persist configuration identity, not pointers, MPI requests or pinned-allocation
handles. This does not admit execution-backend migration or repartitioning.

The device-aware selection requests CUDA binding before MPI initialization even
when solver halos use host buffers. It does not change the solver's selection.
Malformed or rank-inconsistent configuration is still rejected by the later
collective parser. The native extraction/rendering bridge has passed its bounded
gates; unsupported configurations still fail explicitly, with no host fallback.
