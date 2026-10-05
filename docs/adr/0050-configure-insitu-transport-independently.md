# Configure in situ transport independently

Accepted design, 2026-10-05; not yet implemented. In situ configuration explicitly
selects device-buffer MPI or pinned-host face-buffer MPI independently of the
solver's transport selection. Analysis and solver buffers have different
ownership, lifetimes and acceptance scopes, so either context may select a
different backend without changing the other.

Each selection needs its own qualification and resource accounting. There is no
silent backend switch and no full-volume host-mirror exception. This accepts the
configuration boundary, not a working runtime interface.

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
