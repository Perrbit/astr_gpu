# Use pluggable HaloTransport backends

Full ASTR GPU migration will keep halo-exchange semantics separate from the transport backend. The current host-staged blocking MPI path remains the L0 correctness baseline, while later backends may add nonblocking host-staged MPI, pinned host buffers, CUDA-aware MPI, HIP-aware MPI, or multi-node topology-aware transport without changing CPU orchestration call sites.

**Considered Options**

- CUDA-aware MPI as the only target: attractive for NVIDIA performance, but too tightly coupled to the current CUDA/NVHPC environment and not aligned with future HIP/DCU needs.
- Host-staged blocking MPI only: portable and validated, but not a sufficient long-term performance endpoint.
- Pluggable HaloTransport: selected because it preserves the validated correctness baseline while creating a path to backend-specific performance transports.
