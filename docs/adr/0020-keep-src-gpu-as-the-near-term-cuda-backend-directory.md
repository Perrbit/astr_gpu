# Keep src_gpu as the near-term CUDA backend directory

The full-GPU architecture skeleton phase will keep `src_gpu/` as the CUDA Fortran backend implementation directory while stabilizing backend-neutral facades and internal module boundaries. Large directory moves to names such as `src_backend_cuda/` or `src_backend_hip/` are deferred until a second backend or stronger evidence justifies the churn.

**Consequences**

The near-term architecture work should focus on preventing CUDA-specific imports from leaking into `src/` and clarifying facade boundaries. Directory reshuffling is not a substitute for a stable backend-neutral interface.
