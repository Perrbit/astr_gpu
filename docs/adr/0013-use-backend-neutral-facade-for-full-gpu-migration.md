# Use backend-neutral facades for full GPU migration

ASTR's full GPU migration will keep CPU `src/` code responsible for orchestration, runtime input, output, and case lifecycle while GPU backend implementations provide resident data ownership, kernels, halo exchange, and reductions behind backend-neutral facades. CUDA Fortran remains the first implementation backend, but public call sites must not be shaped around CUDA-specific APIs so future HIP/DCU backends can be introduced without rewriting the CPU orchestration layer.

**Considered Options**

- CUDA-only source architecture: fastest short-term path, but it would make future HIP/DCU support a large rewrite.
- Backend-neutral facade with CUDA Fortran first backend: selected because it preserves current momentum while making backend replacement an explicit architecture boundary.
