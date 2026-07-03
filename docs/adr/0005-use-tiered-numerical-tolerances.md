# Use tiered numerical tolerances

First-stage CPU/GPU validation uses stricter tolerances for algebraic conversion and halo copy, and looser but explicit tolerances for stencil modules. This avoids masking real logic errors while not failing the GPU port solely because kernel launch order, fused operations, or accumulation order differs from CPU execution.
