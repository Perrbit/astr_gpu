# Use host-staged qswap-compatible halo exchange

The first multi-rank GPU validation will exchange `q_d(1:5)` halos through host-staged buffers and blocking `MPI_Sendrecv`, matching CPU `parallel:qswap` by transporting `hm+1` layers so interface planes can be averaged. This keeps full field variables resident on GPU while avoiding CUDA-aware/HIP-aware/device-aware MPI as a first dependency, which preserves a path toward AMD/HIP and domestic DCU backends.

**Considered Options**

- Full-field D2H followed by CPU `qswap`: easiest to debug, but breaks the GPU-resident compute-loop contract.
- CUDA-aware MPI from the start: closest to the long-term performance target, but too dependent on the current NVIDIA/NVHPC stack for the first portable multi-rank baseline.
- Host-staged halo buffers: selected as the correctness baseline because only halo buffers cross the host boundary and the interface can later swap in device-aware MPI.
