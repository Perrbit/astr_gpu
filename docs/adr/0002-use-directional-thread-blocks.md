# Use directional thread blocks

For first-stage derivative and filter kernels, x-direction work uses `(512,1,1)`, y-direction work uses `(32,16,1)`, and z-direction work uses `(64,1,8)`. This replaces the earlier universal `(8,8,8)` idea and makes the kernel launch shape an explicit part of the numerical-porting contract.
