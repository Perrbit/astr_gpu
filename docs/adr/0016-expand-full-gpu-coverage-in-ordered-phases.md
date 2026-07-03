# Expand full GPU coverage in ordered phases

Full ASTR GPU migration will first stabilize the backend-neutral architecture and non-reacting flow path before expanding to species transport, turbulence models, chemistry, and immersed-boundary support. This order avoids letting the most irregular physics and data structures define the architecture before resident data ownership, halo transport, and validation contracts are stable.

**Consequences**

The next phase should organize the current TGV CUDA Fortran path into a reusable full-GPU architecture skeleton and broaden non-reacting-flow coverage. Species, turbulence, chemistry, and immersed-boundary migration remain explicit later phases rather than immediate additions to the current TGV path.
