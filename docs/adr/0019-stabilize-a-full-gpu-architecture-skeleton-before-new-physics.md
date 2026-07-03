# Stabilize a full-GPU architecture skeleton before new physics

The next full ASTR GPU migration phase will stabilize a reusable architecture skeleton instead of immediately adding species, turbulence, chemistry, or immersed-boundary support. The phase will organize backend-neutral facades, GPU-authoritative data ownership, HaloTransport boundaries, kernel module taxonomy, and reusable validation scripts so the current TGV CUDA Fortran path becomes an architecture baseline rather than a one-case implementation.

**Consequences**

Success for the next phase is measured by architecture clarity and repeatable validation, not by the number of newly ported physics models. A second non-reacting-flow case should be selected only after the skeleton boundaries are stable enough to absorb it without TGV-specific naming or data ownership leaking into the public GPU interface.
