# Make GPU data authoritative inside the compute loop

Full ASTR GPU migration will treat GPU-resident fields as the authoritative compute-loop state after initialization and before explicit CPU-owned output, checkpoint, or control boundaries. Whole-field transfers are allowed for initialization and CPU-owned output/checkpoint paths, but per-kernel or per-module whole-field D2H/H2D bridges are rejected because they would turn the port into temporary acceleration rather than a resident GPU solver architecture.

**Consequences**

Statistics, halo exchange, and module coupling must operate on device-resident data and move only scalar reductions or halo buffers through the host in the current baseline. CPU arrays remain available for fallback execution and output boundaries, not as the live state for GPU execution.
