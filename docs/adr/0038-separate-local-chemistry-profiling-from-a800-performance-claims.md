# Separate local chemistry profiling from A800 performance claims

Two local 19.2 GB RTX 4000 Ada GPUs will provide correctness, sanitizer,
profiling, and capacity-preflight evidence. Their FP64 results guide kernel
optimization but do not support authoritative production speedup claims.

A800 measurements use one MPI rank per GPU. The initial matrix contains
`256^3 NP=1`, `384^3 NP=1/2/4`, and `512^3 NP=2/4`. TGV uses `2x2x1` on four
GPUs, while long normal-shock and boundary-layer domains prefer `4x1x1`.
Each case has two warmups and five timed runs with a coefficient of variation
no greater than three percent.

**Consequences**

Correctness and Compute Sanitizer gates precede timing. Disabling chemistry
must keep the existing nonreacting time-step overhead within two percent, and
the resident loop must not transfer full fields. The `384^3` two-to-four GPU
parallel-efficiency target is at least 70 percent. Reports include cell updates
per second, seconds per step, chemistry fraction, ROS2 substep and rejection
statistics, register and local-memory behavior, and occupancy. No CPU speedup
target is invented before the first correct A800 baseline; a later decision
freezes that target from measured evidence.
