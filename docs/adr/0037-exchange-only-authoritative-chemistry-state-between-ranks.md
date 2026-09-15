# Exchange only authoritative chemistry state between ranks

Cell-local chemistry advances the existing `is:ie,js:je,ks:ke` active range
without communicating inside adaptive ROS2 substeps. Exterior halos are not
integrated. MPI and periodic duplicate interface planes advance on both sides,
while physical Dirichlet boundary nodes remain boundary-condition owned.

After the first chemistry half-step, ranks exchange only authoritative
`q(1:11)` using existing `qswap` semantics: one duplicate interface plane plus
`hm` exterior halo planes. Primitive variables, species fractions,
translational and vibrational temperatures, and transport properties are
rebuilt from received conservative state. RHS, saved RK state, reaction rates,
and integrator work arrays are not exchanged. The second chemistry half-step
marks halos stale and defers refresh until a real consumer requires them.

**Consequences**

Each chemistry half-step transfers only a compact status record to the host.
MPI reductions combine failure flags, state extrema, substep counts, rejected
steps, and evaluation counts. Any rank failure terminates all ranks together.
Global mass, species, N/O elements, complete total energy, and modal energy are
computed by local cell-volume integration followed by FP64 reduction, avoiding
duplicate interface-node counting. Rank-count comparisons obey conservation
tolerances rather than bitwise reduction reproducibility.
