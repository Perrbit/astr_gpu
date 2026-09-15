# Store vibrational energy in the unified conservative state

The five-species two-temperature model will store vibrational modal energy in
the same conservative state as flow and species. With no turbulence equations,
the state has eleven components: flow variables in `q(1:5)`, five species
densities in `q(6:10)`, and `Ev=rho*e_v` in `q(11)`. Production code will use
named component indices rather than repeating these literal bounds.

The corresponding RHS, Runge-Kutta snapshot, filter, boundary, checkpoint,
and halo operations will use the unified conservative-variable infrastructure.
Vibrational temperature may remain resident as a performance cache, but it is
derived from density, species, and the modal component and cannot be updated or
checkpointed as an independent authoritative state.

**Consequences**

The GPU path must remove its current `numq=5` capability restriction and audit
every kernel whose dummy-array extent or component loop is fixed at five. The
port will not copy the separate `Ev/Evrhs/Evsave` advancement prototype from
the untracked `chem/astr` reference. This requires a broader initial interface
change, but avoids duplicating RK, filtering, halo, and lifecycle logic and
prevents the total-energy and modal-energy states from silently diverging.
