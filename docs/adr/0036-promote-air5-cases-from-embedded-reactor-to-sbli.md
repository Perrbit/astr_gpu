# Promote air-five cases from an embedded reactor to SBLI

Reaction-flow validation will progress through isolated capability gates:

1. a spatially uniform reactor embedded in a small periodic three-dimensional
   grid;
2. frozen-chemistry quasi-one-dimensional species advection, binary diffusion,
   and vibrational-energy transport on extruded three-dimensional grids;
3. a Cartesian quasi-one-dimensional post-normal-shock relaxation case;
4. the same normal shock extruded to three dimensions and multiple MPI ranks;
5. a high-temperature periodic TGV coupling and performance stress case;
6. a laminar high-enthalpy flat plate followed by finite-rate-air SBLI.

**Consequences**

Every stage isolates a new transport, chemistry, dimensional, or communication
contract before the next is enabled. High-temperature TGV is not a chemistry
physics validation case. Existing fuel-flame, ignition, and combustion cases
are incompatible with the fixed `N2/O2/N/O/NO` mechanism and are excluded.
The quasi-one-dimensional gates use extruded three-dimensional meshes that
satisfy existing stencil and halo requirements rather than opening a separate
one-dimensional GPU path.
