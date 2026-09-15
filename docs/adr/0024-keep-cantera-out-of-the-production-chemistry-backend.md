# Keep Cantera out of the production chemistry backend

The first ASTR five-species two-temperature production chemistry path will not
link or call Cantera. The CPU and CUDA Fortran production backends will use the
same fixed Kim-Jo/Park mechanism, thermodynamic constants, reference-energy
convention, and local integration contract. Cantera will remain available only
through an independent optional CPU validation target for the subset it can
represent directly: single-temperature thermodynamics, species reference
energies, selected reaction rates, and equilibrium states.

This boundary avoids host-device field transfers for per-cell chemistry and
removes the Cantera C++ ABI, RPATH, and shared-library requirements from the
production CUDA executable. It also prevents a single-temperature Cantera
state from being treated as an oracle for preferential dissociation,
two-temperature reaction rates, or V-T relaxation.

**Consequences**

The current global `CHEMISTRY/COMB` build path must be separated before the
five-species backend is integrated. The production executable must build and
run without Cantera, and its dynamic-library audit must show no Cantera
dependency. Existing arbitrary-mechanism Cantera combustion cases remain a
separate legacy or reference build path; compatibility with them is not a
property of the new fixed-mechanism backend.
