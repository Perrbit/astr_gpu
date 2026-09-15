# Defer explicit filtering for the first reaction-flow path

The first five-species two-temperature ASTR reaction-flow path will require
`lfilter=f`. The existing full-variable and scalar-workspace tenth-order
explicit central filters remain unchanged for validated non-reacting cases,
but neither filter will be enabled merely by extending its component loop from
five to eleven.

The central filter has negative weights and therefore does not preserve the
positivity of trace species density or vibrational modal energy. Clipping a
negative species or renormalizing all mass fractions would alter elemental
composition and can invalidate the temperature recovered from complete total
energy.

**Consequences**

Chemistry capability checks must reject `lfilter=t` through the first coupled
reaction-flow validation phases. Explicit upwind reconstruction, shock sensing,
and selective characteristic fluxes remain the available stabilization route
for shock-containing chemistry cases. Reactive filtering may be introduced
later only behind a separate numerical-policy decision and tests for species
positivity, total mass, N/O elements, modal-energy admissibility, and complete
total-energy consistency.
