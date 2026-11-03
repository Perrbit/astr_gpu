# Selected Time Statistics Short Test

This is an in-progress TB1 configuration for the existing 16-cubed TGV
fixture, not a production BL/SBLI preset. Read the common configuration through
`ASTR_INSITU_CONFIG`; the existing output configuration selects the dynamic
product list. Volume and legacy field output are not needed. Selected
statistics now use the existing `statistics.h5` within a normal checkpoint;
there is no second restart entry or extra per-product state file. Exact
continuation tests explicitly use `exact_restart=t` and keep the production
default unchanged. Same-backend/topology TGV NP1, NP2 x/y/z and NP4 xy/xz/yz
have passed the 12 vs 5+7 step comparison. The initial NP4 xz failure was traced
to uninitialized CPU primitive-cache corners. The user-approved allocation-time
corner initialization now passes the complete 14-test matrix without relaxing
the bitwise gate. Broader BL/SBLI and product-lifecycle acceptance is pending.

The existing numerical-slice archive is used here as an explicit test oracle.
It transfers only the requested slice, not a complete three-dimensional
field. This is not the final device-rendered mean/fluctuation product.

`catalog_stats` enables sampling without Catalyst or rendering. Its formal
window is in the simulation's physical time units. `selected-time` gathers
the requested primitive fields from completed-step conservative variables,
preserves the established clipped-trapezoid quadrature, and records requested
window, actual interval envelope, covered duration and continuous segment
count in the HDF5 provenance. A gap is excluded from the duration; an interval
envelope is not a statement that all intervening time was sampled.

The present provider admits Reynolds means of density/velocity/pressure/
temperature and Favre means of velocity/pressure/temperature. It does not
label ordinary mean density as Favre density. Gradient quantities use their
primitive dependencies and the existing private stencil; covariance/profile
publication, wall diagnostics, mean images and complete restart acceptance
remain pending.

Test resource settings are local fixture budgets, not production defaults.
