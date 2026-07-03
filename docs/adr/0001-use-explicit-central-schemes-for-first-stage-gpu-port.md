# Use explicit central schemes for the first-stage GPU port

For the first-stage TGV GPU validation, derivatives use the explicit sixth-order central difference and filtering uses the explicit tenth-order central filter. We deliberately exclude compact schemes that require tridiagonal solves so the first GPU correctness loop focuses on stencil equivalence, RHS signs, RK updates, and memory movement rather than line-solver implementation risk.
