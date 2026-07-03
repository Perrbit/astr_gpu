# Limit first-stage RHS to five flow variables

The first-stage GPU RHS covers only the five conservative variables used by non-reacting Taylor-Green Vortex. Species transport, modal energy equations, chemistry source terms, and turbulence-model equations are deferred so the first correctness loop can isolate the compressible-flow stencil, RHS sign, filter, and RK update.
