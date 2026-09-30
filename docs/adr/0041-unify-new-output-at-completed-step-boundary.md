# Unify new output at the completed-step boundary

Accepted for the output redesign on 2026-09-30. New checkpoints, volume fields, slices, and in situ samples observe the same completed-step state, after all updates belonging to that advance and before preprocessing the next one. Products retain independent output cadences and use completed-step counts and actual physical times.

This changes observation placement, not the order or frequency of filters, boundary operations, RK stages, or chemistry updates. Derived fields must correspond to the sampled state, and observation must not mutate authoritative solver state. Exact continuation still requires all necessary restart state rather than only visualization fields. Legacy file compatibility, legacy statistics migration, and initial/final output scheduling remain separate decisions.
