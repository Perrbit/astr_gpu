# Preserve exact continuation when simplifying output and restart

Accepted for the output redesign on 2026-09-30. Reducing checkpoint file count or storage must preserve the exact-continuation target under the same executable, backend, MPI decomposition, numerical configuration, and deterministic execution conditions. Necessary conservative, compensation, boundary-history, and statistical continuation state must be retained or reconstructed with demonstrated exact equivalence; tolerance-only reconstruction is not a substitute.

This is an acceptance requirement, not a claim that all existing configurations satisfy it. Numerical-state equality does not require byte-identical file containers or generation metadata. Cross-backend and cross-decomposition restart scope remains a separate decision.
