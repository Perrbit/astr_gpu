# Retain postprocessing series independently of checkpoints

Accepted for the output redesign on 2026-09-30. Checkpoints, volume-field series, and slice series are separate products with independent enablement and output cadences at the common completed-step boundary. When volume or slice output is enabled, all emitted frames are retained by default; checkpoint retention is a separate policy recorded in the output redesign plan.

This separates exact recovery state from long-lived postprocessing history despite overlap in their numerical fields. Sampling and transfer buffers may be reused, but checkpoint cleanup must not invalidate archived fields or their referenced data. Keeping all emitted frames does not require output at every timestep, and postprocessing field selection cannot remove mandatory restart state.
