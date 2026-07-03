# Choose a non-reacting second validation case

The first non-TGV validation case for the full-GPU architecture skeleton should remain non-reacting, `num_species=0`, turbulence-free, chemistry-free, immersed-boundary-free, and primarily `numq=5`. Its purpose is to test whether the architecture generalizes beyond Taylor-Green Vortex without opening species, chemistry, turbulence, and irregular boundary complexity at the same time.

**Consequences**

Candidate cases should be screened for stable CPU oracle behavior, controlled input/output paths, and useful non-TGV coverage such as different initialization, boundary-condition, or geometry behavior. Flame, reactor, chemistry, turbulence, and immersed-boundary cases are later physics-expansion targets rather than the second architecture validation case.
