# Use reduced mixture-averaged laminar transport

The first five-species reaction-flow transport closure will use Blottner
species viscosities, Wilke mixture rules, modified-Eucken
translational-rotational thermal conductivity, Gupta binary diffusion fits,
and correction-velocity mixture-averaged species diffusion. Vibrational heat
conduction and species-carried vibrational energy will enter the modal-energy
flux.

The total-energy diffusion flux will carry complete species enthalpy,
including translational-rotational, vibrational, formation, and pressure-work
contributions. The corrected species fluxes must sum to zero.

**Consequences**

Transport will be enabled in ordered gates: eleven-variable inviscid
advection, viscous stress and translational-rotational conduction,
non-reacting species diffusion, vibrational conduction and species-carried
modal energy, then full viscous reaction flow. The old `chem/astr` omission of
formation energy from diffusive enthalpy and its arbitrary small-denominator
clipping are defects, not compatibility requirements.

Full Stefan-Maxwell diffusion, Soret and Dufour effects, and bulk viscosity
are deferred. Blottner, Gupta, and collision parameters remain candidates
until their provenance and temperature ranges are independently audited.
