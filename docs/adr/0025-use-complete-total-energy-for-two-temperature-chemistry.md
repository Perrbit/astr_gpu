# Use complete total energy for two-temperature chemistry

The fifth ASTR flow conservative variable for the five-species
two-temperature model will contain kinetic energy, translational-rotational
internal energy, vibrational internal energy, and species formation energy:

$$
q_5=\rho\left[\frac{u_i u_i}{2}+e_{\mathrm{tr}}(T,Y)
+e_{\mathrm v}(T_v,Y)+\sum_sY_s e_s^0\right].
$$

Vibrational modal energy will also be evolved by a separate equation, but it
is a component of this total energy rather than an additional energy copy. An
adiabatic constant-volume chemistry step will hold density, momentum, and
`q5` fixed; update species and vibrational modal energy through the coupled
reaction and V-T operator; and recover translational and vibrational
temperatures from the same thermodynamic model.

This definition matches the complete-energy semantics already used by the
mainline Cantera path and avoids combining that state with the alternative
`chem/astr` bookkeeping in which formation energy is excluded from `q5` and
added through an explicit chemical energy source.

**Consequences**

The production chemistry path must not add
`-sum(e0_s*omega_s)` to `q5`. Temperature recovery becomes part of every
accepted chemistry substep and must reject thermodynamically unrealizable
states instead of clipping temperature or merely renormalizing species. CPU
and GPU validation must cover `q5 -> T -> q5`, `Ev -> Tv -> Ev`, elemental
conservation, and complete-energy conservation independently.
