# Bound the first air-five validation domain

The authoritative zero-dimensional validation domain is 300 to 8000 K for
both translational and vibrational temperature and `1e3` to `1e6` Pa for
pressure. A representative state matrix will cover cold and undissociated
air, partial dissociation, near-equilibrium mixtures, initially absent product
species, and strong translational-vibrational nonequilibrium without taking a
full Cartesian product.

The historical-scale state at `rho=0.01 kg/m3`, `T=Tv=4000 K`, and
`Y_N2/Y_O2=0.7653/0.2347` will be retained, but its trajectory will be
regenerated from the authoritative mechanism and independent oracle.

**Consequences**

States at 200 K, 10000 K, `1e2` Pa, and `1e7` Pa test rejection, finite
behavior, and failure reporting only. Results above 8000 K do not support a
physical-accuracy claim for the fixed-heat-capacity neutral-air model without
electronic excitation or ionization. Zero-dimensional references use SI
units; a separate adapter maps ASTR nondimensional states to chemistry SI
states.
