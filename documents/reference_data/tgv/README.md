# DLR spectral TGV reference

`spectral_Re1600_512.gdiag` is the DLR high-order CFD workshop spectral
reference for the `512^3`, `Re=1600` Taylor-Green vortex.

- Source: `http://www.as.dlr.de/hiocfd/spectral_Re1600_512.gdiag`
- Retrieved: 2026-09-08
- SHA256: `60bfeaee2dd32de1b9e7171c70cf3774b6053d82a785d526c055547a0757017a`
- Columns: time, kinetic energy, dissipation rate `-dE/dt`, enstrophy
- Time interval: `0.00 <= t <= 19.99`
- Sampling interval: `0.01`

The file is retained byte-for-byte after download. Record its SHA256 in the
campaign environment report before comparing production results.

ASTR uses a weakly compressible `Ma=0.1` formulation, explicit sixth-order
spatial derivatives, and an explicit tenth-order filter. The DLR data therefore
provides a physical diagnostic reference, not a bitwise or discretization-level
oracle.
