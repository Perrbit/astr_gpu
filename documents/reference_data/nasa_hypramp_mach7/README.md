# NASA WIND Mach 7 Hypersonic Ramp Reference

This directory pins the small public comparison files for NASA NPARC/WIND
Hypersonic Ramp Study 1, Run E:

- Mach 7 laminar flow over a 15 degree ramp;
- finite-rate nonequilibrium five-species air;
- freestream pressure `14.7 psia` and temperature `520 R`;
- second-order Roe spatial flux, CFL `0.5`, and 8000 steady iterations.

Source: [NASA Hypersonic Ramp Study 1](https://www.grc.nasa.gov/www/wind/valid/hypramp/hypramp01/hypramp01.html)

Pinned files:

| File | Meaning | SHA-256 |
|---|---|---|
| `run.E.dat` | Run E solver input | `6012b6d772afd49e92efe8cb6ab402eb0691c435027646140e131ce25dc7e7ba` |
| `T.E.d` | ramp-surface static temperature versus x | `41e010984aad293557a57206666ff709333e185220f7058e2f21521a154fdc4f` |
| `p.E.d` | ramp-surface static pressure versus x | `b08ef00ff3e378ebcbbb6e7e906bef636381104cd731ec9f219216a3022d7cad` |
| `u.E.d` | streamwise velocity profile at the ramp exit | `f2176b55448a172e9f6d6bfcf6b60d6abe8696e947993c3a5c4859abbb3bdfc4` |

The NASA page explicitly states that no analytical or experimental comparison
is available for this study and that high-temperature-model differences are
small for the selected condition. This case is therefore retained as a public,
reproducible code-to-code benchmark for the later C5-6B shock-capable air5 path.
It is not an experimental validation oracle and does not replace the C5-6A1
independent FP64 two-temperature flat-plate reference.

`tests/gpu_validation/nasa_hypramp_reference.py` verifies the hashes, input
contract, coordinates, units, and basic reported shock response before the data
can be used by an ASTR comparison.
