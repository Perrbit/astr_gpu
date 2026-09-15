# Generate fixed air-five constants from one versioned JSON source

`chemMech/air5_kimjo12.json` will be the only manually maintained source for
the first five-species mechanism, thermodynamics, and transport data. A
repository generator will produce a shared
`src/chemistry_air5_data.F90` module containing compile-time constants for
both CPU and CUDA Fortran production paths.

The JSON and generated Fortran module will both be tracked. A validation test
will regenerate the module, compare it byte for byte, and verify a content
hash. The independent Python oracle reads the JSON directly.

**Consequences**

The production executable accepts only the fixed mechanism identifier and
does not parse or broadcast mechanism files at runtime. Startup output reports
the mechanism identifier and version but not SHA-256. This removes per-rank
file access and parser ambiguity while keeping parameter provenance auditable.
Support for arbitrary runtime mechanisms requires a separate future decision.
