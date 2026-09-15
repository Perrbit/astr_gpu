# Use five-species neutral air for the first GPU chemistry backend

The first ASTR GPU chemistry backend will use the fixed
`N2/O2/N/O/NO` twelve-reaction neutral-air mechanism derived from the
Kim-Jo/Park model. This bounded mechanism is selected instead of general
Cantera execution or the complete eleven-species air model so that reaction
tables, thermochemical state, conservation tests, and per-cell CUDA Fortran
execution can be stabilized before introducing ions, electrons, or arbitrary
mechanisms.

**Consequences**

The mechanism input and all derived tables must be versioned and verified
against their literature sources. Historical results in the untracked `chem/`
reference directory are regression hints rather than an authoritative oracle;
the project must regenerate an FP64 CPU baseline with the frozen mechanism.
