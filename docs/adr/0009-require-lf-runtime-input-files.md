# Require LF runtime input files

ASTR runtime input files under `examples/**/datin` must use LF line endings. NVHPC can preserve `\r` from CRLF input as part of list-directed string tokens, causing `select case(trim(...))` to miss flow types such as `tgv` and skip initialization; normalizing inputs to LF prevents this class of false numerical crashes.
