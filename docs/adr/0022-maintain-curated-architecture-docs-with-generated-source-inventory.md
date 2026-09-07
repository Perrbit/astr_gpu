---
status: accepted
---

# Maintain curated architecture docs with a generated source inventory

ASTR core-solver maintenance documentation will keep architecture conclusions and Mermaid diagrams human-curated while using a deterministic read-only source inventory to index files, modules, imports, calls, includes, and CMake membership. This separates maintainers' interpretation of ownership and runtime contracts from lexical facts that can be regenerated and checked for drift.

## Considered options

- A single manually maintained document was rejected because its structure and evidence would become difficult to review as the solver evolves.
- A fully generated call graph was rejected because Fortran generic interfaces, procedure pointers, conditional compilation, and shared module state make lexical edges insufficient as architecture truth.
- Committing both Mermaid and rendered SVG was rejected because it creates two versioned representations and renderer-dependent diffs without a current distribution requirement.

## Consequences

The Markdown Mermaid source is canonical. Current and target architecture are documented separately. A tracked generated Markdown inventory is updated through a standalone script and checked through an explicit maintenance command, without changing CMake, CTest, or CI in this phase. Temporary rendering may be used for validation under `/tmp`, but no rendered image is retained in the repository.
