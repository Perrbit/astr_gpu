# ASTR Core Solver Maintenance Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-audited Chinese maintenance documentation system for the tracked ASTR core solver, with ten Mermaid architecture views and a deterministic inventory drift check.

**Architecture:** Human-curated documents own architectural meaning, runtime contracts, and maintenance guidance. A standard-library Python scanner owns reproducible lexical facts for tracked `src/`, `src_gpu/`, and `src/CMakeLists.txt` content, and checks machine-readable source anchors of the form `` `src/path::symbol` ``. Current implementation, target architecture, numerical evidence, physical validation, and performance evidence remain separate.

**Tech Stack:** Fortran and CUDA Fortran source inspection, CMake source-list inspection, Python 3 standard library, `unittest`, Markdown, Mermaid, Git, temporary Mermaid CLI rendering through `npx` and system Chrome.

## Global Constraints

- Audit only files tracked by the current Git `HEAD`; record the latest commit that changed `CMakeLists.txt`, `src/`, or `src_gpu/` as the stable source baseline.
- Detailed scope is the root and `src/` CMake configuration, production code under `src/` and `src_gpu/`, and validation scripts used as evidence.
- Do not modify solver behavior, Fortran or CUDA Fortran source, CMake, CTest, CI, case inputs, or runtime output.
- Do not stage or rewrite unrelated untracked files.
- Do not interrupt `astr-opensbli-np2.service`.
- Write maintenance prose in Chinese while preserving source identifiers and standard technical terms in English.
- Keep current and target architecture in separate files and separate diagrams.
- Use file path plus module, program, subroutine, or function name for source evidence; do not use fixed line numbers.
- Mermaid source in Markdown is canonical. Every diagram uses `accTitle`, `accDescr`, `snake_case` node IDs, and `classDef` instead of inline styles.
- Render diagrams only under a unique `/tmp` directory and delete that directory after inspection. Do not retain SVG, PNG, PDF, EPS, or JPEG renders.
- Use only the Python standard library for `scripts/maintenance/audit_source_inventory.py`.
- Keep `use_gpu` as a runtime input decision; `ASTR_WITH_CUDA` only determines whether the binary contains the CUDA backend.
- Document CPU-owned HDF5, checkpoint, restart, and controller boundaries without implying GPU HDF5 support.
- Preserve explicit central and upwind GPU scope. Compact, species, chemistry, turbulence, moving-grid, and immersed-boundary GPU support remain unsupported or deferred unless current source proves otherwise.

---

### Task 1: Establish the tracked-source scanner contract

**Files:**
- Create: `tests/maintenance/test_audit_source_inventory.py`
- Create: `scripts/maintenance/audit_source_inventory.py`

**Interfaces:**
- Consumes: repository root containing `.git`, `src/`, `src_gpu/`, and `src/CMakeLists.txt`.
- Produces: `FortranFacts`, `tracked_core_files(repo)`, `logical_fortran_lines(text)`, and `parse_fortran(path, text)` for later inventory and evidence checks.

- [x] **Step 1: Write scanner unit tests against temporary miniature repositories**

Create tests that load the script with `importlib.util.spec_from_file_location`. The fixture helper is concrete and commits only the supplied files:

```python
def make_repo(self, files: dict[str, str]) -> Path:
    repo = Path(tempfile.mkdtemp(prefix="astr-inventory-test-"))
    self.addCleanup(shutil.rmtree, repo)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    for relative, content in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return repo
```

Assert all of the following:

```python
def test_logical_fortran_lines_fold_continuations_and_strip_comments(self):
    text = """module demo
      use alpha, only: first, &
           & second ! trailing comment
      call advance(state, &
           & dt)
    end module demo
    """
    self.assertEqual(
        audit.logical_fortran_lines(text),
        ["module demo", "use alpha, only: first, second",
         "call advance(state, dt)", "end module demo"],
    )

def test_parse_fortran_records_declarations_and_edges(self):
    facts = audit.parse_fortran(
        Path("src/demo.F90"),
        "module demo\nuse alpha\ncontains\nsubroutine run\n"
        "call step\nend subroutine run\nend module demo\n",
    )
    self.assertEqual(facts.modules, ("demo",))
    self.assertEqual(facts.subroutines, ("run",))
    self.assertEqual(facts.uses, ("alpha",))
    self.assertEqual(facts.calls, ("step",))

def test_tracked_core_files_excludes_untracked_sources(self):
    repo = self.make_repo({"src/a.F90": "module a\nend module a\n"})
    (repo / "src/untracked.F90").write_text("module hidden\nend module hidden\n")
    self.assertEqual(
        [p.as_posix() for p in audit.tracked_core_files(repo)],
        ["src/a.F90"],
    )
```

The test helper initializes a temporary Git repository, configures a local test identity, adds only the supplied files, and commits them. Add tests for a `program`, typed `function`, `attributes(global) subroutine`, Fortran `include`, preprocessor `#include`, semicolon-separated statements, quoted `!`, and stable case-insensitive deduplication.

- [x] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
python3 -m unittest discover -s tests/maintenance -p 'test_audit_source_inventory.py' -v
```

Expected: failure because `scripts/maintenance/audit_source_inventory.py` does not exist.

- [x] **Step 3: Implement source discovery and Fortran lexical parsing**

Implement the `FortranFacts` data contract:

```python
@dataclass(frozen=True)
class FortranFacts:
    path: str
    programs: tuple[str, ...]
    modules: tuple[str, ...]
    subroutines: tuple[str, ...]
    functions: tuple[str, ...]
    uses: tuple[str, ...]
    calls: tuple[str, ...]
    includes: tuple[str, ...]
    sha256: str

SOURCE_SUFFIXES = {".f90", ".cuf", ".inc"}
```

Expose the exact signatures `tracked_core_files(repo: Path) -> list[Path]`,
`logical_fortran_lines(text: str) -> list[str]`, and
`parse_fortran(path: Path, text: str) -> FortranFacts`.

`tracked_core_files` must call `git -C <repo> ls-files -z -- src src_gpu`, filter by `SOURCE_SUFFIXES` case-insensitively, return repository-relative paths, and sort with POSIX path ordering. The logical-line parser must preserve `!` inside quoted strings, split semicolon statements outside strings, fold trailing and leading ampersands, and ignore blank or pure-comment statements. Declaration matching must exclude `end module`, `module procedure`, and interface-only false positives where possible; unresolved lexical ambiguity is reported later instead of guessed.

- [x] **Step 4: Run scanner unit tests**

Run the focused `unittest` command again.

Expected: all Task 1 tests pass.

- [x] **Step 5: Commit the scanner foundation**

```bash
git add scripts/maintenance/audit_source_inventory.py tests/maintenance/test_audit_source_inventory.py
git commit -m "test(maintenance): define tracked source scanner contract"
```

### Task 2: Add CMake membership, source anchors, deterministic rendering, and CLI checks

**Files:**
- Modify: `scripts/maintenance/audit_source_inventory.py`
- Modify: `tests/maintenance/test_audit_source_inventory.py`
- Create: `documents/maintenance/generated/source-inventory.md`

**Interfaces:**
- Consumes: Task 1 `FortranFacts` and tracked source paths.
- Produces: `CMakeMembership`, `parse_cmake_membership`, `validate_source_anchors`, `render_inventory`, `--write`, and `--check`.

- [x] **Step 1: Add failing tests for CMake and evidence contracts**

Add fixture helpers that construct `FortranFacts` and `InventorySnapshot` with explicit empty tuples and fixed hashes. Add tests with these assertions:

```python
def test_parse_cmake_membership_resolves_cpu_and_gpu_lists(self):
    cmake = """set(ASTR_SOURCES a.F90 body.inc)
    set(ASTR_GPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/k.cuf)
    add_executable(astr ${ASTR_SOURCES} ${ASTR_GPU_SOURCES})
    """
    membership = audit.parse_cmake_membership(cmake)
    self.assertEqual(membership["src/a.F90"], "astr CPU")
    self.assertEqual(membership["src_gpu/k.cuf"], "astr CUDA")

def test_validate_source_anchors_rejects_missing_symbol(self):
    facts = {"src/a.F90": self.facts("src/a.F90", modules=("a",))}
    errors = audit.validate_source_anchors("Evidence `src/a.F90::missing`", facts)
    self.assertEqual(errors, ["source anchor symbol not found: src/a.F90::missing"])

def test_two_inventory_renders_are_byte_identical(self):
    snapshot = audit.InventorySnapshot(
        baseline="0123456789abcdef",
        facts=(self.facts("src/a.F90", modules=("a",)),),
        memberships=(("src/a.F90", "astr CPU"),),
        include_owners=(),
        fingerprint="f" * 64,
    )
    first = audit.render_inventory(snapshot)
    second = audit.render_inventory(snapshot)
    self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

def test_check_fails_for_stale_inventory(self):
    repo = self.make_repo({
        "src/a.F90": "module a\nend module a\n",
        "src/CMakeLists.txt": "set(ASTR_SOURCES a.F90)\n"
        "set(ASTR_GPU_SOURCES)\n"
        "add_executable(astr ${ASTR_SOURCES} ${ASTR_GPU_SOURCES})\n",
    })
    self.assertEqual(audit.main(["--repo", str(repo), "--write"]), 0)
    inventory = repo / audit.INVENTORY_PATH
    inventory.write_text(inventory.read_text() + "stale\n")
    self.assertEqual(audit.main(["--repo", str(repo), "--check"]), 1)
```

Define `self.facts(path, modules=())` to return all other `FortranFacts` tuple fields empty and `sha256="0" * 64`. Also test missing CMake membership, duplicate source-list membership, include-body ownership, missing evidence path, mutually exclusive or absent CLI mode, and a docs-only commit leaving the source baseline unchanged.

- [x] **Step 2: Run tests and verify failures identify missing interfaces**

Run the focused `unittest` command.

Expected: Task 1 tests pass and new tests fail because CMake, evidence, rendering, and CLI functions are absent.

- [x] **Step 3: Implement the inventory snapshot and command interface**

Add these exact interfaces:

```python
@dataclass(frozen=True)
class InventorySnapshot:
    baseline: str
    facts: tuple[FortranFacts, ...]
    memberships: tuple[tuple[str, str], ...]
    include_owners: tuple[tuple[str, tuple[str, ...]], ...]
    fingerprint: str

INVENTORY_PATH = Path("documents/maintenance/generated/source-inventory.md")
SOURCE_ANCHOR = re.compile(r"`((?:src|src_gpu)/[^`:]+)::([A-Za-z][A-Za-z0-9_]*)`")
```

Expose the exact signatures `parse_cmake_membership(text: str) -> dict[str, str]`,
`build_snapshot(repo: Path) -> tuple[InventorySnapshot, list[str]]`,
`validate_source_anchors(markdown: str, facts_by_path: Mapping[str, FortranFacts]) -> list[str]`,
`render_inventory(snapshot: InventorySnapshot) -> str`, and
`main(argv: Sequence[str] | None = None) -> int`.

Normalize CMake entries relative to `src/CMakeLists.txt`; classify files in `ASTR_SOURCES` as `astr CPU` and files in `ASTR_GPU_SOURCES` as `astr CUDA`. Classify `.inc` files through parsed include owners. Treat a tracked `.F90` or `.cuf` absent from the `astr` source lists as an error. Compute `baseline` with `git log -1 --format=%H -- CMakeLists.txt src src_gpu`, so documentation-only commits do not invalidate the inventory. Render a Chinese Markdown inventory containing baseline commit, combined SHA-256 fingerprint, scanner limitations, one row per tracked production source, declarations, `use`, `call`, include edges, and membership. Derive each concise responsibility as `Defines program <name>`, `Defines module <name>`, or `Implementation include owned by <source>`; do not infer physics from file names.

`--write` writes with a final newline only when there are no membership errors. `--check` compares exact generated bytes, scans `documents/maintenance/*.md` recursively for source anchors, prints each error to stderr, and returns nonzero for stale output, absent membership, duplicate membership, missing path, or missing symbol.

- [x] **Step 4: Run unit tests and generate the real inventory**

```bash
python3 -m unittest discover -s tests/maintenance -p 'test_audit_source_inventory.py' -v
python3 scripts/maintenance/audit_source_inventory.py --write
python3 scripts/maintenance/audit_source_inventory.py --check
```

Expected: tests pass; `--write` reports 67 tracked production sources at the current implementation baseline; `--check` reports the inventory is current. If the count differs, stop and reconcile it with `git ls-files 'src/*' 'src_gpu/*'` rather than forcing the expected number.

- [x] **Step 5: Prove the stale-inventory gate fails without changing tracked output**

Copy the repository to a temporary directory with `git archive HEAD`, run `--write`, append one line to the temporary inventory, and run `--check` there.

Expected: nonzero exit with `generated source inventory is stale`; the working repository remains unchanged.

- [x] **Step 6: Commit the complete inventory tool**

```bash
git add scripts/maintenance/audit_source_inventory.py tests/maintenance/test_audit_source_inventory.py documents/maintenance/generated/source-inventory.md
git commit -m "feat(maintenance): add deterministic source inventory audit"
```

### Task 3: Document navigation and repository ownership

**Files:**
- Create: `documents/maintenance/README.md`
- Create: `documents/maintenance/repository-structure.md`

**Interfaces:**
- Consumes: generated inventory and tracked root/`src` CMake membership.
- Produces: the entry page and ownership map referenced by every later guide.

- [x] **Step 1: Write the navigation page with one Mermaid flowchart**

Use these sections in order: `Purpose and scope`, `Choose a reader path`, `Document map`, `Evidence and freshness`, `Excluded scope`. The Mermaid diagram must route `new_student` to build/run/restart and runtime documents, and `core_maintainer` to current architecture, parallel GPU, numerical contracts, change impact, and validation documents. Add links for all ten planned maintenance files even if later files do not yet exist, and state that running the audit `--check` is mandatory before accepting documentation changes.

- [x] **Step 2: Write repository ownership with one Mermaid flowchart and complete source groups**

Group all inventory rows into these audited responsibilities: entry/orchestration, shared state, MPI, geometry/grid, initialization, numerics, boundary treatment, statistics/I/O, CUDA facade/runtime, CUDA state, CUDA numerics, CUDA boundaries, and CUDA halo transport. Cite the current build boundary with `` `src/astr.F90::astr` ``, `` `src/CMakeLists.txt` `` as a plain path, and the GPU facade with `` `src_gpu/gpu_runtime.cuf::gpu_runtime` ``. Explicitly label `pastr/`, `miniapps/`, `chemMech/`, most examples, presentations, and runtime output as outside detailed scope.

- [x] **Step 3: Validate coverage and links available at this stage**

Run a shell check that extracts all `src/` and `src_gpu/` paths from the generated inventory and confirms each is represented by exactly one repository responsibility group. Run `audit_source_inventory.py --check`; missing later document links are allowed only until Task 10, so record them in command output rather than weakening the final link gate.

- [x] **Step 4: Commit repository navigation**

```bash
git add documents/maintenance/README.md documents/maintenance/repository-structure.md
git commit -m "docs(maintenance): map core repository ownership"
```

### Task 4: Document the confirmed current architecture

**Files:**
- Create: `documents/maintenance/architecture-current.md`

**Interfaces:**
- Consumes: source inventory, `ASTR_SRC_ARCHITECTURE_MEMORY.md`, and current source anchors.
- Produces: source-audited current layered architecture and core dependency views.

- [x] **Step 1: Build the current layered architecture diagram**

Document the implemented layers only: command/runtime input, CPU orchestration, backend facade, CPU solver, CUDA backend, MPI/HaloTransport, shared dependencies, and CPU-owned I/O. The diagram must show runtime `use_gpu` selection after `readinput`, compile-time `_CUDA` containment, and the stop path when a non-CUDA binary receives `use_gpu=t`. Evidence must include `` `src/astr.F90::astr` ``, `` `src/readwrite.F90::readinput` ``, `` `src/mainloop.F90::time_integration_rk` ``, and `` `src_gpu/gpu_runtime.cuf::gpu_runtime` ``.

- [x] **Step 2: Build the core module dependency diagram**

Use grouped nodes for state (`commvar`, `commarray`), setup (`parallel`, `gridgeneration`, `geom`, `initialisation`), CPU compute (`mainloop`, `solver`, `filter`, `derivative`, `bc`), GPU compute (`gpu_runtime`, `mainloop_gpu`, `solver_gpu`, `boundary_gpu`, `commarray_gpu`), communication (`parallel`, `halo_exchange_gpu`, `halo_transport_gpu`), and outputs (`statistic`, `statistic_gpu`, `readwrite`, `hdf5io`). Include only confirmed `use` or facade-call edges and place a source-evidence table immediately below the diagram.

- [x] **Step 3: State implemented ownership and explicit limits**

Add concise sections for CPU orchestration ownership, GPU-authoritative compute-loop fields, CPU-owned output boundaries, runtime capability rejection, and the current CUDA-specific backend. Separate each statement into `confirmed behavior`, `evidence`, and `maintenance consequence`. Link detailed numerical and target material instead of copying formulas or roadmaps.

- [x] **Step 4: Validate all current claims against source anchors and commit**

```bash
python3 scripts/maintenance/audit_source_inventory.py --check
git add documents/maintenance/architecture-current.md
git commit -m "docs(maintenance): describe current core architecture"
```

Expected: every machine-readable source anchor resolves; no target-only backend is presented as implemented.

### Task 5: Document startup, RK ordering, and host/device field ownership

**Files:**
- Create: `documents/maintenance/runtime-and-dataflow.md`

**Interfaces:**
- Consumes: `astr`, CPU `steploop`/`time_integration_rk`/`rkfirst`, GPU facade, GPU RK driver, and field-copy routines.
- Produces: startup flow, RK sequence, and field ownership views.

- [x] **Step 1: Write the startup flow diagram from the program call order**

Preserve this audited sequence: MPI initialization, command parsing, input read, decomposition, reference values, optional device bind, file setup, host allocation, optional device allocation, grid/geometry/sponge/flow initialization, optional flow upload and GPU backend initialization, `steploop`, optional GPU finalization, MPI finalization. Mark `pp` and `test` as alternate command paths without expanding them. Cite `` `src/astr.F90::astr` ``, `` `src_gpu/gpu_runtime.cuf::gpu_bind_device` ``, `` `src_gpu/gpu_runtime.cuf::gpu_after_alloc` ``, and `` `src_gpu/gpu_runtime.cuf::gpu_after_flowinit` ``.

- [x] **Step 2: Write separate CPU and GPU RK lanes in one sequence diagram**

Show the common `steploop` control lane, statistics preparation, checkpoint/output boundary, three RK substages, filter and boundary ordering, solution halo exchange, primitive refresh, convection, diffusion, source terms, RHS update, statistics, and explicit synchronization policy. Where a case-specific branch changes order, label it as a branch and link to the owning procedure instead of drawing every kernel. Cite `` `src/mainloop.F90::steploop` ``, `` `src/mainloop.F90::time_integration_rk` ``, `` `src_gpu/gpu_runtime.cuf::gpu_prepare_rkfirst_stats` ``, and `` `src_gpu/mainloop_gpu.cuf::time_integration_rk_gpu` ``.

- [x] **Step 3: Write the field ownership state diagram**

Use states `host_initialized`, `device_authoritative`, `host_output_snapshot`, and `finalized`. Label whole-field H2D at initialization, device-resident kernels and halo packs during integration, whole-field or boundary D2H at explicit output/statistics/checkpoint points, and return to device-authoritative state. Distinguish `copy_flow_from_gpu` from `copy_output_boundary_from_gpu`; do not claim device-side HDF5.

- [x] **Step 4: Validate ordering and commit**

Run the inventory check and manually compare diagram order with the named procedures. Record unresolved case-specific branches in a limitations table rather than flattening them into a false universal sequence.

```bash
git add documents/maintenance/runtime-and-dataflow.md
git commit -m "docs(maintenance): trace runtime and field ownership"
```

### Task 6: Document MPI decomposition and GPU HaloTransport

**Files:**
- Create: `documents/maintenance/parallel-gpu-architecture.md`

**Interfaces:**
- Consumes: CPU `parallel` exchanges, GPU pack/transport/unpack modules, private tag range, and validation matrices.
- Produces: a halo sequence diagram and maintainable communication contract.

- [x] **Step 1: Write the decomposition and neighbor contract**

Describe Cartesian `isize x jsize x ksize`, rank neighbors, physical versus homogeneous axes, fixed `hm` stencil halos, and the CPU `qswap` use of `hm+1` where endpoint swap/averaging requires it. Do not state that every exchange uses the same depth. Cite `` `src/parallel.F90::parallelini` ``, `` `src/parallel.F90::qswap` ``, and `` `src_gpu/halo_exchange_gpu.cuf::exchange_solution_halo_gpu` ``.

- [x] **Step 2: Draw the GPU halo sequence**

Show pack kernel, explicit synchronization, device-to-host staging, `exchange_host_pair`, host-to-device staging, unpack kernel, explicit synchronization, and primitive halo refresh. Add branches for local periodic copy, physical boundary ownership, pageable, pinned, and pinned-overlap transport. Record private MPI tags `21001..21006` as a contract owned by `halo_exchange_gpu`, not as a global MPI convention.

- [x] **Step 3: Compare CPU and GPU communication semantics**

Use a table covering storage authority, exchanged payload, MPI API, staging, synchronization, endpoint treatment, error handling, and current portability. Explain that the GPU backend preserves halo semantics but not CPU implementation mechanics. State that CUDA-aware and HIP-aware transport are target options, not current claims.

- [x] **Step 4: Bind communication changes to evidence**

Link `ASTR_GPU_MULTI_RANK_PORTING_PLAN.md`, `GPU_VALIDATION_MATRIX.md`, and current halo validation scripts. Require same-topology CPU/GPU field or statistics comparison for correctness and separate rank-scaling/profile evidence for performance.

- [x] **Step 5: Validate and commit**

```bash
python3 scripts/maintenance/audit_source_inventory.py --check
git add documents/maintenance/parallel-gpu-architecture.md
git commit -m "docs(maintenance): define MPI GPU halo contracts"
```

### Task 7: Document target architecture and numerical contracts

**Files:**
- Create: `documents/maintenance/architecture-target.md`
- Create: `documents/maintenance/numerical-contracts.md`

**Interfaces:**
- Consumes: accepted ADRs, `ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`, numerical-format documentation, and current capability predicates.
- Produces: clearly labeled target backend view and concise numerical maintenance contracts.

- [x] **Step 1: Draw the target backend architecture without implying completion**

Show backend-neutral CPU orchestration and facades feeding current CUDA Fortran plus planned HIP/DCU and future transport backends. Visually classify nodes as `implemented`, `planned`, `restricted`, or `deferred`. Keep `src_gpu/` as the current CUDA directory; do not prescribe a directory rename before a second backend exists. Keep species, chemistry, turbulence, moving grid, and immersed boundary deferred.

- [x] **Step 2: Write target migration rules**

State the stable contracts for runtime selection, device field ownership, HaloTransport replacement, output boundaries, capability admission, and validation layers. Cite plans and ADRs as design evidence, while source anchors are used only for current components.

- [x] **Step 3: Write numerical contract summaries**

For each supported family, provide `purpose`, `entry points`, `required data`, `ordering invariant`, `boundary/halo requirement`, `validation evidence`, and `unsupported combinations`. Cover explicit sixth-order central derivatives, explicit tenth-order central filtering with ping-pong storage, explicit upwind reconstruction, flux splitting, Ducros sensor, selective Roe characteristic override, RK3 sign/order, diffusion, sponge/source terms, and physical-boundary closure.

- [x] **Step 4: Link formulas instead of duplicating them**

Link the existing numerical-format and porting documents for coefficients and detailed formulas. State that a formula is not validated merely because a corresponding kernel exists; identify separate CPU/GPU equivalence, physical, and performance gates.

- [x] **Step 5: Validate terminology, target labels, and commit**

```bash
python3 scripts/maintenance/audit_source_inventory.py --check
git add documents/maintenance/architecture-target.md documents/maintenance/numerical-contracts.md
git commit -m "docs(maintenance): define target and numerical contracts"
```

### Task 8: Document build, run, restart, and CPU-owned output boundaries

**Files:**
- Create: `documents/maintenance/build-run-restart.md`

**Interfaces:**
- Consumes: root/`src` CMake, runtime input parsing, restart source, OpenSBLI service wrapper, and platform constraints.
- Produces: reproducible platform-neutral operations guide plus local NVHPC and A800/HPC differences.

- [x] **Step 1: Write build contracts**

Document required MPI and parallel HDF5 dependencies, GNU CPU build, NVHPC CUDA-capable build with `-DASTR_WITH_CUDA=ON`, and runtime `use_gpu`. Use placeholder-free shell variables such as `ASTR_ROOT="$PWD"` and `BUILD_DIR="$ASTR_ROOT/build_gpu"`; do not include private accounts, hosts, credentials, or remote paths. Explain that login-node compilation and compute-node execution are site policy choices, not solver semantics.

- [x] **Step 2: Write run and device-binding contracts**

Document `astr run <input>`, CRLF input rejection/normalization, one-rank-per-GPU intent, oversubscription as correctness-only evidence, topology consistency, GPU visibility checks, and `nvitop`/`nvidia-smi` observation. Keep all commands non-destructive and state stop conditions for invalid inputs or unavailable GPU support.

- [x] **Step 3: Write the restart state machine and checklist**

Cover checkpoint immutability, step/time agreement, topology constraints, host read, physical and MPI halo reconstruction, upload into authoritative device fields, statistics continuation, and output ownership. Cite `` `src/readwrite.F90::readcheckpoint` ``, `` `src/readwrite.F90::writechkpt` ``, and `` `src_gpu/gpu_runtime.cuf::gpu_sync_flow_to_device` ``. Link the OpenSBLI restart validation script without embedding service-specific runtime state.

- [x] **Step 4: Validate commands without launching a solver job**

Run `cmake --help`-level syntax checks only where they do not create project build output. Verify all referenced scripts and input keys exist. Do not compile or run the production case during this documentation task.

- [x] **Step 5: Commit the operations guide**

```bash
git add documents/maintenance/build-run-restart.md
git commit -m "docs(maintenance): document build run and restart"
```

### Task 9: Write change-impact recipes and diagnostic stop conditions

**Files:**
- Create: `documents/maintenance/change-impact-guide.md`
- Create: `documents/maintenance/validation-and-troubleshooting.md`

**Interfaces:**
- Consumes: all current architecture and operations documents plus validation scripts.
- Produces: six repeatable maintenance workflows and the final validation-gate diagram.

- [x] **Step 1: Write six change-impact recipes**

Create one section each for adding a case, boundary condition, numerical scheme, device field, HaloTransport behavior, and statistic. Every section must contain: decision boundary, exact source entry points, ownership changes, MPI/GPU semantics, invariants, minimum tests, evidence required for acceptance, and stop conditions. For CPU bugs discovered during GPU work, require explicit user/maintainer decision before reproducing or changing behavior.

- [x] **Step 2: Draw the layered validation gate**

Use ordered gates for build/runtime, module contract, one-step field equivalence, multi-step statistics, multi-rank same-topology comparison, physical validation, residency/profile evidence, and performance. Add failure exits that stop promotion at each layer. Do not allow a later gate to compensate for an earlier failure.

- [x] **Step 3: Write troubleshooting entries in the fixed five-part format**

Use `Symptom -> Diagnostic criterion -> Root cause -> Action -> Stop condition` for CRLF strings, NaN/CFL failures, invalid CPU oracle, boundary/halo ownership, restart mismatch, HDF5 dependencies, rank-to-device binding, low GPU utilization, unexpected whole-field transfers, Compute Sanitizer errors, Nsight Systems gaps, and `ERR_NVGPUCTRPERM`. Commands must be diagnostic and non-destructive. A detected code-logic defect ends the run path and is reported instead of being masked by further execution.

- [x] **Step 4: Link concrete validation entry points**

Reference representative TGV, multi-rank, CURVE, wall-family, shock/SBLI, restart, field-compare, Compute Sanitizer, Nsight Systems, and Nsight Compute scripts. Clearly label each as equivalence, physics, robustness, residency, or performance evidence.

- [x] **Step 5: Validate and commit**

```bash
python3 scripts/maintenance/audit_source_inventory.py --check
git add documents/maintenance/change-impact-guide.md documents/maintenance/validation-and-troubleshooting.md
git commit -m "docs(maintenance): add change and diagnostic guides"
```

### Task 10: Run full documentation acceptance and close the maintenance set

**Files:**
- Modify: `documents/maintenance/README.md`
- Modify: `documents/maintenance/generated/source-inventory.md`
- Modify only if source audit proves drift: other `documents/maintenance/*.md`

**Interfaces:**
- Consumes: all nine curated documents, generated inventory, audit script, tests, and ten Mermaid blocks.
- Produces: one internally linked, source-audited maintenance set with no retained render artifacts.

- [x] **Step 1: Refresh the inventory at final `HEAD` and prove determinism**

```bash
python3 scripts/maintenance/audit_source_inventory.py --write
cp documents/maintenance/generated/source-inventory.md /tmp/astr-source-inventory.first.md
python3 scripts/maintenance/audit_source_inventory.py --write
cmp /tmp/astr-source-inventory.first.md documents/maintenance/generated/source-inventory.md
rm /tmp/astr-source-inventory.first.md
python3 scripts/maintenance/audit_source_inventory.py --check
```

Expected: `cmp` is silent and `--check` succeeds.

- [x] **Step 2: Run tests, placeholder scan, relative-link scan, and Git whitespace checks**

```bash
python3 -m unittest discover -s tests/maintenance -p 'test_*.py' -v
rg -n 'T[B]D|T[O]DO|implement[ ]later|fill[ ]in[ ]details' documents/maintenance scripts/maintenance tests/maintenance
git diff --check
```

Expected: unit tests pass; placeholder scan returns no matches; whitespace check is silent. Run a standard-library Markdown link checker over `documents/maintenance/` and require every relative target to exist.

- [x] **Step 3: Extract and temporarily render all ten Mermaid blocks**

Create a unique directory with `mktemp -d /tmp/astr-maintenance-mermaid.XXXXXX`, extract each Mermaid fence into a numbered `.mmd`, and run Mermaid CLI with an isolated npm cache and `PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome`. Render PNG only for inspection. Check every image for clipped labels, overlapping nodes, unreadable text, and incoherent edges. Delete the exact temporary directory after inspection.

Expected: ten diagrams render successfully and no `.svg`, `.png`, `.pdf`, `.eps`, or `.jpeg` is added under the repository.

- [x] **Step 4: Perform the acceptance audit**

Confirm:

```text
all tracked src/src_gpu production files appear exactly once
all CMake memberships and include owners are resolved
all source anchors resolve to a tracked file and declared symbol
current and target architecture remain separate
six change recipes are present
all required troubleshooting topics use the five-part structure
build, run, restart, ownership, and validation claims agree with source
no solver, CMake, CTest, CI, runtime output, or unrelated untracked file changed
astr-opensbli-np2.service remains active
```

- [x] **Step 5: Commit the final synchronized maintenance set**

```bash
git add documents/maintenance/README.md documents/maintenance/generated/source-inventory.md
git commit -m "docs(maintenance): complete core solver maintenance set"
```

If final audit corrections touched another maintenance document, add that exact file explicitly. Do not use `git add -A`.

- [x] **Step 6: Report completion evidence**

Report the final commit range, inventory source count and fingerprint, unit-test count, Mermaid render count, unresolved lexical limitations, OpenSBLI service state, and unchanged unrelated untracked files. Do not claim numerical or physical solver validation from this documentation-only implementation.

## Plan Self-Review

- Spec coverage: all deliverables, ten diagrams, generated inventory, six change recipes, operational boundaries, numerical contracts, troubleshooting topics, and validation gates map to Tasks 1 through 10.
- Scope: the plan adds only maintenance documentation, its read-only generator, and focused standard-library tests.
- Type consistency: Task 2 consumes the exact `FortranFacts` interface defined in Task 1; later tasks consume the exact source-anchor and `--write`/`--check` contracts from Task 2.
- Placeholder scan: implementation steps name concrete files, interfaces, commands, expected results, diagram contents, and stop conditions.
- Artifact policy: only Mermaid source and generated Markdown are tracked; rendered images are temporary and deleted.
- Safety: the plan explicitly excludes solver/build behavior changes, broad staging, production runs, and OpenSBLI service interruption.
