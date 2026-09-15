# ASTR Repository Instructions

These repository-specific rules override broader global guidance while working
under `/home/dell/workspace/astr_gpu`.

## Scope And Autonomy

- Treat inspection, review, discussion, and explicit no-edit requests as
  read-only.
- Do not use subagents or create another worktree unless the user explicitly
  requests them. Work in the current checkout and honor the user's stated
  branch.
- For reversible local implementation work already authorized by the user,
  diagnose defects, make the smallest scoped fix, and run affected checks
  without requesting confirmation at every step.
- Ask one concise question only when a decision changes the physical model,
  numerical method, validation meaning, destructive Git state, or an external
  production job.
- Do not modify, submit, cancel, or reuse remote jobs unless explicitly asked.
  Any Zhongke Suanlian Cloud work must remain below
  `/data/user/hd56000/weiph`.

## Scientific And Numerical Boundaries

- If a physical definition, normalization, boundary condition, or experimental
  premise is invalid and would invalidate downstream conclusions, stop the
  downstream run and report the contradiction directly.
- If an existing CPU implementation appears to contain a numerical or boundary
  bug, present the evidence for human decision before porting or repairing it.
- Keep CPU and GPU implementations separated where practical. Prefer GPU code
  in `src_gpu/` and limit changes in `src/` to shared interfaces or approved CPU
  corrections.
- Preserve FP64 as the authoritative default. Keep explicit post-kernel
  synchronization unless a performance task explicitly authorizes a narrower
  dependency-based mode and correctness has been revalidated.
- Build through `/home/dell/workspace/astr_gpu/CMakeLists.txt`. Runtime `usegpu`
  remains an input-file choice, not a compile-time case selection.
- Do not add `chem/` or unrelated manuals, presentations, rendered artifacts,
  and local profiling output to Git.

## Verification

- Match verification effort to the changed surface. Build, numerical
  equivalence, physical validation, memory safety, and performance are separate
  gates and must not be presented as interchangeable evidence.
- Run the smallest sufficient affected test set first. Reuse immutable expensive
  benchmark artifacts when executable code, inputs, and environment are
  unchanged; do not rerun long GPU or remote production cases for documentation
  changes alone.
- Stop on the first failed scientific or numerical acceptance gate, identify the
  cause, and do not continue producing downstream plots or performance claims.

## Skills And Documentation

- Use `cuda-gpu-toolchain` for CUDA Fortran debugging, Compute Sanitizer, Nsight
  Systems, Nsight Compute, occupancy, memory, and MPI-GPU overlap work. Do not
  load the Python-oriented `optimize-for-gpu` skill for ASTR CUDA Fortran.
- Use one debugging workflow at a time. Prefer `diagnosing-bugs` for hard,
  repeated, or performance failures; do not also load `systematic-debugging`.
- Use brainstorming, grilling, detailed planning, TDD, branch-finishing, and
  reviewer/subagent workflows only when the user explicitly requests that
  workflow or the task cannot proceed without a design decision.
- Global scientific plotting rules apply only when creating or revising a
  scientific figure. Global LaTeX and sentence-revision rules apply only to
  formal manuscripts or explicitly requested research reports, not to source
  code, tests, commit messages, Markdown plans, or brief status summaries.
- For internal technical reports, provenance may be recorded in the document.
  For formal manuscript prose, keep code paths and implementation records in
  LaTeX comments or a separate evidence map.

## Git Boundary

- Preserve unrelated user changes and untracked files.
- Stage, commit, or push only when the user explicitly requests it or an
  approved written plan explicitly includes that Git operation.
