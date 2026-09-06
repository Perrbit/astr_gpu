# ASTR GPU Optimization Candidate Protocol

## 1. Purpose

This protocol controls performance changes to the CUDA Fortran path after the
numerical implementation has passed its case-level correctness gates. It
adopts the useful optimization-loop concept from CUDA-Agent without importing
its PyTorch/CUDA C++ implementation, build system, numerical tolerances, or
source code.

An optimization candidate is accepted only when it preserves the frozen ASTR
numerical contract and passes correctness, residency, safety, and complete-RK
performance gates. A faster isolated kernel is not sufficient evidence.

## 2. Frozen Numerical Contract

Every candidate must preserve all of the following unless a separate numerical
method change has been approved:

- FP64 solution variables and arithmetic;
- explicit sixth-order central convection and diffusion for the TGV gate;
- explicit tenth-order central filtering with the current ping-pong semantics;
- the prescribed x/y/z block shapes `(512,1,1)`, `(32,16,1)`, and `(64,1,8)`;
- RK3 stage ordering, physical-boundary ordering, and MPI halo semantics;
- GPU-resident solution fields inside the time loop;
- explicit synchronization after every kernel;
- CPU/GPU field and statistic tolerances of `atol=rtol=1e-10` for the TGV
  acceptance case;
- the top-level `CMakeLists.txt` as the only supported build entry point.

The following are prohibited in a performance-only candidate:

- `--use_fast_math`, `-Mfprelaxed`, `-fast`, or equivalent relaxed arithmetic;
- FP32, TF32, mixed precision, tensor-core substitution, or tolerance changes;
- compact differences, compact filters, or changes to stencil coefficients;
- deleting required synchronization or fusing across RK, boundary, filter, or
  MPI communication phases;
- new full-field H2D or D2H transfers in the resident RK interval;
- using Nsight Compute replay duration as application performance evidence.

## 3. Candidate Record

Before implementation, record these fields:

| Field | Requirement |
|---|---|
| Candidate ID | Stable short identifier |
| Baseline reference | Git commit or immutable source snapshot |
| Baseline timings | Five-repeat timing TSV from the same machine and build contract |
| Target kernels | Exact kernel names or kernel group |
| Allowed paths | Smallest set of source files that may change |
| Hypothesis | Expected bottleneck and proposed mechanism |
| Expected evidence | NCU metrics expected to change |
| Numerical impact | Must state whether arithmetic order or storage order changes |
| Hardware | GPU, driver, compiler, MPI, and build type |

The runner writes these values, the current Git state, and every stage result
under one candidate output directory. Generated output directories must remain
outside Git tracking.

## 4. Optimization Order

Use measured evidence in this order:

1. Remove redundant global-memory reads, writes, and repeated arithmetic.
2. Reuse existing compile-time constants instead of repeated device division
   where the expression is mathematically identical.
3. Improve Fortran-order contiguous access and eliminate inactive threads.
4. Consider register lifetime reduction when NCU reports occupancy or local
   memory pressure.
5. Consider shared-memory tiling only when measured data reuse can repay the
   synchronization, indexing, and occupancy cost.
6. Consider limited loop unrolling or prefetch only after source-correlated NCU
   evidence identifies the instruction or dependency bottleneck.

Kernel fusion, CUDA Graphs, launch-shape changes, mixed precision, and removal
of explicit synchronization are outside this protocol. Each requires a
separate design and approval because it changes a project-level contract.

## 5. Required Gates

### 5.1 Preflight and build

- Candidate metadata is complete.
- Changed paths are within the declared allow-list.
- CPU and GPU binaries build from the repository top-level CMake file.
- GPU compile commands contain no prohibited relaxed-math option.

### 5.2 Numerical correctness

- NP=1 TGV statistics and full-field comparisons pass for ten steps.
- NP=2 `2x1x1` TGV statistics and full-field comparisons pass for ten steps.
- Every comparison uses `atol=rtol=1e-10`.
- A candidate affecting physical boundaries, shock capturing, curvilinear
  metrics, or another case must additionally rerun that subsystem's existing
  regression matrix. The TGV runner does not replace subsystem validation.

### 5.3 Complete-RK performance

- Grid is `256^3`, NP=1, filter and diffusion enabled, checkpoint disabled.
- Synchronization mode is `explicit`.
- One process warm-up is followed by at least five measured processes.
- The first in-process RK sample is discarded.
- Candidate run-to-run spread is no greater than 5%.
- Default acceptance requires at least 3% lower median complete-RK time than
  the frozen baseline and no more than 5% peak-memory growth.

The baseline and candidate must run on the same GPU, software stack, power
policy, and benchmark contract. Historical timings from another machine are
not an acceptance baseline.

### 5.4 Residency and safety

- Nsight Systems reports no forbidden H2D/D2H transfer at or above 64 KiB in
  the RK interval.
- Only the named TGV diagnostic partial reductions may use the existing large
  D2H exception.
- Compute Sanitizer memcheck reports zero errors.

### 5.5 Causal profile

The 12-kernel NCU matrix must support the optimization hypothesis. Report at
least duration, executed instructions, SM and DRAM throughput, registers,
achieved occupancy, dominant stalls, and relevant source lines. Regressions in
other common kernels must be reported rather than hidden by the target result.

## 6. Decision Rule

A candidate is retained only if all applicable gates pass. On failure:

1. Stop at the first failed gate.
2. Preserve logs, reports, candidate metadata, and the failure reason.
3. Do not relax tolerances or exclude failing ranks to obtain a pass.
4. Revert the candidate source separately after the evidence is recorded.

No CPU design defect discovered during a candidate may be silently mirrored
on the GPU. Report it for an explicit CPU/GPU behavior decision.

## 7. Unified Runner

The entry point is:

```bash
CANDIDATE_ID=conv-load-reuse-01 \
BASELINE_REF=<git-ref> \
BASELINE_TIMINGS=/absolute/path/to/baseline_timings.tsv \
TARGET_KERNELS='convection x/y/z' \
ALLOWED_PATHS='src_gpu/solver_gpu.cuf' \
HYPOTHESIS='Reuse primitive and metric loads without changing arithmetic order' \
GATE_SET=full \
OUT_DIR=/tmp/astr_candidate_conv_load_reuse_01 \
  tests/gpu_validation/run_gpu_optimization_candidate_gate.sh
```

Available gate sets:

| Gate set | Stages |
|---|---|
| `correctness` | preflight, build, NP=1 and NP=2 field/statistic comparisons |
| `performance` | correctness plus five-repeat benchmark, baseline comparison, and Nsight Systems residency |
| `full` | performance plus the 12-kernel NCU matrix and Compute Sanitizer |

Use `DRY_RUN=t` to inspect commands and metadata without building or running
ASTR. `performance` and `full` require `BASELINE_TIMINGS`. A final acceptance
for any kernel change requires `full` plus the affected subsystem regression
matrix.
