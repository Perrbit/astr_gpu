# ASTR CURVE-C21 Aggregate Regression Report

## 1. Conclusion

CURVE-C21 passed on 2026-09-05. The same final CPU and GPU builds completed
all supported CURVE-C0-C20 matrices, required unsupported-geometry rejects,
representative x/y/z memory checks, positive-J and finite-field checks, and a
`256^3` no-checkpoint residency audit.

The run used Git HEAD `fa36a0d657d0c22ae6d456ec442013d2f723364f`
plus the recorded uncommitted C19/C20 solver changes. The tracked solver diff
SHA-256 is `fb480cb051acaaba402143f390bada4039cf607a6527c01239ca62562ca19939`.
The CPU/GPU executable SHA-256 values are
`b83c7e5726d9f6974a979933d7b6f3bde77c592aeb53482e6a4d98b31291c092`
and `a63128060c332729417b4f760147f4c2f7cccd6f5e7bee924945a6dde9557876`.
These hashes identify the tested binaries even though the source changes had
not yet been committed.

This result closes the tested static single-block, three-dimensional,
nonreacting, explicit-scheme curvilinear GPU scope. It is not evidence for
arbitrary curved characteristic boundaries, physical SBLI fidelity,
moving/multi-block grids, GPU HDF5, or scaling beyond two GPUs.

## 2. Reproducible Entry Point

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_c21_aggregate_20260905 \
  tests/gpu_validation/run_curvilinear_c21_aggregate.sh
```

The aggregate driver supports `START_STAGE` and `STOP_AFTER` for diagnosis,
but a release closure requires an uninterrupted full run. C11 is a retained
closed-negative policy gate: the known non-finite CPU global-filter shock
probe is checked as policy and is not rerun.

## 3. Numerical And Boundary Evidence

| Evidence | Result | Acceptance |
|---|---:|---:|
| Aggregate stages | `23/23` pass | all pass |
| CPU/GPU field reports | `136` | all pass |
| Maximum field $L_\infty$ | `q5=6.2527760746888816e-13` | `<=1e-10` |
| CPU/GPU statistic reports | `128` | all pass |
| Maximum statistic difference | `massflux=2.5093260802577788e-11` | `<=1e-10` for the owning C12 gate |
| Boundary-invariant reports | `226` | all pass |
| Maximum required boundary residual | `7.6170181273482740e-12` | `<=1e-10` for C17 extrapolation |
| Explicit finite-field checks | `70` | all true |
| Minimum numerical Jacobian | `2.5296638468231652e-7` | positive and finite |
| x/y/z Compute Sanitizer | `3/3`, zero errors | zero errors |

The field maximum occurs in the C16 y-wall NP=2 filter-plus-diffusion case.
The statistic maximum occurs in the C12 NP=1 long sponge case. The boundary
maximum is the C17 upper-y pressure extrapolation residual. Each value is
below the threshold of its owning test rather than a newly relaxed C21
threshold.

## 4. Compute-Loop Residency

The `256^3` C10 trace begins at the first selected characteristic x-RHS
kernel and contains 221 subsequent kernels.

| Transfer measure | Result |
|---|---:|
| H2D count/total/maximum | `2 / 352 B / 176 B` |
| D2H count/total/maximum | `2 / 96 B / 48 B` |
| H2D/D2H transfers `>=64 KiB` | `0` |

The trace therefore shows resident compute-loop fields for this case. It
does not include GPU HDF5 or claim that host-staged MPI halo traffic has been
eliminated.

## 5. Current-Machine Two-GPU Timing

The timing case is the `256^3` C10 path with six actual RK advances, three
timed repeats, initialization and final output included.

| Configuration | min/median/max wall time |
|---|---:|
| GPU NP=1 | `58.420/58.490/58.646 s` |
| GPU NP=2, x-slab | `38.638/38.989/39.182 s` |

NP=2 is `1.5002x` faster than GPU NP=1, corresponding to `75.01%` parallel
efficiency relative to ideal two-GPU scaling. This is a current-machine
repeatability result, not a CPU speedup or an NP=4/8 scaling claim.

## 6. Evidence Files

- Aggregate machine-readable stages:
  `tests/gpu_validation/out/curvilinear_c21_aggregate_20260905/c21_stage_summary.tsv`
- Solver status, tracked diff hash, and executable hashes:
  `tests/gpu_validation/out/curvilinear_c21_aggregate_20260905/solver_git_status.txt`,
  `solver_diff_sha256.txt`, and `binary_sha256.txt`
- Generated aggregate report:
  `tests/gpu_validation/out/curvilinear_c21_aggregate_20260905/c21_aggregate_report.md`
- Residency report:
  `tests/gpu_validation/out/curvilinear_c21_aggregate_20260905/c15_residency/residency_report.txt`
- Repeated timing report:
  `tests/gpu_validation/out/curvilinear_c21_aggregate_20260905/c15_benchmark/benchmark_summary.md`
- Per-stage logs:
  `tests/gpu_validation/out/curvilinear_c21_aggregate_20260905/logs/`

## 7. Next Target

The next physics milestone is a real laminar shock-wave/boundary-layer
interaction case. It must separate two claims: CPU/GPU numerical equivalence
and grid/time-converged physical fidelity. HaloTransport optimization and
real NP=4/8 one-rank-per-GPU scaling remain independent engineering tracks.
