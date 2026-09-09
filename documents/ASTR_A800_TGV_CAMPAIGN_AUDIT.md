# ASTR A800 TGV Campaign Pre-Submission Audit

## Decision

**Status: approved and submitted as Slurm job `451398` on 2026-09-08. As
checked on 2026-09-09, the job remains `PENDING (Priority)` and has not
produced A800 runtime evidence.**

The campaign contains only Taylor-Green vortex cases. It first measures the
A800 CFL limit, then runs CPU/GPU timing, one-to-four GPU strong scaling, a
pageable-memory control, an Nsight Systems trace, and one `512^3` production
run to approximately `t=20`.

## Case inventory

| ID | Grid | Resources | Purpose | Failure behavior |
|---|---:|---|---|---|
| T0 | `512^3` | GPU NP=1 | measure `dt_CFL=1`, memory and finite-state gate | failure skips T1-T7 because their safe time step is unknown |
| T1a | `256^3` | CPU NP=1 | five independent end-to-end CPU baseline runs | record failure and continue |
| T1b | `256^3` | GPU NP=1 | five independent end-to-end GPU baseline runs | record failure and continue |
| T2 | `512^3` | GPU NP=1 | five short timing repeats | record failure and continue |
| T3 | `512^3` | GPU NP=2, `1x1x2` | z-slab strong scaling | record failure and continue |
| T4 | `512^3` | GPU NP=4, `1x1x4` | four-A800 production topology | record failure and continue |
| T5 | `512^3` | GPU NP=4, `1x1x4` | pageable halo transport control | record failure and continue |
| T6 | `512^3` | GPU NP=4, `1x1x4` | five-advance Nsight Systems trace | record failure and continue |
| T7 | `512^3` | GPU NP=4, `1x1x4` | segmented production integration to `t` near 20 | retry each segment once, then record failure |

T7 uses the measured T0 value and sets
`deltat=floor(0.5*dt_CFL=1, 1e-12)`. Its number of steps is
`ceil(20/deltat)`, so the checkpoint time is at or just above 20 and the
initial CFL target is 0.50. Every reported CFL must remain in `(0,1)`.

## Numerical contract

- `Re=1600`, `Ma=0.1`, periodic Cartesian TGV
- sixth-order explicit central derivatives
- tenth-order explicit central filter
- RK3, diffusion enabled, filtering enabled
- explicit synchronization retained
- default multi-GPU halo transport: `pinned-overlap`
- production checkpoints every 2,000 steps, with one retry per segment

The DLR `spectral_Re1600_512.gdiag` history is used as an external physical
reference for kinetic energy, dissipation and enstrophy. The comparison is not
a bitwise acceptance test because DLR uses an incompressible spectral solver,
whereas ASTR uses a weakly compressible finite-difference formulation with
explicit filtering.

## Remote evidence

All remote files and commands are confined to `/data/user/hd56000/weiph`.

| Item | Audited value |
|---|---|
| Source base | `9fad55b7cbefbcbf7e3b8e4732525f8951647b26` plus the listed uncommitted campaign files |
| CPU executable | `cc85493443e9738eed390f612d7cdcc8e914b107f081e39a53ed3ff9f51e65f6` |
| GPU executable | `8c674a46722d002b026994b321ed3a5e1f8328f9c0d5841e34bd7d48a0535df4` |
| DLR reference | `60bfeaee2dd32de1b9e7171c70cf3774b6053d82a785d526c055547a0757017a` |
| Toolchain | NVHPC 25.5, HPC-X MPI, HDF5 1.14.5 |
| Python | 3.11.7 in `/data/user/hd56000/weiph/venv_tgv_campaign_py311` |
| Plotting | Matplotlib 3.11.1, SciencePlots 2.2.2 |
| Profiler | Nsight Systems 2025.3.1 |
| Partition | `A800-N`, state UP, maximum time unlimited, preemption off |
| Slurm requeue | disabled (`JobRequeue=0`) |

Both CPU and GPU builds completed from the top-level `CMakeLists.txt`. Dynamic
library inspection found no unresolved libraries. Both executables resolve the
same HDF5 1.14.5 and HPC-X MPI installation. The GPU executable additionally
resolves the NVHPC CUDA Fortran and CUDA 12 runtime libraries. No relaxed-math
flag was found in the GPU `compile_commands.json`.

## Login-node checks

- Bash syntax checks passed for all campaign drivers.
- Campaign dry-run listed exactly T0 through T7 above.
- Twenty-one Python unit and integration tests passed.
- The DLR reference hash matched the pinned value.
- A CPU `16^3`, one-advance TGV preflight completed and wrote a readable HDF5
  checkpoint.
- The corresponding GPU preflight parsed `input.tgv`, created the forced
  `1x1x1` MPI topology, and then failed at the expected check: the login node
  has no CUDA device.

The CPU preflight also printed UCX and PMIx permission warnings associated with
the login-node environment, but it fell back to the serial path and exited
successfully. The actual campaign runs inside a Slurm GPU allocation and does
not use the login node for computation.

## Fault containment

The job script intentionally does not use global `set -e`. Every case writes a
row to `case_status.tsv`; a failed independent case does not prevent later
cases from running. T0 is the only dependency gate because proceeding without
an A800-measured safe time step would violate the CFL requirement. T7 preserves
the previous valid checkpoint in `bakup`, retries a failed segment once, and
never overwrites a pre-existing campaign result root.

ASTR writes statistics and checkpoints at a complete-RK state. Its loop then
performs one additional non-persisted RK advance at `maxstep`; a restart reads
the saved complete-RK state, so segmented execution does not duplicate the
persisted statistics sequence.

## Residual risks requiring acceptance

1. T0 must still prove that one A800 can hold the `512^3` case and provide the
   platform-specific `dt_CFL=1`; this cannot be tested on the login node.
2. Slurm automatic requeue is disabled. T7 can recover from a solver or segment
   failure within the allocation, but a node loss requires resubmitting the
   campaign or production driver from the retained checkpoint.
3. The first DLR comparison intentionally reports metrics without imposing an
   uncalibrated pass threshold. Finite values, checkpoint continuity, complete
   time coverage and `CFL<1` remain hard gates.
4. The seven-day allocation is conservative. T2-T4 will provide the first
   A800 timing evidence; the production wall time is not known before those
   measurements.

## Submission record

The approved campaign was submitted with:

```bash
sbatch /data/user/hd56000/weiph/astr_gpu_tgv_campaign/tests/gpu_validation/run_zhongke_a800_tgv_campaign.sbatch
```

Expected result root:

```text
/data/user/hd56000/weiph/results/a800_tgv_campaign_<job-id>
```

The earlier broad staging synchronization was stopped, its temporary checkout
was removed, and the current remote checkout was recreated from the pinned Git
bundle. A final `git status --short` shows only the campaign whitelist above;
unrelated local documents are not present in the audited checkout.
