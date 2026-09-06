# ASTR Phase P2 Shock And SBLI GPU Baseline Report

## 1. Scope

Phase P2-0 establishes a performance baseline for the shock-sensor-coupled
selective-Roe path before changing a numerical kernel. It uses two NP=1 cases:

- periodic three-dimensional Shu-Osher, `256x64x32`, for controlled shock-path
  profiling;
- three-dimensional SBLI, `256x192x32`, with an analytic oblique-shock initial
  field, diffusion, isothermal wall, y-upper `bctype=52` NSCBC, and x-max
  sponge.

Both performance runs use FP64, MP7, explicit synchronization after every
kernel, 20 configured steps, one discarded in-process warm-up advance, one
process warm-up, and five measured processes. Checkpoint and list frequencies
are greater than `maxstep`. Shock-mask dumps are separate diagnostic runs and
do not contribute timing samples.

## 2. Environment

- source reference before the uncommitted P2 changes: `0d684a84bb689654bfc5a53a6928140c3a8e6fdb`;
- GPU: NVIDIA RTX 4000 Ada Generation;
- driver: `595.84`;
- compiler: NVHPC `26.1-0`;
- Nsight Systems: `2025.6.1`;
- Nsight Compute: `2025.4.0.0`.

The local HPC-X `sharedfp=sm` component blocked inside HDF5 `MPI_File_open`
after `/tmp` pressure. P2 runners use an output-local `TMPDIR` and default to
`OMPI_MCA_sharedfp=individual`; this only affects initialization I/O and may be
overridden after another MPI/HDF5 stack is verified.

## 3. P2-0 Infrastructure

- `run_p2_shock_performance_benchmark.sh` prepares either case, executes one
  warm-up and five measured runs, records complete-RK timings, and samples GPU
  memory and utilization.
- `run_p2_shock_activity.sh` performs a separate one-step mask dump.
- `summarize_shock_activity.py` reports active nodes and the exact launched
  interface convention used by `shock_interface_active`: adjacent-node OR for
  interior interfaces plus the two endpoint interfaces.
- `run_p2_shock_performance_profile.sh` captures Nsight Systems or one selected
  Nsight Compute kernel without enabling field output.
- characteristic Shu-Osher capability is no longer coupled to
  `ASTR_SHOCK_SENSOR_DUMP`. With `lchardecomp=t`, sensor computation is already
  part of the numerical path; the dump is only an optional diagnostic.

## 4. Correctness And Safety Gates

The post-change top-level CUDA build succeeds. The Python validation suite
passes all 43 tests.

The `64x8x8` Shu-Osher one-step CPU/GPU comparison passes:

- raw sensor maximum absolute difference: `5.5511e-17`;
- shock-mask mismatches: `0`;
- maximum reported field difference: `1.3878e-17` for `q2`;
- statistics remain within the `1e-10` gate.

The `64x64x8` SBLI one-step same-phase CPU/GPU comparison, with analytic shock,
diffusion, NSCBC, and sponge, passes:

- maximum primitive-field difference: `1.9984e-15` for temperature;
- maximum conserved-field difference: `1.1102e-15` for `q1`;
- statistics remain within the configured `1e-10` absolute/relative gate.

Compute Sanitizer memcheck on the complete small SBLI path reports
`ERROR SUMMARY: 0 errors`.

## 5. Complete-RK Baselines

| case | cells | median RK | run spread | throughput | peak memory | peak utilization |
|---|---:|---:|---:|---:|---:|---:|
| Shu-Osher | 524,288 | `0.151583855 s` | `1.933%` | `3.458732e6 cell-RK/s` | 890 MiB | 100% |
| SBLI | 1,572,864 | `0.523264916 s` | `0.363%` | `3.005866e6 cell-RK/s` | 1,578 MiB | 100% |

Raw evidence:

- `tests/gpu_validation/out/p2_shuosher_performance_256x64x32/`
- `tests/gpu_validation/out/p2_sbli_performance_256x192x32/`

## 6. Shock Activity

| case | active nodes | active x interfaces | active y interfaces | active z interfaces |
|---|---:|---:|---:|---:|
| Shu-Osher | `9.3385%` | `10.4651%` | `9.3385%` | `9.3385%` |
| SBLI | `3.1713%` | `3.4743%` | `3.3014%` | `3.1713%` |

The active regions are sparse. This fact alone does not justify a compacted
interface list because scan and indirect-index costs have not been measured.

## 7. Nsight Systems Hotspots

On Shu-Osher, x/y/z characteristic interface-flux kernels account for about
`31.1%`, `31.7%`, and `31.3%` of total GPU kernel time, or `94.1%` combined.
Raw sensor and expanded mask account for about `0.6%` and `0.2%`.

On SBLI, x/y/z characteristic interface-flux kernels account for about
`27.0%`, `38.9%`, and `21.6%`, or `87.5%` combined. Raw sensor and expanded
mask account for about `0.5%` and `0.2%`.

Therefore sensor fusion is not the first optimization target. The y-direction
characteristic interface flux has the largest complete-RK weight.

Raw evidence:

- `tests/gpu_validation/out/p2_shuosher_nsys_256x64x32/`
- `tests/gpu_validation/out/p2_sbli_nsys_256x192x32/`

## 8. Nsight Compute Mechanism

| metric, y characteristic flux | Shu-Osher | SBLI |
|---|---:|---:|
| registers/thread | 128 | 128 |
| achieved occupancy | `31.98%` | `32.42%` |
| compute throughput | `83.65%` | `87.07%` |
| DRAM throughput | `17.83%` | `1.52%` |
| local-memory spilling requests | 20,869,398 | 68,911,722 |
| L1TEX sectors attributed to local memory | `87.16%` | `49.19%` |
| branch efficiency | `99.95%` | `99.94%` |
| short-scoreboard stall | `87.69 cycles/inst` | `72.94 cycles/inst` |

NCU replay duration is not compared with complete-RK Nsight Systems duration.
The useful mechanism evidence is high register allocation, extensive local
memory caused largely by spills, low eligible-warps-per-scheduler, and dominant
short-scoreboard stalls. Branch efficiency is already near 100%, so the main
problem is not severe warp divergence.

Raw evidence:

- `tests/gpu_validation/out/p2_shuosher_ncu_y_256x64x32/`
- `tests/gpu_validation/out/p2_sbli_ncu_y_256x192x32/`

## 9. P2-1A Candidate Result

P2-1A screened a y-direction split path:

1. one physical-space MP7 interface-flux kernel computes all five fluxes for
   every y interface and writes the existing `flux_characteristic_work_d`;
2. one characteristic-only override kernel returns immediately on inactive
   interfaces and overwrites only sensor-active fluxes;
3. the existing y RHS kernel remains unchanged;
4. every kernel retains explicit synchronization.

This candidate does not claim that branch divergence is expensive. Its
hypothesis is that separating the common physical path lets the compiler avoid
the 128-register characteristic state and large local arrays for more than 90%
of interfaces. The override kernel still launches over the regular grid, so it
does not introduce a prefix scan or irregular compacted list.

The implementation passed the source contract, top-level CMake build,
Shu-Osher and SBLI one-step CPU/GPU field comparisons, NP=2 y-slab halo
comparison, and Compute Sanitizer with `0 errors`. Its numerical behavior was
therefore acceptable.

The performance hypothesis did not hold on RTX 4000 Ada. The physical-base
kernel still used 128 registers/thread with `32.54%` achieved occupancy and
about 63.96 million local-memory spilling requests. The characteristic
override added about 3.90 million spilling requests. The split did not create
a low-resource physical kernel and added a second full interface-grid pass.

| case | retained baseline | P2-1A split | change | spread | decision |
|---|---:|---:|---:|---:|---|
| Shu-Osher `256x64x32` | `0.151583855 s/RK` | `0.184307279 s/RK` | `+21.588%` | `0.928%` | reject |
| SBLI `256x192x32` | `0.523264916 s/RK` | `0.529303436 s/RK` | `+1.154%` | `0.176%` | reject |

The candidate source was removed after measurement. The retained production
path remains the single sensor-coupled characteristic interface-flux kernel.
The x/z split was not attempted because the y screen failed the complete-RK
gate.

Raw candidate evidence:

- `tests/gpu_validation/out/p2_p21a_correctness_shuosher/`
- `tests/gpu_validation/out/p2_p21a_correctness_sbli/`
- `tests/gpu_validation/out/p2_p21a_correctness_shuosher_np2_y/`
- `tests/gpu_validation/out/p2_p21a_memcheck_sbli.log`
- `tests/gpu_validation/out/p2_p21a_shuosher_nsys_256x64x32/`
- `tests/gpu_validation/out/p2_p21a_sbli_nsys_256x192x32/`
- `tests/gpu_validation/out/p2_p21a_sbli_ncu_y_base_256x192x32/`
- `tests/gpu_validation/out/p2_p21a_sbli_ncu_y_override_256x192x32/`
- `tests/gpu_validation/out/p2_p21a_shuosher_performance_256x64x32/`
- `tests/gpu_validation/out/p2_p21a_sbli_performance_256x192x32/`

## 10. P2-1B Physical Split Reuse Result

P2-1B keeps the single sensor-coupled interface kernel and changes only its
nonshock branch. At each of the seven reconstruction points, one device helper
now evaluates all five physical-space Steger-Warming split-flux components.
The previous path called the scalar helper once per component and repeated the
metric normalization, sound speed, eigenvalue split, and square roots five
times. The candidate uses the existing per-thread `split_plus(5,7)` and
`split_minus(5,7)` arrays, adds no global array or kernel launch, and retains
explicit synchronization after the enclosing kernel.

The physical helper preserves the existing `sqrt(tmp)/mach` sound-speed
definition and the distinction between `rho` and `q1`. Physical x/y sampling
in both the nonshock and Roe branches explicitly preserves the `npdc=3`
internal-rank range `-hm:dim+hm`; this prevents future decompositions with an
interior rank from clamping a seven-point stencil to local physical endpoints.

Correctness evidence includes NP=1 one-step Shu-Osher and SBLI comparisons,
NP=2 x/y/z halo comparisons, NP=8 `2x2x2`, ten-step Shu-Osher, and SBLI
Compute Sanitizer with `ERROR SUMMARY: 0 errors`. The ten-step SBLI HDF5 field
also passes: the largest primitive difference is about `1.02e-14` and the
largest reconstructed conserved-field difference is about `6.00e-15`.

The earlier ten-step GPU online `massflux` error of `9.05e-9` was traced to a
statistics-phase defect. CPU `boucon` applies the `bctype=52` upper-y x filter,
halo update, and z filter before `rkfirst` statistics. GPU statistics observed
the state before those filters, while the formal RK path applied them later.
The corrected GPU path snapshots the complete RK state in the existing
`qsave_d`, prepares the filtered NSCBC state for statistics, then restores the
snapshot and repeats the normal boundary preparation for time integration.
The ten-step field evolution is unchanged. Online `massflux` now has a maximum
absolute CPU/GPU difference of `5.0293103015519591e-13` and a final difference
of `2.8865798640254070e-14`, both within the existing `1e-10` gate.

Nsight Compute on the SBLI y-interface kernel shows that replay duration falls
from `103.30 ms` to `33.53 ms`. Local-memory spilling requests fall from
`68,911,722` to `15,059,748`, a reduction of about `78.1%`; registers remain
at 128/thread and achieved occupancy is `32.20%`. Nsight Systems confirms that
the improvement applies to all three interface directions.

| case | retained baseline | P2-1B | change | spread | peak memory | decision |
|---|---:|---:|---:|---:|---:|---|
| Shu-Osher `256x64x32` | `0.151583855 s/RK` | `0.109288716 s/RK` | `-27.902%` | `0.244%` | 820 MiB | retain |
| SBLI `256x192x32` | `0.523264916 s/RK` | `0.268521579 s/RK` | `-48.684%` | `0.431%` | 1,510 MiB | retain |

Raw candidate evidence:

- `tests/gpu_validation/out/p2_p21b_correctness_shuosher/`
- `tests/gpu_validation/out/p2_p21b_correctness_sbli/`
- `tests/gpu_validation/out/p2_p21b_correctness_shuosher_np2_x/`
- `tests/gpu_validation/out/p2_p21b_correctness_shuosher_np2_y/`
- `tests/gpu_validation/out/p2_p21b_correctness_shuosher_np2_z/`
- `tests/gpu_validation/out/p2_p21b_correctness_shuosher_np8_2x2x2/`
- `tests/gpu_validation/out/p2_p21b_correctness_shuosher_10step/`
- `tests/gpu_validation/out/p2_p21b_correctness_sbli_10step/`
- `tests/gpu_validation/out/p2_p21b_ab_baseline_sbli_10step/`
- `tests/gpu_validation/out/p2_p21b_ab_candidate_sbli_10step/`
- `tests/gpu_validation/out/p2_p21b_memcheck_sbli.log`
- `tests/gpu_validation/out/p2_p21b_final_memcheck_sbli/memcheck_clean.log`
- `tests/gpu_validation/out/p2_p21b_shuosher_nsys_256x64x32/`
- `tests/gpu_validation/out/p2_p21b_sbli_nsys_256x192x32/`
- `tests/gpu_validation/out/p2_p21b_sbli_ncu_y_256x192x32/`
- `tests/gpu_validation/out/p2_p21b_shuosher_performance_256x64x32/`
- `tests/gpu_validation/out/p2_p21b_sbli_performance_256x192x32/`

## 11. Corrected P2-1B Hotspot Matrix

The corrected statistics path adds no new hotspot. The SBLI snapshot save and
restore kernels together account for about `0.4%` of GPU kernel time, and the
NSCBC transverse-filter kernels remain below reportable precision.

| case | characteristic x/y/z | raw sensor | expanded mask |
|---|---:|---:|---:|
| Shu-Osher | `31.0% / 30.6% / 30.1%` (`91.7%`) | `0.9%` | `0.3%` |
| SBLI | `20.4% / 25.9% / 26.8%` (`73.1%`) | `1.0%` | `0.4%` |

The sensor and mask class has a complete-RK Amdahl upper bound below `3%` in
both cases, so no sensor-kernel candidate was opened. Full NCU captures show
all sampled characteristic kernels still use 128 registers/thread. SBLI x/y/z
local spilling requests are `19,472,739`, `15,059,748`, and `6,762,124`; the
Shu-Osher x kernel has `8,485,620`. Compute throughput is `78.32-87.23%`, while
achieved occupancy remains near `32%`. Characteristic flux therefore remained
the only P2 class with enough complete-RK weight for another candidate.

Raw evidence:

- `tests/gpu_validation/out/p2_full_p21b_shuosher_nsys_256x64x32/`
- `tests/gpu_validation/out/p2_full_p21b_sbli_nsys_256x192x32/`
- `tests/gpu_validation/out/p2_full_p21b_shuosher_ncu_x_256x64x32/`
- `tests/gpu_validation/out/p2_full_p21b_sbli_ncu_x_256x192x32/`
- `tests/gpu_validation/out/p2_full_p21b_sbli_ncu_y_256x192x32/`
- `tests/gpu_validation/out/p2_full_p21b_sbli_ncu_z_256x192x32/`

## 12. P2-1C Sequential Split-Array Screen

P2-1C replaced the per-thread positive and negative `5x7` split arrays with
one sequentially reused `5x7` array. Five positive characteristic scalars were
retained while the array was refilled for the negative side. The candidate
preserved the interface mask, Roe state, MP7/WENO selection, physical-boundary
clamping, FP64 arithmetic, fixed thread blocks, and explicit synchronization.

One-step Shu-Osher and SBLI CPU/GPU comparisons passed. The candidate did not
meet the complete-RK acceptance gate:

| case | restored P2-1B, 10 samples/run | P2-1C, 10 samples/run | improvement | spread | peak memory | decision |
|---|---:|---:|---:|---:|---:|---|
| Shu-Osher `256x64x32` | `0.110301406 s/RK` | `0.108122964 s/RK` | `1.975%` | `1.614%` | 788 MiB | reject |
| SBLI `256x192x32` | `0.267333482 s/RK` | `0.264103678 s/RK` | `1.208%` | `0.713%` | 1,508 MiB | reject |

The earlier frozen corrected P2-1B runs retained 20 RK samples per process and
were not used for this formal comparison because P2-1C retained 10. The table
uses the restored five-run P2-1B measurements with the same 10-sample contract.
Both generated performance-comparison reports fail only the required `3%`
time-reduction gate.

NCU on the SBLI y kernel reports 128 registers/thread in both variants. Local
spilling requests increase from `15,059,748` to `18,007,902`, while replay
duration decreases only from `33.53 ms` to `32.88 ms`. Reducing the declared
array count did not shorten the compiler-generated live state. The candidate
source and contract were removed from production; raw timing, NCU, and a
candidate source copy are retained under:

- `tests/gpu_validation/out/p2_full_p21c_split_reuse_shuosher_256x64x32/`
- `tests/gpu_validation/out/p2_full_p21c_split_reuse_sbli_256x192x32/`
- `tests/gpu_validation/out/p2_full_p21c_split_reuse_sbli_ncu_y_256x192x32/`

## 13. Multi-GPU Sensor-Halo Timeline

The P2 profile driver now supports explicit `NP`, topology, visible GPU list,
and `cuda,mpi` tracing. A tested SQLite analyzer measures the interval from the
end of each raw-sensor kernel to the start of its paired expanded-mask kernel
on each GPU, and counts only MPI events in the same process and interval.

| NP=2 y-slab case | halo critical path | complete RK in profile | upper-bound fraction | in-window MPI time, both ranks |
|---|---:|---:|---:|---:|
| Shu-Osher `256x64x32` | `7.706589 ms` | `220.945897 ms` | `3.488%` | `3.621127 ms` |
| SBLI `256x192x32` | `8.765948 ms` | `616.751308 ms` | `1.421%` | `2.542511 ms` |

The sensor-specific pack and unpack kernels themselves account for only about
`0.074%` of the Shu-Osher complete RK. A P2-local kernel change cannot reach
the required `3%`; the remaining interval is host staging and blocking MPI,
which belongs to P3 HaloTransport. No communication candidate was opened in
P2.

Raw evidence:

- `tests/gpu_validation/out/p2_full_p21b_shuosher_nsys_np2_y_256x64x32/`
- `tests/gpu_validation/out/p2_full_p21b_sbli_nsys_np2_y_256x192x32/`

## 14. Final P2 Verification And Decision

The restored P2-1B production path passes the top-level CPU/GPU builds, 54
Python tests, all validation shell syntax checks, ten-step Sod, Shu-Osher, and
SBLI CPU/GPU field and statistics comparisons, Shu-Osher NP=2 x/y/z and NP=8
`2x2x2` sensor/field/statistics comparisons, and SBLI Compute Sanitizer with
`ERROR SUMMARY: 0 errors`. Ten-step SBLI maximum errors are
`massflux=5.0293103015519591e-13`, primitive temperature
`1.0214051826551440e-14`, and `q1=5.9952043329758453e-15`.

A final five-repeat rebuild check gives `0.110301406 s/RK` for Shu-Osher
(`0.958%` spread, 810 MiB) and `0.267333482 s/RK` for SBLI (`0.770%` spread,
1,500 MiB). These differ from the frozen corrected baseline by `+0.927%` and
`-0.442%`, respectively, and remain within the measured run-to-run envelope.
Relative to P2-0, the final reductions are `27.234%` and `48.910%`.

P2 is closed. P2-1B is the only retained shock-path kernel optimization.
Sensor/mask has insufficient Amdahl weight, P2-1A and P2-1C are rejected, and
the remaining halo opportunity is explicitly transferred to P3.

## 15. A800 Interpretation

P1 candidates rejected on RTX 4000 Ada remain documented A800 retest
candidates. A800 has different FP64 throughput, bandwidth, cache, and register
residency behavior. Neither the P1 rejection percentages nor the P2 workstation
hotspot ordering can be transferred directly. Every A800 candidate requires a
fresh same-machine baseline, five repeats, Nsight Systems kernel weights, and
Nsight Compute mechanism metrics under the same correctness gates.

P2-1A is also an A800 retest candidate, not an A800 performance claim. Its
source is not retained in the workstation default, but the algorithm, metrics,
and raw evidence above are sufficient to reconstruct it after an A800-local
baseline is frozen. The same `3%` complete-RK acceptance gate remains in force.

P2-1B is retained because it passes the RTX 4000 Ada complete-RK gate with a
large margin. Its corrected frozen-baseline `27.902%` and `48.684%`
improvements are workstation results,
not A800 predictions. A800 deployment must establish a fresh unoptimized and
P2-1B pair with the same compiler, input, and five-repeat protocol before
reporting a platform-specific gain.
