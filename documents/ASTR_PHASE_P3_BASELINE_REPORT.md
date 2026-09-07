# Phase P3 Baseline Report

## Status On 2026-09-07

Design A is approved. Pageable blocking remains the default L0 implementation.
The local P3 acceptance gates are complete. Pinned blocking and periodic stored-
diffusion overlap are retained as explicit opt-in modes, not new defaults.
Post-endpoint-cleanup CFD, physical-filter and production
pack/unpack contracts pass, including host-only MPI memchecks. Standalone
nonblocking was rejected; CUDA-aware admission failed on the current stack.
Final-source timing results, noisy-group dispositions and the final admission
decision are appended below. This is local host-staged admission, not A800 or
CUDA-aware qualification, nor a resolved physical SBLI validation claim.
See `ASTR_PHASE_P3_IMPLEMENTATION_PLAN.md` for full scope and acceptance gates.

## Timing Admission

Existing `ASTR_GPU_RK_TIMING` logs only the I/O rank. P3 adds opt-in
`ASTR_GPU_RANK_RK_TIMING=1`, enabling per-rank records containing rank, step,
preparation duration, integration duration, and their sum. This adds no MPI
barrier or reduction. Old timing output remains compatible.

`tests/gpu_validation/summarize_rank_rk_timing.py` validates every rank/step,
rejects missing, duplicate, nonfinite or inconsistent records, and reports the
maximum local complete-RK duration per step. This is not a globally synchronized
wall-clock interval; also retain end-to-end wall time and separate NSYS traces.

The CUDA executable built through the top-level CMake build directory. All 56
Python tests passed. The real NP=2 x-slab admission run used:

```bash
env ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_SYNC_MODE=explicit \
  CUDA_VISIBLE_DEVICES=0,1 OMPI_MCA_sharedfp=individual MAXSTEP=1 \
  OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p3_rank_timing_np2_x \
  tests/gpu_validation/run_shuosher_characteristic_s0a7_mpi_compare.sh
python3 tests/gpu_validation/summarize_rank_rk_timing.py \
  --log tests/gpu_validation/out/p3_rank_timing_np2_x/gpu/gpu.log \
  --ranks 2 --steps 2 --discard 1
```

CPU/GPU sensor, field, and statistics reports all pass. Mask mismatches: zero.
Maximum conserved-field difference: 7.1054273576010019e-15. The parser found
both ranks at steps zero and one. This small-grid admission test does not
establish a performance baseline or an optimization gain.

## Evidence Scope

The sections below retain the chronological checkpoints, including superseded
pending items and unsuccessful groups. The final admission section is the
current decision. Host-only MPI sanitizer results do not qualify the default
UCX CUDA path. Clocks were observed, not locked; initial historical clock
samples cannot be reconstructed. No A800 or multi-node result is claimed.

## Frozen L0 Executable

The baseline is commit 566b5bc plus only the per-rank timing change. Its SHA256
is `6c6dbdf072f525d88e777f01288b772d2ffeb9e8d219cc7a707fce48a8e639b0`.
The executable, timing patch, and linked-library list are retained in
`tests/gpu_validation/out/p3_l0_frozen/`. Linked MPI comes from NVHPC 26.1
HPC-X 2.25.1, and HDF5 from `/opt/hdf5-1.14.6`. Both RTX 4000 Ada GPUs are on
NUMA node zero, linked through PCIe host bridges (nvidia-smi topology: NODE).

The benchmark driver now accepts NP, TOPOLOGY, and GPU_IDS, checks that their
counts agree, records the executable hash, and preserves per-GPU monitoring
with index/timestamp. Every repeat is validated for all rank/step timing
records. Existing timing output is not overwritten.

Reproduction pattern (substitute the case and topology):

```bash
env CASE=shuosher NP=2 TOPOLOGY=2,1,1 GPU_IDS=0,1 MAXSTEP=10 \
  GPU_EXE=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p3_l0_frozen/astr \
  LABEL=p3_l0_shuosher_x \
  OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p3_l0_shuosher_x_new \
  bash tests/gpu_validation/run_p2_shock_performance_benchmark.sh
```

Each run retains ten samples after discarding step zero; one complete process
warmup precedes five measured processes. Summaries and raw logs are under
`tests/gpu_validation/out/p3_l0_<case>_<axis>/`. Timing is maximum local rank
duration per step. Reported memory is the maximum sampled total device usage
on either GPU, including desktop activity, not an isolated allocation count.

The first Shu-Osher y-slab group has spread 31.456%, caused by one slower
process run. It is retained as inconclusive evidence and requires a complete
five-run remeasurement. Do not delete the slow run or selectively combine runs.

| Case | Slab | Median s/RK | Spread | Baseline admission |
|---|---|---:|---:|---|
| TGV 256^3 | x | 0.626163567 | 3.316% | pass |
| TGV 256^3 | y | 0.597632141 | 1.818% | pass |
| TGV 256^3 | z | 0.595934550 | 1.325% | pass |
| Shu-Osher 256x64x32 | x | 0.067698196 | 1.485% | pass |
| Shu-Osher 256x64x32 | y, first group | 0.070679454 | 31.456% | inconclusive |
| Shu-Osher 256x64x32 | y, repeat group | 0.070366241 | 2.052% | pass |
| Shu-Osher 256x64x32 | z | 0.083179662 | 2.011% | pass |
| SBLI 256x192x32 | x | 0.219455813 | 1.336% | pass |
| SBLI 256x192x32 | y | 0.199765283 | 0.960% | pass |
| SBLI 256x192x32 | z | 0.551301089 | 2.157% | pass |

These are baseline times, not optimization gains. SBLI z costs substantially
more than x/y, but attribution to transport requires the pending NSYS trace.

## L0 Interface Extraction

`halo_transport_gpu.cuf::exchange_host_pair` now owns the two MPI_Sendrecv
operations previously repeated at nine call sites. Pack/unpack and local
boundary logic remain in halo_exchange_gpu.cuf. This implementation checks
message buffer sizes and MPI return codes, and preserves MPI_PROC_NULL buffers.
It has no pinned or nonblocking implementation yet.

The top-level CMake target `halo_transport_test` checks exact element values
for periodic and nonperiodic neighbors, widths 5/6 and component counts 1/3/6.
It passes at NP=1, NP=2 and NP=3. The NP=3 test exercises an interior process
in a one-dimensional topology; it is not a three-dimensional CFD halo proof.
Shu-Osher NP=2 x/y/z field, statistics and sensor comparisons also pass after
extraction. TGV NP=2 x/y/z one-step filtered viscous field comparisons pass
with maximum q5 error 2.8421709430404007e-13. SBLI 64x64x16 NP=2 x/y/z one-step
field and wall-statistics comparisons also pass. Their output directories are
`tests/gpu_validation/out/p3_transport_l0_<case>_<axis>/`.
Full P3 regression, sanitizer and performance equivalence remain open gates.

## Pinned Prerequisite Tests

At the prerequisite-test checkpoint the solver remained pageable. The standalone `halo_pinned_probe` registers four
independent host allocations of 1, 37, 4096 and 262144 FP64 elements, checks
registration, performs both copy directions, unregisters and releases them.
All sizes pass; memcheck reports zero errors in
`tests/gpu_validation/out/p3_pinned_probe_memcheck.log`.

`halo_transport_test pinned` now exercises the same exact-value MPI contracts
using registered buffers, with explicit unregister before deallocation. Both
pageable and pinned modes pass NP=1/2/3, including duplicate periodic peers
and a nonperiodic interior rank. Logs are
`out/p3_transport_<pageable|pinned>_np<N>.log`. NP=3 shares two devices and is
correctness evidence only. These probes do not establish solver integration,
registration-failure fallback, asynchronous lifetime safety or a speedup.

The default HPC-X/UCX memcheck run fails with 31 CUDA API context errors per
rank in both pageable and pinned modes. The first stack is in
`MPI_Init -> uct_cuda_copy_set_ctx_sync_memops -> cuCtxSetFlags`, before the
test registers any host buffers. Raw logs are retained as
`out/p3_<pageable|pinned>_mpi_memcheck_<pid>.log`; neither run passes sanitizer.
The controlled host-only MPI configuration (ob1, pt2pt, self/tcp, CUDA-aware
support disabled, hcoll/ucc disabled) passes the pinned NP=2 test with zero
errors on both ranks, recorded in `out/p3_pinned_hostmpi_memcheck_<pid>.log`.
This isolates a UCX CUDA-path issue, not its underlying driver/library cause.
The diagnostic configuration does not replace the formal performance environment
and cannot validate CUDA-aware device communication.

## Initial SBLI Z Timeline

Frozen L0 was profiled separately with NP=2, topology 1,1,2, grid 256x192x32
and MAXSTEP=1 using the existing profile driver. Artifacts are in
`out/p3_l0_sbli_z_nsys/`, including `p2_shock_nsys.nsys-rep` and SQLite export.
The full trace has 192 private-tag GPU halo MPI events (21005/21006), totaling
555.268083 ms across both processes. This sum is not elapsed wall time or an
MPI percentage of complete RK. The overall MPI table also includes startup,
CPU initialization and file I/O, so its percentages must not guide P3 admission.

Using `summarize_p2_sensor_halo_nsys.py::sensor_intervals` to select only the
interval from raw-sensor completion to expanded-sensor launch gives six windows
per device. Their median durations are 3.685346 and 3.814912 ms, with totals
26.218706 and 26.975074 ms. Each process transfers 23808480 D2H bytes in those
windows, matching twelve messages of 1984040 bytes, where
1984040 = 5 * 257 * 193 * 8. H2D includes small parameter uploads in addition
to the field data. These windows include staging, MPI, synchronization and
launch gaps; they are not pure network time.

The three characteristic-flux kernels account for 70.4% of aggregate kernel
time in this trace. That is a kernel-only fraction, not a complete-step
fraction. The runtime log contains two io-rank complete-RK records of
0.582327339 and 0.537244006 seconds (including step zero). The generated
`sensor_halo_timeline.md` compares the summed per-window maximum with that
io-rank total. This is an exploratory estimate, not the formal slowest-rank
steady-state metric. A longer profile with per-rank timing remains required;
no optimization gain is claimed from the short trace alone.

## Pinned Blocking Candidate Integration

The experimental solver now accepts `ASTR_GPU_HALO_TRANSPORT=pageable|pinned`.
Unset means pageable. Selection is checked collectively; invalid or inconsistent
rank settings abort before solver halo traffic. Pinned setup allocates the three
transport families only on distributed axes, registers their full host buffers
once, and admits or rolls back registration collectively before `init_gpu`.
Only active component prefixes are transmitted as before. There is no change to
MPI_Sendrecv ordering, tags, widths, averaging, kernels or synchronizations.

The initial candidate preallocates all three families for each distributed axis,
including a filter family when the case does not subsequently use filtering.
This is intentionally recorded as candidate allocation overhead, not hidden
from the memory gate. Pinned memory is host memory; it is reported separately
from device memory. Normal shutdown unregisters all buffers before `mpistop`.
The sole new CPU-program hook is CUDA-guarded cleanup in `src/astr.F90`.
Allocation failure itself remains fatal, as in L0; collective fallback covers
registration admission, not arbitrary memory-allocation failure or MPI failure.

The actual setup/release module is tested by `halo_transport_setup_test` in
normal and overflow modes. Overflow fills the 36-entry registry on rank zero
only, requiring all ranks to fall back. All allocations can subsequently be
registered again, proving cleanup of previous registrations. Both modes pass
NP=2 host-only memcheck with zero errors on each rank. Evidence is
`out/p3_setup_<normal|overflow>.log` and corresponding memcheck logs. This tests
partial-registration rollback without requiring a driver allocation failure.

The first solver check, SBLI 64x64x16 NP=2 z-slab MAXSTEP=1, passes field and
wall-statistics comparison at unchanged 1e-10 combined tolerance. Maximum
primitive-field difference is 1.3322676295501878e-15; maximum q5 difference is
4.4408920985006262e-16. Both ranks log pinned=T with 11492000 registered bytes.
Evidence: `out/p3_pinned_sbli_z_smoke/`. This is not the full P3 correctness gate.

Candidate executable snapshot: `out/p3_pinned_frozen/astr`, SHA256
`5568de3069d775a1b3aa6811ffda3f9ca35e30b10879d6e6559bcbe7fd2d1877`.
Changed solver sources and benchmark helpers are archived beside it in
`source_snapshot.tar.gz`; unchanged sources are based on 566b5bc.
The initial five-repeat SBLI z result is 0.415962317 s/RK, spread 0.791%,
1462 MiB sampled device peak and 134914720 host-registered bytes per rank.
Against frozen L0 0.551301089 s/RK this is a 24.549% time reduction. This
candidate is not admitted yet: the remaining directions, contemporaneous L0
cross-check, full correctness and solver sanitizer gates remain required.

### Nine-Group Initial Screening

Each candidate group uses the same ten retained RK samples and five-repeat
contract as L0. All 54 initial process runs (nine groups including warmups)
log pinned=T and positive registered bytes on both ranks. No fallback is
silently included as pinned timing.

| Case | Slab | Frozen L0 s/RK | Pinned s/RK | Time reduction % | Pinned spread % | Device peak growth % |
|---|---|---:|---:|---:|---:|---:|
| TGV | x | 0.626163567 | 0.532401600 | 14.974 | 1.843 | 0.000 |
| TGV | y, repeat | 0.597632141 | 0.515103768 | 13.809 | 0.709 | 0.018 |
| TGV | z | 0.595934550 | 0.512718097 | 13.964 | 1.038 | -0.018 |
| Shu-Osher | x | 0.067698196 | 0.066903056 | 1.175 | 2.539 | 0.000 |
| Shu-Osher | y | 0.070366241 | 0.066530663 | 5.451 | 1.165 | 0.772 |
| Shu-Osher | z | 0.083179662 | 0.076229673 | 8.355 | 2.344 | 1.468 |
| SBLI | x | 0.219455813 | 0.199566019 | 9.063 | 0.413 | 0.319 |
| SBLI | y | 0.199765283 | 0.176882256 | 11.455 | 0.995 | 0.475 |
| SBLI | z | 0.551301089 | 0.415962317 | 24.549 | 0.791 | 2.813 |

Files follow `out/p3_pinned_<case>_<axis>/`. The first TGV y group
(0.513466830 s/RK, spread 8.103%) is inconclusive and retained unchanged. Its
complete five-repeat replacement is `out/p3_pinned_tgv_y_repeat/`; no samples
were mixed between groups. L0 Shu-Osher y likewise uses its complete repeat
group. These screening numbers satisfy the timing/spread/device-memory bounds,
but final admission still requires the full correctness and fallback matrices.
The larger SBLI z gain is a 24.549% reduction in time, or about 1.325x speedup;
it must not be described as a 24.549% increase in throughput.

Host-registered bytes per rank are 179653280 for each TGV slab; Shu-Osher x/y/z
use 5834400 / 23068320 / 45437600; SBLI x/y/z use
17323680 / 23068320 / 134914720. These do not represent extra full-field arrays.

### Same-Binary Timeline Cross-Check

`out/p3_<pageable|pinned>_sbli_z_nsys_ab/` profiles the frozen candidate binary
in both modes, NP=2 z-slab, MAXSTEP=3, per-rank RK timing enabled. Both traces
contain 384 private-tag halo MPI events. Their summed MPI call durations are
1091.029 and 1060.327 ms across processes. Across the full trace, cudaMemcpy
API duration decreases from 2246.884 to 1243.672 ms. These are aggregate event
durations, include initialization, and are not wall-clock percentages.

After discarding the first RK, maximum-rank local complete-RK records are
0.538521 / 0.541415 / 0.537736 seconds for pageable and
0.417229 / 0.415765 / 0.415905 seconds for pinned. The nine remaining sensor
windows per device have median durations 3.604638 / 4.039232 ms for pageable
and 2.351348 / 2.330258 ms for pinned. Both exchange exactly 36 sensor-sized
copies per direction across the two ranks in these windows. H2D GPU copy time
drops from 15.060314 to 9.739981 ms, whereas D2H increases from 8.904839 to
9.941647 ms in this sample. Thus not every transfer direction improves.
The reduced staging cost is supported by the trace; a general MPI-throughput
improvement or overlap is not claimed.

TGV 128^3 NP=2 x/y/z MAXSTEP=10 field and kinetic-energy/enstrophy/dissipation
statistics comparisons pass with filter and diffusion enabled.
Shu-Osher NP=2 x/y/z MAXSTEP=10 sensor, field and statistics comparisons pass,
with zero sensor-mask mismatches. SBLI 64x64x16 NP=2 x/y/z MAXSTEP=10 field
and wall-statistics comparisons pass. Evidence directories are
`out/p3_pinned_<case>_<axis>_10step/`. CPU/GPU top-level builds, all 56 Python
tests and individual shell syntax checks pass at this checkpoint. Full NP=1,
NP=8, fallback performance and solver sanitizer gates remain pending.

An additional solver memcheck passes for pinned TGV 32^3, NP=2 z-slab,
MAXSTEP=1, explicit filter and diffusion enabled. Both ranks report zero errors
under the documented host-only MPI configuration. The case, run log and separate
rank memcheck logs are in `out/p3_pinned_tgv_np2_memcheck/`. This covers actual
solution/filter/diffusion communication plus shutdown release, not the full
shock/wall sanitizer matrix or default-UCX CUDA API diagnostics.

## NP=1/8 Regression And Nonblocking Entry

Both pageable and pinned blocking pass NP=1 and NP=8 topology 2,2,2 with
MAXSTEP=10 for all three cases: TGV 128^3 (filter/diffusion enabled),
Shu-Osher 400x8x8 at NP=1 and 400x16x16 at NP=8, and SBLI 64x64x16.
TGV field and energy/enstrophy/dissipation statistics, Shu-Osher field/statistics
and sensor masks, and SBLI field/wall statistics all pass unchanged tolerances.
The twelve output directories are `out/p3_<pageable|pinned>_<case>_np<1|8>_10step/`.
These checks use the pinned-blocking checkpoint executable, before adding the
nonblocking branch. Every rank's transport-selection record was checked.
NP=1 correctly registers zero halo bytes; NP=8 pinned registers 34476000,
10061280 and 4577760 bytes per rank for TGV, Shu-Osher and SBLI respectively.
NP=8 shares two physical GPUs and establishes no scaling or performance claim.

The archived experimental value `ASTR_GPU_HALO_TRANSPORT=pinned-nonblocking` posts
both receives, posts both sends and waits for all four requests before return.
Dummy buffers are ASYNCHRONOUS for the lifetime of this routine. The routine
retains ownership until MPI_Waitall; no field exchange overlaps another field
exchange or a kernel. MPI errors abort, while pre-communication registration
failure collectively selects pageable blocking (both pinned and nonblocking
flags become false). This path is not compute/communication overlap.

The exact-value test was extended to use runtime transport selection. Before
implementation, the new value correctly failed as unsupported. After adding
the branch, NP=1/2/3 pass periodic, nonperiodic, widths 5/6 and component-count
1/3/6 tests. NP=2 host-only MPI memcheck gives zero errors on both ranks;
the registry-overflow test also confirms fallback clears nonblocking on both
ranks. Evidence: `out/p3_nonblocking_exact_np<N>.log` and
`out/p3_nonblocking_exact_memcheck_<pid>.log`.

The first nonblocking solver test, SBLI 64x64x16 NP=2 z-slab MAXSTEP=10,
passes field and wall-statistics comparison, in
`out/p3_nonblocking_sbli_z_10step/`. Its executable snapshot is
`out/p3_nonblocking_frozen/astr`, SHA256
`2edb5d7c634dbb08f13240c94b9e99ac52e3197d8dde0f179960f9e8919d2c53`.
The source snapshot is archived beside it. Its final standalone screening
decision is rejection, as detailed below. Default remains pageable.

## Paired Nonblocking Screening Decision: Reject Standalone Candidate

Screen this change against pinned blocking, not by crediting it with the
host-registration gain already achieved in the previous tier. Nine five-repeat
groups were measured with ten retained RK samples each, with both ranks verified
as pinned and nonblocking in all 54 initial process logs. The SBLI x first group
(spread 32.967%) is inconclusive and retained; a complete repeat replaces it.
SBLI y and TGV x also receive same-binary pinned controls to investigate the
apparent gain/regression without relying solely on older measurements.

| Case | Slab | Pinned control s/RK | Nonblocking s/RK | Incremental time reduction % | Nonblocking spread % |
|---|---|---:|---:|---:|---:|
| TGV | x, same-binary control | 0.534927102 | 0.539075880 | -0.776 | 1.159 |
| TGV | y | 0.515103768 | 0.512214354 | 0.561 | 0.964 |
| TGV | z | 0.512718097 | 0.512091690 | 0.122 | 0.388 |
| Shu-Osher | x | 0.066903056 | 0.066969488 | -0.099 | 2.436 |
| Shu-Osher | y | 0.066530663 | 0.066615810 | -0.128 | 2.232 |
| Shu-Osher | z | 0.076229673 | 0.075722955 | 0.665 | 1.150 |
| SBLI | x, repeated candidate | 0.199566019 | 0.195732256 | 1.921 | 0.730 |
| SBLI | y, same-binary control/repeat | 0.175711448 | 0.171450129 | 2.425 | 1.140 |
| SBLI | z | 0.415962317 | 0.415631076 | 0.080 | 1.007 |

The largest reproducible incremental gain is 2.425%, below the 3% independent
candidate gate. The earlier SBLI y result compared against an older pinned
group appeared above 3%, but the same-binary control does not reproduce that
threshold. The original TGV x comparison suggested a regression above 1%; the
same-binary control reduces it to 0.776%, so rejection is based on insufficient
incremental gain, not a proven regression beyond the limit. No individual run
was removed or mixed with another group. Raw results and device memory samples
remain under `out/p3_nonblocking_<case>_<axis>/`, with `_repeat` groups for
SBLI x/y and `out/p3_nonblocking_control_pinned_<case>_<axis>/` for the controls.

Same-binary SBLI y traces are in
`out/p3_nbcheck_<pinned|pinned-nonblocking>_sbli_y_nsys/`, MAXSTEP=3.
Pinned has 384 private-tag MPI_Sendrecv point-to-point events, totaling
61.186705 ms across ranks. Nonblocking has 192 non-null Irecv and 192 non-null
Isend events, plus null-peer calls classified separately. It has 192 unique
MPI_Waitall calls, totaling 71.083549 ms. The SQLite start/wait table contains
four request records per Waitall; deduplicate by start/end/thread/text before
summing durations. Zero same-process GPU kernels intersect these wait intervals.
Neither API names nor this timeline establish useful compute/communication
overlap. Aggregate trace durations are not complete-RK wall-clock fractions.

The nonblocking branch and runtime selection have been removed from active
source. Its executable and source snapshot remain in `out/p3_nonblocking_frozen/`
for later hardware-specific investigation. This rejection does not reject an
actual overlap design, which would change request lifetime and require its own
dependency and timing proof. Current supported settings remain pageable/pinned.

After removal, the four modified solver source files match the pinned checkpoint
archive byte for byte. Rebuilt executable SHA256 is
`7f93e6ec45bb978c5994cc9e1bf52fc0d0b6a68f93c54c0b2c11a6b3b875cb1a`;
do not equate a source match with an identical linked binary. Pageable/pinned
exact-value tests at NP=1/2/3, all 56 Python tests and shell syntax checks pass.
The rebuilt pinned solver also passes SBLI NP=2 z-slab MAXSTEP=10 field/wall
comparison in `out/p3_post_nb_pinned_sbli_z_10step/`. The removed runtime value
is explicitly rejected by the current executable rather than silently aliased.

## Interior Overlap Prerequisite: MPI Progress Probe

Status: prerequisite passed, solver overlap not implemented or accepted.
`halo_overlap_probe.cuf` isolates paired pinned-host MPI messages and independent
GPU work. It does not include solver halo packing, staging copies, boundary
unpacking or a complete RK step. Its derivative kernel uses `constdef:num1d60`,
FP64 and a 512-thread x block. Every launch retains explicit
`cudaDeviceSynchronize`. Payloads change by rank, direction and iteration;
every received element and the derivative of a linear field are checked.

Modes are blocking Sendrecv followed by compute (`blocking`), posted requests
followed by compute and Waitall (`wait`), and posted requests with MPI_Testall
progress during GPU execution followed by explicit synchronization (`poll`).
The probe uses two ranks on two GPUs, HPC-X MPI and `coll_hcoll_enable=0`.
It does not enable an MPI progress thread or change the solver MPI settings.

Each message contains 1981470 FP64 values, or 15851760 bytes. Work has 8388608
elements. Iteration zero is discarded. With three derivative launches per
iteration, three independent process runs each retain eight iterations:

| Probe mode | Run 1 median s | Run 2 median s | Run 3 median s |
|---|---:|---:|---:|
| blocking | 0.010279812 | 0.010177647 | 0.010127097 |
| wait | 0.010064250 | 0.010695594 | 0.010094602 |
| poll | 0.008697437 | 0.008595654 | 0.008627968 |

Raw logs: `out/p3_overlap_probe_<mode>_large_<1|2|3>.log`.
With thirty launches, a separate single process run per mode gives medians
0.024601221, 0.024386491 and 0.016977533 s respectively. In `poll`, both ranks
report requests completed before final Waitall in every retained iteration;
median final wait is 0.000001273 s, versus 0.008855154 s for `wait`.
These longer-work runs establish a progress opportunity, not a five-repeat
solver performance result. Logs: `out/p3_overlap_probe_<mode>_long.log`.

Separate thirty-launch traces with four retained iterations plus warmup are in
`out/p3_overlap_probe_nsys/<mode>.nsys-rep` and `.sqlite`. The wait trace has
10 distinct Waitall calls, 89.066877 ms summed across ranks, with no same-process
kernel intersections. The poll trace has 10 recorded distinct Testall calls
intersecting GPU kernels, totaling 0.091319 ms of API/kernel intersection.
Its final 10 Waitall calls total 0.018241 ms. Nsight stores four request rows for
each recorded Testall/Waitall here. These are not forty independent API calls,
and recorded Testall calls are not the total polling count reported by the
probe. The trace proves API/kernel concurrency, not the number of bytes moved
concurrently or a communication-engine utilization percentage.

`summarize_mpi_kernel_overlap.py` reproduces these counts. It deduplicates events,
matches process IDs and unions overlapping kernel intervals before intersection.
Five tests cover duplicate request rows, cross-process false matches, concurrent
kernels, optional MPI tables and missing CUDA trace rejection. All 61 Python
tests pass. NP=2 small probes in all three modes also pass Compute Sanitizer
memcheck with zero errors on both ranks using the documented host-only MPI
diagnostic configuration. This is not default-UCX sanitizer acceptance.

The solver TGV 256^3 NP=2 x-slab trace in `out/p3_overlap_bound_tgv_x/` contains
four RK advances. Per-rank summed stored diffusion RHS kernel time is about
208 ms across those advances, approximately 52 ms/RK. This is a gross work
budget, not a predicted saving: only the independent interior can overlap,
and splitting adds launches and index tests. Message size alone cannot separate
sigma communication from qswap: six components times hm and five components
times hm+1 both give 15851760 bytes for this case. Attribute exchanges using
their position between diffusion-flux production and diffusion RHS, not bytes
alone. The available work does not justify dismissing overlap as below 3%.

Next gate: a periodic stored-diffusion interior prototype with explicit MPI
progress and unchanged per-cell accumulation order. Keep the default solver
pageable and pinned paths unchanged until that prototype passes correctness,
actual solver concurrency and independent complete-RK performance gates.

## Periodic Diffusion Overlap Prototype

The experimental runtime value `pinned-overlap` now registers the same pinned
buffers and enables an optional paired-exchange work callback. Default pageable
and pinned calls still use Sendrecv. Registration failure collectively disables
overlap and falls back to pageable. The callback runs only for fully periodic
stored diffusion, distributed topology and nonempty radius-three core.
NP=1 and physical-boundary cases retain the unsplit blocking path. Runtime logs
distinguish requested overlap from actual periodic-diffusion activation.

Only the first distributed sigma axis receives the callback. It computes core
x/y/z contributions in their original order, actively progresses MPI between
launch and explicit synchronization, and returns before Waitall and halo upload.
Later sigma axes and all qflux axes retain their order. The usual x/y/z kernels
then compute the complementary cells. No new device arrays are allocated.
Nested paired exchanges abort rather than reusing outstanding buffers.

The frozen trial executable is `out/p3_overlap_frozen/astr`, SHA256
`4617908fd52a0678e65b9b435a457e78edde1b5bf06ba25a7b87785f83419532`.
Its source snapshot is stored beside it. Transport callback tests pass NP=1/2/3
for both pageable and pinned buffers, including widths 5/6, components 1/3/6,
self/duplicate peers and physical chains. Each rank verifies twelve callbacks.
Logs: `out/p3_overlap_exact_<pageable|pinned>_np<N>.log`.

TGV 128^3 NP=2 x-slab MAXSTEP=10 passes primitive and reconstructed-conservative
field comparisons, and energy/enstrophy/dissipation statistics, at unchanged
1e-10 tolerances. Maximum absolute q5 difference is 3.126388037344441e-13.
Evidence: `out/p3_overlap_tgv_x_10step_individual/`. TGV 32^3 NP=2 z-slab
MAXSTEP=1 with filter and diffusion enabled passes actual-solver memcheck with
zero errors on both ranks under the host-only diagnostic MPI configuration.
Both ranks report overlap active. Evidence: `out/p3_overlap_memcheck_individual/`.

The initial direct validation invocation omitted `OMPI_MCA_sharedfp=individual`,
which the existing P3 performance driver already supplies. CPU rank zero stopped
in `mca_sharedfp_sm_file_open` while creating `datin/grid.h5`; rank one waited
in the corresponding collective. The saved stack is
`out/p3_overlap_tgv_x_10step/cpu_grid_open_stack.txt`. Those runs were terminated
before time integration, not counted as numerical results. Repeating with the
established individual shared-file-pointer setting passes. Do not attribute this
initialization failure to the new overlap callback or change CFD equations.

An actual TGV 256^3 NP=2 x-slab MAXSTEP=3 solver trace is in
`out/p3_overlap_tgv_x_nsys/`. Both ranks report overlap active. It records 24
distinct Testall calls intersecting same-process GPU kernels, totaling 0.122117 ms
of API/kernel intersection. The 24 final Waitall calls total 0.036581 ms.
Most exchanges remain blocking: overlap applies only to first-axis sigma, not
all halo traffic. These totals include warmup and both ranks, are not wall-clock
fractions, and do not measure bytes moved during kernels. Other topology/fallback
comparisons and controlled complete-RK performance screening remain required.

The subsequent TGV 128^3 MAXSTEP=10 NP=2 y/z slabs, NP=1 and NP=8 2x2x2 all
pass field and energy/enstrophy/dissipation comparisons. Every NP=2/8 rank
reports overlap active, and NP=1 reports inactive as required. Evidence directories
are `out/p3_overlap_tgv_<y|z|np1|np8>_10step/`. NP=8 remains correctness-only.
Shu-Osher 400x8x8 NP=2 x-slab MAXSTEP=10 passes sensor-mask, statistics and field
checks with overlap inactive (diffusion disabled), in
`out/p3_overlap_fallback_shuosher_x_10step/`. SBLI 192x192x8 NP=2 z-slab
MAXSTEP=10 passes field and wall-statistics checks with overlap inactive, in
`out/p3_overlap_fallback_sbli_z_10step/`. The SBLI script takes IM/JM/KM rather
than GRID; report the actual generated dimensions, not an ignored GRID argument.

Normal setup and rank-zero registry overflow are checked with `pinned-overlap`
in `out/p3_overlap_setup_<normal|overflow>.log`. Overflow leaves both ranks at
pinned=false, overlap=false, registered_bytes=0 and permits re-registration.
Top-level CPU/GPU builds, 61 Python tests and individual shell syntax checks pass.
The first same-binary five-repeat complete-RK screen ran separately from
profiling and correctness jobs; no performance admission is claimed yet.

### First Independent Overlap Screen: TGV X-Slab

TGV 256^3, NP=2, topology 2x1x1, one rank per physical GPU, MAXSTEP=10.
Each mode has a separate process warmup plus five runs, each retaining ten RK
samples after discarding step zero. Both use the frozen overlap executable,
identical explicit synchronization, MPI-IO settings and inputs. Neither mode
ran concurrently with another CFD test or profiler.

| Transport | Median complete RK s | Five-run spread % | Peak device memory MiB |
|---|---:|---:|---:|
| pinned blocking control | 0.5379573425 | 1.704549 | 5611 |
| pinned overlap | 0.5259029470 | 1.619935 | 5611 |

The incremental reduction is 2.240772%, below the independent 3% threshold for
this benchmark. Both spreads satisfy 5% and sampled peak device memory is equal.
This single slab neither admits nor rejects the entire candidate: y/z screens
and the remaining no-regression matrix are still required. Do not credit the
overlap candidate with the earlier pageable-to-pinned gain.

All twelve warmup/measured process logs were inspected. Every rank selects
pinned with 179653280 registered bytes. Control logs disable overlap; every
candidate rank enables periodic diffusion overlap. `nvitop --once` observed
separate ASTR processes on both GPUs with approximately 70% device utilization
during the screen. The recorded monitor sample is
`out/p3_overlap_screen_nvitop.txt`. Raw timing, per-run utilization/memory, hash
and metadata are in `out/p3_overlap_screen_<pinned|pinned-overlap>_tgv_x/`.

The split introduces extra launches and whole-grid region tests. In the
overlap NSYS trace, each stored RHS direction has 48 launches across both ranks
and four RK advances, versus 24 in an unsplit run. Its summed x/y/z kernel time
is 548.124410 ms across both ranks. These costs must be weighed against the
hidden communication in controlled solver timings, not ignored because MPI
Waitall becomes short. Further splitting changes require independent screening.

### TGV Y-Slab Screen

The same frozen binary, process warmup and five-repeat protocol gives:

| Transport | Median complete RK s | Five-run spread % | Peak device memory MiB |
|---|---:|---:|---:|
| pinned blocking control | 0.5152436355 | 0.918656 | 5611 |
| pinned overlap | 0.4976075430 | 0.685412 | 5612 |

The y-slab reduction is 3.422865%, satisfying the gain threshold for this group.
The one-MiB difference is a device-total sampled peak, not a new solver array.
All twelve process logs select the intended pinned/overlap settings on both
ranks. Evidence: `out/p3_overlap_screen_<pinned|pinned-overlap>_tgv_y/`.
This result keeps the candidate under consideration; it does not replace the
remaining z-slab and Shu-Osher/SBLI no-regression screens or full fallback gates.

### TGV Z-Slab Screen

The z-slab screen completes the three TGV orientations using the same protocol:

| Transport | Median complete RK s | Five-run spread % | Peak device memory MiB |
|---|---:|---:|---:|
| pinned blocking control | 0.5127152045 | 0.536220 | 5612 |
| pinned overlap | 0.4959914740 | 0.749881 | 5612 |

Reduction is 3.261797%. All twelve process logs confirm the intended backend
on both ranks, with the same 179653280 registered host bytes per rank.
Evidence: `out/p3_overlap_screen_<pinned|pinned-overlap>_tgv_z/`.
Thus y and z pass the independent improvement threshold, while x improves
2.240772% without regression. The six Shu-Osher/SBLI control/candidate groups
and final retained-backend validation remain open; TGV results alone do not
admit the backend.

## Overlap Nine-Case Timing Screen

All nine formal case/slab pairs now have same-binary pinned controls and overlap
selections, each with five full process repeats after a separate warmup. All
108 initial process logs were checked for per-rank backend selection; periodic
diffusion is active only for TGV and is inactive for Shu-Osher and SBLI.
Raw groups use `out/p3_overlap_screen_<pinned|pinned-overlap>_<case>_<axis>/`.

| Case | Slab | Pinned s/RK | Overlap s/RK | Time reduction % | Control spread % | Candidate spread % |
|---|---|---:|---:|---:|---:|---:|
| TGV 256^3 | x | 0.537957343 | 0.525902947 | 2.241 | 1.705 | 1.620 |
| TGV 256^3 | y | 0.515243636 | 0.497607543 | 3.423 | 0.919 | 0.685 |
| TGV 256^3 | z | 0.512715205 | 0.495991474 | 3.262 | 0.536 | 0.750 |
| Shu-Osher 256x64x32 | x, initial | 0.066414812 | 0.067300726 | -1.334 | 2.772 | 2.520 |
| Shu-Osher 256x64x32 | y | 0.066927483 | 0.066299585 | 0.938 | 1.552 | 2.932 |
| Shu-Osher 256x64x32 | z | 0.076448163 | 0.076304647 | 0.188 | 2.518 | 2.664 |
| SBLI 256x192x32 | x | 0.199274557 | 0.199407292 | -0.067 | 0.917 | 0.519 |
| SBLI 256x192x32 | y | 0.176474010 | 0.176728883 | -0.144 | 0.715 | 1.187 |
| SBLI 256x192x32 | z | 0.420678607 | 0.418481130 | 0.522 | 0.599 | 0.784 |

The initial Shu-Osher x result exceeds the 1% regression threshold but is smaller
than either group's internal spread. Two additional complete pairs reverse the
backend execution order (candidate first, then control), without changing the
binary, inputs, retained steps or MPI configuration:

| Shu-Osher x group | Pinned s/RK | Overlap s/RK | Time reduction % | Control spread % | Candidate spread % |
|---|---:|---:|---:|---:|---:|
| reverse-order repeat | 0.066795915 | 0.067157304 | -0.541 | 1.925 | 26.927 |
| reverse-order repeat2 | 0.067258950 | 0.066545705 | 1.060 | 0.741 | 1.372 |

The first reverse-order candidate group is inconclusive: one process median is
0.084731673 s/RK and the five-run spread exceeds 5%. The entire group is retained,
not repaired by deleting that process. The next full pair does not reproduce
the original regression. Thus a stable regression greater than 1% has not been
demonstrated for this inactive path; the initial adverse result remains visible
above. This is not evidence that overlap accelerates an inactive path. The 24
additional process logs were checked. Evidence directories append `_repeat`
and `_repeat2` to the original Shu-Osher x group names.

Per-device peaks were also recomputed from all five measured runs, not just the
maximum across GPUs in the compact tables. GPU 0 peaks are unchanged in all
nine pairs. GPU 1 differs only by one MiB for TGV y (5611 to 5612 MiB, 0.017822%).
All other per-device peaks are identical. These samples include desktop usage.

Screening decision: keep the experimental candidate for final validation.
TGV y/z exceed 3%, accepted timing groups have spreads below 5%, no stable
greater-than-1% regression is reproduced, and device-memory growth is below 5%.
This is not final backend admission. Complete retained-backend fallback/field
and sanitizer matrices, L0 revalidation and CUDA-aware screening remain open.

### Y-Slab Timeline And Request Accounting

The improved y-slab case has its own solver trace in `out/p3_overlap_tgv_y_nsys/`,
NP=2, MAXSTEP=3, same frozen executable. Both ranks enable overlap. There are
48 recorded Testall API calls intersecting same-process kernels, but only 24
have request records in MPI_START_WAIT_EVENTS. The other calls are classified
in MPI_OTHER_EVENTS, including tests after requests have become null; they must
not be used as evidence of pending communication. The 24 request-record calls
all intersect kernels, totaling 0.123053 ms of API/kernel overlap. Final Waitall
has 24 calls totaling 0.022354 ms, with no request records in that table.

The analyzer now reports `start_wait_calls` and
`start_wait_calls_overlapping_kernel` separately from all API calls, and unions
kernel intervals before duration summation. Six analyzer tests, including an
OTHER-only false-positive case, pass; all 62 Python tests pass. Recorded API
intersections demonstrate request-associated MPI calls during useful GPU work,
not bytes transferred or communication-engine utilization. Use these traces
together with payload/field checks and independent unprofiled complete-RK timing.

## CUDA-Aware Admission Probe: Deferred On This Stack

The standalone `halo_cuda_aware_probe` is not a solver backend. Top-level CMake
adds this excluded diagnostic target only when the C MPIX query symbol links.
It checks device-buffer Sendrecv payloads exactly for widths 5/6, components
1/3/6, periodic peers and nonperiodic chains with MPI_PROC_NULL. The default
transverse shape is 3x2; the large test uses 256x256. Tags are 21001/21002.
Host reference payloads are distinct by rank and direction. Receives start at
-999 so unchanged buffers cannot pass. CUDA errors and mismatches stop the run.

Tested environment: NVHPC 26.1, HPC-X 2.25.1/Open MPI 4.1.9a1, two RTX 4000 Ada
GPUs, driver 595.84. The HPC-X wrapper loads UCX's `mt` library directory.
Both peer-access read/write topology queries report OK. These results do not
establish a hardware-wide lack of CUDA-aware support.

| Configuration | Exact payload result | Evidence under tests/gpu_validation/out/ |
|---|---|---|
| Default UCX, small NP=1/2/3 | Pass | p3_cuda_aware_exact_np*.log |
| Default UCX, large NP=2 | Fail, received sentinel unchanged | p3_cuda_aware_exact_large_diagnostic.log |
| UCX_TLS=self,sm,cuda_copy, large NP=2 | Pass | p3_cuda_aware_no_ipc_large.log |
| IPC enabled, UCX_CUDA_IPC_CACHE=n | Fail | p3_cuda_aware_ipc_nocache_large.log |
| UCX_PROTO_ENABLE=n | Fail | p3_cuda_aware_proto_v1_large_fixed_diagnostic.log |
| OB1/self,smcuda, btl_smcuda_use_cuda_ipc=0 | Pass | p3_cuda_aware_smcuda_no_ipc_large.log |
| Bind CUDA device before MPI_Init, default UCX | Fail | p3_cuda_aware_early_bind_large.log |

The first failing large payload has width 5 and one component: 327680 doubles,
2621440 bytes. The largest tested message is 18874368 bytes. Small-message
success therefore cannot admit the solver's communication sizes.

Sanitizer admission also fails. The non-IPC UCX configuration reports 49 errors
on each rank (`p3_cuda_aware_no_ipc_memcheck_881201.log` and `_881202.log`).
OB1/smcuda without IPC reports 36 errors on each rank
(`p3_cuda_aware_smcuda_no_ipc_memcheck_882596.log` and `_882597.log`), including
cuMemRetainAllocationHandle invalid-value reports in MPI's GPU-buffer query.
Early binding with that latter configuration reports 124/128 errors
(`p3_cuda_aware_early_no_ipc_memcheck_883205.log` and `_883206.log`). Some reports
may arise from library capability queries, but the required zero-error gate is
not met. They are not suppressed or reclassified as a clean sanitizer run.
Host-only MPI sanitizer passes elsewhere do not qualify device-buffer MPI.

The installed MPIX_Query_cuda_support implementation returns constant 1,
verified in `p3_cuda_query_disassembly.txt`, even when opal_cuda_support is
disabled. It is not a sufficient runtime admission test on this installation.
An isolated OB1 CUDA-disabled probe fails when handed device buffers; this is
not an ASTR fallback test. A future backend must require a qualified transport
configuration and representative payload tests, not just the extension query.

Two early diagnostic versions themselves had bad mismatch reporting (a zero
maxloc index and an unsupported FINDLOC runtime). Their failed logs are kept,
but the decision above uses the corrected explicit-index scan and repeat tests.
The corrected diagnostic executable and source archive are frozen in
`out/p3_cuda_aware_probe_frozen/`. Executable SHA-256:
`b27e043d38e6389c2c436f33d3997566fb4568521b79b206d7c527079fceb205`.

Disposition: stop CUDA-aware integration on the current stack. No solver mode,
automatic runtime fallback, or device-buffer performance result is delivered.
Keep the diagnostic and raw evidence for future platform qualification.
Pageable remains default; pinned and pinned-overlap still require final full
validation. This closes only the CUDA-aware screening tier, not the P3 goal.

## Final-Matrix Checkpoint Before Endpoint Upload Cleanup

Top-level CPU/GPU CMake builds pass after adding the excluded CUDA-aware probe.
All 62 Python unit tests, 96 individual shell syntax checks and git diff
whitespace checks pass. The GPU executable was relinked by this build, so its
hash differs from the screening executable. A tar comparison confirms identical
contents for halo_transport, halo_exchange, mainloop, solver, gpu_runtime and
src/astr.F90 against the overlap screening source archive.

The current executables and relevant source snapshot are frozen in
`tests/gpu_validation/out/p3_final_frozen/`:

- GPU `astr_gpu`: `c3d0248d52ec3f6db6c45a2a2f54258bf147707b0e76248e2db1b8bdd09a3390`.
- CPU `astr_cpu`: `5c223d1c417b9da543e98fdfe7e2de726a8ba8b1458062a990ee6278d4321bda`.

All 45 ten-step comparisons pass: pageable, pinned and pinned-overlap, each
covering TGV, Shu-Osher and SBLI with NP=1, NP=2 x/y/z and NP=8 2x2x2.
NP=8 remains two-GPU oversubscription correctness evidence, not scaling data.

| Case | Grid | Maximum reconstructed q(1:5) absolute difference across all 15 comparisons |
|---|---|---:|
| TGV, explicit filter and diffusion enabled | 128x128x128 | 3.410605131648481e-13 |
| Shu-Osher, MP7 characteristic reconstruction | 400x16x16 | 1.4210854715202004e-14 |
| SBLI boundary/viscous slice | 64x64x16 | 3.1086244689504383e-15 |

Primitive fields, reconstructed conservative fields and flowstate statistics
pass the unchanged 1e-10 absolute/relative combination. Shu-Osher and SBLI
sensor comparisons also pass the existing 1e-12 combination with no mask
mismatch. SBLI flowstate includes mass flux, friction and wall heat flux.
This short SBLI slice has no marked shock nodes; it is a boundary/viscous
regression, not resolved SBLI physical validation. Shu-Osher supplies nonzero
shock-mask coverage. Rankwise marked-node counts include duplicated interfaces
and must not be compared across topologies as unique physical cell counts.
The Shu-Osher x driver additionally sets ASTR_SHUOSHER_SHOCK_X=0.d0 to put the
shock on the partition interface; other drivers retain the default position.
Thus those topology slices are not identical initial-condition experiments.

Logs confirm pinned selection on every distributed rank and actual overlap
only for distributed periodic TGV diffusion. All Shu-Osher/SBLI paths remain
inactive for overlap. Evidence directories are
`out/p3_final_phase_<backend>_<case>_<np1|x|y|z|np8>_10step/`; structured results
are in `out/p3_final_phase_matrix_summary.json`.

SBLI uses IM=64 JM=64 KM=16, SAME_PHASE_FIELD=t and COMPARE_SENSOR=t with
`run_s2_sbli_physical_nscbc_compare.sh`. CPU fields come from the complete-RK
snapshot, not the next boundary-filter phase. An initial driver omitted the
same-phase option; its SBLI NP=1 run was terminated and excluded. Earlier
Shu-Osher passes under `p3_final_*` remain archived; the complete accepted
matrix uses the `p3_final_phase_*` prefix. No field tolerance was relaxed.

For TGV, CPU references are computed once per topology and reused by the two
pinned modes. Their `cpu` directories link to the matching pageable reference.
Each reuse verifies identical GPU input and controller contents and records
the CPU/GPU executable hashes, input hashes and topology in `reference.json`.
The original batch launcher was stopped between completed cases to avoid
recomputing identical CPU references; its completed NP=1/x runs were checked
again. No solver result was inferred from the launcher's termination status.

Nine actual-solver memchecks also pass, with two zero-error rank logs each:
three backends times TGV 32^3 NP=2 z, Shu-Osher 80x16x16 NP=2 x and SBLI
64x64x16 NP=2 y. MAXSTEP=1, checkpoint/list frequency 9999, same frozen GPU
executable. TGV includes explicit filtering, diffusion and active overlap for
the overlap mode. Case folders are `out/p3_final_memcheck_<backend>_<case>/`;
each stores exact command/environment JSON and rank logs. Aggregate:
`out/p3_final_memcheck_summary.json`.

These memchecks use host-only MPI: pml=ob1, osc=pt2pt, btl=self,tcp,
coll=^hcoll,ucc, opal_cuda_support=0, UCX_MEMTYPE_CACHE=n and
OMPI_MCA_sharedfp=individual. They qualify the ASTR host-staged solver paths
under that configuration, not default-UCX library API behavior, device-buffer
MPI or full large-grid memory coverage.

Remaining contract gap: inherited exchange wrappers upload receive buffers
even for MPI_PROC_NULL, although unpack kernels guard those neighbors. Add
guarded uploads and repeat affected endpoint checks before final admission.
The matrix above is a frozen pre-cleanup checkpoint, not completion of P3.

## Endpoint Upload Guards And Revalidation

All 18 receive-buffer H2D assignments in halo_exchange_gpu now check the
corresponding MPI neighbor against MPI_PROC_NULL. The guard also prevents
reading the absent neighbor's host receive buffer. MPI calls and ordering,
hm/hm+1 widths, pack/unpack arithmetic and explicit kernel synchronization
remain unchanged. The new source-contract test first reported 18 failures,
then passed after the guards. All 63 Python tests, 96 shell syntax checks and
top-level CPU/GPU builds pass. This source test is not a substitute for runtime
communication and numerical checks.

Post-cleanup GPU executable: `out/p3_endpoint_frozen/astr_gpu`, SHA-256
`aee22855dc7a27cd1884ff3c33da9da9f287f5ae6900ebb4c9ef2d4e9edb187a`.
Relevant sources and the new contract test are archived beside it.

All 45 main-matrix comparisons were rerun with this executable. Reference CPU
fields are reused from the preceding frozen checkpoint, with identical input
and controller hashes and explicit runtime environment records. SBLI still
uses complete-RK snapshots. Evidence: `out/p3_endpoint_v2_<backend>_<case>_<topology>_10step/`
and `out/p3_endpoint_v2_matrix_summary.json`. Field maxima are unchanged:
TGV 3.410605131648481e-13, Shu-Osher 1.4210854715202004e-14 and SBLI
3.1086244689504383e-15 for reconstructed q. Statistics and sensor masks pass.
Actual overlap activation and per-rank backend selection were checked again.

The first cached-reference runner omitted the x-slab Shu-Osher driver's
ASTR_SHUOSHER_SHOCK_X=0.d0 setting and stopped at a large field mismatch.
The reference placed the initial shock at the rank interface while that run
used the default position. Restoring the exact initial condition passes the
same tolerance with no additional solver edit. Failed evidence is retained in
`out/p3_endpoint_pageable_shuosher_x_10step/`; the accepted complete matrix has
the `p3_endpoint_v2_` prefix and stores environment values in reference.json.
Input-file hashes alone are insufficient provenance for these runtime controls.

Nine additional physical-endpoint tests use the existing wall41 phased driver:
three backends times x/y/z wall-normal NP=2 decomposition, GRID=32,32,32,
MAXSTEP=2, FEQCHKPT=2, LFILTER=t and DIFFTERM=t. Both field and statistics
tolerances are explicitly 1e-10 absolute/relative. All pass; maximum absolute
q difference is 2.842170943040401e-13. This exercises physical filter receive
paths that the short SBLI case, with global lfilter=f, does not cover.
Evidence: `out/p3_endpoint_wall41_<backend>_<axis>/` and sibling driver logs.

Post-cleanup memcheck covers the earlier TGV/Shu/SBLI small cases plus all three
wall41 directions for each backend: 18 runs, 36 rank logs, all zero errors.
The Shu-Osher x memcheck now also sets the interface shock position explicitly.
Wall cases retain MAXSTEP=2; other small cases use MAXSTEP=1. Exact commands,
environments and logs are in `out/p3_endpoint_memcheck_<backend>_<case>/` and
`out/p3_endpoint_memcheck_summary.json`. These remain host-only MPI diagnostic
results, not CUDA-aware or default-UCX sanitizer admission.

Twelve host transport exact-value checks (pageable/pinned, NP=1/2/3, blocking
and callback) pass in `out/p3_final_exact_*.log`. Four normal/collective-overflow
setup checks for pinned and pinned-overlap pass in `out/p3_final_setup_*.log`;
overflow clears both pinned and overlap flags and unregisters all buffers.
These tests do not independently exercise production GPU pack/unpack averaging
with deliberately conflicting interface values. That adversarial contract
check, plus reconciliation of performance acceptance with the final source,
remains open. Earlier screening timings must not be relabeled post-cleanup.

## Production Pack/Unpack Contract Closure

The excluded `halo_exchange_contract_test` CMake target compiles the production
modules in an isolated module directory, replacing only the ASTR main program
with a test fixture. No pack/unpack code is copied or changed. Each rank seeds
different exact FP64 values, including conflicting shared planes and all halo
regions. The rank offset is odd so averages can contain an exactly representable
half-integer; a copy in place of averaging cannot pass.

All three axes are tested in periodic and physical-chain modes. Thirty
production exchanges per run cover qswap hm+1 and interface averaging, fixed-hm
filter exchange with q/qwork ownership, and field component counts 1/3/6.
Whole-array equality also checks untouched transverse edge/corner halos,
MPI_PROC_NULL faces and unused components. NP=2 has duplicate periodic peers;
NP=3 includes a rank between the two physical endpoints.

All six runs (pageable/pinned/pinned-overlap times NP=2/3) pass exactly, without
tolerances. All distributed-rank setup logs confirm the selected pinned state,
with no silent registration fallback. Evidence: `out/p3_production_halo_*.log`.
NP=3 memcheck for each backend also passes, nine rank logs with zero errors,
in `out/p3_production_halo_memcheck_<backend>/`, using the same host-only MPI
diagnostic configuration documented above. This does not add overlap timing
evidence: callbacks are not needed in this pack/unpack fixture.

Executable and production/test source archive:
`out/p3_production_halo_frozen/halo_exchange_contract_test`, SHA-256
`91ce0b8e7496bb2d963d421a874600f2aa2bcf1425fde7ff15cd87878e1030af`.
Top-level CPU/GPU builds, all 63 Python tests and 96 shell syntax checks pass.
This closes the previously missing adversarial averaging/ownership check.

## Final-Source Performance: TGV X Checkpoint

These are new measurements of the frozen endpoint-cleanup executable
`aee22855dc7a27cd1884ff3c33da9da9f287f5ae6900ebb4c9ef2d4e9edb187a`, not relabeled
earlier screening data. TGV 256^3, NP=2 x slab, one GPU per rank, explicit sync,
MAXSTEP=10, five independent measured processes after process warmup, ten
retained complete-RK samples per process. No compilation or other test ran
during these timing groups. Backend flags were checked in all 18 process logs.

| Backend | Median s/RK | Five-process spread % | GPU 0 peak MiB | GPU 1 peak MiB |
|---|---:|---:|---:|---:|
| pageable | 0.627133364 | 4.152 | 5354 | 5610 |
| pinned | 0.540344406 | 1.401 | 5354 | 5611 |
| pinned-overlap | 0.525150649 | 8.020 | 5354 | 5611 |

Pinned reduces complete-RK time by 13.839% relative to pageable in this group.
The per-device memory increase is at most one MiB. Overlap's median reduction
relative to pinned is 2.812%, but its spread exceeds 5%; the entire group is
inconclusive. Its five process medians are 0.524838908, 0.5251506485,
0.5189430085, 0.5610596655 and 0.526145304 s/RK. No process is deleted from the
group to improve the spread. This x result does not admit or reject overlap
across the required full benchmark matrix.

Evidence: `out/p3_final_performance_<backend>_tgv_x/`, sibling driver logs and
`out/p3_final_performance_tgv_x_summary.json`. Per-GPU peaks were recomputed
from all five measured monitor CSVs, excluding the process-warmup group.
The batch dispatcher was stopped after all three complete x groups, before
compiling the isolated contract test; y/z and the Shu-Osher/SBLI final-source
groups were not launched. Next actions are a complete paired x repeat and
the remaining formal matrix, followed by the final evidence audit.

## Final-Source Shu-Osher And SBLI Matrix

All 18 groups use the same endpoint-cleanup executable and five-repeat protocol
as the final TGV x checkpoint: NP=2, one rank per physical GPU, MAXSTEP=10,
one discarded RK sample and ten retained samples per process. Shu-Osher uses
256x64x32; SBLI uses 256x192x32 and the benchmark driver's shock-initialized
field and sponge configuration. No solver source changed during these runs.

| Case / slab | Pageable s/RK | Pinned s/RK | Overlap mode s/RK | Spread %, pageable / pinned / overlap | Pinned time reduction % |
|---|---:|---:|---:|---|---:|
| Shu-Osher x | 0.0684385275 | 0.0665026415 | 0.0666462355 | 2.618 / 1.341 / 2.031 | 2.829 |
| Shu-Osher y | 0.0707247850 | 0.0664312480 | 0.0667785840 | 0.699 / 1.512 / 1.222 | 6.071 |
| Shu-Osher z, initial | 0.0833062895 | 0.0764160340 | 0.0766462540 | 0.911 / 26.623 / 2.416 | inconclusive |
| Shu-Osher z, reverse-order repeat | 0.0824009760 | 0.0761290395 | 0.0755729385 | 2.276 / 3.308 / 2.198 | 7.611 |
| SBLI x | 0.2118033920 | 0.1952345565 | 0.1949228140 | 1.004 / 0.565 / 0.624 | 7.823 |
| SBLI y | 0.1888616200 | 0.1697192440 | 0.1703798160 | 1.766 / 0.485 / 1.640 | 10.136 |
| SBLI z | 0.5486275390 | 0.4184511040 | 0.4192066590 | 3.157 / 0.339 / 1.097 | 23.728 |

The initial Shu-Osher z pinned group exceeds the spread gate and is kept in
full. All three backends were rerun as complete groups in the reverse order:
pinned-overlap, pinned, pageable. The repeat meets the spread gate without
deleting any process. All initial groups and repeat groups remain archived.

Per-device peaks were recomputed from every measured monitor CSV. The largest
increase relative to pageable is GPU 0 in SBLI z, 1164 to 1204 MiB (3.436%),
below 5%. Using only the larger of the two GPU peaks would understate this
per-device increase. These samples include desktop allocations. Pinned and
overlap-mode peaks are identical within each original pair.

All 126 initialization logs (108 original plus 18 repeat) confirm the requested
backend. Every overlap-mode run reports periodic_diffusion_active=F for these
two cases. Its small timing differences from pinned, within 1% in the valid
groups, are inactive-path observations, not communication-overlap gains.
These groups support pinned performance and fallback behavior but do not close
the independent overlap acceptance gate. Final TGV y/z and the noisy TGV x
paired repeat remain outstanding.

Evidence directories: `out/p3_final_performance_<backend>_<case>_<axis>/`;
Shu-Osher z repeats append `_repeat1`. Structured summaries are
`out/p3_final_performance_shu_sbli_summary.json` and
`out/p3_final_performance_shu_z_repeat1_summary.json`. The batch and all case
processes completed before this checkpoint was recorded.

## Environment And Overlap Evidence Audit

`out/p3_final_environment_20260907/environment.json` records the current
compiler, linked libraries, revision, GPU identities, topology and an
instantaneous clock/power snapshot. At capture the two GPUs reported P2,
SM clocks 2310/2280 MHz, memory clocks 8551 MHz and 130 W limits. These are
observations, not locked-clock settings or retroactive clock evidence for
earlier groups.

The existing y-slab NSYS SQLite schema was checked directly. MPI_START_WAIT_EVENTS
stores request handles but no MPI_Testall completion flag. Request-associated
API/kernel intersections therefore remain useful timeline observations, not
standalone proof that requests were incomplete during the kernel. MPI_Testall
reports completion through its flag; this audit does not equate a still-valid
handle with an unfinished transfer. See the [Open MPI 4.1 MPI_Testall reference](https://www.open-mpi.org/doc/v4.1/man3/MPI_Testall.3.php).

Before final overlap admission, capture request-completion state aligned to the
GPU timeline, or equivalent transport-progress evidence. Do not infer network
bandwidth from the duration of MPI_Isend or MPI_Testall. No claim is made here
that overlap is absent; the current evidence cannot settle that question alone.

## Completion-State Timeline Evidence

The profiling-only `mpi_completion_trace` shared library intercepts the existing
Fortran `mpi_testall_` call and calls PMPI_Testall exactly once with the original
arguments. It records the returned flag as an NVTX mark only when the call had
four requests and at least one non-null request on entry. It adds no progress
calls and does not change the solver, transport or kernel synchronization.
The optional top-level CMake target is disabled by default and is not linked
into ASTR. LD_PRELOAD is applied only to profiled rank processes.

The two-rank `mpi_completion_probe` first withholds a send until its peer has
observed an incomplete receive, then completes the transfer and verifies the
payload. Both plain and interposed runs pass. The traced run records 10 PENDING
marks and one COMPLETE mark. The subsequent all-null Testall is not marked.
The plain export has no NVTX_EVENTS table. Evidence is in
`out/p3_completion_probe_{plain,traced}/`, including logs and NSYS/SQLite files.

The real solver trace uses endpoint-cleanup executable SHA256
`aee22855dc7a27cd1884ff3c33da9da9f287f5ae6900ebb4c9ef2d4e9edb187a`,
TGV 256^3, NP=2, topology 1,2,1, MAXSTEP=3, pinned-overlap and explicit sync.
It traces CUDA and NVTX, without the separate NSYS MPI interposer. All rank
processes finish. Across the two processes:

| Returned Testall flag | Total marks | Marks during same-process stored diffusion kernels |
|---|---:|---:|
| false, requests not all complete | 4261 | 4237 |
| true, all four requests complete | 24 | 24 |

Pending marks coincide with x diffusion (2296) and y diffusion (1941).
All completion marks coincide with y diffusion. Each process records twelve
completions during kernels, corresponding to twelve RK stages in this run.
This supports actual overlap of unfinished MPI requests with useful computation
in the instrumented execution. It does not measure transferred bytes, prove
network-engine concurrency, or establish uninstrumented speedup. Instrumentation
can change polling costs and timing; no result from this run enters formal
performance medians. CUDA-aware transport is not involved.

Evidence: `out/p3_completion_tgv_y_nsys/{run.log,trace.nsys-rep,trace.sqlite}`.
Recompute counts with `summarize_mpi_kernel_overlap.py --completion`. Its tests
cover process separation, interval endpoints, concurrent-kernel deduplication,
inline and interned NVTX strings, missing markers and complete-only events.
The new five tests and the existing six API-intersection tests pass. Remaining
overlap admission depends on final uninstrumented performance reconciliation.

## Final TGV Reconciliation And Admission

All final performance groups use the endpoint-cleanup executable above, without
LD_PRELOAD or a profiler. Formal performance consists of NP=2 on two distinct
physical GPUs, one process warmup, five measured processes and ten retained
complete-RK samples per process. The TGV y/z groups now both pass:

| Slab | Pageable s/RK | Pinned s/RK | Overlap s/RK | Spread %, pageable / pinned / overlap | Pinned reduction % | Overlap increment over pinned % |
|---|---:|---:|---:|---|---:|---:|
| y | 0.6007542585 | 0.5141892660 | 0.4983285730 | 1.094 / 0.970 / 0.166 | 14.409 | 3.085 |
| z | 0.5958139980 | 0.5118258160 | 0.4967553505 | 1.640 / 0.174 / 1.393 | 14.096 | 2.944 |

The y order is pageable, pinned, overlap; z reverses that order. These are
reductions in measured GPU complete-RK time, not speedups over a CPU baseline.

TGV x overlap required complete-pair repeats. Neither noisy group was trimmed:

| Group | Pinned s/RK | Overlap s/RK | Spread %, pinned / overlap | Disposition |
|---|---:|---:|---|---|
| original | 0.5403444060 | 0.5251506485 | 1.401 / 8.020 | overlap inconclusive |
| repeat1, overlap then pinned | 0.5410281270 | 0.5280022320 | 2.150 / 5.125 | overlap inconclusive |
| repeat2, pinned then overlap | 0.5371103215 | 0.5230610630 | 1.394 / 1.170 | valid pair, 2.616% increment |

The original valid x pageable/pinned pair remains the pinned admission reference
(13.839% reduction). The final x pair is the independent overlap reference.
Do not mix individual runs between groups. All 34 final-source groups and
204 process logs, including warmups and all inconclusive repeats, remain in
`out/p3_final_performance_*`. The consolidated
`out/p3_final_performance_complete_audit.json` contains medians, spreads, all
five process medians, per-device memory peaks, registration bytes and input
hashes. Inputs/controllers match within each case/topology across backends and
repeats; SBLI also checks the inflow profile. Every backend log matches its
requested mode, with active overlap only in TGV.

Maximum per-device memory growth across the selected matrix remains 3.436%,
from SBLI z. TGV peaks are GPU 0: 5354 MiB and GPU 1: 5610--5612 MiB, including
desktop allocations. Pinned TGV y registers 179653280 host bytes per rank;
this is not device memory. In inactive Shu-Osher/SBLI paths, the largest valid
overlap-mode slowdown against pinned is 0.523%, below the 1% gate.

| Tier | Decision | Reason |
|---|---|---|
| Pageable blocking L0 | retain as default | portable fallback; full numerical/contract matrix passes |
| Pinned blocking | retain, opt-in | valid nine-case RK reductions 2.829%--23.728%; memory gate passes |
| Standalone paired nonblocking | reject | reproducible increment at most 2.425%, below 3%; source removed |
| Pinned periodic-diffusion overlap | retain, opt-in | y increment 3.085%; x/z positive; all valid spreads below 5%; request-state trace and numerical gates pass |
| CUDA-aware MPI | defer on current stack | large-message or sanitizer admission failures; no solver backend installed |

Overlap admission is limited to the implemented fully periodic stored-diffusion
core. Physical closures and cases without diffusion keep pinned blocking.
Default behavior, FP64, numerical operators, halo widths, interface averaging
and every explicit kernel synchronization are unchanged. No stable regression
above 1% appears in the accepted formal matrix.

### Final Build And Reference Checks

Top-level CPU/GPU builds and optional completion-trace targets pass. All 68
Python tests and 96 shell syntax checks pass. Frozen endpoint solver sources
match the working source archive byte-for-byte; only added excluded diagnostic
targets differ in src/CMakeLists.txt. A subsequent relink produces local GPU
executable SHA256 `9308b81a17d12820595a6f92c2fc763e250d9e398373017e8ff8bf41f984cdbc`.
Formal timings remain attributed to the frozen `aee22855...` executable, not
relabeled as measurements of that rebuilt binary.

Additional TGV 64^3 NP=2 y-slab, MAXSTEP=2 comparisons use the frozen pre-P3
L0 GPU executable as reference. Endpoint pageable, endpoint overlap, a real
NSYS/preloaded overlap run and the rebuilt overlap executable all give zero
maximum difference in six primitive fields and five reconstructed conserved
fields. These are GPU-to-GPU checks despite the comparator's historical
`--cpu` reference argument. Logs and reports are in
`out/p3_completion_field_control/`. The rebuilt controlled MPI probe passes
both without and with preload. These supplement the unchanged 45 CPU/GPU
comparisons, physical-wall checks, production halo contracts and host-only
zero-error sanitizer matrices, rather than replacing them.
