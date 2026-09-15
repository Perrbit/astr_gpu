# ASTR P4-0 Benchmark Trustworthiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in, fail-closed TGV performance mode that removes startup field HDF5 writes and produces attributable phase timing before any P4 communication implementation is changed.

**Architecture:** A CUDA-independent `benchmark_runtime` module owns runtime admission and exposes one read-only I/O decision to `gridgeneration` and `initialisation`. A GPU-only `gpu_phase_timing` module records inclusive prepare time and nested RK phases while explicit synchronization remains authoritative. The performance driver proves that no field HDF5 was created and summarizes five independent runs.

**Tech Stack:** Fortran 2008, CUDA Fortran with NVHPC, MPI, Bash, Python 3 standard library, pytest, CMake, HDF5, Nsight Systems.

**Design:** `docs/superpowers/specs/2026-09-15-astr-p4-compute-communication-overlap-design.md`

## Global Constraints

- Configure every build from `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Work in the existing `feature/gpu_dev` worktree; do not create another worktree.
- Do not inspect, cancel, modify, or reuse the running A800 platform job while implementing P4-0.
- Preserve `ASTR_GPU_SYNC_MODE=explicit` as the default and keep explicit post-kernel synchronization unchanged.
- Keep FP64, full five-component `qwork_d`, sixth-order explicit central differences, tenth-order explicit central filtering, RK3, and fixed `hm` unchanged.
- Accept `ASTR_GPU_BENCHMARK_NO_FIELD_IO=1` only for a CUDA build running three-dimensional, fully periodic GPU TGV with `ASTR_GPU_RK_TIMING=1`.
- Skip only generated-grid `writegrid` and startup `writeflfed`; preserve normal output, checkpoint, statistics, and restart semantics.
- Require all MPI ranks to select identical benchmark and phase-timing modes; configuration disagreement aborts collectively.
- Treat local two-GPU measurements as screening evidence, not A800 performance.
- Leave unrelated untracked manuals, presentations, rendered slides, and presentation build scripts untracked.

## File Map

- Create `src/benchmark_runtime.F90`: parse and collectively validate benchmark-only no-field-I/O mode.
- Modify `src/astr.F90`: configure policy after input and MPI topology setup, before grid or field I/O.
- Modify `src/gridgeneration.F90`: guard only generated-grid `writegrid`.
- Modify `src/initialisation.F90`: guard only startup `writeflfed`.
- Create `src_gpu/gpu_phase_timing.cuf`: configure and emit strict rank/stage phase records.
- Modify `src_gpu/gpu_runtime.cuf`: configure phase timing and mark inclusive prepare work.
- Modify `src_gpu/mainloop_gpu.cuf`: mark filter, solution halo, convection, diffusion flux, diffusion halo, diffusion RHS, and RK update.
- Modify `src/CMakeLists.txt`: compile modules in dependency order and add a policy probe.
- Create `tests/gpu_validation/benchmark_runtime_probe.F90`: exercise policy without production arrays.
- Create `tests/gpu_validation/run_benchmark_runtime_contract.sh`: test valid, invalid, and rank-inconsistent configurations.
- Create `tests/gpu_validation/test_benchmark_no_field_io_contract.py`: protect startup call order and narrow I/O guards.
- Modify `tests/gpu_validation/run_tgv_256_performance_benchmark.sh`: enforce no field I/O and collect phase reports.
- Modify `tests/gpu_validation/test_tgv_performance_driver.py`: protect driver forwarding and output rejection.
- Create `tests/gpu_validation/summarize_gpu_phase_timing.py`: validate and summarize slowest-rank phase samples.
- Create `tests/gpu_validation/test_gpu_phase_timing.py`: test parser rejection and summary behavior.
- Modify `tests/gpu_validation/README.md`: document local P4-0 commands and interpretation limits.
- Create `documents/ASTR_PHASE_P4_0_BASELINE_REPORT.md`: record the evidence and P4-1 go/no-go decision.

---

### Task 1: Add The Fail-Closed Benchmark Runtime Policy

**Files:**
- Create: `tests/gpu_validation/benchmark_runtime_probe.F90`
- Create: `tests/gpu_validation/run_benchmark_runtime_contract.sh`
- Modify: `src/CMakeLists.txt`
- Create: `src/benchmark_runtime.F90`

**Interfaces:**
- Consumes: `MPI_COMM_WORLD`, `_CUDA`, `ASTR_GPU_BENCHMARK_NO_FIELD_IO`, and `ASTR_GPU_RK_TIMING`.
- Produces: `configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo,periodic_boundary_case)` and `benchmark_field_io_disabled()`.

- [ ] **Step 1: Write the policy probe and shell contract**

Create `tests/gpu_validation/benchmark_runtime_probe.F90`:

```fortran
program benchmark_runtime_probe
  use mpi
  use benchmark_runtime, only: configure_benchmark_runtime,benchmark_field_io_disabled
  implicit none
  character(32) :: gpu_arg,flow_arg,homogeneous_arg,boundary_arg,expected_arg
  logical :: use_gpu,homogeneous,periodic_boundary,expected
  integer :: ierr,rank

  call mpi_init(ierr)
  call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,gpu_arg)
  call get_command_argument(2,flow_arg)
  call get_command_argument(3,homogeneous_arg)
  call get_command_argument(4,boundary_arg)
  call get_command_argument(5,expected_arg)
  use_gpu=trim(gpu_arg)=='gpu'
  homogeneous=trim(homogeneous_arg)=='periodic'
  periodic_boundary=trim(boundary_arg)=='periodic'
  expected=trim(expected_arg)=='enabled'
  call configure_benchmark_runtime(use_gpu,trim(flow_arg),3,homogeneous,homogeneous,homogeneous, &
                                   periodic_boundary)
  if(benchmark_field_io_disabled().neqv.expected) then
    write(*,'(A,I0)') 'benchmark runtime probe mismatch on rank ',rank
    call mpi_abort(MPI_COMM_WORLD,1,ierr)
  endif
  if(rank==0) write(*,'(A,L1)') 'benchmark runtime probe passed enabled=',expected
  call mpi_finalize(ierr)
end program benchmark_runtime_probe
```

Create `tests/gpu_validation/run_benchmark_runtime_contract.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

GPU_PROBE="${1:?usage: run_benchmark_runtime_contract.sh /absolute/path/to/gpu-probe /absolute/path/to/cpu-probe}"
CPU_PROBE="${2:?usage: run_benchmark_runtime_contract.sh /absolute/path/to/gpu-probe /absolute/path/to/cpu-probe}"

# Avoid unavailable HCOLL transport noise in this local test only.
export OMPI_MCA_coll_hcoll_enable=0

expect_fail() {
  set +e
  "$@" > /tmp/astr_p4_benchmark_expected_failure.log 2>&1
  local status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "expected command to fail: $*" >&2
    exit 1
  fi
}

env -u ASTR_GPU_BENCHMARK_NO_FIELD_IO -u ASTR_GPU_RK_TIMING \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic disabled
ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 2 "$GPU_PROBE" gpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" cpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$CPU_PROBE" gpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=0 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu channel periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv physical physical enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic physical enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=invalid ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic disabled
expect_fail env -u ASTR_GPU_BENCHMARK_NO_FIELD_IO ASTR_GPU_RK_TIMING=invalid \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic disabled
expect_fail mpirun -np 2 bash -c '
  if [[ "$OMPI_COMM_WORLD_RANK" == 0 ]]; then
    export ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1
  else
    unset ASTR_GPU_BENCHMARK_NO_FIELD_IO ASTR_GPU_RK_TIMING
  fi
  exec "$1" gpu tgv periodic periodic enabled
' bash "$GPU_PROBE"
expect_fail mpirun -np 2 bash -c '
  unset ASTR_GPU_BENCHMARK_NO_FIELD_IO
  if [[ "$OMPI_COMM_WORLD_RANK" == 0 ]]; then
    export ASTR_GPU_RK_TIMING=1
  else
    export ASTR_GPU_RK_TIMING=0
  fi
  exec "$1" gpu tgv periodic periodic disabled
' bash "$GPU_PROBE"
echo "benchmark runtime contract passed"
```

Add this target to `src/CMakeLists.txt` before the CUDA-only probe block:

```cmake
add_executable(benchmark_runtime_probe EXCLUDE_FROM_ALL
  benchmark_runtime.F90 ../tests/gpu_validation/benchmark_runtime_probe.F90)
set_target_properties(benchmark_runtime_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/benchmark_runtime_probe_modules")
target_include_directories(benchmark_runtime_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/benchmark_runtime_probe_modules")
target_link_libraries(benchmark_runtime_probe PRIVATE MPI::MPI_Fortran)
```

- [ ] **Step 2: Run the probe build and verify the missing source failure**

```bash
cmake -S /home/dell/workspace/astr_gpu \
  -B /home/dell/workspace/astr_gpu/build_gpu_p4 \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DASTR_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /home/dell/workspace/astr_gpu/build_gpu_p4 \
  --target benchmark_runtime_probe -j2
```

Expected: configure or build fails because `src/benchmark_runtime.F90` is absent.

- [ ] **Step 3: Implement the runtime module**

Create `src/benchmark_runtime.F90`:

```fortran
module benchmark_runtime
  use mpi
  implicit none
  private
  public :: configure_benchmark_runtime,benchmark_field_io_disabled
  logical,save :: configured=.false.,field_io_disabled=.false.
contains
  integer function switch_choice(name)
    character(*),intent(in) :: name
    character(32) :: value
    integer :: length,status
    value=''
    call get_environment_variable(name,value,length=length,status=status)
    switch_choice=0
    if(status==1 .or. length==0) return
    if(status/=0) then
      switch_choice=-1
      return
    endif
    select case(trim(adjustl(value)))
    case('1','t','T','true','TRUE','on','ON')
      switch_choice=1
    case('0','f','F','false','FALSE','off','OFF')
      switch_choice=0
    case default
      switch_choice=-1
    end select
  end function switch_choice

  subroutine configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo,periodic_boundary_case)
    logical,intent(in) :: use_gpu,lihomo,ljhomo,lkhomo,periodic_boundary_case
    character(*),intent(in) :: flowtype
    integer,intent(in) :: ndims
    integer :: requested,rk_timing,lowest,highest,rk_lowest,rk_highest
    integer :: ierr,rank,ignored
#ifdef _CUDA
    logical,parameter :: cuda_build=.true.
#else
    logical,parameter :: cuda_build=.false.
#endif
    if(configured) return
    requested=switch_choice('ASTR_GPU_BENCHMARK_NO_FIELD_IO')
    rk_timing=switch_choice('ASTR_GPU_RK_TIMING')
    call mpi_allreduce(requested,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(requested,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(rk_timing,rk_lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(rk_timing,rk_highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0 .or. lowest/=highest) then
      print *, 'Invalid or inconsistent GPU benchmark environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    if(rk_lowest<0 .or. rk_lowest/=rk_highest) then
      print *, 'Invalid or inconsistent ASTR_GPU_RK_TIMING environment'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    field_io_disabled=requested==1
    if(field_io_disabled .and. (.not.cuda_build .or. .not.use_gpu .or. &
       ndims/=3 .or. trim(flowtype)/='tgv' .or. &
       .not.(lihomo.and.ljhomo.and.lkhomo) .or. &
       .not.periodic_boundary_case .or. &
       rk_timing/=1)) then
      print *, 'ASTR_GPU_BENCHMARK_NO_FIELD_IO requires CUDA 3-D periodic GPU TGV with RK timing'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    configured=.true.
    if(field_io_disabled) then
      call mpi_comm_rank(MPI_COMM_WORLD,rank,ierr)
      if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
      if(rank==0) write(*,'(A)') 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
    endif
  end subroutine configure_benchmark_runtime

  logical function benchmark_field_io_disabled()
    benchmark_field_io_disabled=configured.and.field_io_disabled
  end function benchmark_field_io_disabled
end module benchmark_runtime
```

Add `benchmark_runtime.F90` after `commvar.F90` in `ASTR_SOURCES`.

- [ ] **Step 4: Build and run the policy contract**

```bash
cmake --build /home/dell/workspace/astr_gpu/build_gpu_p4 \
  --target benchmark_runtime_probe -j2
cmake -S /home/dell/workspace/astr_gpu \
  -B /home/dell/workspace/astr_gpu/build_cpu_p4 \
  -DCMAKE_Fortran_COMPILER=nvfortran \
  -DASTR_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /home/dell/workspace/astr_gpu/build_cpu_p4 \
  --target benchmark_runtime_probe -j2
bash tests/gpu_validation/run_benchmark_runtime_contract.sh \
  /home/dell/workspace/astr_gpu/build_gpu_p4/bin/benchmark_runtime_probe \
  /home/dell/workspace/astr_gpu/build_cpu_p4/bin/benchmark_runtime_probe
```

Expected: valid runs pass, invalid runs return nonzero, and the script ends with `benchmark runtime contract passed`.

- [ ] **Step 5: Commit the policy**

```bash
git add src/benchmark_runtime.F90 src/CMakeLists.txt \
  tests/gpu_validation/benchmark_runtime_probe.F90 \
  tests/gpu_validation/run_benchmark_runtime_contract.sh
git commit -m "feat(gpu): add fail-closed benchmark runtime policy"
```

### Task 2: Guard Only The Two Startup Field Writes

**Files:**
- Create: `tests/gpu_validation/test_benchmark_no_field_io_contract.py`
- Modify: `src/astr.F90`
- Modify: `src/gridgeneration.F90`
- Modify: `src/initialisation.F90`

**Interfaces:**
- Consumes: the Task 1 policy procedures.
- Produces: policy configuration before `gridgen` and default-off guards around two startup writes.

- [ ] **Step 1: Write the source contract**

Create `tests/gpu_validation/test_benchmark_no_field_io_contract.py`:

```python
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _source(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line.split("!", 1)[0] for line in lines)


def _compact(source: str) -> str:
    return re.sub(r"[\s&]+", "", source).lower()


ASTR = _source(SRC / "astr.F90")
GRID = _source(SRC / "gridgeneration.F90")
INIT = _source(SRC / "initialisation.F90")
MAINLOOP = _source(SRC / "mainloop.F90")


def test_policy_is_configured_after_mpi_and_before_initialization() -> None:
    source = _compact(ASTR)
    configure = (
        "callconfigure_benchmark_runtime"
        "(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo,all(bctype(1:6)==1))"
    )
    calls = [
        "callparallelini",
        "callrefcal",
        configure,
        "callfileini",
        "callgridgen",
        "callflowinit",
    ]

    assert all(source.count(call) == 1 for call in calls)
    positions = {call: source.index(call) for call in calls}
    assert positions["callparallelini"] < positions["callrefcal"]
    assert positions["callrefcal"] < positions[configure]
    assert positions[configure] < positions["callfileini"]
    assert positions[configure] < positions["callgridgen"]
    assert positions[configure] < positions["callflowinit"]


def test_benchmark_query_has_exactly_two_production_call_sites() -> None:
    query = re.compile(r"\bbenchmark_field_io_disabled\s*\(\s*\)", re.IGNORECASE)
    declaration = re.compile(
        r"\bfunction\s+benchmark_field_io_disabled\s*\(\s*\)", re.IGNORECASE
    )
    sites = {}

    for path in sorted(SRC.rglob("*.F90")):
        source = _source(path)
        call_count = len(query.findall(source)) - len(declaration.findall(source))
        if call_count:
            sites[path.relative_to(ROOT).as_posix()] = call_count

    assert sites == {
        "src/gridgeneration.F90": 1,
        "src/initialisation.F90": 1,
    }


def test_read_grid_branch_is_not_guarded_by_benchmark_policy() -> None:
    read_branch = re.search(
        r"\bif\s*\(\s*lreadgrid\s*\)\s*then\b(?P<body>.*?)^\s*else\b",
        GRID,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    assert read_branch is not None
    body = _compact(read_branch.group("body"))
    assert "callreadgrid(trim(gridfile))" in body
    assert "benchmark_field_io_disabled" not in body


def test_later_output_paths_cannot_use_startup_benchmark_policy() -> None:
    assert "benchmark_field_io_disabled" not in _compact(MAINLOOP)
```

- [ ] **Step 2: Verify the policy-wiring tests fail**

```bash
python3 -m pytest -q tests/gpu_validation/test_benchmark_no_field_io_contract.py
```

Expected: the ordering and exact-call-site tests fail before production startup is wired; the existing unguarded `readgrid` and `mainloop` invariants already pass.

- [ ] **Step 3: Wire configuration and the narrow guards**

In `src/astr.F90`, use:

```fortran
  use commvar, only: use_gpu,prandtl,flowtype,ndims,lihomo,ljhomo,lkhomo
  use benchmark_runtime, only: configure_benchmark_runtime
```

Immediately after `call refcal`, where `commvar::ndims` has been initialized, insert:

```fortran
    call configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo, &
                                     all(bctype(1:6)==1))
```

In `gridgeneration::gridgen`, import the query and replace only line 95:

```fortran
    use benchmark_runtime, only: benchmark_field_io_disabled
```

```fortran
      if(.not.benchmark_field_io_disabled()) call writegrid(trim(gridfile))
```

In `initialisation::flowinit`, import the query and replace only line 203:

```fortran
    use benchmark_runtime, only: benchmark_field_io_disabled
```

```fortran
    if(.not.benchmark_field_io_disabled()) call writeflfed(timerept=.true.)
```

- [ ] **Step 4: Test and build CPU plus GPU**

```bash
python3 -m pytest -q tests/gpu_validation/test_benchmark_no_field_io_contract.py
cmake --build /home/dell/workspace/astr_gpu/build_gpu_p4 --target astr -j2
cmake -S /home/dell/workspace/astr_gpu \
  -B /home/dell/workspace/astr_gpu/build_cpu_p4 \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DASTR_WITH_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /home/dell/workspace/astr_gpu/build_cpu_p4 --target astr -j2
```

Expected: 3 passed tests and both production binaries link from the top-level CMake project.

- [ ] **Step 5: Commit the startup guards**

```bash
git add src/astr.F90 src/gridgeneration.F90 src/initialisation.F90 \
  tests/gpu_validation/test_benchmark_no_field_io_contract.py
git commit -m "feat(gpu): suppress startup field IO in benchmark mode"
```

### Task 3: Make The Performance Driver Prove A Clean Lifecycle

**Files:**
- Modify: `tests/gpu_validation/test_tgv_performance_driver.py`
- Modify: `tests/gpu_validation/run_tgv_256_performance_benchmark.sh`

**Interfaces:**
- Consumes: `ASTR_GPU_BENCHMARK_NO_FIELD_IO=1` and its startup confirmation line.
- Produces: no `grid*.h5` or `flowfield*.h5`, metadata `benchmark_no_field_io=1`, one warm-up process, and five retained processes.

- [ ] **Step 1: Add failing driver contracts**

Append to `TgvPerformanceDriverTests`:

```python
    def test_driver_enables_and_records_no_field_io_mode(self) -> None:
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1", SCRIPT)
        self.assertIn("benchmark_no_field_io=1", SCRIPT)
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled", SCRIPT)

    def test_driver_forces_generated_grid_and_removes_copied_hdf5(self) -> None:
        self.assertIn('args+=(--lreadgrid f)', SCRIPT)
        self.assertIn('rm -f "$CASE_DIR/datin/grid.h5"', SCRIPT)

    def test_driver_rejects_generated_grid_or_flowfield_hdf5(self) -> None:
        self.assertIn("assert_no_field_hdf5", SCRIPT)
        self.assertIn("-name 'grid*.h5'", SCRIPT)
        self.assertIn("-name 'flowfield*.h5'", SCRIPT)
```

- [ ] **Step 2: Verify only the new cases fail**

```bash
python3 -m pytest -q tests/gpu_validation/test_tgv_performance_driver.py
```

Expected: six existing cases pass and three new cases fail.

- [ ] **Step 3: Add launch and post-run HDF5 guards**

Immediately after the `prepare_case` argument array is closed, append:

```bash
  args+=(--lreadgrid f)
```

Add before `run_once`:

```bash
assert_no_field_hdf5() {
  local found
  found="$(find "$CASE_DIR" -type f \
    \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  if [[ -n "$found" ]]; then
    printf 'benchmark generated forbidden field HDF5:\n%s\n' "$found" >&2
    exit 1
  fi
}
```

Add beside the RK timing environment:

```bash
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 \
```

After completion and crash-marker checks, add:

```bash
  grep -q 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log"
  assert_no_field_hdf5
```

Immediately after `prepare_case "$CASE_DIR"`, remove the copied source grid:

```bash
rm -f "$CASE_DIR/datin/grid.h5"
assert_no_field_hdf5
```

Replace the metadata `printf` with:

```bash
printf 'np=%s\ntopology=%s\ngpu_ids=%s\nhalo_transport=%s\nfilter_workspace=%s\nbenchmark_no_field_io=1\n' \
  "$NP" "$TOPOLOGY" "$GPU_IDS" "$HALO_TRANSPORT" "$FILTER_WORKSPACE" \
  > "$OUT_DIR/${LABEL}_transport_metadata.txt"
```

- [ ] **Step 4: Run tests and a 64-cubed lifecycle smoke**

```bash
python3 -m pytest -q tests/gpu_validation/test_tgv_performance_driver.py
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_io_smoke \
LABEL=np1_io_smoke GRID=64,64,64 MAXSTEP=1 DISCARD_STEPS=1 REPEATS=5 \
GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr \
GPU_IDS=0 NP=1 TOPOLOGY=1,1,1 SYNC_MODE=explicit \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
bash tests/gpu_validation/run_tgv_256_performance_benchmark.sh
find tests/gpu_validation/out/p4_0_io_smoke -type f \
  \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print
```

Expected: 9 passed tests, six solver processes finish, and the final `find` prints nothing.

- [ ] **Step 5: Verify default field output remains active**

```bash
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_default_output \
MAXSTEP=1 FEQCHKPT=1 \
CPU_EXE=/home/dell/workspace/astr_gpu/build_cpu_p4/bin/astr \
GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr \
bash tests/gpu_validation/run_tgv_field_compare.sh
find tests/gpu_validation/out/p4_0_default_output -type f -name '*.h5' -print
```

Expected: CPU/GPU comparison passes and the final `find` lists normal startup/checkpoint HDF5 files.

- [ ] **Step 6: Commit the lifecycle-proof driver**

```bash
git add tests/gpu_validation/run_tgv_256_performance_benchmark.sh \
  tests/gpu_validation/test_tgv_performance_driver.py
git commit -m "test(gpu): enforce field-free TGV performance runs"
```

### Task 4: Build A Strict Phase-Timing Parser

**Files:**
- Create: `tests/gpu_validation/test_gpu_phase_timing.py`
- Create: `tests/gpu_validation/summarize_gpu_phase_timing.py`

**Interfaces:**
- Consumes: `ASTR_GPU_PHASE_TIMING rank nstep rkstep label seconds`.
- Produces: `parse_phase_log(text,ranks,steps)` and `summarize_logs(paths,ranks,steps)` plus a TSV report.

- [ ] **Step 1: Write failing parser tests**

Create `tests/gpu_validation/test_gpu_phase_timing.py`:

```python
from pathlib import Path
import math
import tempfile

import pytest

from summarize_gpu_phase_timing import parse_phase_log, summarize_logs


PHASE_COUNTS = {
    "prepare": 1,
    "filter": 3,
    "solution_halo": 3,
    "convection": 3,
    "diffusion_flux": 3,
    "diffusion_halo": 3,
    "diffusion_rhs": 3,
    "rk_update": 3,
}


def valid_log(ranks: int = 2, steps: int = 1, scale: float = 1.0) -> str:
    lines = []
    for step in range(steps):
        for rank in range(ranks):
            lines.append(f"ASTR_GPU_PHASE_TIMING {rank} {step} 0 prepare {scale*(rank+1):.6f}")
            for rkstep in range(1, 4):
                for index, label in enumerate(list(PHASE_COUNTS)[1:], start=1):
                    value = scale * (100 * rank + 10 * rkstep + index)
                    lines.append(
                        f"ASTR_GPU_PHASE_TIMING {rank} {step} {rkstep} {label} {value:.6f}"
                    )
    return "\n".join(lines)


def test_parser_uses_slowest_rank_for_each_occurrence() -> None:
    samples = parse_phase_log(valid_log(), ranks=2, steps=1)
    assert samples["prepare"] == [2.0]
    assert samples["filter"] == [111.0, 121.0, 131.0]
    assert all(len(samples[label]) == count for label, count in PHASE_COUNTS.items())


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "nan", "unknown"])
def test_parser_rejects_bad_records(mutation: str) -> None:
    lines = valid_log().splitlines()
    if mutation == "duplicate":
        lines.append(lines[0])
    elif mutation == "missing":
        lines.pop()
    elif mutation == "nan":
        lines[0] = lines[0].rsplit(" ", 1)[0] + " nan"
    else:
        lines[0] = lines[0].replace("prepare", "other")
    with pytest.raises(ValueError):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


def test_summary_keeps_independent_logs_separate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "first.log"
        second = Path(tmp) / "second.log"
        first.write_text(valid_log(scale=1.0), encoding="ascii")
        second.write_text(valid_log(scale=2.0), encoding="ascii")
        rows = summarize_logs([first, second], ranks=2, steps=1)
    prepare = next(row for row in rows if row[0] == "prepare")
    assert prepare[1] == 2
    assert math.isclose(prepare[2], 3.0)
```

- [ ] **Step 2: Verify collection fails because the parser is absent**

```bash
python3 -m pytest -q tests/gpu_validation/test_gpu_phase_timing.py
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the parser and CLI**

Create `tests/gpu_validation/summarize_gpu_phase_timing.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


PHASE_COUNTS = {
    "prepare": 1,
    "filter": 3,
    "solution_halo": 3,
    "convection": 3,
    "diffusion_flux": 3,
    "diffusion_halo": 3,
    "diffusion_rhs": 3,
    "rk_update": 3,
}


def parse_phase_log(text: str, ranks: int, steps: int) -> dict[str, list[float]]:
    records: dict[tuple[int, int, int, str], float] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "ASTR_GPU_PHASE_TIMING":
            continue
        if len(fields) != 6:
            raise ValueError(f"invalid phase timing record: {line}")
        rank, step, rkstep = map(int, fields[1:4])
        label = fields[4]
        seconds = float(fields[5])
        key = (rank, step, rkstep, label)
        if label not in PHASE_COUNTS or key in records:
            raise ValueError(f"unknown or duplicate phase record: {line}")
        if rank < 0 or rank >= ranks or step < 0 or step >= steps:
            raise ValueError(f"phase record outside expected range: {line}")
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"non-finite or negative phase time: {line}")
        records[key] = seconds

    samples = {label: [] for label in PHASE_COUNTS}
    for step in range(steps):
        occurrences = [(0, "prepare")]
        occurrences.extend(
            (rkstep, label)
            for rkstep in range(1, 4)
            for label in list(PHASE_COUNTS)[1:]
        )
        for rkstep, label in occurrences:
            values = []
            for rank in range(ranks):
                key = (rank, step, rkstep, label)
                if key not in records:
                    raise ValueError(f"missing phase record {key}")
                values.append(records[key])
            samples[label].append(max(values))
    expected = steps * sum(PHASE_COUNTS.values()) * ranks
    if len(records) != expected:
        raise ValueError(f"expected {expected} phase records, found {len(records)}")
    return samples


def summarize_logs(
    paths: list[Path], ranks: int, steps: int
) -> list[tuple[str, int, float, float, float]]:
    combined = {label: [] for label in PHASE_COUNTS}
    for path in paths:
        samples = parse_phase_log(path.read_text(encoding="utf-8"), ranks, steps)
        for label, values in samples.items():
            combined[label].extend(values)
    return [
        (label, len(values), statistics.median(values), min(values), max(values))
        for label, values in combined.items()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = summarize_logs(args.log, args.ranks, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii") as stream:
        stream.write("phase\tsamples\tmedian_seconds\tmin_seconds\tmax_seconds\n")
        for label, count, median, minimum, maximum in rows:
            stream.write(
                f"{label}\t{count}\t{median:.12e}\t{minimum:.12e}\t{maximum:.12e}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all parser tests**

```bash
python3 -m pytest -q tests/gpu_validation/test_gpu_phase_timing.py
```

Expected: 6 passed tests.

- [ ] **Step 5: Commit the parser**

```bash
git add tests/gpu_validation/summarize_gpu_phase_timing.py \
  tests/gpu_validation/test_gpu_phase_timing.py
git commit -m "test(gpu): add strict RK phase timing parser"
```

### Task 5: Instrument The Existing Explicit TGV Path

**Files:**
- Create: `src_gpu/gpu_phase_timing.cuf`
- Modify: `src/CMakeLists.txt`
- Modify: `src_gpu/gpu_runtime.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Modify: `tests/gpu_validation/run_tgv_256_performance_benchmark.sh`
- Modify: `tests/gpu_validation/test_tgv_performance_driver.py`
- Modify: `tests/gpu_validation/test_benchmark_no_field_io_contract.py`

**Interfaces:**
- Consumes: `sync_gpu_boundary`, `parallel::ptime`, `parallel::mpirank`, `commvar::nstep`, and `ASTR_GPU_PHASE_TIMING`.
- Produces: `configure_gpu_phase_timing(allow_tgv,allow_explicit)`, `begin_gpu_phase(label)`, and `end_gpu_phase(label,nstep,rkstep)`.

- [ ] **Step 1: Add failing source and driver contracts**

Append to `tests/gpu_validation/test_benchmark_no_field_io_contract.py`:

```python
GPU_RUNTIME = (ROOT / "src_gpu/gpu_runtime.cuf").read_text(encoding="utf-8")
GPU_LOOP = (ROOT / "src_gpu/mainloop_gpu.cuf").read_text(encoding="utf-8")


def test_phase_timing_covers_required_p4_0_intervals() -> None:
    for label in (
        "prepare",
        "filter",
        "solution_halo",
        "convection",
        "diffusion_flux",
        "diffusion_halo",
        "diffusion_rhs",
        "rk_update",
    ):
        source = GPU_RUNTIME if label == "prepare" else GPU_LOOP
        assert f"begin_gpu_phase('{label}')" in source
        assert f"end_gpu_phase('{label}'" in source
```

Append to `TgvPerformanceDriverTests`:

```python
    def test_driver_forwards_phase_timing_and_writes_summary(self) -> None:
        self.assertIn('PHASE_TIMING="${PHASE_TIMING:-0}"', SCRIPT)
        self.assertIn('ASTR_GPU_PHASE_TIMING="$PHASE_TIMING"', SCRIPT)
        self.assertIn("summarize_gpu_phase_timing.py", SCRIPT)
```

- [ ] **Step 2: Verify the two new contracts fail**

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_benchmark_no_field_io_contract.py \
  tests/gpu_validation/test_tgv_performance_driver.py
```

Expected: previous cases pass; the phase coverage and phase forwarding cases fail.

- [ ] **Step 3: Implement the phase timing module**

Create `src_gpu/gpu_phase_timing.cuf`:

```fortran
module gpu_phase_timing
  use mpi
  implicit none
  private
  public :: configure_gpu_phase_timing,begin_gpu_phase,end_gpu_phase
  integer,parameter :: max_depth=8
  logical,save :: configured=.false.,enabled=.false.
  integer,save :: depth=0
  character(32),save :: labels(max_depth)
  real(8),save :: starts(max_depth)
contains
  integer function switch_choice(name)
    character(*),intent(in) :: name
    character(32) :: value
    integer :: length,status
    value=''
    call get_environment_variable(name,value,length=length,status=status)
    switch_choice=0
    if(status==1 .or. length==0) return
    if(status/=0) then
      switch_choice=-1
      return
    endif
    select case(trim(adjustl(value)))
    case('1','t','T','true','TRUE','on','ON')
      switch_choice=1
    case('0','f','F','false','FALSE','off','OFF')
      switch_choice=0
    case default
      switch_choice=-1
    end select
  end function switch_choice

  subroutine configure_gpu_phase_timing(allow_tgv,allow_explicit)
    logical,intent(in) :: allow_tgv,allow_explicit
    integer :: requested,lowest,highest,ierr,ignored
    if(configured) return
    requested=switch_choice('ASTR_GPU_PHASE_TIMING')
    call mpi_allreduce(requested,lowest,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    call mpi_allreduce(requested,highest,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(lowest<0 .or. lowest/=highest) then
      print *, 'Invalid or inconsistent ASTR_GPU_PHASE_TIMING'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    enabled=requested==1
    if(enabled .and. (.not.allow_tgv .or. .not.allow_explicit)) then
      print *, 'ASTR_GPU_PHASE_TIMING requires 3-D periodic TGV with explicit synchronization'
      call mpi_abort(MPI_COMM_WORLD,1,ignored)
    endif
    configured=.true.
  end subroutine configure_gpu_phase_timing

  subroutine begin_gpu_phase(label)
    use parallel, only: ptime
    character(*),intent(in) :: label
    integer :: ierr
    if(.not.enabled) return
    if(depth==max_depth) then
      print *, 'GPU phase timing nesting exceeds max_depth'
      call mpi_abort(MPI_COMM_WORLD,1,ierr)
    endif
    depth=depth+1
    labels(depth)=trim(label)
    starts(depth)=ptime()
  end subroutine begin_gpu_phase

  subroutine end_gpu_phase(label,nstep,rkstep)
    use gpu_check, only: sync_gpu_boundary
    use parallel, only: mpirank,ptime
    character(*),intent(in) :: label
    integer,intent(in) :: nstep,rkstep
    real(8) :: seconds
    integer :: ierr
    if(.not.enabled) return
    if(depth<1 .or. trim(labels(depth))/=trim(label)) then
      print *, 'GPU phase timing nesting mismatch: ',trim(label)
      call mpi_abort(MPI_COMM_WORLD,1,ierr)
    endif
    call sync_gpu_boundary('phase_'//trim(label))
    seconds=ptime()-starts(depth)
    write(*,'(A,3(1X,I0),1X,A,1X,ES24.16E3)') &
      'ASTR_GPU_PHASE_TIMING',mpirank,nstep,rkstep,trim(label),seconds
    depth=depth-1
  end subroutine end_gpu_phase
end module gpu_phase_timing
```

Add `gpu_phase_timing.cuf` after `gpu_check.cuf` in `ASTR_GPU_SOURCES`.

- [ ] **Step 4: Configure phase timing and mark inclusive prepare**

In `gpu_runtime::gpu_after_flowinit`, import and call:

```fortran
    use gpu_phase_timing, only: configure_gpu_phase_timing
```

```fortran
    allow_phase_timing=allow_selective .and. gpu_periodic_boundary_case() .and. &
                       lfilter .and. diffterm
    call configure_gpu_phase_timing(allow_phase_timing,.not.gpu_selective_sync_enabled())
```

In `gpu_runtime::gpu_prepare_rkfirst_stats`, add `nstep` to the `commvar` import, import the phase API, then place begin before `begin_rk_prepare_timing` and end after `end_rk_prepare_timing`:

```fortran
    use gpu_phase_timing, only: begin_gpu_phase,end_gpu_phase
```

```fortran
    call begin_gpu_phase('prepare')
```

```fortran
    call end_gpu_phase('prepare',nstep,0)
```

- [ ] **Step 5: Mark filter and solution-halo work for all three RK stages**

In `prepare_rkfirst_stats_gpu`, add `nstep` to the `commvar` import and import the phase API. Wrap the existing `apply_explicit_filter_gpu(periodic_filter_case,'prepare_')` call:

```fortran
      call begin_gpu_phase('filter')
      call apply_explicit_filter_gpu(periodic_filter_case,'prepare_')
      call end_gpu_phase('filter',nstep,1)
```

Wrap the periodic `exchange_solution_halo_gpu(.true.)` call in the same subroutine:

```fortran
      call begin_gpu_phase('solution_halo')
      call exchange_solution_halo_gpu(.true.)
      call end_gpu_phase('solution_halo',nstep,1)
```

In `time_integration_rk_gpu`, import the phase API. Wrap the filter call at the existing `if(lfilter)` site:

```fortran
          call begin_gpu_phase('filter')
          call apply_explicit_filter_gpu(periodic_filter_case,'')
          call end_gpu_phase('filter',nstep,rkstep)
```

Wrap the periodic post-filter solution exchange:

```fortran
          call begin_gpu_phase('solution_halo')
          call exchange_solution_halo_gpu(.true.)
          call end_gpu_phase('solution_halo',nstep,rkstep)
```

Only the fully periodic TGV configuration can enable this timer, so physical-boundary and chemistry branches remain structurally unchanged.

- [ ] **Step 6: Mark convection, diffusion, and RK update blocks**

Insert before `if(sod_s0a1_case) then` and after its complete case dispatch:

```fortran
      call begin_gpu_phase('convection')
```

```fortran
      call end_gpu_phase('convection',nstep,rkstep)
```

Inside `if(diffterm)`, insert the following pairs around the existing flux dispatch, halo dispatch, and stored RHS dispatch respectively:

```fortran
        call begin_gpu_phase('diffusion_flux')
```

```fortran
        call end_gpu_phase('diffusion_flux',nstep,rkstep)
        call begin_gpu_phase('diffusion_halo')
```

```fortran
        call end_gpu_phase('diffusion_halo',nstep,rkstep)
        call begin_gpu_phase('diffusion_rhs')
```

```fortran
        call end_gpu_phase('diffusion_rhs',nstep,rkstep)
```

The exact anchors are: flux begins before the branch at current line 1006; flux ends before `diffusion_region=0`; halo ends after the `exchange_diffusion_flux_*` branch; RHS ends immediately before the closing `endif` at current line 1213. Do not move or rewrite any kernel dispatch.

Wrap the existing RK update branch with:

```fortran
      call begin_gpu_phase('rk_update')
```

```fortran
      call end_gpu_phase('rk_update',nstep,rkstep)
```

- [ ] **Step 7: Forward phase timing and summarize five retained logs**

Add to the driver variables and validation:

```bash
PHASE_TIMING="${PHASE_TIMING:-0}"
PHASE_SUMMARY="$OUT_DIR/${LABEL}_phase_summary.tsv"
if [[ "$PHASE_TIMING" != "0" && "$PHASE_TIMING" != "1" ]]; then
  echo "PHASE_TIMING must be 0 or 1" >&2
  exit 2
fi
if [[ "$PHASE_TIMING" == "1" && "$SYNC_MODE" != "explicit" ]]; then
  echo "PHASE_TIMING=1 requires SYNC_MODE=explicit" >&2
  exit 2
fi
```

Add to the solver environment and metadata:

```bash
      ASTR_GPU_PHASE_TIMING="$PHASE_TIMING" \
```

```bash
printf 'phase_timing=%s\n' "$PHASE_TIMING" >> "$OUT_DIR/${LABEL}_transport_metadata.txt"
```

After the retained-run loop and before the final performance summary, add:

```bash
if [[ "$PHASE_TIMING" == "1" ]]; then
  phase_args=()
  for repeat in $(seq 1 "$REPEATS"); do
    phase_args+=(--log "$OUT_DIR/${LABEL}_run_${repeat}/run.log")
  done
  python3 "$ROOT_DIR/tests/gpu_validation/summarize_gpu_phase_timing.py" \
    "${phase_args[@]}" --ranks "$NP" --steps "$((MAXSTEP + 1))" \
    --output "$PHASE_SUMMARY"
fi
```

- [ ] **Step 8: Build, test, and run a phase smoke**

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_benchmark_no_field_io_contract.py \
  tests/gpu_validation/test_tgv_performance_driver.py \
  tests/gpu_validation/test_gpu_phase_timing.py
cmake --build /home/dell/workspace/astr_gpu/build_gpu_p4 --target astr -j2
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_phase_smoke \
LABEL=np1_phase_smoke GRID=64,64,64 MAXSTEP=1 DISCARD_STEPS=1 REPEATS=5 \
GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr \
GPU_IDS=0 NP=1 TOPOLOGY=1,1,1 SYNC_MODE=explicit PHASE_TIMING=1 \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
bash tests/gpu_validation/run_tgv_256_performance_benchmark.sh
column -t -s $'\t' \
  tests/gpu_validation/out/p4_0_phase_smoke/np1_phase_smoke_phase_summary.tsv
```

Expected: focused tests pass; the summary contains eight rows; `prepare` has 10 samples and each RK phase has 30 samples.

- [ ] **Step 9: Commit phase attribution**

```bash
git add src_gpu/gpu_phase_timing.cuf src_gpu/gpu_runtime.cuf \
  src_gpu/mainloop_gpu.cuf src/CMakeLists.txt \
  tests/gpu_validation/run_tgv_256_performance_benchmark.sh \
  tests/gpu_validation/test_tgv_performance_driver.py \
  tests/gpu_validation/test_benchmark_no_field_io_contract.py
git commit -m "feat(gpu): attribute explicit TGV RK phases"
```

### Task 6: Establish The Local P4-0 Baseline And Gate P4-1

**Files:**
- Modify: `tests/gpu_validation/README.md`
- Create: `documents/ASTR_PHASE_P4_0_BASELINE_REPORT.md`

**Interfaces:**
- Consumes: CPU/GPU production binaries, existing TGV comparison scripts, field-free timing driver, strict phase summaries, and one Nsys trace.
- Produces: an evidence-bounded local baseline and a recorded `GO` or `NO-GO` decision for P4-1.

- [ ] **Step 1: Run focused tests and inspect the numerical diff boundary**

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_benchmark_no_field_io_contract.py \
  tests/gpu_validation/test_tgv_performance_driver.py \
  tests/gpu_validation/test_gpu_phase_timing.py \
  tests/gpu_validation/test_rank_rk_timing.py
git diff --check
git diff --stat
git diff -- src_gpu/solver_gpu.cuf src_gpu/gradcal_gpu.cuf
```

Expected: tests pass, both diff checks print no kernel arithmetic changes, and `git diff --check` prints nothing.

- [ ] **Step 2: Run default-output CPU/GPU field equivalence at 1, 10, and 100 steps**

```bash
for steps in 1 10 100; do
  OUT_DIR="/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_np1_${steps}step" \
  MAXSTEP="$steps" FEQCHKPT="$steps" ATOL=1e-10 RTOL=1e-10 \
  CPU_EXE=/home/dell/workspace/astr_gpu/build_cpu_p4/bin/astr \
  GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr \
  bash tests/gpu_validation/run_tgv_field_compare.sh
done
```

Expected: all three reports contain finite field differences within `1e-10`. Stop on the first violation and classify it before further runs.

- [ ] **Step 3: Run two-rank x, y, and z slab equivalence**

```bash
for topology in 2,1,1 1,2,1 1,1,2; do
  label="${topology//,/_}"
  OUT_DIR="/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_np2_${label}" \
  MAXSTEP=10 FEQCHKPT=10 MPI_NP=2 TOPOLOGY="$topology" \
  ATOL=1e-10 RTOL=1e-10 FILTER_WORKSPACE=full \
  CPU_EXE=/home/dell/workspace/astr_gpu/build_cpu_p4/bin/astr \
  GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr \
  bash tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
done
```

Expected: all three slab comparisons pass within `1e-10`. A failure blocks timing work.

- [ ] **Step 4: Run five timing rounds for NP=1 and all NP=2 slabs**

```bash
BASE=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_256_baseline
GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr

OUT_DIR="$BASE" LABEL=np1 GRID=256,256,256 MAXSTEP=20 DISCARD_STEPS=1 REPEATS=5 \
GPU_EXE="$GPU_EXE" GPU_IDS=0 NP=1 TOPOLOGY=1,1,1 \
SYNC_MODE=explicit PHASE_TIMING=0 HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
bash tests/gpu_validation/run_tgv_256_performance_benchmark.sh

for topology in 2,1,1 1,2,1 1,1,2; do
  label="np2_${topology//,/_}"
  OUT_DIR="$BASE" LABEL="$label" GRID=256,256,256 \
  MAXSTEP=20 DISCARD_STEPS=1 REPEATS=5 \
  GPU_EXE="$GPU_EXE" GPU_IDS=0,1 NP=2 TOPOLOGY="$topology" \
  SYNC_MODE=explicit PHASE_TIMING=0 HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
  bash tests/gpu_validation/run_tgv_256_performance_benchmark.sh
done
```

Expected: every label has five rows and 20 retained slowest-rank RK samples per row. No phase-timing diagnostics are active and no field HDF5 exists below `$BASE`.

- [ ] **Step 5: Run separate phase-attribution rounds for NP=1 and NP=2 x-slab**

```bash
PHASE_BASE=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_256_phases
GPU_EXE=/home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr

OUT_DIR="$PHASE_BASE" LABEL=np1_phase GRID=256,256,256 \
MAXSTEP=20 DISCARD_STEPS=1 REPEATS=5 GPU_EXE="$GPU_EXE" \
GPU_IDS=0 NP=1 TOPOLOGY=1,1,1 SYNC_MODE=explicit PHASE_TIMING=1 \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
bash tests/gpu_validation/run_tgv_256_performance_benchmark.sh

OUT_DIR="$PHASE_BASE" LABEL=np2_x_phase GRID=256,256,256 \
MAXSTEP=20 DISCARD_STEPS=1 REPEATS=5 GPU_EXE="$GPU_EXE" \
GPU_IDS=0,1 NP=2 TOPOLOGY=2,1,1 SYNC_MODE=explicit PHASE_TIMING=1 \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
bash tests/gpu_validation/run_tgv_256_performance_benchmark.sh
```

Expected: each raw phase summary has 105 prepare samples and 315 samples for every three-stage RK phase. These runs diagnose phase cost and are not used in the whole-RK speedup table.

- [ ] **Step 6: Capture one NP=2 x-slab Nsight Systems trace**

```bash
NSYS_OUT=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/p4_0_nsys
mkdir -p "$NSYS_OUT"
python3 tests/gpu_validation/prepare_tgv_case.py \
  --src-case examples/Taylor_Green_Vortex \
  --dst-case "$NSYS_OUT/case" --use-gpu t --grid 256,256,256 \
  --maxstep 2 --feqchkpt 9999 --lfilter t --diffterm t \
  --lreadgrid f --scheme 643e
[[ ! -e "$NSYS_OUT/case/datin/grid.h5" ]] || \
  unlink "$NSYS_OUT/case/datin/grid.h5"
(
  cd "$NSYS_OUT/case"
  CUDA_VISIBLE_DEVICES=0,1 OMPI_MCA_coll_hcoll_enable=0 \
  ASTR_FORCE_MPI_TOPOLOGY=2,1,1 \
  ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  ASTR_GPU_PHASE_TIMING=1 ASTR_GPU_SYNC_MODE=explicit \
  ASTR_GPU_HALO_TRANSPORT=pageable ASTR_GPU_FILTER_WORKSPACE=full \
  nsys profile --trace=cuda,mpi,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite=true -o ../np2_xslab \
    mpirun -np 2 /home/dell/workspace/astr_gpu/build_gpu_p4/bin/astr \
    run datin/input.tgv > ../np2_xslab.log 2>&1
)
find "$NSYS_OUT/case" -type f \
  \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print
```

Expected: `np2_xslab.nsys-rep` exists, the final `find` prints nothing, and the timeline exposes the current serialized pack, D2H, MPI, H2D, and unpack sequence. This is diagnostic evidence, not an overlap claim.

- [ ] **Step 7: Write the baseline report from generated evidence**

Create `documents/ASTR_PHASE_P4_0_BASELINE_REPORT.md` with exactly these sections:

```markdown
# ASTR Phase P4-0 Baseline Report

## Scope And Commit

## Local Hardware And Software

## Benchmark Lifecycle Result

## CPU GPU Field Equivalence

## NP1 And NP2 Slab Timing

## RK Phase Attribution

## Nsight Systems Timeline Evidence

## P4-1 Decision
```

Populate numerical tables only from the generated logs. Set `P4-1 Decision` to `GO` only when every line below is true:

```text
no forbidden field HDF5 in any timing run
five independent retained runs for every topology
20 finite slowest-rank RK samples per retained run
all eight phase labels complete at every expected rank, step, and RK occurrence
NP1 and NP2 x/y/z field comparisons are within 1e-10
no numerical kernel arithmetic changed in P4-0
```

Otherwise write `NO-GO`, the failing command, and the first contradictory log line. Do not start pipeline implementation after `NO-GO`.

- [ ] **Step 8: Document commands and commit verified evidence**

Append the build, correctness, timing, and Nsys commands above to `tests/gpu_validation/README.md`. Label all local scaling numbers as screening-only.

```bash
git diff --check
git status --short
git add tests/gpu_validation/README.md \
  documents/ASTR_PHASE_P4_0_BASELINE_REPORT.md
git commit -m "docs(gpu): establish trusted P4 performance baseline"
```

Expected: only the README and baseline report enter the final documentation commit. Unrelated untracked artifacts remain untracked.

## P4-0 Completion Boundary

P4-0 ends with a `GO` report from Task 6. It does not add CUDA streams, CUDA events, asynchronous copies, MPI request contexts, split interior/boundary kernels, `pinned-pipeline`, or `ASTR_GPU_SYNC_MODE=dependency`. Those changes belong to the separate P4-1/P4-2 plan, written after P4-0 identifies the first exchange family and axis with enough exposed computation to justify overlap.
