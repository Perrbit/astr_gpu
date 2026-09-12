# GPU Mixed-Precision Phase MP3 Characteristic-Flux Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in FP32 characteristic-interface-flux workspace for the all-periodic selective-Roe path while preserving FP64 sensing, reconstruction algebra, MPI transport, RHS accumulation, and RK state.

**Architecture:** Extend the existing exactly-one mixed-candidate selector with `characteristic_flux`. Allocate either the current FP64 five-component workspace or an FP32 replacement, dispatch six periodic candidate kernels, and validate CPU, GPU FP64, and GPU candidate results through the established S0-A6 to S0-A10 matrix. Freeze candidate tolerances from the NP=1 pilot before running the multi-rank matrix.

**Tech Stack:** CUDA Fortran with NVHPC, Fortran MPI, CMake, Bash, Python 3, NumPy, h5py, pytest/unittest, Compute Sanitizer, `nvidia-smi`.

## Global constraints

- Build only through `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Work directly in `feature/gpu_dev`; do not create a worktree.
- Keep `ASTR_GPU_PRECISION_MODE=fp64` as the production default.
- Permit exactly one mixed candidate; do not implement candidate combinations.
- Convert only `flux_characteristic_work_d` storage to FP32 for this candidate.
- Keep the raw Ducros sensor, integer mask, Roe averages, eigenvectors, MP7 reconstruction, `qrhs_d`, RK state, reductions, output, and MPI payloads unchanged.
- Admit only `conschm='543e'`, `recon_schem=3`, `lchardecomp=t`, all-periodic, inviscid, unfiltered, five-equation, single-component RK3 cases.
- Reject physical characteristic boundaries, NSCBC, GCBC, CURVE, diffusion, filtering, chemistry, multispecies, and non-RK3 cases.
- Preserve x/y/z block shapes `(512,1,1)`, `(32,16,1)`, and `(64,1,8)`.
- Retain explicit synchronization after every selected kernel.
- Do not change `src/` CPU numerical behavior. Report a CPU defect for manual decision.
- Do not change MPI tags, counts, neighbor ordering, or transport buffers.
- Do not use FP16, BF16, TF32, Tensor Cores, fast math, or compact schemes.
- Do not run Git staging, commit, or push commands without explicit user authorization; commit steps below are review boundaries only.

---

## File map

| File | Responsibility |
| --- | --- |
| `src_gpu/mixed_candidate_gpu.cuf` | Parse, collectively validate, name, and report `characteristic_flux` |
| `src_gpu/commarray_gpu.cuf` | Enforce eligibility; own mutually exclusive FP32/FP64 allocation and byte logs |
| `src_gpu/solver_gpu.cuf` | Provide three FP32 final-store writers and three FP64-accumulating readers |
| `src_gpu/mainloop_gpu.cuf` | Select the periodic FP32 kernel path and preserve explicit synchronization |
| `tests/gpu_validation/test_mixed_precision_mp3_contract.py` | Lock selector, eligibility, precision boundary, dispatch, driver, and documentation contracts |
| `tests/gpu_validation/run_mp3_characteristic_flux_compare.sh` | Run one CPU/GPU-FP64/GPU-candidate characteristic comparison |
| `tests/gpu_validation/freeze_mp3_tolerances.py` | Convert NP=1 pilot evidence into capped immutable field/statistics tolerances |
| `tests/gpu_validation/test_freeze_mp3_tolerances.py` | Unit-test tolerance extraction, rounding, caps, and malformed reports |
| `tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh` | Run S0-A7 through S0-A10 with the frozen pilot tolerances |
| `tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh` | Enforce zero invalid-memory errors for the candidate |
| `tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh` | Collect five interleaved complete-RK timing and memory samples |
| `tests/gpu_validation/summarize_mp2_benchmark.py` | Generalize the existing byte/timing report title through an optional phase argument |
| `tests/gpu_validation/test_summarize_mp2_benchmark.py` | Preserve MP2 output and verify MP3 title selection |
| `tests/gpu_validation/README.md` | Document commands, evidence, limits, and rollback |
| `documents/GPU_VALIDATION_MATRIX.md` | Record MP3 numerical, MPI, sanitizer, memory, and timing gates |
| `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md` | Record local classification and next promotion dependency |
| `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md` | Synchronize the top-level mixed-precision roadmap |

### Task 1: Selector and eligibility contract

**Files:**
- Create: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`
- Modify: `tests/gpu_validation/mixed_candidate_setup_test.cuf`
- Modify: `src_gpu/mixed_candidate_gpu.cuf`
- Modify: `src_gpu/commarray_gpu.cuf`

**Interfaces:**
- Consumes: `precision_mode_gpu::gpu_mixed_workspace_requested() -> logical`
- Produces: `GPU_MIXED_CHARACTERISTIC_FLUX = 3`
- Produces: `gpu_mixed_characteristic_flux_workspace_enabled() -> logical`
- Preserves: `gpu_mixed_candidate_requested(candidate_id) -> logical`

- [x] **Step 1: Write failing selector and eligibility tests.**

Create assertions that require the new enum, accepted string, exported query,
strict eligibility terms, and hard error behavior:

```python
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


CANDIDATE = read(ROOT / "src_gpu" / "mixed_candidate_gpu.cuf")
COMMARRAY = read(ROOT / "src_gpu" / "commarray_gpu.cuf")
SOLVER = read(ROOT / "src_gpu" / "solver_gpu.cuf")
MAINLOOP = read(ROOT / "src_gpu" / "mainloop_gpu.cuf")


class MixedPrecisionMp3Contract(unittest.TestCase):
    def test_selector_accepts_characteristic_flux_as_fourth_candidate(self):
        compact = CANDIDATE.replace(" ", "").lower()
        self.assertIn("gpu_mixed_characteristic_flux=3", compact)
        self.assertIn("case('characteristic_flux')", compact)
        self.assertIn("gpu_mixed_candidate_name='characteristic_flux'", compact)

    def test_characteristic_flux_has_a_strict_eligibility_gate(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn("logicalfunctiongpu_mixed_characteristic_flux_workspace_enabled", compact)
        self.assertIn("gpu_mixed_characteristic_flux", compact)
        for term in (
            "trim(conschm)=='543e'", "recon_schem==3", "lchardecomp",
            "lihomo.and.ljhomo.and.lkhomo", ".not.diffterm", ".not.lfilter",
            "numq==5", "num_species==0", "num_modequ==0", "trim(rkscheme)=='rk3'",
        ):
            self.assertIn(term, compact)
        self.assertIn("requestedmixedcandidateisineligible", compact)
        self.assertIn("mpi_abort", compact)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the new contract test and verify RED.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
```

Expected: FAIL because `GPU_MIXED_CHARACTERISTIC_FLUX`, the selector branch,
and the workspace eligibility query do not exist.

- [x] **Step 3: Extend the candidate selector.**

Add the enum and selector/name branches without changing the existing default:

```fortran
integer,parameter :: GPU_MIXED_CHARACTERISTIC_FLUX=3

case('characteristic_flux')
  choice=GPU_MIXED_CHARACTERISTIC_FLUX

case(GPU_MIXED_CHARACTERISTIC_FLUX)
  gpu_mixed_candidate_name='characteristic_flux'
```

Update both diagnostic strings so they list all four accepted values. Treat an
explicit `characteristic_flux` under FP64 exactly like the existing explicit
non-default candidates and abort collectively.

- [x] **Step 4: Add the strict candidate query and eligibility branch.**

Implement the query in `commarray_gpu`:

```fortran
logical function gpu_mixed_characteristic_flux_workspace_enabled()
  use commvar, only: conschm,recon_schem,lchardecomp,lihomo,ljhomo,lkhomo, &
       diffterm,lfilter,numq,num_species,num_modequ,rkscheme
  use precision_mode_gpu, only: gpu_mixed_workspace_requested
  use mixed_candidate_gpu, only: gpu_mixed_candidate_requested, &
       GPU_MIXED_CHARACTERISTIC_FLUX
  implicit none

  gpu_mixed_characteristic_flux_workspace_enabled = &
       gpu_mixed_workspace_requested() .and. &
       gpu_mixed_candidate_requested(GPU_MIXED_CHARACTERISTIC_FLUX) .and. &
       trim(conschm)=='543e' .and. recon_schem==3 .and. lchardecomp .and. &
       lihomo .and. ljhomo .and. lkhomo .and. &
       (.not.diffterm) .and. (.not.lfilter) .and. &
       numq==5 .and. num_species==0 .and. num_modequ==0 .and. &
       trim(rkscheme)=='rk3'
end function gpu_mixed_characteristic_flux_workspace_enabled
```

In `validate_gpu_mixed_candidate_eligibility`, return only when this complete
query is true. Otherwise retain the shared fatal path and report
`requested mixed candidate is ineligible for this case`.

- [x] **Step 5: Extend the runtime probe and run selector tests.**

Import and report the fourth enum:

```fortran
use mixed_candidate_gpu, only: configure_gpu_mixed_candidate, &
     gpu_mixed_candidate_id,gpu_mixed_candidate_name, &
     gpu_mixed_candidate_requested,GPU_MIXED_FLUX, &
     GPU_MIXED_CHARACTERISTIC_FLUX

if(rank==0) write(*,'(A,L1)') ' characteristic_flux=', &
     gpu_mixed_candidate_requested(GPU_MIXED_CHARACTERISTIC_FLUX)
```

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp2_contract tests.gpu_validation.test_mixed_precision_mp3_contract -v
```

Expected: selector/eligibility tests PASS; array and kernel tests in the new
suite remain RED for Tasks 2 and 3.

- [x] **Step 6: Review boundary.**

Review only selector and eligibility diffs. Do not stage or commit unless the
user explicitly authorizes Git operations. Suggested commit if authorized:

```bash
git add src_gpu/mixed_candidate_gpu.cuf src_gpu/commarray_gpu.cuf tests/gpu_validation/mixed_candidate_setup_test.cuf tests/gpu_validation/test_mixed_precision_mp3_contract.py
git commit -m "feat(gpu): add MP3 characteristic flux candidate gate"
```

### Task 2: Mutually exclusive workspace allocation

**Files:**
- Modify: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`
- Modify: `src_gpu/commarray_gpu.cuf`

**Interfaces:**
- Consumes: `gpu_mixed_characteristic_flux_workspace_enabled() -> logical`
- Produces: `real(4), allocatable, device :: flux_characteristic_work_sp_d(:,:,:,:)`
- Produces logs: `ASTR_GPU_ACTIVE_MIXED_WORKSPACE`, `ASTR_GPU_MIXED_WORKSPACE_BYTES`, `ASTR_GPU_FP64_WORKSPACE_BYTES`

- [x] **Step 1: Add failing allocation and byte-accounting tests.**

```python
def test_characteristic_workspace_allocations_are_mutually_exclusive(self):
    compact = COMMARRAY.replace(" ", "").lower()
    self.assertIn("real(4),allocatable,device::flux_characteristic_work_sp_d(:,:,:,:)", compact)
    self.assertIn("if(characteristic_active)then", compact)
    self.assertIn("allocate(flux_characteristic_work_sp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:5))", compact)
    self.assertIn("allocate(flux_characteristic_work_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:5))", compact)
    self.assertIn("halo_points*5_8*4_8", compact)
    self.assertIn("halo_points*5_8*8_8", compact)

def test_release_deallocates_both_characteristic_workspace_types(self):
    lower = COMMARRAY.lower()
    self.assertIn("if(allocated(flux_characteristic_work_sp_d)) deallocate(flux_characteristic_work_sp_d)", lower)
    self.assertIn("if(allocated(flux_characteristic_work_d)) deallocate(flux_characteristic_work_d)", lower)
```

- [x] **Step 2: Run the focused tests and verify RED.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
```

Expected: allocation, byte-accounting, and release assertions FAIL.

- [x] **Step 3: Implement mutually exclusive allocation.**

Declare the FP32 array beside the FP64 array. In `alloc_gpu_arrays`, compute
`characteristic_active` once after selector validation and replace the current
unconditional characteristic allocation with:

```fortran
if(lchardecomp) then
  if(characteristic_active) then
    if(.not.allocated(flux_characteristic_work_sp_d)) &
      allocate(flux_characteristic_work_sp_d( &
        -hm:im+hm,-hm:jm+hm,-hm:km+hm,1:5))
    mixed_workspace_bytes=halo_points*5_8*4_8
    fp64_workspace_bytes=halo_points*5_8*8_8
    active_workspace='characteristic_flux'
  else
    if(.not.allocated(flux_characteristic_work_d)) &
      allocate(flux_characteristic_work_d( &
        -hm:im+hm,-hm:jm+hm,-hm:km+hm,1:5))
  endif
endif
```

Move the three existing workspace provenance writes after this allocation so
they report the final active candidate. Add FP32 deallocation immediately
before FP64 characteristic-workspace deallocation.

- [x] **Step 4: Run allocation contracts.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp2_contract tests.gpu_validation.test_mixed_precision_mp3_contract -v
```

Expected: selector and allocation tests PASS; kernel/driver tests remain RED.

- [x] **Step 5: Review boundary.**

Confirm that candidate mode never allocates the FP64 mirror and that FP64 mode
never allocates the FP32 array. Suggested commit if authorized:

```bash
git add src_gpu/commarray_gpu.cuf tests/gpu_validation/test_mixed_precision_mp3_contract.py
git commit -m "feat(gpu): allocate FP32 characteristic flux workspace"
```

### Task 3: Periodic FP32 writer and reader kernels

**Files:**
- Modify: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`
- Modify: `src_gpu/solver_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`

**Interfaces:**
- Consumes: `flux_characteristic_work_sp_d(:,:,:,:)` and FP64 `characteristic_reconstruction_interface_flux(i,j,k,idir,im,jm,km,reconstruction_scheme,gamma,hm,npdc,physical_dir,fh)`
- Produces: six `_sp_kernel` entry points for all-periodic x/y/z dispatch
- Preserves: current physical and xy-physical FP64 kernel entry points

- [x] **Step 1: Add failing kernel and dispatch tests.**

```python
def test_periodic_characteristic_fp32_kernels_exist(self):
    lower = SOLVER.lower()
    for axis in "xyz":
        self.assertIn(f"subroutine characteristic_upwind_flux_{axis}_global_sp_kernel", lower)
        self.assertIn(f"subroutine characteristic_upwind_rhs_{axis}_global_sp_kernel", lower)
    compact = SOLVER.replace(" ", "").lower()
    self.assertGreaterEqual(compact.count("real(8)::fh(5)"), 3)
    self.assertGreaterEqual(compact.count("flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)"), 3)
    self.assertGreaterEqual(compact.count("real(flux_characteristic_work_sp_d("), 6)

def test_mainloop_dispatches_only_periodic_characteristic_sp_kernels(self):
    compact = MAINLOOP.replace(" ", "").lower()
    self.assertIn("mixed_characteristic_flux_workspace", compact)
    for axis in "xyz":
        for kind in ("flux", "rhs"):
            name = f"characteristic_upwind_{kind}_{axis}_global_sp_kernel"
            self.assertIn(f"call{name}<<<", compact)
            self.assertIn(f"sync_after_kernel('{name}')", compact)
    self.assertNotIn("characteristic_upwind_flux_x_physical_global_sp_kernel", compact)
    self.assertNotIn("characteristic_upwind_flux_x_xyphysical_global_sp_kernel", compact)
```

- [x] **Step 2: Run the focused tests and verify RED.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
```

Expected: six kernel-name and dispatch assertions FAIL.

- [x] **Step 3: Add the three FP32 final-store writers.**

Duplicate only the three all-periodic writers. Preserve their index ranges and
the FP64 reconstruction call. The x pattern is:

```fortran
attributes(global) subroutine characteristic_upwind_flux_x_global_sp_kernel( &
     im,jm,km,reconstruction_scheme,gamma)
  use commarray_gpu, only: flux_characteristic_work_sp_d
  implicit none
  integer,value :: im,jm,km,reconstruction_scheme
  real(8),value :: gamma
  integer :: i,j,k,m
  real(8) :: fh(5)

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-2
  j=blockIdx%y-1
  k=blockIdx%z-1
  if(i < -1 .or. i > im .or. j < 0 .or. j > jm .or. k < 0 .or. k > km) return
  call characteristic_reconstruction_interface_flux( &
       i,j,k,1,im,jm,km,reconstruction_scheme,gamma,0,0,0,fh)
  do m=1,5
    flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)
  enddo
end subroutine characteristic_upwind_flux_x_global_sp_kernel
```

Add the y writer with its existing interface range and direction:

```fortran
attributes(global) subroutine characteristic_upwind_flux_y_global_sp_kernel( &
     im,jm,km,reconstruction_scheme,gamma)
  use commarray_gpu, only: flux_characteristic_work_sp_d
  implicit none
  integer,value :: im,jm,km,reconstruction_scheme
  real(8),value :: gamma
  integer :: i,j,k,m
  real(8) :: fh(5)

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
  j=(blockIdx%y-1)*blockDim%y+threadIdx%y-2
  k=blockIdx%z-1
  if(i < 0 .or. i > im .or. j < -1 .or. j > jm .or. k < 0 .or. k > km) return
  call characteristic_reconstruction_interface_flux( &
       i,j,k,2,im,jm,km,reconstruction_scheme,gamma,0,0,0,fh)
  do m=1,5
    flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)
  enddo
end subroutine characteristic_upwind_flux_y_global_sp_kernel
```

Add the z writer with its existing interface range and direction:

```fortran
attributes(global) subroutine characteristic_upwind_flux_z_global_sp_kernel( &
     im,jm,km,reconstruction_scheme,gamma)
  use commarray_gpu, only: flux_characteristic_work_sp_d
  implicit none
  integer,value :: im,jm,km,reconstruction_scheme
  real(8),value :: gamma
  integer :: i,j,k,m
  real(8) :: fh(5)

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
  j=blockIdx%y-1
  k=(blockIdx%z-1)*blockDim%z+threadIdx%z-2
  if(i < 0 .or. i > im .or. j < 0 .or. j > jm .or. k < -1 .or. k > km) return
  call characteristic_reconstruction_interface_flux( &
       i,j,k,3,im,jm,km,reconstruction_scheme,gamma,0,0,0,fh)
  do m=1,5
    flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)
  enddo
end subroutine characteristic_upwind_flux_z_global_sp_kernel
```

Do not duplicate or modify `characteristic_reconstruction_interface_flux`.

- [x] **Step 4: Add the three FP64-accumulating readers.**

Add only periodic RHS kernels. Promote both operands before subtraction. The
complete x reader is:

```fortran
attributes(global) subroutine characteristic_upwind_rhs_x_global_sp_kernel(im,jm,km)
  use commarray_gpu, only: flux_characteristic_work_sp_d,qrhs_d
  implicit none
  integer,value :: im,jm,km
  integer :: i,j,k,m

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
  j=blockIdx%y-1
  k=blockIdx%z-1
  if(i < 0 .or. i > im .or. j < 0 .or. j > jm .or. k < 0 .or. k > km) return
  do m=1,5
    qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
         (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
          real(flux_characteristic_work_sp_d(i-1,j,k,m),8))
  enddo
end subroutine characteristic_upwind_rhs_x_global_sp_kernel
```

Implement the complete y and z readers:

```fortran
attributes(global) subroutine characteristic_upwind_rhs_y_global_sp_kernel(im,jm,km)
  use commarray_gpu, only: flux_characteristic_work_sp_d,qrhs_d
  implicit none
  integer,value :: im,jm,km
  integer :: i,j,k,m

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
  j=(blockIdx%y-1)*blockDim%y+threadIdx%y-1
  k=blockIdx%z-1
  if(i < 0 .or. i > im .or. j < 0 .or. j > jm .or. k < 0 .or. k > km) return
  do m=1,5
    qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
         (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
          real(flux_characteristic_work_sp_d(i,j-1,k,m),8))
  enddo
end subroutine characteristic_upwind_rhs_y_global_sp_kernel

attributes(global) subroutine characteristic_upwind_rhs_z_global_sp_kernel(im,jm,km)
  use commarray_gpu, only: flux_characteristic_work_sp_d,qrhs_d
  implicit none
  integer,value :: im,jm,km
  integer :: i,j,k,m

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
  j=blockIdx%y-1
  k=(blockIdx%z-1)*blockDim%z+threadIdx%z-1
  if(i < 0 .or. i > im .or. j < 0 .or. j > jm .or. k < 0 .or. k > km) return
  do m=1,5
    qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
         (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
          real(flux_characteristic_work_sp_d(i,j,k-1,m),8))
  enddo
end subroutine characteristic_upwind_rhs_z_global_sp_kernel
```

- [x] **Step 5: Dispatch candidate kernels only in the periodic branch.**

Import the query and six kernels, compute one host-side flag before the RK
loop, and branch inside each existing all-periodic x/y/z section:

```fortran
mixed_characteristic_flux_workspace = &
     gpu_mixed_characteristic_flux_workspace_enabled()

if(mixed_characteristic_flux_workspace) then
  call characteristic_upwind_flux_x_global_sp_kernel<<< &
       grid_x_interface,block_x>>>(im,jm,km,recon_schem,gamma)
  call sync_after_kernel('characteristic_upwind_flux_x_global_sp_kernel')
  call characteristic_upwind_rhs_x_global_sp_kernel<<<grid_x,block_x>>>(im,jm,km)
  call sync_after_kernel('characteristic_upwind_rhs_x_global_sp_kernel')
else
  call characteristic_upwind_flux_x_global_kernel<<< &
       grid_x_interface,block_x>>>(im,jm,km,recon_schem,gamma)
  call sync_after_kernel('characteristic_upwind_flux_x_global_kernel')
  call characteristic_upwind_rhs_x_global_kernel<<<grid_x,block_x>>>(im,jm,km)
  call sync_after_kernel('characteristic_upwind_rhs_x_global_kernel')
endif
```

Use these explicit y and z branches after the x branch:

```fortran
if(mixed_characteristic_flux_workspace) then
  call characteristic_upwind_flux_y_global_sp_kernel<<< &
       grid_y_interface,block_y>>>(im,jm,km,recon_schem,gamma)
  call sync_after_kernel('characteristic_upwind_flux_y_global_sp_kernel')
  call characteristic_upwind_rhs_y_global_sp_kernel<<<grid_y,block_y>>>(im,jm,km)
  call sync_after_kernel('characteristic_upwind_rhs_y_global_sp_kernel')
else
  call characteristic_upwind_flux_y_global_kernel<<< &
       grid_y_interface,block_y>>>(im,jm,km,recon_schem,gamma)
  call sync_after_kernel('characteristic_upwind_flux_y_global_kernel')
  call characteristic_upwind_rhs_y_global_kernel<<<grid_y,block_y>>>(im,jm,km)
  call sync_after_kernel('characteristic_upwind_rhs_y_global_kernel')
endif
if(mixed_characteristic_flux_workspace) then
  call characteristic_upwind_flux_z_global_sp_kernel<<< &
       grid_z_interface,block_z>>>(im,jm,km,recon_schem,gamma)
  call sync_after_kernel('characteristic_upwind_flux_z_global_sp_kernel')
  call characteristic_upwind_rhs_z_global_sp_kernel<<<grid_z,block_z>>>(im,jm,km)
  call sync_after_kernel('characteristic_upwind_rhs_z_global_sp_kernel')
else
  call characteristic_upwind_flux_z_global_kernel<<< &
       grid_z_interface,block_z>>>(im,jm,km,recon_schem,gamma)
  call sync_after_kernel('characteristic_upwind_flux_z_global_kernel')
  call characteristic_upwind_rhs_z_global_kernel<<<grid_z,block_z>>>(im,jm,km)
  call sync_after_kernel('characteristic_upwind_rhs_z_global_kernel')
endif
```

Leave physical and xy-physical branches byte-for-byte unchanged.

- [x] **Step 6: Run source contracts and build.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp2_contract tests.gpu_validation.test_mixed_precision_mp3_contract -v
cmake -S . -B build_gpu_probe -DASTR_ENABLE_CUDA_FORTRAN=ON
cmake --build build_gpu_probe --target mixed_candidate_setup_test astr -j2
```

Expected: all source contracts PASS and both NVHPC targets build successfully.
If compilation exposes a logic or type mismatch, stop and report it instead of
running numerical cases.

- [x] **Step 7: Exercise the runtime selector.**

Run the probe with default, explicit FP64, valid candidate, invalid candidate,
and a rank-inconsistent candidate. Expected results:

```text
default: mixed_candidate=flux
valid: mixed_candidate=characteristic_flux id=3 characteristic_flux=T
invalid: nonzero exit with accepted-value diagnostic
FP64 mismatch: nonzero exit requiring mixed_workspace
rank mismatch: collective nonzero exit
```

- [x] **Step 8: Review boundary.**

Confirm that only six periodic kernels were added, all math before the final
store remains FP64, and all launches synchronize. Suggested commit if
authorized:

```bash
git add src_gpu/solver_gpu.cuf src_gpu/mainloop_gpu.cuf tests/gpu_validation/test_mixed_precision_mp3_contract.py
git commit -m "feat(gpu): add mixed characteristic flux kernels"
```

### Task 4: Three-reference pilot and frozen tolerances

**Files:**
- Modify: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_compare.sh`
- Create: `tests/gpu_validation/freeze_mp3_tolerances.py`
- Create: `tests/gpu_validation/test_freeze_mp3_tolerances.py`

**Interfaces:**
- Consumes environment: `NP`, `TOPOLOGY`, `GRID`, `MAXSTEP`, `SHOCK_X`, `RANKWISE`, candidate tolerances
- Produces: CPU versus GPU-FP64 reports and GPU-FP64 versus candidate reports
- Produces: `mp3_tolerances.env` with frozen absolute tolerances and zero relative tolerances

- [x] **Step 1: Add failing driver and tolerance-tool contracts.**

Require the driver to create `cpu`, `gpu_fp64`, and `gpu_characteristic_flux`
runs; assert exact candidate sensor comparison and provenance checks:

```python
COMPARE_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_compare.sh"
)


def test_compare_driver_has_three_references_and_exact_candidate_sensor_gate(self):
    text = COMPARE_DRIVER
    for target in ("cpu", "gpu_fp64", "gpu_characteristic_flux"):
        self.assertIn(target, text)
    self.assertIn("ASTR_GPU_PRECISION_MODE=\"$precision\"", text)
    self.assertIn("ASTR_GPU_MIXED_CANDIDATE=\"$candidate\"", text)
    self.assertIn("--atol 0 --rtol 0", text)
    self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux", text)
    self.assertIn("refusing to overwrite", text)
```

Unit-test tolerance freezing with synthetic field and statistics reports. The
algorithm must use `10 * observed_max`, round upward on the `1-2-5`
engineering sequence, apply a floor of `1e-12`, and reject values above `1e-5`
for both fields and statistics.

- [x] **Step 2: Run tests and verify RED.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract tests.gpu_validation.test_freeze_mp3_tolerances -v
```

Expected: FAIL because the driver and tolerance tool do not exist.

- [x] **Step 3: Implement the reusable three-reference driver.**

Use the current S0-A6 preparation arguments exactly:

```bash
--flowtype shuosher --homogeneous t,t,t --bctype 1,1,1,1,1,1 \
--lfilter f --diffterm f --scheme 643e --conschm 543e \
--difschm 643e --recon-schem 3 --lchardecomp t
```

Run CPU without GPU precision variables. Run GPU FP64 with
`ASTR_GPU_PRECISION_MODE=fp64`. Run the candidate with:

```bash
ASTR_GPU_PRECISION_MODE=mixed_workspace \
ASTR_GPU_MIXED_CANDIDATE=characteristic_flux \
ASTR_GPU_SYNC_MODE=explicit
```

For CPU versus GPU FP64, retain the existing sensor, field, and statistics
tolerances. For GPU FP64 versus candidate, compare sensors with `--atol 0
--rtol 0`, then compare field and statistics with candidate-specific absolute
tolerances and zero relative tolerance. Use `--rankwise` only when
`RANKWISE=t`.

Implement `CALIBRATE=t` as evidence collection, not acceptance: force only the
candidate field/statistics comparison tolerances to `1.0` while keeping sensor
and mask tolerances exactly zero. The subsequent freezing tool rejects field
or statistics tolerances above the approved common `1e-5` ceiling.

- [x] **Step 4: Implement deterministic tolerance freezing.**

Expose these pure functions for unit testing:

```python
FIELD_PATTERN = re.compile(r"\blinf=([+\-0-9.eE]+)")


def read_field_max(path: Path) -> float:
    values = [
        float(match.group(1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := FIELD_PATTERN.search(line)) is not None
    ]
    if not values:
        raise ValueError(f"{path}: no field linf values")
    return max(values)


def read_stats_max(path: Path) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(
            "metric max_abs max_rel final_cpu final_gpu final_abs final_rel"
        ) + 1
    except ValueError as exc:
        raise ValueError(f"{path}: missing statistics metric table") from exc
    values: list[float] = []
    for line in lines[start:]:
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "first_failure:":
            break
        if len(fields) != 7:
            raise ValueError(f"{path}: malformed statistics row: {line}")
        values.append(float(fields[1]))
    if not values:
        raise ValueError(f"{path}: no statistics max_abs values")
    return max(values)


def frozen_tolerance(observed: float, floor: float, ceiling: float) -> float:
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError("observed error must be finite and nonnegative")
    target = max(floor, 10.0 * observed)
    exponent = math.floor(math.log10(target))
    decade = 10.0**exponent
    scaled = target / decade
    frozen = 10.0 * decade
    for mantissa in (1.0, 2.0, 5.0, 10.0):
        if scaled <= mantissa * (1.0 + 1.0e-12):
            frozen = mantissa * decade
            break
    if frozen > ceiling:
        raise ValueError(
            f"frozen tolerance {frozen:.1e} exceeds ceiling {ceiling:.1e}"
        )
    return frozen
```

`frozen_tolerance` must reject non-finite/negative observations, calculate
`target=max(floor, 10.0*observed)`, round upward to the first value in
`(1, 2, 5, 10) * 10**floor(log10(target))`, and raise `ValueError` when the
result exceeds the ceiling. Write ASCII output:

```text
CANDIDATE_FIELD_ATOL=<frozen value>
CANDIDATE_FIELD_RTOL=0
CANDIDATE_STATS_ATOL=<frozen value>
CANDIDATE_STATS_RTOL=0
```

- [x] **Step 5: Run unit and shell syntax tests.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_freeze_mp3_tolerances tests.gpu_validation.test_mixed_precision_mp3_contract -v
bash -n tests/gpu_validation/run_mp3_characteristic_flux_compare.sh
```

Expected: PASS.

- [x] **Step 6: Run the S0-A6 pilot and freeze thresholds.** Pilot completed
  with exact sensor/mask equality, field maximum `1.5973888878306752e-7`, and
  statistics maximum `1.0126266403176487e-7`. The original decade-rounding
  rule was rejected as too coarse. The approved `1-2-5` rule freezes both
  tolerances at `2e-6` under a common `1e-5` ceiling.

Use three steps to expose accumulation while retaining a short local gate:

```bash
OUT_DIR=tests/gpu_validation/out/mp3_characteristic_flux_pilot_v2 \
NP=1 TOPOLOGY=1,1,1 GRID=400,8,8 MAXSTEP=3 CALIBRATE=t \
tests/gpu_validation/run_mp3_characteristic_flux_compare.sh

python3 tests/gpu_validation/freeze_mp3_tolerances.py \
  --field-report tests/gpu_validation/out/mp3_characteristic_flux_pilot_v2/gpu_fp64_vs_candidate_flowfield.txt \
  --stats-report tests/gpu_validation/out/mp3_characteristic_flux_pilot_v2/gpu_fp64_vs_candidate_flowstate.txt \
  --output tests/gpu_validation/out/mp3_characteristic_flux_pilot_v2/mp3_tolerances.env
```

Expected: raw sensor maximum difference `0`, mask mismatches `0`, finite field
and statistics differences, and a frozen file within the fixed ceilings. Stop
for user review if any ceiling is exceeded.

- [x] **Step 7: Re-run S0-A6 as a real gate with frozen tolerances.**

Run in a new non-overwriting evidence directory:

```bash
source tests/gpu_validation/out/mp3_characteristic_flux_pilot_v2/mp3_tolerances.env
OUT_DIR=tests/gpu_validation/out/mp3_characteristic_flux_s0a6 \
NP=1 TOPOLOGY=1,1,1 GRID=400,8,8 MAXSTEP=3 \
CANDIDATE_FIELD_ATOL="$CANDIDATE_FIELD_ATOL" CANDIDATE_FIELD_RTOL=0 \
CANDIDATE_STATS_ATOL="$CANDIDATE_STATS_ATOL" CANDIDATE_STATS_RTOL=0 \
tests/gpu_validation/run_mp3_characteristic_flux_compare.sh
```

Expected: CPU/GPU FP64 baseline PASS; GPU FP64/candidate exact sensor and mask
PASS; candidate field/statistics PASS.

- [x] **Step 8: Review boundary.**

Record the frozen values before continuing. Never regenerate them in response
to a later MPI failure. Suggested commit if authorized:

```bash
git add tests/gpu_validation/test_mixed_precision_mp3_contract.py tests/gpu_validation/run_mp3_characteristic_flux_compare.sh tests/gpu_validation/freeze_mp3_tolerances.py tests/gpu_validation/test_freeze_mp3_tolerances.py
git commit -m "test(gpu): add MP3 characteristic flux pilot gate"
```

### Task 5: MPI, sanitizer, and benchmark drivers

**Files:**
- Modify: `tests/gpu_validation/test_mixed_precision_mp3_contract.py`
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh`
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh`
- Create: `tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh`
- Modify: `tests/gpu_validation/summarize_mp2_benchmark.py`
- Modify: `tests/gpu_validation/test_summarize_mp2_benchmark.py`

**Interfaces:**
- Consumes: frozen `mp3_tolerances.env` and the Task 4 comparison driver
- Produces: S0-A7/A8/A9/A10 evidence directories, memcheck log, timing TSVs, and `summary.md`

- [x] **Step 1: Add failing MPI, sanitizer, and benchmark contracts.**

Assert that the matrix contains these exact cases:

```python
MPI_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_mpi_matrix.sh"
)
MEMCHECK_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_memcheck.sh"
)
BENCHMARK_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_benchmark.sh"
)


expected = (
    "NP=2 TOPOLOGY=2,1,1 GRID=400,8,8 SHOCK_X=0.d0 RANKWISE=f",
    "NP=2 TOPOLOGY=1,2,1 GRID=400,16,8 RANKWISE=t",
    "NP=2 TOPOLOGY=1,1,2 GRID=400,8,16 RANKWISE=t",
    "NP=8 TOPOLOGY=2,2,2 GRID=400,16,16 RANKWISE=t",
)
for line in expected:
    self.assertIn(line, MPI_DRIVER)
```

Also require `compute-sanitizer --tool memcheck`, `--error-exitcode`, `ERROR
SUMMARY: 0 errors`, at least five interleaved repeats, `ASTR_GPU_RK_TIMING=1`,
analytical `halo_points*5` accounting, and no `MIN_SPEEDUP` gate.

- [x] **Step 2: Run contracts and verify RED.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract tests.gpu_validation.test_summarize_mp2_benchmark -v
```

Expected: new driver and phase-title assertions FAIL.

- [x] **Step 3: Implement the fixed MPI matrix.**

Require `TOLERANCE_FILE`, source it once, refuse an existing output directory,
and call the Task 4 driver with `MAXSTEP=3` for each exact S0-A7 to S0-A10
combination. Do not regenerate tolerances in this script. Use one subdirectory
per gate so one failure leaves earlier evidence intact.

- [x] **Step 4: Implement the candidate memcheck.**

Prepare a reduced all-periodic characteristic case and run:

```bash
ASTR_GPU_PRECISION_MODE=mixed_workspace \
ASTR_GPU_MIXED_CANDIDATE=characteristic_flux \
ASTR_GPU_SYNC_MODE=explicit \
mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$GPU_EXE" run datin/input.shuosher
```

Require both `The job is done!` and `ERROR SUMMARY: 0 errors` in the log.

- [x] **Step 5: Generalize the existing benchmark summary title.**

Add a `phase: str = "MP2"` argument to `summarize`, validate it against
`{"MP2", "MP3"}`, add `--phase` with default `MP2`, and generate:

```python
f"# {phase} {candidate_name} Workspace Benchmark"
```

Existing MP2 calls and tests must retain their output unchanged.

- [x] **Step 6: Implement the interleaved characteristic benchmark.**

Base it on `run_mp2_derivative_benchmark.sh`, but prepare Shu-Osher with
`lchardecomp=t`, `recon_schem=3`, no diffusion, and no filter. Use warmups and
five or more alternating FP64/candidate repeats. Compute:

```bash
workspace_elements="$(python3 -c "i,j,k=map(int,'$GRID'.split(',')); h=5; print((i+2*h+1)*(j+2*h+1)*(k+2*h+1)*5)")"
expected_mixed_bytes="$((workspace_elements * 4))"
expected_fp64_bytes="$((workspace_elements * 8))"
```

Call the summary tool with `--phase MP3`. Record complete-RK median, spread,
sampled peak memory, utilization, exact byte reduction, and descriptive
speedup. Do not impose a speedup acceptance threshold.

- [x] **Step 7: Run tests and shell syntax checks.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract tests.gpu_validation.test_summarize_mp2_benchmark -v
bash -n tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh
bash -n tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh
bash -n tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh
```

Expected: PASS.

- [x] **Step 8: Review boundary.**

Confirm the MPI driver cannot calibrate, the benchmark has no speedup gate,
and the expected bytes represent five haloed components. Suggested commit if
authorized:

```bash
git add tests/gpu_validation/test_mixed_precision_mp3_contract.py tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh tests/gpu_validation/summarize_mp2_benchmark.py tests/gpu_validation/test_summarize_mp2_benchmark.py
git commit -m "test(gpu): add MP3 MPI safety and timing gates"
```

### Task 6: Execute the complete local evidence matrix

**Files:**
- No source changes unless a reproducible implementation defect is diagnosed
- Produce ignored output under `tests/gpu_validation/out/`

**Interfaces:**
- Consumes: built GPU executable and frozen pilot tolerance file
- Produces: numerical, MPI, sanitizer, memory, utilization, and timing evidence

- [ ] **Step 1: Run the full contract suite.** The 2026-09-12 run completed
  260 tests with one unrelated failure: `test_halo_endpoint_upload_contract`
  still expects 22 receive uploads while the current source contains 28. All
  MP3 tests passed.

Run:

```bash
python3 -m unittest discover -s tests/gpu_validation -p 'test_*.py' -v
```

Expected: all tests PASS. If a pre-existing unrelated test fails, record it
separately and do not claim a clean suite.

- [x] **Step 2: Reconfigure and build from the root project.**

Run:

```bash
cmake -S . -B build_gpu_probe -DASTR_ENABLE_CUDA_FORTRAN=ON
cmake --build build_gpu_probe --target mixed_candidate_setup_test astr -j2
```

Expected: successful NVHPC build. Stop if source logic or type errors remain.

- [x] **Step 3: Run S0-A7 through S0-A10.** All four gates passed under the
  frozen `2e-6` field/statistics tolerances. GPU FP64 versus candidate raw
  sensors were bitwise identical and every mask mismatch count was zero.

Run:

```bash
TOLERANCE_FILE=tests/gpu_validation/out/mp3_characteristic_flux_pilot_v2/mp3_tolerances.env \
OUT_DIR=tests/gpu_validation/out/mp3_characteristic_flux_mpi_matrix_v2 \
tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh
```

Expected: all four gates PASS with exact GPU-FP64/candidate sensor and mask.
The NP=8 result is correctness evidence only because local ranks share two
GPUs.

- [x] **Step 4: Run Compute Sanitizer.** The corrected `ob1/self/pt2pt`
  isolation run completed with `ERROR SUMMARY: 0 errors` at
  `tests/gpu_validation/out/mp3_characteristic_flux_memcheck_v2`.

Run:

```bash
OUT_DIR=tests/gpu_validation/out/mp3_characteristic_flux_memcheck \
tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh
```

Expected: exit zero and `ERROR SUMMARY: 0 errors`.

- [x] **Step 5: Run five interleaved timing repetitions.** FP64/candidate
  complete-RK medians were `0.023973636/0.025151473 s`; the candidate was
  `4.913%` slower while reducing the selected workspace by exactly 50%.

Run:

```bash
OUT_DIR=tests/gpu_validation/out/mp3_characteristic_flux_benchmark \
REPEATS=5 GRID=400,16,16 MAXSTEP=5 \
tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh
```

Expected: `summary.md` reports exact 50 percent workspace-byte reduction,
finite complete-RK medians, run-to-run spreads, and observed GPU utilization.
Timing regression is recorded rather than hidden.

- [ ] **Step 6: Run final static checks.**

Run:

```bash
git diff --check
rg -n "flux_characteristic_work_sp_d|characteristic_flux" src_gpu tests/gpu_validation documents docs/superpowers
```

Expected: no whitespace errors and every candidate occurrence belongs to the
approved source, test, or documentation scope.

### Task 7: Documentation and classification

**Files:**
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`
- Modify: `docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md`

**Interfaces:**
- Consumes: exact Task 6 evidence paths and measurements
- Produces: one bounded MP3-A classification and reproducible commands

- [x] **Step 1: Add failing documentation assertions.**

Extend the MP3 contract test to require all four documents to contain:

```text
characteristic_flux
local-pass-not-promoted
S0-A6
S0-A10
```

Require the validation README to include the pilot, MPI matrix, memcheck, and
benchmark command names.

- [x] **Step 2: Run the documentation tests and verify RED.**

Run:

```bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
```

Expected: documentation assertions FAIL before updates.

- [x] **Step 3: Record exact evidence without broadening the claim.**

Document:

- the exact frozen field and statistics tolerances;
- S0-A6 to S0-A10 maximum field/statistics differences;
- exact raw sensor difference and mask mismatch counts;
- sanitizer summary;
- FP64 and candidate workspace bytes;
- five-run medians, spreads, peak memory, and timing change;
- the unchanged FP64 default and rollback command;
- the exclusion of physical boundaries and OpenSBLI production promotion.

Use `rejected` if any required gate failed. Use
`local-pass-not-promoted` only if every required local gate passed. Do not use
`validated`, `production-ready`, or physical-equivalence language for the
OpenSBLI path.

- [ ] **Step 4: Run final verification.**

Run:

```bash
python3 -m unittest discover -s tests/gpu_validation -p 'test_*.py' -v
git diff --check
```

Expected: complete suite PASS and no whitespace errors. Report any skipped
hardware or physical gate explicitly.

- [x] **Step 5: Mark this plan with actual completion states.**

Change each completed checkbox only after its command has passed. Leave failed
or skipped steps unchecked and record the concrete reason beside that step.

- [ ] **Step 6: Final review boundary.**

Audit the diff against the design document, especially precision ownership,
periodic-only dispatch, explicit synchronization, and claim strength. If the
user authorizes Git operations, stage an explicit whitelist rather than
`git add -A`:

```bash
git add src_gpu/mixed_candidate_gpu.cuf src_gpu/commarray_gpu.cuf src_gpu/solver_gpu.cuf src_gpu/mainloop_gpu.cuf tests/gpu_validation/mixed_candidate_setup_test.cuf tests/gpu_validation/test_mixed_precision_mp3_contract.py tests/gpu_validation/run_mp3_characteristic_flux_compare.sh tests/gpu_validation/freeze_mp3_tolerances.py tests/gpu_validation/test_freeze_mp3_tolerances.py tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh tests/gpu_validation/summarize_mp2_benchmark.py tests/gpu_validation/test_summarize_mp2_benchmark.py tests/gpu_validation/README.md documents/GPU_VALIDATION_MATRIX.md documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md docs/superpowers/specs/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux-design.md docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md
git commit -m "feat(gpu): validate mixed characteristic flux workspace"
```
