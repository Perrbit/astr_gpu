# GPU Mixed-Precision MP3 Cartesian Viscous HBL Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Admit and validate FP32 storage of the five-component characteristic interface-flux workspace for the exact Cartesian viscous S2-C3 HBL case while retaining reconstruction, sensing, viscous, state, RHS, and RK arithmetic in FP64.

**Architecture:** Reuse the mutually exclusive flux_characteristic_work_sp_d allocation and add FP32 writer/reader counterparts for the x/y/z xyphysical characteristic kernels. Eligibility stays fail-closed through gpu_s2_hbl_selective_roe_diffusion_supported(). Validation uses CPU FP64, GPU FP64, and GPU MP3 cases so pre-existing CPU/GPU error is separated from the mixed-workspace increment.

**Tech Stack:** CUDA Fortran with NVHPC, Fortran 2008, MPI/OpenMPI, HDF5, Bash, Python unittest, NumPy, Compute Sanitizer, and the root CMake build.

## Global Constraints

- Work directly in /home/dell/workspace/astr_gpu on feature/gpu_dev; do not create a worktree.
- Build only through /home/dell/workspace/astr_gpu/CMakeLists.txt and the existing build_cpu_probe and build_gpu_probe trees.
- usegpu remains a runtime input-file option.
- The FP64 path remains the production default. MP3-HBL1 is opt-in through ASTR_GPU_PRECISION_MODE=mixed_workspace and ASTR_GPU_MIXED_CANDIDATE=characteristic_flux.
- Admit only gpu_s2_hbl_selective_roe_diffusion_supported(): Cartesian bl, 543e, 643e, MP7 characteristic reconstruction, diffusion enabled, filter disabled, 11/21/41/51/1/1, no sponge, five-equation single-component RK3.
- Keep q_d, primitive fields, Roe algebra, characteristic matrices, MP7 reconstruction, sensor, mask, derivatives, viscous terms, qrhs_d, qsave_d, MPI halos, diagnostics, and output in their existing precision.
- Cast only final characteristic interface-flux stores to FP32. Promote both neighboring loads to FP64 before subtraction.
- Preserve sync_after_kernel after every new kernel launch.
- Do not modify CPU numerical or boundary code. If CPU FP64 versus GPU FP64 fails, stop and report.
- Formal comparisons use complete-RK same-phase fields and include all physical boundary planes.
- Calibrate once, freeze tolerances before the MPI matrix, and never relax them after a formal failure.
- Raw sensor uses a frozen tolerance after state divergence. Binary shock-mask mismatch must remain zero.
- Do not claim HBL physical validation, SBLI, NSCBC, CURVE, filtering, chemistry, production speedup, or mixed-precision promotion.
- Never use git add -A. Stage explicit path whitelists and inspect the staged diff before every commit.

---

### Task 1: Seal the validated MP3-XP prerequisite

**Files:**
- Modify/commit existing: src_gpu/commarray_gpu.cuf
- Modify/commit existing: src_gpu/mainloop_gpu.cuf
- Modify/commit existing: src_gpu/solver_gpu.cuf
- Modify/commit existing: tests/gpu_validation/test_mixed_precision_mp3_contract.py
- Create/commit existing: tests/gpu_validation/run_mp3_characteristic_flux_xphysical_compare.sh
- Create/commit existing: tests/gpu_validation/run_mp3_characteristic_flux_xphysical_matrix.sh
- Create/commit existing: tests/gpu_validation/run_mp3_characteristic_flux_xphysical_memcheck.sh
- Modify/commit existing: tests/gpu_validation/README.md
- Modify/commit existing: documents/GPU_VALIDATION_MATRIX.md
- Modify/commit existing: documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md
- Modify/commit existing: documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md
- Modify/commit existing: docs/superpowers/specs/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux-design.md
- Modify/commit existing: docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md
- Create/commit existing: docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-xphysical-gate.md

**Interfaces:**
- Consumes: completed MP3-XP evidence under tests/gpu_validation/out/mp3_xphysical_*_20260912*
- Produces: a clean prerequisite commit containing S0-B0 admission and x-physical FP32 writer/reader kernels

- [x] **Step 1: Re-run the focused MP3 contract.**

~~~bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
~~~

Expected: all MP3 tests pass.

- [x] **Step 2: Rebuild the root GPU targets.**

~~~bash
cmake --build build_gpu_probe --target astr mixed_candidate_setup_test -j2
~~~

Expected: both targets build successfully.

- [x] **Step 3: Re-run the complete Python validation suite.**

~~~bash
python3 -m unittest discover -s tests/gpu_validation -p 'test_*.py'
~~~

Expected: all tests pass. Stop on any failure.

- [x] **Step 4: Stage and audit only the MP3-XP prerequisite.**

~~~bash
git add -- \
  src_gpu/commarray_gpu.cuf \
  src_gpu/mainloop_gpu.cuf \
  src_gpu/solver_gpu.cuf \
  tests/gpu_validation/test_mixed_precision_mp3_contract.py \
  tests/gpu_validation/run_mp3_characteristic_flux_xphysical_compare.sh \
  tests/gpu_validation/run_mp3_characteristic_flux_xphysical_matrix.sh \
  tests/gpu_validation/run_mp3_characteristic_flux_xphysical_memcheck.sh \
  tests/gpu_validation/README.md \
  documents/GPU_VALIDATION_MATRIX.md \
  documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md \
  documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md \
  docs/superpowers/specs/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux-design.md \
  docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md \
  docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-xphysical-gate.md
git diff --cached --check
git diff --cached --name-status
git diff --cached --stat
~~~

Expected: exactly the fourteen listed paths and no manuals, presentations, rendered files, or unrelated scripts.

- [x] **Step 5: Commit the prerequisite.**

~~~bash
git commit -m "feat(gpu): validate MP3 x-physical characteristic flux"
~~~

---

### Task 2: Add failing MP3-HBL1 contracts

**Files:**
- Modify: tests/gpu_validation/test_mixed_precision_mp3_contract.py
- Test: tests/gpu_validation/test_mixed_precision_mp3_contract.py

**Interfaces:**
- Consumes: source text constants COMMARRAY, SOLVER, MAINLOOP
- Produces: failing contracts for HBL eligibility, six FP32 xyphysical kernels, dispatch/synchronization, three drivers, and sensor tolerance freezing

- [x] **Step 1: Register planned driver contents.**

Add these constants:

~~~python
HBL_COMPARE_DRIVER = read(
    ROOT / "tests" / "gpu_validation"
    / "run_mp3_characteristic_flux_hbl_compare.sh"
)
HBL_MATRIX_DRIVER = read(
    ROOT / "tests" / "gpu_validation"
    / "run_mp3_characteristic_flux_hbl_matrix.sh"
)
HBL_MEMCHECK_DRIVER = read(
    ROOT / "tests" / "gpu_validation"
    / "run_mp3_characteristic_flux_hbl_memcheck.sh"
)
FREEZE_TOLERANCES = read(
    ROOT / "tests" / "gpu_validation" / "freeze_mp3_tolerances.py"
)
~~~

- [x] **Step 2: Add failing source and driver contracts.**

Rename the existing
`test_characteristic_flux_admits_only_the_existing_xphysical_case_gate` method
to `test_characteristic_flux_retains_the_existing_xphysical_case_gate`. Keep its
two S0-B0 assertions unchanged; the old name would contradict the new bounded
HBL admission.

~~~python
def test_characteristic_flux_admits_exact_viscous_s2c3_hbl_gate(self):
    compact = COMMARRAY.replace(" ", "").lower()
    self.assertIn("gpu_s2_hbl_selective_roe_diffusion_supported", compact)
    self.assertIn(
        "hbl_viscous_case=gpu_s2_hbl_selective_roe_diffusion_supported()",
        compact,
    )
    self.assertIn(
        "periodic_case.or.xphysical_case.or.hbl_viscous_case", compact
    )

def test_xyphysical_characteristic_fp32_kernels_preserve_fp64_algebra(self):
    lower = SOLVER.lower()
    for axis in "xyz":
        for kind in ("flux", "rhs"):
            self.assertIn(
                f"subroutine characteristic_upwind_{kind}_{axis}"
                "_xyphysical_global_sp_kernel",
                lower,
            )
    compact = SOLVER.replace(" ", "").lower()
    self.assertGreaterEqual(
        compact.count(
            "flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)"
        ),
        7,
    )
    self.assertIn(
        "real(flux_characteristic_work_sp_d(i-1,j,k,m),8)", compact
    )
    self.assertIn(
        "real(flux_characteristic_work_sp_d(i,j-1,k,m),8)", compact
    )
    self.assertIn(
        "real(flux_characteristic_work_sp_d(i,j,k-1,m),8)", compact
    )

def test_mainloop_dispatches_and_synchronizes_xyphysical_sp_kernels(self):
    compact = MAINLOOP.replace(" ", "").lower()
    for axis in "xyz":
        for kind in ("flux", "rhs"):
            name = (
                f"characteristic_upwind_{kind}_{axis}"
                "_xyphysical_global_sp_kernel"
            )
            self.assertIn(f"call{name}<<<", compact)
            self.assertIn(f"sync_after_kernel('{name}')", compact)

def test_hbl_compare_driver_is_three_way_same_phase_and_sensor_bounded(self):
    text = HBL_COMPARE_DRIVER
    for target in ("cpu", "gpu_fp64", "gpu_characteristic_flux"):
        self.assertIn(target, text)
    self.assertIn("ASTR_VALIDATION_RK_SNAPSHOT", text)
    self.assertIn("CANDIDATE_SENSOR_ATOL", text)
    self.assertIn("gpu_fp64_vs_candidate_shock_sensor.txt", text)
    self.assertIn(
        "ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux", text
    )

def test_hbl_matrix_locks_np1_and_three_np2_slabs(self):
    text = HBL_MATRIX_DRIVER
    for line in (
        "NP=1 TOPOLOGY=1,1,1",
        "NP=2 TOPOLOGY=2,1,1",
        "NP=2 TOPOLOGY=1,2,1",
        "NP=2 TOPOLOGY=1,1,2",
    ):
        self.assertIn(line, text)
    self.assertIn("TOLERANCE_FILE", text)
    self.assertNotIn("CALIBRATE=t", text)

def test_hbl_memcheck_locks_x_and_y_slabs(self):
    text = HBL_MEMCHECK_DRIVER
    self.assertIn("2,1,1", text)
    self.assertIn("1,2,1", text)
    self.assertIn("compute-sanitizer --tool memcheck", text)
    self.assertIn("--error-exitcode 99", text)
    self.assertIn("ERROR SUMMARY: 0 errors", text)
    self.assertIn("OMPI_MCA_opal_cuda_support=0", text)

def test_tolerance_freezer_accepts_optional_sensor_report(self):
    self.assertIn("--sensor-report", FREEZE_TOLERANCES)
    self.assertIn("CANDIDATE_SENSOR_ATOL", FREEZE_TOLERANCES)
    self.assertIn("MP3_PILOT_SENSOR_MAX", FREEZE_TOLERANCES)
~~~

- [x] **Step 3: Verify intended failures.**

~~~bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
~~~

Expected: only the new HBL tests fail. Existing MP3 tests pass.

- [x] **Step 4: Commit the failing contracts.**

~~~bash
git add -- tests/gpu_validation/test_mixed_precision_mp3_contract.py
git diff --cached --check
git diff --cached --name-status
git commit -m "test(gpu): define MP3 viscous HBL contracts"
~~~

---

### Task 3: Implement eligibility and six FP32 xyphysical kernels

**Files:**
- Modify: src_gpu/commarray_gpu.cuf:94-115
- Modify: src_gpu/solver_gpu.cuf:2199-2561
- Modify: src_gpu/mainloop_gpu.cuf:625-706
- Test: tests/gpu_validation/test_mixed_precision_mp3_contract.py

**Interfaces:**
- Consumes: gpu_s2_hbl_selective_roe_diffusion_supported(), flux_characteristic_work_sp_d, existing FP64 xyphysical bounds, mixed_characteristic_flux_workspace
- Produces: exact S2-C3 admission and characteristic_upwind_{flux|rhs}_{x|y|z}_xyphysical_global_sp_kernel

- [x] **Step 1: Extend eligibility without weakening existing branches.**

Use this exact structure in gpu_mixed_characteristic_flux_workspace_enabled():

~~~fortran
use case_capability_gpu, only: gpu_shock_characteristic_s0b0_xphysical_supported, &
     gpu_s2_hbl_selective_roe_diffusion_supported
logical :: periodic_case,xphysical_case,hbl_viscous_case

periodic_case=lihomo .and. ljhomo .and. lkhomo .and. (.not.diffterm)
xphysical_case=gpu_shock_characteristic_s0b0_xphysical_supported()
hbl_viscous_case=gpu_s2_hbl_selective_roe_diffusion_supported()

gpu_mixed_characteristic_flux_workspace_enabled = &
  gpu_mixed_workspace_requested() .and. &
  gpu_mixed_candidate_requested(GPU_MIXED_CHARACTERISTIC_FLUX) .and. &
  trim(conschm)=='543e' .and. recon_schem==3 .and. lchardecomp .and. &
  (periodic_case .or. xphysical_case .or. hbl_viscous_case) .and. &
  (.not.lfilter) .and. &
  numq==5 .and. num_species==0 .and. num_modequ==0 .and. &
  trim(rkscheme)=='rk3'
~~~

Do not replace hbl_viscous_case with a loose diffterm or boundary expression.

- [x] **Step 2: Add exact FP32 writer bodies.**

Each writer keeps real(8) fh(5), calls the existing FP64 reconstruction routine, and casts only the store. Use these exact index and call contracts:

| Axis | Interface bounds | Reconstruction tail |
| --- | --- | --- |
| x | i=is-1:ie, j=js:je, k=ks:ke | hm,npdci,1,fh |
| y | i=is:ie, j=js-1:je, k=ks:ke | hm,npdcj,2,fh |
| z | i=is:ie, j=js:je, k=ks-1:ke | 0,0,0,fh |

The x implementation is:

~~~fortran
attributes(global) subroutine characteristic_upwind_flux_x_xyphysical_global_sp_kernel( &
     im,jm,km,hm,is,ie,js,je,ks,ke,npdci,reconstruction_scheme,gamma)
  use commarray_gpu, only: flux_characteristic_work_sp_d
  implicit none
  integer,value :: im,jm,km,hm,is,ie,js,je,ks,ke,npdci,reconstruction_scheme
  real(8),value :: gamma
  integer :: i,j,k,m
  real(8) :: fh(5)

  i=(blockIdx%x-1)*blockDim%x+threadIdx%x-2
  j=blockIdx%y-1
  k=blockIdx%z-1
  if(i < is-1 .or. i > ie .or. j < js .or. j > je .or. k < ks .or. k > ke) return
  call characteristic_reconstruction_interface_flux( &
       i,j,k,1,im,jm,km,reconstruction_scheme,gamma,hm,npdci,1,fh)
  do m=1,5
    flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)
  enddo
end subroutine characteristic_upwind_flux_x_xyphysical_global_sp_kernel
~~~

Implement y and z as separate subroutines using the exact table and the launch-index expressions from their FP64 counterparts. Do not call one axis kernel from another.

- [x] **Step 3: Add exact FP64-accumulating reader bodies.**

Use these exact differences:

~~~fortran
! x
qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
     (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
      real(flux_characteristic_work_sp_d(i-1,j,k,m),8))

! y
qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
     (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
      real(flux_characteristic_work_sp_d(i,j-1,k,m),8))

! z
qrhs_d(i,j,k,m)=qrhs_d(i,j,k,m)- &
     (real(flux_characteristic_work_sp_d(i,j,k,m),8)- &
      real(flux_characteristic_work_sp_d(i,j,k-1,m),8))
~~~

All readers use interior bounds i=is:ie, j=js:je, k=ks:ke and write only FP64 qrhs_d.

- [x] **Step 4: Add explicit mixed/FP64 dispatch per direction.**

For x, y, and z independently, branch on mixed_characteristic_flux_workspace. Launch the matching xyphysical_global_sp writer and reader in the true branch and retain the existing FP64 writer and reader in the false branch. Keep each sync_after_kernel immediately after its launch. Keep conv_after_x, conv_after_y, and conv_after_z snapshots after their whole direction branch.

- [x] **Step 5: Run focused contracts and root build.**

~~~bash
python3 -m unittest -v \
  tests.gpu_validation.test_mixed_precision_mp3_contract.MixedPrecisionMp3Contract.test_characteristic_flux_admits_exact_viscous_s2c3_hbl_gate \
  tests.gpu_validation.test_mixed_precision_mp3_contract.MixedPrecisionMp3Contract.test_xyphysical_characteristic_fp32_kernels_preserve_fp64_algebra \
  tests.gpu_validation.test_mixed_precision_mp3_contract.MixedPrecisionMp3Contract.test_mainloop_dispatches_and_synchronizes_xyphysical_sp_kernels
cmake --build build_gpu_probe --target astr mixed_candidate_setup_test -j2
~~~

Expected: all three source contracts pass and the GPU targets build. Driver
contracts are intentionally not invoked until their files exist.

- [x] **Step 6: Commit implementation.**

~~~bash
git add -- src_gpu/commarray_gpu.cuf src_gpu/solver_gpu.cuf src_gpu/mainloop_gpu.cuf
git diff --cached --check
git diff --cached --name-status
git commit -m "feat(gpu): add MP3 viscous HBL characteristic workspace"
~~~

---

### Task 4: Build three-way comparison and sensor-aware tolerance freezing

**Files:**
- Create: tests/gpu_validation/run_mp3_characteristic_flux_hbl_compare.sh
- Modify: tests/gpu_validation/freeze_mp3_tolerances.py
- Modify: tests/gpu_validation/test_freeze_mp3_tolerances.py
- Test: tests/gpu_validation/test_mixed_precision_mp3_contract.py

**Interfaces:**
- Consumes: prepare_s1_flatplate_case.py, generate_compressible_blasius_profile.py, CPU/GPU executables, and the three comparison tools
- Produces: six attribution reports and a tolerance file containing field, statistics, and sensor limits

- [x] **Step 1: Add failing sensor-freezer tests.**

~~~python
def test_sensor_parser_reads_max_abs(self):
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / "sensor.txt"
        report.write_text(
            "status: pass\nmax_abs: 3.5e-8\nmask_mismatches: 0\n",
            encoding="ascii",
        )
        self.assertEqual(MODULE.read_sensor_max(report), 3.5e-8)

def test_sensor_tolerance_uses_engineering_rounding(self):
    self.assertEqual(
        MODULE.frozen_tolerance(3.5e-8, 1.0e-12, 1.0e-5),
        5.0e-7,
    )
~~~

Run the freezer tests and expect read_sensor_max to be undefined.

- [x] **Step 2: Add optional sensor parsing.**

~~~python
SENSOR_PATTERN = re.compile(
    r"^max_abs:\s*([+\-0-9.eE]+)\s*$", re.MULTILINE
)

def read_sensor_max(path: Path) -> float:
    match = SENSOR_PATTERN.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path}: missing sensor max_abs")
    value = float(match.group(1))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{path}: invalid sensor max_abs")
    return value
~~~

Add optional --sensor-report. When present, append:

~~~python
sensor_observed = read_sensor_max(args.sensor_report)
sensor_tolerance = frozen_tolerance(
    sensor_observed, 1.0e-12, 1.0e-5
)
lines.extend(
    [
        f"CANDIDATE_SENSOR_ATOL={sensor_tolerance:.1e}",
        "CANDIDATE_SENSOR_RTOL=0",
        f"MP3_PILOT_SENSOR_MAX={sensor_observed:.16e}",
    ]
)
~~~

Existing calls without --sensor-report must keep their old output.

- [x] **Step 3: Create the exact HBL comparison driver.**

Use 192x192x8, two steps, deltat=1.0e-5, Mach 5, Reynolds 1.83052e6, reference temperature 226.65, wall temperature 5.191440547760865, x from -1 to 10, y stretch 5, z length 0.25, and bctype 11/21/41/51/1/1. Set diffterm=t, lchardecomp=t, lfilter=f, no sponge, and shock threshold 0.001.

Generate identical profile and initial fields for CPU, GPU FP64, and GPU characteristic_flux using generate_compressible_blasius_profile.py with profile and field oblique-shock initialization, shock angle 35 degrees, shock x=-1, shock y=0.18, and provided density/pressure.

Always set ASTR_SHOCK_SENSOR_DUMP. For CPU also set:

~~~bash
ASTR_VALIDATION_RK_SNAPSHOT=outdat/rk_complete_snapshot.h5
~~~

Run GPU reference with ASTR_GPU_PRECISION_MODE=fp64. Run candidate with:

~~~bash
ASTR_GPU_PRECISION_MODE=mixed_workspace
ASTR_GPU_MIXED_CANDIDATE=characteristic_flux
ASTR_GPU_SYNC_MODE=explicit
~~~

Verify the completion marker and all three MP3 log markers. Produce:

~~~text
cpu_vs_gpu_fp64_shock_sensor.txt
cpu_vs_gpu_fp64_flowstate.txt
cpu_vs_gpu_fp64_flowfield.txt
gpu_fp64_vs_candidate_shock_sensor.txt
gpu_fp64_vs_candidate_flowstate.txt
gpu_fp64_vs_candidate_flowfield.txt
~~~

Use the CPU RK snapshot for complete-field comparison. Candidate sensor comparison uses CANDIDATE_SENSOR_ATOL and CANDIDATE_SENSOR_RTOL. CALIBRATE=t may set candidate field/stat/sensor atol to 1.0 and rtol to zero, but compare_shock_sensor.py must still reject any mask mismatch.

- [x] **Step 4: Run focused tests.**

~~~bash
python3 -m unittest -v \
  tests.gpu_validation.test_freeze_mp3_tolerances \
  tests.gpu_validation.test_mixed_precision_mp3_contract.MixedPrecisionMp3Contract.test_hbl_compare_driver_is_three_way_same_phase_and_sensor_bounded \
  tests.gpu_validation.test_mixed_precision_mp3_contract.MixedPrecisionMp3Contract.test_tolerance_freezer_accepts_optional_sensor_report
~~~

Expected: freezer and compare-driver tests pass. Matrix and memcheck contracts
are intentionally not invoked until their files exist.

- [x] **Step 5: Commit comparison tooling.**

~~~bash
chmod +x tests/gpu_validation/run_mp3_characteristic_flux_hbl_compare.sh
git add -- \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_compare.sh \
  tests/gpu_validation/freeze_mp3_tolerances.py \
  tests/gpu_validation/test_freeze_mp3_tolerances.py
git diff --cached --check
git diff --cached --name-status
git commit -m "test(gpu): add MP3 viscous HBL calibration gate"
~~~

---

### Task 5: Add formal MPI and sanitizer drivers

**Files:**
- Create: tests/gpu_validation/run_mp3_characteristic_flux_hbl_matrix.sh
- Create: tests/gpu_validation/run_mp3_characteristic_flux_hbl_memcheck.sh
- Modify: tests/gpu_validation/test_mixed_precision_mp3_contract.py

**Interfaces:**
- Consumes: HBL compare driver and frozen CANDIDATE_FIELD, CANDIDATE_STATS, and CANDIDATE_SENSOR tolerances
- Produces: four formal numerical runs and two NP=2 sanitizer runs

- [x] **Step 1: Create the fixed matrix.**

Require and source TOLERANCE_FILE. Refuse to overwrite OUT_DIR. Run exactly:

~~~bash
NP=1 TOPOLOGY=1,1,1 IM=192 JM=192 KM=8 MAXSTEP=2 \
  OUT_DIR="$OUT_DIR/np1" "$COMPARE_DRIVER"
NP=2 TOPOLOGY=2,1,1 IM=192 JM=192 KM=8 MAXSTEP=2 \
  OUT_DIR="$OUT_DIR/np2_2x1x1" "$COMPARE_DRIVER"
NP=2 TOPOLOGY=1,2,1 IM=192 JM=192 KM=8 MAXSTEP=2 \
  OUT_DIR="$OUT_DIR/np2_1x2x1" "$COMPARE_DRIVER"
NP=2 TOPOLOGY=1,1,2 IM=192 JM=192 KM=8 MAXSTEP=2 \
  OUT_DIR="$OUT_DIR/np2_1x1x2" "$COMPARE_DRIVER"
~~~

Pass all six candidate tolerance variables into every call. Never enable calibration.

- [x] **Step 2: Create candidate-only x/y memcheck runs.**

Prepare independent 64x64x8 one-step S2-C3 viscous cases for topologies 2x1x1 and 1x2x1. Use:

~~~bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" \
ASTR_GPU_PRECISION_MODE=mixed_workspace \
ASTR_GPU_MIXED_CANDIDATE=characteristic_flux \
ASTR_GPU_SYNC_MODE=explicit \
OMPI_MCA_rmaps_base_oversubscribe=1 \
OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
OMPI_MCA_btl=self,tcp OMPI_MCA_osc=pt2pt \
OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
mpirun -np 2 compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$GPU_EXE" run datin/input.flatplate
~~~

Set ASTR_FORCE_MPI_TOPOLOGY separately to 2,1,1 and 1,2,1. Each log must contain the normal completion marker, active characteristic_flux marker, and exactly two ERROR SUMMARY: 0 errors lines.

- [x] **Step 3: Run all MP3 contracts.**

~~~bash
python3 -m unittest tests.gpu_validation.test_mixed_precision_mp3_contract -v
~~~

Expected: all focused tests pass.

- [x] **Step 4: Commit formal drivers.**

~~~bash
chmod +x \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_matrix.sh \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_memcheck.sh
git add -- \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_matrix.sh \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_memcheck.sh \
  tests/gpu_validation/test_mixed_precision_mp3_contract.py
git diff --cached --check
git diff --cached --name-status
git commit -m "test(gpu): add MP3 viscous HBL MPI safety gates"
~~~

---

### Task 6: Execute evidence and synchronize bounded status

**Files:**
- Modify after passing: tests/gpu_validation/test_mixed_precision_mp3_contract.py
- Modify after passing: tests/gpu_validation/README.md
- Modify after passing: documents/GPU_VALIDATION_MATRIX.md
- Modify after passing: documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md
- Modify after passing: documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md
- Modify after passing: docs/superpowers/specs/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux-design.md
- Modify after passing: docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md
- Modify after execution: docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-hbl-gate.md
- Evidence only: tests/gpu_validation/out/mp3_hbl_*

**Interfaces:**
- Consumes: built CPU/GPU executables and Tasks 2-5
- Produces: frozen tolerances, four numerical gates, two sanitizer gates, storage accounting, and a bounded classification

- [x] **Step 1: Build from root CMake.** CPU `astr` and GPU `astr` plus
  `mixed_candidate_setup_test` completed successfully.

~~~bash
cmake --build build_cpu_probe --target astr -j2
cmake --build build_gpu_probe --target astr mixed_candidate_setup_test -j2
~~~

Expected: successful CPU and GPU builds.

- [x] **Step 2: Run the full Python suite.** The pre-documentation run completed
  277 tests successfully.

~~~bash
python3 -m unittest discover -s tests/gpu_validation -p 'test_*.py'
~~~

Expected: all tests pass.

- [x] **Step 3: Run calibration.** The three-way run completed under
  `tests/gpu_validation/out/mp3_hbl_calibration_20260912`; candidate sensor
  difference and mask mismatch count were zero.

~~~bash
OUT_DIR=tests/gpu_validation/out/mp3_hbl_calibration_20260912 \
CALIBRATE=t NP=1 TOPOLOGY=1,1,1 IM=192 JM=192 KM=8 MAXSTEP=2 \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_compare.sh
~~~

Expected: CPU/GPU FP64 baseline passes, all candidate reports exist, and shock-mask mismatch is zero. Stop otherwise.

- [x] **Step 4: Freeze field, statistics, and sensor tolerances.** The frozen
  values are field `5.0e-07`, statistics `1.0e-12`, sensor `1.0e-12`, and zero
  relative tolerances.

~~~bash
python3 tests/gpu_validation/freeze_mp3_tolerances.py \
  --field-report tests/gpu_validation/out/mp3_hbl_calibration_20260912/gpu_fp64_vs_candidate_flowfield.txt \
  --stats-report tests/gpu_validation/out/mp3_hbl_calibration_20260912/gpu_fp64_vs_candidate_flowstate.txt \
  --sensor-report tests/gpu_validation/out/mp3_hbl_calibration_20260912/gpu_fp64_vs_candidate_shock_sensor.txt \
  --output tests/gpu_validation/out/mp3_hbl_calibration_20260912/mp3_hbl_tolerances.env
~~~

Expected: all three absolute tolerances are finite and no greater than 1e-5. All relative tolerances are zero.

- [x] **Step 5: Run formal NP=1 and three-slab matrix.** NP=1 and NP=2 x/y/z
  slabs passed. Maximum field/statistics differences were
  `2.0437756598212786e-08` and `1.6875389974302379e-14`; every mask mismatch
  count was zero.

~~~bash
OUT_DIR=tests/gpu_validation/out/mp3_hbl_matrix_20260912 \
TOLERANCE_FILE=tests/gpu_validation/out/mp3_hbl_calibration_20260912/mp3_hbl_tolerances.env \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_matrix.sh
~~~

Expected: all four runs pass without calibration or tolerance changes.

- [x] **Step 6: Run x/y NP=2 Compute Sanitizer.** Both topologies completed;
  each log contains two `ERROR SUMMARY: 0 errors` records.

~~~bash
OUT_DIR=tests/gpu_validation/out/mp3_hbl_memcheck_20260912 \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_memcheck.sh
~~~

Expected: both topologies complete with two clean rank summaries each.

- [x] **Step 7: Verify storage accounting.** NP=1 and all three NP=2 slabs
  report `characteristic_flux`; mixed bytes are exactly half the FP64 bytes.

~~~bash
rg 'ASTR_GPU_(ACTIVE_MIXED_WORKSPACE|MIXED_WORKSPACE_BYTES|FP64_WORKSPACE_BYTES)' \
  tests/gpu_validation/out/mp3_hbl_matrix_20260912
~~~

Expected: active workspace is characteristic_flux and twice the mixed bytes equals the FP64 bytes. The source contract verifies no simultaneous FP64 mirror.

- [x] **Step 8: Update status documents from measured reports.** A RED/GREEN
  documentation contract locks the exact classification and measured maxima.

Record the actual frozen tolerances, observed field/stat/sensor maxima, zero mask mismatch, four completed topologies, both sanitizer results, and exact 50 percent workspace reduction. Use only this classification:

~~~text
hbl-cartesian-local-pass-not-promoted
~~~

List exclusions: long-time HBL physics, SBLI, NSCBC, bctype=52, sponge, filter, CURVE, chemistry, and production speedup.

- [x] **Step 9: Run final verification.** The final suite completed 278 tests,
  both root GPU targets built, and `git diff --check` is required again before
  staging.

~~~bash
python3 -m unittest discover -s tests/gpu_validation -p 'test_*.py'
cmake --build build_gpu_probe --target astr mixed_candidate_setup_test -j2
git diff --check
~~~

Expected: full suite passes, targets build, and no whitespace errors are reported.

- [x] **Step 10: Stage and audit final documentation.** The staged whitelist
  contains only the MP3-HBL1 plan, contract, and evidence-facing documents;
  runtime evidence and unrelated untracked files are excluded.

~~~bash
git add -- \
  tests/gpu_validation/test_mixed_precision_mp3_contract.py \
  tests/gpu_validation/README.md \
  documents/GPU_VALIDATION_MATRIX.md \
  documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md \
  documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md \
  docs/superpowers/specs/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux-design.md \
  docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-characteristic-flux.md \
  docs/superpowers/plans/2026-09-12-gpu-mixed-precision-mp3-hbl-gate.md
git diff --cached --check
git diff --cached --name-status
git diff --cached --stat
~~~

Expected: only evidence-facing source-controlled documents are staged. Runtime output directories remain untracked or ignored.

- [x] **Step 11: Commit the bounded result.** The bounded MP3-HBL1 result was
  committed after the staged whitelist and final checks passed.

~~~bash
git commit -m "docs(gpu): close MP3 Cartesian viscous HBL gate"
~~~

If a runtime gate fails, document the failure instead. Do not use the passing classification.
