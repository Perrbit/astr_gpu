#!/usr/bin/env python3
"""Static contracts for viscous-source coupling in curved upper-y NSCBC."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), f"missing production source: {relative}"
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    match = re.search(
        rf"(?:attributes\(global\)\s+)?(?:logical function|subroutine)\s+{name}\b(.*?)"
        rf"end (?:function|subroutine)\s+{name}",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert match is not None, f"missing routine {name}"
    return match.group(1)


def compact(text: str) -> str:
    return text.replace(" ", "").replace("\n", "").replace("&", "").lower()


def test_viscous_capability_is_separate_from_inviscid_capability() -> None:
    text = source("src_gpu/case_capability_gpu.cuf")
    inviscid = compact(
        function_body(text, "gpu_curvilinear_nscbc52_nonreflecting_supported")
    )
    viscous = compact(
        function_body(
            text, "gpu_curvilinear_nscbc52_viscous_nonreflecting_supported"
        )
    )

    assert "(.not.diffterm)" in inviscid
    for token in (
        "diffterm",
        "trim(difschm)=='643e'",
        "(.not.lfilter)",
        "numq==5",
        "num_species==0",
        "num_modequ==0",
        "trim(turbmode)=='none'",
        "bctype(1)==11",
        "bctype(2)==21",
        "bctype(3)==41",
        "bctype(4)==52",
        "all(bctype(5:6)==1)",
        "spg_im==0",
        "nscbc_farfield_nonreflecting_enabled()",
    ):
        assert token in viscous


def test_cpu_stage_order_is_capture_diffusion_correct() -> None:
    body = compact(function_body(source("src/solver.F90"), "rhscal"))
    sign = body.index("qrhs=-qrhs")
    capture = body.index("callcapture_nscbc_farfield_y_upper_prediff_rhs", sign)
    diffusion = body.index("calldiffrsdcal6", capture)
    correct = body.index("callapply_nscbc_farfield_y_upper_viscous_source", diffusion)
    snapshot = body.index("callwrite_rhs_validation_snapshot('full')", correct)
    assert sign < capture < diffusion < correct < snapshot


def test_gpu_stage_order_is_capture_diffusion_correct() -> None:
    body = compact(
        function_body(source("src_gpu/mainloop_gpu.cuf"), "time_integration_rk_gpu")
    )
    capture = body.index("callcapture_nscbc_farfield_y_upper_prediff_rhs_gpu")
    diffusion = body.index("calldiffusion_flux_xyphysical_global_kernel", capture)
    correct = body.index(
        "callapply_nscbc_farfield_y_upper_viscous_source_gpu", diffusion
    )
    sources = body.index("callapply_case_sources_gpu", correct)
    assert capture < diffusion < correct < sources


def test_cpu_face_snapshot_has_ready_state_and_upper_owner_guard() -> None:
    text = source("src/bc.F90")
    capture = compact(
        function_body(text, "capture_nscbc_farfield_y_upper_prediff_rhs")
    )
    correct = compact(
        function_body(text, "apply_nscbc_farfield_y_upper_viscous_source")
    )
    assert "nscbc52_prediff_rhs" in capture
    assert "nscbc52_prediff_rhs_ready=.true." in capture
    assert (
        "npdcj==2.or.npdcj==4" in capture
        or "if(npdcj/=2.and.npdcj/=4)return" in capture
    )
    assert "if(.not.nscbc52_prediff_rhs_ready)" in correct
    assert "nscbc52_prediff_rhs_ready=.false." in correct
    assert "callnscbc_remove_incoming_source" in correct


def test_gpu_face_snapshot_is_resident_released_and_owner_only() -> None:
    text = source("src_gpu/boundary_gpu.cuf")
    assert "real(8),allocatable,device::nscbc52_prediff_rhs_d" in compact(text)
    release = compact(function_body(text, "release_boundary_gpu"))
    capture = compact(
        function_body(text, "capture_nscbc_farfield_y_upper_prediff_rhs_gpu")
    )
    correct = compact(
        function_body(text, "apply_nscbc_farfield_y_upper_viscous_source_gpu")
    )
    assert "deallocate(nscbc52_prediff_rhs_d)" in release
    assert "if(apply_up==0)return" in capture
    assert "if(apply_up==0)return" in correct
    assert "nscbc52_prediff_rhs_ready" in capture
    assert "nscbc52_prediff_rhs_ready" in correct


def test_new_gpu_kernels_have_explicit_synchronization() -> None:
    text = compact(source("src_gpu/boundary_gpu.cuf"))
    assert (
        "callsync_after_kernel('nscbc_farfield_y_upper_prediff_capture_kernel')"
        in text
    )
    assert (
        "callsync_after_kernel('nscbc_farfield_y_upper_viscous_source_kernel')"
        in text
    )


def test_cpu_gpu_projection_helpers_share_the_characteristic_contract() -> None:
    cpu = compact(
        function_body(source("src/bc.F90"), "nscbc_remove_incoming_source")
    )
    gpu = compact(
        function_body(
            source("src_gpu/nscbc_characteristic_policy_gpu.cuf"),
            "nscbc_remove_incoming_source_gpu",
        )
    )
    for body in (cpu, gpu):
        assert "64.d0*epsilon(1.d0)" in body
        assert "incoming=lambda<-wave_tolerance" in body
        assert "source_characteristic" in body
        assert "rhs" in body
        assert "pnor" in body
        assert "pinv" in body


def test_viscous_hbl_matrix_covers_all_admitted_topologies() -> None:
    script = source("tests/gpu_validation/run_curvilinear_nscbc52_viscous_matrix.sh")
    body = compact(script)
    for entry in (
        "1:1,1,1",
        "2:2,1,1",
        "2:1,2,1",
        "2:1,1,2",
        "4:2,2,1",
        "4:2,1,2",
        "4:1,2,2",
        "8:2,2,2",
    ):
        assert entry in body
    assert "run_curvilinear_nscbc52_viscous_hbl_compare.sh" in script
    assert "ASTR_VALIDATION_NSCBC52_OWNER_TRACE=1" in script


def test_viscous_hbl_runner_locks_the_approved_case() -> None:
    script = source(
        "tests/gpu_validation/run_curvilinear_nscbc52_viscous_hbl_compare.sh"
    )
    body = compact(script)
    for token in (
        "upper_bctype=52",
        "diffterm=t",
        "lfilter=f",
        "conschm=643e",
        "astr_nscbc_farfield_mode=nonreflecting",
    ):
        assert token in body


def test_viscous_safety_and_profile_runners_lock_diffusion() -> None:
    memcheck = compact(
        source("tests/gpu_validation/run_curvilinear_nscbc52_viscous_memcheck.sh")
    )
    profile = compact(
        source("tests/gpu_validation/run_curvilinear_nscbc52_viscous_profile.sh")
        + source("tests/gpu_validation/run_curvilinear_nscbc52_profile.sh")
    )
    for body in (memcheck, profile):
        assert "diffterm=t" in body
        assert "astr_nscbc_farfield_mode=nonreflecting" in body
    assert "--viscous" in profile
    assert "analyze_nsys_rk_residency.py" in profile
    assert "np=2" in profile
    assert "topology=1,2,1" in profile
