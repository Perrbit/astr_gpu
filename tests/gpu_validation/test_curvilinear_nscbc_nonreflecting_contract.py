#!/usr/bin/env python3
"""Static contracts for the curved upper-y non-reflecting GCBC path."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), f"missing production source: {relative}"
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    match = re.search(
        rf"(?:logical function|integer function|subroutine)\s+{name}\b(.*?)"
        rf"end (?:function|subroutine)\s+{name}",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert match is not None, f"missing routine {name}"
    return match.group(1)


def compact(text: str) -> str:
    return text.replace(" ", "").replace("\n", "").replace("&", "").lower()


def test_farfield_parser_has_distinct_zero_incoming_policy() -> None:
    bc = source("src/bc.F90")
    for name in (
        "NSCBC_FARFIELD_POLICY_COMPATIBILITY",
        "NSCBC_FARFIELD_POLICY_TARGET_RELAXATION",
        "NSCBC_FARFIELD_POLICY_NONREFLECTING",
    ):
        assert name in bc
    parser = compact(function_body(bc, "configure_nscbc_farfield"))
    assert "case('nonreflecting')" in parser
    assert "callbcast(nscbc_farfield_policy)" in parser


def test_legacy_mode_strings_remain_supported() -> None:
    parser = compact(function_body(source("src/bc.F90"), "configure_nscbc_farfield"))
    for mode in ("compatibility", "incoming_only", "sbli_shock"):
        assert f"case('{mode}')" in parser


def test_policy_queries_are_separate_from_legacy_booleans() -> None:
    bc = source("src/bc.F90")
    policy = compact(function_body(bc, "nscbc_farfield_policy_id"))
    nonreflecting = compact(
        function_body(bc, "nscbc_farfield_nonreflecting_enabled")
    )
    assert "nscbc_farfield_policy_id=nscbc_farfield_policy" in policy
    assert "nscbc_farfield_policy==nscbc_farfield_policy_nonreflecting" in nonreflecting
    assert "nscbc_farfield_incoming_only_enabled" in bc
    assert "nscbc_farfield_sbli_shock_enabled" in bc


def test_cpu_farfield_selects_source_balanced_incoming_without_mach_relaxation() -> None:
    body = function_body(source("src/bc.F90"), "farfield_nscbc")
    packed = compact(body)
    assert "case(nscbc_farfield_policy_nonreflecting)" in packed
    assert "callnscbc_farfield_balance_incoming_lodi" in packed
    reduction = re.search(
        r"if\s*\(\s*nscbc_farfield_policy\s*/=\s*"
        r"nscbc_farfield_policy_nonreflecting\s*\)\s*then"
        r".*?gmachmax2\s*=\s*pmax\s*\(\s*gmachmax2\s*\)\s*endif",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert reduction is not None


def test_cpu_source_closes_x_edges_with_one_sided_transverse_derivatives() -> None:
    source_body = compact(
        function_body(source("src/bc.F90"), "nscbc_farfield_y_upper_source")
    )
    assert "if(i==0.and.(npdci==1.or.npdci==4))then" in source_body
    assert "source=source+flux_p-flux_m" in source_body
    assert "elseif(i==im.and.(npdci==2.or.npdci==4))then" in source_body

    farfield = compact(function_body(source("src/bc.F90"), "farfield_nscbc"))
    assert "case(nscbc_farfield_policy_nonreflecting)dEcs(:)=lodi1(:)+source(:)".lower() in farfield


def test_gpu_nonreflecting_path_skips_relaxation_work() -> None:
    mainloop = compact(
        function_body(
            source("src_gpu/mainloop_gpu.cuf"), "time_integration_rk_gpu"
        )
    )
    assert "nscbc_farfield_nonreflecting_enabled" in mainloop
    assert "flatplate_nscbc_nonreflecting_case" in mainloop
    assert (
        mainloop.index("if(flatplate_nscbc_nonreflecting_case)then")
        < mainloop.index("callgpu_nscbc_farfield_y_upper_mach2")
    )

    policy = compact(source("src_gpu/nscbc_characteristic_policy_gpu.cuf"))
    assert "attributes(device)subroutinenscbc_balance_incoming_lodi_gpu" in policy

    boundary = source("src_gpu/boundary_gpu.cuf")
    body = compact(
        function_body(
            boundary, "nscbc_farfield_y_upper_nonreflecting_rhs_kernel"
        )
    )
    assert "callnscbc_balance_incoming_lodi_gpu" in body
    assert "callnscbc_zero_incoming_lodi_gpu" not in body
    assert body.index("source(:)=rest(:)+transverse(:)") < body.index(
        "callnscbc_balance_incoming_lodi_gpu"
    )
    assert body.index("callnscbc_balance_incoming_lodi_gpu") < body.index(
        "lodi1(m)=sum(pnor(m,:)*lodi(:))*jacob_d(i,jm,k)"
    )
    assert "if(i==0.and.(npdci==1.or.npdci==4))then" in body
    assert "transverse(:)=transverse(:)+flux_p(:)-flux_m(:)" in body
    assert "elseif(i==im.and.(npdci==2.or.npdci==4))then" in body
    assert "rho_target" not in body
    assert "gmachmax2" not in body
    wrapper = compact(
        function_body(
            boundary, "apply_nscbc_farfield_y_upper_nonreflecting_rhs_gpu"
        )
    )
    assert (
        "callsync_after_kernel('nscbc_farfield_y_upper_nonreflecting_rhs_kernel')"
        in wrapper
    )


def test_gpu_nonreflecting_skips_legacy_prepare_filter_but_keeps_rk_snapshot() -> None:
    boundary = source("src_gpu/boundary_gpu.cuf")
    filter_case = compact(
        function_body(boundary, "gpu_s1_flatplate_nscbc_farfield_filter_case")
    )
    assert "gpu_s1_flatplate_nscbc_farfield_case()" in filter_case
    assert ".not.nscbc_farfield_nonreflecting_enabled()" in filter_case

    prepare = compact(
        function_body(source("src_gpu/mainloop_gpu.cuf"), "prepare_rkfirst_stats_gpu")
    )
    runtime = compact(
        function_body(source("src_gpu/gpu_runtime.cuf"), "gpu_prepare_rkfirst_stats")
    )
    integration = compact(
        function_body(source("src_gpu/mainloop_gpu.cuf"), "time_integration_rk_gpu")
    )
    assert "gpu_s1_flatplate_nscbc_farfield_filter_case()" in prepare
    assert "gpu_s1_flatplate_nscbc_farfield_case()" in runtime
    assert "if(flatplate_nscbc_farfield_case)prepared=.false." in integration


def test_gpu_nonowner_skips_the_upper_face_kernel_and_reports_ownership() -> None:
    boundary = source("src_gpu/boundary_gpu.cuf")
    wrapper = compact(
        function_body(
            boundary, "apply_nscbc_farfield_y_upper_nonreflecting_rhs_gpu"
        )
    )
    report = compact(
        function_body(boundary, "report_nscbc52_nonreflecting_owner")
    )

    assert "callreport_nscbc52_nonreflecting_owner(apply_up)" in wrapper
    assert wrapper.index("if(apply_up==0)return") < wrapper.index(
        "callnscbc_farfield_y_upper_nonreflecting_rhs_kernel"
    )
    assert "astr_validation_nscbc52_owner_trace" in report
    for field in ("mpirank", "irk", "jrk", "krk", "apply_up"):
        assert field in report


def test_legacy_gpu_nscbc_z_transverse_term_respects_cpu_x_active_range() -> None:
    boundary = source("src_gpu/boundary_gpu.cuf")
    x_inactive = (
        "(((npdci==1.or.npdci==4).and.i==0).or."
        "((npdci==2.or.npdci==4).and.i==im))"
    )
    z_inactive = (
        "(((npdck==1.or.npdck==4).and.k==0).or."
        "((npdck==2.or.npdck==4).and.k==km))"
    )
    kernel = compact(function_body(boundary, "nscbc_farfield_y_upper_rhs_kernel"))
    assert f"if(.not.{x_inactive}.and..not.{z_inactive})then" in kernel


def test_cpu_gpu_characteristic_masks_have_a_machine_zero_deadband() -> None:
    cpu = compact(
        function_body(source("src/bc.F90"), "nscbc_farfield_balance_incoming_lodi")
    )
    gpu = compact(
        function_body(
            source("src_gpu/nscbc_characteristic_policy_gpu.cuf"),
            "nscbc_balance_incoming_lodi_gpu",
        )
    )
    for body in (cpu, gpu):
        assert "64.d0*epsilon(1.d0)" in body
        assert "incoming=lambda<-wave_tolerance" in body


def test_curved_nonreflecting_capability_has_separate_flatplate_and_acoustic_contracts() -> None:
    capability = compact(
        function_body(
            source("src_gpu/case_capability_gpu.cuf"),
            "gpu_curvilinear_nscbc52_nonreflecting_supported",
        )
    )
    for required in (
        "ndims==3",
        "numq==5",
        "num_species==0",
        "num_modequ==0",
        "(.not.diffterm)",
        "(.not.lfilter)",
        "bctype(3)==41",
        "bctype(4)==52",
        "all(bctype(5:6)==1)",
        "nscbc_farfield_nonreflecting_enabled()",
        "flatplate_configuration",
        "trim(flowtype)=='bl'",
        "(.not.lihomo).and.(.not.ljhomo).and.lkhomo",
        "bctype(1)==11.and.bctype(2)==21",
        "trim(turbinf)=='prof'",
        "acoustic_box_configuration",
        "trim(flowtype)=='tgv'",
        "lihomo.and.(.not.ljhomo).and.lkhomo",
        "bctype(1)==1.and.bctype(2)==1",
        "flatplate_configuration.or.acoustic_box_configuration",
    ):
        assert required in capability


def test_gpu_validation_gate_accepts_the_curved_nonreflecting_capability() -> None:
    validation = compact(
        function_body(source("src_gpu/mainloop_gpu.cuf"), "validate_first_stage_gpu")
    )
    assert validation.count(
        "(.not.gpu_curvilinear_nscbc52_nonreflecting_supported())"
    ) >= 2


def test_gpu_physical_boundary_mask_is_zero_on_homogeneous_axes() -> None:
    boundary = compact(
        function_body(
            source("src_gpu/boundary_gpu.cuf"),
            "gpu_physical_boundary_apply_mask",
        )
    )
    guard = "if(.not.gpu_axis_is_physical(axis))return"
    assert guard in boundary
    assert boundary.index(guard) < boundary.index("selectcase(axis)")


def test_xyphysical_flux_uses_the_mpi_metric_halo_without_local_wrapping() -> None:
    body = compact(
        function_body(
            source("src_gpu/solver_gpu.cuf"), "flux_at_xyphysical_global"
        )
    )
    assert "idx_periodic" not in body
    assert "jacob_d(i,j,k)" in body


def test_nonreflecting_skips_cpu_boundary_filter() -> None:
    farfield = compact(function_body(source("src/bc.F90"), "farfield_nscbc"))
    assert (
        "if(.not.lfilter.and.nscbc_farfield_policy/="
        "nscbc_farfield_policy_nonreflecting)then" in farfield
    )
    mainloop = compact(
        function_body(source("src/mainloop.F90"), "time_integration_rk")
    )
    halo_contract = (
        "nscbc_boundary_halo_required="
        "any(bctype==22).or.any(bctype==52)"
    )
    assert mainloop.count(halo_contract) == 2
    assert (
        "if(nscbc_boundary_halo_required)callqswap(timerept=ltimrpt)"
        in mainloop
    )
    assert (
        "if(flowtype(1:2)/='0d'.and.nscbc_boundary_halo_required)then"
        in mainloop
    )


def test_curvilinear_nscbc52_matrix_covers_all_admitted_topologies() -> None:
    script = source("tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh")
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
        assert f"'{entry}'" in script
    assert "ASTR_VALIDATION_NSCBC52_OWNER_TRACE=1" in script
    assert "expected_owners = isize * ksize" in script
    assert "jrk != jsize - 1" in script
    assert "apply_up != 0" in script
    assert 'GRID="$GRID"' in script
    assert script.count('MAXSTEP="$MAXSTEP"') >= 2
    assert "MAXSTEP='440'" in script


def test_curvilinear_nscbc52_memcheck_covers_owner_and_nonowner_ranks() -> None:
    script = source("tests/gpu_validation/run_curvilinear_nscbc52_memcheck.sh")
    assert "'1:1,1,1'" in script
    assert "'2:1,2,1'" in script
    assert "compute-sanitizer --tool memcheck" in script
    assert "OMPI_MCA_pml=ob1" in script
    assert "OMPI_MCA_osc=pt2pt" in script
    assert "ERROR SUMMARY: 0 errors" in script


def test_curvilinear_nscbc52_profile_audits_legacy_work_and_rhs_count() -> None:
    script = source("tests/gpu_validation/run_curvilinear_nscbc52_profile.sh")
    assert "ASTR_NSCBC_FARFIELD_MODE=nonreflecting" in script
    assert "--trace=cuda" in script
    assert "--maxstep \"$((RK_STEPS-1))\"" in script
    assert "analyze_curvilinear_nscbc52_nsys.py" in script
