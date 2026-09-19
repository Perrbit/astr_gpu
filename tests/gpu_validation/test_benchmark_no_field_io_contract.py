import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SRC_GPU = ROOT / "src_gpu"


def _source(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line.split("!", 1)[0] for line in lines)


def _compact(source: str) -> str:
    return re.sub(r"[\s&]+", "", source).lower()


ASTR = _source(SRC / "astr.F90")
GRID = _source(SRC / "gridgeneration.F90")
INIT = _source(SRC / "initialisation.F90")
MAINLOOP = _source(SRC / "mainloop.F90")
BENCHMARK = _source(SRC / "benchmark_runtime.F90")
GPU_RUNTIME = _source(SRC_GPU / "gpu_runtime.cuf")
GPU_LOOP = _source(SRC_GPU / "mainloop_gpu.cuf")
GPU_PHASE = _source(SRC_GPU / "gpu_phase_timing.cuf")


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
    assert (
        "if(.not.benchmark_field_io_disabled())callwritegrid(trim(gridfile))"
        in _compact(GRID)
    )
    assert (
        "if(.not.benchmark_field_io_disabled())callwriteflfed(timerept=.true.)"
        in _compact(INIT)
    )


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


def test_benchmark_admission_requires_independent_periodic_boundary_codes() -> None:
    source = _compact(BENCHMARK)
    assert "periodic_boundary_case" in source
    assert ".not.periodic_boundary_case" in source


def test_no_field_io_supports_matched_cpu_gpu_performance_cases() -> None:
    source = _compact(BENCHMARK)
    assert "trim(flowtype)=='tgv'" in source
    assert "trim(flowtype)=='shuosher'" in source
    assert "use_gpu.and.rk_timing/=1" in source
    assert ".not.cuda_build" not in source


def test_cpu_rk_timing_is_benchmark_only_and_matches_gpu_output_schema() -> None:
    benchmark = _compact(BENCHMARK)
    mainloop = _compact(MAINLOOP)

    assert "switch_choice('astr_cpu_rk_timing')" in benchmark
    assert "benchmark_cpu_rk_timing_enabled" in benchmark
    assert "cpu_rk_timing==1.and.(use_gpu.or..not.field_io_disabled)" in benchmark

    start = mainloop.index("time_beg=ptime()")
    advance = mainloop.index("calltime_integration_rk", start)
    output = mainloop.index("'astr_cpu_rk_timing'", advance)
    assert start < advance < output
    assert "3(1x,es24.16e3)" in mainloop[output - 160 : output]
    assert "0.d0,cpu_rk_seconds,cpu_rk_seconds" in mainloop[output : output + 200]


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


def test_phase_timing_admission_requires_complete_explicit_tgv_schema() -> None:
    source = _compact(GPU_RUNTIME)
    assert "allow_selective=trim(flowtype)=='tgv'.and.ndims==3" in source
    assert "lihomo.and.ljhomo.and.lkhomo" in source
    assert (
        "allow_phase_timing=allow_selective.and.gpu_periodic_boundary_case()"
        ".and.lfilter.and.diffterm" in source
    )
    assert (
        "callconfigure_gpu_phase_timing"
        "(allow_phase_timing,.not.(gpu_selective_sync_enabled().or."
        "gpu_dependency_sync_enabled()))" in source
    )


def test_phase_timing_policy_is_collective_and_stack_checks_are_bounds_safe() -> None:
    source = _compact(GPU_PHASE)
    assert "switch_choice('astr_gpu_phase_timing')" in source
    assert source.count("callmpi_allreduce(requested") == 2
    assert "if(lowest<0.or.lowest/=highest)then" in source
    assert "if(enabled.and.(.not.allow_tgv.or..not.allow_explicit))then" in source
    assert "if(depth>=max_depth)then" in source
    assert "if(depth<1)then" in source
    assert "if(trim(labels(depth))/=trim(label))then" in source

    disabled_return = source.index("if(.not.enabled)return", source.index("subroutineend_gpu_phase"))
    sync = source.index("callsync_gpu_boundary('phase_'//trim(label))")
    assert disabled_return < sync
