from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(path: str) -> str:
    return "".join((ROOT / path).read_text(encoding="utf-8").lower().split())


def test_normal_shock_is_a_reacting_open_x_flowtype() -> None:
    runtime = compact("src/chemistry_runtime.F90")

    assert "'air5normalshock'" in runtime
    assert "air5_normal_shock_flowtype" in runtime
    assert "air5_open_x_flowtype" in runtime


def test_normal_shock_initializer_reads_independent_jump_states() -> None:
    grid = compact("src/gridgeneration.F90")
    initializer = compact("src/initialisation.F90")

    assert "trim(flowtype)=='air5normalshock'" in grid
    assert "case('air5normalshock')" in initializer
    assert "subroutineair5normalshockini" in initializer
    assert "datin/air5_normal_shock_states.dat" in initializer
    assert "callconfigure_air5_postshock_boundary" in initializer


def test_normal_shock_uses_inflow_and_extrapolating_outflow_on_cpu_and_gpu() -> None:
    cpu = compact("src/chemistry_boundary.F90")
    gpu = compact("src_gpu/chemistry_boundary_gpu.cuf")

    assert "normal_shock_case" in cpu
    assert "q(i,0:jm,0:km,component)=q(im-1,0:jm,0:km,component)" in cpu
    assert "normal_shock_case" in gpu
    assert "q_d(i,j,k,component)=q_d(im-1,j,k,component)" in gpu
    assert "sync_after_kernel('air5_postshock_boundary_kernel',.true.)" in gpu


def test_case_preparer_selects_normal_shock_numq11_mode() -> None:
    preparer = compact("tests/gpu_validation/prepare_air5_c4_case.py")

    assert 'initial_condition=="normal-shock"' in preparer
    assert "air5normalshock" in preparer
    assert 'replace_after_marker(lines,"lihomo,ljhomo,lkhomo","f,t,t")' in preparer
    assert 'replace_boundary_types(lines,("11,free",21,1,1,1,1))' in preparer
    assert 'initial_conditionin("shock-tube","normal-shock")' in preparer


def test_normal_shock_has_a_coupled_cpu_gpu_validation_driver() -> None:
    runner = compact("tests/gpu_validation/run_air5_c5_normal_shock_compare.sh")
    checker = compact("tests/gpu_validation/check_air5_c5_normal_shock.py")

    assert "generate_air5_normal_shock_states.py" in runner
    assert "--initial-conditionnormal-shock" in runner
    assert "astr_air5_source_mode=coupled" in runner
    assert "compare_q_validation_snapshots.py" in runner
    assert "check_air5_c5_normal_shock.py" in runner
    assert "marked_shock_interfaces" in checker
    assert "upstream_mach" in checker
    assert "downstream_mach" in checker
