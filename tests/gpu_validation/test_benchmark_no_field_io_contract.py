from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASTR = (ROOT / "src/astr.F90").read_text(encoding="utf-8")
GRID = (ROOT / "src/gridgeneration.F90").read_text(encoding="utf-8")
INIT = (ROOT / "src/initialisation.F90").read_text(encoding="utf-8")


def test_policy_is_configured_before_grid_and_flow_initialization() -> None:
    call = "call configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo)"
    assert call in ASTR
    assert ASTR.index(call) < ASTR.index("call gridgen")
    assert ASTR.index(call) < ASTR.index("call flowinit")


def test_generated_grid_write_is_the_only_grid_guard() -> None:
    assert "if(.not.benchmark_field_io_disabled()) call writegrid(trim(gridfile))" in GRID
    assert "if(lreadgrid) then" in GRID
    assert "call readgrid(trim(gridfile))" in GRID


def test_startup_field_write_has_one_narrow_guard() -> None:
    assert "if(.not.benchmark_field_io_disabled()) call writeflfed(timerept=.true.)" in INIT
    assert INIT.count("benchmark_field_io_disabled()") == 1
