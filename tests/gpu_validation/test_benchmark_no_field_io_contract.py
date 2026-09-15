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
        "(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo)"
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
