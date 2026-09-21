from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SRC_GPU = ROOT / "src_gpu"
CMAKE = (SRC / "CMakeLists.txt").read_text(encoding="utf-8").lower()

CPU_LAYOUT = {
    "chemistry_air5_data.F90": {"chemistry_air5_data"},
    "chemistry_core.F90": {
        "chemistry_model",
        "chemistry_state_layout",
    },
    "chemistry_properties.F90": {
        "chemistry_thermo",
        "chemistry_flow_state",
        "chemistry_transport",
        "chemistry_relaxation",
    },
    "chemistry_kinetics.F90": {
        "chemistry_source",
        "chemistry_linear6",
        "chemistry_ros2",
    },
    "chemistry_boundary_state.F90": {
        "chemistry_hbl_initial_field",
        "chemistry_hbl_profile",
        "chemistry_hbl_boundary_state",
    },
    "chemistry_boundary.F90": {
        "chemistry_postshock_boundary",
        "chemistry_hbl_boundary",
    },
    "chemistry_runtime.F90": {"chemistry_flow_runtime"},
    "chemistry_solver.F90": {"chemistry_flow_solver"},
}

GPU_LAYOUT = {
    "chemistry_core_gpu.cuf": {
        "chemistry_model_gpu",
        "chemistry_thermo_gpu",
    },
    "chemistry_flow_state_gpu.cuf": {"chemistry_flow_state_gpu"},
    "chemistry_transport_gpu.cuf": {"chemistry_transport_gpu"},
    "chemistry_relaxation_gpu.cuf": {"chemistry_relaxation_gpu"},
    "chemistry_kinetics_gpu.cuf": {
        "chemistry_linear6_gpu",
        "chemistry_source_gpu",
        "chemistry_ros2_gpu",
    },
    "chemistry_boundary_gpu.cuf": {
        "chemistry_postshock_boundary_gpu",
        "chemistry_hbl_boundary_gpu",
    },
    "chemistry_solver_gpu.cuf": {"chemistry_flow_solver_gpu"},
    "chemistry_coupling_gpu.cuf": {"chemistry_coupling_gpu"},
}


def modules(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"(?im)^module\s+(?!procedure\b)(\w+)", text))


def test_cpu_chemistry_uses_grouped_compilation_units() -> None:
    for filename, expected_modules in CPU_LAYOUT.items():
        path = SRC / filename
        assert path.is_file()
        assert modules(path) == expected_modules
        assert filename.lower() in CMAKE


def test_gpu_chemistry_uses_grouped_compilation_units() -> None:
    for filename, expected_modules in GPU_LAYOUT.items():
        path = SRC_GPU / filename
        assert path.is_file()
        assert modules(path) == expected_modules
        assert f"../src_gpu/{filename.lower()}" in CMAKE


def test_no_legacy_chemistry_compilation_units_remain() -> None:
    current = set(CPU_LAYOUT) | set(GPU_LAYOUT)
    legacy = {
        path.name
        for directory, suffix in ((SRC, ".F90"), (SRC_GPU, ".cuf"))
        for path in directory.glob(f"chemistry_*{suffix}")
        if path.name not in current
    }
    assert legacy == set()
