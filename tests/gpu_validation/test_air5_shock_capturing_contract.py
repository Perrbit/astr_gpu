from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(path: Path) -> str:
    return "".join(path.read_text(encoding="utf-8").lower().split())


def test_air5_shock_mode_has_an_explicit_runtime_contract() -> None:
    runtime = compact(ROOT / "src/chemistry_runtime.F90")

    assert "air5_shock_capturing_enabled" in runtime
    assert "trim(conschm)=='643e'" in runtime
    assert "recon_schem==3" in runtime
    assert ".not.lchardecomp" in runtime


def test_cpu_numq11_flux_uses_selective_llf_mp7() -> None:
    solver = compact(ROOT / "src/chemistry_solver.F90")

    for token in (
        "air5_frozen_spectral_radius",
        "air5_llf_split_flux",
        "air5_mp7",
        "air5_selective_convective_face_flux",
        "air5_shock_interface_active",
        "public::air5_convection_rhs",
    ):
        assert token in solver
    assert "fplus=0.5_real64*(flux+alpha*metric_jacobian*state)" in solver
    assert "fminus=0.5_real64*(flux-alpha*metric_jacobian*state)" in solver


def test_gpu_numq11_flux_matches_cpu_structure_and_keeps_explicit_sync() -> None:
    solver = compact(ROOT / "src_gpu/chemistry_solver_gpu.cuf")

    for token in (
        "air5_frozen_spectral_radius_gpu",
        "air5_llf_split_flux_gpu",
        "air5_mp7_gpu",
        "air5_selective_convective_face_flux_gpu",
        "air5_shock_interface_active_gpu",
        "air5_selective_convective_rhs_x_kernel",
        "air5_selective_convective_rhs_y_kernel",
        "air5_selective_convective_rhs_z_kernel",
    ):
        assert token in solver
    for kernel in ("x", "y", "z"):
        assert (
            f"sync_after_kernel('air5_selective_convective_rhs_{kernel}_kernel',.true.)"
            in solver
        )


def test_air5_shock_mode_drives_the_existing_ducros_sensor() -> None:
    cpu_solver = compact(ROOT / "src/solver.F90")
    gpu_solver = compact(ROOT / "src_gpu/chemistry_solver_gpu.cuf")
    arrays = compact(ROOT / "src_gpu/commarray_gpu.cuf")
    sensor = compact(ROOT / "src_gpu/shock_sensor_gpu.cuf")

    assert "air5_shock_capturing_enabled()" in cpu_solver
    assert "callducrossensor" in cpu_solver
    assert "callcompute_shock_sensor_gpu(rkstep,.true.)" in gpu_solver
    assert "air5_shock_capturing_enabled()" in arrays
    assert "force_active" in sensor


def test_numq11_shock_stencil_requires_four_halo_layers() -> None:
    cpu_solver = compact(ROOT / "src/chemistry_solver.F90")
    gpu_solver = compact(ROOT / "src_gpu/chemistry_solver_gpu.cuf")
    gpu_arrays = compact(ROOT / "src_gpu/commarray_gpu.cuf")

    assert "air5shockcapturingrequireshm>=4" in cpu_solver
    assert "air5shockcapturingrequireshm>=4" in gpu_solver
    assert "if(lcomb)thenif(.not.allocated(diffusion_ratio_d))" in gpu_arrays


def test_numq11_periodic_interface_pairs_shared_plane_sensor_decisions() -> None:
    cpu_solver = compact(ROOT / "src/chemistry_solver.F90")
    gpu_solver = compact(ROOT / "src_gpu/chemistry_solver_gpu.cuf")

    boundary_pair = "face==0.and.(ntype==2.or.ntype==3)"
    opposite_pair = "face==dim-1.and.(ntype==1.or.ntype==3)"
    assert boundary_pair in cpu_solver
    assert opposite_pair in cpu_solver
    assert boundary_pair in gpu_solver
    assert opposite_pair in gpu_solver


def test_air5_shock_tube_routes_through_the_numq11_mode() -> None:
    initialization = compact(ROOT / "src/initialisation.F90")
    grid = compact(ROOT / "src/gridgeneration.F90")
    preparer = compact(ROOT / "tests/gpu_validation/prepare_air5_c4_case.py")

    assert "case('air5shocktube')" in initialization
    assert "subroutineair5shocktubeini" in initialization
    assert "target_temperature=1.0e3_real64" in initialization
    assert "trim(flowtype)=='air5shocktube'" in grid
    assert 'initial_condition=="shock-tube"' in preparer
    assert '"3,f,0.3d0,0.05d0"' in preparer


def test_air5_shock_tube_has_a_frozen_cpu_gpu_validation_driver() -> None:
    runner = compact(ROOT / "tests/gpu_validation/run_air5_numq11_shock_tube_compare.sh")
    checker = compact(ROOT / "tests/gpu_validation/check_air5_numq11_shock_tube.py")
    runtime = compact(ROOT / "src/chemistry_runtime.F90")

    assert "initial-conditionshock-tube" in runner
    assert "--difftermf" in runner
    assert "compare_q_validation_snapshots.py" in runner
    assert "check_air5_numq11_shock_tube.py" in runner
    assert "case('air5shocktube')" not in runtime
    assert "mass_closure" in checker
    assert "minimum_species_density" in checker
    assert "marked_shock_interfaces" in checker
