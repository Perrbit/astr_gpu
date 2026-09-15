from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(path: str) -> str:
    return "".join((ROOT / path).read_text(encoding="utf-8").lower().split())


def test_air5_hbl_is_a_dedicated_reacting_flowtype() -> None:
    runtime = compact("src/chemistry_flow_runtime.F90")

    assert "air5hbl" in runtime
    assert "public::air5_hbl_flowtype" in runtime


def test_hbl_initializer_uses_the_versioned_complete_profile() -> None:
    initializer = compact("src/initialisation.F90")
    grid = compact("src/gridgeneration.F90")

    assert "case('air5hbl')" in initializer
    assert "callair5hblini" in initializer
    assert "datin/air5_hbl_profile.dat" in initializer
    assert "callconfigure_air5_hbl_boundary" in initializer
    assert "trim(flowtype)=='air5hbl'" in grid


def test_cpu_hbl_boundary_is_applied_at_every_required_phase() -> None:
    mainloop = compact("src/mainloop.F90")

    assert "usechemistry_hbl_boundary,only:apply_air5_hbl_boundary" in mainloop
    assert mainloop.count("callapply_air5_hbl_boundary()") >= 4


def test_cpu_hbl_does_not_retain_backend_specific_debug_phases() -> None:
    mainloop = compact("src/mainloop.F90")

    assert "post_chemistry_raw" not in mainloop
    assert "post_updatefvar" not in mainloop


def test_gpu_hbl_boundary_updates_q11_and_synchronizes_each_face_kernel() -> None:
    boundary = compact("src_gpu/chemistry_hbl_boundary_gpu.cuf")

    for kernel in (
        "air5_hbl_outflow_boundary_kernel",
        "air5_hbl_wall_boundary_kernel",
        "air5_hbl_farfield_boundary_kernel",
        "air5_hbl_inflow_boundary_kernel",
    ):
        assert f"call{kernel}<<<" in boundary
        assert f"callsync_after_kernel('{kernel}',.true.)" in boundary
    assert "q_d(i,j,k,1:air5_num_conservative)" in boundary
    assert "fvar2q" not in boundary


def test_gpu_hbl_status_collectives_are_called_by_every_rank_in_face_order() -> None:
    boundary = compact("src_gpu/chemistry_hbl_boundary_gpu.cuf")
    start = boundary.index("subroutineapply_air5_hbl_boundary_gpu()")
    body = boundary[start : boundary.index("endsubroutineapply_air5_hbl_boundary_gpu", start)]

    outflow_if = body.index("if(mpiright==mpi_proc_null)then")
    outflow_end = body.index("endif", outflow_if)
    outflow_check = body.index("callcheck_hbl_status('outflow')", outflow_if)
    wall_if = body.index("if(mpidown==mpi_proc_null)then")
    wall_end = body.index("endif", wall_if)
    wall_check = body.index("callcheck_hbl_status('wall')", wall_if)
    assert outflow_end < outflow_check < wall_if
    assert wall_end < wall_check


def test_gpu_hbl_diffusion_uses_xy_physical_gradient_closure() -> None:
    solver = compact("src_gpu/chemistry_flow_solver_gpu.cuf")

    assert "air5_hbl_flowtype" in solver
    assert "callgradcal_dvel_xyphysical_kernel<<<" in solver
    assert "fixedair5hbltransportrequiresphysicalx/yandperiodicz" in solver


def test_hbl_noncatalytic_species_flux_is_zeroed_before_flux_construction() -> None:
    cpu_solver = compact("src/chemistry_flow_solver.F90")
    gpu_solver = compact("src_gpu/chemistry_flow_solver_gpu.cuf")

    assert "dspc(:,0,:,:,2)=0.0_real64" in cpu_solver
    assert "zero_species_flux_lower_y" in gpu_solver
    assert "if(zero_species_flux_lower_y/=0.and.j==0)" in gpu_solver
    assert "dspc_d(i,j,k,species,2)=0.0_real64" in gpu_solver


def test_hbl_modules_are_ordered_before_their_consumers() -> None:
    cmake = compact("src/CMakeLists.txt")

    profile = cmake.index("chemistry_hbl_profile.f90")
    state = cmake.index("chemistry_hbl_boundary_state.f90")
    boundary = cmake.index("chemistry_hbl_boundary.f90")
    runtime = cmake.index("chemistry_flow_solver.f90")
    gpu_boundary = cmake.index("../src_gpu/chemistry_hbl_boundary_gpu.cuf")
    gpu_solver = cmake.index("../src_gpu/chemistry_flow_solver_gpu.cuf")
    assert profile < boundary < runtime
    assert state < boundary < runtime
    assert gpu_boundary < gpu_solver


def test_case_preparer_can_build_the_htr_hbl_contract() -> None:
    preparer = compact("tests/gpu_validation/prepare_air5_c4_case.py")

    assert '"high-enthalpy-boundary-layer"' in preparer
    assert 'replace_after_marker(lines,"lihomo,ljhomo,lkhomo","f,f,t")' in preparer
    assert 'replace_boundary_types(lines,("11,free",50,"41,2925.d0",51,1,1))' in preparer
    assert "air5_hbl_profile.dat" in preparer
