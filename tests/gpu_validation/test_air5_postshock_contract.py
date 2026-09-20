from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(path: str) -> str:
    return "".join((ROOT / path).read_text(encoding="utf-8").lower().split())


def test_postshock_grid_and_initializer_use_the_reference_profile() -> None:
    grid = compact("src/gridgeneration.F90")
    initializer = compact("src/initialisation.F90")

    assert "trim(flowtype)=='air5postshock'" in grid
    assert "callgridcube(ref_len,ref_len,ref_len)" in grid
    assert "case('air5postshock')" in initializer
    assert "callair5postshockini" in initializer
    assert "datin/air5_postshock_profile.dat" in initializer
    assert "profile_index=ig0+i" in initializer
    assert "callconfigure_air5_postshock_boundary" in initializer


def test_case_preparer_selects_physical_x_and_periodic_yz() -> None:
    preparer = compact("tests/gpu_validation/prepare_air5_c4_case.py")

    assert '"postshock"' in preparer
    assert 'replace_after_marker(lines,"lihomo,ljhomo,lkhomo","f,t,t")' in preparer
    assert 'replace_boundary_types(lines,("11,free",21,1,1,1,1))' in preparer
    assert '"500.d0,10.d0,2.d-2,1.d0"' in preparer


def test_cpu_boundary_sets_all_q11_x_halos_without_legacy_conversion() -> None:
    boundary = compact("src/chemistry_boundary.F90")

    assert "subroutineapply_air5_postshock_boundary" in boundary
    assert "docomponent=1,air5_num_conservative" in boundary
    assert "q(-hm:0,0:jm,0:km,component)=left_q(component)" in boundary
    assert "q(im:im+hm,0:jm,0:km,component)=right_q(component)" in boundary
    assert "mpileft==mpi_proc_null" in boundary
    assert "mpiright==mpi_proc_null" in boundary
    assert "fvar2q" not in boundary


def test_cpu_applies_postshock_boundary_before_halo_reconstruction() -> None:
    mainloop = compact("src/mainloop.F90")

    first_chemistry = mainloop.index("callair5_chemistry_half_step(0.5_real64*deltat,1)")
    first_boundary = mainloop.index("callapply_air5_postshock_boundary", first_chemistry)
    first_swap = mainloop.index("callqswap(timerept=ltimrpt)", first_chemistry)
    assert first_chemistry < first_boundary < first_swap
    assert "if(air5_postshock_case)then" in mainloop


def test_cpu_restores_postshock_boundary_after_each_rk_update() -> None:
    mainloop = compact("src/mainloop.F90")

    update = mainloop.index("callspongefilter")
    boundary = mainloop.index("callapply_air5_postshock_boundary()", update)
    primitive_refresh = mainloop.index("callupdatefvar", update)
    assert update < boundary < primitive_refresh


def test_gpu_boundary_is_resident_and_explicitly_synchronized() -> None:
    boundary = compact("src_gpu/chemistry_boundary_gpu.cuf")

    assert "attributes(global)subroutineair5_postshock_boundary_kernel" in boundary
    assert "q_d(i,j,k,component)=left_q_d(component)" in boundary
    assert "q_d(i,j,k,component)=right_q_d(component)" in boundary
    assert "callair5_postshock_boundary_kernel<<<grid,block>>>" in boundary
    assert "callsync_after_kernel('air5_postshock_boundary_kernel',.true.)" in boundary


def test_gpu_applies_boundary_before_first_half_step_halo_exchange() -> None:
    coupling = compact("src_gpu/chemistry_coupling_gpu.cuf")

    half_one = coupling.index("if(half_index==1)then")
    boundary = coupling.index("callapply_air5_postshock_boundary_gpu()", half_one)
    exchange = coupling.index("callexchange_solution_halo_gpu(.true.)", half_one)
    assert half_one < boundary < exchange


def test_gpu_postshock_transport_updates_only_active_nodes() -> None:
    solver = compact("src_gpu/chemistry_solver_gpu.cuf")

    assert "subroutineair5_transport_rk3_step_gpu" in solver
    for kernel in (
        "air5_convective_rhs_x_kernel",
        "air5_convective_rhs_y_kernel",
        "air5_convective_rhs_z_kernel",
        "air5_rk3_first_update_kernel",
        "air5_rk3_update_kernel",
    ):
        signature = solver.index(f"subroutine{kernel}")
        body = solver[signature : signature + 2500]
        assert "is,ie,js,je,ks,ke" in body
        assert "i<is.or.i>ie.or.j<js.or.j>je.or.k<ks.or.k>ke" in body


def test_gpu_convective_derivative_matches_cpu_physical_boundary_closure() -> None:
    solver = compact("src_gpu/chemistry_solver_gpu.cuf")

    assert "functionair5_deriv6_boundary" in solver
    assert "(ntype==1.or.ntype==4).and.index==1" in solver
    assert "derivative=0.5_real64*(fp1-fm1)" in solver
    assert "(ntype==1.or.ntype==4).and.index==2" in solver
    assert "derivative=num2d3*(fp1-fm1)-num1d12*(fp2-fm2)" in solver
    assert "(ntype==2.or.ntype==4).and.index==dim-1" in solver
    assert "(ntype==2.or.ntype==4).and.index==dim-2" in solver
    assert "air5_deriv6_boundary(" in solver


def test_postshock_boundary_modules_are_in_the_top_level_build() -> None:
    cmake = compact("src/CMakeLists.txt")

    assert "chemistry_boundary.f90" in cmake
    gpu_boundary = cmake.index("../src_gpu/chemistry_boundary_gpu.cuf")
    gpu_solver = cmake.index("../src_gpu/chemistry_solver_gpu.cuf")
    assert gpu_boundary < gpu_solver


def test_postshock_runner_covers_all_source_modes_and_independent_gate() -> None:
    runner = compact("tests/gpu_validation/run_air5_c5_postshock_compare.sh")

    assert "forsource_modeinchemicalvtcoupled" in runner
    assert "astr_air5_source_mode=\"$source_mode\"" in runner
    assert "diffterm=\"${diffterm:-f}\"" in runner
    assert "--diffterm\"$diffterm\"" in runner
    assert "generate_air5_postshock_profile.py" in runner
    assert "--source-mode\"$source_mode\"" in runner
    assert "check_air5_c5_postshock.py" in runner
    assert "--boundary-cut\"$boundary_cut\"" in runner
    assert "--labels" in runner
    assert "pre_chemistry,post_chemistry,pre_rhs,post_update,post_transport" in runner


def test_cpu_air5_diffusion_uses_physical_boundary_flux_closure() -> None:
    solver = compact("src/chemistry_solver.F90")

    assert "any([npdci,npdcj,npdck]/=3)" not in solver
    assert "subroutineprojected_air5_face_flux(" in solver
    assert "d0=-0.5_real64*fn(2,:)+2.0_real64*fn(1,:)-1.5_real64*fn(0,:)" in solver
    assert "d1=0.5_real64*(fn(2,:)-fn(0,:))" in solver
    assert "d2=0.5_real64*fn(3,:)-2.0_real64*fn(4,:)+1.5_real64*fn(5,:)" in solver
    assert "f=anchor-d2-d1-d0" in solver
    assert "f=anchor+d0+d1+d2" in solver


def test_gpu_air5_diffusion_closes_gradients_and_flux_divergence() -> None:
    solver = compact("src_gpu/chemistry_solver_gpu.cuf")

    assert "functionair5_deriv6_full_boundary" in solver
    assert "index==0" in solver
    assert "index==dim" in solver
    assert (
        "subroutineair5_species_tv_gradient_kernel(im,jm,km,npdci,npdcj,npdck,"
        "&zero_species_flux_lower_y)"
    ) in solver
    assert "callgradcal_dvel_xphysical_kernel<<<grid3,block3>>>" in solver
    for direction, ntype in (("x", "npdci"), ("y", "npdcj"), ("z", "npdck")):
        signature = (
            f"subroutineair5_diffusion_rhs_{direction}_kernel(im,jm,km,{ntype},"
            "is,ie,js,je,ks,ke)"
        )
        assert signature in solver
    assert "subroutineair5_diffusive_face_flux(" in solver
    assert "callair5_diffusive_face_flux(i,j,k,1,i-1,im,npdci,left_flux)" in solver
    assert "callair5_diffusive_face_flux(i,j,k,2,j-1,jm,npdcj,left_flux)" in solver
    assert "callair5_diffusive_face_flux(i,j,k,3,k-1,km,npdck,left_flux)" in solver
