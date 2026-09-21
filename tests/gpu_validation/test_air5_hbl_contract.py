from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def compact(path: str) -> str:
    return "".join((ROOT / path).read_text(encoding="utf-8").lower().split())


def test_air5_hbl_is_a_dedicated_reacting_flowtype() -> None:
    runtime = compact("src/chemistry_runtime.F90")

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


def test_hbl_initializer_optionally_uses_matched_xy_conservative_field() -> None:
    state = compact("src/chemistry_boundary_state.F90")
    initializer = compact("src/initialisation.F90")
    preparer = compact("tests/gpu_validation/prepare_air5_c4_case.py")
    runner = compact("tests/gpu_validation/run_air5_c5_hbl_compare.sh")

    assert "modulechemistry_hbl_initial_field" in state
    assert "read_air5_hbl_initial_field" in state
    assert "sample_air5_hbl_initial_field" in state
    assert "datin/air5_hbl_initial_field.dat" in initializer
    assert "inquire(file='datin/air5_hbl_initial_field.dat',exist=has_initial_field)" in initializer
    assert "callsample_air5_hbl_initial_field(initial_field,x(i,j,k,1),x(i,j,k,2)" in initializer
    assert 'choices=("uniform","matched")' in preparer
    assert "generate_similarity_initial_field" in preparer
    assert "write_astr_air5_initial_field" in preparer
    assert 'hbl_initial_field="${hbl_initial_field:-uniform}"' in runner
    assert '--hbl-initial-field"$hbl_initial_field"' in runner


def test_fixed_air5_restart_persists_independent_vibrational_temperature() -> None:
    initializer = compact("src/initialisation.F90")
    readwrite = compact("src/readwrite.F90")

    assert "fixedair5restartisunavailable" not in initializer
    assert "callreadcheckpoint(folder='outdat',mode='h')" in initializer
    assert "if(lcomb.and.trim(flowtype)=='air5hbl')callair5hblboundaryini" in initializer
    assert "callupdateq" in initializer
    assert "callh5read(varname='tv',var=tve(" in readwrite
    assert "callh5wa3d_r8_struct(varname='tv',var=data2write" in readwrite
    assert "if(lcomb)then" in readwrite
    assert "callpgather_across_k(array=tve(0:im,0:jm,1:km)" in readwrite


def test_fixed_air5_restart_runner_compares_complete_q11_state() -> None:
    runner = compact("tests/gpu_validation/run_air5_c5_hbl_restart.sh")

    assert "continuous_step=4" in runner
    assert "split_step=2" in runner
    assert "hbl_initial_field=matched" in runner
    assert "set_restart_case" in runner
    assert "checkpointfileread" in runner
    assert "compare_q_validation_snapshots.py" in runner
    assert runner.count("compare_restart") >= 3
    assert "--labelspost_chemistry,pre_rhs,post_update,post_transport" in runner
    assert "--scaled-tol5e-9" in runner
    assert "air5_hbl_restart_equivalence_pass" in runner


def test_gpu_hbl_checkpoint_does_not_replay_generic_host_boundary() -> None:
    mainloop = compact("src/mainloop.F90")

    checkpoint = mainloop.index("if(gpu_checkpoint_due.or.gpu_slice_due)then")
    integration = mainloop.index("callgpu_time_integration_rk", checkpoint)
    body = mainloop[checkpoint:integration]
    assert ".not.air5_postshock_case" in body
    assert ".not.air5_hbl_case" in body


def test_cpu_hbl_boundary_is_applied_at_every_required_phase() -> None:
    mainloop = compact("src/mainloop.F90")

    assert "usechemistry_hbl_boundary,only:apply_air5_hbl_boundary" in mainloop
    assert mainloop.count("callapply_air5_hbl_boundary()") >= 4


def test_cpu_chemistry_half_steps_reconstruct_primitives_after_hbl_boundary() -> None:
    mainloop = compact("src/mainloop.F90")

    first = mainloop.index("callair5_chemistry_half_step(0.5_real64*deltat,1)")
    first_boundary = mainloop.index("callapply_air5_hbl_boundary()", first)
    first_update = mainloop.index("callupdatefvar", first)
    first_halo = mainloop.index("callqswap", first)
    assert first < first_boundary < first_update < first_halo

    second = mainloop.index("callair5_chemistry_half_step(0.5_real64*deltat,2)")
    second_boundary = mainloop.index("callapply_air5_hbl_boundary()", second)
    second_update = mainloop.index("callupdatefvar", second)
    assert second < second_boundary < second_update


def test_cpu_hbl_does_not_retain_backend_specific_debug_phases() -> None:
    mainloop = compact("src/mainloop.F90")

    assert "post_chemistry_raw" not in mainloop
    assert "post_updatefvar" not in mainloop


def test_gpu_hbl_boundary_updates_q11_and_synchronizes_each_face_kernel() -> None:
    boundary = compact("src_gpu/chemistry_boundary_gpu.cuf")

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


def test_gpu_chemistry_reconstructs_active_primitives_after_hbl_boundary() -> None:
    coupling = compact("src_gpu/chemistry_coupling_gpu.cuf")

    first = coupling.index("if(half_index==1)then")
    boundary = coupling.index("callapply_air5_hbl_boundary_gpu()", first)
    interior = coupling.index("calllaunch_air5_interior_primitive_gpu()", first)
    halo = coupling.index("callexchange_solution_halo_gpu(.true.)", first)
    faces = coupling.index("calllaunch_air5_face_primitives_gpu()", first)
    assert first < boundary < interior < halo < faces

    second = coupling.index("elseif(half_index==2)then")
    second_boundary = coupling.index("callapply_air5_hbl_boundary_gpu()", second)
    second_interior = coupling.index("calllaunch_air5_interior_primitive_gpu()", second)
    assert second < second_boundary < second_interior


def test_gpu_transport_reapplies_air5_boundaries_after_each_rk_update() -> None:
    solver = compact("src_gpu/chemistry_solver_gpu.cuf")
    start = solver.index("subroutineair5_transport_rk3_step_gpu")
    body = solver[start : solver.index("endsubroutineair5_transport_rk3_step_gpu", start)]

    update_end = body.index("endif", body.index("callair5_rk3_first_update_kernel<<<"))
    postshock = body.index("callapply_air5_postshock_boundary_gpu()", update_end)
    hbl = body.index("callapply_air5_hbl_boundary_gpu()", update_end)
    snapshot = body.index("callwrite_q_validation_snapshot('post_update'", update_end)
    assert update_end < postshock < hbl < snapshot


def test_air5_filter_supports_full_and_scalar_q11_workspaces() -> None:
    solver = compact("src_gpu/solver_gpu.cuf")
    chemistry = compact("src_gpu/chemistry_solver_gpu.cuf")
    preparer = compact("tests/gpu_validation/prepare_air5_c4_case.py")
    runner = compact("tests/gpu_validation/run_air5_c5_hbl_compare.sh")

    assert "subroutineapply_explicit_filter_gpu" in solver
    for kernel in (
        "filter_x_periodic_interior_global_kernel",
        "filter_x_periodic_boundary_global_kernel",
        "filter_y_periodic_interior_global_kernel",
        "filter_y_periodic_boundary_global_kernel",
        "filter_z_periodic_interior_global_kernel",
        "filter_z_periodic_boundary_global_kernel",
        "filter_x_global_kernel",
        "filter_y_pong_to_q_halo_global_kernel",
        "filter_z_halo_global_kernel",
        "copy_qwork_global_kernel",
    ):
        start = solver.index(f"subroutine{kernel}")
        end = solver.index(f"endsubroutine{kernel}", start)
        body = solver[start:end]
        assert "ncomponents" in body
        assert "dom=1,ncomponents" in body
    assert "docomponent=1,numq" in solver
    assert "callapply_explicit_filter_gpu" in chemistry
    save = chemistry.index("callair5_save_filter_species_base_kernel<<<")
    apply_filter = chemistry.index("callapply_explicit_filter_gpu", save)
    limit_state = chemistry.index("callair5_limit_filtered_state_kernel<<<", apply_filter)
    assert save < apply_filter < limit_state
    save_start = chemistry.index("subroutineair5_save_filter_species_base_kernel")
    save_end = chemistry.index("endsubroutineair5_save_filter_species_base_kernel")
    save_body = chemistry[save_start:save_end]
    assert "air5_idx_density" in save_body
    assert "air5_idx_species_first" in save_body
    assert "do component=1,air5_num_conservative" not in save_body
    assert "air5_filter_roundoff_relative_tolerance" not in chemistry
    assert "nearest(" in chemistry
    assert "closure_species" in chemistry
    assert "low_species" in chemistry
    assert "theta" in chemistry
    limiter_start = chemistry.index("subroutineair5_limit_filtered_state_kernel")
    limiter_end = chemistry.index("endsubroutineair5_limit_filtered_state_kernel")
    assert "candidate_state" not in chemistry[limiter_start:limiter_end]
    assert "callair5_limit_filtered_state()" in compact("src/mainloop.F90")
    assert "lfilter.or.lchardecomp" not in chemistry
    assert 'parser.add_argument("--lfilter",choices=("t","f"),default="f")' in preparer
    assert "f\"f,{diffterm},{lfilter},f,f,f,t,t,{use_gpu}\"" in preparer
    assert 'lfilter="${lfilter:-t}"' in runner
    assert 'filter_workspace="${filter_workspace:-full}"' in runner
    assert '--lfilter"$lfilter"' in runner
    assert 'astr_gpu_filter_workspace="$filter_workspace"' in runner


def test_gpu_hbl_outflow_limits_species_before_full_state_fallback() -> None:
    boundary = compact("src_gpu/chemistry_boundary_gpu.cuf")

    species_limit = boundary.index("species_limited=.false.")
    species_closure = boundary.index(
        "q_trial(air5_idx_species_first)=q_high(air5_idx_density)"
    )
    full_state_limit = boundary.index("theta_upper=1.0_real64", species_closure)
    assert species_limit < species_closure < full_state_limit
    assert "callbuild_air5_hbl_outflow_state_gpu" in boundary


def test_gpu_hbl_status_collectives_are_called_by_every_rank_in_face_order() -> None:
    boundary = compact("src_gpu/chemistry_boundary_gpu.cuf")
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
    solver = compact("src_gpu/chemistry_solver_gpu.cuf")

    assert "air5_hbl_flowtype" in solver
    assert "callgradcal_dvel_xyphysical_kernel<<<" in solver
    assert "fixedair5hbltransportrequiresphysicalx/yandperiodicz" in solver


def test_hbl_noncatalytic_species_flux_is_zeroed_before_flux_construction() -> None:
    cpu_solver = compact("src/chemistry_solver.F90")
    gpu_solver = compact("src_gpu/chemistry_solver_gpu.cuf")

    assert "dspc(:,0,:,:,2)=0.0_real64" in cpu_solver
    assert "zero_species_flux_lower_y" in gpu_solver
    assert "if(zero_species_flux_lower_y/=0.and.j==0)" in gpu_solver
    assert "dspc_d(i,j,k,species,2)=0.0_real64" in gpu_solver


def test_hbl_modules_are_ordered_before_their_consumers() -> None:
    cmake = compact("src/CMakeLists.txt")

    state = cmake.index("chemistry_boundary_state.f90")
    boundary = cmake.index("chemistry_boundary.f90")
    runtime = cmake.index("chemistry_solver.f90")
    gpu_boundary = cmake.index("../src_gpu/chemistry_boundary_gpu.cuf")
    gpu_solver = cmake.index("../src_gpu/chemistry_solver_gpu.cuf")
    assert state < boundary < runtime
    assert gpu_boundary < gpu_solver


def test_case_preparer_can_build_the_htr_hbl_contract() -> None:
    preparer = compact("tests/gpu_validation/prepare_air5_c4_case.py")

    assert '"high-enthalpy-boundary-layer"' in preparer
    assert 'replace_after_marker(lines,"lihomo,ljhomo,lkhomo","f,f,t")' in preparer
    assert 'replace_boundary_types(lines,("11,free",50,"41,2925.d0",51,1,1))' in preparer
    assert "air5_hbl_profile.dat" in preparer
