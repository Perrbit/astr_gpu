"""Admission and wiring contracts; numerical evidence comes from Fortran probes."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class M12IntegrationContractTests(unittest.TestCase):
    def test_mp_positivity_is_opt_in_and_allows_explicit_crashfix(self):
        source = (ROOT/'src_gpu/mp_positivity_gpu.cuf').read_text()
        self.assertIn('mp_positivity_enabled=.false.', source)
        self.assertIn("get_environment_variable('ASTR_GPU_MP_POSITIVITY'", source)
        self.assertIn("trim(value)=='flux'", source)
        self.assertIn(".not.lfilter.and.trim(conschm)=='743e'", source)
        self.assertNotIn('.not.lcracon', source)
        self.assertNotIn('lfilter.or.lcracon', source)
        self.assertIn('numq==5.and.num_species==0.and.num_modequ==0', source)
        self.assertNotIn('copy_flow_from_gpu', source)
        self.assertIn("check_mp_failure('complete low-order baseline')", source)
        self.assertIn("check_mp_failure('complete limited RK update')", source)

    def test_mp_positivity_limits_complete_rhs_before_rk_update(self):
        source = (ROOT/'src_gpu/mainloop_gpu.cuf').read_text()
        start = source.index('call apply_case_sources_gpu()')
        end = source.index("call begin_gpu_phase('rk_update')", start)
        self.assertIn('call limit_mp_update_gpu(', source[start:end])
        source = (ROOT/'src_gpu/mp_positivity_gpu.cuf').read_text()
        self.assertIn('call exchange_mp_planes(axis,3)', source)
        self.assertIn('call exchange_field_halo_gpu(ratio_d,1)', source)
        self.assertIn('tag=21401+2*(axis-1)', source)
        self.assertIn('! Physical boundary nodes have their own RHS', source)

    def test_mp_positivity_is_part_of_exact_restart_contract(self):
        source = (ROOT/'src/output_runtime.F90').read_text()
        self.assertIn("get_environment_variable('ASTR_GPU_MP_POSITIVITY'", source)
        self.assertIn("if(trim(value)=='flux') contract(7)=ibset(contract(7),1)", source)

    def test_new_output_does_not_forbid_crashfix(self):
        source = (ROOT/'src/output_runtime.F90').read_text()
        self.assertNotIn('.and..not.lcracon', source)
        self.assertNotIn('lcracon.and.options%checkpoint%enabled', source)
        self.assertIn('call checkpoint_crashfix_gpu(',source)
        self.assertIn('call checkpoint_crashfix_state(',source)
        io=(ROOT/'src/checkpoint_state_io.F90').read_text()
        self.assertIn("dataset(file,'crashfix_history',dims,H5T_STD_I64LE",io)
        self.assertIn('checkpoint lacks required crashfix history',io)

    def test_gpu_crashfix_stays_on_device_after_stage_boundary(self):
        source = (ROOT/'src_gpu/mainloop_gpu.cuf').read_text()
        repair = (ROOT/'src_gpu/crashfix_gpu.cuf').read_text()
        self.assertIn('call ordered_repair_kernel<<<1,1>>>', repair)
        self.assertIn('do i=first_d(j,k),last_d(j,k)', repair)
        self.assertNotIn('copy_flow_from_gpu', repair)
        self.assertNotIn('call crashfix(', repair)
        self.assertNotIn('q_d=q', repair)
        self.assertNotIn('qsave_d=', repair)
        self.assertNotIn('copy_flow_to_gpu', repair)
        stage_boundary = source.index('call apply_conservative_stage_gpu(twall(3))')
        self.assertLess(stage_boundary,source.index('call repair_crashfix_state_gpu(rkstep,',stage_boundary))
        runtime = (ROOT/'src_gpu/gpu_runtime.cuf').read_text()
        self.assertIn('call time_integration_rk_gpu(first_stage_prepared,copy_back)', runtime)
        main = (ROOT/'src/mainloop.F90').read_text()
        self.assertNotIn('gpu_crashfix_stage', main)
        self.assertIn('call check_crashfix_state_gpu()', main)
        self.assertIn('if(lcracon) call crashfix(ctime(16))', main)

    def test_mp_input_repair_precedes_flux_and_refreshes_halos_without_reapplying_wall(self):
        source = (ROOT/'src_gpu/mainloop_gpu.cuf').read_text()
        begin = source.index('subroutine repair_mp_stage_inputs_gpu(')
        end = source.index('end subroutine repair_mp_stage_inputs_gpu',begin)
        repair = source[begin:end]
        self.assertIn('if(.not.lcracon) return',repair)
        self.assertNotIn('mp_positivity_enabled',repair)
        self.assertIn('if(.not.mp_inputs_need_repair_gpu()) return',repair)
        self.assertIn('call repair_reconciled_state_gpu(stage)',repair)
        recovery = (ROOT/'src_gpu/crashfix_gpu.cuf').read_text().split(
            'subroutine repair_reconciled_state_gpu(stage,already_repaired)',1)[1].split('end subroutine',1)[0]
        self.assertLess(recovery.index('call exchange_field_halo_gpu(q_d,5)'),
                        recovery.index('call repair_crashfix_state_gpu(stage)'))
        self.assertLess(recovery.index('call repair_crashfix_state_gpu(stage)'),
                        recovery.index('call exchange_solution_halo_gpu()'))
        self.assertIn('if(.not.mp_inputs_need_repair_gpu()) exit',recovery)
        self.assertNotIn('call apply_',repair)
        start = source.index('do rkstep=1,3')
        self.assertLess(source.index("rkstep,'physical boundary'",start),
                        source.index('call build_mp_convection_gpu()',start))

    def test_gpu_crashfix_is_opt_in_nonreacting_and_checks_the_repaired_state(self):
        gpu = (ROOT/'src_gpu/mainloop_gpu.cuf').read_text()
        self.assertIn('if(lcomb.or.numq/=5.or.num_species/=0)', gpu)
        main = (ROOT/'src/mainloop.F90').read_text()
        self.assertNotIn('if(use_gpu.and.lcracon) call gpu_sync_flow_to_host()', main)
        self.assertIn('if(lcracon.and..not.new_output_enabled()) then', main)
        self.assertIn('new output does not support legacy crash rollback', main)
        repair = (ROOT/'src_gpu/crashfix_gpu.cuf').read_text()
        self.assertIn('if(global/=0) then', repair)
        self.assertIn("'ASTR_GPU_CRASHFIX'", repair)
        self.assertIn('call MPI_Abort(MPI_COMM_WORLD,92,ierr)', repair)
        self.assertIn('if(lcracon.and..not.allocated(crinod_d)) call sync_crashfix_mask_to_gpu()',gpu)

    def test_staged_recovery_rebuilds_only_boundary_rhs(self):
        source = (ROOT/'src_gpu/mainloop_gpu.cuf').read_text()
        preparation = source.split('subroutine prepare_profile_nscbc_gpu(stage)',1)[1].split('end subroutine',1)[0]
        self.assertLess(preparation.index("stage,'profile outlet filter'"),
                        preparation.index('call apply_nscbc_outflow_x_rhs_gpu()'))
        rebuild = preparation.split('if(rebuild_rhs) then',1)[1].split('endif',1)[0]
        self.assertIn('call zero_rhs_kernel',rebuild)
        self.assertIn('call apply_nscbc_outflow_x_rhs_gpu()',rebuild)
        self.assertIn('call apply_nscbc_farfield_y_upper_rhs_gpu(mach2)',rebuild)
        self.assertNotIn('call apply_s1_flatplate_wall_gpu',rebuild)
        self.assertNotIn('filter_gpu()',rebuild)
        self.assertIn("'post_input_crashfix_',snapshot_event",source)

    def test_gpu_crashfix_preserves_cpu_critical_node_flux_fallback(self):
        gpu = (ROOT/'src_gpu/solver_gpu.cuf').read_text()
        self.assertIn('hdiss=critical_interface_active(i,j,k,idir,im,jm,km)', gpu)
        self.assertIn('hc(m)=fp(4)+fm(4)', gpu)
        arrays = (ROOT/'src_gpu/commarray_gpu.cuf').read_text()
        self.assertIn('crinod_d=crinod', arrays)
        self.assertIn('if(.not.lcracon.and..not.allocated(crinod_d)) return', arrays)
        self.assertIn('if(.not.crashfix_enabled_d) return', gpu)

    def test_generated_profile_grid_is_supplied_as_physical_coordinates(self):
        source = (ROOT/'src_gpu/insitu_products_gpu.cuf').read_text()
        self.assertIn("physical_coordinates=lreadgrid.or.profile=='boundary_layer'", source)
        self.assertIn('merge(1_c_int,0_c_int,physical_coordinates)', source)

    def test_legacy_random_uses_approved_nonzero_seed_mapping(self):
        source = (ROOT / "src/wall_blowing_random.F90").read_text()
        self.assertIn('seed=1', source)
        self.assertIn('seed=rank+1', source)
        self.assertNotIn('seed=0', source)
        self.assertIn('do m=1,15', source)
        self.assertIn('do k=0,ubound(velocity,2)', source)
        self.assertIn('do i=0,ubound(velocity,1)', source)

    def test_userdefine_is_selected_through_root_cmake(self):
        root = (ROOT / "CMakeLists.txt").read_text()
        sources = (ROOT / "src/CMakeLists.txt").read_text()
        self.assertIn('CACHE FILEPATH "Fortran source providing the userdefine module"', root)
        self.assertIn('${ASTR_USERDEFINE_SOURCE}', sources)
        self.assertIn('user_define_module/userdefine.F90', root)
        self.assertIn('ASTR_USERDEFINE_SOURCE must name an existing Fortran source file', root)

    def test_legacy_random_is_explicit_and_rejects_inexact_restart(self):
        source = (ROOT / "src/bc.F90").read_text()
        self.assertIn("ASTR_WALL_BLOWING_MODE", source)
        self.assertIn("mode='current'", source)
        self.assertIn("case('legacy_random')", source)
        self.assertIn('if(lrestart) stop \'legacy_random restart requires RNG state support', source)
        self.assertIn('wall_blowing_nmod_t>0 .and. .not.wall_blowing_legacy_random', source)
        self.assertIn('call sample_legacy_wall_velocity', source)

    def test_native_checkpoint_restore_binds_frozen_resources_and_requires_rng(self):
        source = (ROOT / "src/output_runtime.F90").read_text()
        begin = source.index('subroutine bootstrap_output_resources()')
        end = source.index('end subroutine', begin)
        restore = source[begin:end]
        self.assertIn('call configure_wall_blowing()', restore)
        self.assertIn('call set_wall_resource_root(', restore)
        self.assertLess(restore.index('call set_wall_resource_root('),restore.index('call configure_wall_blowing()'))
        self.assertNotIn('file_fingerprint', source)
        self.assertIn("if(wall_blowing_legacy_random) magic='ASTROC09'", source)
        self.assertIn('random wall checkpoint RNG mode/version mismatch', source)
        self.assertIn('call put_legacy_random_state(', source)

    def test_gpu_uploads_only_the_shared_wall_supply(self):
        source = (ROOT / "src_gpu/boundary_gpu.cuf").read_text()
        self.assertIn('legacy_wall_velocity_h=wall_blowing_velocity()', source)
        self.assertIn('legacy_wall_velocity_d=legacy_wall_velocity_h', source)
        self.assertIn('wall_blowing_velocity_gpu=legacy_wall_velocity_d(i,k)', source)
        begin = source.index('subroutine prepare_legacy_wall_blowing_gpu(')
        end = source.index('end subroutine prepare_legacy_wall_blowing_gpu', begin)
        self.assertNotIn('q_d', source[begin:end])
        self.assertNotIn('vel_d', source[begin:end])

    def test_sponge_uses_dataswap_semantics_not_shared_node_averaging(self):
        source = (ROOT / "src_gpu/halo_exchange_gpu.cuf").read_text()
        begin = source.index('subroutine exchange_solution_sponge_halo_gpu()')
        end = source.index('end subroutine exchange_solution_sponge_halo_gpu', begin)
        body = source[begin:end]
        self.assertIn('call exchange_field_halo_gpu(q_d,numq)', body)
        self.assertNotIn('call exchange_x_halo_gpu', body)
        self.assertNotIn('call qswap_', body)
        self.assertNotIn('call refresh_reconciled_solution_halos_gpu', body)

    def test_y_sponge_is_initialized_independently_and_exchanges_before_local_return(self):
        source = (ROOT / "src_gpu/sponge_gpu.cuf").read_text()
        begin = source.index('subroutine init_sponge_gpu()')
        end = source.index('end subroutine init_sponge_gpu', begin)
        self.assertNotIn('return', source[begin:end])
        self.assertIn('sponge_jm_coef_d = sponge_damp_coef_jm', source[begin:end])
        begin = source.index('subroutine apply_ymax_sponge_gpu()')
        end = source.index('end subroutine apply_ymax_sponge_gpu', begin)
        body = source[begin:end]
        self.assertLess(body.index('call exchange_solution_sponge_halo_gpu()'),
                        body.index('if(.not.gpu_ymax_sponge_active()) return'))

    def test_private_diagnostic_is_explicit_and_after_complete_rk(self):
        source = (ROOT / 'tests/gpu_validation/m12_insitu_diagnostic_check.cuf').read_text()
        self.assertIn('ASTR_M12_DIAGNOSTIC_CHECK', source)
        self.assertIn('subroutine check_profile_device_sample_gpu()', source)
        self.assertIn('sample_velocity=sample_velocity', source)
        main = (ROOT / 'src/mainloop.F90').read_text()
        call = main.index('call check_profile_device_sample_gpu()')
        self.assertLess(main.index('call time_integration_rk'), call)
        self.assertLess(call, main.index('call sample_insitu_step(nstep+1'))

    def test_sensor_detail_uses_gpu_local_stage_without_changing_solver_stage(self):
        validation = (ROOT / 'src/validation_io.F90').read_text()
        gpu = (ROOT / 'src_gpu/shock_sensor_gpu.cuf').read_text()
        self.assertIn('logical function sensor_detail_requested(stage_index)', validation)
        self.assertIn('if(present(stage_index)) stage=stage_index', validation)
        self.assertIn('if(sensor_detail_requested(stage_index)) then', gpu)
        self.assertIn('write_sensor_detail_validation_snapshot(gradient_host,pressure_host,stage_index)', gpu)


if __name__ == '__main__':
    unittest.main()
