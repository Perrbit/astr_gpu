"""Tie the analytic transverse-ownership probe to production dispatch."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class NscbcPhysicsContract(unittest.TestCase):
    def test_rk_probe_does_not_admit_new_output_or_restart(self):
        source = (ROOT / 'src/output_runtime.F90').read_text()
        block = source.split("call get_environment_variable('ASTR_VALIDATION_NSCBC_RK'", 1)[1].split('#endif', 1)[0]
        self.assertIn('maxstep==5.and.deltat==1.d-5', block)
        self.assertIn('all([ia,ja,ka]==[64,8,8])', block)
        self.assertIn('len_trim(options%restore_directory)==0', block)
        for product in ('checkpoint', 'volume', 'slices', 'adaptive'):
            self.assertIn(f'.not.options%{product}%enabled', block)
        self.assertIn('#ifdef ASTR_BUILD_TESTING', source.split('subroutine check_capability()', 1)[1].split(
            "call get_environment_variable('ASTR_VALIDATION_NSCBC_RK'", 1)[0])

    def test_open_shock_transverse_has_single_owner(self):
        source = (ROOT / 'src_gpu/mainloop_gpu.cuf').read_text()
        source = source.split('elseif(explicit_reconstructed_case) then', 1)[1]
        for kernel in ('explicit_upwind_flux_y_physical_global_kernel',
                       'explicit_upwind_flux_z_xyphysical_global_kernel'):
            prefix = source.split('call ' + kernel, 1)[0]
            condition = prefix.rsplit('if(', 1)[1].split(') then', 1)[0]
            self.assertIn('open_shock_nscbc_case', condition)

    def test_profile_outlet_does_not_reapply_extrapolation(self):
        source = (ROOT / 'src_gpu/boundary_gpu.cuf').read_text()
        block = source.split('subroutine apply_s1_flatplate_boundary_conditions_gpu(', 1)[1].split(
            'end subroutine apply_s1_flatplate_boundary_conditions_gpu', 1)[0]
        outlet = block.split('call outflow_x_kernel', 1)[0]
        self.assertTrue(outlet.rstrip().endswith('if(bctype(2)==GPU_BC_OUTFLOW) then'))

    def test_joint_profile_reuses_actual_stage_preparation(self):
        source = (ROOT / 'src_gpu/gpu_runtime.cuf').read_text()
        self.assertIn('stats_snapshot_ready = gpu_s1_flatplate_nscbc_farfield_case().and.bctype(2)/=22', source)
        source = (ROOT / 'src_gpu/mainloop_gpu.cuf').read_text()
        self.assertIn('if(flatplate_nscbc_farfield_case.and..not.profile_nscbc_case) prepared = .false.', source)
        self.assertIn('if(profile_nscbc_case) call apply_nscbc_outflow_x_rhs_gpu()', source)

    def test_metric_roundoff_exception_requires_exact_profile_coordinates(self):
        source = (ROOT / 'src_gpu/mainloop_gpu.cuf').read_text()
        block = source.split('if(any(bctype==22) .and. geometry_error>1.d-10) then', 1)[1].split(
            'if(bctype(2)==21)', 1)[0]
        self.assertIn('if(gpu_profile_mp_ld_flatplate_supported()) then', block)
        self.assertIn('x(i,j,k,:)/=[x(i,0,0,1),x(0,j,0,2),x(0,0,k,3)]', block)
        self.assertIn('rectilinear_coordinates=pmax(local_error)==0.d0', block)
        self.assertIn('if(.not.rectilinear_coordinates)', block)


if __name__ == '__main__':
    unittest.main()
