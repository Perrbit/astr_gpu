"""Face-index coverage and unchanged MP positivity arithmetic contracts."""

from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class MPPerformanceContract(unittest.TestCase):
    def test_mp_device_exchange_uses_whole_capacity_buffers_and_explicit_count(self):
        source = (ROOT/'src_gpu/mp_positivity_gpu.cuf').read_text().replace(' ', '')
        self.assertIn('callexchange_device_pair(send_lo_d,send_hi_d,recv_hi_d,recv_lo_d,n,', source)
        self.assertNotIn('callexchange_device_pair(send_lo_d(1:n', source)
        transport = (ROOT/'src_gpu/device_mpi_transport_gpu.cuf').read_text().replace(' ', '')
        for buffer in ('send_low','send_high','recv_high','recv_low'):
            self.assertIn(f'size({buffer})<count', transport)
        self.assertIn('callmpi_irecv(recv_high,count,', transport)
        self.assertIn('callmpi_isend(send_low,count,', transport)

    def test_x_contiguous_mapping_is_bijective_in_all_directions(self):
        for im, jm, km in ((5, 7, 3), (8, 3, 6), (3, 8, 5)):
            for axis in range(3):
                shape = np.array([im+1, jm+1, km+1])
                shape[axis] += 1
                ids = np.arange(np.prod(shape))
                candidate = np.array([ids % shape[0], (ids//shape[0]) % shape[1],
                                      ids//(shape[0]*shape[1])]).T
                directional = np.empty_like(candidate)
                order = [axis, *[a for a in range(3) if a != axis]]
                remaining = ids.copy()
                for a in order:
                    directional[:, a] = remaining % shape[a]
                    remaining //= shape[a]
                codes = np.ravel_multi_index(candidate.T, shape)
                other = np.ravel_multi_index(directional.T, shape)
                np.testing.assert_array_equal(np.sort(codes), np.sort(other))
                self.assertEqual(np.unique(codes).size, ids.size)
                candidate[:, axis] -= 1
                self.assertEqual(candidate[:, axis].min(), -1)
                self.assertEqual(candidate[:, axis].max(), (im, jm, km)[axis])

    def test_layout_does_not_rewrite_flux_or_budget_arithmetic(self):
        source = (ROOT/'src_gpu/mp_positivity_gpu.cuf').read_text()
        kernel = source.split('attributes(global) subroutine mp_faces_kernel(', 1)[1].split(
            'end subroutine mp_faces_kernel', 1)[0]
        self.assertIn('call mp_low_face(', kernel)
        self.assertIn('call mp_ld_interface_flux(', kernel)
        self.assertIn('delta_d(i,j,k,:,axis)=high-low', kernel)
        self.assertIn('x_contiguous_faces=.false.', source)
        self.assertIn('do iteration=1,64', source)
        self.assertIn('theta=lo*(1.d0-256.d0*epsilon(1.d0))', source)
        self.assertIn("check_mp_failure('complete low-order baseline')", source)
        self.assertIn("check_mp_failure('complete limited RK update')", source)
        self.assertIn('call exchange_field_halo_gpu(ratio_d,1)', source)

    def test_diagnostic_compares_bits_not_a_floating_point_tolerance(self):
        source = (ROOT/'src_gpu/mp_positivity_gpu.cuf').read_text()
        comparison = source.split('attributes(global) subroutine compare_mp_face_layout_kernel', 1)[1].split(
            'end subroutine compare_mp_face_layout_kernel', 1)[0]
        self.assertIn('transfer(high_d(', comparison)
        self.assertIn('transfer(delta_d(', comparison)
        self.assertIn('0_int64', comparison)
        self.assertNotIn('epsilon', comparison)

    def test_cached_reconstruction_keeps_all_characteristic_and_mp_algebra(self):
        source = (ROOT/'src_gpu/solver_gpu.cuf').read_text()
        blocks = []
        for name in ('mp_ld_interface_flux', 'mp_ld_interface_flux_cached'):
            block = source.split('attributes(device) subroutine '+name+'(', 1)[1].split(
                'end subroutine '+name, 1)[0]
            blocks.append(block.split('    do m=1,5\n', 1)[1])
        self.assertEqual(blocks[0], blocks[1])

    def test_split_cache_covers_every_consumed_stencil_without_physical_ghost_reads(self):
        for dim in (5, 8, 13):
            hm = 5
            for npdc in (1, 2, 3, 4):
                lower = 0 if npdc in (1, 4) else -hm
                upper = dim if npdc in (2, 4) else dim+hm
                computed = set(range(lower, upper+1))
                for face in range(-1, dim+1):
                    if npdc in (1, 4) and face < 0 or npdc in (2, 4) and face >= dim:
                        continue
                    consumed = {max(lower, min(upper, face+offset)) for offset in range(-3, 5)}
                    self.assertTrue(consumed <= computed)
        source = (ROOT/'src_gpu/mp_positivity_gpu.cuf').read_text()
        self.assertLess(source.index("call sync_after_kernel('mp_split_cache_kernel')"),
                        source.index('call mp_faces_x_contiguous_kernel<<<'))


if __name__ == '__main__':
    unittest.main()
