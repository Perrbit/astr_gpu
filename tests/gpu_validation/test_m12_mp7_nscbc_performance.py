"""Timing identity and exact-payload contracts for the local M12 driver."""

from pathlib import Path
import json
import tempfile
import unittest

import h5py
import numpy as np

from run_m12_mp7_nscbc_performance import compare_stages, compare_state, configure_case, mpi_command, performance_summary, runtime, timing_window


def records():
    result = []
    for step in range(2):
        for rank in range(2):
            value = 10 if step == rank else 1
            result += [f'ASTR_GPU_RANK_RK_TIMING {rank} {step} 0 {value} {value}',
                       f'ASTR_COMPLETE_STEP_TIMING {step} {rank} {value+1}']
        for stage in (1, 2, 3):
            result.append(f'ASTR_MP_POSITIVITY step= {step} stage= {stage} limited_rank_nodes= 0')
    return '\n'.join(result)


class M12PerformanceTest(unittest.TestCase):
    def test_stage_comparison_rejects_missing_required_phases(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/name for name in ('a', 'b')]
            for path in paths:
                (path/'diagnostics').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'stage snapshots'):
                compare_stages(*paths, 1000, 2)

    def test_device_aware_uses_existing_ucx_profile_not_host_only_launcher(self):
        command = mpi_command(Path('/mpi/mpiexec'), 2, 'device-aware')
        self.assertIn('ucx', command)
        self.assertNotIn('ob1', command)
        self.assertEqual(runtime('2,1,1', 'device-aware')['UCX_TLS'],
                         'self,sm,cuda_copy,cuda_ipc')
        self.assertEqual(runtime('2,1,1', 'device-aware')['UCX_MEMTYPE_CACHE'], 'n')
        self.assertIn('ob1', mpi_command(Path('/mpi/mpiexec'), 2, 'pinned'))
        self.assertNotIn('UCX_TLS', runtime('2,1,1', 'pinned'))

    def test_retention_does_not_require_minimum_speedup(self):
        with tempfile.TemporaryDirectory() as folder:
            references, candidates = [], []
            for prefix, speed in (('reference', 1.), ('candidate', .995)):
                for i in range(5):
                    path = Path(folder)/f'{prefix}{i}'
                    path.mkdir()
                    receipt = dict(status='passed', timing_scope='no-field-io-no-profiling',
                                   provenance={key:0 for key in ('intervals','topology','first_step',
                                                                'stop_step','warmup_steps','input_sha256')},
                                   timing=dict(limiter_counts={}, rk_seconds_per_step=speed,
                                               complete_seconds_per_step=speed+.01))
                    (path/'receipt.json').write_text(json.dumps(receipt))
                    (references if prefix == 'reference' else candidates).append(path)
            report = performance_summary(references, candidates)
            self.assertLess(report['rk_seconds_per_step']['reduction'], .05)
            self.assertTrue(report['measured_rk_reduction'])
            self.assertNotIn('performance_gate_passed', report)
            self.assertIn('no-minimum-speedup', report['optional_retention'])
            for path in candidates:
                value = json.loads((path/'receipt.json').read_text())
                value['timing']['rk_seconds_per_step'] = 1.1
                value['timing']['complete_seconds_per_step'] = 1.11
                (path/'receipt.json').write_text(json.dumps(value))
            self.assertFalse(performance_summary(references, candidates)['measured_rk_reduction'])
            value['timing']['limiter_counts'] = {'1':[0,0,1]}
            (candidates[0]/'receipt.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'activation workload differs'):
                performance_summary(references, candidates)

    def test_window_maximum_after_rank_sum(self):
        report = timing_window(records(), 0, 2, 2)
        self.assertEqual(report['rk_seconds_per_step'], 5.5)
        self.assertEqual(report['complete_seconds_per_step'], 6.5)
        self.assertEqual(report['limiter_window'], 'inactive')

    def test_missing_duplicate_and_nonfinite_records_rejected(self):
        for text in (records().replace('ASTR_GPU_RANK_RK_TIMING 0 0', 'omitted 0 0'),
                     records()+'\nASTR_GPU_RANK_RK_TIMING 0 0 0 10 10',
                     records().replace(' 0 10 10', ' 0 nan nan')):
            with self.assertRaises(ValueError):
                timing_window(text, 0, 2, 2)

    def test_warmup_and_activation(self):
        text = records().replace('step= 1 stage= 3 limited_rank_nodes= 0',
                                 'step= 1 stage= 3 limited_rank_nodes= 7')
        report = timing_window(text, 1, 2, 2)
        self.assertEqual(report['limited_rank_node_stage_events'], 7)
        self.assertEqual(report['measured_steps'], 1)
        wrapped = text.replace(' limited_rank_nodes=', '\n limited_rank_nodes=')
        self.assertEqual(timing_window(wrapped, 1, 2, 2), report)

    def test_rank_precedes_step_in_production_rk_log(self):
        text = 'ASTR_GPU_RANK_RK_TIMING 0 4 .1 .2 .3\nASTR_COMPLETE_STEP_TIMING 4 0 .4\n'
        text += '\n'.join(f'ASTR_MP_POSITIVITY step= 4 stage= {s} limited_rank_nodes= 0'
                          for s in (1, 2, 3))
        self.assertAlmostEqual(timing_window(text, 4, 5, 1)['rk_seconds_per_step'], .3)

    def test_runtime_freezes_case_controls_and_avoids_tgv_bypass(self):
        env = runtime('2,1,1')
        for key, value in {'ASTR_GPU_MP_POSITIVITY':'flux', 'ASTR_GPU_SYNC_MODE':'explicit',
                           'ASTR_WALL_BLOWING_MODE':'legacy_random',
                           'ASTR_GPU_PRECISION_MODE':'fp64',
                           'ASTR_NSCBC_FARFIELD_MODE':'compatibility'}.items():
            self.assertEqual(env[key], value)
        self.assertNotIn('ASTR_GPU_BENCHMARK_NO_FIELD_IO', env)
        self.assertNotIn('ASTR_VALIDATION_RHS_PREFIX', env)

    def test_formal_restore_overrides_output_without_enabling_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder)
            (case/'datin').mkdir()
            (case/'outdat').mkdir()
            (case/'datin/input.dat').write_text('# bctype\n11,prof\n21\n41,10.d0\n50\n1\n1\n')
            (case/'datin/controller').write_text('# maxstep,feqchkpt\n9,1,9,9,1,9\n')
            configure_case(case, 120, restore=Path('/frozen/checkpoint'))
            text = (case/'datin/input.output').read_text()
            self.assertIn("restart_output='override'", text)
            for section in ('checkpoint', 'volume', 'slices'):
                self.assertIn('&'+section+'\n enabled=f', text)
            self.assertIn('22,1.d0\n41,10.d0\n52', (case/'datin/input.dat').read_text())

    def test_payload_comparison_includes_signed_zero_and_rank_extras(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/name for name in ('a.h5', 'b.h5')]
            for path in paths:
                with h5py.File(path, 'w') as state:
                    state['identity'] = [0]
                    state['rank_extras'] = np.array([[0.]], dtype=np.float64)
                    for i in range(1, 6):
                        state[f'q{i:04d}'] = np.full((2, 2, 2), 1. if i == 1 else 4. if i == 5 else 0.)
            self.assertEqual(compare_state(*paths)['rank_extras'], 'bitwise-equal')
            with h5py.File(paths[1], 'r+') as state:
                state['rank_extras'][0, 0] = -0.
            with self.assertRaisesRegex(ValueError, 'rank_extras'):
                compare_state(*paths)


if __name__ == '__main__':
    unittest.main()
