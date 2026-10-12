"""Regression checks for output, diagnostics and optimization admission ownership."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from run_m12_hd_coverage import insitu_configuration, prepare_probe_map
from run_output_restart_validation import run_case, compare_fields
from test_output_repartition_runtime import ROOT, MPIEXEC, arguments, clocks, compare_global_state

CPU = ROOT / 'build_release_restart_cpu/bin/astr'
GPU = ROOT / 'build_gpu_probe/bin/astr'
PARSER = ROOT / 'build_insitu_check/bin/insitu_run_config_probe'
FINAL = 'outdat/new/checkpoints/step000000000012'


def product_subset(ids):
    content = insitu_configuration(images=False)
    start = content.index(' product_ids=')
    return content[:start] + (
        ' product_ids=' + ','.join(repr(name) for name in ids) + ',\n'
        " product_modes=" + ','.join("'steps'" for _ in ids) + ',\n'
        ' product_steps=' + ','.join('1000000000' for _ in ids) + '\n/\n'
        if ids else '/\n')


@pytest.mark.parametrize('ids,accepted', [
    (['velocity_slice.image'], True),
    (['q_surface.image', 'velocity_slice.image'], True),
    (['q_surface.image', 'velocity_slice.image', 'instantaneous_streamlines.image'], True),
    ([], False), (['velocity_slice.image']*2, False), (['unknown.image'], False),
])
def test_m12_product_subsets(tmp_path, ids, accepted):
    config = tmp_path/'input.insitu'
    config.write_text(product_subset(ids))
    result = subprocess.run([str(PARSER), str(config)], capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('archive', [False, True])
def test_ordinary_vortex_does_not_require_checkpoint_case(tmp_path, backend, archive):
    args = arguments(tmp_path)
    args.executable = CPU if backend == 'cpu' else GPU
    args.statistics = False
    args.wall_samples = False
    basic_archive = """&volume
 enabled=t, interval_steps=2, initial_frame=f, final_frame=t
/
&slices
 enabled=f
/
""" if archive else None
    case, _ = run_case(args, ROOT, backend, 1, 'vortex', 2, enabled=False,
        checkpoint_enabled=False, flow_type='2dvort', archive_groups=basic_archive)
    assert 'The job is done!' in (case/'run.log').read_text()
    assert not list((case/'outdat/new/checkpoints').glob('*/COMPLETE'))
    if archive:
        assert list((case/'outdat/new/fields').rglob('data.h5'))


@pytest.mark.parametrize('precision,sync', [('mixed_workspace', 'explicit'), ('fp64', 'selective')])
def test_checkpoint_free_tgv_keeps_solver_mode_admission(tmp_path, precision, sync):
    args = arguments(tmp_path)
    args.statistics = False
    args.wall_samples = False
    candidate = 'viscous_flux' if precision == 'mixed_workspace' else None
    case, _ = run_case(args, ROOT, 'gpu', 1, 'candidate', 2, enabled=False,
        gpu_precision_mode=precision, gpu_sync_mode=sync, gpu_mixed_candidate=candidate)
    log = (case/'run.log').read_text()
    assert 'The job is done!' in log
    assert log.count('ASTR_GPU static configuration validated') == 1
    run_case(args, ROOT, 'gpu', 1, 'checkpoint_reject', 2,
        gpu_precision_mode=precision, gpu_sync_mode=sync, gpu_mixed_candidate=candidate,
        reject='new restart requires FP64' if precision != 'fp64' else 'new restart first gate requires explicit sync')


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('source_topology,target_topology', [
    ([2, 1, 1], [4, 1, 1]), ([4, 1, 1], [2, 1, 1]), ([4, 1, 1], [1, 4, 1]),
])
def test_tgv_four_rank_repartition(tmp_path, backend, source_topology, target_topology, record_property):
    args = arguments(tmp_path)
    args.statistics = False
    args.wall_samples = False
    target_ranks = int(np.prod(target_topology))
    source_ranks = int(np.prod(source_topology))
    reference, _ = run_case(args, ROOT, backend, target_ranks, 'reference', 12,
        grid='32,32,32', topology=target_topology)
    seed, _ = run_case(args, ROOT, backend, source_ranks, 'seed', 5,
        grid='32,32,32', topology=source_topology)
    restored, _ = run_case(args, ROOT, backend, target_ranks, 'restored', 12,
        grid='32,32,32', topology=target_topology,
        restore=seed/'outdat/new/checkpoints/step000000000005')
    assert clocks(seed) + clocks(restored) == clocks(reference)
    maxima = compare_global_state(reference/FINAL/'state.h5', restored/FINAL/'state.h5')
    record_property('field_max_abs', max(maxima.values()))
    for name in ('control.bin', 'archives.bin'):
        assert (reference/FINAL/name).read_bytes() == (restored/FINAL/name).read_bytes()


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_air5_parsed_contract_and_true_mismatch(tmp_path, backend):
    from run_output_air5_restart_validation import launch
    from test_output_air5_conservation import arguments as air5_arguments
    args = air5_arguments(tmp_path)
    args.mean_statistics = False
    conservation = backend == 'gpu'
    reference, _ = launch(args, backend, 1, 'reference', 4, interval=2, conservation=conservation)
    source = reference/'outdat/new/checkpoints/step000000000002'
    restored, _ = launch(args, backend, 1, 'restored', 4, interval=2,
        restore=source, explicit_contract_defaults=True, conservation=conservation)
    final = 'outdat/new/checkpoints/step000000000004'
    compare_fields(reference/final/'state.h5', restored/final/'state.h5')
    assert (reference/final/'air5_config.bin').read_bytes() == (restored/final/'air5_config.bin').read_bytes()
    launch(args, backend, 1, 'changed_compensation', 4, interval=2,
        restore=source, fault='compensation', conservation=conservation,
        reject='AIR5 source/limiter/compensation/top/resource contract mismatch')
    launch(args, backend, 1, 'changed_tau', 4, interval=2,
        restore=source, fault='tau', conservation=conservation,
        reject='AIR5 source/limiter/compensation/top/resource contract mismatch')


@pytest.mark.parametrize('mode,supported,expected', [
    ('split', True, False), ('fused', True, True), ('fused', False, False),
])
def test_flux_pair_applied_mode(mode, supported, expected):
    env = os.environ.copy()
    env['ASTR_GPU_FLUX_PAIR_MODE'] = mode
    command = [str(MPIEXEC), '--mca', 'coll_hcoll_enable', '0', '-np', '2',
        str(ROOT/'build_gpu_probe/bin/upwind_flux_mode_setup_test'), 'supported' if supported else 'unsupported']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'fused_applied={"T" if expected else "F"}' in result.stdout
    assert f'ASTR_GPU_FLUX_PAIR_MODE_APPLIED={"fused" if expected else "split"}' in result.stdout
    if mode == 'fused' and not supported:
        assert 'reason=fused kernel is unavailable' in result.stdout


@pytest.mark.skipif(not os.environ.get('ASTR_M12_DIAGNOSTIC_EXE'), reason='select M12 device-render build explicitly')
def test_m12_diagnostic_clock_restores_and_segments(tmp_path):
    from prepare_m12_local_case import prepare
    from run_m12_boundary_compare import configuration, environment
    exe = Path(os.environ['ASTR_M12_DIAGNOSTIC_EXE']).resolve(strict=True)
    shape = (64, 32, 24)

    def launch(name, steps, restore=None):
        case = tmp_path/name
        prepare(ROOT.parent/'M12_C13', case, 'gpu', shape)
        configuration(case, 'extrapolation', shape, steps, archive=False, checkpoint_interval=5, restore=restore)
        (case/'outdat/render').mkdir()
        (case/'datin/input.insitu').write_text(product_subset(['velocity_slice.image']))
        prepare_probe_map(case, shape)
        env = environment('2,1,1')
        env.update(ASTR_INSITU_CONFIG='datin/input.insitu', ASTR_GPU_MP_POSITIVITY='flux')
        command = [str(MPIEXEC), '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
            '--bind-to', 'none', '-np', '2', str(exe), 'run', 'datin/input.dat']
        with (case/'run.log').open('w') as log:
            result = subprocess.run(command, env=env, cwd=case, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, (case/'run.log').read_text()[-6000:]
        assert 'The job is done!' in (case/'run.log').read_text()
        assert sum(p.stat().st_size for p in case.rglob('*') if p.is_file()) <= 64*1024**2
        return case

    reference = launch('reference', 7)
    source = reference/'outdat/new/checkpoints/step000000000005'
    resumed = launch('resumed', 7, source)
    segment = resumed/'outdat/render/probes.from_step000000000005.csv'
    assert not (resumed/'outdat/render/probes.csv').exists()
    rows = np.genfromtxt(segment, delimiter=',', names=True)
    original = np.genfromtxt(reference/'outdat/render/probes.csv', delimiter=',', names=True)
    assert rows.tobytes() == original[original['step'] > 5].tobytes()
    assert rows['step'].tolist() == [6]*72+[7]*72
    final = 'outdat/new/checkpoints/step000000000007/state.h5'
    compare_fields(reference/final, resumed/final)
