"""Platform preflight must preserve allocation and reject wrong rendering devices."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image

import run_platform_insitu_acceptance as gate
import check_platform_rendering_runtime as runtime


def test_rendering_runtime_accepts_selected_library_symlink(tmp_path, monkeypatch):
    expected = tmp_path/'overlay/lib.so.6.1'
    expected.parent.mkdir()
    expected.touch()
    link = expected.with_name(runtime.SONAME)
    link.symlink_to(expected.name)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=f'\t{runtime.SONAME} => {link} (0x1234)\n', stderr=''))
    assert runtime.check_library('/probe', expected) == expected


def test_rendering_runtime_rejects_existing_but_wrong_library(tmp_path, monkeypatch):
    expected = tmp_path/'overlay.so'
    old = tmp_path/'old.so'
    expected.touch()
    old.touch()
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=f'\t{runtime.SONAME} => {old} (0x1234)\n', stderr=''))
    with pytest.raises(ValueError, match='Wrong rendering library'):
        runtime.check_library('/probe', expected)


@pytest.mark.parametrize('listing', ['libother.so => /lib/libother.so (0x1234)\n',
                                     'libother.so => not found\n', 'undefined symbol: example\n'])
def test_rendering_runtime_rejects_missing_or_unresolved_library(tmp_path, monkeypatch, listing):
    expected = tmp_path/'overlay.so'
    expected.touch()
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=listing, stderr=''))
    with pytest.raises(ValueError, match='Missing rendering|Unresolved runtime'):
        runtime.check_library('/probe', expected)


def arguments(tmp_path, topologies=('1,1,2',)):
    return SimpleNamespace(output=tmp_path, probe_directory=Path('/probe'), mpiexec=Path('/mpi'),
                           m12_topologies=topologies)


def test_default_m12_topologies_use_x_slabs():
    assert gate.M12_TOPOLOGIES == ('2,1,1', '4,1,1')


def probe_runner(monkeypatch, wrong_graphics=False, device_count=2, clipping_planes=6):
    devices = [dict(uuid=f'00000000-0000-0000-0000-{i+1:012d}', egl_index=i+3)
               for i in range(device_count)]
    mask = ','.join(str(4+2*i) for i in range(device_count))
    commands = []

    def execute(command, output, timeout=600):
        commands.append(command)
        assert timeout == 600
        assert gate.os.environ['CUDA_VISIBLE_DEVICES'] == mask
        name = Path(str(command[-1])).name
        if name == 'insitu_device_identity.py':
            return json.dumps(devices)
        if name == 'insitu_device_map_probe':
            ranks = int(command[command.index('-np')+1])
            return '\n'.join(f"MAPPING rank={r} egl={devices[r]['egl_index']} uuid={devices[r]['uuid']}"
                             for r in range(ranks))
        if str(command[-1]) == '--conduit-clipped':
            device = int(command[-2])
            receipt = dict(cuda_device=device, gl_cuda_device=1-device if wrong_graphics else device,
                uuid=devices[device]['uuid'], geometry_host_bytes=0, host_access_refused=True,
                pixels_passed=True, conduit=True, clipping_planes=clipping_planes)
            return 'ASTR_INSITU_STANDARD_DEVICE_PROBE '+json.dumps(receipt)
        return 'supported=1\nsupported=1\nCUDA-aware device-buffer exact tests passed'

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setattr(gate, 'execute', execute)
    return commands


def test_allocated_device_mask_is_not_replaced(tmp_path, monkeypatch):
    commands = probe_runner(monkeypatch)
    receipt = gate.dependency_gate(arguments(tmp_path))
    assert len(receipt['graphics']) == 2
    assert receipt['cuda_aware_payload'] == 'passed'
    assert len(commands) == 6
    assert gate.os.environ['CUDA_VISIBLE_DEVICES'] == '4,6'


def test_wrong_graphics_device_stops_before_transport_and_flow(tmp_path, monkeypatch):
    commands = probe_runner(monkeypatch, wrong_graphics=True)
    with pytest.raises(ValueError, match='Device buffer rendering probe failed'):
        gate.dependency_gate(arguments(tmp_path))
    assert len(commands) == 4


def test_render_dependency_gate_requires_six_plane_clipping(tmp_path, monkeypatch):
    commands = probe_runner(monkeypatch)
    receipt = gate.dependency_gate(arguments(tmp_path))
    assert all(row['clipping_planes'] == 6 for row in receipt['graphics'])
    assert sum(str(command[-1]) == '--conduit-clipped' for command in commands) == 2


@pytest.mark.parametrize('clipping_planes', [0, None])
def test_old_render_dependency_fails_before_transport_and_flow(tmp_path, monkeypatch, clipping_planes):
    commands = probe_runner(monkeypatch, clipping_planes=clipping_planes)
    with pytest.raises(ValueError, match='Device buffer rendering probe failed'):
        gate.dependency_gate(arguments(tmp_path))
    assert len(commands) == 4


def test_four_gpu_job_checks_both_m12_rank_counts(tmp_path, monkeypatch):
    commands = probe_runner(monkeypatch, device_count=4)
    receipt = gate.dependency_gate(arguments(tmp_path, gate.M12_TOPOLOGIES))
    assert [item['ranks'] for item in receipt['mappings']] == [1, 2, 4]
    assert len(receipt['graphics']) == receipt['used_devices'] == 4
    assert len(commands) == 9
    assert gate.os.environ['CUDA_VISIBLE_DEVICES'] == '4,6,8,10'


def test_graphics_probe_accepts_runtime_device_range_instead_of_two_gpu_whitelist():
    source = (gate.ROOT/'tests/gpu_validation/insitu_standard_device_probe.cu').read_text()
    assert 'cudaGetDeviceCount(&device_count)' in source
    assert 'selected>=device_count' in source
    assert 'std::strtol(argv[1],&end,10)' in source
    assert 'std::strcmp(argv[1],"0")' not in source


def test_platform_default_budgets_allow_initialization_and_dependency_work():
    options = gate.runtime_options(SimpleNamespace())
    assert options['runtime_timeout_seconds'] == 3600
    assert options['directory_budget_bytes'] == 32*1024**3
    assert options['resource_host_extra_budget_bytes'] == 32*1024**3
    assert options['resource_device_extra_budget_bytes'] == 8*1024**3
    assert options['resource_device_reserve_bytes'] == 2*1024**3


def test_extra_allocated_devices_do_not_reject_smaller_test(tmp_path, monkeypatch):
    probe_runner(monkeypatch, device_count=4)
    receipt = gate.dependency_gate(arguments(tmp_path))
    assert len(receipt['devices']) == 4
    assert len(receipt['graphics']) == receipt['used_devices'] == 2


def test_topology_exceeding_allocation_reports_required_devices(tmp_path, monkeypatch):
    commands = probe_runner(monkeypatch)
    with pytest.raises(ValueError, match='need 4 distinct.*found 2'):
        gate.dependency_gate(arguments(tmp_path, gate.M12_TOPOLOGIES))
    assert len(commands) == 1


@pytest.mark.parametrize('entry', ['m12_gate', 'native_gate'])
def test_m12_debug_and_native_cover_requested_topologies(tmp_path, monkeypatch, entry):
    calls = []

    def run_m12(args, backend, topology, **options):
        calls.append((backend, topology, options))
        assert args.runtime_timeout_seconds == 1200
        assert args.resource_host_extra_budget_bytes == 24*1024**3
        case = args.output / f'{backend}_{topology}_{options["case_label"]}'
        case.mkdir()
        (case/'resources.sampled.json').write_text('{}')
        if options.get('insitu_config'):
            assert 'host_budget_bytes=25769803776' in options['insitu_config']
        if entry == 'native_gate':
            assert options['checkpoint_interval'] == 10
        return case

    def run_case(args, root, backend, ranks, label, steps, **options):
        case = args.output / label
        case.mkdir()
        (case/'resources.sampled.json').write_text('{}')
        return case, None

    monkeypatch.setattr(gate, 'run_m12', run_m12)
    monkeypatch.setattr(gate, 'run_case', run_case)
    monkeypatch.setattr(gate, 'check_pair', lambda *a, **kw: dict(q=dict(max_abs=0.), statistics_max_abs={}))
    monkeypatch.setattr(gate, 'check_images', lambda case, ranks, **kw: dict(ranks=ranks))
    monkeypatch.setattr(gate, 'check_device_render', lambda *a, **kw: True)
    monkeypatch.setattr(gate, 'compare_fields', lambda *a, **kw: None)
    args = SimpleNamespace(output=tmp_path, source=Path('/inputs'), executable=Path('/astr'),
        native_executable=Path('/astr-native'), mpiexec=Path('/mpi'), library=Path('/catalyst'),
        m12_topologies=gate.M12_TOPOLOGIES, runtime_timeout_seconds=1200,
        resource_host_extra_budget_bytes=24*1024**3)
    receipt = getattr(gate, entry)(args)
    results = receipt['m12'] if entry == 'native_gate' else receipt
    assert [item['topology'] for item in results] == list(gate.M12_TOPOLOGIES)
    per_topology = 2 if entry == 'native_gate' else 3
    assert len(calls) == per_topology*2
    assert {topology for _, topology, _ in calls} == set(gate.M12_TOPOLOGIES)


def test_configurable_render_budgets_match_external_monitor(tmp_path):
    args = SimpleNamespace(library=Path('/catalyst'), resource_host_extra_budget_bytes=12*1024**3,
                           resource_device_extra_budget_bytes=3*1024**3,
                           resource_device_reserve_bytes=2*1024**3)
    for factory in (gate.tgv_configuration, gate.m12_configuration):
        config = gate.render_configuration(factory, args)
        assert 'host_budget_bytes=12884901888' in config
        assert 'device_budget_bytes=3221225472' in config
        assert 'device_reserve_bytes=2147483648' in config


def test_tgv_uses_explicit_resident_rendering_and_aware_postprocessing():
    config = gate.tgv_configuration(Path('/dependency/catalyst'))
    assert "rendering_pipeline='standard-device'" in config
    assert "postprocess_transport='device-aware'" in config
    assert 'statistics=f' in config and 'step_interval=5' in config
    assert 'device_render_pipeline.py' in config


@pytest.fixture
def render_case(tmp_path):
    def prepare(ranks=1, pipeline='standard-device'):
        case = tmp_path / 'render_case'
        output = case / 'outdat/render'
        output.mkdir(parents=True)
        steps = [5, 10]
        log = []
        products = ('q_surface', 'velocity_slice', 'instantaneous_streamlines', 'crossing_streamlines')
        stages = ['baseline', 'initialized'] + ['frame_before', 'frame_after'] * len(steps) + [
            'finalize_before', 'finalize_after', 'session_released']
        for rank in range(ranks):
            device = f'00000000-0000-0000-0000-{rank+1:012d}'
            (output / f'resources.rank{rank:08d}.csv').write_text(
                f'# device=GPU-{device}\n'
                'stage,host_increment_bytes,device_increment_bytes,device_free_bytes\n' +
                ''.join(f'{stage},0,0,2147483648\n' for stage in stages))
            (output / f'lifecycle_rank{rank}.json').write_text(json.dumps(
                dict(finalized=True, frames=[[step, step * .001] for step in steps])))
            for step in steps:
                receipt = dict(step=step, time=step * .001, processing_backend='device',
                    rendering_pipeline=pipeline, geometry_host_bytes=0,
                    products={name: dict(egl_uuid=device, scalar_bar_visible=True,
                        scalar_bar_label_color=[0., 0., 0.], color_range=[-1., 1.], color_field='u')
                        for name in products})
                if pipeline == 'compatible':
                    if rank == 0:
                        receipt['crossing_maxabs'] = 1e-15
                else:
                    log.append(f'ASTR_INSITU_RESIDENT_CROSSING rank={rank} endpoint_maxabs=1e-15')
                (output / f'mesh_step{step:08d}_rank{rank}.json').write_text(json.dumps(receipt))
                log.append(f'ASTR_INSITU_DEVICE_FRAME rank={rank} step={step} transport=device-aware')
            log.append(f'ASTR_INSITU_GPU_FLOW rank={rank} frame_downloads=0')
        (case / 'run.log').write_text('\n'.join(log) + '\n')
        image = Image.new('RGB', (800, 600), 'white')
        image.paste((200, 0, 0), (680, 50, 700, 250))
        for step in steps:
            for name in products:
                path = output / f'{name}.step{step:08d}.jpeg'
                image.save(path)
                image.save(path.with_suffix('.eps'))
        return case, steps
    return prepare


@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
@pytest.mark.parametrize('ranks', [1, 2])
def test_resident_render_checks_logged_endpoints_without_legacy_json(render_case, ranks, pipeline):
    case, steps = render_case(ranks, pipeline)
    assert gate.check_device_render(case, ranks, steps, statistics=False)


@pytest.mark.parametrize('replacement', [
    '',
    'ASTR_INSITU_RESIDENT_CROSSING rank=1 endpoint_maxabs=1e-15',
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=1e-15\n'
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=1e-15',
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=nan',
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=inf',
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=-1e-15',
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=3e-10',
    'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=invalid',
])
def test_resident_render_rejects_missing_or_invalid_endpoint_diagnostics(render_case, replacement):
    case, steps = render_case()
    log = case / 'run.log'
    log.write_text(log.read_text().replace(
        'ASTR_INSITU_RESIDENT_CROSSING rank=0 endpoint_maxabs=1e-15', replacement, 1))
    with pytest.raises(AssertionError, match='[Cc]rossing'):
        gate.check_device_render(case, 1, steps, statistics=False)


def test_compatible_render_keeps_legacy_crossing_receipt_check(render_case):
    case, steps = render_case(pipeline='compatible')
    assert gate.check_device_render(case, 1, steps, statistics=False)
    path = case / 'outdat/render/mesh_step00000005_rank0.json'
    receipt = json.loads(path.read_text())
    receipt['crossing_maxabs'] = 3e-10
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        gate.check_device_render(case, 1, steps, statistics=False)


@pytest.mark.parametrize('entry', ['tgv_gate', 'native_gate'])
def test_correctness_runs_keep_final_checkpoint_instead_of_disabling_all_field_io(
        tmp_path, monkeypatch, entry):
    class ConfigurationChecked(Exception):
        pass

    def run_case(args, root, backend, ranks, label, steps, **options):
        assert options.get('no_field_io', False) is False
        assert options.get('checkpoint_enabled') is True
        assert options['checkpoint_interval'] == steps == 10
        assert args.initial_restart is False
        raise ConfigurationChecked

    monkeypatch.setattr(gate, 'run_case', run_case)
    args = SimpleNamespace(output=tmp_path, executable=Path('/astr'),
        native_executable=Path('/astr-native'), mpiexec=Path('/mpi'), library=Path('/catalyst'))
    with pytest.raises(ConfigurationChecked):
        getattr(gate, entry)(args)


@pytest.fixture
def dependency_build_environment(tmp_path):
    binary = tmp_path / 'bin'
    binary.mkdir()
    runtime = tmp_path / 'gcc/lib64/libstdc++.so.6'
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    compiler = binary / 'cxx'
    compiler.write_text(f'#!/bin/sh\nprintf "%s\\n" "{runtime}"\n')
    compiler.chmod(0o755)
    cmake = binary / 'cmake'
    cmake.write_text('#!/bin/sh\nprintf "RUNTIME=%s\\n" "$LD_LIBRARY_PATH"\nprintf "%s\\n" "$@"\n')
    cmake.chmod(0o755)
    env = dict(PATH=str(binary)+':'+os.environ['PATH'], HOME=os.environ['HOME'],
        LD_LIBRARY_PATH='/stale/anaconda/lib', PV_DEVICE_CXX=str(compiler),
        CONFIGURE_ONLY='1', PV_DEVICE_SOURCE=str(tmp_path/'source'),
        PV_DEVICE_BUILD=str(tmp_path/'build'), PV_DEVICE_PREFIX=str(tmp_path/'prefix'),
        CATALYST_DIR=str(tmp_path/'catalyst'), MPI_ROOT=str(tmp_path/'mpi'),
        CUDA_ROOT=str(tmp_path/'cuda'))
    return env, runtime.parent


def test_dependency_build_prefers_compiler_runtime_over_anaconda(dependency_build_environment):
    env, runtime = dependency_build_environment
    result = subprocess.run(['bash', str(gate.ROOT/'scripts/insitu/build_device_paraview.sh')],
        env=env, text=True, capture_output=True, check=True, timeout=10)
    assert f'RUNTIME={runtime}:/stale/anaconda/lib' in result.stdout


def test_dependency_library_directory_matches_deployment_consumers(dependency_build_environment):
    env, _ = dependency_build_environment
    result = subprocess.run(['bash', str(gate.ROOT/'scripts/insitu/build_device_paraview.sh')],
        env=env, text=True, capture_output=True, check=True, timeout=10)
    assert '-DCMAKE_INSTALL_LIBDIR=lib' in result.stdout.splitlines()


def test_missing_compiler_runtime_stops_before_configuration(dependency_build_environment):
    env, runtime = dependency_build_environment
    (runtime/'libstdc++.so.6').unlink()
    result = subprocess.run(['bash', str(gate.ROOT/'scripts/insitu/build_device_paraview.sh')],
        env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert 'Cannot locate the selected C++ compiler runtime' in result.stderr
    assert 'RUNTIME=' not in result.stdout
