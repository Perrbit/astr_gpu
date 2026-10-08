#!/usr/bin/env python3
"""Matched CURVE Q=0/instantaneous trace timing, separate from diagnostics.

Four complete steps, frames at 2/4, one excluded process warm-up per backend,
then five counterbalanced rounds. Optional 100-step image-only runs use the
lowest median complete-window backend for each mapping, without changing RK45.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from run_insitu_curve_performance import digest, later_output
from run_output_restart_validation import run_case
from test_insitu_curve_demo import check_frames, scale_configuration
from test_insitu_curve_derivatives import ROOT
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_x4_observation import no_large_io, timing

BACKENDS = tuple((pipeline, mode) for pipeline in ('standard-device', 'direct-device')
                 for mode in ('pinned', 'device-aware'))
METRICS = ('completed_window', 'advance_inclusive', 'resident_streamlines_inclusive')
GROUP_BUDGET = 256*1024**3
IMAGE_BUDGET = 4*1024**3


def backend_key(pipeline, mode):
    return pipeline+'/'+mode


def round_order(round_id):
    # Four-round Williams design balances positions and directed adjacencies.
    design = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    return tuple(BACKENDS[index] for index in design[round_id % 4])


def production_configuration(pipeline, mode):
    config = scale_configuration(pipeline, mode)
    if config.count('step_interval=2') != 1:
        raise ValueError('Expected exactly one two-step rendering schedule')
    return config.replace('step_interval=2', 'step_interval=1')


def summarize(records, rounds):
    result = {}
    for pipeline, mode in BACKENDS:
        key = backend_key(pipeline, mode)
        rows = [row for row in records if row['backend'] == key and row['round'] >= 0]
        if sorted(row['round'] for row in rows) != list(range(rounds)):
            raise ValueError('Missing or repeated matched timing round: '+key)
        result[key] = {}
        for metric in (*METRICS, 'step4_output'):
            values = [row['step4_output_seconds'] if metric == 'step4_output' else
                      row['timing'][metric]['seconds'] for row in rows]
            if any(not (0. < value < float('inf')) for value in values):
                raise ValueError('Timing must be positive and finite: '+metric)
            result[key][metric] = dict(median=statistics.median(values), minimum=min(values),
                                      maximum=max(values), values=values)
    winner = min(result, key=lambda key: result[key]['completed_window']['median'])
    return dict(backends=result, selected=winner, selection_metric='completed_window median',
                nested_timings_are_not_additive=True)


def group_bytes(build):
    # Ignore pytest's "current" aliases; retain failed and immutable evidence.
    total = 0
    for entry in build.glob('curve_*'):
        if entry.is_symlink():
            continue
        paths = entry.rglob('*') if entry.is_dir() else (entry,)
        total += sum(path.stat().st_size for path in paths
                     if path.is_file() and not path.is_symlink())
    return total


def exclusive_devices():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], check=True, text=True, capture_output=True)
    if result.stdout.strip():
        raise RuntimeError('An existing GPU compute process would invalidate matched timing: '+result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', choices=('periodic', 'y-wavy'), required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--production-steps', type=int, choices=(100,))
    args = parser.parse_args()
    if args.rounds != 5:
        parser.error('This approved scale matrix uses exactly five measured rounds')
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.absolute()
    if not args.mpiexec.is_file():
        parser.error('MPI launcher is missing')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    build = ROOT/'build_insitu_air5_device_render'
    libraries = ROOT.parent/'astr_dependencies/build/paraview-6.1.1-astr-device-gcc13/lib'
    dependencies = [args.executable, ROOT/'scripts/insitu/device_render_pipeline.py',
                    libraries/'libviskores_cont-pv6.1.so.1.1.0',
                    libraries/'libvtkRenderingOpenGL2-pv6.1.so.6.1']
    hashes = {str(path): digest(path) for path in dependencies}
    report = dict(mapping=args.mapping, grid=[256]*3, topology=[2, 1, 1], steps=4, frames=[2, 4],
                  dt=2e-5, maximum_cfl=.5, rounds=args.rounds, fingerprints=hashes,
                  wall_temperature_nondimensional=1. if args.mapping == 'y-wavy' else None,
                  scope='Four-step timing: no field/checkpoint/statistics/oracle I/O, profiler or external resource sampler; JPEG/EPS only',
                  warmup='One excluded independent process per backend; first-frame lazy setup remains in each complete window',
                  records=[])
    report_path = args.output/'timing.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n')

    def run(pipeline, mode, name, steps, render, resource_baseline=None):
        exclusive_devices()
        if group_bytes(build)+IMAGE_BUDGET > GROUP_BUDGET:
            raise RuntimeError('Approved 256 GiB group budget cannot contain another image run')
        if any(digest(path) != hashes[str(path)] for path in dependencies):
            raise RuntimeError('Executable or dependency changed during matched timing')
        settings = current_arguments(args.output, samples=False)
        settings.executable, settings.mpiexec = args.executable, args.mpiexec
        settings.runtime_timeout_seconds = 3600 if steps == 100 else 600
        settings.directory_budget_bytes = IMAGE_BUDGET
        settings.scale_timestep, settings.maximum_cfl = 2e-5, .5
        settings.resource_device_extra_budget_bytes = 6*1024**3
        settings.resource_host_extra_budget_bytes = 16*1024**3
        settings.resource_device_reserve_bytes = 2*1024**3
        config = scale_configuration(pipeline, mode) if render else '&insitu_run enabled=f /\n'
        if steps == 100 and render:
            config = production_configuration(pipeline, mode)
        case, size = run_case(settings, ROOT, 'gpu', 2, name, steps,
            grid='256,256,256', tgv_mapping=args.mapping, enabled=False, checkpoint_enabled=False,
            insitu_config=config, postprocess_transport=mode, insitu_timing=True,
            monitor_resources=steps == 100, resource_baseline=resource_baseline,
            curve_wall_temperature=report['wall_temperature_nondimensional'])
        no_large_io(case)
        if render:
            check_frames(case, 2, pipeline, tuple(range(1, 101)) if steps == 100 else (2, 4), audit=False)
        else:
            assert not list((case/'outdat').rglob('*.jpeg'))
        return dict(case=str(case), directory_bytes=size, timing=timing(case, 2),
                    cfl=json.loads((case/'cfl_gate.json').read_text()))

    save()
    for round_id in range(-1, args.rounds):
        for pipeline, mode in round_order(round_id):
            record = run(pipeline, mode, f'{pipeline}_{mode}_{round_id+1}', 4, True)
            record.update(backend=backend_key(pipeline, mode), round=round_id,
                          step4_output_seconds=later_output(Path(record['case']), 2))
            report['records'].append(record)
            save()
            print(args.mapping, record['backend'], round_id,
                  record['timing']['completed_window']['seconds'], flush=True)
    report['summary'] = summarize(report['records'], args.rounds)
    save()
    if args.production_steps:
        pipeline, mode = report['summary']['selected'].split('/')
        report['production'] = dict(selected=report['summary']['selected'])
        baseline = None
        for render in (False, True):
            label = 'on' if render else 'off'
            report['production'][label] = run(pipeline, mode, 'production_'+label, 100, render, baseline)
            observed = json.loads((Path(report['production'][label]['case'])/'resources.sampled.json').read_text())
            report['production'][label]['external_20ms_resources'] = observed
            if not render:
                baseline = observed
            save()
            print(args.mapping, 'production_'+label,
                  report['production'][label]['timing']['completed_window']['seconds'], flush=True)
        on = report['production']['on']['timing']['completed_window']['seconds']
        off = report['production']['off']['timing']['completed_window']['seconds']
        report['production'].update(additional_complete_window_seconds=on-off,
                                    complete_window_ratio=on/off,
                                    timing_scope='Resource-observed production, not clean five-round performance')
        save()
    if any(digest(path) != hashes[str(path)] for path in dependencies):
        raise RuntimeError('Executable or dependency changed after matched timing')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
