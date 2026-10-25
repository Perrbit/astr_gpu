"""Paired native-precision long TGV; ASTR alone advances the flow."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from compare_tgv_flowstate import read_flowstate
from run_compact_execution import read_rank_output,retained_checkpoints


ROOT = Path(__file__).resolve().parents[2]
DT = 2.5e-4
KINDS = ('fp64', 'fp32')
SCHEMES = ('643e', '643c')
DIAGNOSTICS = ('kenergy', 'enstrophy', 'dissipation')


def clock_at_steps(steps, dt, precision):
    real = np.dtype(precision.replace('fp', 'float')).type
    result = np.empty(len(steps))
    time = real(0)
    previous = 0
    for index, step in enumerate(steps):
        for _ in range(int(step)-previous):
            time = real(time+real(dt))
        result[index] = time
        previous = int(step)
    return result


def read_case(case, precision, steps, interval):
    summary = json.loads((case/'summary.json').read_text())
    if (summary['numerical_precision'] != precision or summary['steps'] != steps or
        summary['statistics_interval'] != interval or not summary['tgv_conservation'] or
        summary['filter'] != 't' or summary['profiled'] or not summary['state_check']):
        raise ValueError('Long TGV configuration differs from the requested pair')
    _, rows = read_flowstate(case)
    flow = np.asarray(rows)
    sample_steps = np.arange(interval, steps, interval)
    if not np.array_equal(flow[:, 0], sample_steps):
        raise ValueError('TGV samples do not have the expected complete-step identity')
    conservation = np.loadtxt(case/'tgv_conservation.dat', skiprows=1, ndmin=2)
    if (conservation.shape != (len(sample_steps)+1, 4) or
        not np.array_equal(conservation[:, 0], np.r_[0, sample_steps]) or
        not np.isfinite(conservation).all()):
        raise ValueError('Missing or nonfinite conservative-mean samples')
    expected_clock = clock_at_steps(conservation[:, 0], DT, precision)
    if not np.array_equal(conservation[:, 1], expected_clock) or not np.array_equal(flow[:, 1], expected_clock[1:]):
        raise ValueError('TGV clock differs from repeated source-kind addition')
    if np.any(flow[:, 2:] < 0) or np.any(conservation[:, 2:] <= 0):
        raise ValueError('Invalid TGV scalar diagnostics')
    text = read_rank_output(case, 2)
    clocks = re.findall(r'^ASTR_TGV_FINAL_CLOCK\s+(\d+)\s+(\S+)\s+(\S+)', text, re.M)
    if len(clocks) != 1 or int(clocks[0][0]) != steps:
        raise ValueError('Missing exact final step/clock record')
    final_clock, native_dt = map(float, clocks[0][1:])
    if final_clock != clock_at_steps([steps], DT, precision)[0]:
        raise ValueError('Final native clock is inconsistent')
    checkpoint_interval=summary.get('checkpoint_interval',0)
    checkpoints=retained_checkpoints(case,steps,checkpoint_interval) if checkpoint_interval else []
    if not checkpoint_interval and (list(case.rglob('*.h5')) or list(case.rglob('*.jpeg')) or list(case.rglob('*.bin'))):
        raise ValueError('Statistics-only long TGV emitted a field or render file')
    nominal = sample_steps*DT
    report = dict(precision=precision, steps=steps, native_dt=native_dt,
        final_native_time=final_clock, final_nominal_time=steps*DT,
        final_clock_drift=final_clock-steps*DT,
        final_native_dt_accumulation_drift=final_clock-steps*native_dt,
        diagnostic_window_nominal=[float(nominal[0]), float(nominal[-1])],
        diagnostic_phase='prepared/filter-before-RK, existing flowstate phase',
        checkpoint_interval=checkpoint_interval,retained_checkpoints=checkpoints,
        sampled_peaks={name: dict(value=float(flow[:, column+2].max()),
            step=int(sample_steps[flow[:, column+2].argmax()]),
            nominal_time=float(nominal[flow[:, column+2].argmax()]),
            native_time=float(flow[flow[:, column+2].argmax(), 1]))
            for column, name in enumerate(DIAGNOSTICS)},
        conservative_means={}, crashfix=summary['crashfix'],
        crashfix_repaired_node_visits=summary['crashfix_repaired_node_visits'],
        executable_sha256=summary['executable_sha256'])
    for column, name in enumerate(('density_mean', 'total_energy_mean'), 2):
        values = conservation[:, column]
        drift = values-values[0]
        report['conservative_means'][name] = dict(initial=float(values[0]), last=float(values[-1]),
            last_sample_step=int(conservation[-1, 0]), max_abs_drift=float(abs(drift).max()),
            last_relative_drift=float(drift[-1]/values[0]))
    return flow, conservation, report


def compare_curves(reference, candidate, dt=DT):
    if not np.array_equal(reference[:, 0], candidate[:, 0]):
        raise ValueError('Cannot compare TGV curves with different sample identities')
    time = reference[:, 0]*dt
    overlap = [max(reference[0, 1], candidate[0, 1]), min(reference[-1, 1], candidate[-1, 1])]
    common = (reference[:, 1]>=overlap[0]) & (reference[:, 1]<=overlap[1])
    if not np.any(common):
        raise ValueError('TGV native-clock samples have no common observation interval')
    result = {}
    for column, name in enumerate(DIAGNOSTICS, 2):
        a, b = reference[:, column], candidate[:, column]
        difference = b-a
        maximum = int(abs(difference).argmax())
        scale = float(abs(a).max())
        result[name] = dict(max_abs_difference=float(abs(difference[maximum])),
            max_difference_step=int(reference[maximum, 0]),
            max_difference_nominal_time=float(time[maximum]),
            rms_difference=float(np.sqrt(np.mean(difference**2))),
            max_difference_over_reference_peak=float(abs(difference).max()/scale) if scale else None,
            sampled_peak_difference=float(b.max()-a.max()),
            sampled_peak_nominal_time_difference=float(time[b.argmax()]-time[a.argmax()]),
            signed_time_integral_difference=float(np.trapezoid(difference, time)),
            same_native_clock_interpolated_max_abs_difference=float(abs(
                np.interp(reference[common, 1], candidate[:, 1], b)-a[common]).max()),
            same_native_clock_overlap=list(map(float, overlap)),
            same_native_clock_reference_samples=int(np.count_nonzero(common)))
    return result


def analyze(root, steps, interval):
    spec = json.loads((root/'specification.json').read_text())
    report = dict(status='completed-report-only-cross-precision', nominal_dt=DT,
        grid_intervals=spec['grid_intervals'], production_matrix=spec['production_matrix'],
        steps=steps, topology=[2, 1, 1], filter='t', cases={}, pairs={})
    for scheme in SCHEMES:
        flows = {}
        for precision in KINDS:
            name = scheme+'_'+precision
            flow, conservation, record = read_case(root/name, precision, steps, interval)
            flows[precision] = flow
            report['cases'][name] = record
            nominal = flow[:, 0]*DT
            energy_rate = -np.gradient(flow[:, 2], nominal)
            table = np.column_stack((flow[:, :2], nominal, flow[:, 2:], energy_rate))
            np.savetxt(root/(name+'_curves.csv'), table, delimiter=',',
                       header='step,native_time,nominal_time,kenergy,enstrophy,dissipation,negative_dEk_dt', comments='')
        report['pairs'][scheme] = compare_curves(flows['fp64'], flows['fp32'])
    source = ROOT/'documents/reference_data/tgv/spectral_Re1600_512.gdiag'
    dlr = np.loadtxt(source)
    report['dlr_background'] = dict(source=str(source.relative_to(ROOT)),
        columns=['time', 'kinetic energy', '-dEk/dt', 'enstrophy'],
        enstrophy_sampled_peak_time=float(dlr[dlr[:, 3].argmax(), 0]),
        use='background only; ASTR uses density-weighted enstrophy and its existing thermodynamics')
    report['diagnostic_definitions'] = dict(
        kenergy='0.5 mean(rho |u|^2)/(rho_inf U0^2)',
        enstrophy='0.5 mean(rho |curl(u)|^2)/(rho_inf (U0/L0)^2), L0=xmax/(2 pi)',
        dissipation='mean(2 mu (S:S - div(u)^2/3)), existing nondimensional Sutherland viscosity',
        negative_dEk_dt='offline centered differences on nominal sample time; one-sided at ends; '
                        'its difference from viscous dissipation includes pressure work and numerical effects',
        conservative_means='unique periodic nodes, native-kind GPU/MPI sums; reduction rounding is included')
    (root/'accuracy.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for precision in KINDS:
        parser.add_argument('--'+precision, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('run', 'analyze'), required=True)
    parser.add_argument('--only', choices=[s+'_'+p for s in SCHEMES for p in KINDS])
    parser.add_argument('--steps', type=int, default=40000)
    parser.add_argument('--grid', default='256,256,256')
    parser.add_argument('--interval', type=int, default=40)
    parser.add_argument('--checkpoint-interval', type=int, default=0)
    args = parser.parse_args()
    if args.steps <= args.interval or args.interval <= 0:
        parser.error('require steps > statistics interval > 0')
    if args.checkpoint_interval<0:
        parser.error('--checkpoint-interval must be nonnegative')
    root = args.out.resolve()
    if args.mode == 'analyze':
        analyze(root, args.steps, args.interval)
        return
    if any(getattr(args, kind) is None for kind in KINDS):
        parser.error('both native executables are required')
    root.mkdir(parents=True, exist_ok=True)
    spec = dict(steps=args.steps, interval=args.interval, grid_intervals=list(map(int, args.grid.split(','))),
                topology=[2, 1, 1], deltat=DT, filter='t',
                production_matrix=args.steps==40000 and args.grid=='256,256,256' and args.interval==40)
    if args.checkpoint_interval:
        spec.update(checkpoint_interval=args.checkpoint_interval,checkpoint_keep=2,exact_restart=True)
    path = root/'specification.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Existing long TGV group has a different specification')
    path.write_text(json.dumps(spec, indent=2)+'\n')
    for scheme in SCHEMES:
        for precision in KINDS:
            name = scheme+'_'+precision
            if args.only and args.only != name:
                continue
            print('START '+name, flush=True)
            subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/run_compact_execution.py'),
                '--exe', str(getattr(args, precision).resolve(strict=True)), '--out', str(root/name),
                '--case', 'tgv', '--native-precision', precision, '--scheme', scheme, '--filter', 't',
                '--grid', args.grid, '--deltat', str(DT), '--topology', '2,1,1',
                '--steps', str(args.steps), '--warmup', '0', '--statistics-interval', str(args.interval),
                '--listing-interval', str(max(args.interval, 1000)), '--buffer-output', '--tgv-conservation', '--state-check',
                '--checkpoint-interval', str(args.checkpoint_interval),
                '--compact-sync', 'explicit', '--transport', 'pinned', '--workspace', 'scalar'], check=True)
            _, _, record = read_case(root/name, precision, args.steps, args.interval)
            (root/(name+'_diagnostics.json')).write_text(json.dumps(record, indent=2)+'\n')
    if not args.only:
        analyze(root, args.steps, args.interval)


if __name__ == '__main__':
    main()
