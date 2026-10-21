"""Native-kind completed-step archives, exact/cross-kind restart and real reader."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from run_native_precision_restart_gate import check_state, FINAL
from run_native_precision_representative_cross_restart_gate import check_probe
from run_output_archive_validation import check_frame, check_series, check_series_reader, frames, groups
from run_output_restart_validation import compare_fields, run_case
from test_output_archive_segments import check_parent_reader
from test_output_fields import DERIVED_NAMES


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/output'))
from repair_series import repair_segment, metadata


def snapshot(root):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def check_products(case, backend, width, *, exact):
    report = {}
    checkpoint = case / FINAL
    for product in ('fields', 'slices'):
        path = case / 'outdat/new' / product / 'segment00000000'
        report[product] = dict(times=check_series(case, product),
            offline=repair_segment(path), reader=check_series_reader(case, product, extra_names=DERIVED_NAMES))
        for step, frame in frames(case, product).items():
            with h5py.File(frame / 'data.h5') as data:
                leaves = [data] if product == 'fields' else [data[name] for name in data if name != 'frame_identity']
                for leaf in leaves:
                    header = metadata(leaf)
                    if header[0] != (2 if width == 4 else 1):
                        raise AssertionError('native archive precision identity differs')
                    for name in (*DERIVED_NAMES, 'density', 'velocity_x', 'velocity_y', 'velocity_z',
                                 'pressure', 'temperature', 'velocity'):
                        values = leaf[name][:]
                        if values.dtype != np.dtype(f'<f{width}') or not np.isfinite(values).all():
                            raise AssertionError('invalid native archive payload')
                    if not (leaf['density'][:] > 0).all() or not (leaf['temperature'][:] > 0).all():
                        raise AssertionError('invalid native archive physical state')
            for item in ET.parse(frame / 'data.xdmf').findall('.//DataItem'):
                if item.attrib['Precision'] != str(width):
                    raise AssertionError('native frame XDMF precision differs')
            if step == 12 and exact:
                report[product]['final_payload_bytes'] = check_frame(frame, product, checkpoint,
                    case / 'outdat/new/resources/geometry.h5', backend, extra_names=DERIVED_NAMES)
        # The runtime parent index and stopped-source reconstruction must agree,
        # including each ancestor's own precision, clocks and data references.
        report[product]['lineage'] = check_parent_reader(path, case / f'lineage_reader_{product}',
                                                       extra_names=DERIVED_NAMES)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in ('cpu', 'gpu'):
        for precision in ('fp32', 'fp64'):
            parser.add_argument(f'--{backend}-{precision}', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--precisions', nargs='+', choices=['fp32', 'fp64'], default=['fp32', 'fp64'])
    parser.add_argument('--cross', action='store_true')
    parser.add_argument('--memcheck-resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    archives = groups('steps').replace('interval_steps=2', 'interval_steps=5').replace(
        'interval_steps=3', 'interval_steps=5').replace('initial_frame=.true.',
        'velocity_gradient=t,vorticity=t,qcriterion=t,initial_frame=.true.')
    report = dict(grid_intervals=[16]*3, ranks=2, topology=[2, 1, 1], cases={})
    for backend in args.backends:
        for precision in args.precisions:
            local = SimpleNamespace(output=args.output, executable=getattr(args, f'{backend}_{precision}').resolve(),
                mpiexec=args.mpiexec.resolve(), case='tgv', mode='steps', restart_step=5,
                initial_dimension=0, legacy_statistics=False, statistics=True, initial_restart=False,
                filter_workspace='scalar', force='feedback', axis='x', no_samples=True, wall_samples=False)
            tag = f'{backend}_{precision}'
            options = dict(gpu_precision_mode='', buffer_bytes=4096)
            a, _ = run_case(local, ROOT, backend, 2, precision + '_continuous', 12, archive_groups=archives, **options)
            off, _ = run_case(local, ROOT, backend, 2, precision + '_off', 12, **options)
            b, _ = run_case(local, ROOT, backend, 2, precision + '_seed', 5, archive_groups=archives, **options)
            before = snapshot(b)
            source = b / 'outdat/new/checkpoints/step000000000005'
            c, _ = run_case(local, ROOT, backend, 2, precision + '_resumed', 12, restore=source,
                            archive_groups=archives,
                            memcheck=args.memcheck_resume and backend == 'gpu' and precision == 'fp32', **options)
            result = dict(state=check_state(c / FINAL / 'state.h5', precision), exact={}, output_invariance={})
            for name in ('state.h5', 'statistics.h5'):
                result['exact'][name] = compare_fields(a / FINAL / name, c / FINAL / name)
                result['output_invariance'][name] = compare_fields(a / FINAL / name, off / FINAL / name)
            for name in ('control.bin', 'insitu_control.bin'):
                if (a / FINAL / name).read_bytes() != (c / FINAL / name).read_bytes():
                    raise AssertionError(f'exact control differs: {name}')
            result['products'] = check_products(c, backend, 4 if precision == 'fp32' else 8, exact=True)
            if snapshot(b) != before:
                raise AssertionError('source run changed during exact restart/readers')
            result['source_unchanged'] = True
            report['cases'][tag] = result
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(tag + ': exact archive/state/statistics and real reader passed', flush=True)
            if args.cross:
                new = 'fp64' if precision == 'fp32' else 'fp32'
                local.executable = getattr(args, f'{backend}_{new}').resolve()
                d, _ = run_case(local, ROOT, backend, 2, f'{new}_cross_from_{precision}', 12, restore=source,
                    archive_groups=archives, exact_restart=False, restore_probe=True, **options)
                crossed = dict(restored_fields=check_probe(source, d, new, (True, True, True)),
                    state=check_state(d / FINAL / 'state.h5', new),
                    products=check_products(d, backend, 4 if new == 'fp32' else 8, exact=False))
                if snapshot(b) != before:
                    raise AssertionError('source run changed during cross-kind restart/readers')
                crossed['source_unchanged'] = True
                report['cases'][f'{backend}_{precision}_to_{new}'] = crossed
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(f'{backend} {precision} to {new}: native archive/parent reader passed', flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
