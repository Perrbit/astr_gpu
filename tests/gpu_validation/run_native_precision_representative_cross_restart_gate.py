"""Static CURVE or Ma12 cross-kind restart, with explicit same-phase probes."""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np

from run_native_precision_m12_restart_gate import launch as launch_m12
from run_native_precision_restart_gate import check_state
from run_output_restart_validation import run_case


ROOT = Path(__file__).resolve().parents[2]


def wall_rng_packet(path, precision):
    """Decode the documented stream records, without searching payload bytes."""
    data = path.read_bytes()
    real = 'f' if precision == 'fp32' else 'd'
    prefix = struct.Struct('<8s8qq' + 3*real + 'iiq16sq' + real + 'iq' + real + '384s' + 6*real)
    header = prefix.unpack_from(data)
    if header[0] != b'ASTROC09':
        raise AssertionError('expected current legacy-random control schema')
    offset = prefix.size
    schedule = struct.Struct('<8s64sq' + real + '5sq' + real + 'q' + real + 'q' + real + 'qq3i')
    if schedule.unpack_from(data, offset)[0] != b'ASTRSC01':
        raise AssertionError('missing checkpoint schedule before RNG record')
    offset += schedule.size
    rng = struct.Struct('<8sii16384s16384s8q')
    magic, ranks, packet, compiler, options, *signature = rng.unpack_from(data, offset)
    if magic != b'ASTRWR02' or ranks < 1 or packet < 8:
        raise AssertionError('invalid current wall RNG record')
    offset += rng.size
    state = np.frombuffer(data, '<i4', ranks*packet, offset).reshape((packet, ranks), order='F').copy()
    offset += state.nbytes
    if offset != len(data):
        compact = struct.Struct('<8s64s8s17q' + 3*real)
        if compact.unpack_from(data, offset)[0] != b'ASTRCM01' or offset + compact.size != len(data):
            raise AssertionError('unsupported control record after wall RNG')
    return compiler, state


def check_probe(source, resumed, precision, periodic, cast_probe=None):
    dtype = np.dtype('<f4' if precision == 'fp32' else '<f8')
    report = {}
    with h5py.File(source / 'state.h5') as a, h5py.File(resumed / 'outdat/restore_probe.h5') as b:
        original_header, actual_header = a['identity'][:], b['identity'][:]
        width = 8 if original_header[1] == 2 else int(a['numerical_kind_bytes'][0])
        words = original_header[9:12]
        clocks = words.astype('<i4').view('<f4') if width == 4 else words.view('<f8')
        expected_words = clocks.astype(dtype).view(f'<i{dtype.itemsize}').astype('<i8')
        np.testing.assert_array_equal(actual_header[9:12], expected_words)
        np.testing.assert_array_equal(actual_header[3:6], original_header[3:6])
        if actual_header[8] != original_header[8]:
            raise AssertionError('restore probe moved the completed-step identity')
        for component in range(1, 6):
            name = f'q{component:04d}'
            original = a[name][:]
            numpy_cast = original.astype(dtype)
            converted = numpy_cast
            if cast_probe is not None:
                directory = resumed / 'native_cast_reference'
                directory.mkdir(exist_ok=True)
                input_path, output_path = directory / f'{name}.source.bin', directory / f'{name}.native.bin'
                original.tofile(input_path)
                result = subprocess.run([str(cast_probe.resolve()), 'cast', str(width),
                    str(input_path.resolve()), str(output_path.resolve())], text=True, capture_output=True, timeout=30)
                if result.returncode:
                    raise AssertionError(result.stdout + result.stderr)
                converted = np.fromfile(output_path, dtype=dtype).reshape(original.shape)
            expected = converted.copy()
            for physical_axis in range(3):
                if not periodic[physical_axis]:
                    continue
                axis = 2 - physical_axis
                lower, upper = [slice(None)]*3, [slice(None)]*3
                lower[axis], upper[axis] = 0, -1
                mean = dtype.type(.5) * (expected[tuple(lower)] + expected[tuple(upper)])
                expected[tuple(lower)] = mean
                expected[tuple(upper)] = mean
            actual = b[name][:]
            if actual.dtype != dtype or actual.tobytes() != expected.tobytes():
                raise AssertionError(f'{name}: conversion plus original periodic/shared-node rebuild differs')
            report[name] = dict(periodic_rebuild_max_abs=float(np.max(np.abs(expected - converted))))
            if cast_probe is not None:
                report[name]['native_vs_numpy_cast_changed_values'] = int(np.count_nonzero(
                    converted.view(f'i{dtype.itemsize}') != numpy_cast.view(f'i{dtype.itemsize}')))
                report[name]['native_vs_numpy_cast_max_abs'] = float(np.max(np.abs(
                    converted.astype('f8') - numpy_cast.astype('f8'))))
        for name in ('crashfix_layout', 'crashfix_partitions', 'crashfix_history'):
            if name in a:
                if name not in b or a[name][:].tobytes() != b[name][:].tobytes():
                    raise AssertionError('cross-kind restore changed crashfix history')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['curve', 'm12'], required=True)
    for backend in ('cpu', 'gpu'):
        for precision in ('fp32', 'fp64'):
            parser.add_argument(f'--{backend}-{precision}', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--mapping', choices=['periodic', 'y-wavy'], default='periodic')
    parser.add_argument('--diffusion', choices=['643e', '643c'], default='643e')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--ranks', nargs='+', type=int, choices=[1, 2], default=[2])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                      OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc',
                      OMPI_MCA_opal_cuda_support='0')
    report = dict(case=args.case, mapping=args.mapping if args.case == 'curve' else None, cases={})
    for backend in args.backends:
        for ranks in args.ranks:
            for old, new in (('fp32', 'fp64'), ('fp64', 'fp32')):
                tag = f'{backend}_np{ranks}_{old}_to_{new}'
                if args.case == 'curve':
                    local = SimpleNamespace(output=args.output, mpiexec=args.mpiexec,
                        case='tgv', mode='steps', restart_step=5, initial_dimension=0, legacy_statistics=False,
                        statistics=False, initial_restart=False, filter_workspace='scalar', force='feedback',
                        axis='y' if args.mapping == 'y-wavy' else 'x', no_samples=True, wall_samples=False,
                        maximum_cfl=.5)
                    options = dict(gpu_precision_mode='', grid='32,32,32', tgv_mapping=args.mapping)
                    local.executable = getattr(args, f'{backend}_{old}').resolve()
                    local.native_precision = old
                    local.scale_timestep = float(np.float32(2e-5)) if old == 'fp32' else 2e-5
                    b, _ = run_case(local, ROOT, backend, ranks, f'{old}_seed_for_{new}', 5, **options)
                    source = b / 'outdat/new/checkpoints/step000000000005'
                    local.executable = getattr(args, f'{backend}_{new}').resolve()
                    local.native_precision = new
                    local.scale_timestep = float(np.float32(2e-5)) if new == 'fp32' else 2e-5
                else:
                    local = SimpleNamespace(output=args.output, mpiexec=args.mpiexec, source=args.source,
                        cpu=getattr(args, f'cpu_{old}'), gpu=getattr(args, f'gpu_{old}'),
                        diffusion=args.diffusion, filter=args.filter)
                    b, _ = launch_m12(local, backend, ranks, f'{old}_seed_for_{new}', 5)
                    source = b / 'outdat/new/checkpoints/step000000000005'
                    local.cpu, local.gpu = getattr(args, f'cpu_{new}'), getattr(args, f'gpu_{new}')
                for path in (b / 'outdat/new/resources').iterdir():
                    if path.suffix == '.h5' or path.name == 'grid.2d':
                        path.chmod(0o444)
                before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in b.rglob('*') if p.is_file()}
                if args.case == 'curve':
                    c, _ = run_case(local, ROOT, backend, ranks, f'{new}_resumed_from_{old}', 12,
                        restore=source, exact_restart=False, restore_probe=True, **options)
                    periodic = (True, args.mapping == 'periodic', True)
                else:
                    c, _ = launch_m12(local, backend, ranks, f'{new}_resumed_from_{old}', 12,
                        restore=source, exact=False, restore_probe=True)
                    periodic = (False, False, True)
                cast_probe = (getattr(args, f'{backend}_{new}').resolve().parent / 'native_precision_probe'
                              if args.case == 'm12' else None)
                result = dict(restored_fields=check_probe(source, c, new, periodic, cast_probe),
                    final_state=check_state(c / 'outdat/new/checkpoints/step000000000012/state.h5', new))
                if args.case == 'm12':
                    old_compiler, old_rng = wall_rng_packet(source / 'control.bin', old)
                    new_compiler, new_rng = wall_rng_packet(c / 'outdat/restore_control_probe.bin', new)
                    if old_compiler != new_compiler:
                        raise AssertionError('wall RNG compiler/runtime identity differs')
                    np.testing.assert_array_equal(old_rng, new_rng)
                    result['latent_rng_packet_bit_identical'] = True
                    result['wall_rng_initialized_rank_flags'] = new_rng[6].tolist()
                after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in b.rglob('*') if p.is_file()}
                if before != after:
                    raise AssertionError('source run changed during cross-kind continuation')
                result['source_unchanged'] = True
                report['cases'][tag] = result
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(args.case + ' ' + tag + ': restored field/history gate passed', flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
