"""Exercise actual compact10 field coupling, layouts and global line MPI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


def read_field(path):
    with path.open('rb') as stream:
        header = np.fromfile(stream, np.int32, 12)
        if header.size != 12 or header[10] != 16:
            raise AssertionError('compact10 record identity')
        dtype = {4: np.float32, 8: np.float64}[int(header[0])]
        shape = tuple(header[4:7] + 1) + (5,)
        values = np.fromfile(stream, dtype, int(np.prod(shape)))
        if values.size != np.prod(shape) or stream.read(1) or not np.isfinite(values).all():
            raise AssertionError('compact10 finite record payload')
    return header, values.reshape(shape, order='F')


def fourier_reference():
    alpha = .49
    c = np.array([(193 + 126*alpha)/512, (105 + 302*alpha)/512,
        (-15 + 30*alpha)/128, (45 - 90*alpha)/1024, (-5 + 10*alpha)/512,
        (1 - 2*alpha)/1024])

    def gain(wave):
        angle = 2*np.pi*wave/16
        return 2*np.sum(c*np.cos(np.arange(6)*angle))/(1 + 2*alpha*np.cos(angle))

    i, j, k = np.indices((17, 17, 17))
    x, y, z = 2*np.pi*np.array([i, j, k])/16
    result = np.empty((17, 17, 17, 5))
    for component, base in enumerate([2., .125, .25, .5, 16.], 1):
        result[..., component-1] = base + .03*component*(gain(1)*np.sin(x) + gain(5)*np.cos(5*y)) + \
            .01*component*gain(1)*gain(2)*gain(3)*np.cos(x+2*y+3*z) + \
            .005*component*gain(8)**3*(-1.)**(i+j+k)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'mpiexec', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--precision', choices=('fp32', 'fp64'), required=True)
    parser.add_argument('--transport', choices=('pinned', 'device-aware'), required=True)
    parser.add_argument('--cases', nargs='+', choices=('np1', 'np2x', 'np2y', 'np2z'),
        default=['np1', 'np2x', 'np2y', 'np2z'])
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--sanitizer', choices=('memcheck',))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.binary, root/'tested_probe')
    shutil.copy2(__file__, root/'runner.py')
    env = os.environ | dict(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    if args.transport == 'device-aware':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_opal_cuda_support='1')
    width = 4 if args.precision == 'fp32' else 8
    analytic = fourier_reference()
    report = dict(precision=args.precision, transport=args.transport,
        binary_sha256=hashlib.sha256((root/'tested_probe').read_bytes()).hexdigest(), cases={})
    observed = {}
    for case in args.cases:
        ranks = 1 if case == 'np1' else 2
        axis = 1 if ranks == 1 else 'xyz'.index(case[-1])+1
        directory = root/case
        directory.mkdir()
        command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', str(ranks)]
        if args.sanitizer:
            command += ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool', 'memcheck', '--error-exitcode', '99', '--leak-check', 'full',
                '--log-file', str(directory/'sanitizer.%p.log')]
        command += [str(root/'tested_probe'), str(axis), args.transport, str(directory)]
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, check=True)
        text = (directory/'run.log').read_text()
        rows = re.findall(r'^CPP_COMPACT10\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.Ee+-]+)$', text, re.M)
        if len(rows) != 36*ranks or text.count('CPP_COMPACT10_PASS ') != ranks:
            raise AssertionError('compact10 incomplete rank/layout/synchronization/sample/axis observations')
        lu_error = max(float(row[-1]) for row in rows)
        fourier_error = 0.
        data = {}
        for rank in range(ranks):
            header, reference = read_field(directory/f'rank{rank:08d}.debug0.mode1.bin')
            if header[0] != width or header[1] != rank or header[2] != axis:
                raise AssertionError('compact10 type/rank/axis identity')
            for debug in (0, 1):
                for mode in (1, 2, 3):
                    h, values = read_field(directory/f'rank{rank:08d}.debug{debug}.mode{mode}.bin')
                    if h[3] != debug or h[11] != mode or not np.array_equal(h[4:11], header[4:11]) or \
                            values.tobytes() != reference.tobytes():
                        raise AssertionError('compact10 layout/batch/synchronization words')
            body = tuple(slice(int(o), int(o+n+1)) for o, n in zip(header[7:10], header[4:7]))
            target = analytic[body]
            scale = np.maximum(1., np.max(np.abs(target), axis=(0, 1, 2)))
            fourier_error = max(fourier_error, float(np.max(np.abs(reference-target)/scale)))
            if args.compare:
                h, values = read_field(args.compare/case/f'rank{rank:08d}.debug0.mode1.bin')
                if not np.array_equal(header, h) or values.tobytes() != reference.tobytes():
                    raise AssertionError('compact10 transport changed output words')
            data[rank] = (header, reference)
        if not np.isfinite([lu_error, fourier_error]).all() or width == 8 and max(lu_error, fourier_error) > 2e-10:
            raise AssertionError('compact10 finite/FP64 numerical gate')
        if args.sanitizer:
            paths = list(directory.glob('sanitizer.*.log'))
            if len(paths) != ranks or any('ERROR SUMMARY: 0 errors' not in path.read_text() or
                    'LEAK SUMMARY: 0 bytes leaked' not in path.read_text() for path in paths):
                raise AssertionError('compact10 Memcheck records')
        observed[case] = data
        report['cases'][case] = dict(command=command, cpu_rhs_lu_scaled_max=lu_error,
            fourier_scaled_max=fourier_error, layout_batch_sync_words_equal=True,
            transport_words_equal=args.compare is not None, zero_field_exact=True, sanitizer=args.sanitizer)
        print(case, json.dumps(report['cases'][case]), flush=True)
    if 'np1' in observed:
        full = observed['np1'][0][1]
        for case, data in observed.items():
            error = 0.
            for header, values in data.values():
                body = tuple(slice(int(o), int(o+n+1)) for o, n in zip(header[7:10], header[4:7]))
                target = full[body]
                error = max(error, float(np.max(np.abs(values-target)/np.maximum(1.,
                    np.max(np.abs(target), axis=(0, 1, 2))))))
            if width == 8 and error > 2e-10:
                raise AssertionError('compact10 decomposition comparison')
            report['cases'][case]['np1_scaled_max'] = error
    report['scope'] = 'Periodic actual compact10 field/halo/global PCR coupling; not FP32 main, physical boundaries, CFD accuracy or performance'
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
