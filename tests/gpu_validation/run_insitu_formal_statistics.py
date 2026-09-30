"""Native namelist statistics versus an immutable test-oracle run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu', 'gpu', 'mpiexec', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--production', action='store_true',
                        help='Also verify no-config default-off with test environment variables present')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'checks': []}
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_HALO_TRANSPORT='pinned',
               ASTR_GPU_PRECISION_MODE='fp64')
    try:
        for ranks in (1, 2):
            env['ASTR_FORCE_MPI_TOPOLOGY'] = '2,1,1' if ranks == 2 else '1,1,1'
            for backend in ('cpu', 'gpu'):
                reference = args.reference/f'np{ranks}_{backend}_on'
                for enabled in ((None, False, True) if args.production else (False, True)):
                    label = 'unset' if enabled is None else str(int(enabled))
                    case = (args.output/f'np{ranks}_{backend}_{label}').resolve()
                    case.mkdir()
                    shutil.copytree(reference/'datin', case/'datin')
                    config = case/'insitu.nml'
                    config.write_text(
                        '&insitu_run\n'
                        f'enabled={"t" if enabled else "f"}, statistics=t, render=f,\n'
                        "statistics_window=0.0005,0.0025, output_directory='outdat',\n"
                        'host_budget_bytes=4294967296, device_budget_bytes=2147483648,\n'
                        'device_reserve_bytes=1073741824\n/\n')
                    runtime = dict(env, ASTR_INSITU_CONFIG=str(config))
                    if enabled is None:
                        runtime.pop('ASTR_INSITU_CONFIG')
                    # Formal configuration must not inherit legacy test controls.
                    runtime.update(ASTR_INSITU_SAMPLE_PREFIX='forbidden/sample',
                                   ASTR_INSITU_TEST_STATISTICS_WINDOW='invalid',
                                   ASTR_INSITU_TEST_PIPELINE='missing.py')
                    with (case/'run.log').open('w') as log:
                        subprocess.run([str(args.mpiexec.resolve()), '--mca', 'coll_hcoll_enable', '0',
                                        '-np', str(ranks), str(getattr(args, backend).resolve()),
                                        'run', 'datin/input.tgv'], cwd=case, env=runtime,
                                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                    assert 'The job is done!' in (case/'run.log').read_text()
                    products = sorted((case/'outdat').glob('sample*.bin'))
                    assert len(products) == (ranks if enabled else 0), products
                    for path in products:
                        assert path.name.startswith('sample.statistics.step00000003.rank'), path
                        if backend=='cpu':
                            assert path.read_bytes() == (reference/'outdat'/path.name).read_bytes(), path
                        else:
                            from insitu_statistics_reference import read_statistics,compare
                            ha,ma,va=read_statistics(path)
                            hb,mb,vb=read_statistics(reference/'outdat'/path.name)
                            np.testing.assert_array_equal(ha,hb)
                            np.testing.assert_array_equal(ma[:3],mb[:3])
                            np.testing.assert_allclose(ma[3],mb[3],atol=2e-10,rtol=0)
                            np.testing.assert_allclose(ma[4:]**2,mb[4:]**2,atol=2e-10,rtol=0)
                            report['checks'].append(dict(np=ranks,native_device_statistics=compare(va,vb)))
                    with h5py.File(case/'outdat/flowfield.h5') as actual, \
                            h5py.File(reference/'outdat/flowfield.h5') as expected:
                        for field in ('ro', 'u1', 'u2', 'u3', 'p', 't', 'nstep', 'time'):
                            np.testing.assert_array_equal(actual[field][...], expected[field][...])
                            assert actual[field][...].tobytes() == expected[field][...].tobytes()
                    report['checks'].append(dict(np=ranks, backend=backend, enabled=enabled,
                                                 statistics=('byte-identical' if backend=='cpu' else 'approved-D7-tolerance') if enabled else 'absent',
                                                 solver_fields='bitwise-identical'))
        for label, settings, expected in (
            ('host_budget', 'host_budget_bytes=1, device_budget_bytes=2147483648',
             'statistics host allocation exceeds node budget'),
            ('device_budget', 'host_budget_bytes=4294967296, device_budget_bytes=0',
             'GPU statistics require an explicit device budget'),
            ('unpaired_restart', "host_budget_bytes=4294967296, restore_batch='unused'",
             'formal restart requires both lrestart=t and restore_batch'),
        ):
            case = (args.output/f'reject_{label}').resolve()
            case.mkdir()
            shutil.copytree(args.reference/'np2_gpu_on/datin', case/'datin')
            config = case/'insitu.nml'
            config.write_text("&insitu_run enabled=t, statistics=t, statistics_window=0.0005,0.0025, "
                              "output_directory='outdat', "+settings+' /\n')
            runtime = dict(env, ASTR_INSITU_CONFIG=str(config), ASTR_FORCE_MPI_TOPOLOGY='2,1,1')
            with (case/'run.log').open('w') as log:
                result = subprocess.run([str(args.mpiexec.resolve()), '--mca', 'coll_hcoll_enable', '0',
                                         '-np', '2', str(args.gpu.resolve()), 'run', 'datin/input.tgv'],
                                        cwd=case, env=runtime, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            assert result.returncode != 0, label
            assert expected in (case/'run.log').read_text(), label
            assert not list((case/'outdat').glob('sample*.bin')), label
            report['checks'].append(dict(rejected=label, np=2))
        report['status'] = 'passed-native-statistics-lifecycle-not-full-IS3'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
