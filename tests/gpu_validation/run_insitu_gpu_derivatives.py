"""32^3 native same-phase GPU gradient/Q gate; no alternative fluid solver."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case, compare_fields

ROOT = Path(__file__).resolve().parents[2]
FINAL = 'outdat/new/checkpoints/step000000000004'


def configuration(backend, library, pipeline):
    return f"""&insitu_run
 enabled=t, statistics=f, render=t, derivative_backend='{backend}',
 output_directory='outdat/render', schedule_mode='steps', step_interval=2,
 initial_frame=t, final_frame=t,
 host_budget_bytes=4294967296,device_budget_bytes=2147483648,device_reserve_bytes=1073741824,
 implementation_path='{library}',pipeline_file='{pipeline}'
/
"""


def check_resources(case, ranks, steps=(0,2,4), capture=True, statistics=False, dt=.001):
    records = []
    for rank in range(ranks):
        out = case/'outdat/render'
        with (out/f'resources.rank{rank:08d}.csv').open() as stream:
            identity = stream.readline().strip()
            rows = list(csv.DictReader(stream))
        stages=['baseline','initialized']+['frame_before','frame_after']*len(steps)+['finalize_before','finalize_after']
        if statistics:
            stages.append('statistics_exported')
        assert [row['stage'] for row in rows] == stages+['session_released']
        assert max(int(row['host_increment_bytes']) for row in rows) <= 4*1024**3
        assert max(int(row['device_increment_bytes']) for row in rows) <= 2*1024**3
        assert min(int(row['device_free_bytes']) for row in rows) >= 1024**3
        path=out/(f'capture.rank{rank:08d}.json' if capture else f'lifecycle_rank{rank}.json')
        receipt = json.loads(path.read_text())
        assert receipt == dict(frames=[[s,s*dt] for s in steps], finalized=True)
        records.append(dict(identity=identity, host_increment=max(int(row['host_increment_bytes']) for row in rows),
                            device_increment=max(int(row['device_increment_bytes']) for row in rows)))
    return records


def compare_captures(reference, candidate, cpu, ranks):
    differences = {}
    closure = 0.
    analytic = 0.
    h = 2*np.pi/32
    symbol = (1.5*np.sin(h)-.3*np.sin(2*h)+np.sin(3*h)/30)/h
    with h5py.File(cpu/FINAL/'state.h5') as state:
        for step in (0, 2, 4):
            for rank in range(ranks):
                name = f'fields.step{step:08d}.rank{rank:08d}.npz'
                with np.load(reference/'outdat/render'/name) as a, np.load(candidate/'outdat/render'/name) as b:
                    assert a.files == b.files
                    np.testing.assert_array_equal(a['xyz'], b['xyz'])
                    for field in a.files[1:]:
                        error = float(np.max(np.abs(a[field]-b[field])))
                        differences[field] = max(differences.get(field, 0.), error)
                        if field in ('q1','q2','q3','q4','q5','rho','u','v','w','pressure','temperature'):
                            np.testing.assert_array_equal(a[field], b[field])
                        else:
                            assert error <= 2e-10, (field, step, rank, error)
                    gradient = np.stack([b[f'd{component}_d{direction}'] for direction in 'xyz'
                                         for component in 'uvw'], axis=-1).reshape(-1,3,3).transpose(0,2,1)
                    expected_q = -.5*np.einsum('nij,nji->n', gradient, gradient)
                    closure = max(closure, float(np.max(abs(b['Q_rs']-expected_q))))
                    if step == 0:
                        x,y,z = b['xyz'].T
                        expected = symbol**2*(np.sin(x)**2*np.sin(y)**2-np.cos(x)**2*np.cos(y)**2)*np.cos(z)**2
                        analytic = max(analytic, float(np.max(abs(b['Q_rs']-expected))))
                    if step == 4:
                        indices = np.rint(b['xyz']/h).astype(int)
                        for component in range(1, 12):
                            field = ('q1 q2 q3 q4 q5 rho u v w pressure temperature').split()[component-1]
                            expected = state[f'q{component:04d}'][...][indices[:,2],indices[:,1],indices[:,0]]
                            error = float(np.max(abs(b[field]-expected)))
                            differences['cpu_'+field] = max(differences.get('cpu_'+field, 0.), error)
                            assert error <= 2e-10, ('CPU solver', field, error)
    assert closure <= 2e-10 and analytic <= 2e-10
    return dict(maxabs=differences, q_definition_maxabs=closure, initial_discrete_q_maxabs=analytic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'mpiexec', 'backend', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--memcheck', action='store_true')
    parsed = parser.parse_args()
    parsed.output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(output=parsed.output.resolve(), executable=parsed.executable.resolve(),
        mpiexec=parsed.mpiexec.resolve(), case='tgv', mode='steps', restart_step=5, initial_dimension=0,
        legacy_statistics=False, statistics=False, initial_restart=False, filter_workspace='scalar',
        force='feedback', axis='x', no_samples=True)
    report = dict(status='running', scope='32^3 periodic TGV NP1/2 private FP64 derivatives', checks=[])
    try:
        for ranks in (1, 2):
            cpu, _ = run_case(args, ROOT, 'cpu', ranks, 'solver_reference', 4,
                              grid='32,32,32', insitu_config='&insitu_run\n enabled=f\n/\n')
            cases = {}
            for backend in ('cpu', 'gpu'):
                config = configuration(backend, parsed.backend.resolve(), ROOT/'tests/gpu_validation/insitu_diagnostic_capture.py')
                case, size = run_case(args, ROOT, 'gpu', ranks, 'diagnostics_'+backend, 4,
                                     grid='32,32,32', insitu_config=config, memcheck=parsed.memcheck and backend=='gpu')
                cases[backend] = case
                report['checks'].append(dict(np=ranks, derivative_backend=backend, directory_bytes=size,
                                              resources=check_resources(case,ranks)))
            compare_fields(cases['cpu']/FINAL/'state.h5', cases['gpu']/FINAL/'state.h5')
            record = compare_captures(cases['cpu'], cases['gpu'], cpu, ranks)
            report['checks'].append(dict(np=ranks, comparison=record, solver_state='bitwise-identical-between-diagnostic-backends'))
        report['status']='passed-bounded-gpu-derivatives-not-field-selection-or-production'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (parsed.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
