import numpy as np
from analyze_air5_flow_monitor import analyze, block_statistics, separated_intervals


def monitor_difference(old, new, filename):
    assert np.isfinite(old).all() and np.isfinite(new).all()
    if filename == 'air5_wall.dat':
        assert old.shape[1] in (9,11) and new.shape[1] in (9,11)
        width = min(old.shape[1],new.shape[1])
        old,new = old[:,:width],new[:,:width]
    assert old.shape == new.shape
    return float(np.max(abs(new-old)/np.maximum(1.,abs(old))))


def test_monitor_comparison_registered_wall_schemas():
    old = np.ones((2,9))
    new = np.ones((2,11))
    assert monitor_difference(old,new,'air5_wall.dat') == 0.
    new[:,10] = 2.
    assert monitor_difference(np.ones((2,11)),new,'air5_wall.dat') == 1.
    with np.testing.assert_raises(AssertionError):
        monitor_difference(old,new[:,:10],'air5_wall.dat')
    with np.testing.assert_raises(AssertionError):
        monitor_difference(old,new,'air5_profiles.dat')
    new[:,10] = np.nan
    with np.testing.assert_raises(AssertionError):
        monitor_difference(old,new,'air5_wall.dat')


def test_constant_has_no_false_fluctuations():
    a=np.ones((4,2,12))
    result=block_statistics(a)
    np.testing.assert_array_equal(result['rms'],0)
    np.testing.assert_array_equal(result['favre_velocity_rms'],0)


def test_favre_and_reynolds_are_distinct():
    a=np.ones((2,1,12)); a[:,0,0]=[1,3]; a[:,0,1]=[0,4]
    result=block_statistics(a)
    assert result['mean'][0][1]==2
    assert result['favre_velocity'][0][0]==3


def test_multiple_bubbles_and_open_boundaries():
    x=np.arange(7,dtype=float)
    intervals=separated_intervals(x,np.array([1,-1,1,1,-1,1,-1]))
    assert len(intervals)==3
    assert intervals[0]['length']==1
    assert intervals[-1]['end'] is None
    assert separated_intervals(x,np.ones(7))==[]


def test_original_probe_configuration_analysis(tmp_path):
    rows = np.ones((4,18))
    rows[:,0] = np.arange(4)
    rows[:,1] = np.arange(1,5)*0.25
    np.savetxt(tmp_path/'air5_probes.dat', rows)
    result = analyze(tmp_path, 2)
    assert result['keys'] == [[1.0]]
    assert result['recent_time_range'] == [0.75,1.0]
    np.testing.assert_array_equal(result['recent']['rms'], 0)


def test_wall_heat_components_keep_original_columns(tmp_path):
    profiles, walls = [], []
    for step in range(4):
        time = (step+1)*.25
        for j in range(3):
            profiles.append([step,time,1,1,j,1.,float(j),0.,
                             1.,float(j),0.,0.,1500.,1500.,20000.,.767,.233,0.,0.,0.])
        walls.append([step,time,1,1.,0.,0.,2.,-5.,20000.,-2.,-3.])
    np.savetxt(tmp_path/'air5_profiles.dat', profiles)
    np.savetxt(tmp_path/'air5_wall.dat', walls)
    result = analyze(tmp_path, 2)
    assert result['wall_variables'] == ['tau_xy','q_into_gas','p',
                                       'q_tr_into_gas','q_v_into_gas']
    assert result['wall_recent_mean'] == [[2.,-5.,20000.,-2.,-3.]]
    np.savetxt(tmp_path/'air5_wall.dat', np.asarray(walls)[:,:9])
    old = analyze(tmp_path, 2)
    assert old['wall_variables'] == ['tau_xy','q_into_gas','p']


def integration(baseline, output, executable):
    """Two-update gate using an immutable startup case, including shared probes."""
    import json
    import os
    import shutil
    import subprocess
    from run_air5_sbli_preflight import set_value

    output.mkdir(parents=True, exist_ok=False)
    reference = None
    report = {}
    for name, topology, gpu, profiles in (
        ('cpu', '1,1,1', False, True),
        ('gpu_x', '2,1,1', True, True),
        ('gpu_y', '1,2,1', True, True),
        ('gpu_probes_only', '2,1,1', True, False),
    ):
        case = output/name
        shutil.copytree(baseline/'gpu_np2/datin', case/'datin')
        # Interface nodes, corners, and duplicate user probes exercise original ownership.
        (case/'datin/monitor.dat').write_text('# i j k\n0 0 0\n32 64 8\n63 127 15\n32 64 8\n')
        set_value(case/'datin/input.air5_c4',
                  'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                  'f,t,f,f,f,f,t,t,'+('t' if gpu else 'f'))
        (case/'validation').mkdir()
        launch = json.loads((baseline/'gpu_np2/launch.json').read_text())
        env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
        env.update(launch['environment'])
        env['ASTR_FORCE_MPI_TOPOLOGY'] = topology
        if profiles:
            env['ASTR_AIR5_FLOW_MONITOR_STRIDE'] = '1'
        else:
            env.pop('ASTR_AIR5_FLOW_MONITOR_STRIDE', None)
        command = ['timeout', '900s', launch['command'][3], '--oversubscribe',
                   '-np', '2' if gpu else '1', str(executable), 'run', 'datin/input.air5_c4']
        print(name, flush=True)
        with (case/'run.log').open('w') as log:
            subprocess.run(command, cwd=case, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        probes = np.loadtxt(case/'monitor/air5_probes.dat')
        assert probes.shape == (8, 18)
        assert np.isfinite(probes).all()
        np.testing.assert_array_equal(probes[:, 2], [1,2,3,4]*2)
        np.testing.assert_array_equal(probes[[1,5], 3:], probes[[3,7], 3:])
        if reference is None:
            reference = probes
        error = float(np.max(abs(probes-reference)/np.maximum(1.,abs(reference))))
        assert error <= 2e-10, error
        report[name] = dict(probe_max_scaled=error)
        if profiles:
            for filename in ('air5_profiles.dat', 'air5_wall.dat'):
                old = np.loadtxt(baseline/('gpu_np2' if gpu else 'cpu_np1')/'monitor'/filename)
                new = np.loadtxt(case/'monitor'/filename)
                change = monitor_difference(old,new,filename)
                assert change <= 2e-10, change
                report[name][filename] = change
        else:
            assert not (case/'monitor/air5_profiles.dat').exists()
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    args = parser.parse_args()
    integration(args.baseline.resolve(), args.output.resolve(), args.executable.resolve())
