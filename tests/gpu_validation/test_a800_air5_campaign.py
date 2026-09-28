import pytest
from run_a800_air5_campaign import live_problem, verify_mapping
from check_air5_mach4_error_replay import check_window_contract


def test_runtime_monitor():
    assert live_problem('current CFL: 0.2') is None
    assert live_problem('current CFL: 1.01')
    assert live_problem('current CFL: nan')
    prefix = 'AIR5_CHEMISTRY half=1 min_rhos/min_T/max_T/min_Tv/max_Tv= '
    assert live_problem(prefix+'0 1500 3000 1500 3000 max_acc/rej/rhs/jac=1 0 2 1') is None
    assert live_problem(prefix+'0 900 3000 1500 3000')
    assert live_problem(prefix+'-1e-30 1500 3000 1500 3000')


def test_four_distinct_devices():
    text = '\n'.join(f' ** MPI rank {i} bound to GPU device {i} on local rank {i}' for i in range(4))
    verify_mapping(text)
    with pytest.raises(ValueError):
        verify_mapping(text.replace('device 3', 'device 1'))
    with pytest.raises(ValueError):
        verify_mapping(text+'\n'+text)


@pytest.mark.parametrize('topology', ['4,1,1', '2,2,1', '1,4,1'])
def test_four_rank_matched_window(topology):
    a = dict(checkpoint_sha256='x', executable_sha256='x', start_step=3000,
             start_time=4.09e-6, baseline='seed', topology=topology,
             convection_limiter='symmetric_species', diffusion_limiter='layered',
             updates=400, dt=5e-10, target_time=4.29e-6)
    assert check_window_contract(a, dict(a, updates=800, dt=2.5e-10)) == pytest.approx(4.29e-6)
