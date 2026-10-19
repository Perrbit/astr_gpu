"""Keep M12 per-step diagnostics without per-step configuration/file churn."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_static_checks_and_file_lifetime_are_outside_each_sample():
    source=(ROOT/'src_gpu/insitu_m12_gpu.cuf').read_text()
    sample=source.split('subroutine sample_m12_completed_gpu(',1)[1].split('end subroutine',1)[0]
    initialize=source.split('subroutine initialize_m12()',1)[1].split('end subroutine',1)[0]
    assert 'invalid M12 diagnostic reference state' in initialize
    assert sample.count('call MPI_Allreduce(')==1
    assert 'checkpoint_state_require(' not in sample
    assert 'position=\'append\'' not in sample
    assert 'close(probe_unit' not in sample
    assert 'flush(probe_unit' in sample
    assert 'call finish_m12_completed_gpu()' in (ROOT/'src/insitu_session.F90').read_text()


def test_folded_status_retains_clock_probe_and_physical_rejection():
    source=(ROOT/'src_gpu/insitu_m12_gpu.cuf').read_text()
    sample=source.split('subroutine sample_m12_completed_gpu(',1)[1].split('end subroutine',1)[0]
    for condition in ('step==last_step+1','ieee_is_finite(t)','all(ieee_is_finite(local))',
                      'all(ieee_is_finite(summary(1:3)))','all(summary(1:3)>0.d0)','minimum(4)==1.d0'):
        assert condition in sample
    assert sample.index('minimum(4)==1.d0')<sample.index('call MPI_Reduce(local,values')
    failure=source.split('subroutine diagnostic_require(',1)[1].split('end subroutine',1)[0]
    assert 'call MPI_Abort(MPI_COMM_WORLD,71,ierr)' in failure
