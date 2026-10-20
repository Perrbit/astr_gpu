"""CP9 rank-window cost accounting, independent of GPU availability."""
from pathlib import Path
import json

import pytest
import numpy as np

from run_compact_execution import window_times,phase_times
from run_compact_performance import summarize
from run_compact_flatplate import state_minima


def test_rank_max_is_after_window_sum_not_per_step():
    text='\n'.join(f'ASTR_GPU_RANK_RK_TIMING {r} {s} 0 0 {v}'
                   for r,values in enumerate(([100,1,3],[100,3,1]))
                   for s,v in enumerate(values))
    sums,cost=window_times(text,2,3,1)
    assert sums==[4,4]
    assert cost==2
    with pytest.raises(ValueError,match='missing/duplicate'):
        window_times(text+'\n'+text.splitlines()[0],2,3,1)


def test_three_rk_stage_costs_are_summed_per_complete_step():
    labels=('filter','convection','diffusion_flux','diffusion_halo','diffusion_rhs','rk_update')
    text='\n'.join(f'ASTR_GPU_PHASE_TIMING {rank} {step} {stage} {label} {rank+1}'
                   for rank in range(2) for step in range(3) for stage in (1,2,3) for label in labels)
    result=phase_times(text,2,3,1)
    assert result['filter']['rank_window_seconds']==[6,12]
    assert result['filter']['seconds_per_step']==6
    with pytest.raises(ValueError,match='missing/invalid'):
        phase_times(text.replace('1 2 3 filter 2','1 2 3 other 2'),2,3,1)


def test_unfiltered_diagnostics_do_not_invent_filter_time():
    labels=('convection','diffusion_flux','diffusion_halo','diffusion_rhs','rk_update')
    text='\n'.join(f'ASTR_GPU_PHASE_TIMING 0 0 {stage} {label} .1'
                   for stage in (1,2,3) for label in labels)
    result=phase_times(text,1,1,0,'f')
    assert 'filter' not in result
    assert result['diffusion_flux']['seconds_per_step']==pytest.approx(.3)


def test_state_probe_reuses_read_only_admissibility_outside_timing():
    root=Path(__file__).resolve().parents[2]
    source=(root/'src_gpu/gpu_runtime.cuf').read_text()
    body=source.split('subroutine gpu_time_integration_rk(',1)[1].split('end subroutine',1)[0]
    assert body.index("'ASTR_GPU_RANK_RK_TIMING'")<body.index('if(benchmark_state_check)')
    check=(root/'src_gpu/mp_positivity_gpu.cuf').read_text().split('logical function mp_inputs_need_repair_gpu()',1)[1].split('end function',1)[0]
    assert 'mp_input_check_kernel' in check
    assert 'refresh_point' not in check


def test_compact_nscbc_filter_diagnostic_covers_all_stages():
    source=(Path(__file__).resolve().parents[2]/'src_gpu/mainloop_gpu.cuf').read_text()
    body=source.split('subroutine prepare_profile_nscbc_gpu(stage)',1)[1].split('end subroutine',1)[0]
    assert "call end_gpu_phase('filter',nstep,stage)" in body
    assert 'call prepare_profile_nscbc_gpu(1)' in source
    assert 'call prepare_profile_nscbc_gpu(rkstep)' in source
    assert "call apply_explicit_filter_gpu(.false.,'')" in body


def test_explicit_boundary_correction_uses_a_shared_scoped_predicate():
    root=Path(__file__).resolve().parents[2]
    cpu=(root/'src/bc.F90').read_text()
    predicate=cpu.split('logical function explicit_profile_nscbc_corrections_enabled()',1)[1].split('end function',1)[0]
    assert "all(bctype==[11,22,41,52,1,1])" in predicate
    assert "difschm=='643e'" in predicate
    assert 'nscbc_farfield_incoming_only_enabled()' in predicate
    assert 'compact_stage_wall=compact_stage_wall.or.(explicit_profile_nscbc_corrections_enabled().and.jrk==0)' in cpu
    assert 'compact_plate_outlet=compact_plate_outlet.or.(ndir==2.and.explicit_profile_nscbc_corrections_enabled())' in cpu
    gpu=(root/'src_gpu/mainloop_gpu.cuf').read_text()
    assert 'staged_profile_nscbc_case=profile_nscbc_case.and.' in gpu
    assert '(compact_global_enabled().or.explicit_profile_nscbc_corrections_enabled())' in gpu


def test_profile_stage_probe_is_test_only_and_does_not_repair_state():
    root=Path(__file__).resolve().parents[2]
    source=(root/'src_gpu/mp_positivity_gpu.cuf').read_text()
    body=source.split('subroutine check_profile_validation_state_gpu(label,stage)',1)[1].split('end subroutine',1)[0]
    assert "'ASTR_VALIDATION_PROFILE_STAGE_STATE'" in body
    assert 'if(.not.enabled_check) return' in body
    assert 'mp_inputs_need_repair_gpu()' in body
    assert 'repair_crashfix' not in body
    loop=(root/'src_gpu/mainloop_gpu.cuf').read_text().split('subroutine time_integration_rk_gpu(',1)[1].split('end subroutine time_integration_rk_gpu',1)[0]
    assert 'use mp_positivity_gpu, only: check_profile_validation_state_gpu' in loop


def test_filtered_recovery_is_runtime_optional_and_precedes_the_stage_check():
    root=Path(__file__).resolve().parents[2]
    source=(root/'src_gpu/mainloop_gpu.cuf').read_text()
    helper=source.split('subroutine repair_filtered_state_gpu(stage)',1)[1].split('end subroutine',1)[0]
    assert 'if(.not.lcracon) return' in helper
    assert helper.count('call exchange_solution_halo_gpu()')==2
    assert helper.index('call exchange_solution_halo_gpu()')<helper.index('call repair_crashfix_state_gpu(stage)')
    preparation=source.split('subroutine prepare_profile_nscbc_gpu(stage)',1)[1].split('end subroutine',1)[0]
    assert preparation.index('call repair_filtered_state_gpu(stage)')<preparation.index('call check_profile_validation_state_gpu')
    cpu=(root/'src/mainloop.F90').read_text()
    filtered=cpu.split('! Derivatives and fluxes must use the filtered state',1)[1].split('#ifdef',1)[0]
    assert 'if(lcracon) then' in filtered
    assert 'call crashfix(ctime(16))' in filtered
    assert filtered.count('call qswap(timerept=ltimrpt)')==2


def write_timing_reports(tmp_path):
    for scheme,factor in (('643e',1),('643c',4)):
        for filtering in ('f','t'):
            for ranks in (1,2):
                for repeat,cost in enumerate((.2,.1,.3)):
                    path=tmp_path/'timing/tgv'/f'{scheme}_{filtering}_{ranks}_{repeat}'
                    path.mkdir(parents=True)
                    (path/'summary.json').write_text(json.dumps(dict(scheme=scheme,filter=filtering,
                        topology=f'{ranks},1,1',profiled=False,phase_timing=False,preflight=False,
                        executable_sha256='same-executable',crashfix=False,deltat=.001,steps=40,
                        warmup=10,workspace='full',compact_sync='dependency',transport='device-aware',
                        batch_rhs=65535,grid_intervals=[32,32,32],
                        rk_seconds_per_step=cost*factor/ranks)))


def test_performance_ratios_use_three_run_medians(tmp_path):
    write_timing_reports(tmp_path)
    summarize(tmp_path,'tgv')
    rows=json.loads((tmp_path/'tgv_performance.json').read_text())
    assert len(rows)==8
    for row in rows:
        assert row['speedup_2']==2
        assert row['efficiency_2']==1
        assert row['compact_over_explicit']==(4 if row['scheme']=='643c' else 1)


@pytest.mark.parametrize('field,value',[('crashfix',True),('executable_sha256','other-executable')])
def test_performance_rejects_mixed_runtime_identity(tmp_path,field,value):
    write_timing_reports(tmp_path)
    path=next((tmp_path/'timing/tgv').glob('*/summary.json'))
    record=json.loads(path.read_text())
    record[field]=value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='cannot mix'):
        summarize(tmp_path,'tgv')


def test_performance_does_not_accept_a_partial_matrix(tmp_path):
    write_timing_reports(tmp_path)
    next((tmp_path/'timing/tgv').glob('*/summary.json')).unlink()
    with pytest.raises(ValueError,match='requires three'):
        summarize(tmp_path,'tgv')


def test_flatplate_positive_state_gate():
    q=np.array([[1.,1.,0.,0.,.5+1/(.4*1.4*144)]])
    assert state_minima(q)['temperature']==pytest.approx(1.)
    q[0,4]=.4
    with pytest.raises(ValueError,match='pressure or temperature'):
        state_minima(q)
