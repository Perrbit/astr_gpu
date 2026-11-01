from pathlib import Path
from types import SimpleNamespace

import pytest

from run_compact_a800_preflight import CPU_TOPOLOGIES, configuration_matrix, timing_samples
from compact_benchmark import mpi_environment, formal_configuration_matrix
from run_compact_performance import preflight_matches
from run_compact_a800_campaign import summarize_runs


def test_matrix_covers_a800_gpu_topologies_and_cpu_physical_core_counts():
    rows = configuration_matrix(32, 16)
    assert len(rows) == 58
    for scheme in ('643e', '643c'):
        gpu = [r for r in rows if r['backend'] == 'gpu' and r['scheme'] == scheme]
        assert len(gpu) == 13
        assert sorted(r['ranks'] for r in gpu if r['family'] == 'strong') == [1,2,2,2,4,4,4,4,4,4]
        cpu = [r for r in rows if r['backend'] == 'cpu' and r['scheme'] == scheme]
        assert {r['ranks'] for r in cpu} == {1,2,4,8,16,32}
        for row in gpu:
            assert any(c['topology'] == row['topology'] and c['grid'] == row['grid'] for c in cpu)
        weak = [r for r in gpu if r['family'] == 'weak']
        assert [r['grid'] for r in weak] == [(16,16,16),(16,16,32),(16,32,32)]
    assert CPU_TOPOLOGIES[32] == (4,4,2)
    formal=formal_configuration_matrix()
    assert len(formal)==38
    assert len([r for r in formal if r['backend']=='gpu'])==26
    assert len([r for r in formal if r['backend']=='cpu'])==12


@pytest.mark.parametrize('halo,compact',[('device-aware','pinned'),('pinned','device-aware'),('device-aware','device-aware')])
def test_either_device_transport_selects_ucx_without_forcing_no_ipc(halo,compact):
    env=mpi_environment('gpu',halo,compact,'self,sm,cuda_copy,cuda_ipc')
    assert env['OMPI_MCA_pml']=='ucx'
    assert env['UCX_TLS']=='self,sm,cuda_copy,cuda_ipc'
    assert 'OMPI_MCA_opal_cuda_support' not in env


def test_cpu_uses_host_mpi_even_when_gpu_choices_are_device_aware():
    env=mpi_environment('cpu','device-aware','device-aware')
    assert env['OMPI_MCA_pml']=='ob1'
    assert 'UCX_TLS' not in env


def test_formal_timing_cannot_reuse_a_different_transport_preflight():
    args=SimpleNamespace(crashfix=False,halo_transport='device-aware',compact_transport='device-aware',
                         ucx_tls='self,sm,cuda_copy,cuda_ipc')
    record=dict(executable_sha256='same',status='pass',crashfix=False,transport='device-aware',
                halo_transport='device-aware',ucx_tls=args.ucx_tls)
    assert preflight_matches(record,'same',args)
    for name,value in [('halo_transport','pinned'),('transport','pinned'),('ucx_tls','self,sm,cuda_copy')]:
        assert not preflight_matches(dict(record,**{name:value}),'same',args)


def test_timing_is_rank_max_per_step_not_window_rank_max():
    text = '\n'.join(f'ASTR_CPU_RANK_RK_TIMING {rank} {step} 0 {value} {value}'
                     for rank, values in enumerate(([99,1,4],[99,4,1]))
                     for step, value in enumerate(values))
    assert timing_samples(text, 'cpu', 2, 3, 1) == [4,4]
    with pytest.raises(ValueError, match='missing|duplicate'):
        timing_samples(text+'\n'+text.splitlines()[0], 'cpu', 2, 3, 1)


def test_formal_summary_requires_five_repeats_and_same_operator_cpu_baseline():
    runs=[]
    for scheme,factor in [('643e',1),('643c',3)]:
        for backend,n,topology,cost in [('cpu',1,(1,1,1),100),('cpu',32,(4,4,2),10),
                                      ('gpu',1,(1,1,1),2),('gpu',4,(1,2,2),.6)]:
            for _ in range(5):
                runs.append(dict(scheme=scheme,backend=backend,ranks=n,topology=topology,grid=(512,)*3,
                    family='strong',median_rk_seconds=cost*factor,rank_max_rk_samples=[cost*factor]*5))
    summary=summarize_runs(runs,5)
    assert len(summary)==8
    for row in summary:
        if row['backend']=='gpu' and row['ranks']==4:
            assert row['speedup_vs_cpu']['32']==pytest.approx(10/.6)
            assert row['efficiency']==pytest.approx(2/.6/4)
    assert len(summarize_runs(runs[:-1],5))==7


def test_cpu_rk_timer_excludes_rkfirst_without_reordering_the_advance():
    source = (Path(__file__).resolve().parents[2]/'src/mainloop.F90').read_text()
    body = source.split('subroutine time_integration_rk(timerept)', 1)[1].split('end subroutine time_integration_rk', 1)[0]
    assert body.index('cpu_statistics_start=ptime()') < body.index('call rkfirst(skip_checkpoint=air5_reacting_case)')
    assert 'cpu_statistics_seconds=cpu_statistics_seconds+ptime()-cpu_statistics_start' in body
    assert 'cpu_rk_seconds=ptime()-time_beg-cpu_statistics_seconds' in body
    assert "'ASTR_CPU_RANK_RK_TIMING',mpirank,nstep" in body
    assert 'do rkstep=1,n_rk_steps' in body
