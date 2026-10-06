"""Bounded complete-step TGV adaptive monitoring and native publication."""
import os
from pathlib import Path
import re
import sqlite3

import h5py
import numpy as np
import pytest

from test_output_insitu_restart import arguments
from test_output_insitu_restart import configuration
from run_output_restart_validation import run_case, archive_schedule_payload,compare_fields

ROOT = Path(__file__).resolve().parents[2]
CPU = Path(os.environ.get('ASTR_OUTPUT_CPU_EXE', ROOT/'build_release_restart_cpu/bin/astr'))
GPU = Path(os.environ.get('ASTR_OUTPUT_INSITU_EXE', ROOT/'build_insitu_device_render/bin/astr'))
CONFIG = '''&adaptive_output
 enabled=t,monitor_steps=1,event_ids='decay',s_ref=1,t_ref=1,
 r_on=1,r_off=0.5,hold_time=0.003,
 window_ids='burst',window_modes='steps',window_steps(:,1)=3,7
/
'''
GROUPS = '''&volume
 enabled=t,interval_steps=5,adaptive=t,dense_interval_steps=1,window_ids='burst'
/
&slices
 enabled=t,mode='time',interval_time=0.006,adaptive=t,dense_interval_time=0.002,
 window_ids='burst',k_indices=4
/
'''
# A separate Re=1 scheduling fixture gives a decreasing kinetic-energy slope in
# the approved twelve-step window. These immutable thresholds are not DNS events.
EVENT_CONFIG=CONFIG.replace('r_on=1,r_off=0.5,hold_time=0.003', 'r_on=0.74,r_off=0.723,hold_time=0.005')
EVENT_GROUPS=GROUPS.replace("window_ids='burst'","event_ids='decay'")


def params(path, backend):
    p = arguments(path)
    p.executable = CPU if backend == 'cpu' else GPU
    p.statistics = False
    p.runtime_timeout_seconds = 120
    p.directory_budget_bytes = 67108864
    return p


def monitors(path):
    records = []
    for line in (path/'run.log').read_text().splitlines():
        if not line.startswith('ASTR_AP_MONITOR '):
            continue
        m = re.search(r'step=(\d+) time=\s*(\S+) kinetic_energy=\s*(\S+)',line)
        records.append((int(m[1]), float(m[2]), float(m[3])))
    return records


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('ranks', [1, 2])
def test_native_adaptive_windows_restart_and_isolation(tmp_path,backend,ranks):
    args = params(tmp_path,backend)
    continuous,_ = run_case(args,ROOT,backend,ranks,'continuous',12,checkpoint_interval=5,
                           archive_groups=GROUPS,adaptive_config=CONFIG,buffer_bytes=1048576)
    partial,_ = run_case(args,ROOT,backend,ranks,'partial',5,checkpoint_interval=5,
                        archive_groups=GROUPS,adaptive_config=CONFIG,buffer_bytes=1048576)
    cp = partial/'outdat/new/checkpoints/step000000000005'
    assert cp.is_dir()
    resumed,_ = run_case(args,ROOT,backend,ranks,'resumed',12,restore=cp,checkpoint_interval=5,
                        archive_groups=GROUPS,adaptive_config=CONFIG,buffer_bytes=1048576)
    off,_ = run_case(args,ROOT,backend,ranks,'off',12,checkpoint_interval=5,buffer_bytes=1048576)
    full = continuous/'outdat/new/checkpoints/step000000000012'
    tail = resumed/'outdat/new/checkpoints/step000000000012'
    disabled = off/'outdat/new/checkpoints/step000000000012'
    for name in ('control.bin','insitu_control.bin'):
        assert (full/name).read_bytes() == (tail/name).read_bytes(), name
    assert archive_schedule_payload(full/'archives.bin') == archive_schedule_payload(tail/'archives.bin')
    for other in (tail,disabled):
        compare_fields(full/'state.h5',other/'state.h5')
    assert [x for x in monitors(continuous) if x[0]>5] == monitors(resumed)
    log=(continuous/'run.log').read_text()
    assert 'id=fields step=3' in log and 'id=fields step=6' in log
    assert 'id=fields step=11' in log
    assert list(map(int,re.findall(r'ASTR_AP_PRODUCT id=fields step=(\d+) ',log))) == [3,4,5,6,11]
    assert sorted(x.name for x in (continuous/'outdat/new/checkpoints').iterdir() if x.name.startswith('step')) == [
        'step000000000010','step000000000012']


def test_monitor_characterization_cpu_gpu(tmp_path):
    results={}
    for backend in ('cpu','gpu'):
        args=params(tmp_path/backend,backend)
        case,_=run_case(args,ROOT,backend,2,'monitor',12,checkpoint_interval=5,
                        adaptive_config=CONFIG,buffer_bytes=1048576)
        results[backend]=case
    a=np.asarray(monitors(results['cpu'])); b=np.asarray(monitors(results['gpu']))
    np.testing.assert_array_equal(a[:,:2],b[:,:2])
    assert np.max(np.abs(a[:,2]-b[:,2])) <= 2e-10
    for name in [f'q{i:04d}' for i in range(1,12)]+['rank_extras']:
        with h5py.File(results['cpu']/'outdat/new/checkpoints/step000000000012/state.h5') as a, \
             h5py.File(results['gpu']/'outdat/new/checkpoints/step000000000012/state.h5') as b:
            assert np.max(np.abs(a[name][:]-b[name][:])) <= 2e-10


def test_low_reynolds_event_characterization(tmp_path):
    args=params(tmp_path,'cpu')
    case,_=run_case(args,ROOT,'cpu',1,'event_pilot',12,adaptive_config=CONFIG,
                    tgv_reynolds=1,buffer_bytes=1048576)
    values=np.asarray(monitors(case))
    rates=np.abs(np.diff(values[:,2])/np.diff(values[:,1]))
    assert rates[-1]<rates[0]


@pytest.mark.parametrize('ranks',[1,2])
def test_real_event_entry_exit_hold_exact_resume_and_statistics(tmp_path,ranks,record_property):
    results={}
    errors={}
    final='outdat/new/checkpoints/step000000000012'
    for backend in ('cpu','gpu'):
        args=params(tmp_path/backend,backend); args.statistics=True
        options=dict(checkpoint_interval=5,buffer_bytes=1048576,tgv_reynolds=1,
                     insitu_config=configuration(render=False,statistics=True))
        full,_=run_case(args,ROOT,backend,ranks,'event_continuous',12,
                       archive_groups=EVENT_GROUPS,adaptive_config=EVENT_CONFIG,**options)
        part,_=run_case(args,ROOT,backend,ranks,'event_partial',5,
                       archive_groups=EVENT_GROUPS,adaptive_config=EVENT_CONFIG,**options)
        tail,_=run_case(args,ROOT,backend,ranks,'event_resumed',12,
                       restore=part/'outdat/new/checkpoints/step000000000005',
                       archive_groups=EVENT_GROUPS,adaptive_config=EVENT_CONFIG,**options)
        off,_=run_case(args,ROOT,backend,ranks,'event_off',12,**options)
        for other in (tail,off):
            for name in ('state.h5','statistics.h5'):
                compare_fields(full/final/name,other/final/name)
        assert (full/final/'control.bin').read_bytes()==(tail/final/'control.bin').read_bytes()
        assert archive_schedule_payload(full/final/'archives.bin')==archive_schedule_payload(tail/final/'archives.bin')
        assert [r for r in monitors(full) if r[0]>5]==monitors(tail)
        lines=[line for line in (full/'run.log').read_text().splitlines() if line.startswith('ASTR_AP_MONITOR ')]
        active=[line.split(' active=')[1].split()[0] for line in lines]
        assert active==['F']+['T']*6+['F']*6
        identities=list(map(int,re.findall(r'ASTR_AP_PRODUCT id=fields step=(\d+) ',(full/'run.log').read_text())))
        assert identities==[1,2,3,4,5,6,11]
        results[backend]=full
    a=np.asarray(monitors(results['cpu'])); b=np.asarray(monitors(results['gpu']))
    errors['kinetic_energy']=float(np.max(np.abs(a[:,2]-b[:,2])))
    assert errors['kinetic_energy']<=2e-10
    for name in [f'q{i:04d}' for i in range(1,12)]+['rank_extras']:
        with h5py.File(results['cpu']/final/'state.h5') as a,h5py.File(results['gpu']/final/'state.h5') as b:
            errors[name]=float(np.max(np.abs(a[name][:]-b[name][:])))
            assert errors[name]<=2e-10
    record_property('absolute_errors',errors)


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('change',['threshold','scale','monitor','hold','window','period','association','dense','disable_product','disable_monitor'])
def test_selective_override_preserves_fixed_products_and_statistics(tmp_path,backend,change):
    args=params(tmp_path,backend); args.statistics=True
    groups=EVENT_GROUPS.replace("enabled=t,mode='time',interval_time=0.006,adaptive=t,dense_interval_time=0.002,\n event_ids='decay',k_indices=4",
                                "enabled=t,interval_steps=4,k_indices=4")
    assert groups!=EVENT_GROUPS
    opts=dict(checkpoint_interval=5,buffer_bytes=1048576,tgv_reynolds=1,
              insitu_config=configuration(render=False,statistics=True))
    first,_=run_case(args,ROOT,backend,2,'first',5,adaptive_config=EVENT_CONFIG,archive_groups=groups,**opts)
    source=first/'outdat/new/checkpoints/step000000000005'
    cfg,new=EVENT_CONFIG,groups
    if change=='threshold': cfg=cfg.replace('r_on=0.74','r_on=1')
    if change=='scale': cfg=cfg.replace('s_ref=1','s_ref=2')
    if change=='monitor': cfg=cfg.replace('monitor_steps=1','monitor_steps=2')
    if change=='hold': cfg=cfg.replace('hold_time=0.005','hold_time=0.001')
    if change=='window': cfg=cfg.replace('window_step'+'s(:,1)=3,7','window_steps(:,1)=6,10')
    if change=='period': new=new.replace('interval_steps=5','interval_steps=4')
    if change=='association': new=new.replace("event_ids='decay'","window_ids='burst'")
    if change=='dense': new=new.replace('dense_interval_steps=1','dense_interval_steps=2')
    if change=='disable_product': new=new.replace("enabled=t,interval_steps=5,adaptive=t,dense_interval_steps=1,event_ids='decay'","enabled=f,interval_steps=5")
    if change=='disable_monitor':
        cfg='&adaptive_output\n/\n'
        new=new.replace(",adaptive=t,dense_interval_steps=1,event_ids='decay'",'')
    reject='adaptive monitor/control metadata tail' if change in ('threshold','scale','monitor','hold','window','disable_monitor') else 'product options changed without override'
    run_case(args,ROOT,backend,2,'reject',12,restore=source,adaptive_config=cfg,archive_groups=new,reject=reject,**opts)
    changed,_=run_case(args,ROOT,backend,2,'changed',12,restore=source,adaptive_config=cfg,
                       archive_groups=new,override=True,**opts)
    same,_=run_case(args,ROOT,backend,2,'same',12,restore=source,adaptive_config=EVENT_CONFIG,
                    archive_groups=groups,**opts)
    final='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'):
        compare_fields(changed/final/name,same/final/name)
    def identities(root,product):
        return [int(p.name[4:]) for p in sorted((root/'outdat/new'/product).glob('segment*/step*'))]
    assert identities(changed,'slices')==identities(same,'slices')==[8,12]
    log=(changed/'run.log').read_text()
    if change=='window':
        assert 'ASTR_AP_RESET window burst' in log and 'ASTR_AP_RESET event decay' not in log
        assert identities(changed,'fields')==identities(same,'fields')
    elif change in ('threshold','scale','monitor','hold'):
        assert 'ASTR_AP_RESET event decay' in log
    elif change!='disable_monitor':
        assert monitors(changed)==monitors(same)
    if change=='disable_product': assert identities(changed,'fields')==[]
    if change=='disable_monitor': assert monitors(changed)==[]


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_binding_and_cpu_render_admission_rejected_before_output(tmp_path,backend):
    args=params(tmp_path,backend)
    # CPU rendering remains unadmitted; the GPU mode reaches event-ID validation.
    args.executable=GPU
    cfg=configuration(render=True,statistics=False).replace('step_interval=8',
        "product_ids='q_surface.image',product_modes='steps',product_steps=5,product_adaptive=t,product_dense_steps=1,product_events(1,1)='missing'")
    expected='native EGL rendering requires GPU solver binding' if backend=='cpu' else 'unknown/invalid adaptive in-situ event/window association'
    run_case(args,ROOT,backend,1,'reject_unknown',12,adaptive_config=EVENT_CONFIG,insitu_config=cfg,
             reject=expected,buffer_bytes=1048576)


@pytest.mark.parametrize('ranks',[1,2])
def test_adaptive_monitor_memcheck(tmp_path,ranks,record_property):
    args=params(tmp_path,'gpu'); args.runtime_timeout_seconds=300
    case,size=run_case(args,ROOT,'gpu',ranks,'memcheck',12,enabled=False,no_field_io=True,
                       adaptive_config=EVENT_CONFIG,tgv_reynolds=1,buffer_bytes=1048576,memcheck=True,
                       insitu_config='&insitu_run enabled=f /')
    assert len(monitors(case))==13
    logs=list(case.glob('memcheck.*.log'))
    assert len(logs)==ranks
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)
    record_property('directory_bytes',size)


def test_adaptive_scalar_transfer_attribution(tmp_path,record_property):
    args=params(tmp_path,'gpu')
    case,size=run_case(args,ROOT,'gpu',2,'scalar_trace',2,enabled=False,no_field_io=True,
                       adaptive_config=EVENT_CONFIG,tgv_reynolds=1,buffer_bytes=1048576,nsys_trace=True,
                       insitu_config='&insitu_run enabled=f /')
    paths=sorted(case.glob('trace.rank*.sqlite')); assert len(paths)==2
    ledger=[]
    for path in paths:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            kernel=db.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                'join StringIds s on s.id=k.demangledName order by k.end').fetchall()
            reductions=[k for k in kernel if 'adaptive_energy_reduce_kernel' in k[2]]
            partials=[k for k in kernel if 'adaptive_energy_partial_kernel' in k[2]]
            assert len(reductions)==len(partials)==3
            copies=db.execute('select m.start,e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind order by m.start').fetchall()
            payload=[]
            for start,kind,n in copies:
                preceding=[k for k in kernel if k[1]<=start]
                if preceding and 'adaptive_energy_reduce_kernel' in preceding[-1][2]:
                    payload.append((kind,n))
            assert payload==[('CUDA_MEMCPY_KIND_DTOH',16)]*3,payload
            ledger.append(dict(rank=path.name,monitor_calls=3,d2h_bytes=48,partial_bytes=80))
    assert not list((case/'outdat/new').rglob('*.h5'))
    record_property('transfer_ledger',ledger); record_property('directory_bytes',size)


def test_monitor_event_costs_without_products(tmp_path,record_property):
    args=params(tmp_path,'gpu'); costs={}
    for active in (False,True):
        case,_=run_case(args,ROOT,'gpu',2,'on' if active else 'off',12,enabled=False,no_field_io=True,
                        adaptive_config=EVENT_CONFIG if active else None,tgv_reynolds=1,
                        buffer_bytes=1048576,insitu_timing=True,insitu_config='&insitu_run enabled=f /')
        values={}
        for stage,step,rank,seconds in re.findall(r'ASTR_INSITU_STAGE_TIMING\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)',(case/'run.log').read_text()):
            values.setdefault(stage,{}).setdefault(int(rank),[]).append(float(seconds))
        costs[str(active)]=dict(max_rank_seconds={stage:max(map(sum,ranks.values())) for stage,ranks in values.items()},
                               monitor_samples=len(monitors(case)))
        if active:
            assert all(len(values[stage][rank])==13 for stage in ('adaptive_event_clock','adaptive_monitor','adaptive_event_update') for rank in (0,1))
        else:
            assert not any(stage.startswith('adaptive_') for stage in values)
    record_property('bounded_costs',costs)
