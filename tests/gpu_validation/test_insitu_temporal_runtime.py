"""TB4 true complete-step one-dimensional histories, independent of picture cadence."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
import pytest
from scipy.signal import welch

from test_insitu_sbli_statistics import launch,TOPOLOGIES,DT,compare
from test_insitu_selected_runtime import configuration
from test_insitu_temporal_catalog import definition
from run_output_restart_validation import compare_fields,run_case
from test_insitu_device_products import resident_configuration
from test_runtime_catalog_images import catalog_replay_library
from test_index_slice_runtime import arguments as output_arguments

ROOT=Path(__file__).resolve().parents[2]
EXE=os.environ.get('ASTR_SELECTED_RUNTIME_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the current root FP64 CPU/GPU executable')
spec=importlib.util.spec_from_file_location('astr_temporal_archive',ROOT/'scripts/insitu/temporal_archive.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


def records(raw=True):
    line=definition();line['sampling'].update(index=24,index_ranges=dict(i=[0,64],k=[8,8]))
    line['temporal']['block_samples']=4;line['schedule']['interval']=3
    probe=definition('probe_spectrum');probe['sampling']['indices']=[32,24,8]
    probe['temporal']=dict(segment_length=8)
    if raw: probe['temporal'].update(archive_samples=True,block_samples=4)
    probe['schedule']['interval']=3
    return [line,probe]


def config(window=(0.,1.)):
    text=configuration(window)
    return re.sub(r'(host_budget_bytes|device_budget_bytes)\s*=\s*\d+',r'\1=-1',text).replace('device_reserve_bytes=1073741824','device_reserve_bytes=0')


def history(case,name):
    paths=list((case/'outdat/new/temporal_products'/name).glob('*/segment*/block.*.csv'))
    groups=archive.read_histories(paths)
    assert len(groups)==1,(paths,groups.keys())
    return next(iter(groups.values()))


def spectrum_csv(path):
    text=path.read_text()
    header=next(line for line in text.splitlines() if not line.startswith('#'))
    lines=[line for line in text.splitlines() if not line.startswith('#')]
    data=np.loadtxt(lines[1:],delimiter=',',ndmin=2) if len(lines)>1 else np.empty((0,len(header.split(','))))
    return text,data


def metadata_objects(text):
    result={}
    for name in ('units','psd_units','reference_state'):
        line=next(line for line in text.splitlines() if line.startswith('# '+name+'='))
        result[name]=json.loads(line.split('=',1)[1])
    assert result['units']['time']=='native_time'
    assert result['units']['frequency']=='1/native_time'
    assert result['psd_units']['pressure']=='(solver-dimensionless)^2*native_time'
    assert len(result['reference_state'])==12 and all(np.isfinite(list(result['reference_state'].values())))
    return result


def check(case,backend,ranks,steps=12):
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    line=history(case,'time_space');probe=history(case,'probe_spectrum')
    assert len(line)==(steps+1)*65 and len(probe)==steps+1
    np.testing.assert_array_equal(np.unique(line['step']),np.arange(steps+1))
    np.testing.assert_array_equal(probe['step'],np.arange(steps+1))
    assert not np.any(line['gap_before']) and not np.any(probe['gap_before'])
    compare(probe['time'],np.arange(steps+1)*DT)
    paths=sorted((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.*.csv'))
    assert len(paths)==steps//3+1
    text,actual=spectrum_csv(paths[-1]);assert '# available=T' in text and 'valid_segments=2' in text
    source=np.column_stack([probe['velocity_x'],probe['pressure']])
    window=.5*(1-np.cos(2*np.pi*np.arange(8)/8))
    frequency,power=welch(source,fs=1/DT,window=window,nperseg=8,noverlap=4,detrend='constant',scaling='density',axis=0)
    compare(actual,np.column_stack([frequency,power]))
    with h5py.File(case/f'outdat/new/checkpoints/step{steps:012d}/state.h5') as state:
        rho=state['q0001'][8,24,:];mx=state['q0002'][8,24,:];my=state['q0003'][8,24,:]
        mz=state['q0004'][8,24,:];energy=state['q0005'][8,24,:]
    last=line[line['step']==steps]
    compare(last['velocity_x'],mx/rho)
    compare(last['pressure'],.4*(energy-.5*(mx**2+my**2+mz**2)/rho))
    compare(probe['pressure'][-1],last['pressure'][32])
    samples=re.findall(r'ASTR_TEMPORAL_SAMPLE rank=(\d+) id=(\S+) step=(\d+) one_dimensional_d2h_bytes=(\d+) ',
                       (case/'run.log').read_text())
    assert len(samples)==2*(steps+1)*ranks
    for name,nodes in [('time_space',65),('probe_spectrum',1)]:
        download=sum(int(row[3]) for row in samples if row[1]==name)
        assert download==(nodes*2*8*(steps+1) if backend=='gpu' else 0),(name,download)
    return line,probe


@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_actual_temporal_line_probe_and_welch(tmp_path,topology):
    cases={backend:launch(tmp_path/backend,backend,topology,records=records(),insitu_config=config(),
                          checkpoint=True,unrestricted=True) for backend in ('cpu','gpu')}
    data={backend:check(case,backend,int(np.prod(topology))) for backend,case in cases.items()}
    for cpu,gpu in zip(data['cpu'],data['gpu']):
        for field in cpu.dtype.names: compare(cpu[field],gpu[field])


def test_actual_spanwise_line_across_z_partitions(tmp_path):
    line=records()[0];line['id']='spanwise-line'
    line['sampling'].update(index=1,index_ranges=dict(i=[32,32],k=[0,16]))
    cases={backend:launch(tmp_path/backend,backend,(1,1,2),records=[line],insitu_config=config(),
        checkpoint=True,unrestricted=True) for backend in ('cpu','gpu')}
    rows={}
    for backend,case in cases.items():
        data=history(case,'spanwise-line');rows[backend]=data
        np.testing.assert_array_equal(np.unique(data['step']),np.arange(13))
        assert len(data)==13*17 and not np.any(data['gap_before'])
        last=data[data['step']==12]
        with h5py.File(case/'outdat/new/checkpoints/step000000000012/state.h5') as state:
            rho,mx,my,mz,energy=[state[f'q{c:04d}'][:,1,32] for c in range(1,6)]
        compare(last['velocity_x'],mx/rho)
        compare(last['pressure'],.4*(energy-.5*(mx**2+my**2+mz**2)/rho))
        with h5py.File(case/'datin/grid.flatplate.h5') as mesh:
            for field in ('x','y','z'):compare(last[field],mesh[field][:,1,32])
        assert np.all(np.diff(last['z'])>0)
        samples=re.findall(r'ASTR_TEMPORAL_SAMPLE rank=\d+ id=spanwise-line step=\d+ one_dimensional_d2h_bytes=(\d+)',
            (case/'run.log').read_text())
        assert len(samples)==26
        assert sum(map(int,samples))==(17*2*8*13 if backend=='gpu' else 0)
    for field in rows['cpu'].dtype.names:compare(rows['gpu'][field],rows['cpu'][field])


@pytest.mark.parametrize('bounds',[(2.5*DT,8.5*DT),(0.,4*DT)])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_welch_common_window_and_exact_continuation(tmp_path,backend,bounds):
    items=records();items[1]['temporal']['segment_length']=4;items[1]['schedule']['interval']=1
    common=dict(records=items,insitu_config=config(bounds),checkpoint=True,unrestricted=True)
    full=launch(tmp_path/'full',backend,(1,1,2),**common)
    raw=history(full,'probe_spectrum');line=history(full,'time_space')
    np.testing.assert_array_equal(raw['step'],np.arange(13))
    np.testing.assert_array_equal(np.unique(line['step']),np.arange(13))
    eligible=raw[(raw['time']>=bounds[0])&(raw['time']<=bounds[1])]
    n=len(eligible);count=1+(n-4)//2
    window=.5*(1-np.cos(2*np.pi*np.arange(4)/4))
    f,p=welch(np.column_stack([eligible['velocity_x'],eligible['pressure']]),fs=1/DT,
        window=window,nperseg=4,noverlap=2,detrend='constant',scaling='density',axis=0)
    folder=full/'outdat/new/temporal_products/probe_spectrum'
    for step in (10,11,12):
        path=next(folder.glob(f'*/segment*/spectrum.step{step:012d}.csv'))
        text,data=spectrum_csv(path)
        assert f'valid_segments={count};' in text
        assert f'welch_sampled_endpoints={n};' in text
        assert f'welch_first_sample_step={eligible["step"][0]};' in text
        assert f'welch_last_sample_step={eligible["step"][-1]};' in text
        compare(data,np.column_stack([f,p]))
    prefix=launch(tmp_path/'prefix',backend,(1,1,2),steps=5,**common)
    restores=[('inside-or-after',prefix/'outdat/new/checkpoints/step000000000005'),
        ('after',full/'outdat/new/checkpoints/step000000000010')]
    if bounds[0]>0:
        before=launch(tmp_path/'before',backend,(1,1,2),steps=2,**common)
        restores.append(('before',before/'outdat/new/checkpoints/step000000000002'))
    ending='outdat/new/checkpoints/step000000000012'
    for label,batch in restores:
        tail=launch(tmp_path/('tail-'+label),backend,(1,1,2),restore=batch,**common)
        for filename in ('state.h5','statistics.h5'):compare_fields(full/ending/filename,tail/ending/filename)
        path=next((tail/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
        text,data=spectrum_csv(path)
        assert f'welch_sampled_endpoints={n};' in text
        compare(data,np.column_stack([f,p]))


@pytest.mark.parametrize('keep_shared',[False,True])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_common_window_respects_actual_shared_sampling_gaps(tmp_path,backend,keep_shared,catalog_replay_library):
    line,probe=records();probe['temporal']['segment_length']=4;probe['schedule']['interval']=1
    keeper=deepcopy(probe);keeper['id']='sampling-keeper';keeper['temporal']=dict(segment_length=4)
    keepers=[line,*([keeper] if keep_shared else [])]
    bounds=(2.5*DT,8.5*DT)
    case=launch(tmp_path/backend,backend,(2,1,2),records=[*keepers,probe],insitu_config=config(bounds),
        unrestricted=True,catalog_replay=(catalog_replay_library,{6:keepers,8:[*keepers,probe]}))
    path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,data=spectrum_csv(path)
    assert f'welch_sampled_endpoints={6 if keep_shared else 4};' in text
    assert 'welch_first_sample_step=3;' in text and 'welch_last_sample_step=8;' in text
    if keep_shared:
        line=history(case,'time_space');line=line[line['node']==33]
        eligible=line[(line['time']>=bounds[0])&(line['time']<=bounds[1])]
        window=.5*(1-np.cos(2*np.pi*np.arange(4)/4))
        f,p=welch(np.column_stack([eligible['velocity_x'],eligible['pressure']]),fs=1/DT,
            window=window,nperseg=4,noverlap=2,detrend='constant',scaling='density',axis=0)
        assert 'valid_segments=2;' in text
        compare(data,np.column_stack([f,p]))
    else:
        assert 'valid_segments=0;' in text and '# available=F;' in text and len(data)==0


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_temporal_exact_restart_and_output_isolation(tmp_path,backend):
    common=dict(records=records(),insitu_config=config(),checkpoint=True,unrestricted=True)
    full=launch(tmp_path/'full',backend,(2,1,1),**common)
    prefix=launch(tmp_path/'prefix',backend,(2,1,1),steps=5,**common)
    tail=launch(tmp_path/'tail',backend,(2,1,1),restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    off=launch(tmp_path/'off',backend,(2,1,1),records=[],insitu_config=config(),checkpoint=True,unrestricted=True)
    ending='outdat/new/checkpoints/step000000000012'
    for filename in ('state.h5','statistics.h5'):
        compare_fields(full/ending/filename,tail/ending/filename)
    compare_fields(full/ending/'state.h5',off/ending/'state.h5')
    for name in ('time_space','probe_spectrum'):
        reference=history(full,name)
        paths=[p for case in (prefix,tail) for p in (case/'outdat/new/temporal_products'/name).glob('*/segment*/block.*.csv')]
        restored=next(iter(archive.read_histories(paths).values()))
        for field in reference.dtype.names: np.testing.assert_array_equal(reference[field],restored[field])


def test_raw_probe_archive_remains_opt_in(tmp_path):
    case=launch(tmp_path/'cpu','cpu',(1,1,1),records=records(raw=False),insitu_config=config(),
                checkpoint=True,unrestricted=True)
    assert not list((case/'outdat/new/temporal_products/probe_spectrum').rglob('block.*.csv'))
    assert list((case/'outdat/new/temporal_products/probe_spectrum').rglob('spectrum.*.csv'))


@pytest.mark.parametrize('keep_shared',[False,True])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_temporal_sampling_gap_and_live_shared_history(tmp_path,backend,keep_shared,catalog_replay_library):
    line,probe=records()
    probe['temporal']['segment_length']=4;probe['schedule']['interval']=1
    keeper=deepcopy(probe);keeper['id']='sampling-keeper';keeper['schedule']['interval']=1000
    keeper['temporal']=dict(segment_length=4)
    keepers=[line,*([keeper] if keep_shared else [])]
    case=launch(tmp_path/backend,backend,(1,1,2),records=[*keepers,probe],insitu_config=config(),
        checkpoint=True,unrestricted=True,catalog_replay=(catalog_replay_library,{6:keepers,8:[*keepers,probe]}))
    reference=history(case,'time_space');reference=reference[reference['node']==33]
    values=np.column_stack([reference['velocity_x'],reference['pressure']])
    pieces=[values] if keep_shared else [values[:5],values[7:]]
    counts=[1+(len(piece)-4)//2 for piece in pieces]
    estimates=[];window=.5*(1-np.cos(2*np.pi*np.arange(4)/4))
    for piece in pieces:
        frequency,power=welch(piece,fs=1/DT,window=window,nperseg=4,noverlap=2,
            detrend='constant',scaling='density',axis=0)
        estimates.append(power)
    expected=np.column_stack([frequency,sum(p*n for p,n in zip(estimates,counts))/sum(counts)])
    path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,actual=spectrum_csv(path)
    metadata_objects(text)
    assert f'valid_segments={sum(counts)};' in text,text
    compare(actual,expected)
    raw=history(case,'probe_spectrum')
    expected_steps=np.arange(13) if keep_shared else np.r_[np.arange(5),np.arange(7,13)]
    np.testing.assert_array_equal(raw['step'],expected_steps)
    assert raw['step'][raw['gap_before']!=0].tolist()==([] if keep_shared else [7])


@pytest.mark.parametrize('keep_shared',[False,True])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_dynamic_history_exact_restart_while_product_is_absent(tmp_path,backend,keep_shared,catalog_replay_library):
    line,probe=records();probe['temporal']['segment_length']=4;probe['schedule']['interval']=1
    keeper=deepcopy(probe);keeper['id']='sampling-keeper';keeper['temporal']=dict(segment_length=4)
    keepers=[line,*([keeper] if keep_shared else [])]
    all_records=[*keepers,probe]
    common=dict(insitu_config=config(),checkpoint=True,unrestricted=True)
    full=launch(tmp_path/'full',backend,(1,1,2),records=all_records,
        catalog_replay=(catalog_replay_library,{6:keepers,8:all_records}),**common)
    prefix=launch(tmp_path/'prefix',backend,(1,1,2),steps=5,records=all_records,
        catalog_replay=(catalog_replay_library,{6:keepers}),**common)
    # Restore rebases the watcher twice; polls 3/4 then observe steps 6/7.
    tail=launch(tmp_path/'tail',backend,(1,1,2),restore=prefix/'outdat/new/checkpoints/step000000000005',
        records=keepers,catalog_replay=(catalog_replay_library,{4:all_records}),**common)
    ending='outdat/new/checkpoints/step000000000012'
    for filename in ('state.h5','statistics.h5'):
        compare_fields(full/ending/filename,tail/ending/filename)
    for name in ('time_space','probe_spectrum'):
        expected=history(full,name)
        paths=[path for case in (prefix,tail) for path in
               (case/'outdat/new/temporal_products'/name).glob('*/segment*/block.*.csv')]
        actual=next(iter(archive.read_histories(paths).values()))
        for field in expected.dtype.names: np.testing.assert_array_equal(actual[field],expected[field])
    path=next((full/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    other=next((tail/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    assert path.read_bytes()==other.read_bytes()


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_shared_spectrum_survives_a_long_archive_gap(tmp_path,backend,catalog_replay_library):
    line,probe=records();probe['temporal']['segment_length']=4;probe['schedule']['interval']=1
    keeper=deepcopy(probe);keeper['id']='sampling-keeper';keeper['temporal']=dict(segment_length=4)
    keepers=[line,keeper]
    case=launch(tmp_path/backend,backend,(2,1,1),records=[*keepers,probe],insitu_config=config(),
        unrestricted=True,catalog_replay=(catalog_replay_library,{4:keepers,11:[*keepers,probe]}))
    raw=history(case,'probe_spectrum')
    np.testing.assert_array_equal(raw['step'],np.r_[np.arange(3),np.arange(7,13)])
    assert raw['step'][raw['gap_before']!=0].tolist()==[7]
    line=history(case,'time_space');line=line[line['node']==33]
    source=np.column_stack([line['velocity_x'],line['pressure']])
    window=.5*(1-np.cos(2*np.pi*np.arange(4)/4))
    f,p=welch(source,fs=1/DT,window=window,nperseg=4,noverlap=2,detrend='constant',scaling='density',axis=0)
    path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,data=spectrum_csv(path)
    assert 'valid_segments=5;' in text and 'sampled_endpoints=13;' in text
    compare(data,np.column_stack([f,p]))


@pytest.mark.parametrize('keep_shared',[False,True])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_late_probe_reuses_only_actual_matching_history(tmp_path,backend,keep_shared,catalog_replay_library):
    line,probe=records();probe['temporal']['segment_length']=4;probe['schedule']['interval']=1
    keeper=deepcopy(probe);keeper['id']='sampling-keeper';keeper['temporal']=dict(segment_length=4)
    initial=[line,*([keeper] if keep_shared else [])]
    case=launch(tmp_path/backend,backend,(1,2,1),records=initial,insitu_config=config(),unrestricted=True,
        catalog_replay=(catalog_replay_library,{8:[*initial,probe]}))
    raw=history(case,'probe_spectrum');np.testing.assert_array_equal(raw['step'],np.arange(7,13))
    line=history(case,'time_space');line=line[line['node']==33]
    source=np.column_stack([line['velocity_x'],line['pressure']])
    if not keep_shared: source=source[7:]
    window=.5*(1-np.cos(2*np.pi*np.arange(4)/4))
    f,p=welch(source,fs=1/DT,window=window,nperseg=4,noverlap=2,detrend='constant',scaling='density',axis=0)
    path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,data=spectrum_csv(path)
    assert f'first_sample_step={0 if keep_shared else 7};' in text
    assert f'valid_segments={5 if keep_shared else 2};' in text
    compare(data,np.column_stack([f,p]))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_equal_stride_different_phase_does_not_reuse_spectrum(tmp_path,backend,catalog_replay_library):
    line,probe=records();probe['temporal'].update(segment_length=2,sample_steps=3)
    probe['schedule']['interval']=1
    keeper=deepcopy(probe);keeper['id']='sampling-keeper';keeper['temporal']=dict(segment_length=2,sample_steps=3)
    initial=[line,keeper]
    case=launch(tmp_path/backend,backend,(1,1,2),records=initial,insitu_config=config(),unrestricted=True,
        catalog_replay=(catalog_replay_library,{8:[*initial,probe]}))
    raw=history(case,'probe_spectrum');np.testing.assert_array_equal(raw['step'],[7,10])
    path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,data=spectrum_csv(path)
    assert 'first_sample_step=7;' in text and 'valid_segments=1;' in text
    f,p=welch(np.column_stack([raw['velocity_x'],raw['pressure']]),fs=1/(3*DT),window=np.array([0.,1.]),
        nperseg=2,noverlap=1,detrend='constant',scaling='density',axis=0)
    compare(data,np.column_stack([f,p]))


@pytest.mark.parametrize('change',['storage','spectrum'])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_temporal_live_storage_and_sampling_changes(tmp_path,backend,change,catalog_replay_library):
    line,probe=records();probe['schedule']['interval']=1
    changed=deepcopy(probe)
    if change=='storage':
        changed['temporal']['block_samples']=6
    else:
        changed['temporal']['segment_length']=4
    case=launch(tmp_path/backend,backend,(2,1,1),records=[line,probe],insitu_config=config(),
        unrestricted=True,catalog_replay=(catalog_replay_library,{6:[line,changed]}))
    paths=list((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/block.*.csv'))
    groups=archive.read_histories(paths)
    assert len(groups)==(1 if change=='storage' else 2)
    ordered=sorted(groups.values(),key=lambda rows:rows['step'][0])
    combined=np.concatenate(ordered)
    np.testing.assert_array_equal(combined['step'],np.arange(13))
    assert not np.any(combined['gap_before'])
    expected_rows=combined if change=='storage' else ordered[-1]
    if change=='spectrum': np.testing.assert_array_equal(expected_rows['step'],np.arange(5,13))
    n=8 if change=='storage' else 4
    window=.5*(1-np.cos(2*np.pi*np.arange(n)/n))
    f,p=welch(np.column_stack([expected_rows['velocity_x'],expected_rows['pressure']]),
        fs=1/DT,window=window,nperseg=n,noverlap=n//2,detrend='constant',scaling='density',axis=0)
    path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,data=spectrum_csv(path)
    assert f'valid_segments={2 if change=="storage" else 3};' in text
    assert f'first_sample_step={0 if change=="storage" else 5};' in text
    compare(data,np.column_stack([f,p]))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_temporal_publication_failure_does_not_retry_or_change_flow(tmp_path,backend,catalog_replay_library):
    library=os.environ.get('ASTR_TEMPORAL_FAULT_LIBRARY')
    if not library: pytest.skip('Select the root-built temporal publication fault library')
    normal=records();faulted=deepcopy(normal[0]);faulted['id']='faulted'
    common=dict(insitu_config=config(),checkpoint=True,unrestricted=True)
    reference=launch(tmp_path/'reference',backend,(2,1,1),records=normal,**common)
    failed=launch(tmp_path/'failed',backend,(2,1,1),records=[*normal,faulted],
        catalog_replay=(library,{}),**common)
    text=(failed/'run.log').read_text()
    assert text.count('ASTR_TEST_TEMPORAL_LINK_FAILURE ')==1,text[-7000:]
    status={row['id']:row for row in json.loads((failed/'outdat/new/products.status.json').read_text())['products']}
    assert status['faulted']['state']=='resource-disabled',status
    assert status['time_space']['state']==status['probe_spectrum']['state']=='active',status
    folder=failed/'outdat/new/temporal_products/faulted'
    assert not list(folder.rglob('block.*.csv'))
    assert len(list(folder.rglob('*.writing')))==1
    ending='outdat/new/checkpoints/step000000000012'
    compare_fields(reference/ending/'state.h5',failed/ending/'state.h5')
    for name in ('time_space','probe_spectrum'):
        a,b=history(reference,name),history(failed,name)
        for field in a.dtype.names: np.testing.assert_array_equal(a[field],b[field])
    repaired=deepcopy(faulted);repaired['enable_request']=2
    recovered=launch(tmp_path/'recovered',backend,(2,1,1),records=[*normal,faulted],
        restore=failed/'outdat/new/checkpoints/step000000000010',
        catalog_replay=(catalog_replay_library,{3:[*normal,repaired]}),**common)
    status={row['id']:row for row in json.loads((recovered/'outdat/new/products.status.json').read_text())['products']}
    assert status['faulted']['state']=='active' and status['faulted']['consumed_enable_request']==2,status
    assert 'ASTR_TEST_TEMPORAL_LINK_FAILURE' not in (recovered/'run.log').read_text()
    compare_fields(reference/ending/'state.h5',recovered/ending/'state.h5')
    raw=history(recovered,'faulted')
    np.testing.assert_array_equal(np.unique(raw['step']),np.arange(8,13))


def automatic_run(steps):
    return f"""&run_control
 max_steps={steps},reload_steps=1000
/
&timestep
 mode='auto',cfl_target=.5,growth_factor=1.1,initial_dt=1.e-4,max_dt=1.e-3
/
&logging
 list_steps=1
/
&repair
 crashfix=f
/
&statistics
 enabled=f
/
"""


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_actual_variable_timestamps_and_exact_history_restart(tmp_path,backend):
    args=output_arguments(tmp_path);args.executable=Path(EXE).resolve();args.unrestricted=True
    items=records()
    items[0]['sampling'].update(index=4,index_ranges=dict(i=[0,32],k=[4,4]))
    items[1]['sampling']['indices']=[8,4,4]
    common=dict(grid='32,32,32',insitu_config=config(),product_catalog_records=items,checkpoint_interval=5)
    full,_=run_case(args,ROOT,backend,2,'automatic-full',12,run_config_text=automatic_run(12),**common)
    prefix,_=run_case(args,ROOT,backend,2,'automatic-prefix',5,run_config_text=automatic_run(5),**common)
    tail,_=run_case(args,ROOT,backend,2,'automatic-tail',12,run_config_text=automatic_run(12),
        restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    for name in ('time_space','probe_spectrum'):
        expected=history(full,name)
        paths=[path for case in (prefix,tail) for path in
               (case/'outdat/new/temporal_products'/name).glob('*/segment*/block.*.csv')]
        actual=next(iter(archive.read_histories(paths).values()))
        for field in expected.dtype.names: np.testing.assert_array_equal(actual[field],expected[field])
    probe=history(full,'probe_spectrum');delta=np.diff(probe['time'])
    assert np.ptp(delta)>1.e-5 and np.all(delta>0)
    for case in (full,tail):
        files=list((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.*.csv'))
        assert files
        for path in files:
            text,values=spectrum_csv(path)
            assert '# available=F;' in text and 'valid_segments=0;' in text and len(values)==0
    ending='outdat/new/checkpoints/step000000000012'
    for filename in ('state.h5','statistics.h5'):
        compare_fields(full/ending/filename,tail/ending/filename)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_temporal_restart_without_saved_statistics_starts_actual_history(tmp_path,backend):
    args=output_arguments(tmp_path);args.executable=Path(EXE).resolve();args.unrestricted=True
    items=records()
    items[0]['sampling'].update(index=4,index_ranges=dict(i=[0,32],k=[4,4]))
    items[1]['sampling']['indices']=[8,4,4]
    text=re.sub(r'statistics_window=[^,\n]+,[^,\n]+','statistics_window=0,1',config())
    common=dict(grid='32,32,32',insitu_config=text,product_catalog_records=items,
        checkpoint_interval=5,checkpoint_statistics=False)
    def fixed_run(steps):
        return f"""&run_control
 max_steps={steps},reload_steps=1000
/
&timestep
 mode='fixed',fixed_dt={DT}
/
&logging
 list_steps=1
/
&repair
 crashfix=f
/
&statistics
 enabled=f
/
"""
    prefix,_=run_case(args,ROOT,backend,2,'no-statistics-prefix',5,run_config_text=fixed_run(5),**common)
    batch=prefix/'outdat/new/checkpoints/step000000000005'
    assert not (batch/'statistics.h5').exists()
    tail,_=run_case(args,ROOT,backend,2,'no-statistics-tail',12,restore=batch,
        run_config_text=fixed_run(12),**common)
    for name in ('time_space','probe_spectrum'):
        data=history(tail,name)
        np.testing.assert_array_equal(np.unique(data['step']),np.arange(5,13))
        assert not np.any(data['gap_before'])
        paths=list((tail/'outdat/new/temporal_products'/name).glob('*/segment*/block.*.csv'))
        assert all(int(re.search(r'block.g(\d+)',p.name)[1])>=6 for p in paths)
    probe=history(tail,'probe_spectrum')
    path=next((tail/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
    text,data=spectrum_csv(path)
    assert 'valid_segments=1;' in text and 'first_sample_step=5;' in text
    window=.5*(1-np.cos(2*np.pi*np.arange(8)/8))
    f,p=welch(np.column_stack([probe['velocity_x'],probe['pressure']]),fs=1/DT,
        window=window,nperseg=8,noverlap=4,detrend='constant',scaling='density',axis=0)
    compare(data,np.column_stack([f,p]))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_actual_temporal_mean_fluctuation_coverage_and_continuation(tmp_path,backend):
    raw=records()[0];raw['fields']=['density','velocity_x','pressure','temperature']
    items=[raw]
    for source in ('reynolds','favre'):
        for fluctuation in (False,True):
            row=deepcopy(raw);row.update(id=source+('-fluctuation' if fluctuation else '-mean'),
                source=source,statistics_source='selected-time',fluctuation=fluctuation)
            row['fields']=['velocity_x'] if source=='favre' else raw['fields'][1:]
            items.append(row)
    window=(2.5*DT,8.5*DT)
    text=re.sub(r'statistics_window=[^,\n]+,[^,\n]+',f'statistics_window={window[0]},{window[1]}',config())
    common=dict(records=items,insitu_config=text,checkpoint=True,unrestricted=True)
    full=launch(tmp_path/'full',backend,(1,1,2),**common)
    line=history(full,'time_space')
    fields=raw['fields'];primitive={f:line[f].reshape(13,65) for f in fields}
    for row in items[1:]:
        data=history(full,row['id'])
        np.testing.assert_array_equal(np.unique(data['step']),np.arange(3,13))
        for step in range(3,13):
            weights=np.zeros(13)
            for right in range(1,step+1):
                t0,t1=(right-1)*DT,right*DT
                a,b=max(t0,window[0]),min(t1,window[1])
                if b<=a:continue
                fraction=.5*((a-t0)/DT)+.5*((b-t0)/DT)
                weights[right-1]+=(b-a)*(1-fraction);weights[right]+=(b-a)*fraction
            density=primitive['density'];mass=weights@density
            sample=data[data['step']==step]
            for f in row['fields']:
                mean=(weights@primitive[f]/weights.sum() if row['source']=='reynolds'
                    else weights@(density*primitive[f])/mass)
                expected=primitive[f][step]-mean if row['fluctuation'] else mean
                compare(sample[f],expected)
            compare(sample['mean_window_start'],np.full(65,window[0]))
            compare(sample['mean_window_end'],np.full(65,min(step*DT,window[1])))
            compare(sample['mean_measure'],np.full(65,weights.sum()))
    prefix=launch(tmp_path/'prefix',backend,(1,1,2),steps=5,**common)
    tail=launch(tmp_path/'tail',backend,(1,1,2),restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    ending='outdat/new/checkpoints/step000000000012'
    for filename in ('state.h5','statistics.h5'):compare_fields(full/ending/filename,tail/ending/filename)
    for row in items:
        paths=[path for case in (prefix,tail) for path in
            (case/'outdat/new/temporal_products'/row['id']).glob('*/segment*/block.*.csv')]
        restored=next(iter(archive.read_histories(paths).values()))
        expected=history(full,row['id'])
        for field in expected.dtype.names:np.testing.assert_array_equal(restored[field],expected[field])


def test_temporal_resource_cost_by_product_and_trigger_step(tmp_path):
    cases={};summary=dict(scope='NP2 x, twelve GRCBC complete steps; initial sampling/rendering is reported separately; no field/checkpoint HDF5; profiling logs enabled; not a production benchmark',cases={})
    for name,items in [('off',[]),('data',records()),('images',image_records())]:
        # Both candidate window policies include every sample in this cost pair.
        text=re.sub(r'statistics_window=[^,\n]+,[^,\n]+','statistics_window=0,1',render_config('standard-device'))
        baseline=json.loads((cases['off']/'resources.sampled.json').read_text()) if name!='off' else None
        case=launch(tmp_path/name,'gpu',(2,1,1),records=items,insitu_config=text,monitor=True,
                    resource_baseline=baseline,unrestricted=True)
        cases[name]=case
        assert not list((case/'outdat').rglob('*.h5'))
        completed={};timings=[];storage={};transfers={}
        for line in (case/'run.log').read_text().splitlines():
            match=re.fullmatch(r'ASTR_INSITU_STAGE_TIMING completed_step_inclusive (\d+) (\d+)\s+(\S+)',line)
            if match:completed[tuple(map(int,match.group(1,2)))]=float(match[3])
            prefix='ASTR_TEMPORAL_TIMING '
            if line.startswith(prefix):
                row=json.loads(line[len(prefix):]);assert np.isfinite(row['seconds']) and row['seconds']>=0
                timings.append(row)
            match=re.fullmatch(r'ASTR_TEMPORAL_STORAGE id=(\S+) step=(\d+) retained_host_array_bytes=(\d+) retained_device_array_bytes=(\d+)',line)
            if match:storage.setdefault(match[1],[]).append(dict(step=int(match[2]),host=int(match[3]),device=int(match[4])))
            match=re.match(r'ASTR_TEMPORAL_SAMPLE rank=(\d+) id=(\S+) step=(\d+) one_dimensional_d2h_bytes=(\d+) private_exchange_bytes=(\d+)',line)
            if match:
                item=transfers.setdefault(match[2],dict(one_dimensional_d2h_bytes=0,provider_private_exchange_bytes=0))
                item['one_dimensional_d2h_bytes']+=int(match[4]);item['provider_private_exchange_bytes']+=int(match[5])
        assert set(completed)=={(step,rank) for step in range(1,13) for rank in (0,1)}
        resource=json.loads((case/'resources.sampled.json').read_text())
        assert resource['limits']==dict(device_extra_bytes=None,host_extra_bytes=None,device_reserve_bytes=0,timeout_seconds=None)
        record=dict(complete_step_max_rank_seconds=[max(completed[step,rank] for rank in (0,1)) for step in range(1,13)],
            complete_window_max_rank_sum_seconds=max(sum(completed[step,rank] for step in range(1,13)) for rank in (0,1)),
            initial_product_events=[row for row in timings if row['step']==0],product_events=timings,
            retained_temporal_arrays=storage,transfers=transfers,resources=resource)
        record['phase_interpretation']='inclusive stages; sample includes packet/collectives/shared fanout, publication includes consumer and file publication, archive is rank-zero file work; do not sum nested stages as disjoint time'
        if name!='off':
            assert set(storage)=={'time_space','probe_spectrum'}
            for product,rows in storage.items():
                assert len({row['host'] for row in rows})==1 and all(row['device']==0 for row in rows)
                packet=[row for row in timings if row['product']==product and row['stage']=='sample']
                assert {(row['step'],row['rank']) for row in packet}=={(step,rank) for step in range(13) for rank in (0,1)}
            width=1 if name=='images' else 2
            assert transfers['time_space']['one_dimensional_d2h_bytes']==65*width*8*13
            assert transfers['probe_spectrum']['one_dimensional_d2h_bytes']==width*8*13
            record['host_collective_buffer_accounting']=dict(
                scope='calculated submitted buffer bytes, not measured MPI wire traffic; includes zero-filled nonowner entries',
                per_rank_ownership_allreduce_bytes=13*(65+1)*4,
                per_rank_line_reduce_bytes=13*(65+1)*(width+3)*8)
        summary['cases'][name]=record
    (tmp_path/'cost.json').write_text(json.dumps(summary,indent=2)+'\n')


def test_temporal_diagnostics_off_preserves_fields_and_histories(tmp_path):
    common=dict(records=records(),insitu_config=config(),checkpoint=True,unrestricted=True)
    profiled=launch(tmp_path/'profiled','gpu',(2,1,1),timing=True,**common)
    quiet=launch(tmp_path/'quiet','gpu',(2,1,1),timing=False,**common)
    text=(quiet/'run.log').read_text()
    assert 'ASTR_TEMPORAL_TIMING ' not in text and 'ASTR_TEMPORAL_STORAGE ' not in text and 'ASTR_TEMPORAL_SAMPLE ' not in text
    ending='outdat/new/checkpoints/step000000000012'
    for filename in ('state.h5','statistics.h5'):
        compare_fields(profiled/ending/filename,quiet/ending/filename)
    for name in ('time_space','probe_spectrum'):
        a,b=history(profiled,name),history(quiet,name)
        for field in a.dtype.names: np.testing.assert_array_equal(a[field],b[field])


def test_temporal_real_field_memcheck(tmp_path):
    declared=config().replace('render=f,',"render=f,derivative_backend='gpu',processing_backend='device',postprocess_transport='device-aware',")
    case=launch(tmp_path/'gpu','gpu',(2,1,2),records=records(),insitu_config=declared,
        memcheck=True,unrestricted=True)
    logs=list(case.glob('memcheck.*.log'))
    assert len(logs)==4
    assert all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)
    assert not list((case/'outdat').rglob('*.h5'))


def image_records():
    line=definition('time_space',True)
    line['sampling'].update(index=24,index_ranges=dict(i=[0,64],k=[8,8]))
    line['temporal']['block_samples']=4;line['schedule']['interval']=3
    line['display']['color_range']=[.17,.23]
    probe=definition('probe_spectrum',True);probe['sampling']['indices']=[32,24,8]
    probe['temporal']['segment_length']=8;probe['schedule']['interval']=3
    probe['display']['y_range']=[0,1e-5]
    return [line,probe]


def render_config(pipeline):
    text=resident_configuration(pipeline,profile='catalog_stats').replace("output_directory='outdat/render'","output_directory='outdat'")
    return re.sub(r'(host_budget_bytes|device_budget_bytes)\s*=\s*\d+',r'\1=-1',text).replace('device_reserve_bytes=1073741824','device_reserve_bytes=0')


@pytest.mark.parametrize('pipeline,topology',[(p,(2,1,1)) for p in ('standard-device','direct-device')])
def test_actual_temporal_images_without_a_volume_channel(tmp_path,pipeline,topology):
    from PIL import Image
    case=launch(tmp_path/'gpu','gpu',topology,records=image_records(),insitu_config=render_config(pipeline),
                unrestricted=True)
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    products={p.parent.parent.name:p for p in (case/'outdat/new/render_products').glob('*/*/segment*')}
    assert set(products)=={'time_space','probe_spectrum'}
    for name,steps in [('time_space',[0,3,6,9,12]),('probe_spectrum',[9,12])]:
        paths=sorted(products[name].glob('frame.step*.jpeg'))
        assert [int(p.stem.split('step')[1]) for p in paths]==steps,(name,paths)
        assert not list(products[name].glob('*.eps'))
        for path in paths:
            with Image.open(path) as picture:
                assert picture.size==(800,600) and np.ptp(np.asarray(picture.convert('RGB')))>100
        for path in products[name].glob('frame.*.rank00000000.json'):
            receipt=json.loads(path.read_text())
            assert receipt['chart_backend']=='existing-host-matplotlib'
            assert receipt['nodes']<=65 and receipt['rows']<=4
            assert receipt['product']['lighting_enabled'] is False
    assert not list((case/'outdat').rglob('*.h5'))


def test_spanwise_time_space_chart_uses_actual_z_axis(tmp_path):
    from PIL import Image
    line=image_records()[0];line['id']='spanwise-chart'
    line['sampling'].update(index=1,index_ranges=dict(i=[32,32],k=[0,16]))
    case=launch(tmp_path/'gpu','gpu',(1,1,2),steps=6,records=[line],
        insitu_config=render_config('standard-device'),unrestricted=True)
    directory=next((case/'outdat/new/render_products/spanwise-chart').glob('*/segment*'))
    pictures=sorted(directory.glob('frame.step*.jpeg'))
    assert [int(p.stem.split('step')[1]) for p in pictures]==[0,3,6]
    for path in pictures:
        with Image.open(path) as image:assert image.size==(800,600)
    history_rows=next(iter(archive.read_histories(directory.glob('block.*.csv')).values()))
    assert len(history_rows)==7*17
    last=history_rows[history_rows['step']==6]
    assert np.all(np.diff(last['z'])>0) and np.ptp(last['x'])==np.ptp(last['y'])==0
    for path in directory.glob('frame.*.rank00000000.json'):
        receipt=json.loads(path.read_text())
        assert receipt['nodes']==17 and receipt['rows']<=4
        assert receipt['product']['lighting_enabled'] is False
    assert not list((case/'outdat').rglob('*.h5'))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_temporal_images_exact_continuation(tmp_path,pipeline):
    eligible=re.sub(r'statistics_window=[^,\n]+,[^,\n]+','statistics_window=0,1',render_config(pipeline))
    common=dict(records=image_records(),insitu_config=eligible,checkpoint=True,unrestricted=True)
    full=launch(tmp_path/'full','gpu',(2,1,1),**common)
    prefix=launch(tmp_path/'prefix','gpu',(2,1,1),steps=5,**common)
    tail=launch(tmp_path/'tail','gpu',(2,1,1),restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    ending='outdat/new/checkpoints/step000000000012'
    for filename in ('state.h5','statistics.h5'):
        compare_fields(full/ending/filename,tail/ending/filename)
    reference={p.parent.parent.name:p for p in (full/'outdat/new/render_products').glob('*/*/segment*')}
    for segment in (tail/'outdat/new/render_products').glob('*/*/segment*'):
        name=segment.parent.parent.name
        steps=[6,9,12] if name=='time_space' else [9,12]
        files=sorted(segment.glob('frame.step*.jpeg'))
        assert [int(p.stem.split('step')[1]) for p in files]==steps
        for path in files:
            assert path.read_bytes()==(reference[name]/path.name).read_bytes()


def analyse_matrix(root):
    root=Path(root);report={};physical={}
    for parent in sorted(root.glob('test_actual_temporal_line_prob*')):
        if parent.is_symlink(): continue
        rows={backend:{name:history(parent/backend,name) for name in ('time_space','probe_spectrum')}
              for backend in ('cpu','gpu')}
        errors={}
        def record(name,a,b):
            a,b=np.asarray(a),np.asarray(b)
            scaled,absolute=compare(a,b)
            errors[name]=dict(maximum_scaled_difference=scaled,maximum_absolute_difference=absolute)
        for name in ('time_space','probe_spectrum'):
            for field in rows['cpu'][name].dtype.names:
                record(name+'.'+field,rows['cpu'][name][field],rows['gpu'][name][field])
        for backend in ('cpu','gpu'):
            case=parent/backend;line=rows[backend]['time_space'];probe=rows[backend]['probe_spectrum']
            with h5py.File(case/'outdat/new/checkpoints/step000000000012/state.h5') as state:
                full=[state[f'q{c:04d}'][...] for c in range(1,6)]
                assert all(np.isfinite(field).all() for field in full)
                density=full[0];assert density.min()>0
                internal=full[4]-.5*(full[1]**2+full[2]**2+full[3]**2)/density
                assert np.isfinite(internal).all() and internal.min()>0
                temperature=2.24*internal/density
                assert np.isfinite(temperature).all() and temperature.min()>0
                physical[parent.name+'.'+backend]=dict(minimum_density=float(density.min()),
                    minimum_internal_energy=float(internal.min()),minimum_temperature=float(temperature.min()))
                rho,mx,my,mz,energy=[field[8,24,:] for field in full]
            last=line[line['step']==12]
            record(backend+'.line_velocity_oracle',last['velocity_x'],mx/rho)
            record(backend+'.line_pressure_oracle',last['pressure'],.4*(energy-.5*(mx**2+my**2+mz**2)/rho))
            with h5py.File(case/'datin/grid.flatplate.h5') as mesh:
                for field in ('x','y','z'):record(backend+'.coordinate_'+field,last[field],mesh[field][8,24,:])
            path=next((case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.step000000000012.csv'))
            _,spectrum=spectrum_csv(path)
            source=np.column_stack([probe['velocity_x'],probe['pressure']])
            window=.5*(1-np.cos(2*np.pi*np.arange(8)/8))
            f,p=welch(source,fs=1/DT,window=window,nperseg=8,noverlap=4,detrend='constant',scaling='density',axis=0)
            record(backend+'.welch_reference',spectrum,np.column_stack([f,p]))
        report[parent.name]=errors
    assert len(report)==7,report.keys()
    output=root/'numeric_summary.json';output.write_text(json.dumps(report,indent=2)+'\n')
    (root/'physical_state_summary.json').write_text(json.dumps(physical,indent=2)+'\n')
    return output,dict(maximum_scaled_difference=max(v['maximum_scaled_difference'] for r in report.values() for v in r.values()),
                       maximum_absolute_difference=max(v['maximum_absolute_difference'] for r in report.values() for v in r.values()))
