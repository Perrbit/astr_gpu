"""IS6 physical-volume statistics, independent endpoints and exact continuation."""
import json
import re

import h5py
import numpy as np
import pytest

from generate_curvilinear_tgv_grid import mapped_grid
from test_insitu_geometry import integrate_numpy
from test_insitu_curve_derivatives import arguments,ROOT,FINAL
from test_insitu_channel_statistics import read_statistics
from run_output_restart_validation import run_case,compare_fields


def configuration():
    return """&insitu_run
 enabled=t,statistics=t,render=f,statistics_window=0.003,0.004,
 output_directory='outdat/render',host_budget_bytes=4294967296,
 device_budget_bytes=2147483648,device_reserve_bytes=1073741824
/
"""


def physical_weights(mapping='periodic'):
    volume,_=integrate_numpy(np.stack(mapped_grid(32,32,32,.15,mapping)[:3]))
    weights=np.zeros((33,33,33))
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                weights[a:a+32,b:b+32,c:c+32]+=volume/8
    for axis in ((0,1,2) if mapping=='periodic' else (0,2)):
        lower=[slice(None)]*3; lower[axis]=0
        upper=[slice(None)]*3; upper[axis]=32
        weights[tuple(lower)]+=weights[tuple(upper)]
        weights[tuple(upper)]=0
    ny=32 if mapping=='periodic' else 33
    assert np.min(weights[:32,:ny,:32])>0
    assert abs(weights.sum()-(2*np.pi)**3)<=2e-10
    return weights[:32,:ny,:32]


def check_independent(case,ranks,mapping='periodic'):
    weights=physical_weights(mapping)
    ny=weights.shape[1]
    velocities=[]; densities=[]
    for step in (3,4):
        with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
            density=state['q0001'][:32,:ny,:32].transpose(2,1,0)
            velocities.append(np.stack([state[f'q{c:04d}'][:32,:ny,:32].transpose(2,1,0)/density
                                         for c in (2,3,4)],axis=-1))
            densities.append(density)
    a,b=velocities; ra,rb=densities
    mean=.5*(a+b)
    covariance=.25*(b-a)[..., :,None]*(b-a)[...,None,:]
    favre=(ra[...,None]*a+rb[...,None]*b)/(ra+rb)[...,None]
    favre_cov=(ra*rb/(ra+rb)**2)[...,None,None]*(b-a)[..., :,None]*(b-a)[...,None,:]
    volume=weights.sum()
    spatial_variance=np.einsum('ijk,ijkc->c',weights,np.diagonal(covariance,axis1=-2,axis2=-1))/volume
    regional_delta=np.einsum('ijk,ijkc->c',weights,b-a)/volume
    for rank in range(ranks):
        header,meta,output=read_statistics(case/f'outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin')
        assert abs(meta[3]-volume)<=2e-10
        np.testing.assert_allclose(meta[4:7],np.sqrt(spatial_variance),rtol=0,atol=2e-10)
        np.testing.assert_allclose(meta[7:10],.5*abs(regional_delta),rtol=0,atol=2e-10)
        # Periodic endpoint nodes use the owner's mass/velocity, not duplicate physical coordinates.
        origin=header[6:9]; extent=header[3:6]
        indices=np.meshgrid(*[(np.arange(o,o+n)%size) for o,n,size in zip(origin,extent,(32,ny,32))],indexing='ij')
        selection=tuple(indices)
        np.testing.assert_allclose(output[...,0],.001,rtol=0,atol=2e-15)
        np.testing.assert_allclose(output[...,1],(.5*(ra+rb))[selection],rtol=0,atol=2e-10)
        np.testing.assert_allclose(output[...,2:5],mean[selection],rtol=0,atol=2e-10)
        np.testing.assert_allclose(output[...,5:8],favre[selection],rtol=0,atol=2e-10)
        for offset,expected in ((14,covariance),(23,favre_cov),(32,.5*(ra+rb)[...,None,None]*favre_cov)):
            np.testing.assert_allclose(output[...,offset:offset+9].reshape((*output.shape[:3],3,3),order='F'),
                                       expected[selection],rtol=0,atol=2e-10)
    return volume


@pytest.mark.parametrize('ranks,axis',[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def test_curve_statistics_physical_measure_and_restart(tmp_path,ranks,axis,record_property):
    cases={}; maxima=[]
    for backend in ('cpu','gpu'):
        args=arguments(tmp_path,backend,axis)
        kwargs=dict(grid='32,32,32',tgv_mapping='periodic',checkpoint_interval=1)
        case,_=run_case(args,ROOT,backend,ranks,'statistics',4,insitu_config=configuration(),**kwargs)
        check_independent(case,ranks)
        cases[backend]=case
        restarted,_=run_case(args,ROOT,backend,ranks,'restart',4,insitu_config=configuration(),
            restore=case/'outdat/new/checkpoints/step000000000003',**kwargs)
        for name in ('state.h5','statistics.h5'):
            compare_fields(case/FINAL/name,restarted/FINAL/name)
        for name in ('control.bin','insitu_control.bin'):
            assert (case/FINAL/name).read_bytes()==(restarted/FINAL/name).read_bytes()
        for rank in range(ranks):
            name=f'outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin'
            assert (case/name).read_bytes()==(restarted/name).read_bytes()
        off,_=run_case(args,ROOT,backend,ranks,'off',4,
            insitu_config='&insitu_run\n enabled=f\n/\n',**kwargs)
        compare_fields(case/FINAL/'state.h5',off/FINAL/'state.h5')
    for rank in range(ranks):
        name=f'outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin'
        ch,cm,cpu=read_statistics(cases['cpu']/name)
        gh,gm,gpu=read_statistics(cases['gpu']/name)
        np.testing.assert_array_equal(ch,gh)
        np.testing.assert_allclose(cpu,gpu,rtol=0,atol=2e-10)
        np.testing.assert_allclose(cm,gm,rtol=0,atol=2e-10)
        maxima.append(float(np.max(abs(cpu-gpu))))
    for case in cases.values():
        volumes=[float(v) for v in re.findall(r'ASTR_INSITU_CURVE_VOLUME=\s*(\S+)',(case/'run.log').read_text())]
        assert len(volumes)==1 and abs(volumes[0]-(2*np.pi)**3)<=2e-10
    record_property('statistics_maxabs',max(maxima))


@pytest.mark.parametrize('axis',('x','y','z'))
def test_curve_statistic_device_weights_memory_safety(tmp_path,axis):
    run_case(arguments(tmp_path,'gpu',axis),ROOT,'gpu',2,'memcheck_weights',4,
        grid='32,32,32',tgv_mapping='periodic',checkpoint_interval=1,
        insitu_config=configuration(),memcheck=True)


def compare_partition_evidence(matrix):
    """Read immutable completed matrix artifacts; do not rerun the flow solver."""
    report={}
    for family,pattern in (('periodic','test_curve_statistics_physica*'),
                           ('walls','test_curve_wall_fields_stats_a*')):
        for backend in ('cpu','gpu'):
            cases=sorted({p.resolve() for d in matrix.glob(pattern)
                          for p in d.glob(backend+'_np*_statistics')})
            assert len(cases)==4,(family,backend,cases)
            reference=next(p for p in cases if '_np1_' in p.name)
            meta=read_statistics(reference/'outdat/render/sample.statistics.step00000004.rank00000000.bin')[1]
            field_max=region_max=wall_max=0.
            for case in cases:
                with h5py.File(reference/FINAL/'state.h5') as a,h5py.File(case/FINAL/'state.h5') as b:
                    assert set(a)==set(b)
                    field_max=max(field_max,max(float(np.max(abs(a[k][:]-b[k][:])))
                                                for k in a if k.startswith('q')))
                region_max=max(region_max,float(np.max(abs(meta-read_statistics(
                    case/'outdat/render/sample.statistics.step00000004.rank00000000.bin')[1]))))
                if family=='walls':
                    def integrals(path):
                        rows=re.findall(r'ASTR_INSITU_CURVE_WALL wall=\d+ area=\s*(\S+) area_means=\s*([^\n]+)',
                                        (path/'run.log').read_text())
                        assert len(rows)>=2
                        return np.array([np.fromstring(area+' '+values,sep=' ') for area,values in rows[-2:]])
                    wall_max=max(wall_max,float(np.max(abs(integrals(case)-integrals(reference)))))
            assert max(field_max,region_max,wall_max)<=2e-10,(family,backend,field_max,region_max,wall_max)
            report[family+'_'+backend]=dict(cases=4,field_maxabs=field_max,
                regional_maxabs=region_max,wall_area_and_means_maxabs=wall_max)
    return report


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Read-only IS6 NP=1/2 partition comparison')
    parser.add_argument('--verify-matrix',type=Path,required=True)
    print(json.dumps(compare_partition_evidence(parser.parse_args().verify_matrix),indent=2))
