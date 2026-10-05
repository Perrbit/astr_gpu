"""AIR5 shear-line sampling is not a separation claim for the zero-inflow fixture."""
import csv

import numpy as np
import pytest

from test_insitu_air5_walls import arguments,TOPOLOGIES,FINAL,read_wall
from test_insitu_wall_scalar_statistics import config
from run_output_air5_restart_validation import launch,DT
from run_output_restart_validation import compare_fields


def settings():
    return config().replace('statistics=.true.','statistics=.true.,wall_separation=t')


def read_records(path):
    return list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')))


@pytest.mark.parametrize('topology',TOPOLOGIES,ids=('single','x','y','z'))
def test_spanwise_shear_and_restart(tmp_path,topology):
    ranks=int(np.prod(topology)); args=arguments(tmp_path,topology); compared=[]
    for backend in ('cpu','gpu'):
        plain,_=launch(args,backend,ranks,'oracle',4,interval=1,wall_samples=True)
        sampled,_=launch(args,backend,ranks,'sampled',4,interval=1,insitu_config=settings())
        compare_fields(plain/FINAL/'state.h5',sampled/FINAL/'state.h5')
        rows=read_records(sampled/'outdat/render/sample.wall_separation.step00000004.csv')
        assert len(rows)==17 and all(row['record']=='profile' for row in rows)
        assert all(row['status']=='not_applicable_no_positive_inflow' for row in rows)
        points={}
        for rank in range(ranks):
            _,xyz,fields,_=read_wall(plain/f'outdat/sample.air5_wall.step00000004.rank{rank:08d}.bin')
            for index in np.ndindex(xyz.shape[:3]):
                point=tuple(xyz[index]); value=fields[index][12]
                if point in points:
                    np.testing.assert_allclose(value,points[point],atol=2e-10,rtol=0)
                points[point]=value
        zz=np.unique([point[2] for point in points]); period=float(zz[-1]-zz[0]); zz=zz[:-1]
        weights=.5*(np.r_[zz[1:],zz[0]+period]-np.r_[zz[-1]-period,zz[:-1]])
        np.testing.assert_allclose(weights.sum(),.002,atol=2e-18,rtol=0)
        for row in rows:
            assert int(row['step'])==4 and float(row['time'])==4*DT
            position=float(row['x_lower'])
            xvalues=np.unique([point[0] for point in points])
            nearest=xvalues[np.argmin(abs(xvalues-position))]
            np.testing.assert_allclose(nearest,position,atol=2e-10,rtol=0)
            expected=sum(weight*points[(nearest,0.,z)] for weight,z in zip(weights,zz))/period
            np.testing.assert_allclose(float(row['value']),expected,atol=2e-10,rtol=0)
        compared.append(np.array([[float(row['x_lower']),float(row['value'])] for row in rows]))
        assert len(list(sampled.glob('outdat/render/*.wall_separation.*.csv')))==5
        source=sampled/'outdat/new/checkpoints/step000000000003'
        resumed,_=launch(args,backend,ranks,'restart',4,restore=source,interval=1,insitu_config=settings())
        for name in ('state.h5','statistics.h5'):
            compare_fields(sampled/FINAL/name,resumed/FINAL/name)
        filename='outdat/render/sample.wall_separation.step00000004.csv'
        assert (sampled/filename).read_bytes()==(resumed/filename).read_bytes()
        launch(args,backend,ranks,'wrong_selection',4,restore=source,interval=1,insitu_config=config(),
            reject='AIR5 separation selection mismatch')
    np.testing.assert_allclose(*compared,atol=2e-10,rtol=0)
