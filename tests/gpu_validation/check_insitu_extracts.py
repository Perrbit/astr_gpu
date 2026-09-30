"""Read TGV slice/Q products in a fresh pvpython process without Catalyst."""
import argparse
import json
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader


FIELDS = ('q1 q2 q3 q4 q5 rho u v w pressure temperature '
          'du_dx dv_dx dw_dx du_dy dv_dy dw_dy du_dz dv_dz dw_dz '
          'Q_rs divergence omega_x omega_y omega_z').split()


def check(directory, steps, streamlines=False, means=False):
    records = []
    products = ['velocity_slice', 'q_surface']
    if streamlines:
        products += ['instantaneous_streamlines','crossing_streamlines']
    if means:
        products += ['mean_reynolds_streamlines','mean_favre_streamlines']
    for product in products:
        product_steps=[s for s in steps if s>0] if product.startswith('mean_') else steps
        expected = {f'{product}.step{step:08d}.pvtp' for step in product_steps}
        actual = {p.name for p in directory.glob(f'{product}.step*.pvtp')}
        if actual != expected:
            raise ValueError(f'{directory}: unexpected {product} frame sequence')
        for step in product_steps:
            path = directory / f'{product}.step{step:08d}.pvtp'
            reader = vtkXMLPPolyDataReader()
            errors = []
            reader.AddObserver('ErrorEvent', lambda *args: errors.append('VTK reader error'))
            reader.SetFileName(str(path))
            reader.Update()
            data = reader.GetOutput()
            if errors or data.GetNumberOfPoints() == 0 or data.GetNumberOfCells() == 0:
                raise ValueError(f'{path}: failed or empty geometry')
            xyz = vtk_to_numpy(data.GetPoints().GetData())
            if not np.all(np.isfinite(xyz)) or np.any(xyz < -2.e-10) or np.any(xyz > 2*np.pi+2.e-10):
                raise ValueError(f'{path}: invalid coordinates')
            fields=list(FIELDS)
            if means and step>0:
                fields += [f'mean_{axis}_{kind}' for kind in ('reynolds','favre') for axis in ('u','v','w')]
                fields += ['statistics_duration']
            for name in fields:
                array = data.GetPointData().GetArray(name)
                if array is None or array.GetNumberOfTuples() != len(xyz):
                    raise ValueError(f'{path}: missing or wrong-sized field {name}')
                if not np.all(np.isfinite(vtk_to_numpy(array))):
                    raise ValueError(f'{path}: nonfinite field {name}')
            if means and step>0:
                duration=vtk_to_numpy(data.GetPointData().GetArray('statistics_duration'))
                np.testing.assert_allclose(duration,min(step*1.e-3,0.0035)-0.0005,atol=2.e-10,rtol=0)
            for name, expected_value in [('simulation_time', step*1.e-3), ('complete_step', step)]:
                array = data.GetFieldData().GetArray(name)
                if array is None or array.GetNumberOfTuples() != 1 or array.GetTuple1(0) != expected_value:
                    raise ValueError(f'{path}: wrong {name}')
            if product=='velocity_slice':
                error=float(np.max(abs(xyz[:,2]-np.pi/4)))
            elif product=='q_surface':
                error=float(np.max(abs(vtk_to_numpy(data.GetPointData().GetArray('Q_rs'))-0.25)))
            elif product=='crossing_streamlines':
                from insitu_streamlines import check_crossing
                error=check_crossing(data)
            else:
                error=0.
                if data.GetPoints().GetData().GetDataTypeAsString()!='double':
                    raise ValueError(f'{path}: streamline coordinates are not double')
            if error > 2.e-10:
                raise ValueError(f'{path}: geometry constraint error {error}')
            records.append(dict(file=str(path), points=len(xyz), cells=data.GetNumberOfCells(),
                                step=step, time=step*1.e-3, constraint_maxabs=error))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--streamlines', action='store_true')
    parser.add_argument('--mean-streamlines', action='store_true')
    parser.add_argument('--continuous-only', action='store_true')
    args = parser.parse_args()
    records = []
    modes=[('continuous',[0,2,4])]
    if not args.continuous_only:
        modes.append(('resumed',[4]))
    for ranks in (1, 2):
        for mode, steps in modes:
            records.extend(check(args.paired_output / f'np{ranks}_{mode}' / 'outdat', steps,
                                 args.streamlines,args.mean_streamlines))
    with args.report.open('x') as stream:
        json.dump(dict(status='passed-independent-offline-readback', records=records), stream, indent=2)
    print(f'PASS: {len(records)} geometry files read in an independent process')


if __name__ == '__main__':
    main()
