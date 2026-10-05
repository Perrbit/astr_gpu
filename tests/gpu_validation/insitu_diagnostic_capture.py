"""Native local-field capture for the bounded GPU derivative gate; no rendering."""
import json
from pathlib import Path

import numpy as np
from paraview import catalyst, simple as pv
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkParallelCore import vtkMultiProcessController

options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 0
source = pv.TrivialProducer(registrationName='grid')
arguments = catalyst.get_args()
output = Path(arguments[0])
profile = arguments[2] if len(arguments)==3 else 'all'
rank = vtkMultiProcessController.GetGlobalController().GetLocalProcessId()
frames = []
names = ('q1 q2 q3 q4 q5 rho u v w pressure temperature '
         'du_dx dv_dx dw_dx du_dy dv_dy dw_dy du_dz dv_dz dw_dz '
         'Q_rs divergence omega_x omega_y omega_z').split()
if profile!='all':
    names=['u','v','w']+([] if profile in ('streamlines','velocity_slice') else ['Q_rs'])


def catalyst_execute(info):
    step, time = int(info.timestep), float(info.time)
    if step not in (0, 2, 4) or abs(time-step*1e-3) > 1e-15 or (frames and step <= frames[-1][0]):
        raise RuntimeError('Unexpected diagnostic capture phase')
    source.UpdatePipeline(time)
    data = source.GetClientSideObject().GetOutputDataObject(0)
    if not data.IsA('vtkDataSet'):
        iterator = data.NewIterator()
        iterator.InitTraversal()
        leaves = []
        while not iterator.IsDoneWithTraversal():
            leaf = iterator.GetCurrentDataObject()
            if leaf is not None and leaf.IsA('vtkDataSet'):
                leaves.append(leaf)
            iterator.GoToNextItem()
        if len(leaves) != 1:
            raise RuntimeError('Expected one local diagnostic block')
        data = leaves[0]
    values = dict(xyz=vtk_to_numpy(data.GetPoints().GetData()))
    if profile!='all':
        available={data.GetPointData().GetArrayName(i) for i in range(data.GetPointData().GetNumberOfArrays())}
        if available!=set(names) and not (profile=='velocity_slice' and data.GetNumberOfPoints()==0 and not available):
            raise RuntimeError('Unexpected selected-field payload: '+str(sorted(available)))
    for name in names:
        array = data.GetPointData().GetArray(name)
        if array is None and profile=='velocity_slice' and data.GetNumberOfPoints()==0:
            values[name]=np.empty(0,dtype=np.float64)
            continue
        if array is None or array.GetNumberOfTuples() != data.GetNumberOfPoints():
            raise RuntimeError('Missing diagnostic field: '+name)
        values[name] = vtk_to_numpy(array)
    if profile=='velocity_slice':
        cells=data.GetCells()
        connectivity=vtk_to_numpy(cells.GetConnectivityArray())
        values['cells']=connectivity.reshape(-1,4)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise RuntimeError('Nonfinite diagnostic payload')
    path = output/f'fields.step{step:08d}.rank{rank:08d}.npz'
    if path.exists():
        raise FileExistsError(path)
    np.savez(path, **values)
    frames.append([step, time])


def catalyst_finalize():
    (output/f'capture.rank{rank:08d}.json').write_text(json.dumps(dict(frames=frames, finalized=True)))
