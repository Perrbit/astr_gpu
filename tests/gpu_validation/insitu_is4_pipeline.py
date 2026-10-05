"""Bounded Cartesian ownership/measure and rotated streamline diagnostics."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from paraview import catalyst, simple as pv
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkParallelCore import vtkCommunicator
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
from tgv_streamlines import make_trace,check_crossing

base=runpy.run_path(str(ROOT/'scripts/insitu/tgv_pipeline.py'))
options=base['options']
source=base['source']
controller=base['controller']
rank=base['rank']
output=base['output']
collective_error=base['collective_error']
traces={}


def reduce(values,operation):
    local=numpy_to_vtk(np.ascontiguousarray(values),deep=True)
    total=local.NewInstance()
    total.SetNumberOfComponents(local.GetNumberOfComponents())
    total.SetNumberOfTuples(local.GetNumberOfTuples())
    controller.AllReduce(local,total,operation)
    return vtk_to_numpy(total)


def mesh_receipt():
    data=source.GetClientSideObject().GetOutputDataObject(0)
    if not data.IsA('vtkDataSet'):
        iterator=data.NewIterator(); iterator.InitTraversal()
        leaves=[]
        while not iterator.IsDoneWithTraversal():
            leaf=iterator.GetCurrentDataObject()
            if leaf is not None and leaf.IsA('vtkDataSet'):
                leaves.append(leaf)
            iterator.GoToNextItem()
        if len(leaves)!=1:
            raise ValueError('Expected one local volume block')
        data=leaves[0]
    if not data.IsA('vtkStructuredGrid'):
        raise ValueError('IS4 volume oracle requires a structured Cartesian block')
    xyz=vtk_to_numpy(data.GetPoints().GetData())
    h=2*np.pi/32
    indices=np.rint(xyz/h).astype(np.int64)
    if not np.isfinite(xyz).all() or np.max(abs(xyz-indices*h))>2e-10:
        raise ValueError('Nonfinite or displaced Cartesian nodes')
    lo,hi=indices.min(axis=0),indices.max(axis=0)
    if np.any(lo<0) or np.any(hi>32) or np.any(hi<=lo):
        raise ValueError('Invalid local physical extent')
    z,y,x=np.meshgrid(*(np.arange(lo[d],hi[d]+1) for d in (2,1,0)),indexing='ij')
    expected=np.stack((x,y,z),axis=-1).reshape(-1,3)
    if not np.array_equal(indices,expected):
        raise ValueError('Structured connectivity and coordinates disagree')
    cells=np.prod(hi-lo)
    if data.GetNumberOfCells()!=cells:
        raise ValueError('Missing or duplicated local cells')
    owned=indices[np.all(indices<hi,axis=1)]
    ids=np.ravel_multi_index(owned.T,(32,32,32))
    node_counts=np.bincount(ids,minlength=32**3).astype(np.int32)
    # Structured cells have exactly these lower corners, with no ghost cells.
    cell_counts=node_counts.copy()
    weights=np.prod(np.where((indices==lo)|(indices==hi),0.5,1.),axis=1)*h**3
    quantities=np.array([float(cells)*h**3,weights.sum()])
    fields=['u','v','w']+(['Q_rs'] if data.GetPointData().GetArray('Q_rs') is not None else [])
    canonical=np.ravel_multi_index((indices%32).T,(32,32,32))
    seams=[]
    for name in fields:
        array=data.GetPointData().GetArray(name)
        if array is None:
            raise ValueError('Missing seam field '+name)
        values=vtk_to_numpy(array)
        if values.shape!=(len(indices),) or not np.isfinite(values).all():
            raise ValueError('Invalid seam field '+name)
        lower=np.full(32**3,np.finfo(float).max)
        upper=-lower.copy()
        np.minimum.at(lower,canonical,values)
        np.maximum.at(upper,canonical,values)
        seams.append((lower,upper))
    return node_counts,cell_counts,quantities,seams,fields,int(cells),len(indices)


def check_mesh(step,time):
    error=''
    try:
        payload=mesh_receipt()
    except Exception as exc:
        error=str(exc)
    collective_error(error)
    nodes,cells,quantities,seams,fields,local_cells,local_nodes=payload
    nodes=reduce(nodes,vtkCommunicator.SUM_OP)
    cells=reduce(cells,vtkCommunicator.SUM_OP)
    measure=reduce(quantities,vtkCommunicator.SUM_OP)
    seam_errors={}
    for name,(lower,upper) in zip(fields,seams):
        lower=reduce(lower,vtkCommunicator.MIN_OP)
        upper=reduce(upper,vtkCommunicator.MAX_OP)
        seam_errors[name]=float(np.max(abs(upper-lower)))
    expected=(2*np.pi)**3
    error=''
    if not np.all(nodes==1) or not np.all(cells==1):
        error='Nonunique or missing global node/cell ownership'
    elif np.max(abs(measure-expected))>2e-10:
        error='Physical volume/constant integral differs from the analytic box'
    elif max(seam_errors.values())>2e-10:
        error='Shared-node or periodic seam fields disagree'
    collective_error(error)
    return dict(step=step,time=time,unique_nodes=int(nodes.sum()),unique_cells=int(cells.sum()),
                physical_volume=float(measure[0]),constant_integral=float(measure[1]),
                local_cells=local_cells,local_nodes=local_nodes,seam_maxabs=seam_errors)


def catalyst_execute(info):
    step,time=int(info.timestep),float(info.time)
    source.UpdatePipeline(time)
    receipt=check_mesh(step,time)
    base['catalyst_execute'](info)
    receipt['crossing_maxabs']={}
    for axis in 'xyz':
        if axis not in traces:
            traces[axis]=make_trace(source,constant=True,axis=axis)
        trace,_=traces[axis]
        trace.UpdatePipeline(time)
        path=output/f'axis_crossing_{axis}.step{step:08d}.pvtp'
        collective_error('Refusing diagnostic geometry overwrite' if path.exists() else '')
        pv.SaveData(str(path),proxy=trace)
        controller.Barrier()
        error=''
        if rank==0:
            try:
                reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
                receipt['crossing_maxabs'][axis]=check_crossing(reader.GetOutput(),axis)
            except Exception as exc:
                error=str(exc)
        collective_error(error)
    (output/f'is4.step{step:08d}.rank{rank:08d}.json').write_text(json.dumps(receipt,indent=2))


def catalyst_finalize():
    for _,objects in traces.values():
        for proxy in objects:
            pv.Delete(proxy)
    traces.clear()
    base['catalyst_finalize']()
