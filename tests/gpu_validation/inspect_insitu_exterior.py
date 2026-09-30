"""Check constant-field exterior termination; launch pvbatch --symmetric for MPI."""
import argparse
import json
from pathlib import Path

import numpy as np
from paraview import simple as pv
from vtkmodules.vtkParallelCore import vtkMultiProcessController
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from insitu_streamlines import make_trace


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
controller = vtkMultiProcessController.GetGlobalController()
rank = controller.GetLocalProcessId()
mpi_size = controller.GetNumberOfProcesses()
if mpi_size not in (1, 2):
    raise ValueError('This probe supports NP=1/2 only')
if rank == 0:
    args.output.mkdir(parents=True, exist_ok=False)
controller.Barrier()
mesh = pv.ProgrammableSource()
mesh.OutputDataSetType = 'vtkStructuredGrid'
mesh.Script = '''
import math
from vtkmodules.vtkParallelCore import vtkMultiProcessController
from vtkmodules.vtkCommonCore import vtkPoints
c=vtkMultiProcessController.GetGlobalController()
n=32//c.GetNumberOfProcesses()
offset=c.GetLocalProcessId()*n
p=vtkPoints()
p.SetDataTypeToDouble()
for k in range(33):
    for j in range(33):
        for i in range(n+1):
            p.InsertNextPoint((offset+i)*math.pi/16,j*math.pi/16,k*math.pi/16)
g=self.GetStructuredGridOutput()
g.SetDimensions(n+1,33,33)
g.SetPoints(p)
'''
trace, proxies = make_trace(mesh, constant=True)
trace.IntegrationDirection = 'BACKWARD'
merged = pv.MergeBlocks(Input=trace)
surface = pv.ExtractSurface(Input=merged)
pv.SaveData(str(args.output/'exterior.pvtp'), proxy=surface)
controller.Barrier()
if rank == 0:
    reader = vtkXMLPPolyDataReader()
    reader.SetFileName(str(args.output/'exterior.pvtp'))
    reader.Update()
    data = reader.GetOutput()
    xyz = vtk_to_numpy(data.GetPoints().GetData())
    endpoints = [data.GetPoint(data.GetCell(i).GetPointId(data.GetCell(i).GetNumberOfPoints()-1))
                 for i in range(data.GetNumberOfCells())]
    reasons = vtk_to_numpy(data.GetCellData().GetArray('ReasonForTermination')).tolist()
    seed_ids=vtk_to_numpy(data.GetCellData().GetArray('SeedIds'))
    if data.GetNumberOfCells()!=16 or sorted(seed_ids.tolist())!=list(range(16)):
        raise ValueError('Missing or duplicated exterior trajectories')
    if not np.all(np.isfinite(xyz)) or np.any(xyz < -2.e-10) or np.any(xyz > 2*np.pi+2.e-10):
        raise ValueError('Nonfinite or out-of-box coordinates')
    if np.any(np.asarray(reasons)!=vtkStreamTracer.OUT_OF_DOMAIN):
        raise ValueError('Trace stopped for a reason other than leaving the domain')
    distance=float(np.max(np.abs(np.asarray(endpoints)[:,0])))
    if distance>2.e-10:
        raise ValueError(f'Exterior endpoint error {distance} exceeds 2e-10')
    for index in range(16):
        cell=data.GetCell(index)
        line=np.asarray([data.GetPoint(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())])
        seed=int(seed_ids[index])
        np.testing.assert_allclose(line[:,1],np.pi/8+seed*3*np.pi/4/15,atol=2.e-10,rtol=0)
        np.testing.assert_allclose(line[:,2],np.pi/4,atol=2.e-10,rtol=0)
        if np.any(np.diff(line[:,0])>=0):
            raise ValueError('Trace reversed or wrapped instead of stopping')
    report = dict(status='passed-bounded-constant-field-exterior', np=mpi_size,
                  cells=data.GetNumberOfCells(), reasons=reasons,
                  min_x=float(xyz[:,0].min()), max_x=float(xyz[:,0].max()),
                  endpoint_wall_distance_max=distance,
                  coordinate_type=data.GetPoints().GetData().GetDataTypeAsString())
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
