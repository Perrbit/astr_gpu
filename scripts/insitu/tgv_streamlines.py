"""Approved TGV RK45 preset and independent constant-field crossing oracle."""
import numpy as np
from paraview import simple as pv
from vtkmodules.util.numpy_support import vtk_to_numpy


def make_trace(source, constant=False, mean=None, h=2*np.pi/32):
    velocity = pv.Calculator(Input=source)
    velocity.ResultArrayName = 'trace_velocity'
    velocity.Function = 'iHat' if constant else 'u*iHat+v*jHat+w*kHat'
    if mean is not None:
        velocity.Function = f'mean_u_{mean}*iHat+mean_v_{mean}*jHat+mean_w_{mean}*kHat'
    seeds = pv.ProgrammableSource()
    seeds.OutputDataSetType = 'vtkPolyData'
    seeds.Script = '''
import math
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray
points = vtkPoints()
points.SetDataTypeToDouble()
vertices = vtkCellArray()
for i in range(16):
    index = points.InsertNextPoint(math.pi/2, math.pi/8 + i*(3*math.pi/4)/15, math.pi/4)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(index)
output = self.GetPolyDataOutput()
output.SetPoints(points)
output.SetVerts(vertices)
'''
    trace = pv.StreamTracerWithCustomSource(Input=velocity, SeedSource=seeds)
    trace.Vectors = ['POINTS', 'trace_velocity']
    trace.IntegratorType = 'Runge-Kutta 4-5'
    trace.IntegrationStepUnit = 'Length'
    trace.InitialStepLength = 0.1*h
    trace.MinimumStepLength = 0.01*h
    trace.MaximumStepLength = 0.5*h
    trace.MaximumError = 1.e-8
    trace.MaximumStreamlineLength = np.pi
    trace.IntegrationDirection = 'FORWARD' if constant else 'BOTH'
    trace.MaximumSteps = 100000
    trace.UseLocalSeedSource = 0
    # The constant-field oracle isolates partition crossing, not wall termination.
    return trace, (trace, seeds, velocity)


def check_crossing(data):
    seed_ids = data.GetCellData().GetArray('SeedIds')
    if seed_ids is None or sorted(set(vtk_to_numpy(seed_ids).tolist())) != list(range(16)):
        raise ValueError('Missing or duplicated streamline seeds')
    segments = [[] for _ in range(16)]
    for index in range(data.GetNumberOfCells()):
        cell = data.GetCell(index)
        xyz = np.array([data.GetPoint(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())])
        if len(xyz)<2 or not np.all(np.isfinite(xyz)) or np.any(np.diff(xyz[:,0])<=0):
            raise ValueError('Nonfinite or reversed constant-field segment')
        segments[int(seed_ids.GetTuple1(index))].append(xyz)
    worst = 0.
    for seed, parts in enumerate(segments):
        parts.sort(key=lambda xyz:xyz[0,0])
        for previous,following in zip(parts,parts[1:]):
            if np.max(abs(previous[-1]-following[0]))>2.e-10:
                raise ValueError('Disconnected or overlapping partition segments')
        xyz = np.concatenate(parts)
        y = np.pi/8 + seed*(3*np.pi/4)/15
        error = max(abs(xyz[0,0]-np.pi/2), abs(xyz[-1,0]-3*np.pi/2),
                    np.max(abs(xyz[:,1]-y)), np.max(abs(xyz[:,2]-np.pi/4)))
        if not np.any(xyz[:,0]<np.pi) or not np.any(xyz[:,0]>np.pi):
            raise ValueError('Streamline did not cross x=pi')
        worst = max(worst,float(error))
    if worst > 2.e-10:
        raise ValueError(f'Constant-field endpoint/straightness error {worst} exceeds 2e-10')
    return worst
