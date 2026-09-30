# script-version: 2.0
"""Three-frame TGV lifecycle test, not a production visualization preset."""
import json
import os
from pathlib import Path
import sys
import resource

import numpy as np
from paraview import catalyst
from paraview import simple as pv
from paraview.modules.vtkRemotingViews import vtkPVProcessWindow
from vtkmodules.vtkParallelCore import vtkMultiProcessController,vtkCommunicator
from vtkmodules.vtkCommonCore import vtkIntArray
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader

sys.path.insert(0, os.environ["ASTR_PROBE_IDENTITY_MODULE"])
from insitu_device_identity import EGL
from insitu_streamlines import make_trace,check_crossing
from insitu_statistics_reference import read_statistics

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 0
pv._DisableFirstRenderCameraReset()
source = pv.TrivialProducer(registrationName="grid")
controller = vtkMultiProcessController.GetGlobalController()
rank = controller.GetLocalProcessId()
output = Path(os.environ["ASTR_PROBE_OUTPUT"])
completed_frames = []
expected_frames = json.loads(os.environ.get('ASTR_INSITU_TEST_EXPECTED_STEPS', '[1,2,3]'))


def leaf(data):
    if data.IsA("vtkDataSet"):
        return data
    iterator = data.NewIterator()
    iterator.InitTraversal()
    blocks = []
    while not iterator.IsDoneWithTraversal():
        obj = iterator.GetCurrentDataObject()
        if obj is not None and obj.IsA("vtkDataSet"):
            blocks.append(obj)
        iterator.GoToNextItem()
    if len(blocks) != 1:
        raise RuntimeError("Expected one local structured block")
    return blocks[0]


def catalyst_execute(info):
    step = int(info.timestep)
    if step != expected_frames[len(completed_frames)] or abs(info.time-step*1.e-3) > 1.e-15:
        raise RuntimeError("Wrong completed-step phase")
    source.UpdatePipeline(info.time)
    data = leaf(source.GetClientSideObject().GetOutputDataObject(0))
    prefix = os.environ["ASTR_INSITU_SAMPLE_PREFIX"]
    oracle_io=os.environ.get('ASTR_INSITU_TEST_ORACLE_IO','1')!='0'
    n=data.GetNumberOfPoints()
    cells=data.GetNumberOfCells()
    names = ["q1","q2","q3","q4","q5","rho","u","v","w","pressure","temperature"]
    diagnostics = ["du_dx","dv_dx","dw_dx","du_dy","dv_dy","dw_dy",
                   "du_dz","dv_dz","dw_dz","Q_rs","divergence","omega_x","omega_y","omega_z"]
    if oracle_io:
        raw = Path(f"{prefix}.canonical.step{step:08d}.rank{rank:08d}.bin").read_bytes()
        if raw[:8] != b"ASTRIC01":
            raise RuntimeError("Not a canonical sample")
        header = np.frombuffer(raw, dtype="<i4", count=9, offset=8)
        values = np.frombuffer(raw, dtype="<f8", offset=60).reshape(14, n)
        if n!=int(np.prod(header[3:6])) or cells!=int(np.prod(header[3:6]-1)):
            raise RuntimeError("Wrong local mesh counts")
        np.testing.assert_array_equal(vtk_to_numpy(data.GetPoints().GetData()), values[:3].T)
        derived = Path(f"{prefix}.derived.step{step:08d}.rank{rank:08d}.bin").read_bytes()
        if derived[:8] != b"ASTRID01" or derived[8:60] != raw[8:60]:
            raise RuntimeError("Derived metadata mismatch")
        expected = np.concatenate((values[3:], np.frombuffer(derived, dtype="<f8", offset=60).reshape(14,n)))
        for name, reference in zip(names+diagnostics, expected):
            np.testing.assert_array_equal(vtk_to_numpy(data.GetPointData().GetArray(name)), reference)
    else:
        for name in names+diagnostics:
            array=data.GetPointData().GetArray(name)
            if array is None or array.GetNumberOfTuples()!=n or not np.isfinite(vtk_to_numpy(array)).all():
                raise RuntimeError('Missing or invalid in-memory field: '+name)
    mean_names=[]
    if os.environ.get('ASTR_INSITU_TEST_MEAN_STREAMLINES')=='1':
        if step>0:
            mean_names=[f'mean_{axis}_{kind}' for kind in ('reynolds','favre') for axis in ('u','v','w')]
            mean_names+=['statistics_duration']
            if oracle_io:
                _,_,statistics=read_statistics(f'{prefix}.statistics.step{step:08d}.rank{rank:08d}.bin')
                for name,column in zip(mean_names,[2,3,4,5,6,7,0]):
                    np.testing.assert_array_equal(vtk_to_numpy(data.GetPointData().GetArray(name)),
                                                  statistics[...,column].ravel(order='F'))
        elif data.GetPointData().GetArray('mean_u_reynolds') is not None:
            raise RuntimeError('Mean field was emitted before statistical coverage')

    contour = pv.Contour(Input=source)
    contour.ContourBy = ["POINTS", "Q_rs"]
    contour.Isosurfaces = [0.25]
    cut = pv.Slice(Input=source)
    cut.SliceType = "Plane"
    cut.SliceType.Origin = [np.pi, np.pi, np.pi/4]
    cut.SliceType.Normal = [0., 0., 1.]
    record = {"rank": rank, "time": info.time, "step": info.timestep,
              "mesh_cells": cells, "point_fields_exact": 25 if oracle_io else None,
              "oracle_file_io":oracle_io, "products": {}}
    schedule = Path(f"{prefix}.schedule.step{step:08d}.rank{rank:08d}.txt").read_text().split()
    if int(schedule[0]) != step or float(schedule[1]) != info.time:
        raise RuntimeError('Schedule record is not the actual state time')
    record['crossed_targets'] = int(schedule[2])
    record['persistent_bridge_bytes'] = 28*n*8
    record['persistent_bridge_bytes'] += len(mean_names)*n*8
    record['retained_final_snapshot_bytes'] = 28*n*8 if os.environ.get('ASTR_INSITU_TEST_FINAL')=='1' else 0
    products = [("q_surface", contour), ("velocity_slice", cut)]
    trace_objects = []
    if os.environ.get('ASTR_INSITU_TEST_STREAMLINES')=='1':
        for name,constant in [('instantaneous_streamlines',False),('crossing_streamlines',True)]:
            trace,objects=make_trace(source,constant)
            products.append((name,trace))
            trace_objects.extend(objects)
        if mean_names:
            for kind in ('reynolds','favre'):
                trace,objects=make_trace(source,mean=kind)
                products.append((f'mean_{kind}_streamlines',trace))
                trace_objects.extend(objects)
    for name, geometry in products:
        geometry.UpdatePipeline(info.time)
        count = geometry.GetDataInformation().GetNumberOfCells()
        if os.environ.get('ASTR_INSITU_TEST_EXTRACTS')=='1':
            merged = pv.MergeBlocks(Input=geometry)
            surface = pv.ExtractSurface(Input=merged)
            surface.UpdatePipeline(info.time)
            tagged = pv.ProgrammableFilter(Input=surface)
            tagged.Script = f'''
from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkTypeInt64Array
result = self.GetOutputDataObject(0)
result.ShallowCopy(self.GetInputDataObject(0, 0))
for name, value, kind in [('simulation_time', {float(info.time)!r}, vtkDoubleArray),
                          ('complete_step', {step}, vtkTypeInt64Array)]:
    item = kind()
    item.SetName(name)
    item.InsertNextValue(value)
    result.GetFieldData().AddArray(item)
'''
            destination = output / f'{name}.step{step:08d}.pvtp'
            pv.SaveData(str(destination),proxy=tagged)
            controller.Barrier()
            crossing_error = ''
            if rank==0:
                reader = vtkXMLPPolyDataReader()
                reader.SetFileName(str(destination))
                reader.Update()
                restored = reader.GetOutput()
                if restored.GetNumberOfCells()==0:
                    raise RuntimeError('Empty reloaded spatial extract')
                for key,value in (('simulation_time',info.time),('complete_step',step)):
                    array=restored.GetFieldData().GetArray(key)
                    if array is None or array.GetTuple1(0)!=value:
                        raise RuntimeError('Extract time/step metadata mismatch')
                for key in names+diagnostics+mean_names:
                    array=restored.GetPointData().GetArray(key)
                    if array is None or not np.isfinite(vtk_to_numpy(array)).all():
                        raise RuntimeError('Missing or nonfinite extract field: '+key)
                coordinates=vtk_to_numpy(restored.GetPoints().GetData())
                if name=='velocity_slice':
                    np.testing.assert_allclose(coordinates[:,2],np.pi/4,rtol=0,atol=2e-10)
                elif name=='q_surface':
                    np.testing.assert_allclose(vtk_to_numpy(restored.GetPointData().GetArray('Q_rs')),
                                               0.25,rtol=0,atol=2e-10)
                elif name=='crossing_streamlines':
                    try:
                        record['crossing_maxabs']=check_crossing(restored)
                    except ValueError as error:
                        crossing_error=str(error)
                record.setdefault('extracts',{})[name]={'file':destination.name,
                    'points':restored.GetNumberOfPoints(),'cells':restored.GetNumberOfCells(),
                    'step':step,'time':info.time,'fields':names+diagnostics+mean_names}
            if name=='crossing_streamlines':
                failed,any_failed=vtkIntArray(),vtkIntArray()
                failed.InsertNextValue(int(bool(crossing_error)))
                any_failed.SetNumberOfTuples(1)
                controller.AllReduce(failed,any_failed,vtkCommunicator.MAX_OP)
                if any_failed.GetValue(0):
                    raise RuntimeError(crossing_error or 'Crossing validation failed on reader rank')
            pv.Delete(tagged)
            pv.Delete(surface)
            pv.Delete(merged)
        view = pv.CreateView("RenderView")
        view.ViewSize = [800, 600]
        view.UseColorPaletteForBackground = 0
        view.Background = [1.,1.,1.]
        view.OrientationAxesVisibility = 0
        view.RemoteRenderThreshold = 0.
        view.CameraPosition = [13.,11.,12.]
        view.CameraFocalPoint = [np.pi]*3
        view.CameraViewUp = [0.,0.,1.]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 5.
        display = pv.Show(geometry, view)
        if 'streamlines' in name:
            display.LineWidth = 2.
        color_field = 'mean_u_'+name.split('_')[1] if name.startswith('mean_') else 'u'
        display.SetScalarColoring(color_field, 0)
        lut = pv.GetColorTransferFunction(color_field)
        # ColorBy auto-rescales from local ranges; empty partitions can skip its
        # collective. Assign the approved fixed map directly on every rank.
        display.ColorArrayName = ['POINTS', color_field]
        display.LookupTable = lut
        lut.AutomaticRescaleRangeMode = 'Never'
        lut.ApplyPreset("Cool to Warm (Extended)", False)
        lut.RescaleTransferFunction(-1.,1.)
        pv.Render(view)
        window = vtkPVProcessWindow.GetRenderWindow()
        window.MakeCurrent()
        actual = EGL().current_uuid()
        if window.GetClassName() != "vtkEGLRenderWindow" or actual != os.environ["ASTR_PROBE_EXPECTED_UUID"]:
            raise RuntimeError("Actual GPU renderer identity mismatch")
        pv.SaveScreenshot(str(output / f"{name}.step{step:08d}.jpeg"), view, ImageResolution=[800,600])
        record["products"][name] = {"local_cells": count, "egl_uuid": actual}
        pv.Delete(view)
    pv.Delete(contour)
    pv.Delete(cut)
    for proxy in trace_objects:
        pv.Delete(proxy)
    record['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record['rss_kib_after_frame'] = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                          if line.startswith('VmRSS:')))
    (output / f"mesh_step{step:08d}_rank{rank}.json").write_text(json.dumps(record, indent=2))
    completed_frames.append(step)


def catalyst_finalize():
    if completed_frames != expected_frames:
        raise RuntimeError("Incomplete persistent Catalyst lifecycle")
    (output / f"lifecycle_rank{rank}.json").write_text(json.dumps({"frames":completed_frames,
                                                                "finalized":True}))
