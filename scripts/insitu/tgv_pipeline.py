# script-version: 2.0
"""Approved 32^3 TGV preset, driven only by in-memory Catalyst fields and time."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from paraview import catalyst
from paraview import simple as pv
from paraview.modules.vtkRemotingViews import vtkPVProcessWindow
from vtkmodules.vtkParallelCore import vtkMultiProcessController,vtkCommunicator
from vtkmodules.vtkCommonCore import vtkIntArray
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader

sys.path.insert(0,str(Path(catalyst.get_script_filename()).resolve().parent))
from egl_identity import EGL
from tgv_streamlines import make_trace,check_crossing

arguments=catalyst.get_args()
if len(arguments)!=2:
    raise RuntimeError('Native TGV pipeline requires output directory and CUDA UUID')
output=Path(arguments[0])
expected_uuid=arguments[1]
if not output.is_dir():
    raise RuntimeError('In-situ output directory must already exist')
options=catalyst.Options()
options.GlobalTrigger='TimeStep'
options.EnableCatalystLive=0
pv._DisableFirstRenderCameraReset()
source=pv.TrivialProducer(registrationName='grid')
controller=vtkMultiProcessController.GetGlobalController()
rank=controller.GetLocalProcessId()
frames=[]


def collective_error(message):
    local,total=vtkIntArray(),vtkIntArray()
    local.InsertNextValue(int(bool(message)))
    total.SetNumberOfTuples(1)
    controller.AllReduce(local,total,vtkCommunicator.MAX_OP)
    if total.GetValue(0):
        raise RuntimeError(message or 'Native in-situ failure on another rank')


def catalyst_execute(info):
    step,t=int(info.timestep),float(info.time)
    if not np.isfinite(t) or step<0 or (frames and (step<=frames[-1][0] or t<=frames[-1][1])):
        raise RuntimeError('Nonmonotone or invalid complete-step render clock')
    source.UpdatePipeline(t)
    contour=pv.Contour(Input=source)
    contour.ContourBy=['POINTS','Q_rs']
    contour.Isosurfaces=[0.25]
    cut=pv.Slice(Input=source)
    cut.SliceType='Plane'
    cut.SliceType.Origin=[np.pi,np.pi,np.pi/4]
    cut.SliceType.Normal=[0.,0.,1.]
    products=[('q_surface',contour),('velocity_slice',cut)]
    trace_objects=[]
    for name,constant in [('instantaneous_streamlines',False),('crossing_streamlines',True)]:
        trace,objects=make_trace(source,constant)
        products.append((name,trace))
        trace_objects.extend(objects)
    if 'mean_u_reynolds' in source.PointData.keys():
        for kind in ('reynolds','favre'):
            trace,objects=make_trace(source,mean=kind)
            products.append((f'mean_{kind}_streamlines',trace))
            trace_objects.extend(objects)
    record={'rank':rank,'step':step,'time':t,'products':{}}
    for name,geometry in products:
        existing=any((output/f'{name}.step{step:08d}.{suffix}').exists()
                     for suffix in ('pvtp','jpeg','eps'))
        collective_error('Refusing to overwrite an existing in-situ frame' if existing else '')
        geometry.UpdatePipeline(t)
        merged=pv.MergeBlocks(Input=geometry)
        surface=pv.ExtractSurface(Input=merged)
        tagged=pv.ProgrammableFilter(Input=surface)
        tagged.Script=f'''
from vtkmodules.vtkCommonCore import vtkDoubleArray,vtkTypeInt64Array
result=self.GetOutputDataObject(0)
result.ShallowCopy(self.GetInputDataObject(0,0))
for name,value,kind in [('simulation_time',{t!r},vtkDoubleArray),('complete_step',{step},vtkTypeInt64Array)]:
    item=kind()
    item.SetName(name)
    item.InsertNextValue(value)
    result.GetFieldData().AddArray(item)
'''
        destination=output/f'{name}.step{step:08d}.pvtp'
        pv.SaveData(str(destination),proxy=tagged)
        controller.Barrier()
        error=''
        if rank==0 and name=='crossing_streamlines':
            try:
                reader=vtkXMLPPolyDataReader()
                reader.SetFileName(str(destination))
                reader.Update()
                record['crossing_maxabs']=check_crossing(reader.GetOutput())
            except Exception as exc:
                error=str(exc)
        collective_error(error)
        pv.Delete(tagged)
        pv.Delete(surface)
        pv.Delete(merged)
        view=pv.CreateView('RenderView')
        view.ViewSize=[800,600]
        view.UseColorPaletteForBackground=0
        view.Background=[1.,1.,1.]
        view.OrientationAxesVisibility=0
        view.RemoteRenderThreshold=0.
        view.CameraPosition=[13.,11.,12.]
        view.CameraFocalPoint=[np.pi]*3
        view.CameraViewUp=[0.,0.,1.]
        view.CameraParallelProjection=1
        view.CameraParallelScale=5.
        display=pv.Show(geometry,view)
        if 'streamlines' in name:
            display.LineWidth=2.
        color='mean_u_'+name.split('_')[1] if name.startswith('mean_') else 'u'
        display.SetScalarColoring(color,0)
        lut=pv.GetColorTransferFunction(color)
        display.ColorArrayName=['POINTS',color]
        display.LookupTable=lut
        lut.AutomaticRescaleRangeMode='Never'
        lut.ApplyPreset('Cool to Warm (Extended)',False)
        lut.RescaleTransferFunction(-1.,1.)
        pv.Render(view)
        error=''
        actual=''
        try:
            window=vtkPVProcessWindow.GetRenderWindow()
            window.MakeCurrent()
            actual=EGL().current_uuid()
            if window.GetClassName()!='vtkEGLRenderWindow' or actual!=expected_uuid:
                raise RuntimeError('Actual EGL context differs from solver CUDA device')
        except Exception as exc:
            error=str(exc)
        collective_error(error)
        picture=output/f'{name}.step{step:08d}.jpeg'
        pv.SaveScreenshot(str(picture),view,ImageResolution=[800,600])
        error=''
        if rank==0:
            try:
                with Image.open(picture) as image:
                    image.convert('RGB').save(picture.with_suffix('.eps'))
            except Exception as exc:
                error=str(exc)
        collective_error(error)
        record['products'][name]={'egl_uuid':actual}
        pv.Delete(view)
    pv.Delete(contour)
    pv.Delete(cut)
    for proxy in trace_objects:
        pv.Delete(proxy)
    (output/f'mesh_step{step:08d}_rank{rank}.json').write_text(json.dumps(record,indent=2))
    frames.append((step,t))


def catalyst_finalize():
    (output/f'lifecycle_rank{rank}.json').write_text(json.dumps({'frames':frames,'finalized':True}))
