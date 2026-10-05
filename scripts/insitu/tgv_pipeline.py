# script-version: 2.0
"""Bounded TGV and bc41 wall presets using only in-memory Catalyst data."""
import json
import io
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from paraview import catalyst
from paraview import simple as pv,servermanager
from paraview.modules.vtkRemotingViews import vtkPVProcessWindow
from vtkmodules.vtkParallelCore import vtkMultiProcessController,vtkCommunicator
from vtkmodules.vtkCommonCore import vtkIntArray,vtkDoubleArray
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
from vtkmodules.vtkIOImage import vtkJPEGWriter
from vtkmodules.util.numpy_support import vtk_to_numpy

sys.path.insert(0,str(Path(catalyst.get_script_filename()).resolve().parent))
from egl_identity import EGL
from tgv_streamlines import make_trace,check_crossing
from image_publication import publish_pair,RECOVERABLE_ERRNOS

arguments=catalyst.get_args()
if len(arguments) not in (2,3):
    raise RuntimeError('Native TGV pipeline requires output directory and CUDA UUID')
output=Path(arguments[0])
expected_uuid=arguments[1]
profile=arguments[2] if len(arguments)==3 else 'all'
product_names={
    'all':('q_surface','velocity_slice','instantaneous_streamlines','crossing_streamlines'),
    'q_surface':('q_surface',),
    'streamlines':('instantaneous_streamlines','crossing_streamlines'),
    'q_streamlines':('q_surface','instantaneous_streamlines','crossing_streamlines'),
    'velocity_slice':('velocity_slice',),
    'channel_walls':('wall_pressure','wall_shear_x','wall_heat_into_gas'),
    'air5_walls':('pressure','temperature','vibrational_temperature','Y_N2','Y_O2','Y_N','Y_O','Y_NO',
                  'wall_shear_x','wall_heat_into_gas'),
}.get(profile)
if product_names is None:
    raise RuntimeError('Unknown TGV product profile: '+profile)
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
geometries={}
pipeline_objects=[]
extraction_objects={}
render_objects={}
owned_colors=[]


def collective_error(message):
    local,total=vtkIntArray(),vtkIntArray()
    local.InsertNextValue(int(bool(message)))
    total.SetNumberOfTuples(1)
    controller.AllReduce(local,total,vtkCommunicator.MAX_OP)
    if total.GetValue(0):
        raise RuntimeError(message or 'Native in-situ failure on another rank')


def global_surface_bounds():
    bounds=np.asarray(source.GetDataInformation().GetBounds()).reshape(3,2)
    if empty_local_plane():
        bounds[:,0]=np.inf
        bounds[:,1]=-np.inf
    local,total=vtkDoubleArray(),vtkDoubleArray()
    # A normal-decomposed rank can contain just one plane: its local y span is zero.
    for minimum,maximum in bounds:
        local.InsertNextValue(-float(minimum))
        local.InsertNextValue(float(maximum))
    total.SetNumberOfTuples(6)
    controller.AllReduce(local,total,vtkCommunicator.MAX_OP)
    result=np.asarray([total.GetValue(i) for i in range(6)]).reshape(3,2)
    result[:,0]*=-1
    spans=result[:,1]-result[:,0]
    valid=spans[0]>0 and spans[2]>0 and (spans[1]==0 if profile=='air5_walls' else spans[1]>0)
    if not np.isfinite(result).all() or not valid:
        raise RuntimeError('Invalid global wall physical bounds')
    return result


def capture_image(view):
    # Capture is collective, but contains no file publication or recovery.
    manager=servermanager.ParaViewPipelineController()
    settings=servermanager.misc.SaveScreenshot()
    manager.PreInitializeProxy(settings)
    settings.View=view
    settings.SaveAllViews=0
    settings.UpdateDefaultsAndVisibilities('frame.jpeg')
    manager.PostInitializeProxy(settings)
    settings.ImageResolution=[800,600]
    settings.UpdateVTKObjects()
    image=settings.SMProxy.CaptureImage()
    return image,int(settings.Format.Quality),int(settings.Format.Progressive)


def encode_image(image,quality,progressive):
    if image is None or tuple(image.GetDimensions())!=(800,600,1):
        raise RuntimeError('Missing or invalid collective screenshot')
    writer=vtkJPEGWriter()
    writer.SetInputData(image)
    writer.SetQuality(quality)
    writer.SetProgressive(progressive)
    writer.WriteToMemoryOn()
    writer.Write()
    if writer.GetErrorCode() or writer.GetResult() is None:
        raise RuntimeError('JPEG encoding failed')
    jpeg=vtk_to_numpy(writer.GetResult()).tobytes()
    eps=io.BytesIO()
    with Image.open(io.BytesIO(jpeg)) as decoded:
        decoded.convert('RGB').save(eps,format='EPS')
    return jpeg,eps.getvalue()


def publish_image(view,picture,name,step,time):
    error=''
    try:
        image,quality,progressive=capture_image(view)
    except Exception as exc:
        error='Collective screenshot failed: '+str(exc)
    collective_error(error)
    error=''
    failure=None
    if rank==0:
        try:
            jpeg,eps=encode_image(image,quality,progressive)
            failure=publish_pair(picture,jpeg,eps)
        except Exception as exc:
            error='Image capture/encoding/publication failed: '+str(exc)
    collective_error(error)
    # The writer is the only source of a recoverable publication errno.
    local,total=vtkIntArray(),vtkIntArray()
    local.InsertNextValue(failure['errno'] if failure else 0)
    total.SetNumberOfTuples(1)
    controller.AllReduce(local,total,vtkCommunicator.MAX_OP)
    code=total.GetValue(0)
    if code==0:
        return {'status':'published'}
    if code not in RECOVERABLE_ERRNOS:
        collective_error('Unexpected recoverable image errno')
    # Broadcast the phase as small integers, not any flow-field data.
    phases=('stage_jpeg','stage_eps','publish_jpeg','publish_eps')
    local.SetValue(0,phases.index(failure['phase'])+1 if failure else 0)
    controller.AllReduce(local,total,vtkCommunicator.MAX_OP)
    receipt={'status':'missing','product':name,'step':step,'time':time,
             'errno':code,'phase':phases[total.GetValue(0)-1],
             'reason':os.strerror(code),'geometry':picture.with_suffix('.pvtp').name}
    error=''
    try:
        journal=output/f'missing.{name}.step{step:08d}.rank{rank:08d}.json'
        with journal.open('x') as stream:
            json.dump(receipt,stream,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception as exc:
        error='Cannot record missing in-situ image: '+str(exc)
    collective_error(error)
    if rank==0:
        print('ASTR_INSITU_IMAGE_MISSING '+json.dumps(receipt),flush=True)
    return receipt


def empty_local_plane():
    data=source.GetClientSideObject().GetOutputDataObject(0)
    if data.IsA('vtkDataSet'):
        return data.GetNumberOfPoints()==0
    iterator=data.NewIterator()
    iterator.InitTraversal()
    points=0
    while not iterator.IsDoneWithTraversal():
        leaf=iterator.GetCurrentDataObject()
        if leaf is not None and leaf.IsA('vtkDataSet'):
            points+=leaf.GetNumberOfPoints()
        iterator.GoToNextItem()
    return points==0


def save_geometry(destination,proxy):
    writer=None
    error=''
    try:
        writer=pv.CreateWriter(str(destination),proxy)
        if writer is None:
            raise RuntimeError('Cannot create geometry writer')
        writer.UpdateVTKObjects()
        writer.UpdatePipeline()
        algorithm=writer.GetClientSideObject()
        if algorithm is None or algorithm.GetErrorCode():
            raise RuntimeError('Geometry writer reported failure')
    except Exception as exc:
        error='In-situ geometry publication failed: '+str(exc)
    finally:
        if writer is not None:
            pv.Delete(writer)
    collective_error(error)


def catalyst_execute(info):
    step,t=int(info.timestep),float(info.time)
    if not np.isfinite(t) or step<0 or (frames and (step<=frames[-1][0] or t<=frames[-1][1])):
        raise RuntimeError('Nonmonotone or invalid complete-step render clock')
    source.UpdatePipeline(t)
    required={'wall_pressure','wall_shear_x','wall_heat_into_gas','wall_normal_y'} if profile=='channel_walls' else {'u','v','w'}
    if profile=='air5_walls':
        required=set(product_names)|{'rho','u','v','w','wall_normal_y','wall_heat_tr_into_gas',
                                    'wall_heat_v_into_gas','wall_heat_species_into_gas'}
    if 'q_surface' in product_names:
        required.add('Q_rs')
    missing=required-set(source.PointData.keys())
    # Conduit's VTK conversion drops zero-length arrays on empty slice domains.
    if profile in ('velocity_slice','channel_walls','air5_walls') and empty_local_plane():
        missing=set()
    collective_error('Missing required product fields: '+','.join(sorted(missing)) if missing else '')
    mean_products=()
    statistics_clocks={}
    if profile in ('channel_walls','air5_walls'):
        local,total=vtkIntArray(),vtkIntArray()
        local.InsertNextValue(int('statistics_duration' in source.PointData.keys()))
        total.SetNumberOfTuples(1)
        controller.AllReduce(local,total,vtkCommunicator.MAX_OP)
        if total.GetValue(0):
            selected=tuple('mean_'+name for name in product_names)
            absent=set(selected)|{'statistics_duration','statistics_window_start','statistics_window_end'}
            absent-=set(source.PointData.keys())
            if empty_local_plane():
                absent=set()
            collective_error('Missing wall statistics: '+','.join(sorted(absent)) if absent else '')
            local_time,total_time=vtkDoubleArray(),vtkDoubleArray()
            covered=np.inf if empty_local_plane() else source.PointData.GetArray('statistics_duration').GetRange(0)[0]
            local_time.InsertNextValue(float(covered)); total_time.SetNumberOfTuples(1)
            controller.AllReduce(local_time,total_time,vtkCommunicator.MIN_OP)
            collective_error('Invalid wall statistics coverage' if not np.isfinite(total_time.GetValue(0)) else '')
            if total_time.GetValue(0)>0:
                mean_products=selected
            for field in ('statistics_duration','statistics_window_start','statistics_window_end'):
                value=np.inf if empty_local_plane() else source.PointData.GetArray(field).GetRange(0)[0]
                local_time.SetValue(0,float(value))
                controller.AllReduce(local_time,total_time,vtkCommunicator.MIN_OP)
                statistics_clocks[field]=total_time.GetValue(0)
            collective_error('Invalid wall statistics window' if not all(
                np.isfinite(value) for value in statistics_clocks.values()) else '')
    if not geometries:
        if profile in ('channel_walls','air5_walls'):
            for name in product_names:
                geometries[name]=source
        if 'q_surface' in product_names:
            contour=pv.Contour(Input=source)
            contour.ContourBy=['POINTS','Q_rs']
            contour.Isosurfaces=[0.25]
            geometries['q_surface']=contour
            pipeline_objects.append(contour)
        if 'velocity_slice' in product_names:
            if profile=='velocity_slice':
                geometries['velocity_slice']=source
            else:
                cut=pv.Slice(Input=source)
                cut.SliceType='Plane'
                cut.SliceType.Origin=[np.pi,np.pi,np.pi/4]
                cut.SliceType.Normal=[0.,0.,1.]
                geometries['velocity_slice']=cut
                pipeline_objects.append(cut)
        for name,constant in [('instantaneous_streamlines',False),('crossing_streamlines',True)]:
            if name not in product_names:
                continue
            trace,objects=make_trace(source,constant)
            geometries[name]=trace
            pipeline_objects.extend(objects)
    products=[(name,geometries[name]) for name in product_names]
    for name in mean_products:
        geometries[name]=source
        products.append((name,source))
    if profile=='all' and 'mean_u_reynolds' in source.PointData.keys():
        for kind in ('reynolds','favre'):
            name=f'mean_{kind}_streamlines'
            if name not in geometries:
                trace,objects=make_trace(source,mean=kind)
                geometries[name]=trace
                pipeline_objects.extend(objects)
            products.append((name,geometries[name]))
    record={'rank':rank,'step':step,'time':t,'products':{}}
    if profile!='all':
        record.update(profile=profile,available_fields=sorted(source.PointData.keys()))
    if profile=='air5_walls':
        record['units']={name:('kg/m^3' if name=='rho' else 'm/s' if name in ('u','v','w')
                         else 'K' if 'temperature' in name else 'Pa' if name in ('pressure','wall_shear_x')
                         else 'W/m^2' if 'heat' in name else '1') for name in sorted(required)}
        record['units'].update({'mean_'+name:record['units'][name] for name in product_names if mean_products})
        if mean_products:
            for field in source.PointData.keys():
                for kind in ('mean_','variance_','rms_'):
                    if field.startswith(kind) and field[len(kind):] in record['units']:
                        base=record['units'][field[len(kind):]]
                        record['units'][field]=f'({base})^2' if kind=='variance_' else base
            record['units']['statistics_duration']='s'
            record['units']['statistics_window_start']='s'
            record['units']['statistics_window_end']='s'
    if statistics_clocks:
        record['statistics']=dict(statistics_clocks,average='Reynolds')
    for name,geometry in products:
        existing=any((output/f'{name}.step{step:08d}.{suffix}').exists()
                     for suffix in ('pvtp','jpeg','eps'))
        collective_error('Refusing to overwrite an existing in-situ frame' if existing else '')
        geometry.UpdatePipeline(t)
        if name not in extraction_objects:
            merged=pv.MergeBlocks(Input=geometry)
            surface=pv.ExtractSurface(Input=merged)
            tagged=pv.ProgrammableFilter(Input=surface)
            extraction_objects[name]=(tagged,surface,merged)
        tagged,surface,merged=extraction_objects[name]
        tagged.Script=f'''
from vtkmodules.vtkCommonCore import vtkDoubleArray,vtkTypeInt64Array,vtkStringArray
result=self.GetOutputDataObject(0)
result.ShallowCopy(self.GetInputDataObject(0,0))
for name,value,kind in [('simulation_time',{t!r},vtkDoubleArray),('complete_step',{step},vtkTypeInt64Array)]:
    item=kind()
    item.SetName(name)
    item.InsertNextValue(value)
    result.GetFieldData().AddArray(item)
for name,value in {statistics_clocks!r}.items():
    item=vtkDoubleArray()
    item.SetName(name)
    item.InsertNextValue(value)
    result.GetFieldData().AddArray(item)
units=vtkStringArray()
units.SetName('field_units')
for name,unit in {record.get('units',{})!r}.items():
    units.InsertNextValue(name+'='+unit)
result.GetFieldData().AddArray(units)
'''
        destination=output/f'{name}.step{step:08d}.pvtp'
        save_geometry(destination,tagged)
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
        if name not in render_objects:
            manager=servermanager.ProxyManager()
            initial_colors={group:set(manager.GetProxiesInGroup(group)) for group in
                            ('lookup_tables','piecewise_functions')}
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
            if profile in ('channel_walls','air5_walls'):
                bounds=global_surface_bounds()
                center=bounds.mean(axis=1)
                length=bounds[:,1]-bounds[:,0]
                view.CameraPosition=(center+length*np.array([1.2,1.7,1.7])).tolist()
                if profile=='air5_walls':
                    view.CameraPosition=(center+float(length.max())*np.array([.3,1.7,.5])).tolist()
                view.CameraFocalPoint=center.tolist()
                view.CameraViewUp=[0.,1.,0.]
                view.CameraParallelScale=.6*float(length.max())
            display=pv.Show(geometry,view)
            if 'streamlines' in name:
                display.LineWidth=2.
            color=name if profile in ('channel_walls','air5_walls') else ('mean_u_'+name.split('_')[1] if name.startswith('mean_') else 'u')
            display.SetScalarColoring(color,0)
            lut=pv.GetColorTransferFunction(color)
            display.ColorArrayName=['POINTS',color]
            display.LookupTable=lut
            lut.AutomaticRescaleRangeMode='Never'
            lut.ApplyPreset('Cool to Warm (Extended)',False)
            physical_name=name.removeprefix('mean_')
            color_range={'wall_pressure':(0.,12.),'wall_shear_x':(-.02,.02),
                         'wall_heat_into_gas':(-.2,.2)}.get(physical_name,(-1.,1.))
            if profile=='air5_walls':
                color_range=({'pressure':(0.,60000.),'temperature':(1500.,3500.),
                    'vibrational_temperature':(1500.,3500.),'wall_shear_x':(-1.,1.),
                    'wall_heat_into_gas':(-60000.,60000.)}.get(physical_name,(0.,1.)))
            lut.RescaleTransferFunction(*color_range)
            render_objects[name]=(view,display)
            for group,initial in initial_colors.items():
                owned_colors.extend(proxy for key,proxy in manager.GetProxiesInGroup(group).items()
                                    if key not in initial)
        view,display=render_objects[name]
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
        status=publish_image(view,picture,name,step,t)
        record['products'][name]={'egl_uuid':actual,'image':status}
    error=''
    try:
        (output/f'mesh_step{step:08d}_rank{rank}.json').write_text(json.dumps(record,indent=2))
    except Exception as exc:
        error='Cannot record in-situ frame: '+str(exc)
    collective_error(error)
    frames.append((step,t))


def catalyst_finalize():
    for view,display in render_objects.values():
        pv.Delete(display)
        pv.Delete(view)
    render_objects.clear()
    for objects in extraction_objects.values():
        for proxy in objects:
            pv.Delete(proxy)
    extraction_objects.clear()
    for proxy in pipeline_objects:
        pv.Delete(proxy)
    pipeline_objects.clear()
    geometries.clear()
    for proxy in owned_colors:
        pv.Delete(proxy)
    owned_colors.clear()
    (output/f'lifecycle_rank{rank}.json').write_text(json.dumps({'frames':frames,'finalized':True}))
