# script-version: 2.1
"""Strict device products in ParaView views, without host geometry filters."""
import ctypes
import io
import json
from math import isfinite
import os
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
from paraview import catalyst, servermanager
from paraview import simple as pv
from paraview.modules.vtkRemotingViews import vtkPVProcessWindow
from vtkmodules.vtkCommonCore import vtkIntArray, vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkPolyData
from vtkmodules.vtkIOImage import vtkJPEGWriter
from vtkmodules.vtkParallelCore import vtkCommunicator, vtkMultiProcessController
from vtkmodules.vtkRenderingCore import vtkActor
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper
from vtkmodules.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(catalyst.get_script_filename()).resolve().parent))
from egl_identity import EGL
from image_publication import publish_pair, RECOVERABLE_ERRNOS
import product_dispatch as dispatch

arguments = catalyst.get_args()
if len(arguments) != 4 or arguments[3] not in (
        'standard-device', 'direct-device'):
    raise RuntimeError('Strict rendering requires an explicit resident pipeline')
output = Path(arguments[0])
expected_uuid, profile, pipeline = arguments[1:]
product_names = {
    'all': ('q_surface', 'velocity_slice', 'instantaneous_streamlines', 'crossing_streamlines'),
    'q_surface': ('q_surface',),
    'velocity_slice': ('velocity_slice',),
    'streamlines': ('instantaneous_streamlines', 'crossing_streamlines'),
    'q_streamlines': ('q_surface', 'instantaneous_streamlines', 'crossing_streamlines'),
    'tgv256_demo': ('q_surface', 'instantaneous_streamlines'),
    'curve_demo': ('q_surface', 'instantaneous_streamlines'),
    'boundary_layer': ('q_surface', 'velocity_slice', 'instantaneous_streamlines'),
    'channel_walls': ('wall_pressure', 'wall_shear_x', 'wall_heat_into_gas'),
    'curve_walls': ('wall_pressure', 'wall_shear_x', 'wall_heat_into_gas'),
    'air5_walls': ('pressure', 'temperature', 'vibrational_temperature', 'Y_N2', 'Y_O2',
                   'Y_N', 'Y_O', 'Y_NO', 'wall_shear_x', 'wall_heat_into_gas'),
    'physical_plane': ('velocity_slice',),
}.get(profile)
if product_names is None:
    raise RuntimeError('Unsupported bounded resident product profile')
if not output.is_dir() or os.environ.get('ASTR_VTK_STRICT_DEVICE_ACCESS') != '1':
    raise RuntimeError('Missing output directory or strict device access guard')
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 0
pv._DisableFirstRenderCameraReset()
controller = vtkMultiProcessController.GetGlobalController()
rank = controller.GetLocalProcessId()
sources = {name: pv.TrivialProducer(registrationName=name) for name in (
    'q_surface', 'velocity_slice', 'instantaneous_streamlines', 'crossing_streamlines',
    'mean_reynolds_streamlines', 'mean_favre_streamlines',
    'wall_pressure', 'wall_shear_x', 'wall_heat_into_gas', 'pressure', 'temperature',
    'vibrational_temperature', 'Y_N2', 'Y_O2', 'Y_N', 'Y_O', 'Y_NO')}
wall_profile = profile in ('channel_walls', 'curve_walls', 'air5_walls')
if wall_profile:
    sources.update({('mean_'+name): pv.TrivialProducer(registrationName='mean_'+name) for name in product_names})
render_objects = {}
frames = []
image_size = [1280, 960] if profile in ('tgv256_demo','curve_demo') else [800, 600]
color_range = [0., 1.] if profile in ('tgv256_demo','curve_demo') else [-1., 1.]
if profile == 'boundary_layer':
    image_size, color_range = [1536, 384], [0., 1.1]
timing = os.environ.get('ASTR_INSITU_TIMING', '') in ('1', 't', 'T', 'true', 'TRUE', 'on', 'ON')


def collective_error(message):
    local, total = vtkIntArray(), vtkIntArray()
    local.InsertNextValue(int(bool(message)))
    total.SetNumberOfTuples(1)
    controller.AllReduce(local, total, vtkCommunicator.MAX_OP)
    if total.GetValue(0):
        raise RuntimeError(message or 'Strict device rendering failed on another rank')


def local_piece(data):
    if data.IsA('vtkDataSet'):
        return data
    iterator = data.NewIterator()
    iterator.InitTraversal()
    pieces = []
    while not iterator.IsDoneWithTraversal():
        piece = iterator.GetCurrentDataObject()
        if piece is not None and piece.IsA('vtkDataSet'):
            pieces.append(piece)
        iterator.GoToNextItem()
    if len(pieces) != 1:
        raise RuntimeError('Strict entry requires one local homogeneous product, without merging')
    return pieces[0]


def metadata(piece, name):
    array = piece.GetFieldData().GetArray(name)
    if array is None or array.GetNumberOfTuples() != 1:
        raise RuntimeError('Missing bounded frame metadata: ' + name)
    return array.GetTuple1(0)


def update_geometry(name, piece):
    item = render_objects.setdefault(name, {})
    points = int(metadata(piece, 'device_points'))
    cells = int(metadata(piece, 'device_cells'))
    arity = int(metadata(piece, 'device_arity'))
    if arity not in (2, 3):
        raise RuntimeError('Unsupported resident product cell arity')
    if pipeline == 'standard-device':
        poly = vtkPolyData()
        poly.SetPoints(piece.GetPoints())
        if arity == 3:
            poly.SetPolys(piece.GetCells())
        else:
            poly.SetLines(piece.GetCells())
        poly.GetPointData().ShallowCopy(piece.GetPointData())
        poly.GetInformation().Set(vtkDataObject.BOUNDING_BOX(),
            [metadata(piece, 'bound' + str(d)) for d in range(6)], 6)
        if poly.GetNumberOfPoints() != points or poly.GetNumberOfCells() != cells:
            raise RuntimeError('Device array extent differs from bounded metadata')
        if not item:
            mapper = vtkOpenGLPolyDataMapper()
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            item.update(actor=actor, mapper=mapper)
        item['mapper'].SetInputData(poly)
    elif not item:
        bridge = ctypes.CDLL(None)
        bridge.astr_insitu_resident_actor.argtypes = [ctypes.c_char_p]
        bridge.astr_insitu_resident_actor.restype = ctypes.c_size_t
        address = bridge.astr_insitu_resident_actor(name.encode())
        if not address:
            raise RuntimeError('Missing registered direct device actor')
        actor = vtkActor(f'_{address:016x}_p_vtkActor')
        item.update(actor=actor, mapper=actor.GetMapper())
    if 'view' not in item:
        setup_view(name, item, piece)
    if profile == 'boundary_layer':
        item['local_bounds'] = [metadata(piece, 'bound'+str(d)) for d in range(6)] if points else None
    return points, cells


def global_device_bounds(piece, allow_planar=False, allow_empty=False):
    local, total = vtkDoubleArray(), vtkDoubleArray()
    points = int(metadata(piece, 'device_points'))
    for d in range(6):
        value = metadata(piece, 'bound' + str(d)) if points else (float('inf') if d % 2 == 0 else -float('inf'))
        local.InsertNextValue(-value if d % 2 == 0 else value)
    total.SetNumberOfTuples(6)
    controller.AllReduce(local, total, vtkCommunicator.MAX_OP)
    bounds = [(-1. if d % 2 == 0 else 1.) * total.GetValue(d) for d in range(6)]
    from math import isfinite
    if allow_empty and all(bounds[2*d] == float('inf') and bounds[2*d+1] == -float('inf') for d in range(3)):
        return None
    if not all(isfinite(value) for value in bounds) or any(
            bounds[2*d] > bounds[2*d+1] if allow_planar else bounds[2*d] >= bounds[2*d+1] for d in range(3)):
        raise RuntimeError('Invalid globally reduced device wall bounds')
    if allow_planar and sum(bounds[2*d] < bounds[2*d+1] for d in range(3)) < 2:
        raise RuntimeError('Physical slice has no two-dimensional extent')
    return bounds


def setup_view(name, item, piece):
    field = name.removeprefix('mean_') if wall_profile else name
    if 'streamlines' in name:
        item['actor'].GetProperty().SetLineWidth(2.)
    view = pv.CreateView('RenderView')
    view.ViewSize = image_size
    view.UseColorPaletteForBackground = 0
    view.Background = [1., 1., 1.]
    view.OrientationAxesVisibility = 0
    view.RemoteRenderThreshold = 0.
    view.CameraPosition = [13., 11., 12.]
    view.CameraFocalPoint = [3.141592653589793] * 3
    view.CameraViewUp = [0., 0., 1.]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 5.
    bounds = [0., 6.283185307179586] * 3
    if profile=='channel_walls':
        # Approved Cartesian fixture bounds; no scan of device geometry.
        from math import pi
        center=[pi,1.,.5*pi]
        length=[2*pi,2.,pi]
        view.CameraPosition=[center[d]+length[d]*[1.2,1.7,1.7][d] for d in range(3)]
        view.CameraFocalPoint=center
        view.CameraViewUp=[0.,1.,0.]
        view.CameraParallelScale=.6*max(length)
        bounds=[0.,2*pi,0.,2.,0.,pi]
    elif profile=='curve_walls':
        # Six metadata scalars were reduced on the GPU, not scanned in VTK.
        bounds=global_device_bounds(piece)
        center=[.5*(bounds[2*d]+bounds[2*d+1]) for d in range(3)]
        length=[bounds[2*d+1]-bounds[2*d] for d in range(3)]
        view.CameraPosition=[center[d]+length[d]*[1.2,1.7,1.7][d] for d in range(3)]
        view.CameraFocalPoint=center
        view.CameraViewUp=[0.,1.,0.]
        view.CameraParallelScale=max(length)
    elif profile=='air5_walls':
        from math import sqrt
        bounds=global_device_bounds(piece, allow_planar=True)
        center=[.5*(bounds[2*d]+bounds[2*d+1]) for d in range(3)]
        length=[bounds[2*d+1]-bounds[2*d] for d in range(3)]
        offset=[.3,1.7,.5]
        view.CameraPosition=[center[d]+max(length)*offset[d] for d in range(3)]
        view.CameraFocalPoint=center
        view.CameraViewUp=[0.,1.,0.]
        magnitude=sqrt(sum(value*value for value in offset))
        forward=[-value/magnitude for value in offset]
        right=[-forward[2],0.,forward[0]]
        magnitude=sqrt(sum(value*value for value in right))
        right=[value/magnitude for value in right]
        up=[right[1]*forward[2]-right[2]*forward[1],
            right[2]*forward[0]-right[0]*forward[2],right[0]*forward[1]-right[1]*forward[0]]
        half_height=.5*sum(abs(up[d])*length[d] for d in range(3))
        half_width=.5*sum(abs(right[d])*length[d] for d in range(3))
        view.CameraParallelScale=max(.6*max(length),1.05*half_height,
            1.05*half_width/(image_size[0]/image_size[1]))
        item['physical_bounds']=bounds
    elif profile=='boundary_layer':
        view.CameraPosition = [550., 375. if name == 'q_surface' else 75., 900.]
        view.CameraFocalPoint = [550., 75., 45.]
        view.CameraViewUp = [0., 1., 0.]
        view.CameraParallelScale = 150.
        bounds = [0., 1100., 0., 150., 0., 90.]
        item['physical_bounds'] = bounds
        if name == 'velocity_slice':
            item['plane'] = {
                'origin': [metadata(piece, 'plane_origin'+str(d)) for d in range(3)],
                'normal': [metadata(piece, 'plane_normal'+str(d)) for d in range(3)]}
    elif profile=='physical_plane':
        from math import sqrt
        plane_origin=[metadata(piece, 'plane_origin'+str(d)) for d in range(3)]
        normal=[metadata(piece, 'plane_normal'+str(d)) for d in range(3)]
        plane_bounds=global_device_bounds(piece, allow_planar=True, allow_empty=True)
        center=plane_origin if plane_bounds is None else [.5*(plane_bounds[2*d]+plane_bounds[2*d+1]) for d in range(3)]
        length=6.283185307179586 if plane_bounds is None else max(plane_bounds[2*d+1]-plane_bounds[2*d] for d in range(3))
        axis=min(range(3), key=lambda d: abs(normal[d]))
        up=[float(d==axis)-normal[axis]*normal[d] for d in range(3)]
        norm=sqrt(sum(value*value for value in up))
        view.CameraPosition=[center[d]+2.*length*normal[d] for d in range(3)]
        view.CameraFocalPoint=center
        view.CameraViewUp=[value/norm for value in up]
        view.CameraParallelScale=.8*length
        if plane_bounds is not None:
            # Camera clipping padding only; never used to classify intersections.
            bounds=[value+(-.05 if d%2==0 else .05)*length for d,value in enumerate(plane_bounds)]
        item['plane']={'origin':plane_origin, 'normal':normal}
    view.UpdateVTKObjects()
    client = view.GetClientSideObject()
    client.GetRenderer().AddActor(item['actor'])
    client.SetMaxClipBounds(bounds)
    client.SetLockBounds(True)
    selected_range=color_range
    if profile in ('channel_walls','curve_walls'):
        color=name
        selected_range={'wall_pressure':[0.,12.], 'wall_shear_x':[-.02,.02],
                        'wall_heat_into_gas':[-.2,.2]}[field]
        if profile=='curve_walls':
            selected_range={'wall_pressure':[80.,120.], 'wall_shear_x':[-.2,.2],
                            'wall_heat_into_gas':[0.,15000.]}[field]
    elif profile=='air5_walls':
        color=name
        selected_range=({'pressure':[0.,60000.], 'temperature':[1500.,3500.],
            'vibrational_temperature':[1500.,3500.], 'wall_shear_x':[-1.,1.],
            'wall_heat_into_gas':[-60000.,60000.]}).get(field,[0.,1.])
    elif profile in ('tgv256_demo','curve_demo','boundary_layer'):
        color = 'speed'
    elif name.startswith('mean_'):
        color = 'mean_u_' + ('reynolds' if name=='mean_reynolds_streamlines' else 'favre')
    else:
        color = 'u'
    lut = pv.GetColorTransferFunction(color)
    lut.AutomaticRescaleRangeMode = 'Never'
    lut.ApplyPreset('Cool to Warm (Extended)', False)
    lut.RescaleTransferFunction(*selected_range)
    legend = pv.GetScalarBar(lut, view)
    legend.Visibility = 1
    legend.Title = {'mean_u_reynolds': 'Reynolds u', 'mean_u_favre': 'Favre u', 'speed': 'Speed'}.get(color, color)
    legend.ComponentTitle = ''
    legend.WindowLocation = 'Upper Right Corner'
    legend.ScalarBarLength = .45
    legend.ScalarBarThickness = 16
    legend.TitleFontSize = 16
    legend.LabelFontSize = 14
    legend.TitleColor = [0., 0., 0.]
    legend.LabelColor = [0., 0., 0.]
    if profile in ('curve_walls','physical_plane','air5_walls'):
        legend.WindowLocation='Any Location'
        legend.Position=[.8,.5]
        legend.ScalarBarLength=.4
    item.update(view=view, legend=legend, lut=lut, color=color, color_range=selected_range)


def capture_image(view):
    manager = servermanager.ParaViewPipelineController()
    settings = servermanager.misc.SaveScreenshot()
    manager.PreInitializeProxy(settings)
    settings.View = view
    settings.SaveAllViews = 0
    settings.UpdateDefaultsAndVisibilities('frame.jpeg')
    manager.PostInitializeProxy(settings)
    settings.ImageResolution = image_size
    settings.UpdateVTKObjects()
    return settings.SMProxy.CaptureImage(), int(settings.Format.Quality), int(settings.Format.Progressive)


def catalyst_execute(info):
    started = perf_counter()
    step, time = int(info.timestep), float(info.time)
    sources[product_names[0]].UpdatePipeline(time)
    first = local_piece(sources[product_names[0]].GetClientSideObject().GetOutputDataObject(0))
    names = [name for name in product_names if dispatch.scene_due(name)]
    statistics = None
    if wall_profile and first.GetFieldData().GetArray('mean_requested') is not None:
        covered = metadata(first, 'mean_covered')
        statistics = {key: metadata(first, key) for key in (
            'statistics_duration', 'statistics_window_start', 'statistics_window_end')}
        duration = statistics['statistics_duration']
        if covered not in (0, 1) or not all(isfinite(value) for value in statistics.values()) or duration < 0. or \
                statistics['statistics_window_end'] <= statistics['statistics_window_start'] or bool(covered) != (duration > 0.):
            raise RuntimeError('Invalid device wall mean window/coverage')
        if covered:
            names += ['mean_'+name for name in product_names]
    if profile in ('all', 'streamlines') and metadata(first, 'mean_covered'):
        if profile=='streamlines':
            statistics = {key: metadata(first, key) for key in (
                'statistics_duration', 'statistics_window_start', 'statistics_window_end')}
            span = statistics['statistics_window_end']-statistics['statistics_window_start']
            duration = statistics['statistics_duration']
            if not all(isfinite(value) for value in statistics.values()) or duration<=0. or span<=0. or \
                    abs(span-duration)>64*sys.float_info.epsilon*max(1.,abs(span),abs(duration)):
                raise RuntimeError('Invalid resident mean streamline coverage')
        names += [name for name in ('mean_reynolds_streamlines', 'mean_favre_streamlines') if dispatch.scene_due(name)]
    elif profile in ('all', 'streamlines') and dispatch.independent():
        for name in ('mean_reynolds_streamlines', 'mean_favre_streamlines'):
            if dispatch.scene_due(name):
                dispatch.report_uncovered(name)
    products = {}
    for name in names:
        source = sources[name]
        source.UpdatePipeline(time)
        piece = local_piece(source.GetClientSideObject().GetOutputDataObject(0))
        points, cells = update_geometry(name, piece)
        item = render_objects[name]
        products[name] = render_product(name, item, step, time, points, cells)
        if piece.GetFieldData().GetArray('streamline_seed_count') is not None:
            products[name]['seeding'] = {
                'layout': {0: 'line16', 1: 'tgv-stratified', 2: 'bl-layered64'}[
                    int(metadata(piece, 'streamline_seed_layout'))],
                'seeds': int(metadata(piece, 'streamline_seed_count')),
                'direction_trajectories': int(metadata(piece, 'streamline_particle_count'))}
        dispatch.report(name, 'image', products[name]['image'])
    record = {'step': step, 'time': time, 'processing_backend': 'device',
        'rendering_pipeline': pipeline, 'profile':profile, 'geometry_host_bytes': 0,
        'local_points': products[names[0]]['local_points'] if names else 0,
        'local_cells': products[names[0]]['local_cells'] if names else 0,
        'products': products}
    if statistics is not None:
        record['statistics'] = statistics
    adaptive=dispatch.adaptive_receipts(names)
    if adaptive:
        record['adaptive_outputs']=adaptive
    (output / f'mesh_step{step:08d}_rank{rank}.json').write_text(json.dumps(record, indent=2))
    frames.append((step, time))
    if timing:
        print('ASTR_INSITU_PIPELINE_TIMING ' + json.dumps({'rank': rank, 'step': step, 'seconds': {
            'python_execute_inclusive': perf_counter()-started}}, sort_keys=True), flush=True)


def render_product(name, item, step, time, points, cells):
    os.environ['ASTR_INSITU_PIXEL_PRODUCT']=name
    os.environ['ASTR_INSITU_PIXEL_STEP']=str(step)
    view = item['view']
    render_start = perf_counter()
    pv.Render(view)
    window = vtkPVProcessWindow.GetRenderWindow()
    window.MakeCurrent()
    actual = EGL().current_uuid()
    collective_error('Actual CUDA/EGL identity mismatch' if
        window.GetClassName() != 'vtkEGLRenderWindow' or actual != expected_uuid else '')
    render_seconds = perf_counter() - render_start
    capture_start = perf_counter()
    error = ''
    try:
        image, quality, progressive = capture_image(view)
    except Exception as exc:
        error = 'Collective screenshot failed: ' + str(exc)
    collective_error(error)
    capture_seconds = perf_counter() - capture_start
    error = ''
    failure = None
    encoding_seconds = publication_seconds = 0.
    if rank == 0:
        try:
            encoding_start = perf_counter()
            if image is None or tuple(image.GetDimensions()) != (*image_size, 1):
                raise RuntimeError('Missing collective image')
            writer = vtkJPEGWriter()
            writer.SetInputData(image)
            writer.SetQuality(quality)
            writer.SetProgressive(progressive)
            writer.WriteToMemoryOn()
            writer.Write()
            if writer.GetErrorCode() or writer.GetResult() is None:
                raise RuntimeError('Image encoding failed')
            jpeg = vtk_to_numpy(writer.GetResult()).tobytes()
            eps = io.BytesIO()
            with Image.open(io.BytesIO(jpeg)) as decoded:
                decoded.convert('RGB').save(eps, format='EPS')
            encoding_seconds = perf_counter() - encoding_start
            publication_start = perf_counter()
            failure = publish_pair(output / f'{name}.step{step:08d}.jpeg', jpeg, eps.getvalue())
            publication_seconds = perf_counter() - publication_start
        except Exception as exc:
            error = str(exc)
    collective_error(error)
    status, total = vtkIntArray(), vtkIntArray()
    status.InsertNextValue(failure['errno'] if failure else 0)
    total.SetNumberOfTuples(1)
    controller.AllReduce(status, total, vtkCommunicator.MAX_OP)
    code = total.GetValue(0)
    if code:
        if code not in RECOVERABLE_ERRNOS:
            collective_error('Unexpected recoverable image errno')
        phases = ('stage_jpeg', 'stage_eps', 'publish_jpeg', 'publish_eps')
        status.SetValue(0, phases.index(failure['phase'])+1 if failure else 0)
        controller.AllReduce(status, total, vtkCommunicator.MAX_OP)
        receipt = {'status': 'missing', 'product': name, 'step': step, 'time': time,
            'errno': code, 'phase': phases[total.GetValue(0)-1], 'reason': os.strerror(code),
            'geometry': 'device-resident; no geometry file'}
        error = ''
        try:
            with (output / f'missing.{name}.step{step:08d}.rank{rank:08d}.json').open('x') as stream:
                json.dump(receipt, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            error = 'Cannot record missing in-situ image: ' + str(exc)
        collective_error(error)
        if rank == 0:
            print('ASTR_INSITU_IMAGE_MISSING ' + json.dumps(receipt), flush=True)
    if timing:
        print('ASTR_INSITU_PIPELINE_TIMING ' + json.dumps({'rank': rank, 'step': step, 'product': name, 'seconds': {
            'render_inclusive': render_seconds,
            'screenshot_capture_inclusive': capture_seconds,
            'image_encoding': encoding_seconds,
            'image_publication': publication_seconds}}, sort_keys=True), flush=True)
    result = {'egl_uuid': actual, 'color_field': item['color'], 'color_range': item['color_range'],
        'scalar_bar_visible': bool(item['legend'].Visibility), 'scalar_bar_label_color': [0., 0., 0.],
        'local_points': points, 'local_cells': cells, 'image': 'published' if code == 0 else receipt}
    if 'plane' in item:
        result['plane']=item['plane']
    if profile=='boundary_layer':
        result['physical_bounds']=item['physical_bounds']
        result['local_geometry_bounds']=item['local_bounds']
        result['camera'] = dict(position=list(view.CameraPosition), focal_point=list(view.CameraFocalPoint),
                                view_up=list(view.CameraViewUp), parallel_scale=float(view.CameraParallelScale))
    if profile=='air5_walls':
        field=name.removeprefix('mean_')
        result['field_unit']=({'pressure':'Pa','temperature':'K','vibrational_temperature':'K',
            'wall_shear_x':'Pa','wall_heat_into_gas':'W/m^2'}).get(field,'1')
        result['physical_bounds']=item['physical_bounds']
    return result


def catalyst_finalize():
    for item in render_objects.values():
        view = item['view']
        window = vtkPVProcessWindow.GetRenderWindow()
        window.MakeCurrent()
        item['mapper'].ReleaseGraphicsResources(window)
        view.GetClientSideObject().GetRenderer().RemoveActor(item['actor'])
        pv.Delete(item['legend'])
        pv.Delete(view)
        pv.Delete(item['lut'])
    for source in sources.values():
        pv.Delete(source)
    (output / f'lifecycle_rank{rank}.json').write_text(json.dumps({'frames': frames, 'finalized': True}))
