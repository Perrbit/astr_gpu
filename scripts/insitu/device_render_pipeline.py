# script-version: 2.0
"""Strict device products in ParaView views, without host geometry filters."""
import ctypes
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
from paraview import catalyst, servermanager
from paraview import simple as pv
from paraview.modules.vtkRemotingViews import vtkPVProcessWindow
from vtkmodules.vtkCommonCore import vtkIntArray
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
    'mean_reynolds_streamlines', 'mean_favre_streamlines')}
render_objects = {}
frames = []
image_size = [1280, 960] if profile=='tgv256_demo' else [800, 600]
color_range = [0., 1.] if profile=='tgv256_demo' else [-1., 1.]
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
        setup_view(name, item)
    return points, cells


def setup_view(name, item):
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
    view.UpdateVTKObjects()
    client = view.GetClientSideObject()
    client.GetRenderer().AddActor(item['actor'])
    # The fixed physical box supplies clipping bounds, not a host geometry scan.
    client.SetMaxClipBounds([0., 6.283185307179586] * 3)
    client.SetLockBounds(True)
    if profile=='tgv256_demo':
        color = 'speed'
    elif name.startswith('mean_'):
        color = 'mean_u_' + ('reynolds' if name=='mean_reynolds_streamlines' else 'favre')
    else:
        color = 'u'
    lut = pv.GetColorTransferFunction(color)
    lut.AutomaticRescaleRangeMode = 'Never'
    lut.ApplyPreset('Cool to Warm (Extended)', False)
    lut.RescaleTransferFunction(*color_range)
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
    item.update(view=view, legend=legend, lut=lut, color=color)


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
    if profile == 'all' and metadata(first, 'mean_covered'):
        names += [name for name in ('mean_reynolds_streamlines', 'mean_favre_streamlines') if dispatch.scene_due(name)]
    elif profile == 'all' and dispatch.independent():
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
        dispatch.report(name, 'image', products[name]['image'])
    record = {'step': step, 'time': time, 'processing_backend': 'device',
        'rendering_pipeline': pipeline, 'geometry_host_bytes': 0,
        'local_points': products[names[0]]['local_points'] if names else 0,
        'local_cells': products[names[0]]['local_cells'] if names else 0,
        'products': products}
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
    return {'egl_uuid': actual, 'color_field': item['color'], 'color_range': color_range,
        'scalar_bar_visible': bool(item['legend'].Visibility), 'scalar_bar_label_color': [0., 0., 0.],
        'local_points': points, 'local_cells': cells, 'image': 'published' if code == 0 else receipt}


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
