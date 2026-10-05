# script-version: 2.0
"""256^3 render-only TGV demonstration; not the IS3 acceptance preset."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
from paraview import catalyst
from paraview import simple as pv, servermanager
from paraview.modules.vtkRemotingViews import vtkPVProcessWindow
from vtkmodules.vtkParallelCore import vtkMultiProcessController, vtkCommunicator
from vtkmodules.vtkCommonCore import vtkIntArray

sys.path.insert(0, str(Path(catalyst.get_script_filename()).resolve().parent))
from egl_identity import EGL
from tgv_streamlines import make_trace

output_path, expected_uuid = catalyst.get_args()
output = Path(output_path)
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 0
pv._DisableFirstRenderCameraReset()
source = pv.TrivialProducer(registrationName='grid')
controller = vtkMultiProcessController.GetGlobalController()
rank = controller.GetLocalProcessId()
frames = []
products = []
pipeline_objects = []
profiles = []
owned_colors = []


def require_collective(message):
    local, total = vtkIntArray(), vtkIntArray()
    local.InsertNextValue(int(bool(message)))
    total.SetNumberOfTuples(1)
    controller.AllReduce(local, total, vtkCommunicator.MAX_OP)
    if total.GetValue(0):
        raise RuntimeError(message or 'Demo rendering failed on another rank')


def initialize_products():
    if products:
        return
    manager = servermanager.ProxyManager()
    initial_colors = {group: set(manager.GetProxiesInGroup(group)) for group in
                      ('lookup_tables', 'piecewise_functions')}
    speed = pv.Calculator(Input=source)
    speed.ResultArrayName = 'speed'
    speed.Function = 'sqrt(u*u+v*v+w*w)'
    contour = pv.Contour(Input=speed)
    contour.ContourBy = ['POINTS', 'Q_rs']
    contour.Isosurfaces = [0.0]
    trace, trace_objects = make_trace(speed, h=2*np.pi/256)
    pipeline_objects.extend([*trace_objects, contour, speed])
    for name, geometry in [('q0_speed', contour), ('streamlines_speed', trace)]:
        view = pv.CreateView('RenderView')
        view.ViewSize = [1280, 960]
        view.UseColorPaletteForBackground = 0
        view.Background = [1., 1., 1.]
        view.OrientationAxesVisibility = 0
        view.RemoteRenderThreshold = 0.
        view.CameraPosition = [13., 11., 12.]
        view.CameraFocalPoint = [np.pi]*3
        view.CameraViewUp = [0., 0., 1.]
        view.CameraParallelProjection = 1
        view.CameraParallelScale = 5.
        display = pv.Show(geometry, view)
        display.LineWidth = 3.
        display.Ambient = 0.4
        display.Diffuse = 0.6
        # Avoid ColorBy's data-range collective on empty streamline partitions.
        display.SetScalarColoring('speed', 0)
        lut = pv.GetColorTransferFunction('speed')
        display.ColorArrayName = ['POINTS', 'speed']
        display.LookupTable = lut
        lut.AutomaticRescaleRangeMode = 'Never'
        lut.ApplyPreset('Cool to Warm (Extended)', False)
        lut.RescaleTransferFunction(0., 1.)
        display.SetScalarBarVisibility(view, True)
        bar = pv.GetScalarBar(lut, view)
        bar.Title = 'Speed'
        bar.ComponentTitle = ''
        bar.TitleColor = [0., 0., 0.]
        bar.LabelColor = [0., 0., 0.]
        bar.WindowLocation = 'Any Location'
        bar.Position = [0.88, 0.12]
        bar.ScalarBarLength = 0.35
        bar.TitleFontSize = 18
        bar.LabelFontSize = 16
        products.append((name, geometry, view, display, bar))
    for group, initial in initial_colors.items():
        owned_colors.extend(proxy for key, proxy in manager.GetProxiesInGroup(group).items()
                            if key not in initial)


def catalyst_execute(info):
    step, time = int(info.timestep), float(info.time)
    require_collective('Invalid frame sequence' if step < 1 or not np.isfinite(time) or
                       (frames and (step != frames[-1][0]+1 or time <= frames[-1][1])) else '')
    frame_start = perf_counter()
    source.UpdatePipeline(time)
    setup_start = perf_counter()
    initialize_products()
    profile = {'step': step, 'time': time, 'setup_seconds': perf_counter()-setup_start,
               'products': {}}
    for name, geometry, view, display, bar in products:
        picture = output/f'{name}.step{step:08d}.jpeg'
        require_collective('Refusing to overwrite frame' if picture.exists() or
                           picture.with_suffix('.eps').exists() else '')
        start = perf_counter()
        geometry.UpdatePipeline(time)
        details = geometry.GetDataInformation()
        timings = {'extract_seconds': perf_counter()-start,
                   'points': details.GetNumberOfPoints(), 'cells': details.GetNumberOfCells(),
                   'geometry_kib': details.GetMemorySize()}
        start = perf_counter()
        pv.Render(view)
        timings['render_seconds'] = perf_counter()-start
        error = ''
        try:
            window = vtkPVProcessWindow.GetRenderWindow()
            window.MakeCurrent()
            if window.GetClassName() != 'vtkEGLRenderWindow' or EGL().current_uuid() != expected_uuid:
                raise RuntimeError('Solver CUDA and actual EGL device do not match')
        except Exception as exc:
            error = str(exc)
        require_collective(error)
        start = perf_counter()
        pv.SaveScreenshot(str(picture), view, ImageResolution=[1280, 960])
        # SaveScreenshot also re-renders/captures; do not label this pure JPEG encoding.
        timings['screenshot_jpeg_seconds'] = perf_counter()-start
        start = perf_counter()
        error = ''
        if rank == 0:
            try:
                with Image.open(picture) as image:
                    image.convert('RGB').save(picture.with_suffix('.eps'))
            except Exception as exc:
                error = str(exc)
        require_collective(error)
        timings['eps_seconds'] = perf_counter()-start
        profile['products'][name] = timings
    manager = servermanager.ProxyManager()
    profile['proxies'] = {group: len(manager.GetProxiesInGroup(group)) for group in
                          ('sources', 'views', 'representations', 'lookup_tables', 'scalar_bars')}
    profile['execute_seconds'] = perf_counter()-frame_start
    profiles.append(profile)
    frames.append([step, time])
    (output/f'demo_rank{rank}.json').write_text(json.dumps(
        {'frames': frames, 'profiles': profiles, 'finalized': False}))


def catalyst_finalize():
    for _, _, view, display, bar in products:
        pv.Delete(bar)
        pv.Delete(display)
        pv.Delete(view)
    products.clear()
    for proxy in pipeline_objects:
        pv.Delete(proxy)
    pipeline_objects.clear()
    for proxy in owned_colors:
        pv.Delete(proxy)
    owned_colors.clear()
    (output/f'demo_rank{rank}.json').write_text(json.dumps(
        {'frames': frames, 'profiles': profiles, 'finalized': True}))
