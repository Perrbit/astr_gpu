"""Read native due decisions and return publication outcomes, never schedule here."""
import ctypes
import json
from functools import wraps


class ProductFailure(RuntimeError):
    """All ranks have left the failed optional product stage."""


_disabled = set()

_native = ctypes.CDLL(None)
try:
    _active = _native.astr_insitu_product_clocks_active
    _due = _native.astr_insitu_product_due
    _scene_due = _native.astr_insitu_product_scene_due
    _report = _native.astr_insitu_product_report
except AttributeError:
    _active = None
else:
    _active.argtypes = []
    _active.restype = ctypes.c_int
    for function in (_due, _scene_due):
        function.argtypes = [ctypes.c_char_p]
        function.restype = ctypes.c_int
    _report.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    _report.restype = ctypes.c_int

try:
    _disable = _native.astr_insitu_product_disable
except AttributeError:
    _disable = None
else:
    _disable.argtypes = [ctypes.c_char_p]
    _disable.restype = ctypes.c_int


def independent():
    return bool(_active and _active())


def scene_due(scene):
    if _active is not None:
        return bool(_scene_due(scene.encode('ascii')))
    return due(scene, 'image') or due(scene, 'geometry')


def due(scene, kind):
    name = scene + '.' + kind
    return name not in _disabled and (_active is None or bool(_due(name.encode('ascii'))))


def disable(scene, kind, step, time, reason):
    name = scene + '.' + kind
    if name not in _disabled:
        if _disable and _disable(name.encode('ascii')):
            raise RuntimeError('Cannot retire failed native product: ' + name)
        _disabled.add(name)
        print('ASTR_INSITU_PRODUCT_DISABLED ' + json.dumps(dict(
            product=name, step=step, time=time, reason=str(reason))), flush=True)
    return dict(status='disabled', product=name, step=step, time=time, reason=str(reason))


def enabled(scene, kind):
    name = scene + '.' + kind
    function = getattr(_native, 'astr_insitu_product_enabled', None)
    if function is not None:
        function.argtypes = [ctypes.c_char_p]
        function.restype = ctypes.c_int
    return name not in _disabled and (function is None or bool(function(name.encode('ascii'))))


def isolate_image(operation):
    @wraps(operation)
    def render(name, item, step, time, points, cells):
        if not enabled(name, 'image'):
            return dict(image=dict(status='disabled', product=name+'.image'), local_points=points, local_cells=cells)
        try:
            return operation(name, item, step, time, points, cells)
        except ProductFailure as exc:
            return dict(image=disable(name, 'image', step, time, exc), local_points=points, local_cells=cells)
    return render


def collective_call(controller, operation):
    """Rendezvous after a product stage; never wrap mismatched MPI call sequences."""
    from vtkmodules.vtkCommonCore import vtkIntArray
    from vtkmodules.vtkParallelCore import vtkCommunicator
    error = ''
    result = None
    try:
        result = operation()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    local, total = vtkIntArray(), vtkIntArray()
    local.InsertNextValue(int(bool(error)))
    total.SetNumberOfTuples(1)
    controller.AllReduce(local, total, vtkCommunicator.MAX_OP)
    if total.GetValue(0):
        collective_failure(controller, error or 'Product stage failed on another rank')
    return result


def collective_failure(controller, reason):
    # Called by all ranks after consensus, not from a rank-local exception handler.
    health = getattr(_native, 'astr_insitu_product_device_health', None)
    from vtkmodules.vtkCommonCore import vtkIntArray
    from vtkmodules.vtkParallelCore import vtkCommunicator
    local, total = vtkIntArray(), vtkIntArray()
    local.InsertNextValue(int(bool(health and health())))
    total.SetNumberOfTuples(1)
    controller.AllReduce(local, total, vtkCommunicator.MAX_OP)
    if total.GetValue(0):
        raise RuntimeError('Unrecoverable CUDA error after optional product failure')
    raise ProductFailure(reason)


def report(scene, kind, receipt=None, uncovered=False):
    if isinstance(receipt, dict) and receipt['status'] == 'disabled':
        return
    if not independent():
        return
    code, phase = (-1 if uncovered else 0), 0
    if isinstance(receipt, dict) and receipt['status'] == 'missing':
        code = receipt['errno']
        phase = ('stage_jpeg', 'stage_eps', 'publish_jpeg', 'publish_eps').index(receipt['phase']) + 1
    if _report((scene + '.' + kind).encode('ascii'), code, phase):
        raise RuntimeError('Unexpected or repeated native product outcome: ' + scene + '.' + kind)


def report_uncovered(scene):
    for kind in ('image', 'geometry'):
        if due(scene, kind):
            report(scene, kind, uncovered=True)


def adaptive_info(scene, kind):
    """Read the native decision/committed clock without duplicating its logic."""
    if not independent():
        return None
    function = _native.astr_insitu_product_adaptive_info
    function.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    integers, times = (ctypes.c_int64 * 8)(), (ctypes.c_double * 4)()
    if not function((scene+'.'+kind).encode('ascii'), integers, times):
        return None
    return dict(zip(('dense', 'reason', 'target_step', 'next_step', 'success_step', 'attempt_step', 'code', 'phase'), integers)) | \
        dict(zip(('target_time', 'next_time', 'success_time', 'attempt_time'), times))


def adaptive_receipts(scenes):
    records = {}
    for scene in scenes:
        for kind in ('image', 'geometry'):
            if due(scene, kind):
                info = adaptive_info(scene, kind)
                if info is not None:
                    records[scene+'.'+kind] = info
    return records
