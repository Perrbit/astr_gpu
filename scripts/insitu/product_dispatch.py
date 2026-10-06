"""Read native due decisions and return publication outcomes, never schedule here."""
import ctypes

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


def independent():
    return bool(_active and _active())


def scene_due(scene):
    return not independent() or bool(_scene_due(scene.encode('ascii')))


def due(scene, kind):
    return not independent() or bool(_due((scene + '.' + kind).encode('ascii')))


def report(scene, kind, receipt=None, uncovered=False):
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
