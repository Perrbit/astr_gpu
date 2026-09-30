"""Read the actual current EGL display identity; no device selection."""
import ctypes as c
import uuid

class EGL:
    def __init__(self):
        self.lib = c.CDLL("libEGL.so.1")
        self.lib.eglGetProcAddress.argtypes = [c.c_char_p]
        self.lib.eglGetProcAddress.restype = c.c_void_p
        self.query = self.function("eglQueryDevicesEXT", c.c_uint, c.c_int,
                                   c.POINTER(c.c_void_p), c.POINTER(c.c_int))
        self.string = self.function("eglQueryDeviceStringEXT", c.c_char_p, c.c_void_p, c.c_int)
        self.binary = self.function("eglQueryDeviceBinaryEXT", c.c_uint, c.c_void_p,
                                    c.c_int, c.c_int, c.c_void_p, c.POINTER(c.c_int))

    def function(self, name, result, *args):
        address = self.lib.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"EGL function unavailable: {name}")
        return c.CFUNCTYPE(result, *args)(address)

    def identity(self, device):
        extensions = (self.string(device, 0x3055) or b"").decode().split()
        # Multiple drivers can enumerate one physical GPU; select the CUDA-capable driver.
        if not {"EGL_NV_device_cuda", "EGL_EXT_device_persistent_id"}.issubset(extensions):
            return None
        value, length = (c.c_ubyte * 16)(), c.c_int()
        if not self.binary(device, 0x335C, 16, value, c.byref(length)) or length.value != 16:
            raise RuntimeError("Cannot query CUDA-capable EGL device UUID")
        return str(uuid.UUID(bytes=bytes(value)))

    def devices(self):
        count = c.c_int()
        if not self.query(0, None, c.byref(count)) or count.value <= 0:
            raise RuntimeError("EGL device enumeration failed")
        devices = (c.c_void_p * count.value)()
        if not self.query(len(devices), devices, c.byref(count)):
            raise RuntimeError("EGL device enumeration failed")
        return [{"egl_index": index, "uuid": self.identity(devices[index])}
                for index in range(count.value)]

    def current_uuid(self):
        self.lib.eglGetCurrentDisplay.restype = c.c_void_p
        display = self.lib.eglGetCurrentDisplay()
        if not display:
            raise RuntimeError("No current EGL display")
        query = self.function("eglQueryDisplayAttribEXT", c.c_uint, c.c_void_p,
                              c.c_int, c.POINTER(c.c_ssize_t))
        device = c.c_ssize_t()
        if not query(display, 0x322C, c.byref(device)):
            raise RuntimeError("Cannot query the actual EGL display device")
        identity = self.identity(c.c_void_p(device.value))
        if identity is None:
            raise RuntimeError("Actual EGL display is not a supported CUDA-capable device")
        return identity
