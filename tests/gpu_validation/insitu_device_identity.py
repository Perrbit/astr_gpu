"""Local Linux NVIDIA dependency probe; not a production device-binding policy."""

import ctypes as c
import json
import os
from pathlib import Path
import sys
import uuid


def checked(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed: CUDA status {code}")


def cuda_devices():
    driver = c.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [c.c_uint]
    checked(driver.cuInit(0), "cuInit")
    driver.cuDeviceGetCount.argtypes = [c.POINTER(c.c_int)]
    count = c.c_int()
    checked(driver.cuDeviceGetCount(c.byref(count)), "cuDeviceGetCount")
    driver.cuDeviceGet.argtypes = [c.POINTER(c.c_int), c.c_int]
    driver.cuDeviceGetUuid_v2.argtypes = [c.c_void_p, c.c_int]
    driver.cuDeviceGetPCIBusId.argtypes = [c.c_void_p, c.c_int, c.c_int]
    devices = []
    for ordinal in range(count.value):
        device, value, pci = c.c_int(), (c.c_ubyte * 16)(), c.create_string_buffer(32)
        checked(driver.cuDeviceGet(c.byref(device), ordinal), "cuDeviceGet")
        checked(driver.cuDeviceGetUuid_v2(value, device), "cuDeviceGetUuid_v2")
        checked(driver.cuDeviceGetPCIBusId(pci, len(pci), device), "cuDeviceGetPCIBusId")
        devices.append({"cuda_ordinal": ordinal, "uuid": str(uuid.UUID(bytes=bytes(value))),
                        "pci_bus_id": pci.value.decode()})
    return devices


sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/insitu'))
from egl_identity import EGL


def match_device(cuda, egl):
    matches = [item for item in egl if item["uuid"] == cuda["uuid"]]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one EGL device for CUDA UUID {cuda['uuid']}, found {len(matches)}")
    return {**cuda, **matches[0]}


def main():
    devices = cuda_devices()
    if not devices:
        raise RuntimeError("No visible CUDA devices")
    egl = EGL().devices()
    if len(sys.argv) == 1:
        print(json.dumps([match_device(device, egl) for device in devices], indent=2))
        return
    if sys.argv[1] != "--" or len(sys.argv) < 3:
        raise RuntimeError("Usage: insitu_device_identity.py [-- probe pipeline backend]")
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    if local_rank < 0:
        raise RuntimeError("Invalid local rank")
    selection = match_device(devices[local_rank % len(devices)], egl)
    output = Path(os.environ["ASTR_PROBE_OUTPUT"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    (output / f"mapping{rank}.json").write_text(json.dumps(selection, indent=2))
    os.environ["VTK_EGL_DEVICE_INDEX"] = str(selection["egl_index"])
    os.environ["ASTR_PROBE_EXPECTED_UUID"] = selection["uuid"]
    os.environ["ASTR_PROBE_IDENTITY_MODULE"] = str(Path(__file__).resolve().parent)
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
