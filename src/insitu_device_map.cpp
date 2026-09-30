#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {
struct Libraries {
  void* cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  void* egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
  ~Libraries() { if (egl) dlclose(egl); if (cuda) dlclose(cuda); }
};
bool has_extension(const char* extensions, const char* name)
{
  if (!extensions) return false;
  const std::string list = std::string(" ") + extensions + " ";
  return list.find(std::string(" ") + name + " ") != std::string::npos;
}
}

// Linux NVIDIA/EGL mapping only. Never creates, changes, or guesses a CUDA context.
extern "C" int astr_insitu_map_current_cuda(int* index, char* uuid, int uuid_size,
                                            char* message, int message_size)
{
  if (!index || !uuid || uuid_size < 37 || !message || message_size < 1) return 1;
  *index = -1;
  uuid[0] = message[0] = '\0';
  auto fail = [&](const char* reason) {
    std::snprintf(message, message_size, "%s", reason);
    return 1;
  };
  try {
    static Libraries libraries;
    if (!libraries.cuda || !libraries.egl) return fail("CUDA driver or EGL library unavailable");
    // CUDA driver ABI: CUdevice/CUresult are int, CUuuid contains 16 bytes.
    struct Uuid { unsigned char bytes[16]; } wanted{};
    auto current = reinterpret_cast<int (*)(int*)>(dlsym(libraries.cuda, "cuCtxGetDevice"));
    auto get_uuid = reinterpret_cast<int (*)(Uuid*, int)>(dlsym(libraries.cuda, "cuDeviceGetUuid_v2"));
    auto get_proc = reinterpret_cast<decltype(&eglGetProcAddress)>(dlsym(libraries.egl, "eglGetProcAddress"));
    if (!current || !get_uuid || !get_proc) return fail("required CUDA/EGL identity API unavailable");
    int device = -1;
    if (current(&device) != 0) return fail("no current solver CUDA context");
    if (get_uuid(&wanted, device) != 0) return fail("cannot read current CUDA device UUID");
    auto query = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(get_proc("eglQueryDevicesEXT"));
    auto string = reinterpret_cast<PFNEGLQUERYDEVICESTRINGEXTPROC>(get_proc("eglQueryDeviceStringEXT"));
    auto binary = reinterpret_cast<PFNEGLQUERYDEVICEBINARYEXTPROC>(get_proc("eglQueryDeviceBinaryEXT"));
    if (!query || !string || !binary) return fail("EGL device identity extensions unavailable");
    EGLint count = 0;
    if (!query(0, nullptr, &count) || count <= 0) return fail("cannot enumerate EGL devices");
    std::vector<EGLDeviceEXT> devices(count);
    EGLint found = 0;
    if (!query(count, devices.data(), &found) || found < 0 || found > count)
      return fail("inconsistent EGL device enumeration");
    int selected = -1, matches = 0;
    for (EGLint i = 0; i < found; ++i) {
      const char* extensions = string(devices[i], EGL_EXTENSIONS);
      if (!has_extension(extensions, "EGL_NV_device_cuda") ||
          !has_extension(extensions, "EGL_EXT_device_persistent_id")) continue;
      Uuid candidate{};
      EGLint size = 0;
      if (!binary(devices[i], EGL_DEVICE_UUID_EXT, sizeof(candidate), &candidate, &size) || size != 16)
        return fail("cannot query CUDA-capable EGL device UUID");
      if (std::memcmp(&candidate, &wanted, sizeof(wanted)) == 0) { selected = i; ++matches; }
    }
    if (matches != 1) return fail("current CUDA UUID has no unique EGL match (MIG unsupported)");
    const auto* b = wanted.bytes;
    std::snprintf(uuid, uuid_size,
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
    *index = selected;
    return 0;
  } catch (const std::exception& error) {
    return fail(error.what());
  } catch (...) {
    return fail("unexpected device mapping failure");
  }
}
