#include "insitu_graphics_interop.h"
#include <vtkEGLRenderWindow.h>
#include <vtkNew.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkOpenGLBufferObject.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

extern "C" int astr_insitu_map_current_cuda(int*, char*, int, char*, int);

namespace {
__global__ void fill_triangles(float* vertices, int frame) {
  const int vertex = threadIdx.x;
  if (vertex >= 6) return;
  const float xy[6] = {-0.75f, -0.75f, 0.75f, -0.75f, 0.f, 0.75f};
  const bool front = vertex < 3;
  const int local = vertex % 3;
  const float scale = front ? 0.5f : 1.f;
  const bool green = front && frame != 1;
  const bool blue = front && frame == 1;
  float* point = vertices + 6 * vertex;
  point[0] = scale * xy[2 * local];
  point[1] = scale * xy[2 * local + 1];
  point[2] = front ? -0.5f : 0.5f;
  point[3] = front ? 0.f : 1.f;
  point[4] = green ? 1.f : 0.f;
  point[5] = blue ? 1.f : 0.f;
}

GLuint shader(GLenum type, const char* source) {
  const GLuint result = glCreateShader(type);
  glShaderSource(result, 1, &source, nullptr);
  glCompileShader(result);
  GLint compiled = 0;
  glGetShaderiv(result, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    char message[2048]{};
    glGetShaderInfoLog(result, sizeof(message), nullptr, message);
    glDeleteShader(result);
    throw std::runtime_error(std::string("Probe shader: ") + message);
  }
  return result;
}

void probe_buffer_growth(int device) {
  constexpr int frames=100;
  constexpr std::size_t mib=1024*1024;
  constexpr std::size_t initial[3]={8*mib,4*mib,mib};
  constexpr std::size_t increment[3]={8192,4096,1024};
  void* source=nullptr;
  astr_insitu::require_cuda_graphics(cudaMalloc(&source,initial[0]+frames*increment[0]),"growth source");
  astr_insitu::require_cuda_graphics(cudaMemset(source,0,initial[0]+frames*increment[0]),"growth source fill");
  std::array<vtkSmartPointer<vtkOpenGLBufferObject>,3> buffers;
  std::size_t baseline=0,total=0,minimum=0,last=0,live=0,capacity=0;
  astr_insitu::require_cuda_graphics(cudaMemGetInfo(&baseline,&total),"growth baseline");
  minimum=baseline;
  int allocations=0;
  for(int frame=0;frame<frames;++frame) {
    live=capacity=0;
    for(int component=0;component<3;++component) {
      if(!buffers[component]) buffers[component]=vtkSmartPointer<vtkOpenGLBufferObject>::New();
      auto* buffer=buffers[component].GetPointer();
      const auto before=buffer->GetSize();
      const auto bytes=initial[component]+frame*increment[component];
      if(!buffer->UploadDevice(source,bytes,vtkOpenGLBufferObject::ArrayBuffer))
        throw std::runtime_error(buffer->GetError());
      if(buffer->GetSize()<bytes) throw std::runtime_error("Growth buffer capacity is smaller than upload");
      allocations+=buffer->GetSize()!=before;
      live+=bytes;capacity+=buffer->GetSize();
    }
    glFinish();
    astr_insitu::require_cuda_graphics(cudaDeviceSynchronize(),"growth completion");
    astr_insitu::require_cuda_graphics(cudaMemGetInfo(&last,&total),"growth sample");
    minimum=std::min(minimum,last);
    if(baseline-minimum>2*1024*mib || last<1024*mib)
      throw std::runtime_error("Growth probe exceeds existing local memory budget/reserve");
  }
  const auto final_increment=baseline>last?baseline-last:0;
  const auto peak_increment=baseline>minimum?baseline-minimum:0;
  for(auto& buffer:buffers) buffer=nullptr;
  astr_insitu::require_cuda_graphics(cudaFree(source),"release growth source");
  std::printf("ASTR_INSITU_BUFFER_GROWTH {\"cuda_device\":%d,\"frames\":%d,"
    "\"allocations\":%d,\"live_bytes\":%zu,\"capacity_bytes\":%zu,"
    "\"device_increment_bytes\":%zu,\"device_peak_increment_bytes\":%zu,\"geometry_host_bytes\":0}\n",
    device,frames,allocations,live,capacity,final_increment,peak_increment);
}
}

int main(int argc, char** argv) {
  try {
    if ((argc != 2 && argc != 3) || (std::strcmp(argv[1], "0") != 0 && std::strcmp(argv[1], "1") != 0) ||
        (argc==3 && std::strcmp(argv[2],"--growth")!=0))
      throw std::invalid_argument("Usage: insitu_graphics_interop_probe CUDA_DEVICE (0 or 1) [--growth]");
    const int device = std::atoi(argv[1]);
    astr_insitu::require_cuda_graphics(cudaSetDevice(device), "select CUDA device");
    astr_insitu::require_cuda_graphics(cudaFree(nullptr), "initialize CUDA context");
    int egl_index = -1;
    char uuid[40]{}, message[1024]{};
    if (astr_insitu_map_current_cuda(&egl_index, uuid, sizeof(uuid), message, sizeof(message)))
      throw std::runtime_error(message);
    // VTK 6.1.1 initializes its private EGL device index in the constructor.
    const std::string egl_selection = std::to_string(egl_index);
    if (setenv("VTK_EGL_DEVICE_INDEX", egl_selection.c_str(), 1) != 0)
      throw std::runtime_error("Cannot set UUID-matched EGL device selection");
    vtkNew<vtkEGLRenderWindow> window;
    vtkNew<vtkRenderer> renderer;
    window->SetDeviceIndex(egl_index);
    window->SetSize(64, 64);
    window->SetMultiSamples(0);
    window->SetShowWindow(false);
    window->AddRenderer(renderer);
    window->Render();
    window->MakeCurrent();
    const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* hardware = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    if (!vendor || !hardware || !std::strstr(vendor, "NVIDIA"))
      throw std::runtime_error("Probe requires NVIDIA EGL hardware rendering");
    const std::string hardware_name(hardware);
    unsigned int count = 0;
    int gl_device = -1;
    astr_insitu::require_cuda_graphics(cudaGLGetDevices(&count, &gl_device, 1,
      cudaGLDeviceListAll), "query OpenGL CUDA device");
    if (count != 1 || gl_device != device)
      throw std::runtime_error("OpenGL and selected CUDA devices differ");

    if(argc==3) probe_buffer_growth(device);

    const char* vertex_source =
      "#version 330 core\nlayout(location=0) in vec3 p; layout(location=1) in vec3 c;"
      "out vec3 color; void main(){gl_Position=vec4(p,1);color=c;}";
    const char* fragment_source =
      "#version 330 core\nin vec3 color; out vec4 outputColor;"
      "void main(){outputColor=vec4(color,1);}";
    const GLuint vertex = shader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = shader(GL_FRAGMENT_SHADER, fragment_source);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) throw std::runtime_error("Probe program link failed");
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    astr_insitu::GraphicsBuffer buffer;
    buffer.allocate(6 * 6 * sizeof(float));
    const GLuint reusable_buffer = buffer.name();
    std::vector<unsigned char> pixels(64 * 64 * 4);
    for (int frame = 0; frame < 3; ++frame) {
      auto* pointer = static_cast<float*>(buffer.map());
      bool rejected = false;
      try { buffer.name(); } catch (const std::logic_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Mapped graphics buffer exposed to OpenGL");
      fill_triangles<<<1, 32>>>(pointer, frame);
      astr_insitu::require_cuda_graphics(cudaGetLastError(), "launch triangle fill");
      astr_insitu::require_cuda_graphics(cudaDeviceSynchronize(), "complete triangle fill");
      buffer.unmap();
      if (buffer.name() != reusable_buffer) throw std::runtime_error("Graphics buffer was not reused");
      window->GetRenderFramebuffer()->Bind();
      glViewport(0, 0, 64, 64);
      glClearColor(0.f, 0.f, 0.f, 1.f);
      glClearDepth(1.);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);
      glDisable(GL_BLEND);
      glDisable(GL_CULL_FACE);
      glUseProgram(program);
      glBindVertexArray(vao);
      glBindBuffer(GL_ARRAY_BUFFER, buffer.name());
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
        reinterpret_cast<void*>(3 * sizeof(float)));
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
      if (glGetError() != GL_NO_ERROR) throw std::runtime_error("Probe OpenGL operation failed");
      const auto at = [&](int x, int y, int channel) { return pixels[4 * (64 * y + x) + channel]; };
      if (at(32, 32, frame == 1 ? 2 : 1) != 255 || at(32, 32, 0) != 0 ||
          at(16, 12, 0) != 255 || at(1, 1, 0) != 0 || at(1, 1, 1) != 0 || at(1, 1, 2) != 0)
        throw std::runtime_error("Analytic triangle color/depth/background comparison failed");
    }
    buffer.release();
    if (glIsBuffer(reusable_buffer)) throw std::runtime_error("OpenGL buffer survived release");
    glBindVertexArray(0);
    glUseProgram(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    window->Finalize();
    std::printf("ASTR_INSITU_GRAPHICS_PROBE {\"cuda_device\":%d,\"gl_cuda_device\":%d,"
      "\"egl_index\":%d,\"uuid\":\"%s\",\"vtk\":\"%s\",\"renderer\":\"%s\","
      "\"frames\":3,\"registered_buffers\":1,\"geometry_host_bytes\":0,"
      "\"image_host_bytes\":49152,\"pixels_passed\":true,\"released\":true}\n",
      device, gl_device, egl_index, uuid, vtkVersion::GetVTKVersion(), hardware_name.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ASTR graphics interop probe failed: %s\n", error.what());
    return 1;
  }
}
