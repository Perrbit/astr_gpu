#ifndef ASTR_INSITU_GRAPHICS_INTEROP_H
#define ASTR_INSITU_GRAPHICS_INTEROP_H

#include <vtk_glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "../src/insitu_compact_mesh.h"

namespace astr_insitu {
inline void require_cuda_graphics(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// The owning rank must keep this buffer's OpenGL context current until release.
class GraphicsBuffer {
public:
  GraphicsBuffer() = default;
  GraphicsBuffer(const GraphicsBuffer&) = delete;
  GraphicsBuffer& operator=(const GraphicsBuffer&) = delete;
  ~GraphicsBuffer() {
    if (mapped_) cudaGraphicsUnmapResources(1, &resource_);
    if (resource_) cudaGraphicsUnregisterResource(resource_);
    if (name_) glDeleteBuffers(1, &name_);
  }

  void allocate(std::size_t bytes) {
    if (name_ || bytes == 0) throw std::invalid_argument("Invalid graphics buffer allocation");
    glGenBuffers(1, &name_);
    glBindBuffer(GL_ARRAY_BUFFER, name_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, GL_DYNAMIC_DRAW);
    if (glGetError() != GL_NO_ERROR) throw std::runtime_error("OpenGL buffer allocation failed");
    require_cuda_graphics(cudaGraphicsGLRegisterBuffer(&resource_, name_,
      cudaGraphicsRegisterFlagsWriteDiscard), "register OpenGL buffer");
    capacity_ = bytes;
  }

  void* map() {
    if (!resource_ || mapped_) throw std::logic_error("Invalid graphics buffer map");
    require_cuda_graphics(cudaGraphicsMapResources(1, &resource_), "map OpenGL buffer");
    mapped_ = true;
    void* pointer = nullptr;
    std::size_t bytes = 0;
    require_cuda_graphics(cudaGraphicsResourceGetMappedPointer(&pointer, &bytes, resource_),
      "get mapped OpenGL pointer");
    if (!pointer || bytes < capacity_) throw std::runtime_error("Mapped buffer extent mismatch");
    return pointer;
  }

  void unmap() {
    if (!mapped_) throw std::logic_error("Graphics buffer is not mapped");
    require_cuda_graphics(cudaGraphicsUnmapResources(1, &resource_), "unmap OpenGL buffer");
    mapped_ = false;
  }

  GLuint name() const {
    if (!name_ || mapped_) throw std::logic_error("OpenGL cannot consume a mapped buffer");
    return name_;
  }

  void release() {
    if (mapped_) unmap();
    if (resource_) {
      require_cuda_graphics(cudaGraphicsUnregisterResource(resource_), "unregister OpenGL buffer");
      resource_ = nullptr;
    }
    if (name_) {
      glDeleteBuffers(1, &name_);
      name_ = 0;
      capacity_ = 0;
      if (glGetError() != GL_NO_ERROR) throw std::runtime_error("OpenGL buffer release failed");
    }
  }

private:
  GLuint name_ = 0;
  cudaGraphicsResource* resource_ = nullptr;
  std::size_t capacity_ = 0;
  bool mapped_ = false;
};
} // namespace astr_insitu
#endif
