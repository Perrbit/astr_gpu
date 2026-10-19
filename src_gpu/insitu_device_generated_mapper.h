#ifndef ASTR_INSITU_DEVICE_GENERATED_MAPPER_H
#define ASTR_INSITU_DEVICE_GENERATED_MAPPER_H

#include "insitu_device_vtk_mapper.h"
#include <vtkOpenGLRenderWindow.h>
#include <functional>
#include <utility>

namespace astr_insitu {
// Opt-in component candidate. VTK owns GL storage; CUDA registration belongs here.
class MappedVtkBuffer {
public:
  MappedVtkBuffer()=default;
  MappedVtkBuffer(const MappedVtkBuffer&)=delete;
  MappedVtkBuffer& operator=(const MappedVtkBuffer&)=delete;
  ~MappedVtkBuffer() {
    if(mapped) cudaGraphicsUnmapResources(1,&resource);
    if(resource) cudaGraphicsUnregisterResource(resource);
  }
  void prepare(vtkOpenGLBufferObject* buffer,std::size_t bytes,
               vtkOpenGLBufferObject::ObjectType type) {
    if(!resource || capacity<bytes || handle!=buffer->GetHandle()) {
      release();
      const auto limit=static_cast<std::size_t>(std::numeric_limits<GLsizeiptr>::max());
      if(bytes>limit || bytes/4>limit-bytes)
        throw std::overflow_error("Generated display buffer exceeds GLsizeiptr");
      if(!buffer->Allocate(bytes+bytes/4,type,vtkOpenGLBufferObject::DynamicDraw))
        throw std::runtime_error(buffer->GetError());
      handle=buffer->GetHandle();capacity=buffer->GetSize();
      require_cuda_graphics(cudaGraphicsGLRegisterBuffer(&resource,handle,
        cudaGraphicsRegisterFlagsWriteDiscard),"register generated VTK buffer");
    }
  }
  void* map() {
    require_cuda_graphics(cudaGraphicsMapResources(1,&resource),"map generated VTK buffer");
    mapped=true;
    void* pointer=nullptr;std::size_t bytes=0;
    require_cuda_graphics(cudaGraphicsResourceGetMappedPointer(&pointer,&bytes,resource),
      "get generated VTK buffer");
    if(!pointer || bytes<capacity) throw std::runtime_error("Generated VTK extent mismatch");
    return pointer;
  }
  void unmap() {
    if(!mapped) return;
    require_cuda_graphics(cudaGraphicsUnmapResources(1,&resource),"unmap generated VTK buffer");
    mapped=false;
  }
  void release() {
    unmap();
    if(resource) {
      require_cuda_graphics(cudaGraphicsUnregisterResource(resource),"unregister generated VTK buffer");
      resource=nullptr;
    }
    handle=0;capacity=0;
  }
private:
  cudaGraphicsResource* resource=nullptr;
  bool mapped=false;
  int handle=0;
  std::size_t capacity=0;
};

class GeneratedDeviceMapper : public DirectDeviceMapper {
public:
  using Generator=std::function<void(float*,unsigned char*,unsigned int*,cudaStream_t)>;
  static GeneratedDeviceMapper* New() {return new GeneratedDeviceMapper;}
  vtkTypeMacro(GeneratedDeviceMapper,DirectDeviceMapper);
  void set_source(const DeviceDrawView& extent,Generator generate) {
    if(!generate) throw std::invalid_argument("Generated display needs a producer");
    set_extent(extent,false);producer=std::move(generate);
  }
  void clear_source() {producer={};clear_view();}
  void ReleaseGraphicsResources(vtkWindow* window) override {
    auto context=vtkOpenGLRenderWindow::SafeDownCast(window);
    if(context) context->MakeCurrent();
    for(auto& buffer:buffers) buffer.release();
    DirectDeviceMapper::ReleaseGraphicsResources(window);
  }
protected:
  ~GeneratedDeviceMapper() override {if(ready) cudaEventDestroy(ready);}
  void upload_draw_buffers(vtkOpenGLIndexBufferObject* indices) override {
    auto group=static_cast<DirectVertexGroup*>(this->VBOs);
    auto positions=group->buffer("vertexMC"),colors=group->buffer("scalarColor");
    positions->describe(view.points,3,VTK_FLOAT);
    colors->describe(view.points,4,VTK_UNSIGNED_CHAR);
    buffers[0].prepare(positions,view.points*3*sizeof(float),vtkOpenGLBufferObject::ArrayBuffer);
    buffers[1].prepare(colors,view.points*4,vtkOpenGLBufferObject::ArrayBuffer);
    buffers[2].prepare(indices,view.cells*view.arity*sizeof(unsigned int),
                       vtkOpenGLBufferObject::ElementArrayBuffer);
    if(!ready) require_cuda_graphics(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),
                                   "create generated display event");
    try {
      auto xyz=static_cast<float*>(buffers[0].map());
      auto rgba=static_cast<unsigned char*>(buffers[1].map());
      auto connectivity=static_cast<unsigned int*>(buffers[2].map());
      producer(xyz,rgba,connectivity,nullptr);
      require_cuda_graphics(cudaGetLastError(),"generate VTK buffers");
      require_cuda_graphics(cudaEventRecord(ready),"record generated display event");
      require_cuda_graphics(cudaEventSynchronize(ready),"complete generated display event");
    } catch(...) {
      for(auto& buffer:buffers) buffer.unmap();
      throw;
    }
    for(auto& buffer:buffers) buffer.unmap();
    positions->generated();colors->generated();group->Modified();
  }
private:
  Generator producer;
  MappedVtkBuffer buffers[3];
  cudaEvent_t ready=nullptr;
};
} // namespace astr_insitu
#endif
