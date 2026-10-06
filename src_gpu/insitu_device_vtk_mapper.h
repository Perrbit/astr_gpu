#ifndef ASTR_INSITU_DEVICE_VTK_MAPPER_H
#define ASTR_INSITU_DEVICE_VTK_MAPPER_H

#include "insitu_graphics_interop.h"
#include <vtkActor.h>
#include <vtkNew.h>
#include <vtkMath.h>
#include <vtkOpenGLHelper.h>
#include <vtkOpenGLIndexBufferObject.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLVertexBufferObject.h>
#include <vtkOpenGLVertexBufferObjectGroup.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace astr_insitu {
class DirectVertexBuffer : public vtkOpenGLVertexBufferObject {
public:
  static DirectVertexBuffer* New() {return new DirectVertexBuffer;}
  vtkTypeMacro(DirectVertexBuffer,vtkOpenGLVertexBufferObject);
  void upload(const void* pointer,std::size_t count,unsigned int components,int type) {
    if(!count || count>std::numeric_limits<int>::max())
      throw std::invalid_argument("Invalid direct device vertex extent");
    this->SetDataType(type);
    this->NumberOfComponents=components;
    this->NumberOfTuples=count;
    this->Stride=components*this->DataTypeSize;
    this->SetCoordShiftAndScaleMethod(DISABLE_SHIFT_SCALE);
    if(!this->UploadDevice(pointer,count*this->Stride,ArrayBuffer))
      throw std::runtime_error(this->GetError());
    this->Modified();
    this->UploadTime.Modified();
  }
};

class DirectVertexGroup : public vtkOpenGLVertexBufferObjectGroup {
public:
  static DirectVertexGroup* New() {return new DirectVertexGroup;}
  vtkTypeMacro(DirectVertexGroup,vtkOpenGLVertexBufferObjectGroup);
  void upload(const char* name,const void* pointer,std::size_t count,unsigned int components,int type) {
    auto& buffer=this->UsedVBOs[name];
    if(!buffer) buffer=DirectVertexBuffer::New();
    static_cast<DirectVertexBuffer*>(buffer)->upload(pointer,count,components,type);
    this->Modified();
  }
};

// Standard ParaView shaders/cameras consume direct GL buffers. The empty VTK
// input carries no geometry; the caller retains the immutable device view.
class DirectDeviceMapper : public vtkOpenGLPolyDataMapper {
public:
  static DirectDeviceMapper* New() {return new DirectDeviceMapper;}
  vtkTypeMacro(DirectDeviceMapper,vtkOpenGLPolyDataMapper);
  void set_view(const DeviceDrawView& input) {
    int current=-1;
    require_cuda_graphics(cudaGetDevice(&current),"direct view device");
    if(current!=input.device || (input.arity!=2 && input.arity!=3) ||
       input.points>std::size_t(std::numeric_limits<int>::max()) ||
       input.cells>std::size_t(std::numeric_limits<int>::max())/input.arity ||
       (input.cells && !input.points) ||
       (input.points && (!input.positions || !input.colors || !input.indices)))
      throw std::invalid_argument("Invalid direct device view");
    if(input.points) for(int d=0;d<3;++d)
      if(!std::isfinite(input.bounds[2*d]) || !std::isfinite(input.bounds[2*d+1]) ||
         input.bounds[2*d]>input.bounds[2*d+1])
        throw std::invalid_argument("Invalid direct device bounds");
    this->view=input;
    this->Modified();
  }

protected:
  DirectDeviceMapper() {
    this->VBOs->Delete();
    this->VBOs=DirectVertexGroup::New();
    vtkNew<vtkPolyData> empty;
    vtkNew<vtkPoints> points;
    empty->SetPoints(points);
    this->SetInputData(empty);
    this->ScalarVisibilityOff();
    // No VTK input cells participate in the legacy geometry timing query.
    this->TimerQueryCounter=1;
  }
  void ComputeBounds() override {
    if(!view.points) {vtkMath::UninitializeBounds(this->Bounds);return;}
    std::copy(view.bounds,view.bounds+6,this->Bounds);
  }
  bool GetNeedToRebuildBufferObjects(vtkRenderer*,vtkActor* actor) override {
    return this->GetMTime()>this->VBOBuildTime || actor->GetProperty()->GetMTime()>this->VBOBuildTime;
  }
  void BuildBufferObjects(vtkRenderer* renderer,vtkActor* actor) override {
    if(renderer->GetSelector() || this->GetNumberOfClippingPlanes() ||
       actor->GetProperty()->GetEdgeVisibility() ||
       actor->GetProperty()->GetRepresentation()!=VTK_SURFACE || !this->ExtraAttributes.empty())
      throw std::runtime_error("Unsupported direct device rendering configuration");
    for(int i=PrimitiveStart;i<PrimitiveEnd;++i) this->Primitives[i].IBO->IndexCount=0;
    if(view.cells) {
      auto group=static_cast<DirectVertexGroup*>(this->VBOs);
      group->upload("vertexMC",view.positions,view.points,3,VTK_FLOAT);
      group->upload("scalarColor",view.colors,view.points,4,VTK_UNSIGNED_CHAR);
      const auto primitive=view.arity==3?PrimitiveTris:PrimitiveLines;
      auto buffer=this->Primitives[primitive].IBO;
      if(!buffer->UploadDevice(view.indices,view.cells*view.arity*sizeof(unsigned int),
           vtkOpenGLBufferObject::ElementArrayBuffer))
        throw std::runtime_error(buffer->GetError());
      buffer->IndexCount=view.cells*view.arity;
    }
    this->HaveCellScalars=false;
    this->HaveCellNormals=false;
    this->VBOBuildTime.Modified();
  }

private:
  DeviceDrawView view;
};
} // namespace astr_insitu
#endif
