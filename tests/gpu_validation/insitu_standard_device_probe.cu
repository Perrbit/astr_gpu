#include "insitu_device_array.h"
#include "insitu_device_vtk_mapper.h"
#include <viskores/cont/Initialize.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkConduitArrayUtilities.h>
#include <vtkEGLRenderWindow.h>
#include <vtkInformation.h>
#include <vtkNew.h>
#include <vtkOpenGLBufferObject.h>
#include <vtkOpenGLFramebufferObject.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVersion.h>
#include <vtkmDataArray.h>
#include <vtk_glad.h>
#include <cuda_gl_interop.h>
#include <catalyst.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

extern "C" int astr_insitu_map_current_cuda(int*, char*, int, char*, int);

namespace {
using Point=viskores::Vec3f_32;
using Color=viskores::Vec<viskores::UInt8,4>;
template<class T> struct Allocation {
  T* pointer=nullptr;
  explicit Allocation(int count) {
    if(cudaMalloc(&pointer,count*sizeof(T))!=cudaSuccess)
      throw std::runtime_error("Standard device probe allocation failed");
  }
  ~Allocation() {if(pointer) cudaFree(pointer);}
  Allocation(const Allocation&)=delete;
  Allocation& operator=(const Allocation&)=delete;
};
struct Node {
  conduit_node* pointer=conduit_node_create();
  ~Node() {conduit_node_destroy(pointer);}
  Node()=default;
  Node(const Node&)=delete;
  Node& operator=(const Node&)=delete;
};
__global__ void fill(Point* points,Color* colors,viskores::UInt32* indices,int frame) {
  const int i=threadIdx.x;
  if(i>=6) return;
  const float xy[6]={-.75f,-.75f,.75f,-.75f,0.f,.75f};
  const bool front=i<3;
  const float scale=front?.5f:1.f;
  points[i]=Point(scale*xy[2*(i%3)],scale*xy[2*(i%3)+1],front?.5f:-.5f);
  colors[i]=Color(front?0:255,front && frame!=1?255:0,front && frame==1?255:0,255);
  indices[i]=i;
}

__global__ void fill_large(Point* points,Color* colors,viskores::UInt32* indices,
                          int point_count,int index_count,int frame) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<point_count) {
    const float xy[6]={-.75f,-.75f,.75f,-.75f,0.f,.75f};
    const bool front=i<3;
    const float scale=front?.5f:1.f;
    if(i<6) {
      points[i]=Point(scale*xy[2*(i%3)],scale*xy[2*(i%3)+1],front?.5f:-.5f);
      colors[i]=Color(front?0:255,front && frame!=1?255:0,front && frame==1?255:0,255);
    } else {
      // Valid triangles outside the fixed camera keep fragment work bounded.
      points[i]=Point(2.f+(i%3==1?.001f:0.f),i%3==2?.001f:0.f,0.f);
      colors[i]=Color(255,0,0,255);
    }
  }
  if(i<index_count) indices[i]=i%point_count;
}

void probe_large_mesh(vtkEGLRenderWindow* window,vtkRenderer* renderer,int device,
                      bool direct,bool growing) {
  constexpr int frames=100,initial=270000,increment=198;
  const int maximum=initial+(growing?increment*(frames-1):0);
  struct FrameSource {
    Allocation<Point> points;
    Allocation<Color> colors;
    Allocation<viskores::UInt32> indices;
    explicit FrameSource(int count):points(count),colors(count),indices(6*count) {}
  };
  std::unique_ptr<FrameSource> previous;
  vtkSmartPointer<vtkOpenGLPolyDataMapper> mapper;
  if(direct) mapper=vtkSmartPointer<astr_insitu::DirectDeviceMapper>::Take(astr_insitu::DirectDeviceMapper::New());
  else mapper=vtkSmartPointer<vtkOpenGLPolyDataMapper>::New();
  mapper->ScalarVisibilityOff();
  vtkNew<vtkActor> actor;actor->SetMapper(mapper);actor->GetProperty()->LightingOff();
  renderer->AddActor(actor);
  auto* camera=renderer->GetActiveCamera();
  camera->SetPosition(0.,0.,3.);camera->SetFocalPoint(0.,0.,0.);
  camera->ParallelProjectionOn();camera->SetParallelScale(1.);camera->SetClippingRange(1.,5.);
  const double bounds[6]={-.75,2.002,-.75,.75,-.5,.5};
  vtkNew<vtkUnsignedCharArray> image;
  std::vector<unsigned char> pixels(64*64*4);
  image->SetNumberOfComponents(4);image->SetArray(pixels.data(),pixels.size(),1);
  std::size_t first_free=0,last_free=0,minimum=0,total=0;
  for(int frame=0;frame<frames;++frame) {
    const int count=initial+(growing?increment*frame:0),index_count=6*count;
    auto current=std::make_unique<FrameSource>(count);
    auto& points=current->points;
    auto& colors=current->colors;
    auto& indices=current->indices;
    fill_large<<<(index_count+255)/256,256>>>(points.pointer,colors.pointer,indices.pointer,count,index_count,frame);
    astr_insitu::synchronize_device_stage("large graphics fill");
    Node point_node,color_node,index_node;
    const char* axes[]={"x","y","z"};
    for(int d=0;d<3;++d) conduit_node_set_path_external_float32_ptr_detailed(
      point_node.pointer,axes[d],reinterpret_cast<float*>(points.pointer),count,
      d*sizeof(float),sizeof(Point),sizeof(float),0);
    const char* channels[]={"r","g","b","a"};
    for(int d=0;d<4;++d) conduit_node_set_path_external_uint8_ptr_detailed(
      color_node.pointer,channels[d],reinterpret_cast<unsigned char*>(colors.pointer),count,
      d,sizeof(Color),sizeof(unsigned char),0);
    conduit_node_set_external_uint32_ptr(index_node.pointer,indices.pointer,index_count);
    auto point_array=vtkConduitArrayUtilities::MCArrayToVTKArray(point_node.pointer);
    auto color_array=vtkConduitArrayUtilities::MCArrayToVTKArray(color_node.pointer);
    auto cells=vtkConduitArrayUtilities::MCArrayToVTKCellArray(index_count,VTK_TRIANGLE,3,index_node.pointer);
    if(!point_array || !color_array || !cells ||
       point_array->GetMemorySpace()!=vtkDataArray::CudaDeviceMemory ||
       color_array->GetMemorySpace()!=vtkDataArray::CudaDeviceMemory ||
       cells->GetConnectivityArray()->GetDeviceVoidPointer(0)!=indices.pointer)
      throw std::runtime_error("Large graphics probe lost external CUDA geometry");
    vtkNew<vtkPoints> vtk_points;vtk_points->SetData(point_array);
    color_array->SetName("_astr_display_rgba");
    vtkNew<vtkPolyData> mesh;mesh->SetPoints(vtk_points);mesh->SetPolys(cells);
    mesh->GetPointData()->AddArray(color_array);
    mesh->GetInformation()->Set(vtkDataObject::BOUNDING_BOX(),bounds,6);
    if(direct) {
      astr_insitu::DeviceDrawView draw;
      draw.positions=reinterpret_cast<float*>(points.pointer);
      draw.colors=reinterpret_cast<unsigned char*>(colors.pointer);
      draw.indices=indices.pointer;draw.points=count;draw.cells=index_count/3;draw.device=device;
      std::copy(bounds,bounds+6,draw.bounds);
      static_cast<astr_insitu::DirectDeviceMapper*>(mapper.GetPointer())->set_view(draw);
    } else mapper->SetInputData(mesh);
    nvtxRangePushA("ASTR_INSITU_LARGE_GRAPHICS_FRAME");
    window->Render();
    if(!window->GetRGBACharPixelData(0,0,63,63,0,image))
      throw std::runtime_error("Large graphics screenshot failed");
    nvtxRangePop();
    const auto at=[&](int x,int y,int channel){return pixels[4*(64*y+x)+channel];};
    if(glGetError()!=GL_NO_ERROR || at(32,32,frame==1?2:1)!=255 || at(32,32,0)!=0 ||
       at(16,12,0)!=255 || at(1,1,0)!=0 || at(1,1,1)!=0 || at(1,1,2)!=0)
      throw std::runtime_error("Large graphics color/depth/background comparison failed");
    previous=std::move(current);
    astr_insitu::require_cuda_graphics(cudaMemGetInfo(&last_free,&total),"large graphics memory");
    if(frame==0) first_free=minimum=last_free;
    minimum=std::min(minimum,last_free);
    if(first_free-minimum>2ULL*1024*1024*1024 || last_free<1024ULL*1024*1024)
      throw std::runtime_error("Large graphics probe exceeds existing local memory budget/reserve");
  }
  const auto retained=first_free>last_free?first_free-last_free:0;
  const auto peak=first_free-minimum;
  mapper->ReleaseGraphicsResources(window);renderer->RemoveActor(actor);
  astr_insitu::require_cuda_graphics(cudaMemGetInfo(&last_free,&total),"large graphics release");
  std::printf("ASTR_INSITU_LARGE_GRAPHICS {\"cuda_device\":%d,\"frames\":%d,"
    "\"direct\":%s,\"growing\":%s,\"final_points\":%d,\"live_bytes\":%zu,"
    "\"retained_growth_bytes\":%zu,\"peak_growth_bytes\":%zu,\"geometry_host_bytes\":0,"
    "\"image_host_bytes\":%zu,\"pixels_passed\":true}\n",
    device,frames,direct?"true":"false",growing?"true":"false",maximum,
    static_cast<std::size_t>(maximum)*(sizeof(Point)+sizeof(Color)+6*sizeof(viskores::UInt32)),
    retained,peak,frames*pixels.size());
}
}

int main(int argc,char** argv) {
  try {
    if((argc!=2 && argc!=3) || (std::strcmp(argv[1],"0") && std::strcmp(argv[1],"1")) ||
       (argc==3 && std::strcmp(argv[2],"--conduit") && std::strcmp(argv[2],"--direct") &&
        std::strcmp(argv[2],"--large-growth") && std::strcmp(argv[2],"--large-fixed") &&
        std::strcmp(argv[2],"--direct-large-growth") && std::strcmp(argv[2],"--direct-large-fixed")))
      throw std::invalid_argument("Usage: insitu_standard_device_probe CUDA_DEVICE [--conduit|--direct|--large-growth|--large-fixed|--direct-large-growth|--direct-large-fixed]");
    const bool conduit=argc==3 && !std::strcmp(argv[2],"--conduit");
    const bool direct=argc==3 && !std::strcmp(argv[2],"--direct");
    const int device=std::atoi(argv[1]);
    if(cudaSetDevice(device)!=cudaSuccess) throw std::runtime_error("CUDA selection failed");
    viskores::cont::Initialize(argc,argv);
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    int egl_index=-1;
    char uuid[40]{},message[1024]{};
    if(astr_insitu_map_current_cuda(&egl_index,uuid,sizeof(uuid),message,sizeof(message)))
      throw std::runtime_error(message);
    if(setenv("VTK_EGL_DEVICE_INDEX",std::to_string(egl_index).c_str(),1) ||
       setenv("ASTR_VTK_STRICT_DEVICE_ACCESS","1",1))
      throw std::runtime_error("Cannot configure strict CUDA/EGL identity");
    vtkNew<vtkEGLRenderWindow> window;
    vtkNew<vtkRenderer> renderer;
    window->SetSize(64,64);window->SetMultiSamples(0);window->SetShowWindow(false);
    window->AddRenderer(renderer);renderer->SetBackground(0.,0.,0.);
    window->Render();window->MakeCurrent();
    unsigned int count=0;int gl_device=-1;
    if(cudaGLGetDevices(&count,&gl_device,1,cudaGLDeviceListAll)!=cudaSuccess ||
       count!=1 || gl_device!=device) throw std::runtime_error("CUDA/EGL device mismatch");
    if(argc==3 && std::strstr(argv[2],"large")) {
      probe_large_mesh(window,renderer,device,std::strstr(argv[2],"direct")!=nullptr,
                       std::strstr(argv[2],"growth")!=nullptr);
      window->Finalize();
      return 0;
    }
    Allocation<Point> points(6);
    Allocation<Color> colors(6);
    Allocation<viskores::UInt32> indices(6);
    fill<<<1,32>>>(points.pointer,colors.pointer,indices.pointer,0);
    astr_insitu::synchronize_device_stage("standard probe fill");
    auto point_handle=astr_insitu::borrow_cuda_array(points.pointer,6);
    auto color_handle=astr_insitu::borrow_cuda_array(colors.pointer,6);
    auto index_handle=astr_insitu::borrow_cuda_array(indices.pointer,6);
    vtkSmartPointer<vtkDataArray> point_array=vtkSmartPointer<vtkmDataArray<float>>::Take(make_vtkmDataArray(point_handle));
    vtkSmartPointer<vtkDataArray> color_array=vtkSmartPointer<vtkmDataArray<unsigned char>>::Take(make_vtkmDataArray(color_handle));
    vtkSmartPointer<vtkDataArray> index_array=vtkSmartPointer<vtkmDataArray<unsigned int>>::Take(make_vtkmDataArray(index_handle));
    Node point_node,color_node,index_node;
    if(conduit) {
      const char* axes[]={"x","y","z"};
      for(int d=0;d<3;++d) conduit_node_set_path_external_float32_ptr_detailed(
        point_node.pointer,axes[d],reinterpret_cast<float*>(points.pointer),6,
        d*sizeof(float),sizeof(Point),sizeof(float),0);
      const char* components[]={"r","g","b","a"};
      for(int d=0;d<4;++d) conduit_node_set_path_external_uint8_ptr_detailed(
        color_node.pointer,components[d],reinterpret_cast<unsigned char*>(colors.pointer),6,
        d,sizeof(Color),sizeof(unsigned char),0);
      conduit_node_set_external_uint32_ptr(index_node.pointer,indices.pointer,6);
      point_array=vtkConduitArrayUtilities::MCArrayToVTKArray(point_node.pointer);
      color_array=vtkConduitArrayUtilities::MCArrayToVTKArray(color_node.pointer);
      index_array=vtkConduitArrayUtilities::MCArrayToVTKArray(index_node.pointer);
      if(!point_array || !color_array || !index_array ||
         point_array->GetMemorySpace()!=vtkDataArray::CudaDeviceMemory ||
         color_array->GetMemorySpace()!=vtkDataArray::CudaDeviceMemory ||
         index_array->GetMemorySpace()!=vtkDataArray::CudaDeviceMemory)
        throw std::runtime_error("Conduit did not retain CUDA arrays");
    }
    color_array->SetName("_astr_display_rgba");
    vtkNew<vtkPoints> vtk_points;vtk_points->SetData(point_array);
    vtkSmartPointer<vtkCellArray> cells=vtkSmartPointer<vtkCellArray>::New();
    if(conduit) cells=vtkConduitArrayUtilities::MCArrayToVTKCellArray(6,VTK_TRIANGLE,3,index_node.pointer);
    if(!cells) throw std::runtime_error("Conduit device connectivity conversion failed");
    if(!conduit && !cells->SetData(3,index_array))
      throw std::runtime_error("Cannot attach standard device connectivity");
    if(cells->GetConnectivityArray()->GetDataType()!=VTK_UNSIGNED_INT ||
       cells->GetConnectivityArray()->GetMemorySpace()!=vtkDataArray::CudaDeviceMemory ||
       cells->GetConnectivityArray()->GetDeviceVoidPointer(0)!=indices.pointer)
      throw std::runtime_error("Standard connectivity did not retain the external CUDA allocation");
    vtkNew<vtkPolyData> mesh;mesh->SetPoints(vtk_points);mesh->SetPolys(cells);
    const double bounds[6]={-.75,.75,-.75,.75,-.5,.5};
    mesh->GetInformation()->Set(vtkDataObject::BOUNDING_BOX(),bounds,6);
    mesh->GetPointData()->AddArray(color_array);
    vtkSmartPointer<vtkOpenGLPolyDataMapper> mapper;
    astr_insitu::DeviceDrawView draw;
    draw.positions=reinterpret_cast<float*>(points.pointer);
    draw.colors=reinterpret_cast<unsigned char*>(colors.pointer);
    draw.indices=indices.pointer;draw.points=6;draw.cells=2;draw.device=device;
    std::copy(bounds,bounds+6,draw.bounds);
    if(direct) mapper=vtkSmartPointer<astr_insitu::DirectDeviceMapper>::Take(astr_insitu::DirectDeviceMapper::New());
    else {mapper=vtkSmartPointer<vtkOpenGLPolyDataMapper>::New();mapper->SetInputData(mesh);}
    mapper->ScalarVisibilityOff();
    vtkNew<vtkActor> actor;actor->SetMapper(mapper);actor->GetProperty()->LightingOff();
    renderer->AddActor(actor);
    auto camera=renderer->GetActiveCamera();
    camera->SetPosition(0.,0.,3.);camera->SetFocalPoint(0.,0.,0.);
    camera->ParallelProjectionOn();camera->SetParallelScale(1.);camera->SetClippingRange(1.,5.);
    bool refused=false;
    try {double xyz[3];point_array->GetTuple(0,xyz);}
    catch(const std::runtime_error&) {refused=true;}
    if(!refused) throw std::runtime_error("Standard device array allowed CPU coordinate access");
    std::vector<unsigned char> pixels(64*64*4);
    vtkNew<vtkUnsignedCharArray> image;
    image->SetNumberOfComponents(4);
    image->SetArray(pixels.data(),pixels.size(),1);
    double projection_error=0.;
    for(int frame=0;frame<3;++frame) {
      fill<<<1,32>>>(points.pointer,colors.pointer,indices.pointer,frame);
      astr_insitu::synchronize_device_stage("standard probe refill");
      point_array->Modified();color_array->Modified();index_array->Modified();
      if(direct) static_cast<astr_insitu::DirectDeviceMapper*>(mapper.GetPointer())->set_view(draw);
      nvtxRangePushA(direct?"ASTR_IS8_DIRECT_DEVICE_RENDER":"ASTR_IS8_STANDARD_DEVICE_RENDER");
      window->Render();
      if(!window->GetRGBACharPixelData(0,0,63,63,0,image))
        throw std::runtime_error("Standard VTK image read failed");
      nvtxRangePop();
      const auto at=[&](int x,int y,int channel){return pixels[4*(64*y+x)+channel];};
      if(glGetError()!=GL_NO_ERROR || at(32,32,frame==1?2:1)!=255 || at(32,32,0)!=0 ||
         at(16,12,0)!=255 || at(1,1,0)!=0 || at(1,1,1)!=0 || at(1,1,2)!=0)
        throw std::runtime_error("Standard VTK triangle color/depth/background comparison failed frame="+
          std::to_string(frame)+" front="+std::to_string(at(32,32,0))+","+
          std::to_string(at(32,32,1))+","+std::to_string(at(32,32,2))+
          " back="+std::to_string(at(16,12,0))+","+std::to_string(at(16,12,1))+","+
          std::to_string(at(16,12,2))+" background="+std::to_string(at(1,1,0))+","+
          std::to_string(at(1,1,1))+","+std::to_string(at(1,1,2)));
      for(int y=0;y<64;++y) for(int x=0;x<64;++x) {
        const double px=2.*(x+.5)/64.-1.,py=2.*(y+.5)/64.-1.;
        const bool expected=py>=-.75 && py<=.75 && std::abs(px)<=.75-.5*(py+.75);
        const bool covered=at(x,y,0)||at(x,y,1)||at(x,y,2);
        if(covered!=expected) {
          const double distance=std::min(std::abs(py+.75),
            std::min(std::abs(px+.5*py-.375),std::abs(-px+.5*py-.375))/std::sqrt(1.25));
          projection_error=std::max(projection_error,32.*distance);
        }
      }
      if(projection_error>1.) throw std::runtime_error("Analytic triangle projection differs by more than one pixel");
      astr_insitu::require_device_only(point_handle);
      astr_insitu::require_device_only(color_handle);
      astr_insitu::require_device_only(index_handle);
    }
    mapper->ReleaseGraphicsResources(window);renderer->RemoveActor(actor);window->Finalize();
    std::printf("ASTR_INSITU_STANDARD_DEVICE_PROBE {\"cuda_device\":%d,\"gl_cuda_device\":%d,"
      "\"uuid\":\"%s\",\"vtk\":\"%s\",\"frames\":3,\"geometry_host_bytes\":0,"
      "\"image_host_bytes\":49152,\"projection_max_pixels\":%.17g,\"host_access_refused\":true,"
      "\"pixels_passed\":true,\"conduit\":%s,\"direct\":%s}\n",
      device,gl_device,uuid,vtkVersion::GetVTKVersion(),projection_error,conduit?"true":"false",direct?"true":"false");
    return 0;
  } catch(const std::exception& error) {
    std::fprintf(stderr,"Standard device probe failed: %s\n",error.what());
    return 1;
  }
}
