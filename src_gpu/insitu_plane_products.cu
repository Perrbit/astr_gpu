#include "insitu_device_structured.h"
#include "insitu_compact_mesh.h"
#include "insitu_product_dispatch.h"
#include <viskores/cont/Initialize.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
using Real=astr::numerical_real;

namespace {
using namespace astr_insitu;
using PlaneAudit=viskores::Vec<Real,3>;
using SbliFields=viskores::Vec<Real,5>;
std::vector<Real> sbli_wall_values;
struct SbliTriangles : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  viskores::Id nx;
  VISKORES_CONT explicit SbliTriangles(viskores::Id x):nx(x) {}
  VISKORES_EXEC void operator()(viskores::Id id,viskores::Id& index) const {
    const auto cell=id/6,base=cell%nx+(nx+1)*(cell/nx);
    const int corner[6]={0,1,3,0,3,2};
    const auto p=corner[id%6];index=base+(p&1)+(nx+1)*(p>>1);
  }
};
struct SbliScalar : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  int component;
  VISKORES_CONT explicit SbliScalar(int c):component(c) {}
  VISKORES_EXEC void operator()(const SbliFields& fields,Real& scalar) const {
    scalar=component==4?viskores::Log10(Real(1.)+fields[4]):fields[component];
  }
};
struct PlaneSpeed : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& velocity,Real& speed) const {
    speed=viskores::Sqrt(viskores::Dot(velocity,velocity));
  }
};
struct AuditPlane : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  plane_geometry::Plane plane;
  VISKORES_CONT explicit AuditPlane(plane_geometry::Plane p):plane(p) {}
  template<class X,class I>
  VISKORES_EXEC void operator()(viskores::Id cell,const X& xyz,const I& indices,PlaneAudit& result) const {
    const auto a=xyz.Get(indices.Get(3*cell)),b=xyz.Get(indices.Get(3*cell+1)),c=xyz.Get(indices.Get(3*cell+2));
    const auto cross=viskores::Cross(b-a,c-a);
    const Real area=Real(.5)*viskores::Sqrt(viskores::Dot(cross,cross));
    Real direction=Real(0.),distance=Real(0.);
    for(int d=0;d<3;++d) direction+=cross[d]*plane.normal[d];
    bool valid=viskores::IsFinite(area) && area>Real(0.) && viskores::IsFinite(direction) && direction>Real(0.);
    for(const auto& p:{a,b,c}) {
      plane_geometry::Point point{};
      for(int d=0;d<3;++d) point[d]=p[d];
      const auto value=plane_geometry::plane_distance(plane,point);
      valid=valid && value.status==plane_geometry::Status::ok;
      distance=viskores::Max(distance,viskores::Abs(value.value));
    }
    result=PlaneAudit(valid?area:Real(0.),distance,valid?Real(0.):Real(1.));
  }
};
struct MergeAudit {
  VISKORES_EXEC_CONT PlaneAudit operator()(const PlaneAudit& a,const PlaneAudit& b) const {
    return PlaneAudit(a[0]+b[0],viskores::Max(a[1],b[1]),a[2]+b[2]);
  }
};

#ifdef ASTR_BUILD_TESTING
void test_oracle(const structured_geometry::PlaneProduct& product,const DeviceGeometryView& view,
    int rank,int step,std::uint64_t host_budget) {
  const char* prefix=std::getenv("ASTR_INSITU_TEST_PLANE_PREFIX");
  if(!prefix || !*prefix) return;
  const std::uint64_t bytes=std::uint64_t(view.points)*(6*sizeof(Real)+2*sizeof(viskores::Id))+
    std::uint64_t(view.cells)*3*sizeof(viskores::Id);
  if(bytes>host_budget || bytes>64*1024*1024)
    throw std::runtime_error("Plane test oracle host budget exceeded");
  std::vector<astr_insitu::NativeVector> xyz(view.points),velocity(view.points);
  std::vector<structured_geometry::PlaneKey> keys(view.points);
  std::vector<viskores::Id> indices(view.cells*3);
  viskores::cont::Token token;
  if(view.points) {
    const auto field=product.data.GetPointField("velocity").GetData().AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
    const auto vectors=field.PrepareForInput(viskores::cont::DeviceAdapterTagCuda{},token);
    const auto point_keys=product.keys.PrepareForInput(viskores::cont::DeviceAdapterTagCuda{},token);
    if(cudaMemcpy(xyz.data(),view.coordinates,xyz.size()*sizeof(xyz[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(velocity.data(),vectors.GetIteratorBegin(),velocity.size()*sizeof(velocity[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(keys.data(),point_keys.GetIteratorBegin(),keys.size()*sizeof(keys[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(indices.data(),view.connectivity,indices.size()*sizeof(indices[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Plane test oracle download failed");
  }
  const std::string path=std::string(prefix)+".step"+std::to_string(step)+".rank"+std::to_string(rank)+".bin";
  std::ofstream file(path,std::ios::binary|std::ios::out);
  const std::uint64_t counts[2]={std::uint64_t(view.points),std::uint64_t(view.cells)};
  file.write(reinterpret_cast<const char*>(counts),sizeof(counts));
  for(const auto* data:{&xyz,&velocity})
    if(!data->empty()) file.write(reinterpret_cast<const char*>(data->data()),data->size()*sizeof((*data)[0]));
  if(!keys.empty()) file.write(reinterpret_cast<const char*>(keys.data()),keys.size()*sizeof(keys[0]));
  if(!indices.empty()) file.write(reinterpret_cast<const char*>(indices.data()),indices.size()*sizeof(indices[0]));
  file.close();
  if(!file) throw std::runtime_error("Plane test oracle publication failed");
  std::printf("ASTR_INSITU_PLANE_TEST_ORACLE rank=%d step=%d oracle_download_bytes=%llu runtime_zero_readback_evidence=0\n",
    rank,step,static_cast<unsigned long long>(bytes));
}
#endif
}

astr_insitu::DeviceMesh astr_insitu::build_device_plane(int fcomm,int step,Real time,const int global[3],
    const int local[3],const int offset[3],Real* coordinates,Real* velocity,
    const Real origin[3],const Real normal[3],std::int64_t budget,std::int64_t reserve,
    std::int64_t retained,std::int64_t host_budget,bool speed_colors,Real speed_max) {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  const auto comm=MPI_Comm_f2c(fcomm);
  if(step<0 || !std::isfinite(time) ||
      !coordinates || !velocity || !origin || !normal || budget<=0 || retained<0 || reserve<0 || host_budget<=0)
    throw std::invalid_argument("Invalid strict physical plane identity");
#ifndef ASTR_BUILD_TESTING
  if(std::getenv("ASTR_INSITU_TEST_PLANE_PREFIX"))
    throw std::invalid_argument("Plane test oracle is unavailable in production builds");
#endif
  static bool initialized=false;
  if(!initialized) {
    int argc=1;char program[]="astr-insitu-plane";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
  const structured_geometry::Partition partition(viskores::Id3(global[0],global[1],global[2]),
    viskores::Id3(local[0],local[1],local[2]),viskores::Id3(offset[0],offset[1],offset[2]));
  plane_geometry::Point point{},direction{};
  for(int d=0;d<3;++d) {point[d]=origin[d];direction[d]=normal[d];}
  std::size_t free_bytes=0,total_bytes=0;
  if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess ||
      free_bytes<std::uint64_t(std::max<std::int64_t>(reserve,1073741824)))
    throw std::runtime_error("Physical plane device reserve exceeded");
  auto product=structured_geometry::extract_plane(reinterpret_cast<astr_insitu::NativeVector*>(coordinates),
    reinterpret_cast<astr_insitu::NativeVector*>(velocity),nullptr,partition,point,direction,std::uint64_t(budget),
    std::uint64_t(retained),true,true);
  if(speed_colors && product.data.GetNumberOfCoordinateSystems()) {
    const auto vectors=product.data.GetPointField("velocity").GetData().AsArrayHandle<
      viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
    viskores::cont::ArrayHandle<Real> speed;
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    invoke(PlaneSpeed{},vectors,speed);
    synchronize_device_stage("PlaneSpeed");
    require_device_only(speed);
    product.data.AddPointField("speed",speed);
  }
  auto owner=std::make_shared<DeviceGeometryOwner>(product.data,step,time,3,speed_colors?"speed":"u");
  const auto colors=owner->color_display(device_display_palette(),false,speed_colors?Real(0.):-Real(1.),speed_max);
  const auto& view=owner->get();
  PlaneAudit audit(Real(0.));
  if(view.cells) {
    auto xyz=borrow_cuda_array(const_cast<astr_insitu::NativeVector*>(view.coordinates),view.points);
    auto indices=borrow_cuda_array(const_cast<viskores::Id*>(view.connectivity),view.cells*3);
    viskores::cont::ArrayHandle<PlaneAudit> values;
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    invoke(AuditPlane({point,direction}),viskores::cont::ArrayHandleIndex(view.cells),xyz,indices,values);
    synchronize_device_stage("AuditPhysicalPlane");
    audit=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},values,PlaneAudit(Real(0.)),MergeAudit{});
    synchronize_device_stage("ReducePhysicalPlaneAudit");
    if(!std::isfinite(audit[0]) || !std::isfinite(audit[1]) ||
        (sizeof(Real)==8 && audit[1]>Real(2e-10)) || audit[2]!=Real(0.))
      throw std::runtime_error("Physical plane has nonfinite, reversed or zero-area geometry");
    require_device_only(values);require_device_only(xyz);require_device_only(indices);
  }
  int rank=0;
  if(MPI_Comm_rank(comm,&rank)!=MPI_SUCCESS) throw std::runtime_error("Cannot identify physical plane rank");
#ifdef ASTR_BUILD_TESTING
  test_oracle(product,view,rank,step,std::uint64_t(host_budget));
#endif
  DeviceMesh mesh;mesh.name="velocity_slice";mesh.owner=owner;
  mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
  mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
  mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.device=view.device;mesh.draw.arity=3;
  std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
  for(int d=0;d<3;++d) {
    mesh.controls["plane_origin"+std::to_string(d)]=point[d];
    mesh.controls["plane_normal"+std::to_string(d)]=direction[d];
  }
  std::printf("ASTR_INSITU_DEVICE_PLANE rank=%d step=%d points=%lld triangles=%lld area=%.17g "
    "plane_maxabs=%.17g invalid_triangles=%.0f field_download_bytes=0 geometry_host_bytes=0 q_requested=0\n",
    rank,step,static_cast<long long>(view.points),static_cast<long long>(view.cells),audit[0],audit[1],audit[2]);
  if(!retain_quantitative_product_fields()) owner->retain_display_only();
  return mesh;
#else
  throw std::invalid_argument("Strict physical plane rendering is not built; no fallback");
#endif
}

#include "insitu_product_failure.h"
extern "C" int astr_insitu_device_render_plane(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,int nx,int ny,int nz,int ox,int oy,int oz,
    Real* coordinates,Real* velocity,const Real* origin,const Real* normal,
    std::int64_t budget,std::int64_t reserve,std::int64_t retained,std::int64_t host_budget)
try {
  const std::string route=pipeline?pipeline:"";
  if(route!="standard-device" && route!="direct-device")
    throw std::invalid_argument("Physical plane requires a strict device pipeline; no fallback");
  const int global[3]={32,32,32},local[3]={nx,ny,nz},offset[3]={ox,oy,oz};
  astr_insitu::DeviceMesh mesh;
  if(!astr_insitu::local_product(MPI_Comm_f2c(fcomm),"velocity_slice",step,time,[&] {
    mesh=astr_insitu::build_device_plane(fcomm,step,time,global,local,offset,coordinates,velocity,
      origin,normal,budget,reserve,retained,host_budget);
  })) return 0;
  return astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,"physical_plane",{std::move(mesh)});
} catch(const std::exception& error) {
  std::fprintf(stderr,"Physical plane product failed: %s\n",error.what());
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
}

// Only the compact O(Nx) wall profile crosses this host interface. Plane fields
// and their coordinates/connectivity remain borrowed CUDA arrays.
extern "C" const Real* astr_insitu_sbli_wall_data(int* count,int* real_bytes) {
  *count=int(sbli_wall_values.size()/4);
  *real_bytes=int(sizeof(Real));
  return sbli_wall_values.data();
}

extern "C" int astr_insitu_device_render_sbli_2d(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,int nx,int ny,Real* coordinates_pointer,Real* fields_pointer,
    const Real* wall,int nwall,const Real* bounds,std::int64_t budget,std::int64_t reserve,std::int64_t retained)
try {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  using namespace astr_insitu;
  const auto comm=MPI_Comm_f2c(fcomm);
  if(nx<6 || ny<6 || budget<=0 || reserve<0 || retained<0 || retained>budget || nwall<0)
    throw std::invalid_argument("Invalid compact SBLI plane shape/budget");
  static bool initialized=false;
  if(!initialized) {
    int argc=1;char program[]="astr-insitu-sbli";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
  const auto count=viskores::Id(nx+1)*(ny+1);
  auto coordinates=borrow_cuda_array(reinterpret_cast<NativeVector*>(coordinates_pointer),count);
  auto fields=borrow_cuda_array(reinterpret_cast<SbliFields*>(fields_pointer),count);
  const std::uint64_t display_bound=std::uint64_t(count)*256+std::uint64_t(nx)*ny*6*32;
  std::size_t free=0,total=0;
  if(cudaMemGetInfo(&free,&total)!=cudaSuccess || free<std::uint64_t(reserve) ||
      display_bound>std::uint64_t(budget-retained) || display_bound>free-std::uint64_t(reserve))
    throw std::runtime_error("Compact SBLI plane display budget exceeded");
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::ArrayHandle<viskores::Id> indices;
  invoke(SbliTriangles{nx},viskores::cont::ArrayHandleIndex(viskores::Id(6)*nx*ny),indices);
  synchronize_device_stage("SbliPlaneTriangles");
  viskores::cont::CellSetSingleType<> cells;
  cells.Fill(count,viskores::CELL_SHAPE_TRIANGLE,3,indices);
  const char* names[]={"velocity_slice","streamwise_velocity_slice","pressure_ratio_slice", "temperature_slice",
                       "density_gradient_slice","wall_pressure_curve","skin_friction_curve","wall_heat_flux_curve"};
  const Real ranges[5][2]={{0,Real(1.1)},{-Real(1.1),Real(1.1)},{0,8},{0,11},{0,4}};
  std::vector<DeviceMesh> meshes;
  int rank=0;
  if(MPI_Comm_rank(comm,&rank)!=MPI_SUCCESS) throw std::runtime_error("Cannot identify SBLI plane rank");
  sbli_wall_values.clear();
  if(rank==0 && nwall) sbli_wall_values.assign(wall,wall+4*nwall);
  for(int c=0;c<8;++c) {
    DeviceMesh mesh;mesh.name=names[c];
    viskores::cont::DataSet data;
    if(c<5 && astr_insitu_scene_due(names[c])) {
      const bool ok=local_product(comm,names[c],step,time,[&] {
        viskores::cont::ArrayHandle<Real> scalar;
        invoke(SbliScalar{c},fields,scalar);
        synchronize_device_stage("SbliPlaneScalar");
        data.AddCoordinateSystem(viskores::cont::CoordinateSystem("coordinates",coordinates));
        data.SetCellSet(cells);data.AddPointField("value",scalar);
      });
      if(!ok) data=viskores::cont::DataSet{};
    }
    auto owner=std::make_shared<DeviceGeometryOwner>(data,step,time,3,"value");
    mesh.owner=owner;
    const auto colors=owner->color_display(device_display_palette(),false,c<5?ranges[c][0]:Real(0.),
      c<5?ranges[c][1]:Real(1.));
    const auto& view=owner->get();
    mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
    mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
    mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.device=view.device;mesh.draw.arity=3;
    std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
    for(int d=0;d<6;++d) mesh.controls["domain_bound"+std::to_string(d)]=bounds[d];
    mesh.controls["group_member"]=Real(1.);
    mesh.controls["curve_points"]=Real(nwall);
    if(c<5) {
      mesh.controls["color_min"]=ranges[c][0];mesh.controls["color_max"]=ranges[c][1];
      mesh.controls["actual_min"]=view.speed_range[0];mesh.controls["actual_max"]=view.speed_range[1];
    }
    meshes.push_back(std::move(mesh));
    if(!retain_quantitative_product_fields()) owner->retain_display_only();
  }
  require_device_only(coordinates);require_device_only(fields);require_device_only(indices);
  const auto status=render_resident_products(pipeline,backend,script,fcomm,step,time,"sbli_2d",std::move(meshes));
  sbli_wall_values.clear();return status;
#else
  throw std::runtime_error("SBLI device rendering is not built; no host fallback");
#endif
} catch(const std::exception& error) {
  std::fprintf(stderr,"SBLI plane products failed: %s\n",error.what());
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
}
