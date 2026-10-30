#include "insitu_device_structured.h"
#include "insitu_compact_mesh.h"
#include "insitu_product_dispatch.h"
#include "insitu_allocation_budget.h"
#include <viskores/cont/Initialize.h>
#include <viskores/cont/CellSetExplicit.h>
#include <viskores/filter/contour/ClipWithField.h>
#include <viskores/filter/geometry_refinement/Triangulate.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <tuple>
using Real=astr::numerical_real;

namespace {
using namespace astr_insitu;
using PlaneAudit=viskores::Vec<Real,3>;
using SbliFields=viskores::Vec<Real,5>;
std::vector<Real> sbli_wall_values;
std::shared_ptr<std::vector<Real>> runtime_wall_values;
std::shared_ptr<std::vector<unsigned char>> runtime_wall_breaks;
int runtime_wall_nondimensional=0;
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
struct RuntimePlaneScalar : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id nodes;
  int a,b,c;
  Real factor;
  bool magnitude;
  VISKORES_CONT RuntimePlaneScalar(viskores::Id n,int x,int y,int z,Real scale,bool speed)
    :nodes(n),a(x),b(y),c(z),factor(scale),magnitude(speed) {}
  template<class Values>
  VISKORES_EXEC void operator()(viskores::Id node,const Values& fields,Real& value) const {
    value=fields.Get(node+nodes*a);
    if(magnitude) {
      const Real v=fields.Get(node+nodes*b),w=fields.Get(node+nodes*c);
      value=viskores::Sqrt(value*value+v*v+w*w);
    }
    value*=factor;
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
extern "C" void astr_insitu_runtime_failure(const char*);

double astr_insitu::runtime_product_clock() {
  static const bool enabled=[] {
    const char* value=std::getenv("ASTR_INSITU_TIMING");
    return value && (std::string(value)=="1" || std::string(value)=="true" || std::string(value)=="TRUE" ||
      std::string(value)=="t" || std::string(value)=="T" || std::string(value)=="on" || std::string(value)=="ON");
  }();
  return enabled?MPI_Wtime():-1.;
}

#ifdef ASTR_INSITU_DEVICE_RENDERING
namespace {
std::map<const void*,std::shared_ptr<astr_insitu::CompactMesh>> runtime_compact_cache;

std::uint64_t compact_bytes(const astr_insitu::CompactMesh& mesh) {
  std::uint64_t bytes=(mesh.connectivity.size()+mesh.seed_ids.size()+mesh.directions.size())*sizeof(std::int64_t);
  for(const auto& axis:mesh.coordinates) bytes+=axis.size()*sizeof(Real);
  for(const auto& field:mesh.point_fields) bytes+=field.second.size()*sizeof(Real);
  return bytes;
}

std::shared_ptr<astr_insitu::CompactMesh> runtime_compatible_geometry(
    const astr_insitu::RuntimeProductView& context,const astr_insitu::DeviceMesh& mesh) {
  const auto found=runtime_compact_cache.find(mesh.owner.get());
  if(found!=runtime_compact_cache.end()) {
    found->second->controls["geometry_download_bytes"]=0.;
    return found->second;
  }
  const auto owner=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(mesh.owner);
  const auto& view=owner->get();
  auto packet=std::make_shared<astr_insitu::CompactMesh>();
  packet->name="runtime_product";packet->shape=view.arity==3?"tri":"line";
  const bool geometry=context.labels[1]=="geometry";
  std::map<std::string,const Real*> point_fields;
  if(geometry) {
    for(const auto& field:context.fields) point_fields[field.second]=view.points?owner->point_fields().at(field.second):nullptr;
    if(view.arity==2) for(const char* name:{"accepted_length","termination_status","untravelled_arc_length"})
      point_fields[name]=view.points?owner->point_fields().at(name):nullptr;
  } else point_fields["quantity"]=view.scalar;
  const int cell_fields=geometry && view.arity==2?2:0;
  const std::uint64_t bytes=std::uint64_t(view.points)*(3+point_fields.size())*sizeof(Real)+
    std::uint64_t(view.cells)*(view.arity+cell_fields)*sizeof(viskores::Id);
  std::uint64_t held=0;
  const auto previous=astr_insitu::retained_runtime_geometry();bool shared_previous=false;
  for(const auto& entry:runtime_compact_cache) {
    held+=compact_bytes(*entry.second);shared_previous|=entry.second==previous;
  }
  if(previous && !shared_previous) held+=compact_bytes(*previous);
  const std::uint64_t temporary=std::uint64_t(view.points)*sizeof(astr_insitu::NativeVector);
  auto budget=context.host_budget;
#ifdef ASTR_BUILD_TESTING
  if(const char* fault=std::getenv("ASTR_INSITU_TEST_RUNTIME_ALLOCATION_REFUSAL")) {
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(std::string(fault)==context.id+":"+std::to_string(view.step)+":"+std::to_string(rank)+":compatible") budget=0;
  }
#endif
  if(budget<=0 || held>std::uint64_t(budget) || bytes>std::uint64_t(budget)-held ||
      temporary>std::uint64_t(budget)-held-bytes)
    throw std::runtime_error("Explicit compatible geometry exceeds the controlled host-buffer budget");
  std::vector<astr_insitu::NativeVector> xyz(view.points);
  for(auto& axis:packet->coordinates) axis.resize(view.points);
  packet->connectivity.resize(view.cells*view.arity);
  static_assert(sizeof(viskores::Id)==sizeof(std::int64_t),"Native compact connectivity requires int64");
  if(view.points && cudaMemcpy(xyz.data(),view.coordinates,xyz.size()*sizeof(xyz[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("Cannot download explicit compatible native coordinates");
  for(const auto& field:point_fields) {
    auto& values=packet->point_fields[field.first];values.resize(view.points);
    if(view.points && cudaMemcpy(values.data(),field.second,values.size()*sizeof(Real),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Cannot download explicit compatible native field "+field.first);
  }
  if(view.cells && cudaMemcpy(packet->connectivity.data(),view.connectivity,
      packet->connectivity.size()*sizeof(std::int64_t),cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("Cannot download explicit compatible native connectivity");
  if(cell_fields) {
    packet->seed_ids.resize(view.cells);packet->directions.resize(view.cells);
    if(view.cells && (cudaMemcpy(packet->seed_ids.data(),owner->cell_fields().at("SeedIds"),
        view.cells*sizeof(viskores::Id),cudaMemcpyDeviceToHost)!=cudaSuccess ||
        cudaMemcpy(packet->directions.data(),owner->cell_fields().at("IntegrationDirection"),
        view.cells*sizeof(viskores::Id),cudaMemcpyDeviceToHost)!=cudaSuccess))
      throw std::runtime_error("Cannot download native streamline identity");
  }
  for(std::size_t i=0;i<xyz.size();++i) for(int d=0;d<3;++d) packet->coordinates[d][i]=xyz[i][d];
  packet->controls=mesh.controls;
  packet->controls["device_points"]=view.points;packet->controls["device_cells"]=view.cells;
  packet->controls["device_arity"]=view.arity;
  packet->controls["geometry_host_bytes"]=bytes;packet->controls["geometry_download_bytes"]=bytes;
  packet->controls["source_real_bytes"]=sizeof(Real);
  for(int d=0;d<6;++d) packet->controls["bound"+std::to_string(d)]=view.bounds[d];
  runtime_compact_cache.emplace(mesh.owner.get(),packet);
  return packet;
}
}

void astr_insitu::clear_runtime_compatible_geometry() {runtime_compact_cache.clear();}
#endif
int astr_insitu::publish_runtime_product(const char* pipeline,const char* backend,const char* script,int fcomm,int step,Real time,
    astr_insitu::RuntimeProductView& context,astr_insitu::DeviceMesh mesh,std::string error,double started) {
  const auto comm=MPI_Comm_f2c(fcomm);
  const auto prepared=runtime_product_clock();
  if(astr_insitu::consensus_product_failure(comm,error)) {
#ifdef ASTR_INSITU_DEVICE_RENDERING
    astr_insitu_runtime_failure(error.c_str());
#endif
    return 1;
  }
#ifdef ASTR_INSITU_DEVICE_RENDERING
  const auto agreed=runtime_product_clock();
  int result=0;
  if(std::string(pipeline?pipeline:"")=="compatible") {
    std::shared_ptr<CompactMesh> packet;
    try {
      if(context.labels[0]=="wall_curve") {
        packet=std::make_shared<CompactMesh>();packet->name="runtime_product";packet->shape="line";
        packet->controls=mesh.controls;
      } else packet=runtime_compatible_geometry(context,mesh);
    } catch(const std::exception& caught) {error=caught.what();}
    if(consensus_product_failure(comm,error)) {astr_insitu_runtime_failure(error.c_str());return 1;}
    int rank=0;MPI_Comm_rank(comm,&rank);
    std::printf("ASTR_RUNTIME_COMPATIBLE rank=%d id=%s step=%d geometry_download_bytes=%.0f geometry_host_bytes=%.0f\n",
      rank,context.id.c_str(),step,packet->controls["geometry_download_bytes"],packet->controls["geometry_host_bytes"]);
    result=render_compact_products(backend,script,fcomm,step,time,"runtime_catalog",false,Real(0.),Real(0.),Real(0.),{},&context,packet);
  } else result=astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,"runtime_catalog",
    {std::move(mesh)},false,true,&context);
  if(started>=0.) {
    const double finished=runtime_product_clock();int rank=0;MPI_Comm_rank(comm,&rank);
    std::printf("ASTR_RUNTIME_STAGE_TIMING rank=%d id=%s step=%d native_local_seconds=%.17g "
      "consensus_seconds=%.17g consumer_inclusive_seconds=%.17g status=%d\n",rank,context.id.c_str(),step,
      prepared-started,agreed-prepared,finished-agreed,result);
  }
  return result;
#else
  return 1;
#endif
}

#ifdef ASTR_INSITU_DEVICE_RENDERING
extern "C" void astr_insitu_device_allocation_preflight(std::size_t,const char*);
extern "C" int astr_insitu_allocation_guard_active();

void astr_insitu::prepare_runtime_product_allocations(const RuntimeProductView& context,int fcomm,int step) {
#ifdef ASTR_VISKORES_ALLOCATION_PREFLIGHT
  if(!astr_insitu_allocation_guard_active())
    throw std::runtime_error("Runtime native allocations require an active resource guard");
  viskores::cont::GetRuntimeDeviceTracker().SetThreadFriendlyMemAlloc(false);
  viskores::cont::cuda::internal::CudaAllocator::SetAllocationPreflight(astr_insitu_device_allocation_preflight);
  astr_insitu_device_allocation_preflight(0,"runtime native product");
#ifdef ASTR_BUILD_TESTING
  if(const char* fault=std::getenv("ASTR_INSITU_TEST_RUNTIME_ALLOCATION_REFUSAL")) {
    int rank=0;MPI_Comm_rank(MPI_Comm_f2c(fcomm),&rank);
    if(std::string(fault)==context.id+":"+std::to_string(step)+":"+std::to_string(rank))
      astr_insitu_device_allocation_preflight(std::numeric_limits<std::size_t>::max(),"runtime allocation test");
  }
#endif
#else
  (void)context;(void)fcomm;(void)step;
#endif
}

extern "C" const Real* astr_insitu_runtime_wall_data(int* count,int* bytes,int* nondimensional) {
  *count=runtime_wall_values?int(runtime_wall_values->size()/4):0;
  *bytes=sizeof(Real);*nondimensional=runtime_wall_nondimensional;
  return *count?runtime_wall_values->data():nullptr;
}

extern "C" const unsigned char* astr_insitu_runtime_wall_breaks() {
  return runtime_wall_breaks && !runtime_wall_breaks->empty()?runtime_wall_breaks->data():nullptr;
}

extern "C" int astr_insitu_render_wall_product(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,const char* id,const char* source,const char* definition,const char* directory,
    std::int64_t version,const char* labels,const int* words,const double* settings,const Real* values,int nodes,
    int column,Real factor,int nondimensional,int provider,const Real* statistics,std::int64_t samples,std::int64_t host_budget,
    int host) {
  const auto started=astr_insitu::runtime_product_clock();
  astr_insitu::RuntimeProductView context;
  astr_insitu::DeviceMesh mesh;
  std::string error;
  try {
    context=astr_insitu::runtime_product_view(id,source,definition,directory,version,labels,words,settings);
    context.host_budget=host_budget;
    context.statistics_provider=provider;context.samples=samples;std::copy_n(statistics,3,context.statistics.begin());
    int rank=0;MPI_Comm_rank(MPI_Comm_f2c(fcomm),&rank);
    if(context.labels[0]!="wall_curve" || nodes<0 || column<4 || column>6 || !std::isfinite(factor) ||
        host_budget<=0 || std::uint64_t(nodes)*4*sizeof(Real)>std::uint64_t(host_budget))
      throw std::invalid_argument("Invalid runtime wall curve input or host packet budget");
    runtime_wall_values=std::make_shared<std::vector<Real>>();
    runtime_wall_breaks=std::make_shared<std::vector<unsigned char>>();
    runtime_wall_nondimensional=nondimensional;
    if(rank==0) {
      runtime_wall_values->reserve(std::size_t(nodes)*4);
      runtime_wall_breaks->reserve(nodes);
      int previous=-1;
      for(int i=0;i<nodes;++i) {
        const auto* row=values+6*i;
        const Real scalar=row[column-1]*factor;
        bool selected=true;
        for(int d=0;d<3;++d) {
          if(!std::isfinite(row[d])) throw std::runtime_error("Nonfinite wall curve coordinates");
          selected=selected && row[d]>=context.reals[6+2*d] && row[d]<=context.reals[7+2*d];
        }
        if(!selected) continue;
        if(!std::isfinite(scalar)) throw std::runtime_error("Nonfinite normalized wall curve quantity");
        runtime_wall_values->insert(runtime_wall_values->end(),{row[0],row[1],row[2],scalar});
        runtime_wall_breaks->push_back(previous>=0 && i!=previous+1);
        previous=i;
      }
    }
    mesh.name="runtime_product";mesh.actor_key=context.id;mesh.owner=runtime_wall_values;
    mesh.controls["host_source"]=host;
    mesh.draw.arity=2;
    if(cudaGetDevice(&mesh.draw.device)!=cudaSuccess) throw std::runtime_error("Cannot identify the wall chart device binding");
    std::copy_n(context.reals.begin()+6,6,mesh.draw.bounds);
    std::printf("ASTR_RUNTIME_WALL_CHART rank=%d id=%s step=%d host_curve_points=%zu host_curve_bytes=%zu "
      "three_dimensional_field_download_bytes=0\n",rank,id,step,runtime_wall_values->size()/4,
      runtime_wall_values->size()*sizeof(Real));
  } catch(const std::exception& caught) {error=caught.what();}
  const int status=astr_insitu::publish_runtime_product(pipeline,backend,script,fcomm,step,time,context,std::move(mesh),error,started);
  runtime_wall_values.reset();runtime_wall_breaks.reset();
  return status;
}

namespace {
struct RuntimeIndexKey {
  int a,b,c;
  Real factor;
  bool magnitude;
  std::vector<int> fields;
  bool operator<(const RuntimeIndexKey& other) const {
    return std::tie(a,b,c,factor,magnitude,fields)<std::tie(other.a,other.b,other.c,other.factor,other.magnitude,other.fields);
  }
};
struct RuntimeIndexGroup {
  bool active=false,geometry=false;
  int width=0,height=0;
  Real* coordinates=nullptr;
  Real* values=nullptr;
  std::vector<int> fields;
  viskores::cont::DataSet grid;
  std::map<RuntimeIndexKey,std::shared_ptr<astr_insitu::DeviceGeometryOwner>> owners;
  int triangulations=0,scalar_builds=0,owner_reuse=0;
} index_group;

RuntimeIndexKey scalar_selection(const astr_insitu::RuntimeProductView& context,const int* ids,
    int fields,const Real* references,int selected_field) {
  auto slot=[&](int id) {
    for(int c=0;c<fields;++c) if(ids[c]==id) return c;
    throw std::logic_error("Runtime surface source is missing a declared field");
  };
  RuntimeIndexKey key{0,0,0,Real(1.),context.labels[4]=="velocity_magnitude"};
  for(const auto& field:context.fields) key.fields.push_back(field.first);
  const auto& quantity=context.labels[4];
  if(key.magnitude) {key.a=slot(2);key.b=slot(3);key.c=slot(4);}
  else if(quantity=="u_Uinf" || quantity=="w_Uinf" || quantity=="T_Tinf" || quantity=="p_pinf") {
    const int reference=quantity=="T_Tinf"?2:(quantity=="p_pinf"?1:0);
    if(!std::isfinite(references[reference]) || references[reference]==Real(0.))
      throw std::invalid_argument("Normalized surface field requires a finite nonzero reference value");
    key.a=slot(quantity=="u_Uinf"?2:(quantity=="w_Uinf"?4:(quantity=="T_Tinf"?6:5)));
    key.factor=Real(1.)/references[reference];
    if(!std::isfinite(key.factor)) throw std::invalid_argument("Surface reference reciprocal is not representable");
  } else if(selected_field) key.a=slot(selected_field);
  else if(fields!=1) throw std::logic_error("A scalar surface must identify one source component");
  return key;
}

struct RuntimeSurfaceEntry {
  viskores::cont::DataSet data;
  std::map<RuntimeIndexKey,std::shared_ptr<astr_insitu::DeviceGeometryOwner>> owners;
};
struct RuntimePhysicalGroup {
  bool active=false,frozen=false;
  std::map<std::array<double,14>,RuntimeSurfaceEntry> surfaces;
  Real* coordinates=nullptr;
  Real* values=nullptr;
  std::vector<int> fields,partition;
  int extractions=0,scalar_builds=0,owner_reuse=0;
} physical_group;

struct ZeroPlaneVector : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(viskores::Id,astr_insitu::NativeVector& value) const {value=astr_insitu::NativeVector(Real(0.));}
};
struct InterpolateRuntimePlaneField : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id offset;
  VISKORES_CONT explicit InterpolateRuntimePlaneField(viskores::Id first):offset(first) {}
  template<class V>
  VISKORES_EXEC void operator()(const astr_insitu::structured_geometry::PlaneInterpolation& sample,
      const V& source,Real& result) const {
    const Real a=source.Get(offset+sample.lower),b=source.Get(offset+sample.upper);
    result=a+sample.fraction*(b-a);
  }
};
struct RuntimeSurfaceScalar : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  Real factor;bool magnitude;
  VISKORES_CONT RuntimeSurfaceScalar(Real f,bool speed):factor(f),magnitude(speed) {}
  VISKORES_EXEC void operator()(Real a,Real b,Real c,Real& result) const {
    result=(magnitude?viskores::Sqrt(a*a+b*b+c*c):a)*factor;
  }
};
struct RuntimeClipDistance : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  int axis;Real origin,sign;
  VISKORES_CONT RuntimeClipDistance(int d,Real x,Real direction):axis(d),origin(x),sign(direction) {}
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& xyz,Real& value) const {
    value=(xyz[axis]-origin)*sign;
  }
};

struct RuntimeLineStatus : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  template<class X,class I>
  VISKORES_EXEC void operator()(viskores::Id cell,const X& xyz,const I& indices,viskores::Id& status) const {
    const auto a=xyz.Get(indices.Get(2*cell)),b=xyz.Get(indices.Get(2*cell+1));
    status=a==b?1:0;
    for(int d=0;d<3;++d) if(!viskores::IsFinite(a[d]) || !viskores::IsFinite(b[d])) status=2;
  }
};
struct RuntimeLineNumber {
  VISKORES_EXEC_CONT viskores::Id operator()(viskores::Id index) const {return index/2;}
};
}

viskores::cont::DataSet astr_insitu::clip_runtime_device_mesh(viskores::cont::DataSet data,
    const astr_insitu::RuntimeProductView& context,int arity) {
  if(!data.GetNumberOfCoordinateSystems() || !data.GetCellSet().GetNumberOfCells()) return {};
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  for(int d=0;d<3;++d) {
    const Real lower=Real(context.reals[6+2*d]),upper=Real(context.reals[7+2*d]);
    if(!std::isfinite(lower) || !std::isfinite(upper)) throw std::invalid_argument("Surface bounds are not representable");
    if(lower==upper) {
      if(std::abs(context.reals[3+d])==1. && Real(context.reals[d])==lower) continue;
    }
    for(int side=0;side<2;++side) {
      const auto xyz=data.GetCoordinateSystem().GetData().AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
      viskores::cont::ArrayHandle<Real> distance;
      invoke(RuntimeClipDistance(d,side?upper:lower,side?-Real(1.):Real(1.)),xyz,distance);
      astr_insitu::synchronize_device_stage("RuntimeClipDistance");
      astr_insitu::require_device_only(distance);
      data.AddPointField("_astr_clip_distance",distance);
      viskores::filter::contour::ClipWithField clip;
      clip.SetActiveField("_astr_clip_distance");clip.SetClipValue(0.);
      data=clip.Execute(data);astr_insitu::synchronize_device_stage("ClipRuntimeSurface");
      if(!data.GetCellSet().GetNumberOfCells()) return {};
    }
  }
  if(arity==3) {
    viskores::filter::geometry_refinement::Triangulate triangles;
    data=triangles.Execute(data);astr_insitu::synchronize_device_stage("TriangulateRuntimeSurface");
  } else if(arity==2) {
    const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetExplicit<>>();
    const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{});
    if(indices.GetNumberOfValues()!=2*cells.GetNumberOfCells()) throw std::runtime_error("Clipped streamline is not a line mesh");
    viskores::cont::CellSetSingleType<> lines;
    lines.Fill(data.GetCoordinateSystem().GetNumberOfPoints(),viskores::CELL_SHAPE_LINE,2,indices);
    data.SetCellSet(lines);
    const auto xyz=data.GetCoordinateSystem().GetData().AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
    viskores::cont::ArrayHandle<viskores::Id> status,kept;
    invoke(RuntimeLineStatus{},viskores::cont::ArrayHandleIndex(cells.GetNumberOfCells()),xyz,indices,status);
    astr_insitu::synchronize_device_stage("ClassifyRuntimeLineCells");
    const auto device=viskores::cont::DeviceAdapterTagCuda{};
    if(viskores::cont::Algorithm::Reduce(device,status,viskores::Id(0),astr_insitu::MaximumSurfaceStatus{})==2)
      throw std::runtime_error("Clipped streamline has nonfinite coordinates");
    const auto zeros=viskores::cont::Algorithm::Reduce(device,status,viskores::Id(0));
    if(zeros) {
      const auto number=viskores::cont::make_ArrayHandleTransform(viskores::cont::ArrayHandleIndex(indices.GetNumberOfValues()),RuntimeLineNumber{});
      viskores::cont::Algorithm::CopyIf(device,indices,viskores::cont::make_ArrayHandlePermutation(number,status),kept,
        astr_insitu::ValidSurfaceTriangle{});
      astr_insitu::synchronize_device_stage("RemoveZeroLengthRuntimeLines");
      if(!kept.GetNumberOfValues()) return {};
      lines.Fill(xyz.GetNumberOfValues(),viskores::CELL_SHAPE_LINE,2,kept);data.SetCellSet(lines);
    }
    astr_insitu::require_device_only(xyz);astr_insitu::require_device_only(indices);
    astr_insitu::require_device_only(status);astr_insitu::require_device_only(kept);
  } else throw std::invalid_argument("Unsupported runtime geometry cell arity");
  auto coordinates=data.GetCoordinateSystem().GetData().AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
  astr_insitu::require_device_only(coordinates);
  if(arity==3) astr_insitu::remove_zero_area_surface_cells(data);
  return data;
}

namespace {

viskores::cont::DataSet runtime_q_surface(const astr_insitu::RuntimeProductView& context,
    const int* ids,int fields,const int* extent,Real* coordinates,Real* values) {
  const viskores::Id nodes=viskores::Id(extent[0])*extent[1]*extent[2];
  auto grid=viskores::cont::DataSetBuilderUniform().Create(viskores::Id3(extent[0],extent[1],extent[2]));
  grid.AddPointField("_astr_physical_xyz",astr_insitu::borrow_cuda_array(
    reinterpret_cast<astr_insitu::NativeVector*>(coordinates),nodes));
  viskores::filter::FieldSelection selected;
  selected.AddField("_astr_physical_xyz");
  bool q=false;
  for(int f=0;f<fields;++f) {
    const auto name="raw"+std::to_string(ids[f]);
    grid.AddPointField(name,astr_insitu::borrow_cuda_array(values+nodes*f,nodes));selected.AddField(name);
    q=q || ids[f]==22;
  }
  if(!q) throw std::logic_error("Runtime Q surface is missing its native Q field");
  viskores::filter::contour::ContourFlyingEdges contour;
  contour.SetActiveField("raw22");contour.SetIsoValue(context.reals[25]);
  contour.SetGenerateNormals(false);contour.SetFieldsToPass(selected);
  const auto result=contour.Execute(grid);astr_insitu::synchronize_device_stage("RuntimeFlyingEdges");
  if(!result.GetCellSet().GetNumberOfCells()) return {};
  const auto physical=result.GetPointField("_astr_physical_xyz").GetData().AsArrayHandle<
    viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
  astr_insitu::require_device_only(physical);
  viskores::cont::DataSet data;
  data.SetCellSet(result.GetCellSet());data.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",physical));
  for(int f=0;f<fields;++f) {
    const auto field=result.GetPointField("raw"+std::to_string(ids[f]));
    astr_insitu::require_device_only(field.GetData().AsArrayHandle<viskores::cont::ArrayHandle<Real>>());
    data.AddField(field);
  }
  return astr_insitu::clip_runtime_device_mesh(std::move(data),context);
}
}

void astr_insitu::begin_runtime_index_group() {
  clear_runtime_compatible_geometry();
  if(index_group.active) throw std::logic_error("Runtime index group already open");
  index_group=RuntimeIndexGroup{};index_group.active=true;
}
void astr_insitu::end_runtime_index_group() {
  int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  std::printf("ASTR_RUNTIME_INDEX_GROUP rank=%d triangulations=%d scalar_builds=%d owner_reuse=%d retained_after=0\n",
    rank,index_group.triangulations,index_group.scalar_builds,index_group.owner_reuse);
  index_group=RuntimeIndexGroup{};
  clear_runtime_compatible_geometry();
}
extern "C" void astr_insitu_runtime_index_group(int begin) {
  if(begin) astr_insitu::begin_runtime_index_group();else astr_insitu::end_runtime_index_group();
}
extern "C" void astr_insitu_runtime_physical_group(int begin) {
  astr_insitu::clear_runtime_compatible_geometry();
  if(begin) {
    if(physical_group.active) throw std::logic_error("Runtime physical group already open");
    physical_group=RuntimePhysicalGroup{};physical_group.active=true;
  } else {
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    std::printf("ASTR_RUNTIME_PHYSICAL_GROUP rank=%d extractions=%d scalar_builds=%d owner_reuse=%d retained_after=0\n",
      rank,physical_group.extractions,physical_group.scalar_builds,physical_group.owner_reuse);
    physical_group=RuntimePhysicalGroup{};
  }
}

astr_insitu::DeviceMesh astr_insitu::build_runtime_index_mesh(const RuntimeProductView& context,int step,Real time,
    const int* field_ids,int fields,int width,int height,Real* coordinates,Real* values,const Real* references,int selected_field) {
    astr_insitu::DeviceMesh mesh;
    static bool initialized=false;
    if(!initialized) {
      int argc=1;char program[]="astr-runtime-products";char* args[]={program,nullptr};char** argv=args;
      viskores::cont::Initialize(argc,argv);initialized=true;
    }
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
    if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
      viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    const viskores::Id nodes=viskores::Id(width)*height;
    if(index_group.active && index_group.geometry && (width!=index_group.width || height!=index_group.height ||
        coordinates!=index_group.coordinates || values!=index_group.values ||
        std::vector<int>(field_ids,field_ids+fields)!=index_group.fields))
      throw std::logic_error("A runtime extraction group received a different frozen slice");
    viskores::cont::DataSet data;
    const bool geometry=context.labels[1]=="geometry";
    std::vector<std::string> names;
    for(const auto& field:context.fields) names.push_back(field.second);
    std::shared_ptr<astr_insitu::DeviceGeometryOwner> owner;
    RuntimeIndexKey key{};
    if(nodes) {
      if(width<2 || height<2) throw std::invalid_argument("A rendered index surface requires two-dimensional cells");
      auto xyz=astr_insitu::borrow_cuda_array(reinterpret_cast<astr_insitu::NativeVector*>(coordinates),nodes);
      auto input=astr_insitu::borrow_cuda_array(values,nodes*fields);
      key=scalar_selection(context,field_ids,fields,references,selected_field);
      if(index_group.active) {
        const auto found=index_group.owners.find(key);
        if(found!=index_group.owners.end()) {owner=found->second;++index_group.owner_reuse;}
      }
      if(!owner) {
        if(index_group.active && index_group.geometry) data=index_group.grid;
        else {
          viskores::cont::ArrayHandle<viskores::Id> indices;
          viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
          invoke(astr_insitu::PackSliceTriangle(width),
            viskores::cont::ArrayHandleIndex(viskores::Id(6)*(width-1)*(height-1)),indices);
          astr_insitu::synchronize_device_stage("RuntimePlaneTriangles");
          viskores::cont::CellSetSingleType<> cells;
          cells.Fill(nodes,viskores::CELL_SHAPE_TRIANGLE,3,indices);data.SetCellSet(cells);
          data.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",xyz));
          astr_insitu::require_device_only(indices);
          if(index_group.active) {
            index_group.grid=data;index_group.geometry=true;
            index_group.coordinates=coordinates;index_group.values=values;
            index_group.width=width;index_group.height=height;
            index_group.fields.assign(field_ids,field_ids+fields);++index_group.triangulations;
          }
        }
        viskores::cont::ArrayHandle<Real> scalar;
        viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
        if(geometry) {
          scalar=borrow_cuda_array(values+nodes*key.a,nodes);
          for(const auto& field:context.fields) {
            const auto found=std::find(field_ids,field_ids+fields,field.first);
            if(found==field_ids+fields) throw std::logic_error("Native geometry is missing a declared field");
            data.AddPointField(field.second,borrow_cuda_array(values+nodes*(found-field_ids),nodes));
          }
        } else {
          invoke(RuntimePlaneScalar(nodes,key.a,key.b,key.c,key.factor,key.magnitude),viskores::cont::ArrayHandleIndex(nodes),input,scalar);
          astr_insitu::synchronize_device_stage("RuntimePlaneScalar");
        }
        data.AddPointField("quantity",scalar);
        astr_insitu::require_device_only(scalar);
        if(index_group.active) ++index_group.scalar_builds;
      }
      astr_insitu::require_device_only(xyz);astr_insitu::require_device_only(input);
    }
    if(!owner) {
      owner=std::make_shared<astr_insitu::DeviceGeometryOwner>(data,step,time,3,"quantity",false,names,std::vector<std::string>{},geometry);
      if(index_group.active && nodes) index_group.owners.emplace(key,owner);
    }
    const auto colors=geometry?nullptr:owner->color_display(astr_insitu::device_display_palette(context.labels[6]),false,
      Real(context.reals[22]),Real(context.reals[23]));
    const auto& view=owner->get();
    mesh.name="runtime_product";mesh.actor_key=context.id;mesh.owner=owner;
    mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
    mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
    mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.arity=3;mesh.draw.device=view.device;
    std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
    mesh.controls["actual_min"]=view.speed_range[0];mesh.controls["actual_max"]=view.speed_range[1];
    return mesh;
}

astr_insitu::DeviceMesh astr_insitu::build_runtime_physical_mesh(const RuntimeProductView& context,int step,Real time,
    const int* field_ids,int fields,const int* global,const int* extent,const int* offset,
    Real* coordinates,Real* values,const Real* references,std::int64_t budget,std::int64_t retained,int selected_field) {
  static bool initialized=false;
  if(!initialized) {
    int argc=1;char program[]="astr-runtime-physical-plane";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  const auto device=viskores::cont::DeviceAdapterTagCuda{};
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(device);
  if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
  const auto key=scalar_selection(context,field_ids,fields,references,selected_field);
  std::vector<int> partition;
  for(const auto* group:{global,extent,offset}) partition.insert(partition.end(),group,group+3);
  const bool q_surface=context.labels[0]=="q_surface";
  std::array<double,14> selection{};std::copy_n(context.reals.begin(),12,selection.begin());
  selection[12]=q_surface?context.reals[25]:0.;selection[13]=q_surface?1.:0.;
  if(physical_group.active && physical_group.frozen && (physical_group.coordinates!=coordinates ||
      physical_group.values!=values || physical_group.fields!=std::vector<int>(field_ids,field_ids+fields) ||
      physical_group.partition!=partition))
    throw std::logic_error("A surface extraction group received another frozen volume");
  RuntimeSurfaceEntry local;
  auto found=physical_group.surfaces.find(selection);
  if(!physical_group.active || found==physical_group.surfaces.end()) {
    viskores::cont::DataSet data;
    const viskores::Id nodes=viskores::Id(extent[0])*extent[1]*extent[2];
    if(budget<=0 || retained<0 || retained>budget) throw std::runtime_error("Runtime surface input budget exceeded");
    if(nodes && q_surface) data=runtime_q_surface(context,field_ids,fields,extent,coordinates,values);
    else if(nodes) {
      const structured_geometry::Partition piece(viskores::Id3(global[0],global[1],global[2]),
        viskores::Id3(extent[0]-1,extent[1]-1,extent[2]-1),viskores::Id3(offset[0],offset[1],offset[2]));
      plane_geometry::Point origin{},normal{};
      for(int d=0;d<3;++d) {origin[d]=Real(context.reals[d]);normal[d]=Real(context.reals[3+d]);}
      const auto plane=plane_geometry::make_plane(origin,normal);
      viskores::cont::Invoker invoke(device);
      viskores::cont::ArrayHandle<NativeVector> zeros;
      invoke(ZeroPlaneVector{},viskores::cont::ArrayHandleIndex(nodes),zeros);
      synchronize_device_stage("RuntimePlaneGeometryInput");
      require_device_only(zeros);
      structured_geometry::PlaneProduct product;
      {
        viskores::cont::Token token;
        const auto vectors=zeros.PrepareForInput(device,token);
        product=structured_geometry::extract_plane(reinterpret_cast<NativeVector*>(coordinates),
          const_cast<NativeVector*>(vectors.GetIteratorBegin()),nullptr,piece,plane.origin,plane.normal,
          std::uint64_t(budget),std::uint64_t(retained)+nodes*sizeof(NativeVector),true,true,true);
      }
      if(product.data.GetNumberOfCoordinateSystems()) {
        data.SetCellSet(product.data.GetCellSet());data.AddCoordinateSystem(product.data.GetCoordinateSystem());
        auto input=borrow_cuda_array(values,nodes*fields);
        for(int f=0;f<fields;++f) {
          viskores::cont::ArrayHandle<Real> scalar;
          invoke(InterpolateRuntimePlaneField(nodes*f),product.interpolation,input,scalar);
          synchronize_device_stage("InterpolateRuntimePlaneField");
          require_device_only(scalar);
          data.AddPointField("raw"+std::to_string(field_ids[f]),scalar);
        }
        require_device_only(input);
        data=clip_runtime_device_mesh(std::move(data),context);
      }
    }
    local.data=std::move(data);
    if(physical_group.active) {
      found=physical_group.surfaces.emplace(selection,std::move(local)).first;
      physical_group.frozen=true;physical_group.coordinates=coordinates;
      physical_group.values=values;physical_group.fields.assign(field_ids,field_ids+fields);
      physical_group.partition=std::move(partition);++physical_group.extractions;
    }
  }
  auto& entry=physical_group.active?found->second:local;
  const auto& data=entry.data;
  const bool geometry=context.labels[1]=="geometry";
  std::vector<std::string> names;
  for(const auto& field:context.fields) names.push_back(field.second);
  std::shared_ptr<DeviceGeometryOwner> owner;
  if(physical_group.active) {
    const auto existing=entry.owners.find(key);
    if(existing!=entry.owners.end()) {owner=existing->second;++physical_group.owner_reuse;}
  }
  if(!owner) {
    viskores::cont::DataSet selected;
    if(data.GetNumberOfCoordinateSystems()) {
      auto field=[&](int component) {
        return data.GetPointField("raw"+std::to_string(field_ids[component])).GetData().AsArrayHandle<viskores::cont::ArrayHandle<Real>>();
      };
      const auto a=field(key.a),b=field(key.magnitude?key.b:key.a),c=field(key.magnitude?key.c:key.a);
      viskores::cont::ArrayHandle<Real> scalar;
      viskores::cont::Invoker invoke(device);
      if(geometry) scalar=a;
      else {
        invoke(RuntimeSurfaceScalar(key.factor,key.magnitude),a,b,c,scalar);
        synchronize_device_stage("RuntimePhysicalSurfaceScalar");
      }
      require_device_only(a);require_device_only(b);require_device_only(c);require_device_only(scalar);
      selected.SetCellSet(data.GetCellSet());selected.AddCoordinateSystem(data.GetCoordinateSystem());
      selected.AddPointField("quantity",scalar);
      for(const auto& field:context.fields) selected.AddPointField(field.second,
        data.GetPointField("raw"+std::to_string(field.first)).GetData());
      if(physical_group.active) ++physical_group.scalar_builds;
    }
    owner=std::make_shared<DeviceGeometryOwner>(selected,step,time,3,"quantity",false,names,std::vector<std::string>{},geometry);
    if(physical_group.active) entry.owners.emplace(key,owner);
  }
  const auto colors=geometry?nullptr:owner->color_display(device_display_palette(context.labels[6]),false,
    Real(context.reals[22]),Real(context.reals[23]));
  const auto& view=owner->get();
  DeviceMesh mesh;mesh.name="runtime_product";mesh.actor_key=context.id;mesh.owner=owner;
  mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
  mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
  mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.arity=3;mesh.draw.device=view.device;
  std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
  mesh.controls["actual_min"]=view.speed_range[0];mesh.controls["actual_max"]=view.speed_range[1];
  return mesh;
}
#endif

extern "C" int astr_insitu_render_physical_product(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,const char* id,const char* source,const char* definition,const char* directory,
    std::int64_t version,const char* labels,const int* words,const double* settings,
    const int* field_ids,int fields,const int* global,const int* extent,const int* offset,
    Real* coordinates,Real* values,const Real* references,int provider,const Real* statistics,std::int64_t samples,
    int selected_field,std::int64_t budget,std::int64_t retained,std::int64_t host_budget) {
  const auto comm=MPI_Comm_f2c(fcomm);
  const double started=astr_insitu::runtime_product_clock();
  std::string error;
  astr_insitu::DeviceMesh mesh;
  astr_insitu::RuntimeProductView context;
#ifdef ASTR_INSITU_DEVICE_RENDERING
  try {
    astr_insitu::LocalAllocationRecovery local_allocations;
    context=astr_insitu::runtime_product_view(id,source,definition,directory,version,labels,words,settings);
    context.host_budget=host_budget;
    context.statistics_provider=provider;context.samples=samples;
    std::copy_n(statistics,3,context.statistics.begin());
    astr_insitu::prepare_runtime_product_allocations(context,fcomm,step);
    mesh=astr_insitu::build_runtime_physical_mesh(context,step,time,field_ids,fields,global,extent,offset,
      coordinates,values,references,budget,retained,selected_field);
    if(!physical_group.active && std::string(pipeline?pipeline:"")!="compatible")
      std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(mesh.owner)->retain_display_only();
    int rank=0;MPI_Comm_rank(comm,&rank);
    std::printf("ASTR_RUNTIME_PHYSICAL_SURFACE rank=%d id=%s kind=%s step=%d points=%lld triangles=%lld "
      "field_d2h_bytes=0 geometry_host_bytes=0\n",rank,id,context.labels[0].c_str(),step,static_cast<long long>(mesh.draw.points),
      static_cast<long long>(mesh.draw.cells));
  } catch(const std::exception& caught) {error=caught.what();}
#else
  error="Runtime physical surfaces require the strict device rendering build";
#endif
  return astr_insitu::publish_runtime_product(pipeline,backend,script,fcomm,step,time,context,std::move(mesh),error,started);
}

extern "C" int astr_insitu_render_index_product(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,const char* id,const char* source,const char* definition,const char* directory,
    std::int64_t version,const char* labels,const int* words,const double* settings,
    const int* field_ids,int fields,int width,int height,Real* coordinates,Real* values,const Real* references,
    int provider,const Real* statistics,std::int64_t samples,int selected_field,std::int64_t host_budget)
{
  const auto comm=MPI_Comm_f2c(fcomm);
  const double started=astr_insitu::runtime_product_clock();
  std::string error;
  astr_insitu::DeviceMesh mesh;
  astr_insitu::RuntimeProductView context;
#ifdef ASTR_INSITU_DEVICE_RENDERING
  try {
    astr_insitu::LocalAllocationRecovery local_allocations;
    context=astr_insitu::runtime_product_view(id,source,definition,directory,version,labels,words,settings);
    context.host_budget=host_budget;
    context.statistics_provider=provider;context.samples=samples;
    std::copy_n(statistics,3,context.statistics.begin());
    astr_insitu::prepare_runtime_product_allocations(context,fcomm,step);
    mesh=astr_insitu::build_runtime_index_mesh(context,step,time,field_ids,fields,width,height,coordinates,values,references,selected_field);
    const auto owner=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(mesh.owner);
    if(!index_group.active && std::string(pipeline?pipeline:"")!="compatible") owner->retain_display_only();
    int rank=0;MPI_Comm_rank(comm,&rank);
    std::printf("ASTR_RUNTIME_PLANE rank=%d id=%s step=%d points=%lld triangles=%lld "
      "field_d2h_bytes=0 geometry_host_bytes=0\n",rank,id,step,static_cast<long long>(mesh.draw.points),
      static_cast<long long>(mesh.draw.cells));
  } catch(const std::exception& caught) {error=caught.what();}
#else
  error="Runtime device index surfaces require the strict device rendering build";
#endif
  return astr_insitu::publish_runtime_product(pipeline,backend,script,fcomm,step,time,context,std::move(mesh),error,started);
}

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
