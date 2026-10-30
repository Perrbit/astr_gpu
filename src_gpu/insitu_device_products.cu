#include "insitu_device_geometry.h"
#include "insitu_device_streamlines.h"
#include "insitu_compact_mesh.h"
#include "insitu_device_product_audit.h"
#include "insitu_device_wall.h"
#include "insitu_product_dispatch.h"
#include "insitu_product_failure.h"
#include "insitu_allocation_budget.h"
#include "insitu_m12_products.h"
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <fstream>
#include <set>
#include <optional>

extern "C" void astr_insitu_device_allocation_preflight(std::size_t,const char*);
extern "C" int astr_insitu_allocation_guard_active();
extern "C" void astr_insitu_runtime_failure(const char*);
using Real=astr::numerical_real;

namespace {
// Every operation is rank-local; the following agreement precedes any next
// distributed tracer call. A damaged device context is never recovered here.
struct RuntimeTraceStages {
  MPI_Comm comm;
  const char* id;
  int step;
  template<class Operation>
  auto operator()(const char* stage,Operation operation) const -> decltype(operation()) {
    using Result=decltype(operation());
    std::optional<std::conditional_t<std::is_void_v<Result>,bool,Result>> result;
    std::string error;
    try {
      astr_insitu::LocalAllocationRecovery local_allocations;
#ifdef ASTR_BUILD_TESTING
      if(const char* fault=std::getenv("ASTR_INSITU_TEST_TRACE_ALLOCATION_REFUSAL")) {
        int rank=0;astr_insitu::trace_mpi(MPI_Comm_rank(comm,&rank));
        if(std::string(fault)==std::string(id)+":"+std::to_string(step)+":"+std::to_string(rank)+":"+stage)
          astr_insitu_device_allocation_preflight(std::numeric_limits<std::size_t>::max(),"runtime distributed trace test");
      }
#endif
      if constexpr(std::is_void_v<Result>) {operation();result.emplace(true);}
      else result.emplace(operation());
    } catch(const std::exception& caught) {error=caught.what();}
    if(astr_insitu::consensus_product_failure(comm,error)) throw astr_insitu::CollectiveTraceError(error);
    if constexpr(!std::is_void_v<Result>) return std::move(*result);
  }
};

struct SliceCache {
  viskores::Id3 dimensions{0,0,0},offset{0,0,0};
  int device=-1;
  bool valid=false;
  astr_insitu::CompactMesh geometry;
} slice_cache;
bool initialized=false;

struct ProductStage {
  const char* name;
  int rank,step;
  double started;
  bool running=true;
  ProductStage(const char* label,const char* range,int communicator,int frame):name(label),step(frame) {
    MPI_Comm_rank(MPI_Comm_f2c(communicator),&rank);
    started=MPI_Wtime();nvtxRangePushA(range);
  }
  void finish() {
    if(!running) return;
    const double seconds=MPI_Wtime()-started;
    nvtxRangePop();running=false;
    const char* option=std::getenv("ASTR_INSITU_TIMING");
    if(option && (std::string(option)=="1" || std::string(option)=="true" || std::string(option)=="TRUE" ||
       std::string(option)=="t" || std::string(option)=="T" || std::string(option)=="on" || std::string(option)=="ON"))
      std::printf("ASTR_INSITU_STAGE_TIMING %s %d %d %.17g\n",name,step,rank,seconds);
  }
  ~ProductStage() {finish();}
};

void append_coordinate(astr_insitu::CompactMesh& mesh,const astr_insitu::NativeVector& xyz) {
  for(int d=0;d<3;++d) {
    if(!std::isfinite(xyz[d])) throw std::runtime_error("Nonfinite compact coordinate");
    mesh.coordinates[d].push_back(xyz[d]);
  }
}

astr_insitu::CompactMesh geometry_mesh(const viskores::cont::DataSet& data,bool surface,
    const viskores::Id3& dimensions,const viskores::Id3& offset) {
  astr_insitu::CompactMesh mesh;
  mesh.name=surface?"q_surface":"velocity_slice";mesh.shape=surface?"tri":"quad";
  for(const char* field:{"u","v","w"}) mesh.point_fields[field];
  if(surface) mesh.point_fields["Q_rs"];
  if(!data.GetNumberOfCoordinateSystems() || !data.GetCoordinateSystem().GetNumberOfPoints()) return mesh;
  const auto nodes=data.GetCoordinateSystem().GetNumberOfPoints();
  if(surface) {
    auto coords=data.GetCoordinateSystem().GetData().AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
    astr_insitu::require_device_only(coords);
    const auto points=coords.ReadPortal();
    for(viskores::Id i=0;i<nodes;++i) append_coordinate(mesh,points.Get(i));
    const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
    const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{}).ReadPortal();
    for(viskores::Id i=0;i<cells.GetNumberOfCells()*3;++i) mesh.connectivity.push_back(indices.Get(i));
    auto q=data.GetPointField("Q_rs").GetData().AsArrayHandle<viskores::cont::ArrayHandle<Real>>().ReadPortal();
    for(viskores::Id i=0;i<nodes;++i) mesh.point_fields["Q_rs"].push_back(q.Get(i));
  } else {
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess) throw std::runtime_error("Cannot identify static slice device");
    if(!slice_cache.valid || slice_cache.dimensions!=dimensions || slice_cache.offset!=offset || slice_cache.device!=device) {
      slice_cache=SliceCache{};slice_cache.dimensions=dimensions;slice_cache.offset=offset;slice_cache.device=device;
      const Real h=Real(2.)*std::acos(-Real(1.))/Real(32.);
      for(viskores::Id j=0;j<dimensions[1];++j) for(viskores::Id i=0;i<dimensions[0];++i)
        append_coordinate(slice_cache.geometry,astr_insitu::NativeVector(offset[0]*h+i*h,offset[1]*h+j*h,4*h));
      for(viskores::Id j=0;j<dimensions[1]-1;++j) for(viskores::Id i=0;i<dimensions[0]-1;++i) {
        const auto a=j*dimensions[0]+i;
        slice_cache.geometry.connectivity.insert(slice_cache.geometry.connectivity.end(),
          {a,a+1,a+dimensions[0]+1,a+dimensions[0]});
      }
      slice_cache.valid=true;
    }
    if(slice_cache.geometry.coordinates[0].size()!=std::uint64_t(nodes))
      throw std::runtime_error("Static slice identity differs from device plane");
    for(int d=0;d<3;++d) mesh.coordinates[d]=slice_cache.geometry.coordinates[d];
    mesh.connectivity=slice_cache.geometry.connectivity;
  }
  auto vectors=data.GetPointField("velocity").GetData().AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
  auto scalar=data.GetPointField("u").GetData().AsArrayHandle<viskores::cont::ArrayHandle<Real>>();
  astr_insitu::require_device_only(vectors);astr_insitu::require_device_only(scalar);
  const auto v=vectors.ReadPortal();const auto u=scalar.ReadPortal();
  for(viskores::Id i=0;i<nodes;++i) {
    mesh.point_fields["u"].push_back(u.Get(i));
    mesh.point_fields["v"].push_back(v.Get(i)[1]);mesh.point_fields["w"].push_back(v.Get(i)[2]);
  }
  return mesh;
}

astr_insitu::CompactMesh streamline_mesh(const astr_insitu::DeviceStreamlines& trace,const std::string& name) {
  astr_insitu::CompactMesh mesh;mesh.name=name;mesh.shape="line";
  for(const char* field:{"u","v","w","accepted_length","termination_status","untravelled_arc_length"})
    mesh.point_fields[field];
  const Real pi=std::acos(-Real(1.));
  int rank=0;
  if(MPI_Comm_rank(MPI_COMM_WORLD,&rank)!=MPI_SUCCESS) throw std::runtime_error("Cannot identify trajectory rank");
  for(int i=0;i<trace.particles;++i) if(trace.subminimum_stop[i])
    std::printf("ASTR_INSITU_TRAJECTORY_TERMINATION rank=%d product=%s seed=%d direction=%d status=3 "
      "reason=sub_minimum_remaining accepted=%.17g remaining=%.17g minimum=%.17g\n",rank,name.c_str(),
      i%trace.seed_count,i<trace.seed_count?1:-1,trace.final_state[i][4],pi-trace.final_state[i][4],trace.minimum_step);
  const bool mean=name.find("mean_")==0;
  std::string kind;
  if(mean) {
    kind=name=="mean_reynolds_streamlines"?"reynolds":"favre";
    for(const char* component:{"u","v","w"}) mesh.point_fields[std::string("mean_")+component+"_"+kind];
  }
  for(const auto& segment:trace.segments) {
    if(segment.points.size()<2) continue;
    const auto base=mesh.coordinates[0].size();
    for(std::size_t i=0;i<segment.points.size();++i) {
      const auto& p=segment.points[i];const auto& v=segment.velocity[i];
      append_coordinate(mesh,astr_insitu::NativeVector(p[0],p[1],p[2]));
      int d=0;
      for(const char* component:{"u","v","w"}) {
        mesh.point_fields[component].push_back(v[d]);
        if(mean) {
          if(segment.integrating_velocity.size()!=segment.points.size())
            throw std::runtime_error("Mean trajectory is missing its integrating velocity");
          mesh.point_fields[std::string("mean_")+component+"_"+kind].push_back(segment.integrating_velocity[i][d]);
        }
        ++d;
      }
      mesh.point_fields["accepted_length"].push_back(p[3]);
      const auto& final=trace.final_state[segment.seed+(segment.direction>0?0:trace.seed_count)];
      mesh.point_fields["termination_status"].push_back(final.status);
      mesh.point_fields["untravelled_arc_length"].push_back(std::max(Real(0.),pi-final[4]));
      if(i) {
        mesh.connectivity.push_back(base+i-1);mesh.connectivity.push_back(base+i);
        mesh.seed_ids.push_back(segment.seed);mesh.directions.push_back(segment.direction);
      }
    }
  }
  return mesh;
}

struct ConstantVelocity : viskores::worklet::WorkletMapField {
  int axis=0;
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(viskores::Id,astr_insitu::NativeVector& v) const {v=astr_insitu::NativeVector(Real(0.));v[axis]=Real(1.);}
};
#ifdef ASTR_INSITU_DEVICE_RENDERING
struct PhysicalTraceAudit : viskores::worklet::WorkletMapField {
  astr_insitu::DeviceGeometryView view;
  bool crossing;
  int axis;
  using ControlSignature=void(FieldIn,FieldIn,FieldIn,ExecObject,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5);
  template<class Grid>
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& point,const astr_insitu::NativeVector& velocity,
      viskores::Id index,const Grid& grid,viskores::Vec<Real,2>& differences) const {
    viskores::VecVariable<astr_insitu::NativeVector,2> expected;
    const auto status=grid.Evaluate(point,Real(0.),expected);
    Real error=Real(0.),display=Real(0.);
    if(!status.CheckOk() || expected.GetNumberOfComponents()!=1) error=INFINITY;
    else {
      for(int d=0;d<3;++d) {
        if(!isfinite(expected[0][d])) error=INFINITY;
        else error=fmax(error,fabs(velocity[d]-expected[0][d]));
      }
      error=fmax(error,fabs(view.u[index]-expected[0][0]));
    }
    for(int d=0;d<3;++d) {
      if(!isfinite(velocity[d]) || !isfinite(point[d])) error=INFINITY;
      display=fmax(display,fabs(Real(view.display_coordinates[index][d])-Real(float(point[d]))));
    }
    const Real speed=sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
    display=fmax(display,fabs(Real(view.display_speed[index])-Real(float(speed))));
    if(crossing) {
      const Real pi=acos(-Real(1.)),seed[3]={pi/Real(2.),pi/Real(8.),pi/Real(4.)};
      const int along=(axis+1)%3,fixed=(axis+2)%3;
      const Real number=(point[along]-seed[1])/(Real(3.)*pi/Real(4.)/Real(15.));
      error=fmax(error,fabs(point[along]-(seed[1]+round(number)*(Real(3.)*pi/Real(4.)/Real(15.)))));
      error=fmax(error,fabs(point[fixed]-seed[2]));
      error=fmax(error,fmax(Real(0.),seed[0]-point[axis]));
      error=fmax(error,fmax(Real(0.),point[axis]-Real(3.)*pi/Real(2.)));
    }
    differences=viskores::Vec<Real,2>(error,display);
  }
};
struct TraceAuditMaximum {
  VISKORES_EXEC_CONT viskores::Vec<Real,2> operator()(const viskores::Vec<Real,2>& a,
      const viskores::Vec<Real,2>& b) const {
    return {viskores::Max(a[0],b[0]),viskores::Max(a[1],b[1])};
  }
};

astr_insitu::ProductAudit audit_physical_trace(const astr_insitu::DeviceGeometryView& view,
    const astr_insitu::SampleVectorComponents& coordinates,const Real* source,
    const viskores::Id3& extent,const viskores::Id3& offset,const viskores::Id3& global_cells,bool crossing,int axis) {
  astr_insitu::ProductAudit result;
  if(!view.points) return result;
  const auto fields=astr_insitu::view_component_halo(const_cast<Real*>(source),extent+viskores::Id3(7));
  const auto physical=astr_insitu::crop_physical_trace_halo(coordinates,fields,fields,extent,offset,global_cells);
  astr_insitu::PhysicalTraceOwner owner(physical,extent,offset,global_cells);
  astr_insitu::PhysicalTraceGrid grid(owner,astr_insitu::PhysicalTraceField(physical.trace));
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  PhysicalTraceAudit worklet;worklet.view=view;worklet.crossing=crossing;worklet.axis=axis;
  viskores::cont::ArrayHandle<viskores::Vec<Real,2>> differences;
  invoke(worklet,astr_insitu::borrow_cuda_array(const_cast<astr_insitu::NativeVector*>(view.coordinates),view.points),
    astr_insitu::borrow_cuda_array(const_cast<astr_insitu::NativeVector*>(view.velocity),view.points),
    viskores::cont::ArrayHandleIndex(view.points),grid,differences);
  astr_insitu::synchronize_device_stage("ReadOnlyPhysicalTraceAudit");
  astr_insitu::require_device_only(differences);
  const auto maximum=viskores::cont::Algorithm::Reduce(differences,viskores::Vec<Real,2>(Real(0.)),TraceAuditMaximum{});
  result.field=maximum[0];result.display=maximum[1];
  if(!std::isfinite(result.field) || (sizeof(Real)==8 && result.field>Real(2e-10)) || result.display!=Real(0.))
    throw std::runtime_error("Physical trace numerical/display acceptance failed");
  return result;
}

astr_insitu::DeviceMesh resident_mesh(const char* name,const viskores::cont::DataSet& data,
    int step,Real time,int arity=3,const Real* audit_halo=nullptr,
    const astr_insitu::DeviceDiagnostics* audit_diagnostics=nullptr,
    viskores::Id3 extent={0,0,0},viskores::Id3 offset={0,0,0},int global_cells=32,int rank=0,
    const astr_insitu::NativeVector* physical_source=nullptr,std::uint64_t host_budget=0,
    const astr_insitu::SampleVectorComponents* trace_coordinates=nullptr,int trace_axis=0,
    bool speed_colors=false,Real iso=Real(.25),viskores::Id3 global_dimensions=viskores::Id3(0),Real speed_max=Real(1.)) {
  if(global_dimensions==viskores::Id3(0)) global_dimensions=viskores::Id3(global_cells);
  const bool image_only=!astr_insitu::retain_quantitative_product_fields();
  auto owner=std::make_shared<astr_insitu::DeviceGeometryOwner>(data,step,time,arity,nullptr,image_only);
  static const auto palette=astr_insitu::device_display_palette();
  const bool speed=speed_colors || global_cells==256;
  const auto colors=owner->color_display(palette,speed,speed?Real(0.):-Real(1.),speed?speed_max:Real(1.));
  const auto& view=owner->get();
#ifdef ASTR_BUILD_TESTING
  if(physical_source) astr_insitu::test_curve_surface_oracle(data,view,rank,step,host_budget);
  if(trace_coordinates) astr_insitu::test_curve_trace_oracle(data,view,name,rank,step,host_budget);
#endif
  if(audit_halo && std::getenv("ASTR_INSITU_RESIDENT_AUDIT") &&
      std::string(std::getenv("ASTR_INSITU_RESIDENT_AUDIT"))=="1") {
    viskores::cont::Token token;
    const astr_insitu::NativeVector* computational=nullptr;
    if(physical_source && view.points) {
      const auto field=data.GetPointField("_astr_computational_xyz").GetData().AsArrayHandle<
        viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
      astr_insitu::require_device_only(field);
      computational=static_cast<const astr_insitu::NativeVector*>(field.GetBuffers()[0].ReadPointerDevice(
        viskores::cont::DeviceAdapterTagCuda{},token));
    }
    const auto result=trace_coordinates?audit_physical_trace(view,*trace_coordinates,audit_halo,
      extent,offset,global_dimensions,std::string(name)=="crossing_streamlines",trace_axis):
      astr_insitu::audit_device_product(view,audit_halo,audit_diagnostics,
      extent,offset,global_cells,std::string(name)=="q_surface",std::string(name)=="velocity_slice",
      std::string(name)=="crossing_streamlines",computational,physical_source,iso);
    std::printf("ASTR_INSITU_RESIDENT_AUDIT rank=%d step=%d product=%s field_maxabs=%.17g "
      "display_maxabs=%.17g points=%lld\n",rank,step,name,result.field,result.display,
      static_cast<long long>(view.points));
  }
  astr_insitu::DeviceMesh mesh;
  mesh.name=name;mesh.owner=owner;
  mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
  mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);
  mesh.draw.indices=view.display_connectivity;
  mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.device=view.device;mesh.draw.arity=view.arity;
  std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
  if(image_only) owner->retain_display_only();
  return mesh;
}
void report_resident_trace(const astr_insitu::DeviceStreamlines& trace,const char* name,int rank,int axis=0) {
  std::printf("ASTR_INSITU_RESIDENT_TRAJECTORY rank=%d product=%s rounds=%d transfers=%d "
    "control_read_bytes=%llu geometry_device_bytes=%llu owner_query_read_bytes=%llu "
    "seed_layout=%s seeds=%d particles=%d control_round_limit_bytes=%llu owner_round_limit_bytes=%llu\n",
    rank,name,trace.rounds,trace.transfers,
    static_cast<unsigned long long>(trace.control_read_bytes),static_cast<unsigned long long>(trace.geometry_bytes),
    static_cast<unsigned long long>(trace.owner_query_read_bytes),trace.seed_layout.c_str(),trace.seed_count,trace.particles,
    static_cast<unsigned long long>(trace.particles*sizeof(astr_insitu::RK45TraceWorklet::State)),
    static_cast<unsigned long long>(trace.particles*sizeof(viskores::Id)));
  if(std::string(name)=="crossing_streamlines") {
    Real worst=Real(0.);
    const Real pi=std::acos(-Real(1.));
    const auto seeds=astr_insitu::tgv_streamline_seeds(trace.seed_layout);
    for(int i=0;i<trace.seed_count;++i) {
      const auto& point=trace.final_state[i];
      const Real original[3]={seeds[i][0]+pi,seeds[i][1],seeds[i][2]};
      for(int d=0;d<3;++d) {
        const Real error=std::abs(point[d]-original[(d-axis+3)%3]);
        if(!std::isfinite(error)) throw std::runtime_error("Nonfinite resident crossing endpoint");
        worst=std::max(worst,error);
      }
    }
    if(sizeof(Real)==8 && worst>Real(2e-10)) throw std::runtime_error("Resident crossing endpoint acceptance failed");
    std::printf("ASTR_INSITU_RESIDENT_CROSSING rank=%d endpoint_maxabs=%.17g\n",rank,worst);
  }
  for(int i=0;i<trace.particles;++i) if(trace.subminimum_stop[i])
    std::printf("ASTR_INSITU_TRAJECTORY_TERMINATION rank=%d product=%s seed=%d direction=%d status=3 "
      "reason=sub_minimum_remaining accepted=%.17g remaining=%.17g minimum=%.17g\n",rank,name,
      i%trace.seed_count,i<trace.seed_count?1:-1,trace.final_state[i][4],trace.maximum_length-trace.final_state[i][4],trace.minimum_step);
  if(trace.seed_layout=="bl-layered256") for(int i=0;i<trace.particles;++i) {
    const auto& state=trace.final_state[i];
    const int status=state.status;
    const char* reason=status==astr_insitu::TraceSteps?"step_budget":
      (status==astr_insitu::TraceLength?"arc_length":(status==astr_insitu::Outside?"domain_exit":"sub_minimum_remaining"));
    std::printf("ASTR_INSITU_M12_TRAJECTORY rank=%d product=%s seed=%d direction=%d status=%d reason=%s "
      "accepted_steps=%d accepted_length=%.17g remaining=%.17g endpoint=%.17g,%.17g,%.17g\n",rank,name,
      i%trace.seed_count,i<trace.seed_count?1:-1,status,reason,state.steps,state[4],
      trace.maximum_length-state[4],state[0],state[1],state[2]);
  }
}
#endif
}

extern "C" int astr_insitu_device_render_walls(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,int im,int km,int nw,int wall_kind,Real* coordinates,Real* fields,
    int mean_requested,int covered,Real duration,Real window_start,Real window_end,Real* means,
    std::int64_t budget,std::int64_t reserve,std::int64_t retained)
try {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  const auto comm=MPI_Comm_f2c(fcomm);
  const std::string route=pipeline?pipeline:"";
  if(route!="standard-device" && route!="direct-device")
    throw std::invalid_argument("Device walls require a strict pipeline; no host fallback");
  const bool curve=wall_kind==1,air5=wall_kind==2;
  if(im<1 || km<1 || nw<0 || nw>2 || im>32 || km>32 || wall_kind<0 || wall_kind>2 || (air5 && nw>1) ||
      step<0 || !std::isfinite(time))
    throw std::invalid_argument("Invalid bounded device wall identity");
  if((mean_requested!=0 && mean_requested!=1) || (covered!=0 && covered!=1) ||
      covered!=int(mean_requested && duration>Real(0.)) || !std::isfinite(duration) || duration<Real(0.) ||
      (mean_requested && (!std::isfinite(window_start) || !std::isfinite(window_end) || window_end<=window_start)) ||
      (covered && nw && !means))
    throw std::invalid_argument("Invalid device wall mean clock/coverage");
  const std::uint64_t nodes=std::uint64_t(im+1)*(km+1)*nw,bound=nodes*2048*(covered?2:1)+65536;
  std::size_t free_bytes=0,total_bytes=0;
  const auto required=std::max<std::int64_t>(reserve,1073741824);
  if(budget<=0 || reserve<0 || retained<0 || retained>budget || bound>std::uint64_t(budget-retained) ||
      cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess || free_bytes<std::uint64_t(required) ||
      bound>free_bytes-required)
    throw std::runtime_error("Device wall geometry controlled allocation budget/reserve");
  if(!initialized) {
    int argc=1;char program[]="astr-insitu-device";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
  int rank=0;astr_insitu::trace_mpi(MPI_Comm_rank(comm,&rank));
  std::vector<astr_insitu::DeviceMesh> products;
  const char* channel_names[]={"wall_pressure","wall_shear_x","wall_heat_into_gas"};
  const char* air5_names[]={"pressure","temperature","vibrational_temperature","Y_N2","Y_O2",
    "Y_N","Y_O","Y_NO","wall_shear_x","wall_heat_into_gas"};
  const int air5_components[]={6,4,5,7,8,9,10,11,12,16};
  const Real channel_ranges[3][2]={{Real(0.),Real(12.)},{-Real(.02),Real(.02)},{-Real(.2),Real(.2)}};
  const Real curve_ranges[3][2]={{Real(80.),Real(120.)},{-Real(.2),Real(.2)},{Real(0.),Real(15000.)}};
  const Real air5_ranges[10][2]={{Real(0.),Real(60000.)},{Real(1500.),Real(3500.)},{Real(1500.),Real(3500.)},
    {Real(0.),Real(1.)},{Real(0.),Real(1.)},{Real(0.),Real(1.)},{Real(0.),Real(1.)},{Real(0.),Real(1.)},{-Real(1.),Real(1.)},{-Real(60000.),Real(60000.)}};
  const auto names=air5?air5_names:channel_names;
  const auto ranges=air5?air5_ranges:(curve?curve_ranges:channel_ranges);
  const int count=air5?10:3;
  const auto palette=astr_insitu::device_display_palette();
  // Catalyst sources bind on the first frame; retain empty mean channels until covered.
  for(int mean=0;mean<1+mean_requested;++mean) for(int c=0;c<count;++c) {
    const bool placeholder=mean && !covered;
    const std::string name=(mean?"mean_":"")+std::string(names[c]);
    auto product=astr_insitu::extract_bc41_wall(coordinates,mean?means:fields,im,km,placeholder?0:nw,
      air5?air5_components[c]:c,name.c_str(),air5?18:4,air5?17:3);
    Real total_area=Real(0.);
    astr_insitu::trace_mpi(MPI_Allreduce(&product.audit[0],&total_area,1,astr_insitu::numerical_mpi_type(),MPI_SUM,comm));
    if(!placeholder && (!std::isfinite(total_area) || total_area<=Real(0.) ||
        (sizeof(Real)==8 && !curve && std::abs(total_area-(air5?Real(.08)*Real(.002):Real(4.)*std::acos(-Real(1.))*std::acos(-Real(1.))))>
          (air5?Real(2e-17):Real(2e-10)))))
      throw std::runtime_error("Wall global facet area differs");
    Real quadrature_area=Real(0.);
    astr_insitu::trace_mpi(MPI_Allreduce(&product.measure[0],&quadrature_area,1,astr_insitu::numerical_mpi_type(),MPI_SUM,comm));
    if(!placeholder && (!std::isfinite(quadrature_area) || quadrature_area<=Real(0.)))
      throw std::runtime_error("Wall global quadrature area differs");
    auto owner=std::make_shared<astr_insitu::DeviceGeometryOwner>(product.data,step,time,3,name.c_str());
    const auto colors=owner->color_display(palette,false,ranges[c][0],ranges[c][1]);
    const auto& view=owner->get();
    astr_insitu::DeviceMesh mesh;mesh.name=name;mesh.owner=owner;
    mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
    mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
    mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.device=view.device;mesh.draw.arity=3;
    std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
    if(mean_requested) {
      mesh.controls["mean_requested"]=Real(1.);
      mesh.controls["statistics_duration"]=duration;
      mesh.controls["statistics_window_start"]=window_start;
      mesh.controls["statistics_window_end"]=window_end;
    }
    products.push_back(std::move(mesh));
    if(placeholder) continue;
    std::printf("ASTR_INSITU_DEVICE_WALL_GEOMETRY rank=%d step=%d product=%s points=%lld triangles=%lld "
      "local_area=%.17g global_area=%.17g invalid_triangles=%.0f geometry_host_bytes=0\n",rank,step,name.c_str(),
      static_cast<long long>(view.points),static_cast<long long>(view.cells),product.audit[0],total_area,product.audit[1]);
    std::printf("ASTR_INSITU_DEVICE_WALL_MEASURE rank=%d step=%d product=%s curve=%d "
      "local_quad_area=%.17g global_quad_area=%.17g\n",rank,step,name.c_str(),int(curve),product.measure[0],quadrature_area);
    std::printf("ASTR_INSITU_DEVICE_WALL_SCALAR rank=%d step=%d product=%s minimum=%.17g maximum=%.17g "
      "display_minimum=%.17g display_maximum=%.17g control_values=2 field_download_bytes=0\n",
      rank,step,name.c_str(),view.speed_range[0],view.speed_range[1],ranges[c][0],ranges[c][1]);
    if(!astr_insitu::retain_quantitative_product_fields()) owner->retain_display_only();
  }
  return astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,
    air5?"air5_walls":(curve?"curve_walls":"channel_walls"),std::move(products),bool(covered));
#else
  throw std::runtime_error("Device wall rendering is not built; no host fallback");
#endif
} catch(const std::exception& error) {
  std::fprintf(stderr,"ASTR device wall products: %s\n",error.what());
  // A local failure must not leave other ranks waiting in product collectives.
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
} catch(...) {
  std::fprintf(stderr,"ASTR device wall products: unknown exception\n");
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
}

extern "C" int astr_insitu_device_render_m12_planes(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,int nx,int ny,int nz,int ox,int gy,int gz,int global_cells,int plane_j,
    Real* positions,Real* fields,std::int64_t budget,std::int64_t reserve,std::int64_t retained)
try {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  const auto comm=MPI_Comm_f2c(fcomm);
  const viskores::Id3 global(global_cells,gy,gz),extent(nx,ny,nz),offset(ox,0,0);
  if(std::string(pipeline?pipeline:"")!="standard-device" || !astr_insitu::m12_hd_grid(global) ||
      ny!=gy || nz!=gz || nx<=0 || ox<0 || ox+nx>global_cells || plane_j<0 || plane_j>=ny ||
      step<0 || !std::isfinite(time) || !positions || !fields)
    throw std::invalid_argument("Local M12 planes require the admitted standard-device x-slab inputs");
  if(!initialized) {
    int argc=1;char program[]="astr-insitu-device";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
#ifdef ASTR_VISKORES_ALLOCATION_PREFLIGHT
  if(!astr_insitu_allocation_guard_active())
    throw std::runtime_error("Local M12 plane allocations require an active resource guard");
  viskores::cont::GetRuntimeDeviceTracker().SetThreadFriendlyMemAlloc(false);
  viskores::cont::cuda::internal::CudaAllocator::SetAllocationPreflight(astr_insitu_device_allocation_preflight);
  astr_insitu_device_allocation_preflight(0,"local M12 plane admission");
#endif
  if(budget<=0 || reserve<0 || retained<0 || retained>budget)
    throw std::invalid_argument("Local M12 plane input budget");
  std::vector<astr_insitu::DeviceMesh> meshes;
  astr_insitu::append_m12_planes(meshes,comm,step,time,global,extent,offset,positions,fields,nullptr,
    plane_j,std::uint64_t(budget),std::uint64_t(retained),true);
  for(const char* name:{"q_surface","instantaneous_streamlines","crossing_streamlines",
                       "mean_reynolds_streamlines","mean_favre_streamlines"})
    meshes.push_back(resident_mesh(name,viskores::cont::DataSet{},step,time,
      std::string(name).find("streamlines")!=std::string::npos?2:3));
  return astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,
    "m12_diagnostics",std::move(meshes),false);
#else
  throw std::runtime_error("Device M12 plane rendering is not built; no host fallback");
#endif
} catch(const std::exception& error) {
  std::fprintf(stderr,"ASTR local M12 planes: %s\n",error.what());
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
} catch(...) {
  std::fprintf(stderr,"ASTR local M12 planes: unknown exception\n");
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
}

extern "C" int astr_insitu_device_render(const char* pipeline,const char* backend,const char* script,const char* profile,
    int fcomm,int step,Real time,int nx,int ny,int nz,int ox,int oy,int oz,int global_cells,int global_y,int global_z,
    Real* velocity,Real* halo,Real* diagnostic,Real* reynolds_halo,Real* favre_halo,
    int covered,Real duration,Real window_start,Real window_end,std::int64_t host_budget,
    int curve,Real* coordinates,Real* coordinate_halo,std::int64_t budget,std::int64_t reserve,std::int64_t retained,
    const char* seeds,Real* temperature,int plane_j,int diagnostic_components)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  const std::string seed_layout=seeds?seeds:"line16";
  if(seed_layout!="sbli-layered16") astr_insitu::tgv_streamline_seeds(seed_layout);
  Real physical_step_scale=Real(1.);
  const char* test_step_scale=std::getenv("ASTR_INSITU_TEST_CURVE_STEP_SCALE");
  if(test_step_scale) {
#ifdef ASTR_BUILD_TESTING
    if(!curve || !std::getenv("ASTR_INSITU_TEST_CURVE_TRACE_PREFIX") ||
        (std::string(test_step_scale)!="1" && std::string(test_step_scale)!="0.5"))
      throw std::invalid_argument("CURVE step sensitivity requires a diagnostic trace oracle and scale 1 or 0.5");
    physical_step_scale=std::string(test_step_scale)=="0.5"?Real(.5):Real(1.);
#else
    throw std::invalid_argument("CURVE step sensitivity is unavailable in production builds");
#endif
  }
  if(std::getenv("ASTR_INSITU_TEST_CURVE_Q_PREFIX")) {
#ifdef ASTR_BUILD_TESTING
    if(!curve) throw std::invalid_argument("CURVE Q test oracle requires physical contour coordinates");
#else
    throw std::invalid_argument("CURVE Q test oracle is unavailable in production builds");
#endif
  }
  if(std::getenv("ASTR_INSITU_TEST_CURVE_TRACE_PREFIX")) {
#ifdef ASTR_BUILD_TESTING
    if(!curve || (std::string(profile?profile:"")!="streamlines" && std::string(profile?profile:"")!="curve_demo" &&
                 std::string(profile?profile:"")!="boundary_layer"))
      throw std::invalid_argument("CURVE trace test oracle requires physical streamline coordinates");
#else
    throw std::invalid_argument("CURVE trace test oracle is unavailable in production builds");
#endif
  }
  if(!initialized) {
    int argc=1;char program[]="astr-insitu-device";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  const std::string selected=profile?profile:"";
  const bool curve_demo=selected=="curve_demo";
  const bool boundary_layer=selected=="boundary_layer";
  const bool m12_hd=selected=="m12_diagnostics";
  const bool sbli=selected=="sbli_diagnostics";
  const viskores::Id3 global_dimensions(global_cells,global_y,global_z);
  const bool preview_bl=selected=="streamlines" && astr_insitu::streamline_preview_active() &&
    !astr_insitu::streamline_views().empty() && global_dimensions==viskores::Id3(450,130,96);
  const bool demo=selected=="tgv256_demo" || curve_demo || boundary_layer || m12_hd || sbli;
  if(selected!="all" && selected!="q_surface" && selected!="q_streamlines" &&
     selected!="streamlines" && selected!="velocity_slice" && !demo) throw std::invalid_argument("Unsupported device product profile");
  if(sbli && (!curve || covered || seed_layout!="sbli-layered16" || std::string(pipeline?pipeline:"")!="standard-device" ||
      global_cells<6 || global_y<6 || global_z<5 || oy!=0 || oz!=0 || ny!=global_y || nz!=global_z))
    throw std::invalid_argument("SBLI products require physical x-slab inputs without mean fields");
  if(!sbli && ((boundary_layer || m12_hd || preview_bl)?((global_dimensions!=viskores::Id3(64,32,24) &&
      !(m12_hd && astr_insitu::m12_hd_grid(global_dimensions)) && !preview_bl) || !curve || covered ||
      (!preview_bl && seed_layout!=(m12_hd?"bl-layered256":"bl-layered64")) || std::string(pipeline?pipeline:"")!="standard-device"):
      (global_dimensions!=viskores::Id3(global_cells) ||
       (curve_demo?(global_cells!=32 && global_cells!=64 && global_cells!=128 && global_cells!=256):
       global_cells!=(demo?256:32)))))
    throw std::invalid_argument("Device product profile/resolution differs");
  if(curve_demo && global_cells==256) {
    int ranks=0;
    astr_insitu::trace_mpi(MPI_Comm_size(comm,&ranks));
    if(ranks!=2 || nx!=128 || ny!=256 || nz!=256 || (ox!=0 && ox!=128) || oy!=0 || oz!=0)
      throw std::invalid_argument("CURVE 256 demonstration requires NP=2 x-slab");
  }
  if(curve_demo && !curve) throw std::invalid_argument("CURVE demonstration requires physical coordinates");
  if(host_budget<=0 || !std::isfinite(time) || step<0) throw std::invalid_argument("Invalid device frame identity/budget");
  if(curve!=0 && curve!=1) throw std::invalid_argument("Invalid device coordinate mode");
  if(curve) {
    if((selected!="q_surface" && selected!="streamlines" && !curve_demo && !boundary_layer && !m12_hd && !sbli) || !coordinates ||
        (global_cells!=32 && !curve_demo && !boundary_layer && !m12_hd && !preview_bl && !sbli) ||
        ((selected=="q_surface" || curve_demo || boundary_layer || m12_hd || sbli) && covered) ||
        ((selected=="streamlines" || curve_demo || boundary_layer || m12_hd || sbli) && !coordinate_halo) ||
        std::string(pipeline?pipeline:"")=="compatible")
      throw std::invalid_argument("CURVE requires an admitted strict bounded surface or streamline product");
#ifdef ASTR_VISKORES_ALLOCATION_PREFLIGHT
    if(!astr_insitu_allocation_guard_active())
      throw std::runtime_error("CURVE device allocations require an active resource guard");
    viskores::cont::GetRuntimeDeviceTracker().SetThreadFriendlyMemAlloc(false);
    viskores::cont::cuda::internal::CudaAllocator::SetAllocationPreflight(
      astr_insitu_device_allocation_preflight);
    astr_insitu_device_allocation_preflight(0,"CURVE frame admission");
    if(budget<=0 || retained<0 || reserve<0 || retained>budget)
      throw std::runtime_error("CURVE retained input budget");
#else
    if(global_cells>64)
      throw std::runtime_error("CURVE scale requires the private allocation-preflight dependency patch");
    const std::uint64_t bound=std::uint64_t(nx+1)*(ny+1)*(nz+1)*8192;
    std::size_t free_bytes=0,total_bytes=0;
    const auto required=std::uint64_t(std::max<std::int64_t>(reserve,1073741824));
    if(budget<=0 || retained<0 || reserve<0 || retained>budget || bound>std::uint64_t(budget-retained) ||
        cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess || free_bytes<required || bound>free_bytes-required)
      throw std::runtime_error("CURVE Q controlled allocation budget/reserve");
#endif
  }
  const viskores::Id3 extent(nx,ny,nz),dimensions(nx+1,ny+1,nz+1),offset(ox,oy,oz),halo_dimensions(nx+7,ny+7,nz+7);
  std::array<Real,6> sbli_bounds{};
  if(sbli) {
    // Separable monotone grid: only two coordinate metadata tuples leave the GPU.
    std::array<Real,3> first{},last{},lower{},upper{};
    if(cudaMemcpy(first.data(),coordinates,3*sizeof(Real),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(last.data(),coordinates+3*(dimensions[0]*dimensions[1]*dimensions[2]-1),
         3*sizeof(Real),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Cannot obtain SBLI domain metadata");
    astr_insitu::trace_mpi(MPI_Allreduce(first.data(),lower.data(),3,astr_insitu::numerical_mpi_type(),MPI_MIN,comm));
    astr_insitu::trace_mpi(MPI_Allreduce(last.data(),upper.data(),3,astr_insitu::numerical_mpi_type(),MPI_MAX,comm));
    for(int d=0;d<3;++d) {sbli_bounds[2*d]=lower[d];sbli_bounds[2*d+1]=upper[d];}
    astr_insitu::tgv_streamline_seeds(seed_layout,&sbli_bounds);
  }
  const bool slice=(selected=="all" || selected=="velocity_slice") && astr_insitu_scene_due("velocity_slice");
  const bool surface=(selected=="all" || selected=="q_surface" || selected=="q_streamlines" || demo) &&
    astr_insitu_scene_due("q_surface");
  const bool lines=selected=="all" || selected=="streamlines" || selected=="q_streamlines" || demo;
  const bool instant=lines && astr_insitu_scene_due("instantaneous_streamlines");
  const bool crossing=lines && !demo && astr_insitu_scene_due("crossing_streamlines");
  const bool reynolds=(selected=="all" || selected=="streamlines") && covered && astr_insitu_scene_due("mean_reynolds_streamlines");
  const bool favre=(selected=="all" || selected=="streamlines") && covered && astr_insitu_scene_due("mean_favre_streamlines");
  const Real iso=(boundary_layer || m12_hd)?Real(.001):(demo?Real(0.):Real(.25));
  const std::string route=pipeline?pipeline:"";
  if(route!="compatible") {
#ifdef ASTR_INSITU_DEVICE_RENDERING
    if(route!="standard-device" && route!="direct-device")
      throw std::invalid_argument("Unsupported resident rendering pipeline; no host fallback");
    if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
      viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    int rank=0;astr_insitu::trace_mpi(MPI_Comm_rank(comm,&rank));
    std::vector<astr_insitu::DeviceMesh> meshes;
    const bool plane=(boundary_layer || m12_hd || sbli) && astr_insitu_scene_due("velocity_slice");
    const bool traces=instant || crossing || reynolds || favre;
    const int group_count=int(slice || surface)+int(plane)+int(traces);
    int emitted_groups=0;
    const auto render_group=[&]() {
      for(auto& mesh:meshes) {
        mesh.controls["group_member"]=Real(1.);
        if(sbli) for(int d=0;d<6;++d) mesh.controls["domain_bound"+std::to_string(d)]=sbli_bounds[d];
      }
      for(const char* name:{"q_surface","velocity_slice","instantaneous_streamlines","crossing_streamlines",
                           "mean_reynolds_streamlines","mean_favre_streamlines",
                           "temperature_slice","spanwise_slice","spanwise_slice_weak"}) {
        if(!m12_hd && (std::string(name)=="temperature_slice" || std::string(name).find("spanwise_slice")==0)) continue;
        if(std::none_of(meshes.begin(),meshes.end(),[&](const astr_insitu::DeviceMesh& mesh){return mesh.name==name;})) {
          auto empty=resident_mesh(name,viskores::cont::DataSet{},step,time,
            std::string(name).find("streamlines")!=std::string::npos?2:3);
          empty.controls["group_member"]=Real(0.);meshes.push_back(std::move(empty));
        }
      }
      if(selected=="streamlines" && covered) {
        if(!std::isfinite(duration) || !std::isfinite(window_start) || !std::isfinite(window_end) ||
            duration<=Real(0.) || window_end<=window_start)
          throw std::invalid_argument("Invalid mean streamline coverage metadata");
        for(auto& mesh:meshes) {
          mesh.controls["statistics_duration"]=duration;
          mesh.controls["statistics_window_start"]=window_start;
          mesh.controls["statistics_window_end"]=window_end;
        }
      }
      const bool complete=++emitted_groups>=group_count;
      const int status=astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,
        profile,std::move(meshes),covered,complete);
      meshes.clear();
      return status;
    };
    if(slice || surface) {
    ProductStage extraction("device_geometry_extract_inclusive","ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION",fcomm,step);
    for(const bool is_surface:{true,false}) {
      if(is_surface?!surface:!slice) continue;
      const auto before=meshes.size();
      const bool ok=astr_insitu::local_product(comm,is_surface?"q_surface":"velocity_slice",step,time,[&] {
        auto extracted=astr_insitu::extract_tgv_geometry(reinterpret_cast<astr_insitu::NativeVector*>(velocity),
          diagnostic,dimensions,offset,!is_surface,is_surface,iso,nullptr,global_cells,
          curve?reinterpret_cast<astr_insitu::NativeVector*>(coordinates):nullptr,global_dimensions,
          diagnostic_components,!astr_insitu::retain_quantitative_product_fields(),sbli);
        if(is_surface) meshes.push_back(resident_mesh("q_surface",extracted.surface,step,time,3,halo,
          diagnostic_components==14?reinterpret_cast<astr_insitu::DeviceDiagnostics*>(diagnostic):nullptr,extent,offset,global_cells,rank,
          curve?reinterpret_cast<astr_insitu::NativeVector*>(coordinates):nullptr,std::uint64_t(host_budget),nullptr,0,demo,iso,
          global_dimensions,(boundary_layer || m12_hd || sbli)?Real(1.1):Real(1.)));
        else meshes.push_back(resident_mesh("velocity_slice",
          astr_insitu::triangulate_device_slice(extracted.slice,dimensions,offset,global_cells),step,time,3,
          halo,nullptr,extent,offset,global_cells,rank));
      });
      if(!ok) meshes.resize(before);
    }
    if(curve) std::printf("ASTR_INSITU_CURVE_SURFACE_FRAME rank=%d step=%d physical_coordinates=1 "
      "field_download_bytes=0 geometry_host_bytes=0\n",rank,step);
    }
    if(slice || surface) {
      const int status=render_group();if(status) return status;
    }
    if((boundary_layer || sbli) && astr_insitu_scene_due("velocity_slice")) {
      const int global[3]={global_cells,global_y,global_z},local[3]={nx,ny,nz},start[3]={ox,oy,oz};
      const Real origin[3]={Real(0.),Real(0.),sbli?(sbli_bounds[4]+sbli_bounds[5])/Real(2.):Real(45.)};
      const Real normal[3]={Real(0.),Real(0.),Real(1.)};
      const auto before=meshes.size();
      if(!astr_insitu::local_product(comm,"velocity_slice",step,time,[&] {
        meshes.push_back(astr_insitu::build_device_plane(fcomm,step,time,global,local,start,coordinates,velocity,
          origin,normal,budget,reserve,retained,host_budget,true,Real(1.1)));
      })) meshes.resize(before);
    }
    if(m12_hd && astr_insitu_scene_due("velocity_slice")) {
      if(!temperature) throw std::invalid_argument("M12 independent temperature is missing");
      astr_insitu::append_m12_planes(meshes,comm,step,time,global_dimensions,extent,offset,coordinates,velocity,
        temperature,plane_j,std::uint64_t(budget),std::uint64_t(retained));
    }
    if(plane) {
      const int status=render_group();if(status) return status;
    }
    if(traces) {
      auto actual=astr_insitu::view_component_halo(halo,halo_dimensions);
      auto physical=curve?astr_insitu::view_component_halo(coordinate_halo,halo_dimensions):
        astr_insitu::SampleVectorComponents{};
      const int trace_axis=curve?(ny<global_cells?1:(nz<global_cells?2:0)):0;
      const auto add_trace=[&](const char* name,const astr_insitu::SampleVectorComponents& field,
          bool constant,bool mean,const Real* source,const std::string& trace_seeds) {
        ProductStage phase("resident_streamlines_inclusive","ASTR_IS8_RESIDENT_STREAMLINES",fcomm,step);
        auto trace=astr_insitu::trace_tgv_device(field,actual,extent,offset,comm,host_budget,
          constant,constant?trace_axis:0,constant,mean,global_cells,true,curve?&physical:nullptr,physical_step_scale,
          constant?"line16":trace_seeds,global_dimensions,!astr_insitu::retain_quantitative_product_fields(),
          sbli?&sbli_bounds:nullptr);
        report_resident_trace(trace,name,rank,constant?trace_axis:0);
#ifdef ASTR_BUILD_TESTING
        if(test_step_scale && rank==0) for(int i=0;i<trace.particles;++i) {
          const auto& s=trace.final_state[i];
          std::printf("ASTR_INSITU_CURVE_STEP_SENSITIVITY step=%d product=%s scale=%.17g particle=%d "
            "status=%d x=%.17g y=%.17g z=%.17g length=%.17g\n",step,name,physical_step_scale,i,
            s.status,s[0],s[1],s[2],s[4]);
        }
#endif
        auto mesh=resident_mesh(name,trace.resident_geometry,step,time,2,source,nullptr,
          extent,offset,global_cells,rank,nullptr,std::uint64_t(host_budget),curve?&physical:nullptr,trace_axis,demo,Real(.25),
          global_dimensions,(boundary_layer || m12_hd || sbli)?Real(1.1):Real(1.));
        if(!astr_insitu::retain_quantitative_product_fields()) trace.resident_geometry={};
        mesh.controls["streamline_seed_layout"]=trace.seed_layout=="tgv-stratified"?Real(1.):
          (trace.seed_layout=="bl-layered64"?Real(2.):(trace.seed_layout=="bl-layered256"?Real(3.):
            (trace.seed_layout=="sbli-layered16"?Real(4.):Real(0.))));
        mesh.controls["streamline_seed_count"]=trace.seed_count;
        mesh.controls["streamline_particle_count"]=trace.particles;
        if(m12_hd || !astr_insitu::streamline_views().empty()) {
          mesh.controls["streamline_step_limit"]=trace.seed_layout.rfind("bl-layered",0)==0?4096:100000;
          const auto sites=astr_insitu::tgv_streamline_seeds(trace.seed_layout);
          std::vector<Real> layers;
          for(const auto& site:sites) layers.push_back(site[1]);
          std::sort(layers.begin(),layers.end());layers.erase(std::unique(layers.begin(),layers.end()),layers.end());
          mesh.controls["termination_layer_count"]=layers.size();
          for(std::size_t i=0;i<trace.final_state.size();++i) {
            const auto& state=trace.final_state[i];
            const auto reason=std::string(state.status==astr_insitu::TraceSteps?"step_budget":
              (state.status==astr_insitu::TraceLength?"arc_length":
               (state.status==astr_insitu::Outside?"domain_exit":"sub_minimum_remaining")));
            mesh.controls["termination_"+reason]+=Real(1.);
            const auto layer=std::lower_bound(layers.begin(),layers.end(),sites[i%trace.seed_count][1])-layers.begin();
            const auto direction=i<std::size_t(trace.seed_count)?"forward":"backward";
            mesh.controls[std::string("termination_")+direction+"_"+reason]+=Real(1.);
            mesh.controls["termination_layer"+std::to_string(layer)+"_"+direction+"_"+reason]+=Real(1.);
          }
          for(const char* reason:{"step_budget","arc_length","domain_exit","sub_minimum_remaining"}) {
            mesh.controls.emplace(std::string("termination_")+reason,Real(0.));
            for(const char* direction:{"forward","backward"}) {
              mesh.controls.emplace(std::string("termination_")+direction+"_"+reason,Real(0.));
              for(std::size_t layer=0;layer<layers.size();++layer)
                mesh.controls.emplace("termination_layer"+std::to_string(layer)+"_"+direction+"_"+reason,Real(0.));
            }
          }
        }
        return mesh;
      };
      const auto& candidates=astr_insitu::streamline_views();
      if(!candidates.empty()) {
        // One state/frame scopes the cache; display-only variants reuse its owner.
        std::vector<bool> done(candidates.size(),false);
        for(std::size_t i=0;i<candidates.size();++i) if(!done[i]) {
          const auto& first=candidates[i];
          const bool mean=first.velocity!="instantaneous";
          const char* name=mean?(first.velocity=="reynolds"?"mean_reynolds_streamlines":"mean_favre_streamlines"):
            "instantaneous_streamlines";
          Real* source=mean?(first.velocity=="reynolds"?reynolds_halo:favre_halo):halo;
          if(mean && (!covered || !source)) throw std::invalid_argument("Preview mean coverage is unavailable");
          auto field=mean?astr_insitu::view_component_halo(source,halo_dimensions):actual;
          auto mesh=add_trace(name,field,false,mean,source,first.seeds);
          auto owner=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(mesh.owner);
          for(std::size_t j=i;j<candidates.size();++j) {
            const auto& candidate=candidates[j];
            if(candidate.velocity!=first.velocity || candidate.seeds!=first.seeds) continue;
            done[j]=true;
            const int component=candidate.color=="v"?1:(candidate.color=="w"?2:0);
            mesh.draw.colors=reinterpret_cast<const unsigned char*>(owner->color_display(
              astr_insitu::device_display_palette(),candidate.color=="speed",candidate.values[16],candidate.values[17],component,true));
            const auto local_range=owner->color_range();
            Real actual_range[2];
            astr_insitu::trace_mpi(MPI_Allreduce(&local_range[0],actual_range,1,astr_insitu::numerical_mpi_type(),MPI_MIN,comm));
            astr_insitu::trace_mpi(MPI_Allreduce(&local_range[1],actual_range+1,1,astr_insitu::numerical_mpi_type(),MPI_MAX,comm));
            const bool color_empty=!std::isfinite(actual_range[0]) || !std::isfinite(actual_range[1]);
            mesh.controls["actual_color_empty"]=color_empty;
            mesh.controls["actual_color_min"]=color_empty?Real(0.):actual_range[0];
            mesh.controls["actual_color_max"]=color_empty?Real(0.):actual_range[1];
            mesh.controls["trajectory_storage_id"]=reinterpret_cast<std::uintptr_t>(owner->get().coordinates);
            mesh.controls["pv_candidate"]=j;
            mesh.controls["statistics_duration"]=duration;
            mesh.controls["statistics_window_start"]=window_start;
            mesh.controls["statistics_window_end"]=window_end;
            std::vector<astr_insitu::DeviceMesh> frame{mesh};
            for(const char* other:{"instantaneous_streamlines","crossing_streamlines",
                                  "mean_reynolds_streamlines","mean_favre_streamlines"})
              if(std::string(other)!=name) {
                auto empty=resident_mesh(other,viskores::cont::DataSet{},step,time,2);
                empty.controls["pv_candidate"]=j;frame.push_back(std::move(empty));
              }
            const int status=astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,
              profile,std::move(frame),covered);
            if(status) return status;
            std::printf("ASTR_INSITU_PREVIEW_CACHE rank=%d step=%d candidate=%s velocity=%s seeds=%s integrated=%d\n",
              rank,step,candidate.id.c_str(),candidate.velocity.c_str(),candidate.seeds.c_str(),j==i);
          }
        }
        return 0;
      }
      if(instant) meshes.push_back(add_trace("instantaneous_streamlines",actual,false,false,halo,seed_layout));
      if(crossing) {
      astr_insitu::SampleVectorComponents constant;
      viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
      ConstantVelocity constant_field;constant_field.axis=trace_axis;
      invoke(constant_field,viskores::cont::ArrayHandleIndex(actual.GetNumberOfValues()),constant);
      astr_insitu::synchronize_device_stage("ConstantVelocity");
      meshes.push_back(add_trace("crossing_streamlines",constant,true,false,halo,seed_layout));
      }
      if(reynolds || favre) {
        if(!reynolds_halo || !favre_halo || !std::isfinite(duration) || duration<=Real(0.))
          throw std::invalid_argument("Resident mean products require authoritative coverage/halos");
        if(reynolds) meshes.push_back(add_trace("mean_reynolds_streamlines",astr_insitu::view_component_halo(reynolds_halo,halo_dimensions),false,true,reynolds_halo,seed_layout));
        if(favre) meshes.push_back(add_trace("mean_favre_streamlines",astr_insitu::view_component_halo(favre_halo,halo_dimensions),false,true,favre_halo,seed_layout));
      }
      if(curve) std::printf("ASTR_INSITU_CURVE_TRACE_FRAME rank=%d step=%d physical_coordinates=1 "
        "physical_hexahedral_interpolation=1 field_download_bytes=0 geometry_host_bytes=0\n",rank,step);
    }
    if(traces || group_count==0) return render_group();
    return 0;
#else
    throw std::invalid_argument("Strict device rendering is not built; no host fallback");
#endif
  }
  std::vector<astr_insitu::CompactMesh> meshes;
  if(slice || surface) {
    ProductStage extraction("device_geometry_extract_inclusive","ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION",fcomm,step);
    auto extracted=astr_insitu::extract_tgv_geometry(reinterpret_cast<astr_insitu::NativeVector*>(velocity),
      reinterpret_cast<astr_insitu::DeviceDiagnostics*>(diagnostic),dimensions,offset,slice,surface,
      demo?Real(0.):Real(.25),nullptr,global_cells);
    extraction.finish();
    ProductStage read("device_geometry_read","ASTR_IS8_COMPACT_GEOMETRY_READ",fcomm,step);
    if(surface) meshes.push_back(geometry_mesh(extracted.surface,true,dimensions,offset));
    if(slice) meshes.push_back(geometry_mesh(extracted.slice,false,dimensions,offset));
  }
  if(instant || crossing || reynolds || favre) {
    auto actual=astr_insitu::view_component_halo(halo,halo_dimensions);
    if(instant) {
    ProductStage instant("device_instant_lines_inclusive","ASTR_IS8_DEVICE_INSTANT_LINES",fcomm,step);
    auto trace=astr_insitu::trace_tgv_device(actual,actual,extent,offset,comm,host_budget,
      false,0,false,false,global_cells);
    meshes.push_back(streamline_mesh(trace,"instantaneous_streamlines"));
    instant.finish();
    }
    if(crossing) {
    ProductStage crossing("device_crossing_lines_inclusive","ASTR_IS8_DEVICE_CROSSING_LINES",fcomm,step);
    astr_insitu::SampleVectorComponents constant;
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    invoke(ConstantVelocity{},viskores::cont::ArrayHandleIndex(actual.GetNumberOfValues()),constant);
    astr_insitu::synchronize_device_stage("ConstantVelocity");
    auto trace=astr_insitu::trace_tgv_device(constant,actual,extent,offset,comm,host_budget,true,0,true);
    meshes.push_back(streamline_mesh(trace,"crossing_streamlines"));
    crossing.finish();
    }
    if(reynolds || favre) {
      if(!reynolds_halo || !favre_halo || !std::isfinite(duration) || duration<=Real(0.))
        throw std::invalid_argument("Covered device means require authoritative halos/duration");
      if(reynolds) {
      ProductStage reynolds_stage("device_reynolds_lines_inclusive","ASTR_IS8_DEVICE_REYNOLDS_LINES",fcomm,step);
      auto field=astr_insitu::view_component_halo(reynolds_halo,halo_dimensions);
      auto trace=astr_insitu::trace_tgv_device(field,actual,extent,offset,comm,host_budget,false,0,false,true);
      meshes.push_back(streamline_mesh(trace,"mean_reynolds_streamlines"));
      reynolds_stage.finish();
      }
      if(favre) {
      ProductStage favre_stage("device_favre_lines_inclusive","ASTR_IS8_DEVICE_FAVRE_LINES",fcomm,step);
      auto field=astr_insitu::view_component_halo(favre_halo,halo_dimensions);
      auto trace=astr_insitu::trace_tgv_device(field,actual,extent,offset,comm,host_budget,false,0,false,true);
      meshes.push_back(streamline_mesh(trace,"mean_favre_streamlines"));
      }
    }
  }
  // Keep channel identity stable, including valid empty/uncovered products.
  for(const char* name:{"q_surface","velocity_slice","instantaneous_streamlines","crossing_streamlines",
                       "mean_reynolds_streamlines","mean_favre_streamlines"}) {
    if(std::none_of(meshes.begin(),meshes.end(),[&](const astr_insitu::CompactMesh& m){return m.name==name;})) {
      astr_insitu::CompactMesh empty;empty.name=name;
      empty.shape=empty.name=="q_surface"?"tri":(empty.name=="velocity_slice"?"quad":"line");
      meshes.push_back(std::move(empty));
    }
  }
  std::uint64_t bytes=0;
  for(const auto& mesh:meshes) {
    bytes+=mesh.coordinates[0].size()*3*sizeof(Real)+
      (mesh.connectivity.size()+mesh.seed_ids.size()+mesh.directions.size())*sizeof(std::int64_t);
    for(const auto& field:mesh.point_fields) bytes+=field.second.size()*sizeof(Real);
  }
  if(bytes>std::uint64_t(host_budget)/2) throw std::runtime_error("Retained and next compact geometry exceed host budget");
  return astr_insitu::render_compact_products(backend,script,fcomm,step,time,profile,covered,
    duration,window_start,window_end,std::move(meshes));
} catch(const std::exception& error) {
  std::fprintf(stderr,"ASTR device products failed: %s\n",error.what());
  MPI_Abort(MPI_Comm_f2c(fcomm),1);return 1;
}

extern "C" void astr_insitu_device_products_release() {
  slice_cache=SliceCache{};initialized=false;
}

#ifdef ASTR_INSITU_DEVICE_RENDERING
namespace {
struct RuntimeTrajectoryKey {
  std::string source,seeds;
  std::array<double,5> integration;
  int steps,direction;
  bool operator<(const RuntimeTrajectoryKey& other) const {
    return std::tie(source,seeds,integration,steps,direction)<
      std::tie(other.source,other.seeds,other.integration,other.steps,other.direction);
  }
};
struct RuntimeLineQuantity {
  int component=0;
  bool magnitude=false;
  Real factor=Real(1.);
  std::vector<int> fields;
  bool operator<(const RuntimeLineQuantity& other) const {
    return std::tie(component,magnitude,factor,fields)<std::tie(other.component,other.magnitude,other.factor,other.fields);
  }
};
struct RuntimeLineRegion {
  viskores::cont::DataSet data;
  std::map<RuntimeLineQuantity,std::shared_ptr<astr_insitu::DeviceGeometryOwner>> owners;
};
struct RuntimeTrajectory {
  astr_insitu::DeviceStreamlines trace;
  std::map<std::array<double,6>,RuntimeLineRegion> regions;
};
struct RuntimeLines {
  bool active=false,frozen=false;
  Real* halo=nullptr;
  Real* coordinates=nullptr;
  Real* coordinate_halo=nullptr;
  std::array<int,9> partition{};
  int physical=0;
  std::array<Real,6> bounds{},local_bounds{};
  std::map<RuntimeTrajectoryKey,RuntimeTrajectory> trajectories;
  std::set<RuntimeTrajectoryKey> metadata;
  int integrations=0,clipped=0,scalar_builds=0,owner_reuse=0;
} runtime_lines;

struct RuntimeLineScalar : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  int component;bool magnitude;Real factor;
  VISKORES_CONT explicit RuntimeLineScalar(const RuntimeLineQuantity& q):component(q.component),magnitude(q.magnitude),factor(q.factor) {}
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& v,Real& result) const {
    result=(magnitude?viskores::Sqrt(viskores::Dot(v,v)):v[component])*factor;
  }
};

struct RuntimeFinalTraceMetadata : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  template<class Final>
  VISKORES_EXEC void operator()(viskores::Id particle,const Final& final,Real& status,Real& remaining) const {
    const auto value=final.Get(particle);status=value[0];remaining=value[1];
  }
};
struct RuntimeTraceCellIdentity : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5);
  int seeds,direction;
  VISKORES_CONT RuntimeTraceCellIdentity(int count,int mode):seeds(count),direction(mode) {}
  template<class Indices,class Particles>
  VISKORES_EXEC void operator()(viskores::Id cell,const Indices& indices,const Particles& particles,
      viskores::Id& seed,viskores::Id& way) const {
    const auto particle=particles.Get(indices.Get(2*cell));seed=particle%seeds;
    way=direction?direction:(particle<seeds?1:-1);
  }
};

RuntimeLineQuantity line_quantity(const astr_insitu::RuntimeProductView& view,const Real* references) {
  RuntimeLineQuantity result;
  for(const auto& field:view.fields) result.fields.push_back(field.first);
  const auto& field=view.labels[1]=="geometry"?view.fields.front().second:view.labels[4];
  if(field=="velocity_magnitude") result.magnitude=true;
  else if(field=="velocity_y") result.component=1;
  else if(field=="velocity_z" || field=="w_Uinf") result.component=2;
  else if(field!="velocity_x" && field!="u_Uinf")
    throw std::invalid_argument("Existing streamlines support velocity components or magnitude only");
  if(field=="u_Uinf" || field=="w_Uinf") {
    if(!std::isfinite(references[0]) || references[0]==Real(0.))
      throw std::invalid_argument("Normalized streamline color needs a finite nonzero velocity reference");
    result.factor=Real(1.)/references[0];
    if(!std::isfinite(result.factor)) throw std::invalid_argument("Streamline color reference reciprocal is not representable");
  }
  return result;
}

RuntimeTrajectoryKey trajectory_key(const std::string& source,const std::string& seeds,const std::string& direction,
    const int* words,const double* settings) {
  RuntimeTrajectoryKey key;
  key.source=source;key.seeds=seeds;std::copy_n(settings+26,5,key.integration.begin());key.steps=words[6];
  if(direction!="forward" && direction!="backward" && direction!="both")
    throw std::invalid_argument("Unknown explicit streamline direction");
  key.direction=direction=="forward"?1:(direction=="backward"?-1:0);
  return key;
}

void runtime_line_oracle(const astr_insitu::RuntimeProductView& context,const astr_insitu::DeviceGeometryView& view,
    int fcomm,int step,std::int64_t host_budget) {
  const char* prefix=std::getenv("ASTR_INSITU_TEST_RUNTIME_LINE_PREFIX");
  if(!prefix || !*prefix) return;
#ifdef ASTR_BUILD_TESTING
  const std::uint64_t bytes=std::uint64_t(view.points)*4*sizeof(Real)+std::uint64_t(view.cells)*2*sizeof(viskores::Id);
  if(bytes>std::uint64_t(host_budget) || bytes>64*1024*1024)
    throw std::runtime_error("Runtime streamline validation oracle exceeds its bounded host allocation");
  std::vector<astr_insitu::NativeVector> xyz(view.points);
  std::vector<Real> values(view.points);
  std::vector<viskores::Id> indices(2*view.cells);
  if(view.points) {
    if(cudaMemcpy(xyz.data(),view.coordinates,xyz.size()*sizeof(xyz[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(values.data(),view.scalar,values.size()*sizeof(Real),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(indices.data(),view.connectivity,indices.size()*sizeof(viskores::Id),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Cannot download the explicitly requested streamline validation oracle");
  }
  int rank=0;astr_insitu::trace_mpi(MPI_Comm_rank(MPI_Comm_f2c(fcomm),&rank));
  const auto path=std::string(prefix)+"."+context.id+".step"+std::to_string(step)+".rank"+std::to_string(rank)+".bin";
  std::ofstream file(path,std::ios::binary|std::ios::out);
  const std::uint64_t header[]={sizeof(Real),std::uint64_t(view.points),std::uint64_t(view.cells)};
  file.write("ASTRLI01",8);file.write(reinterpret_cast<const char*>(header),sizeof(header));
  if(!xyz.empty()) file.write(reinterpret_cast<const char*>(xyz.data()),xyz.size()*sizeof(xyz[0]));
  if(!values.empty()) file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(Real));
  if(!indices.empty()) file.write(reinterpret_cast<const char*>(indices.data()),indices.size()*sizeof(viskores::Id));
  file.close();if(!file) throw std::runtime_error("Cannot publish the streamline validation oracle");
  std::printf("ASTR_RUNTIME_LINE_ORACLE rank=%d id=%s step=%d oracle_download_bytes=%llu runtime_zero_readback_evidence=0\n",
    rank,context.id.c_str(),step,static_cast<unsigned long long>(bytes));
#else
  throw std::invalid_argument("Runtime streamline validation oracle is unavailable in this production build");
#endif
}

void prepare_runtime_line_bounds(const int* global,const int* extent,const int* offset,int physical,
    Real* halo,Real* coordinate_halo,Real* coordinates) {
  std::array<int,9> partition{};
  std::copy_n(global,3,partition.begin());std::copy_n(extent,3,partition.begin()+3);std::copy_n(offset,3,partition.begin()+6);
  if(runtime_lines.frozen) {
    if(runtime_lines.halo!=halo || runtime_lines.coordinates!=coordinates || runtime_lines.coordinate_halo!=coordinate_halo ||
        runtime_lines.partition!=partition || runtime_lines.physical!=physical)
      throw std::logic_error("A streamline group received another frozen velocity/geometry");
    return;
  }
  const viskores::Id nodes=viskores::Id(extent[0]+1)*(extent[1]+1)*(extent[2]+1);
  auto xyz=astr_insitu::borrow_cuda_array(reinterpret_cast<astr_insitu::NativeVector*>(coordinates),nodes);
  const auto metadata=viskores::cont::make_ArrayHandleTransform(viskores::cont::make_ArrayHandleZip(xyz,
    viskores::cont::ArrayHandleConstant<Real>(Real(0.),nodes)),astr_insitu::ScalarDisplayMetadata{});
  astr_insitu::DisplayMetadata initial;
  for(int d=0;d<4;++d) {initial[2*d]=INFINITY;initial[2*d+1]=-INFINITY;}initial[8]=Real(0.);
  const auto bounds=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},metadata,initial,
    astr_insitu::MergeDisplayMetadata{});
  astr_insitu::synchronize_device_stage("RuntimeStreamlineDomainMetadata");astr_insitu::require_device_only(xyz);
  if(bounds[8]!=Real(0.)) throw std::runtime_error("Nonfinite streamline physical geometry");
  for(int d=0;d<6;++d) runtime_lines.local_bounds[d]=bounds[d];
  runtime_lines.halo=halo;runtime_lines.coordinates=coordinates;runtime_lines.coordinate_halo=coordinate_halo;
  runtime_lines.partition=partition;runtime_lines.physical=physical;
}
}

extern "C" void astr_insitu_runtime_line_group(int begin) {
  astr_insitu::clear_runtime_compatible_geometry();
  if(begin) {
    if(runtime_lines.active) throw std::logic_error("Runtime streamline group already open");
    runtime_lines=RuntimeLines{};runtime_lines.active=true;
  } else {
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    std::printf("ASTR_RUNTIME_LINE_GROUP rank=%d integrations=%d clipped_geometries=%d scalar_builds=%d owner_reuse=%d retained_after=0\n",
      rank,runtime_lines.integrations,runtime_lines.clipped,runtime_lines.scalar_builds,runtime_lines.owner_reuse);
    runtime_lines=RuntimeLines{};
  }
}

extern "C" int astr_insitu_runtime_line_metadata(const char* source,const char* labels,const int* words,
    const double* settings,int fcomm) {
  std::string error;
  try {
    auto text=[&](int index) {
      std::string result(labels+32*index,32);
      while(!result.empty() && result.back()==' ') result.pop_back();
      return result;
    };
    runtime_lines.metadata.insert(trajectory_key(source,text(7),text(9),words,settings));
  } catch(const std::exception& caught) {error=caught.what();}
  if(astr_insitu::consensus_product_failure(MPI_Comm_f2c(fcomm),error)) {
    astr_insitu_runtime_failure(error.c_str());return 1;
  }
  return 0;
}

extern "C" int astr_insitu_render_line_product(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,Real time,const char* id,const char* source,const char* definition,const char* directory,
    std::int64_t version,const char* labels,const int* words,const double* settings,const int* global,const int* local,
    const int* start,int physical,Real* halo,Real* coordinate_halo,Real* coordinates,const Real* references,
    int provider,const Real* statistics,std::int64_t samples,std::int64_t host_budget) {
  const auto comm=MPI_Comm_f2c(fcomm);
  const auto started=astr_insitu::runtime_product_clock();
  astr_insitu::RuntimeProductView context;
  astr_insitu::DeviceMesh mesh;
  std::string error;
  RuntimeTrajectoryKey key{};
  RuntimeLineQuantity quantity;
  astr_insitu::TraceConfiguration parameters{};
  try {
    astr_insitu::LocalAllocationRecovery local_allocations;
    if(!runtime_lines.active) throw std::logic_error("Runtime streamline producer needs a frozen group");
    context=astr_insitu::runtime_product_view(id,source,definition,directory,version,labels,words,settings);
    context.host_budget=host_budget;
    context.statistics_provider=provider;context.samples=samples;std::copy_n(statistics,3,context.statistics.begin());
    quantity=line_quantity(context,references);
    key=trajectory_key(context.source,context.labels[7],context.labels[9],context.integers.data(),context.reals.data());
    parameters={Real(key.integration[0]),Real(key.integration[1]),Real(key.integration[2]),Real(key.integration[3]),
      Real(key.integration[4]),key.steps,key.direction};parameters.validate();
    if(!initialized) {int argc=1;char program[]="astr-runtime-streamlines";char* args[]={program,nullptr};
      char** argv=args;viskores::cont::Initialize(argc,argv);initialized=true;}
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
    if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
      viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    astr_insitu::prepare_runtime_product_allocations(context,fcomm,step);
    prepare_runtime_line_bounds(global,local,start,physical,halo,coordinate_halo,coordinates);
  } catch(const std::exception& caught) {error=caught.what();}
  if(astr_insitu::consensus_product_failure(comm,error)) {
    astr_insitu_runtime_failure(error.c_str());return 1;
  }
  if(!runtime_lines.frozen) {
    Real low[3],high[3],lower[3],upper[3];
    for(int d=0;d<3;++d) {low[d]=runtime_lines.local_bounds[2*d];high[d]=runtime_lines.local_bounds[2*d+1];}
    astr_insitu::trace_mpi(MPI_Allreduce(low,lower,3,astr_insitu::numerical_mpi_type(),MPI_MIN,comm));
    astr_insitu::trace_mpi(MPI_Allreduce(high,upper,3,astr_insitu::numerical_mpi_type(),MPI_MAX,comm));
    for(int d=0;d<3;++d) {runtime_lines.bounds[2*d]=lower[d];runtime_lines.bounds[2*d+1]=upper[d];}
    runtime_lines.frozen=true;
  }
  try {
    const auto seeds=astr_insitu::tgv_streamline_seeds(key.seeds,key.seeds=="sbli-layered16"?&runtime_lines.bounds:nullptr);
    for(const auto& site:seeds) for(int d=0;d<3;++d)
      if(site[d]<runtime_lines.bounds[2*d] || site[d]>runtime_lines.bounds[2*d+1])
        throw std::invalid_argument("The selected existing seed layout lies outside this physical domain");
  } catch(const std::exception& caught) {error=caught.what();}
  if(astr_insitu::consensus_product_failure(comm,error)) {
    astr_insitu_runtime_failure(error.c_str());return 1;
  }
  const viskores::Id3 extent(local[0],local[1],local[2]),offset(start[0],start[1],start[2]),all(global[0],global[1],global[2]);
  auto found=runtime_lines.trajectories.find(key);
  const bool integrated=found==runtime_lines.trajectories.end();
  if(integrated) {
    try {
      const RuntimeTraceStages stages{comm,id,step};
      const auto dimensions=extent+viskores::Id3(7);
      const auto views=stages("views",[&] {
        return std::make_pair(astr_insitu::view_component_halo(halo,dimensions),
          physical?astr_insitu::view_component_halo(coordinate_halo,dimensions):astr_insitu::SampleVectorComponents{});
      });
      const auto& velocity=views.first;
      const auto& xyz=views.second;
      RuntimeTrajectory result;
      result.trace=astr_insitu::trace_tgv_device(velocity,velocity,extent,offset,comm,std::uint64_t(host_budget),
        false,0,false,false,global[0],true,physical?&xyz:nullptr,Real(1.),key.seeds,all,true,
        key.seeds=="sbli-layered16"?&runtime_lines.bounds:nullptr,&parameters,runtime_lines.metadata.count(key)!=0,stages);
      stages("cache",[&] {found=runtime_lines.trajectories.emplace(key,std::move(result)).first;});
      ++runtime_lines.integrations;
    } catch(const astr_insitu::CollectiveTraceError& caught) {
      error=caught.what();
    } catch(const std::exception& caught) {
      // MPI failures or failures outside the matched local phases remain fatal.
      std::fprintf(stderr,"Runtime distributed streamline stage failed: %s\n",caught.what());MPI_Abort(comm,1);return 1;
    }
  }
  if(!error.empty()) return astr_insitu::publish_runtime_product(
    pipeline,backend,script,fcomm,step,time,context,std::move(mesh),error,started);
  try {
    astr_insitu::LocalAllocationRecovery local_allocations;
    auto& trajectory=found->second;
    std::array<double,6> bounds{};std::copy_n(context.reals.begin()+6,6,bounds.begin());
    auto region=trajectory.regions.find(bounds);
    if(region==trajectory.regions.end()) {
      RuntimeLineRegion value;
      value.data=astr_insitu::clip_runtime_device_mesh(trajectory.trace.resident_geometry,context,2);
      region=trajectory.regions.emplace(bounds,std::move(value)).first;++runtime_lines.clipped;
    }
    auto& product=region->second;
    const bool geometry=context.labels[1]=="geometry";
    std::vector<std::string> point_names,cell_names;
    for(const auto& field:context.fields) point_names.push_back(field.second);
    if(geometry) {
      for(const char* name:{"accepted_length","termination_status","untravelled_arc_length"}) point_names.emplace_back(name);
      cell_names={"SeedIds","IntegrationDirection"};
    }
    auto owner=product.owners.find(quantity);
    if(owner==product.owners.end()) {
      viskores::cont::DataSet data;
      if(product.data.GetNumberOfCoordinateSystems()) {
        data.SetCellSet(product.data.GetCellSet());data.AddCoordinateSystem(product.data.GetCoordinateSystem());
        const auto velocity=product.data.GetPointField("velocity").GetData().AsArrayHandle<
          viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
        viskores::cont::ArrayHandle<Real> scalar;
        viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
        invoke(RuntimeLineScalar(quantity),velocity,scalar);astr_insitu::synchronize_device_stage("RuntimeStreamlineColorScalar");
        astr_insitu::require_device_only(velocity);astr_insitu::require_device_only(scalar);data.AddPointField("quantity",scalar);
        if(geometry) {
          for(const auto& field:context.fields) {
            RuntimeLineQuantity component;component.component=field.first-2;
            viskores::cont::ArrayHandle<Real> values;
            if(component.component==quantity.component) values=scalar;
            else {invoke(RuntimeLineScalar(component),velocity,values);astr_insitu::synchronize_device_stage("RuntimeTraceExportComponent");}
            astr_insitu::require_device_only(values);data.AddPointField(field.second,values);
          }
          const auto particles=product.data.GetPointField("particle").GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Id>>();
          const auto arc=product.data.GetPointField("accepted_length").GetData().AsArrayHandle<viskores::cont::ArrayHandle<Real>>();
          std::vector<viskores::Vec<Real,2>> final(trajectory.trace.final_state.size());
          for(std::size_t i=0;i<final.size();++i) final[i]={Real(trajectory.trace.final_state[i].status),
            std::max(Real(0.),parameters.maximum_length-trajectory.trace.final_state[i][4])};
          const auto state=viskores::cont::make_ArrayHandle(final,viskores::CopyFlag::On);
          viskores::cont::ArrayHandle<Real> status,remaining;
          invoke(RuntimeFinalTraceMetadata{},particles,state,status,remaining);
          astr_insitu::synchronize_device_stage("RuntimeTraceExportTermination");
          data.AddPointField("accepted_length",arc);data.AddPointField("termination_status",status);
          data.AddPointField("untravelled_arc_length",remaining);
          const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
          const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{});
          viskores::cont::ArrayHandle<viskores::Id> seeds,directions;
          invoke(RuntimeTraceCellIdentity(trajectory.trace.seed_count,key.direction),
            viskores::cont::ArrayHandleIndex(cells.GetNumberOfCells()),indices,particles,seeds,directions);
          astr_insitu::synchronize_device_stage("RuntimeTraceExportIdentity");
          data.AddCellField("SeedIds",seeds);data.AddCellField("IntegrationDirection",directions);
          astr_insitu::require_device_only(particles);astr_insitu::require_device_only(arc);
          astr_insitu::require_device_only(status);astr_insitu::require_device_only(remaining);
          astr_insitu::require_device_only(seeds);astr_insitu::require_device_only(directions);
        }
      }
      auto value=std::make_shared<astr_insitu::DeviceGeometryOwner>(data,step,time,2,"quantity",false,point_names,cell_names,geometry);
      owner=product.owners.emplace(quantity,std::move(value)).first;++runtime_lines.scalar_builds;
    } else ++runtime_lines.owner_reuse;
    const auto colors=geometry?nullptr:owner->second->color_display(astr_insitu::device_display_palette(context.labels[6]),false,
      Real(context.reals[22]),Real(context.reals[23]));
    const auto& view=owner->second->get();
    runtime_line_oracle(context,view,fcomm,step,host_budget);
    mesh.name="runtime_product";mesh.actor_key=context.id;mesh.owner=owner->second;
    mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
    mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
    mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.arity=2;mesh.draw.device=view.device;
    std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
    mesh.controls["actual_min"]=view.speed_range[0];mesh.controls["actual_max"]=view.speed_range[1];
    mesh.controls["trajectory_integration_performed"]=integrated;
    mesh.controls["trajectory_seed_count"]=trajectory.trace.seed_count;mesh.controls["trajectory_particles"]=trajectory.trace.particles;
    mesh.controls["trajectory_control_read_bytes"]=trajectory.trace.control_read_bytes;
    mesh.controls["trajectory_owner_query_read_bytes"]=trajectory.trace.owner_query_read_bytes;
    for(const char* name:{"length","step_budget","domain_exit","subminimum"}) mesh.controls[std::string("termination_")+name]=0.;
    for(std::size_t i=0;i<trajectory.trace.final_state.size();++i) {
      const auto status=trajectory.trace.final_state[i].status;
      const char* reason=status==astr_insitu::TraceLength?"length":(status==astr_insitu::TraceSteps?"step_budget":
        (status==astr_insitu::Outside?"domain_exit":"subminimum"));
      mesh.controls[std::string("termination_")+reason]+=1.;
    }
  } catch(const std::exception& caught) {error=caught.what();}
  return astr_insitu::publish_runtime_product(pipeline,backend,script,fcomm,step,time,context,std::move(mesh),error,started);
}
#endif
