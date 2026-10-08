#include "insitu_device_geometry.h"
#include "insitu_device_streamlines.h"
#include "insitu_compact_mesh.h"
#include "insitu_device_product_audit.h"
#include "insitu_device_wall.h"
#include "insitu_product_dispatch.h"
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>

extern "C" void astr_insitu_device_allocation_preflight(std::size_t,const char*);
extern "C" int astr_insitu_allocation_guard_active();

namespace {
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

void append_coordinate(astr_insitu::CompactMesh& mesh,const viskores::Vec3f& xyz) {
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
    auto coords=data.GetCoordinateSystem().GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f>>();
    astr_insitu::require_device_only(coords);
    const auto points=coords.ReadPortal();
    for(viskores::Id i=0;i<nodes;++i) append_coordinate(mesh,points.Get(i));
    const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
    const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{}).ReadPortal();
    for(viskores::Id i=0;i<cells.GetNumberOfCells()*3;++i) mesh.connectivity.push_back(indices.Get(i));
    auto q=data.GetPointField("Q_rs").GetData().AsArrayHandle<viskores::cont::ArrayHandle<double>>().ReadPortal();
    for(viskores::Id i=0;i<nodes;++i) mesh.point_fields["Q_rs"].push_back(q.Get(i));
  } else {
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess) throw std::runtime_error("Cannot identify static slice device");
    if(!slice_cache.valid || slice_cache.dimensions!=dimensions || slice_cache.offset!=offset || slice_cache.device!=device) {
      slice_cache=SliceCache{};slice_cache.dimensions=dimensions;slice_cache.offset=offset;slice_cache.device=device;
      const double h=2.*std::acos(-1.)/32.;
      for(viskores::Id j=0;j<dimensions[1];++j) for(viskores::Id i=0;i<dimensions[0];++i)
        append_coordinate(slice_cache.geometry,viskores::Vec3f(offset[0]*h+i*h,offset[1]*h+j*h,4*h));
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
  auto vectors=data.GetPointField("velocity").GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f>>();
  auto scalar=data.GetPointField("u").GetData().AsArrayHandle<viskores::cont::ArrayHandle<double>>();
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
  const double pi=std::acos(-1.);
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
      append_coordinate(mesh,viskores::Vec3f(p[0],p[1],p[2]));
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
      mesh.point_fields["termination_status"].push_back(final[7]);
      mesh.point_fields["untravelled_arc_length"].push_back(std::max(0.,pi-final[4]));
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
  VISKORES_EXEC void operator()(viskores::Id,viskores::Vec3f& v) const {v=viskores::Vec3f(0.);v[axis]=1.;}
};
#ifdef ASTR_INSITU_DEVICE_RENDERING
struct PhysicalTraceAudit : viskores::worklet::WorkletMapField {
  astr_insitu::DeviceGeometryView view;
  bool crossing;
  int axis;
  using ControlSignature=void(FieldIn,FieldIn,FieldIn,ExecObject,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5);
  template<class Grid>
  VISKORES_EXEC void operator()(const viskores::Vec3f& point,const viskores::Vec3f& velocity,
      viskores::Id index,const Grid& grid,viskores::Vec<double,2>& differences) const {
    viskores::VecVariable<viskores::Vec3f,2> expected;
    const auto status=grid.Evaluate(point,0.,expected);
    double error=0.,display=0.;
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
      display=fmax(display,fabs(double(view.display_coordinates[index][d])-double(float(point[d]))));
    }
    const double speed=sqrt(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]);
    display=fmax(display,fabs(double(view.display_speed[index])-double(float(speed))));
    if(crossing) {
      const double pi=acos(-1.),seed[3]={pi/2.,pi/8.,pi/4.};
      const int along=(axis+1)%3,fixed=(axis+2)%3;
      const double number=(point[along]-seed[1])/(3.*pi/4./15.);
      error=fmax(error,fabs(point[along]-(seed[1]+round(number)*(3.*pi/4./15.))));
      error=fmax(error,fabs(point[fixed]-seed[2]));
      error=fmax(error,fmax(0.,seed[0]-point[axis]));
      error=fmax(error,fmax(0.,point[axis]-3.*pi/2.));
    }
    differences=viskores::Vec<double,2>(error,display);
  }
};
struct TraceAuditMaximum {
  VISKORES_EXEC_CONT viskores::Vec<double,2> operator()(const viskores::Vec<double,2>& a,
      const viskores::Vec<double,2>& b) const {
    return {viskores::Max(a[0],b[0]),viskores::Max(a[1],b[1])};
  }
};

astr_insitu::ProductAudit audit_physical_trace(const astr_insitu::DeviceGeometryView& view,
    const viskores::cont::ArrayHandle<viskores::Vec3f>& coordinates,const double* source,
    const viskores::Id3& extent,const viskores::Id3& offset,const viskores::Id3& global_cells,bool crossing,int axis) {
  astr_insitu::ProductAudit result;
  if(!view.points) return result;
  const auto fields=astr_insitu::pack_component_halo(const_cast<double*>(source),extent+viskores::Id3(7));
  const auto physical=astr_insitu::crop_physical_trace_halo(coordinates,fields,fields,extent,offset,global_cells);
  astr_insitu::PhysicalTraceOwner owner(physical,extent,offset,global_cells);
  astr_insitu::PhysicalTraceGrid grid(owner,astr_insitu::PhysicalTraceField(physical.trace));
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  PhysicalTraceAudit worklet;worklet.view=view;worklet.crossing=crossing;worklet.axis=axis;
  viskores::cont::ArrayHandle<viskores::Vec<double,2>> differences;
  invoke(worklet,astr_insitu::borrow_cuda_array(const_cast<viskores::Vec3f*>(view.coordinates),view.points),
    astr_insitu::borrow_cuda_array(const_cast<viskores::Vec3f*>(view.velocity),view.points),
    viskores::cont::ArrayHandleIndex(view.points),grid,differences);
  astr_insitu::synchronize_device_stage("ReadOnlyPhysicalTraceAudit");
  astr_insitu::require_device_only(differences);
  const auto maximum=viskores::cont::Algorithm::Reduce(differences,viskores::Vec<double,2>(0.),TraceAuditMaximum{});
  result.field=maximum[0];result.display=maximum[1];
  if(!std::isfinite(result.field) || result.field>2e-10 || result.display!=0.)
    throw std::runtime_error("Physical trace numerical/display acceptance failed");
  return result;
}

astr_insitu::DeviceMesh resident_mesh(const char* name,const viskores::cont::DataSet& data,
    int step,double time,int arity=3,const double* audit_halo=nullptr,
    const astr_insitu::DeviceDiagnostics* audit_diagnostics=nullptr,
    viskores::Id3 extent={0,0,0},viskores::Id3 offset={0,0,0},int global_cells=32,int rank=0,
    const viskores::Vec3f* physical_source=nullptr,std::uint64_t host_budget=0,
    const viskores::cont::ArrayHandle<viskores::Vec3f>* trace_coordinates=nullptr,int trace_axis=0,
    bool speed_colors=false,double iso=.25,viskores::Id3 global_dimensions=viskores::Id3(0),double speed_max=1.) {
  if(global_dimensions==viskores::Id3(0)) global_dimensions=viskores::Id3(global_cells);
  auto owner=std::make_shared<astr_insitu::DeviceGeometryOwner>(data,step,time,arity);
  static const auto palette=astr_insitu::device_display_palette();
  const bool speed=speed_colors || global_cells==256;
  const auto colors=owner->color_display(palette,speed,speed?0.:-1.,speed?speed_max:1.);
  const auto& view=owner->get();
#ifdef ASTR_BUILD_TESTING
  if(physical_source) astr_insitu::test_curve_surface_oracle(data,view,rank,step,host_budget);
  if(trace_coordinates) astr_insitu::test_curve_trace_oracle(data,view,name,rank,step,host_budget);
#endif
  if(audit_halo && std::getenv("ASTR_INSITU_RESIDENT_AUDIT") &&
      std::string(std::getenv("ASTR_INSITU_RESIDENT_AUDIT"))=="1") {
    viskores::cont::Token token;
    const viskores::Vec3f* computational=nullptr;
    if(physical_source && view.points) {
      const auto field=data.GetPointField("_astr_computational_xyz").GetData().AsArrayHandle<
        viskores::cont::ArrayHandle<viskores::Vec3f>>();
      astr_insitu::require_device_only(field);
      computational=static_cast<const viskores::Vec3f*>(field.GetBuffers()[0].ReadPointerDevice(
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
    double worst=0.;
    const double pi=std::acos(-1.);
    const auto seeds=astr_insitu::tgv_streamline_seeds(trace.seed_layout);
    for(int i=0;i<trace.seed_count;++i) {
      const auto& point=trace.final_state[i];
      const double original[3]={seeds[i][0]+pi,seeds[i][1],seeds[i][2]};
      for(int d=0;d<3;++d) {
        const double error=std::abs(point[d]-original[(d-axis+3)%3]);
        if(!std::isfinite(error)) throw std::runtime_error("Nonfinite resident crossing endpoint");
        worst=std::max(worst,error);
      }
    }
    if(worst>2e-10) throw std::runtime_error("Resident crossing endpoint acceptance failed");
    std::printf("ASTR_INSITU_RESIDENT_CROSSING rank=%d endpoint_maxabs=%.17g\n",rank,worst);
  }
  for(int i=0;i<trace.particles;++i) if(trace.subminimum_stop[i])
    std::printf("ASTR_INSITU_TRAJECTORY_TERMINATION rank=%d product=%s seed=%d direction=%d status=3 "
      "reason=sub_minimum_remaining accepted=%.17g remaining=%.17g minimum=%.17g\n",rank,name,
      i%trace.seed_count,i<trace.seed_count?1:-1,trace.final_state[i][4],trace.maximum_length-trace.final_state[i][4],trace.minimum_step);
}
#endif
}

extern "C" int astr_insitu_device_render_walls(const char* pipeline,const char* backend,const char* script,
    int fcomm,int step,double time,int im,int km,int nw,int wall_kind,double* coordinates,double* fields,
    int mean_requested,int covered,double duration,double window_start,double window_end,double* means,
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
      covered!=int(mean_requested && duration>0.) || !std::isfinite(duration) || duration<0. ||
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
  const double channel_ranges[3][2]={{0.,12.},{-.02,.02},{-.2,.2}};
  const double curve_ranges[3][2]={{80.,120.},{-.2,.2},{0.,15000.}};
  const double air5_ranges[10][2]={{0.,60000.},{1500.,3500.},{1500.,3500.},
    {0.,1.},{0.,1.},{0.,1.},{0.,1.},{0.,1.},{-1.,1.},{-60000.,60000.}};
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
    double total_area=0.;
    astr_insitu::trace_mpi(MPI_Allreduce(&product.audit[0],&total_area,1,MPI_DOUBLE,MPI_SUM,comm));
    if(!placeholder && (!std::isfinite(total_area) || total_area<=0. ||
        (!curve && std::abs(total_area-(air5?.08*.002:4.*std::acos(-1.)*std::acos(-1.)))>
          (air5?2e-17:2e-10))))
      throw std::runtime_error("Wall global facet area differs");
    double quadrature_area=0.;
    astr_insitu::trace_mpi(MPI_Allreduce(&product.measure[0],&quadrature_area,1,MPI_DOUBLE,MPI_SUM,comm));
    if(!placeholder && (!std::isfinite(quadrature_area) || quadrature_area<=0.))
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
      mesh.controls["mean_requested"]=1.;
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

extern "C" int astr_insitu_device_render(const char* pipeline,const char* backend,const char* script,const char* profile,
    int fcomm,int step,double time,int nx,int ny,int nz,int ox,int oy,int oz,int global_cells,int global_y,int global_z,
    double* velocity,double* halo,double* diagnostic,double* reynolds_halo,double* favre_halo,
    int covered,double duration,double window_start,double window_end,std::int64_t host_budget,
    int curve,double* coordinates,double* coordinate_halo,std::int64_t budget,std::int64_t reserve,std::int64_t retained,
    const char* seeds)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  const std::string seed_layout=seeds?seeds:"line16";
  astr_insitu::tgv_streamline_seeds(seed_layout);
  double physical_step_scale=1.;
  const char* test_step_scale=std::getenv("ASTR_INSITU_TEST_CURVE_STEP_SCALE");
  if(test_step_scale) {
#ifdef ASTR_BUILD_TESTING
    if(!curve || !std::getenv("ASTR_INSITU_TEST_CURVE_TRACE_PREFIX") ||
        (std::string(test_step_scale)!="1" && std::string(test_step_scale)!="0.5"))
      throw std::invalid_argument("CURVE step sensitivity requires a diagnostic trace oracle and scale 1 or 0.5");
    physical_step_scale=std::string(test_step_scale)=="0.5"?.5:1.;
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
  const viskores::Id3 global_dimensions(global_cells,global_y,global_z);
  const bool demo=selected=="tgv256_demo" || curve_demo || boundary_layer;
  if(selected!="all" && selected!="q_surface" && selected!="q_streamlines" &&
     selected!="streamlines" && selected!="velocity_slice" && !demo) throw std::invalid_argument("Unsupported device product profile");
  if(boundary_layer?(global_dimensions!=viskores::Id3(64,32,24) || !curve || covered ||
      seed_layout!="bl-layered64" || std::string(pipeline?pipeline:"")!="standard-device"):
      (global_dimensions!=viskores::Id3(global_cells) ||
       (curve_demo?(global_cells!=32 && global_cells!=64 && global_cells!=128 && global_cells!=256):
        global_cells!=(demo?256:32))))
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
    if((selected!="q_surface" && selected!="streamlines" && !curve_demo && !boundary_layer) || !coordinates ||
        (global_cells!=32 && !curve_demo && !boundary_layer) ||
        ((selected=="q_surface" || curve_demo || boundary_layer) && covered) ||
        ((selected=="streamlines" || curve_demo || boundary_layer) && !coordinate_halo) ||
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
  const bool slice=(selected=="all" || selected=="velocity_slice") && astr_insitu_scene_due("velocity_slice");
  const bool surface=(selected=="all" || selected=="q_surface" || selected=="q_streamlines" || demo) &&
    astr_insitu_scene_due("q_surface");
  const bool lines=selected=="all" || selected=="streamlines" || selected=="q_streamlines" || demo;
  const bool instant=lines && astr_insitu_scene_due("instantaneous_streamlines");
  const bool crossing=lines && !demo && astr_insitu_scene_due("crossing_streamlines");
  const bool reynolds=(selected=="all" || selected=="streamlines") && covered && astr_insitu_scene_due("mean_reynolds_streamlines");
  const bool favre=(selected=="all" || selected=="streamlines") && covered && astr_insitu_scene_due("mean_favre_streamlines");
  const double iso=boundary_layer?.001:(demo?0.:.25);
  const std::string route=pipeline?pipeline:"";
  if(route!="compatible") {
#ifdef ASTR_INSITU_DEVICE_RENDERING
    if(route!="standard-device" && route!="direct-device")
      throw std::invalid_argument("Unsupported resident rendering pipeline; no host fallback");
    if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
      viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    int rank=0;astr_insitu::trace_mpi(MPI_Comm_rank(comm,&rank));
    std::vector<astr_insitu::DeviceMesh> meshes;
    if(slice || surface) {
    ProductStage extraction("device_geometry_extract_inclusive","ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION",fcomm,step);
    auto extracted=astr_insitu::extract_tgv_geometry(reinterpret_cast<viskores::Vec3f*>(velocity),
      reinterpret_cast<astr_insitu::DeviceDiagnostics*>(diagnostic),dimensions,offset,slice,surface,
      iso,nullptr,global_cells,curve?reinterpret_cast<viskores::Vec3f*>(coordinates):nullptr,global_dimensions);
    if(surface) meshes.push_back(resident_mesh("q_surface",extracted.surface,step,time,3,halo,
      reinterpret_cast<astr_insitu::DeviceDiagnostics*>(diagnostic),extent,offset,global_cells,rank,
      curve?reinterpret_cast<viskores::Vec3f*>(coordinates):nullptr,std::uint64_t(host_budget),nullptr,0,demo,iso,
      global_dimensions,boundary_layer?1.1:1.));
    if(curve) std::printf("ASTR_INSITU_CURVE_SURFACE_FRAME rank=%d step=%d physical_coordinates=1 "
      "field_download_bytes=0 geometry_host_bytes=0\n",rank,step);
    if(slice) meshes.push_back(resident_mesh("velocity_slice",
      astr_insitu::triangulate_device_slice(extracted.slice,dimensions,offset,global_cells),step,time,3,
      halo,nullptr,extent,offset,global_cells,rank));
    }
    if(boundary_layer && astr_insitu_scene_due("velocity_slice")) {
      const int global[3]={global_cells,global_y,global_z},local[3]={nx,ny,nz},start[3]={ox,oy,oz};
      const double origin[3]={0.,0.,45.},normal[3]={0.,0.,1.};
      meshes.push_back(astr_insitu::build_device_plane(fcomm,step,time,global,local,start,coordinates,velocity,
        origin,normal,budget,reserve,retained,host_budget,true,1.1));
    }
    if(instant || crossing || reynolds || favre) {
      auto actual=astr_insitu::pack_component_halo(halo,halo_dimensions);
      auto physical=curve?astr_insitu::pack_component_halo(coordinate_halo,halo_dimensions):
        viskores::cont::ArrayHandle<viskores::Vec3f>{};
      const int trace_axis=curve?(ny<global_cells?1:(nz<global_cells?2:0)):0;
      const auto add_trace=[&](const char* name,const viskores::cont::ArrayHandle<viskores::Vec3f>& field,
          bool constant,bool mean,const double* source) {
        ProductStage phase("resident_streamlines_inclusive","ASTR_IS8_RESIDENT_STREAMLINES",fcomm,step);
        auto trace=astr_insitu::trace_tgv_device(field,actual,extent,offset,comm,host_budget,
          constant,constant?trace_axis:0,constant,mean,global_cells,true,curve?&physical:nullptr,physical_step_scale,
          constant?"line16":seed_layout,global_dimensions);
        report_resident_trace(trace,name,rank,constant?trace_axis:0);
#ifdef ASTR_BUILD_TESTING
        if(test_step_scale && rank==0) for(int i=0;i<trace.particles;++i) {
          const auto& s=trace.final_state[i];
          std::printf("ASTR_INSITU_CURVE_STEP_SENSITIVITY step=%d product=%s scale=%.17g particle=%d "
            "status=%d x=%.17g y=%.17g z=%.17g length=%.17g\n",step,name,physical_step_scale,i,
            int(s[7]),s[0],s[1],s[2],s[4]);
        }
#endif
        auto mesh=resident_mesh(name,trace.resident_geometry,step,time,2,source,nullptr,
          extent,offset,global_cells,rank,nullptr,std::uint64_t(host_budget),curve?&physical:nullptr,trace_axis,demo,.25,
          global_dimensions,boundary_layer?1.1:1.);
        mesh.controls["streamline_seed_layout"]=trace.seed_layout=="tgv-stratified"?1.:
          (trace.seed_layout=="bl-layered64"?2.:0.);
        mesh.controls["streamline_seed_count"]=trace.seed_count;
        mesh.controls["streamline_particle_count"]=trace.particles;
        meshes.push_back(std::move(mesh));
      };
      if(instant) add_trace("instantaneous_streamlines",actual,false,false,halo);
      if(crossing) {
      viskores::cont::ArrayHandle<viskores::Vec3f> constant;
      viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
      ConstantVelocity constant_field;constant_field.axis=trace_axis;
      invoke(constant_field,viskores::cont::ArrayHandleIndex(actual.GetNumberOfValues()),constant);
      astr_insitu::synchronize_device_stage("ConstantVelocity");
      add_trace("crossing_streamlines",constant,true,false,halo);
      }
      if(reynolds || favre) {
        if(!reynolds_halo || !favre_halo || !std::isfinite(duration) || duration<=0.)
          throw std::invalid_argument("Resident mean products require authoritative coverage/halos");
        if(reynolds) add_trace("mean_reynolds_streamlines",astr_insitu::pack_component_halo(reynolds_halo,halo_dimensions),false,true,reynolds_halo);
        if(favre) add_trace("mean_favre_streamlines",astr_insitu::pack_component_halo(favre_halo,halo_dimensions),false,true,favre_halo);
      }
      if(curve) std::printf("ASTR_INSITU_CURVE_TRACE_FRAME rank=%d step=%d physical_coordinates=1 "
        "physical_hexahedral_interpolation=1 field_download_bytes=0 geometry_host_bytes=0\n",rank,step);
    }
    for(const char* name:{"q_surface","velocity_slice","instantaneous_streamlines","crossing_streamlines",
                         "mean_reynolds_streamlines","mean_favre_streamlines"})
      if(std::none_of(meshes.begin(),meshes.end(),[&](const astr_insitu::DeviceMesh& mesh){return mesh.name==name;}))
        meshes.push_back(resident_mesh(name,viskores::cont::DataSet{},step,time,
          std::string(name).find("streamlines")!=std::string::npos?2:3));
    if(selected=="streamlines" && covered) {
      if(!std::isfinite(duration) || !std::isfinite(window_start) || !std::isfinite(window_end) ||
          duration<=0. || window_end<=window_start)
        throw std::invalid_argument("Invalid mean streamline coverage metadata");
      for(auto& mesh:meshes) {
        mesh.controls["statistics_duration"]=duration;
        mesh.controls["statistics_window_start"]=window_start;
        mesh.controls["statistics_window_end"]=window_end;
      }
    }
    return astr_insitu::render_resident_products(pipeline,backend,script,fcomm,step,time,profile,std::move(meshes),covered);
#else
    throw std::invalid_argument("Strict device rendering is not built; no host fallback");
#endif
  }
  std::vector<astr_insitu::CompactMesh> meshes;
  if(slice || surface) {
    ProductStage extraction("device_geometry_extract_inclusive","ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION",fcomm,step);
    auto extracted=astr_insitu::extract_tgv_geometry(reinterpret_cast<viskores::Vec3f*>(velocity),
      reinterpret_cast<astr_insitu::DeviceDiagnostics*>(diagnostic),dimensions,offset,slice,surface,
      demo?0.:.25,nullptr,global_cells);
    extraction.finish();
    ProductStage read("device_geometry_read","ASTR_IS8_COMPACT_GEOMETRY_READ",fcomm,step);
    if(surface) meshes.push_back(geometry_mesh(extracted.surface,true,dimensions,offset));
    if(slice) meshes.push_back(geometry_mesh(extracted.slice,false,dimensions,offset));
  }
  if(instant || crossing || reynolds || favre) {
    auto actual=astr_insitu::pack_component_halo(halo,halo_dimensions);
    if(instant) {
    ProductStage instant("device_instant_lines_inclusive","ASTR_IS8_DEVICE_INSTANT_LINES",fcomm,step);
    auto trace=astr_insitu::trace_tgv_device(actual,actual,extent,offset,comm,host_budget,
      false,0,false,false,global_cells);
    meshes.push_back(streamline_mesh(trace,"instantaneous_streamlines"));
    instant.finish();
    }
    if(crossing) {
    ProductStage crossing("device_crossing_lines_inclusive","ASTR_IS8_DEVICE_CROSSING_LINES",fcomm,step);
    viskores::cont::ArrayHandle<viskores::Vec3f> constant;
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    invoke(ConstantVelocity{},viskores::cont::ArrayHandleIndex(actual.GetNumberOfValues()),constant);
    astr_insitu::synchronize_device_stage("ConstantVelocity");
    auto trace=astr_insitu::trace_tgv_device(constant,actual,extent,offset,comm,host_budget,true,0,true);
    meshes.push_back(streamline_mesh(trace,"crossing_streamlines"));
    crossing.finish();
    }
    if(reynolds || favre) {
      if(!reynolds_halo || !favre_halo || !std::isfinite(duration) || duration<=0.)
        throw std::invalid_argument("Covered device means require authoritative halos/duration");
      if(reynolds) {
      ProductStage reynolds_stage("device_reynolds_lines_inclusive","ASTR_IS8_DEVICE_REYNOLDS_LINES",fcomm,step);
      auto field=astr_insitu::pack_component_halo(reynolds_halo,halo_dimensions);
      auto trace=astr_insitu::trace_tgv_device(field,actual,extent,offset,comm,host_budget,false,0,false,true);
      meshes.push_back(streamline_mesh(trace,"mean_reynolds_streamlines"));
      reynolds_stage.finish();
      }
      if(favre) {
      ProductStage favre_stage("device_favre_lines_inclusive","ASTR_IS8_DEVICE_FAVRE_LINES",fcomm,step);
      auto field=astr_insitu::pack_component_halo(favre_halo,halo_dimensions);
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
    bytes+=(mesh.coordinates[0].size()*3+mesh.connectivity.size()+mesh.seed_ids.size()+mesh.directions.size())*8;
    for(const auto& field:mesh.point_fields) bytes+=field.second.size()*8;
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
