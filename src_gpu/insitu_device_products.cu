#include "insitu_device_geometry.h"
#include "insitu_device_streamlines.h"
#include "insitu_compact_mesh.h"
#include "insitu_device_product_audit.h"
#include "insitu_product_dispatch.h"
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>

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
      i%16,i<16?1:-1,trace.final_state[i][4],pi-trace.final_state[i][4],trace.minimum_step);
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
      const auto& final=trace.final_state[segment.seed+(segment.direction>0?0:16)];
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
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(viskores::Id,viskores::Vec3f& v) const {v=viskores::Vec3f(1.,0.,0.);}
};
#ifdef ASTR_INSITU_DEVICE_RENDERING
astr_insitu::DeviceMesh resident_mesh(const char* name,const viskores::cont::DataSet& data,
    int step,double time,int arity=3,const double* audit_halo=nullptr,
    const astr_insitu::DeviceDiagnostics* audit_diagnostics=nullptr,
    viskores::Id3 extent={0,0,0},viskores::Id3 offset={0,0,0},int global_cells=32,int rank=0) {
  auto owner=std::make_shared<astr_insitu::DeviceGeometryOwner>(data,step,time,arity);
  static const auto palette=astr_insitu::device_display_palette();
  const auto colors=owner->color_display(palette,global_cells==256);
  const auto& view=owner->get();
  if(audit_halo && std::getenv("ASTR_INSITU_RESIDENT_AUDIT") &&
      std::string(std::getenv("ASTR_INSITU_RESIDENT_AUDIT"))=="1") {
    const auto result=astr_insitu::audit_device_product(view,audit_halo,audit_diagnostics,
      extent,offset,global_cells,std::string(name)=="q_surface",std::string(name)=="velocity_slice",
      std::string(name)=="crossing_streamlines");
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
void report_resident_trace(const astr_insitu::DeviceStreamlines& trace,const char* name,int rank) {
  std::printf("ASTR_INSITU_RESIDENT_TRAJECTORY rank=%d product=%s rounds=%d transfers=%d "
    "control_read_bytes=%llu geometry_device_bytes=%llu\n",rank,name,trace.rounds,trace.transfers,
    static_cast<unsigned long long>(trace.control_read_bytes),static_cast<unsigned long long>(trace.geometry_bytes));
  if(std::string(name)=="crossing_streamlines") {
    double worst=0.;
    const double pi=std::acos(-1.);
    for(int i=0;i<16;++i) {
      const auto& point=trace.final_state[i];
      const double expected[3]={1.5*pi,pi/8.+i*(.75*pi/15.),pi/4.};
      for(int d=0;d<3;++d) {
        const double error=std::abs(point[d]-expected[d]);
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
      i%16,i<16?1:-1,trace.final_state[i][4],std::acos(-1.)-trace.final_state[i][4],trace.minimum_step);
}
#endif
}

extern "C" int astr_insitu_device_render(const char* pipeline,const char* backend,const char* script,const char* profile,
    int fcomm,int step,double time,int nx,int ny,int nz,int ox,int oy,int oz,int global_cells,
    double* velocity,double* halo,double* diagnostic,double* reynolds_halo,double* favre_halo,
    int covered,double duration,double window_start,double window_end,std::int64_t host_budget)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  if(!initialized) {
    int argc=1;char program[]="astr-insitu-device";char* args[]={program,nullptr};char** argv=args;
    viskores::cont::Initialize(argc,argv);initialized=true;
  }
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  const std::string selected=profile?profile:"";
  const bool demo=selected=="tgv256_demo";
  if(selected!="all" && selected!="q_surface" && selected!="q_streamlines" &&
     selected!="streamlines" && selected!="velocity_slice" && !demo) throw std::invalid_argument("Unsupported device product profile");
  if(global_cells!=(demo?256:32)) throw std::invalid_argument("Device product profile/resolution differs");
  if(host_budget<=0 || !std::isfinite(time) || step<0) throw std::invalid_argument("Invalid device frame identity/budget");
  const viskores::Id3 extent(nx,ny,nz),dimensions(nx+1,ny+1,nz+1),offset(ox,oy,oz),halo_dimensions(nx+7,ny+7,nz+7);
  const bool slice=(selected=="all" || selected=="velocity_slice") && astr_insitu_scene_due("velocity_slice");
  const bool surface=(selected=="all" || selected=="q_surface" || selected=="q_streamlines" || demo) &&
    astr_insitu_scene_due("q_surface");
  const bool lines=selected=="all" || selected=="streamlines" || selected=="q_streamlines" || demo;
  const bool instant=lines && astr_insitu_scene_due("instantaneous_streamlines");
  const bool crossing=lines && !demo && astr_insitu_scene_due("crossing_streamlines");
  const bool reynolds=selected=="all" && covered && astr_insitu_scene_due("mean_reynolds_streamlines");
  const bool favre=selected=="all" && covered && astr_insitu_scene_due("mean_favre_streamlines");
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
      demo?0.:.25,nullptr,global_cells);
    if(surface) meshes.push_back(resident_mesh("q_surface",extracted.surface,step,time,3,halo,
      reinterpret_cast<astr_insitu::DeviceDiagnostics*>(diagnostic),extent,offset,global_cells,rank));
    if(slice) meshes.push_back(resident_mesh("velocity_slice",
      astr_insitu::triangulate_device_slice(extracted.slice,dimensions,offset,global_cells),step,time,3,
      halo,nullptr,extent,offset,global_cells,rank));
    }
    if(instant || crossing || reynolds || favre) {
      auto actual=astr_insitu::pack_component_halo(halo,halo_dimensions);
      const auto add_trace=[&](const char* name,const viskores::cont::ArrayHandle<viskores::Vec3f>& field,
          bool constant,bool mean,const double* source) {
        ProductStage phase("resident_streamlines_inclusive","ASTR_IS8_RESIDENT_STREAMLINES",fcomm,step);
        auto trace=astr_insitu::trace_tgv_device(field,actual,extent,offset,comm,host_budget,
          constant,0,constant,mean,global_cells,true);
        report_resident_trace(trace,name,rank);
        meshes.push_back(resident_mesh(name,trace.resident_geometry,step,time,2,source,nullptr,
          extent,offset,global_cells,rank));
      };
      if(instant) add_trace("instantaneous_streamlines",actual,false,false,halo);
      if(crossing) {
      viskores::cont::ArrayHandle<viskores::Vec3f> constant;
      viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
      invoke(ConstantVelocity{},viskores::cont::ArrayHandleIndex(actual.GetNumberOfValues()),constant);
      astr_insitu::synchronize_device_stage("ConstantVelocity");
      add_trace("crossing_streamlines",constant,true,false,halo);
      }
      if(reynolds || favre) {
        if(!reynolds_halo || !favre_halo || !std::isfinite(duration) || duration<=0.)
          throw std::invalid_argument("Resident mean products require authoritative coverage/halos");
        if(reynolds) add_trace("mean_reynolds_streamlines",astr_insitu::pack_component_halo(reynolds_halo,halo_dimensions),false,true,reynolds_halo);
        if(favre) add_trace("mean_favre_streamlines",astr_insitu::pack_component_halo(favre_halo,halo_dimensions),false,true,favre_halo);
      }
    }
    for(const char* name:{"q_surface","velocity_slice","instantaneous_streamlines","crossing_streamlines",
                         "mean_reynolds_streamlines","mean_favre_streamlines"})
      if(std::none_of(meshes.begin(),meshes.end(),[&](const astr_insitu::DeviceMesh& mesh){return mesh.name==name;}))
        meshes.push_back(resident_mesh(name,viskores::cont::DataSet{},step,time,
          std::string(name).find("streamlines")!=std::string::npos?2:3));
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
