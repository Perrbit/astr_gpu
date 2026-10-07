#include <catalyst.h>
#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <exception>
#include "insitu_compact_mesh.h"
#include "insitu_product_dispatch.h"
#ifdef ASTR_INSITU_DEVICE_RENDERING
#include "insitu_device_vtk_mapper.h"
#include <vtkColorTransferFunction.h>
#include <vtkSMTransferFunctionPresets.h>
#include <vtk_jsoncpp.h>
#endif
#ifdef ASTR_INSITU_DEVICE_PRODUCTS
#include <nvtx3/nvToolsExt.h>
extern "C" void astr_insitu_device_range_push(int stage) {
  const char* names[]={"ASTR_IS8_DEVICE_SAMPLE", "ASTR_IS8_DEVICE_MEAN_SUPPLY",
    "ASTR_X4_DEVICE_WALL_SAMPLE", "ASTR_X4_DEVICE_PLANE_SAMPLE",
    "ASTR_X4_DEVICE_WALL_CONSUMER", "ASTR_X4_DEVICE_PLANE_CONSUMER", "ASTR_X4_DEVICE_VOLUME_CONSUMER",
    "ASTR_X4_DEVICE_WALL_STATISTICS", "ASTR_X4_DEVICE_VOLUME_STATISTICS", "ASTR_X4_DEVICE_WALL_MEAN_SUPPLY",
    "ASTR_X4_DEVICE_WALL_SEPARATION_REDUCE"};
  if(stage<1 || stage>11) std::abort();
  nvtxRangePushA(names[stage-1]);
}
extern "C" void astr_insitu_device_range_pop() { nvtxRangePop(); }
#endif
#ifdef ASTR_CUDA_RENDERER
extern "C" int astr_insitu_map_current_cuda(int*,char*,int,char*,int);
extern "C" int astr_insitu_resource_check(const char*);
#endif

namespace {
bool active = false;
MPI_Comm render_comm = MPI_COMM_NULL;
conduit_node* retained = nullptr;
std::vector<double> coordinates, primitive, gradients, means;
std::vector<conduit_int64> plane_cells;
std::vector<astr_insitu::CompactMesh> retained_products;
#ifdef ASTR_INSITU_DEVICE_RENDERING
std::vector<astr_insitu::DeviceMesh> retained_device_products;
std::map<std::string,vtkSmartPointer<vtkActor>> device_actors;
std::string resident_pipeline;
#endif
double empty_value = 0.;
conduit_int64 empty_cell = 0;
int shape[3] = {0,0,0};
int selected_profile = 0;
bool native = false;
std::string output_directory, expected_uuid;
int allocation_failure(const char* stage, const char* detail, MPI_Comm comm)
{
  std::fprintf(stderr,"ASTR INSITU BRIDGE ERROR: %s: %s\n",stage,detail);
  MPI_Abort(comm,1);
  return 1;
}
void check_mpi(int status, const char* stage, MPI_Comm comm)
{
  if (status != MPI_SUCCESS) {
    allocation_failure("MPI",stage,comm);
    std::abort();
  }
}
void check(catalyst_status status)
{
  int bad = status != catalyst_status_ok, any = 0;
  check_mpi(MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, render_comm),
      "Catalyst status consensus MPI failure",render_comm);
  if (any) MPI_Abort(render_comm, 1);
}
}

extern "C" int astr_insitu_mesh_pipeline_available(const char* pipeline)
{
  // New routes are admitted only when their actual bridge has been built.
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(pipeline && (std::string(pipeline)=="standard-device" ||
      std::string(pipeline)=="direct-device")) return 1;
#endif
  return pipeline && std::string(pipeline) == "compatible";
}

extern "C" int astr_insitu_mesh_configure(const char* output, int fcomm)
try {
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  int bad = active || native || !output || !*output;
  char uuid[37] = {}, message[256] = {};
  int index = -1;
#ifdef ASTR_CUDA_RENDERER
  bad |= astr_insitu_map_current_cuda(&index,uuid,sizeof(uuid),message,sizeof(message));
#else
  bad = 1;
  std::snprintf(message,sizeof(message),"Native EGL renderer requires a current CUDA context");
#endif
  bad |= std::getenv("CATALYST_IMPLEMENTATION_PREFER_ENV") != nullptr;
  int any = 0;
  check_mpi(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm),"render mapping consensus",comm);
  if (any) { std::fprintf(stderr,"ASTR native render mapping: %s\n",message); return 1; }
  const std::string ordinal=std::to_string(index);
  bad = setenv("VTK_EGL_DEVICE_INDEX",ordinal.c_str(),1) != 0;
  bad |= setenv("VTK_DEFAULT_OPENGL_WINDOW","vtkEGLRenderWindow",1) != 0;
  check_mpi(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm),"render environment consensus",comm);
  if (any) return 1;
  output_directory=output;
  expected_uuid=uuid;
  native=true;
  return 0;
} catch (const std::exception& error) {
  return allocation_failure("configure",error.what(),MPI_Comm_f2c(fcomm));
} catch (...) {
  return allocation_failure("configure","unknown C++ exception",MPI_Comm_f2c(fcomm));
}

// Synchronous lifecycle; backend-visible storage survives between frames.
static int mesh_execute(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, double time,
    double* xyz, double* fields, double* derived, int has_mean, double* mean_fields, int profile)
try {
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  if (profile<0 || profile>6 || (profile && !native) ||
      (profile && has_mean && (profile<5 || has_mean!=2)))
    return allocation_failure("selection","invalid selected product layout",comm);
  if (profile==4 && (nz!=1 || nx<0 || ny<0 || ((nx==0)!=(ny==0))))
    return allocation_failure("slice","invalid index-plane shape",comm);
  if (profile==5 && (nx<2 || ny<2 || nz<0 || nz>2))
    return allocation_failure("wall","invalid wall plane shape",comm);
  if (profile==6 && (nx<2 || ny<2 || nz<0 || nz>1))
    return allocation_failure("wall","invalid AIR5 wall plane shape",comm);
  if (!active) {
  const double initialize_start=MPI_Wtime();
  check_mpi(MPI_Comm_dup(comm, &render_comm),"create render communicator",comm);
  if (!native && (std::getenv("CATALYST_IMPLEMENTATION_PREFER_ENV") ||
      !std::getenv("VTK_EGL_DEVICE_INDEX") || !std::getenv("ASTR_PROBE_EXPECTED_UUID"))) {
    std::fprintf(stderr, "Mesh probe requires explicit UUID-mapped EGL launcher\n");
    MPI_Abort(comm, 1);
  }
  setenv("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow", 1);
  auto init = conduit_node_create();
  conduit_node_set_path_char8_str(init, "catalyst_load/implementation", "paraview");
  conduit_node_set_path_char8_str(init, "catalyst_load/search_paths/astr", backend);
  conduit_node_set_path_char8_str(init, "catalyst/scripts/probe/filename", script);
  if (native) {
    auto args=conduit_node_fetch(init,"catalyst/scripts/probe/args");
    conduit_node_set_char8_str(conduit_node_append(args),output_directory.c_str());
    conduit_node_set_char8_str(conduit_node_append(args),expected_uuid.c_str());
    if (profile) {
      const char* products[]={"all","q_surface","streamlines","q_streamlines","velocity_slice","channel_walls","air5_walls"};
      conduit_node_set_char8_str(conduit_node_append(args),products[profile]);
    }
  }
  conduit_node_set_path_int64(init, "catalyst/mpi_comm", MPI_Comm_c2f(render_comm));
  check(catalyst_initialize(init));
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("initialized");
#endif
  conduit_node_destroy(init);
  retained = conduit_node_create();
  shape[0]=nx; shape[1]=ny; shape[2]=nz;
  selected_profile=profile;
  coordinates.resize(long(nx)*ny*nz*3);
  primitive.resize(long(nx)*ny*nz*(profile==6 ? 18 : (profile==5 ? 4 : (profile ? 3 : 11))));
  gradients.resize(long(nx)*ny*nz*(profile ? (profile==2 || profile>=4 ? 0 : 1) : 14));
  if (profile>=4) {
    for (int layer=0;layer<nz;++layer) for (int j=0;j<ny-1;++j) for (int i=0;i<nx-1;++i) {
      const conduit_int64 a=(conduit_int64(layer)*ny+j)*nx+i;
      plane_cells.insert(plane_cells.end(),{a,a+1,a+nx+1,a+nx});
    }
  }
  active = true;
  const char* timing=std::getenv("ASTR_INSITU_TIMING");
  if (timing && (std::string(timing)=="1" || std::string(timing)=="true" ||
      std::string(timing)=="TRUE" || std::string(timing)=="t" || std::string(timing)=="T" ||
      std::string(timing)=="on" || std::string(timing)=="ON")) {
    int rank;
    check_mpi(MPI_Comm_rank(comm,&rank),"query initialize rank",comm);
    std::printf("ASTR_INSITU_STAGE_TIMING catalyst_initialization_inclusive %d %d %.16e\n",
        step,rank,MPI_Wtime()-initialize_start);
  }
  }
  if (shape[0]!=nx || shape[1]!=ny || shape[2]!=nz || profile!=selected_profile) MPI_Abort(comm, 1);
  const double copy_start = MPI_Wtime();
  std::size_t copied_values=coordinates.size();
  if (!coordinates.empty()) std::copy(xyz, xyz+coordinates.size(), coordinates.begin());
  const auto nodes=long(nx)*ny*nz;
  if(astr_insitu_independent_products() && !profile) {
    if(nodes) std::copy(fields+6*nodes,fields+9*nodes,primitive.begin()+6*nodes);
    copied_values+=3*nodes;
    if(astr_insitu_scene_due("q_surface")) {
      if(nodes) std::copy(derived+9*nodes,derived+10*nodes,gradients.begin()+9*nodes);
      copied_values+=nodes;
    }
  } else {
    if (!primitive.empty()) std::copy(fields, fields+primitive.size(), primitive.begin());
    if (!gradients.empty()) std::copy(derived, derived+gradients.size(), gradients.begin());
    copied_values+=primitive.size()+gradients.size();
  }
  const double copy_seconds = MPI_Wtime()-copy_start;
  xyz=coordinates.empty() ? &empty_value : coordinates.data();
  fields=primitive.empty() ? &empty_value : primitive.data(); derived=gradients.data();
  auto params = retained;
  conduit_node_set_path_int64(params, "catalyst/state/timestep", step);
  conduit_node_set_path_double(params, "catalyst/state/time", time);
  conduit_node_set_path_char8_str(params, "catalyst/channels/grid/type", "mesh");
  auto mesh = conduit_node_fetch(params, "catalyst/channels/grid/data");
  int rank;
  check_mpi(MPI_Comm_rank(comm, &rank),"query mesh rank",comm);
  conduit_node_set_path_int64(mesh, "state/domain_id", rank);
  conduit_node_set_path_char8_str(mesh, "coordsets/coords/type", "explicit");
  const long count = long(nx)*ny*nz;
  const char* axes[] = {"x", "y", "z"};
  for (int d=0; d<3; ++d) {
    const auto key = std::string("coordsets/coords/values/")+axes[d];
    conduit_node_set_path_external_float64_ptr(mesh, key.c_str(), xyz+d*count, count);
  }
  conduit_node_set_path_char8_str(mesh, "topologies/mesh/coordset", "coords");
  if (profile>=4) {
    conduit_node_set_path_char8_str(mesh, "topologies/mesh/type", "unstructured");
    conduit_node_set_path_char8_str(mesh, "topologies/mesh/elements/shape", "quad");
    conduit_node_set_path_external_int64_ptr(mesh,"topologies/mesh/elements/connectivity",
      plane_cells.empty() ? &empty_cell : plane_cells.data(),plane_cells.size());
  } else {
    conduit_node_set_path_char8_str(mesh, "topologies/mesh/type", "structured");
    conduit_node_set_path_int64(mesh, "topologies/mesh/elements/dims/i", nx-1);
    conduit_node_set_path_int64(mesh, "topologies/mesh/elements/dims/j", ny-1);
    conduit_node_set_path_int64(mesh, "topologies/mesh/elements/dims/k", nz-1);
  }
  auto field = [mesh,count](const char* name, double* data) {
    const auto base = std::string("fields/")+name;
    conduit_node_set_path_char8_str(mesh, (base+"/association").c_str(), "vertex");
    conduit_node_set_path_char8_str(mesh, (base+"/topology").c_str(), "mesh");
    conduit_node_set_path_external_float64_ptr(mesh, (base+"/values").c_str(), data, count);
  };
  const char* names[] = {"q1","q2","q3","q4","q5","rho","u","v","w","pressure","temperature"};
  if (profile==6) {
    const char* wall_names[]={"rho","u","v","w","temperature","vibrational_temperature","pressure",
      "Y_N2","Y_O2","Y_N","Y_O","Y_NO","wall_shear_x","wall_heat_tr_into_gas",
      "wall_heat_v_into_gas","wall_heat_species_into_gas","wall_heat_into_gas","wall_normal_y"};
    for (int c=0;c<18;++c) field(wall_names[c],fields+c*count);
  } else if (profile==5) {
    const char* wall_names[]={"wall_pressure","wall_shear_x","wall_heat_into_gas","wall_normal_y"};
    for (int c=0;c<4;++c) field(wall_names[c],fields+c*count);
  } else if (profile) {
    for (int c=0; c<3; ++c) field(names[c+6], fields+c*count);
  } else {
    for (int c=0; c<11; ++c) {
      if(astr_insitu_independent_products() && (c<6 || c>8)) continue;
      field(names[c], fields+c*count);
    }
  }
  const char* diagnostics[] = {"du_dx","dv_dx","dw_dx","du_dy","dv_dy","dw_dy",
      "du_dz","dv_dz","dw_dz","Q_rs","divergence","omega_x","omega_y","omega_z"};
  if (profile) {
    if (profile!=2 && profile<4 && astr_insitu_scene_due("q_surface")) field("Q_rs",derived);
    else if(conduit_node_has_path(mesh,"fields/Q_rs")) conduit_node_remove_path(mesh,"fields/Q_rs");
  } else {
    for (int c=0; c<14; ++c) {
      const auto key=std::string("fields/")+diagnostics[c];
      if(astr_insitu_independent_products() && (c!=9 || !astr_insitu_scene_due("q_surface"))) {
        if(conduit_node_has_path(mesh,key.c_str())) conduit_node_remove_path(mesh,key.c_str());
      } else field(diagnostics[c], derived+c*count);
    }
  }
  const char* mean_names[] = {"mean_u_reynolds","mean_v_reynolds","mean_w_reynolds",
    "mean_u_favre","mean_v_favre","mean_w_favre","statistics_duration"};
  if (has_mean==2) {
    const char* air5_names[]={"rho","u","v","w","temperature","vibrational_temperature","pressure",
      "Y_N2","Y_O2","Y_N","Y_O","Y_NO","wall_shear_x","wall_heat_tr_into_gas",
      "wall_heat_v_into_gas","wall_heat_species_into_gas","wall_heat_into_gas","wall_normal_y"};
    const char* channel_names[]={"wall_pressure","wall_shear_x","wall_heat_into_gas"};
    const int nf=profile==6 ? 18 : 3;
    means.resize((3*nf+3)*count);
    if (!means.empty()) std::copy(mean_fields,mean_fields+means.size(),means.begin());
    double* mean_data=means.empty() ? &empty_value : means.data();
    const char* kinds[]={"mean_","variance_","rms_"};
    for (int kind=0;kind<3;++kind) for (int c=0;c<nf;++c) {
      const std::string name=std::string(kinds[kind])+(profile==6 ? air5_names[c] : channel_names[c]);
      field(name.c_str(),mean_data+(kind*nf+c)*count);
    }
    field("statistics_duration",mean_data+3*nf*count);
    field("statistics_window_start",mean_data+(3*nf+1)*count);
    field("statistics_window_end",mean_data+(3*nf+2)*count);
  } else if (has_mean) {
    means.resize(7*count);
    double* mean_data=means.empty() ? &empty_value : means.data();
    for (int c=0;c<7;++c) {
      const bool wanted=!astr_insitu_independent_products() || c==6 ||
        astr_insitu_scene_due(c<3 ? "mean_reynolds_streamlines" : "mean_favre_streamlines");
      const auto key=std::string("fields/")+mean_names[c];
      if(wanted) {
        if(count) std::copy(mean_fields+c*count,mean_fields+(c+1)*count,means.begin()+c*count);
        field(mean_names[c],mean_data+c*count);
        copied_values+=count;
      } else if(conduit_node_has_path(mesh,key.c_str())) conduit_node_remove_path(mesh,key.c_str());
    }
  } else {
    for (auto name:mean_names) {
      const auto key=std::string("fields/")+name;
      if (conduit_node_has_path(mesh,key.c_str())) conduit_node_remove_path(mesh,key.c_str());
    }
    means.clear();
  }
  // Cells are disjoint; shared interface points are retained, not hidden as cells.
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_before");
#endif
  const double execute_start = MPI_Wtime();
  check(catalyst_execute(params));
  const double execute_seconds = MPI_Wtime()-execute_start;
  std::printf("ASTR_INSITU_BRIDGE_TIMING rank=%d step=%d copy=%.9g execute_inclusive=%.9g copy_bytes=%zu\n",
      rank,step,copy_seconds,execute_seconds,
      copied_values*sizeof(double));
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_after");
#endif
  return 0;
} catch (const std::exception& error) {
  return allocation_failure("execute",error.what(),MPI_Comm_f2c(fcomm));
} catch (...) {
  return allocation_failure("execute","unknown C++ exception",MPI_Comm_f2c(fcomm));
}

extern "C" int astr_insitu_mesh_probe(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, double time,
    double* xyz, double* fields, double* derived, int has_mean, double* mean_fields)
{
  return mesh_execute(backend,script,fcomm,nx,ny,nz,step,time,xyz,fields,derived,has_mean,mean_fields,0);
}

extern "C" int astr_insitu_mesh_selected(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, double time,
    double* xyz, double* velocity, double* q, int profile)
{
  return mesh_execute(backend,script,fcomm,nx,ny,nz,step,time,xyz,velocity,q,0,nullptr,profile);
}

extern "C" int astr_insitu_mesh_wall_statistics(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, double time,
    double* xyz, double* fields, double* statistics, int profile)
{
  return mesh_execute(backend,script,fcomm,nx,ny,nz,step,time,xyz,fields,nullptr,2,statistics,profile);
}

#ifdef ASTR_INSITU_DEVICE_RENDERING
std::vector<double> astr_insitu::device_display_palette() {
  const auto& preset=vtkSMTransferFunctionPresets::GetInstance()->GetFirstPresetWithName(
    "Cool to Warm (Extended)");
  const auto& points=preset["RGBPoints"];
  if(preset["ColorSpace"].asString()!="Lab" || !points.isArray() ||
      points.size()<8 || points.size()%4)
    throw std::runtime_error("Unexpected fixed ParaView display preset");
  vtkNew<vtkColorTransferFunction> transfer;
  transfer->SetColorSpaceToLab();
  for(Json::ArrayIndex i=0;i<points.size();i+=4)
    transfer->AddRGBPoint(points[i].asDouble(),points[i+1].asDouble(),
      points[i+2].asDouble(),points[i+3].asDouble());
  std::vector<double> table(4096*3);
  for(int i=0;i<4096;++i) transfer->GetColor(i/4095.,table.data()+3*i);
  return table;
}

extern "C" std::uintptr_t astr_insitu_resident_actor(const char* name) {
  const auto found=device_actors.find(name?name:"");
  return found==device_actors.end()?0:reinterpret_cast<std::uintptr_t>(found->second.GetPointer());
}

int astr_insitu::render_resident_products(const char* pipeline,const char* backend,const char* script,int fcomm,
    int step,double time,const char* profile,std::vector<DeviceMesh> products,bool covered)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  const std::string route=pipeline?pipeline:"";
  if(!native || !backend || !script || !profile || products.empty() ||
      (route!="standard-device" && route!="direct-device") ||
      (active && (selected_profile!=8 || resident_pipeline!=route)))
    return allocation_failure("resident products","invalid renderer session",comm);
  if(!active) {
    if(setenv("ASTR_VTK_STRICT_DEVICE_ACCESS","1",1))
      return allocation_failure("resident products","cannot enable strict CPU-access guard",comm);
    check_mpi(MPI_Comm_dup(comm,&render_comm),"create resident render communicator",comm);
    auto init=conduit_node_create();
    conduit_node_set_path_char8_str(init,"catalyst_load/implementation","paraview");
    conduit_node_set_path_char8_str(init,"catalyst_load/search_paths/astr",backend);
    conduit_node_set_path_char8_str(init,"catalyst/scripts/probe/filename",script);
    auto args=conduit_node_fetch(init,"catalyst/scripts/probe/args");
    for(const char* value:{output_directory.c_str(),expected_uuid.c_str(),profile,route.c_str()})
      conduit_node_set_char8_str(conduit_node_append(args),value);
    conduit_node_set_path_int64(init,"catalyst/mpi_comm",MPI_Comm_c2f(render_comm));
    check(catalyst_initialize(init));conduit_node_destroy(init);
    selected_profile=8;active=true;resident_pipeline=route;
    astr_insitu_resource_check("initialized");
  }
  auto params=conduit_node_create();
  conduit_node_set_path_int64(params,"catalyst/state/timestep",step);
  conduit_node_set_path_double(params,"catalyst/state/time",time);
  int rank=0;check_mpi(MPI_Comm_rank(comm,&rank),"resident domain rank",comm);
  for(auto& product:products) {
    const auto& view=product.draw;
    if(!product.owner || (view.arity!=2 && view.arity!=3) || (view.cells && !view.points))
      return allocation_failure("resident products","invalid device owner/extent",comm);
    const auto base="catalyst/channels/"+product.name;
    conduit_node_set_path_char8_str(params,(base+"/type").c_str(),"mesh");
    auto mesh=conduit_node_fetch(params,(base+"/data").c_str());
    conduit_node_set_path_int64(mesh,"state/domain_id",rank);
    for(int d=0;d<6;++d)
      conduit_node_set_path_double(mesh,("state/fields/bound"+std::to_string(d)).c_str(),view.bounds[d]);
    conduit_node_set_path_int64(mesh,"state/fields/device_points",view.points);
    conduit_node_set_path_int64(mesh,"state/fields/device_cells",view.cells);
    conduit_node_set_path_int64(mesh,"state/fields/device_arity",view.arity);
    conduit_node_set_path_int64(mesh,"state/fields/mean_covered",covered);
    for(const auto& field:product.controls) {
      if(!std::isfinite(field.second))
        return allocation_failure("resident products","nonfinite scalar control",comm);
      conduit_node_set_path_double(mesh,("state/fields/"+field.first).c_str(),field.second);
    }
    conduit_node_set_path_char8_str(mesh,"coordsets/coords/type","explicit");
    conduit_node_set_path_char8_str(mesh,"topologies/mesh/coordset","coords");
    conduit_node_set_path_char8_str(mesh,"topologies/mesh/type","unstructured");
    conduit_node_set_path_char8_str(mesh,"topologies/mesh/elements/shape",view.arity==3?"tri":"line");
    const bool arrays=route=="standard-device" && view.points;
    const char* axes[]={"x","y","z"};
    for(int d=0;d<3;++d) {
      const auto key=std::string("coordsets/coords/values/")+axes[d];
      if(arrays) conduit_node_set_path_external_float32_ptr_detailed(mesh,key.c_str(),
        const_cast<float*>(view.positions),view.points,d*sizeof(float),3*sizeof(float),sizeof(float),0);
      else conduit_node_set_path_external_float64_ptr(mesh,key.c_str(),&empty_value,0);
    }
    if(arrays) {
      conduit_node_set_path_external_uint32_ptr(mesh,"topologies/mesh/elements/connectivity",
        const_cast<unsigned int*>(view.indices),view.cells*view.arity);
      conduit_node_set_path_char8_str(mesh,"fields/_astr_display_rgba/association","vertex");
      conduit_node_set_path_char8_str(mesh,"fields/_astr_display_rgba/topology","mesh");
      for(int d=0;d<4;++d)
        conduit_node_set_path_external_uint8_ptr_detailed(mesh,
          ("fields/_astr_display_rgba/values/"+std::to_string(d)).c_str(),
          const_cast<unsigned char*>(view.colors),view.points,d,4,1,0);
    } else conduit_node_set_path_external_int64_ptr(mesh,"topologies/mesh/elements/connectivity",&empty_cell,0);
    if(route=="direct-device") {
      auto& actor=device_actors[product.name];
      if(!actor) {
        actor=vtkSmartPointer<vtkActor>::New();
        vtkNew<astr_insitu::DirectDeviceMapper> mapper;
        actor->SetMapper(mapper);
      }
      static_cast<astr_insitu::DirectDeviceMapper*>(actor->GetMapper())->set_view(view);
    }
  }
  astr_insitu_resource_check("frame_before");
  const double start=MPI_Wtime();
  nvtxRangePushA("ASTR_IS8_RESIDENT_RENDER");
  check(catalyst_execute(params));
  nvtxRangePop();
  if(retained) conduit_node_destroy(retained);
  retained=params;retained_device_products=std::move(products);
  std::printf("ASTR_INSITU_RESIDENT_BRIDGE rank=%d step=%d pipeline=%s geometry_host_bytes=0 execute_inclusive=%.9g\n",
    rank,step,route.c_str(),MPI_Wtime()-start);
  astr_insitu_resource_check("frame_after");
  return 0;
} catch(const std::exception& error) {
  return allocation_failure("resident products",error.what(),MPI_Comm_f2c(fcomm));
} catch(...) {
  return allocation_failure("resident products","unknown exception",MPI_Comm_f2c(fcomm));
}
#endif

int astr_insitu::render_compact_products(const char* backend,const char* script,int fcomm,
    int step,double time,const char* profile,bool covered,double duration,
    double window_start,double window_end,std::vector<CompactMesh> products)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  if(!native || !backend || !script || !profile || products.empty() || (active && selected_profile!=7))
    return allocation_failure("device products","invalid compact renderer session",comm);
  static_assert(sizeof(std::int64_t)==sizeof(conduit_int64),"Conduit geometry requires int64 connectivity");
  if(!active) {
    check_mpi(MPI_Comm_dup(comm,&render_comm),"create compact render communicator",comm);
    auto init=conduit_node_create();
    conduit_node_set_path_char8_str(init,"catalyst_load/implementation","paraview");
    conduit_node_set_path_char8_str(init,"catalyst_load/search_paths/astr",backend);
    conduit_node_set_path_char8_str(init,"catalyst/scripts/probe/filename",script);
    auto args=conduit_node_fetch(init,"catalyst/scripts/probe/args");
    for(const char* value:{output_directory.c_str(),expected_uuid.c_str(),profile,"device"})
      conduit_node_set_char8_str(conduit_node_append(args),value);
    conduit_node_set_path_int64(init,"catalyst/mpi_comm",MPI_Comm_c2f(render_comm));
    check(catalyst_initialize(init));conduit_node_destroy(init);
    selected_profile=7;active=true;
#ifdef ASTR_CUDA_RENDERER
    astr_insitu_resource_check("initialized");
#endif
  }
  auto params=conduit_node_create();
  conduit_node_set_path_int64(params,"catalyst/state/timestep",step);
  conduit_node_set_path_double(params,"catalyst/state/time",time);
  int rank=0;check_mpi(MPI_Comm_rank(comm,&rank),"compact domain rank",comm);
  std::size_t copy_bytes=0;
  for(auto& product:products) {
    const auto nodes=product.coordinates[0].size();
    if(product.coordinates[1].size()!=nodes || product.coordinates[2].size()!=nodes)
      return allocation_failure("device products","coordinate component sizes differ",comm);
    const int arity=product.shape=="tri"?3:(product.shape=="quad"?4:(product.shape=="line"?2:0));
    if(!arity || product.connectivity.size()%arity)
      return allocation_failure("device products","invalid final connectivity",comm);
    const auto cells=product.connectivity.size()/arity;
    for(auto index:product.connectivity) if(index<0 || std::uint64_t(index)>=nodes)
      return allocation_failure("device products","final vertex index out of range",comm);
    const auto base="catalyst/channels/"+product.name;
    conduit_node_set_path_char8_str(params,(base+"/type").c_str(),"mesh");
    auto mesh=conduit_node_fetch(params,(base+"/data").c_str());
    conduit_node_set_path_int64(mesh,"state/domain_id",rank);
    conduit_node_set_path_int64(mesh,"state/fields/device_products",1);
    conduit_node_set_path_int64(mesh,"state/fields/mean_covered",covered);
    conduit_node_set_path_double(mesh,"state/fields/statistics_duration",duration);
    conduit_node_set_path_double(mesh,"state/fields/statistics_window_start",window_start);
    conduit_node_set_path_double(mesh,"state/fields/statistics_window_end",window_end);
    conduit_node_set_path_char8_str(mesh,"coordsets/coords/type","explicit");
    const char* axes[]={"x","y","z"};
    for(int d=0;d<3;++d) {
      const auto path=std::string("coordsets/coords/values/")+axes[d];
      conduit_node_set_path_external_float64_ptr(mesh,path.c_str(),
        nodes?product.coordinates[d].data():&empty_value,nodes);
    }
    conduit_node_set_path_char8_str(mesh,"topologies/mesh/coordset","coords");
    conduit_node_set_path_char8_str(mesh,"topologies/mesh/type","unstructured");
    conduit_node_set_path_char8_str(mesh,"topologies/mesh/elements/shape",product.shape.c_str());
    conduit_node_set_path_external_int64_ptr(mesh,"topologies/mesh/elements/connectivity",
      product.connectivity.empty()?&empty_cell:reinterpret_cast<conduit_int64*>(product.connectivity.data()),
      product.connectivity.size());
    for(auto& field:product.point_fields) {
      if(field.second.size()!=nodes)
        return allocation_failure("device products","final point field size differs",comm);
      const auto path="fields/"+field.first;
      conduit_node_set_path_char8_str(mesh,(path+"/association").c_str(),"vertex");
      conduit_node_set_path_char8_str(mesh,(path+"/topology").c_str(),"mesh");
      conduit_node_set_path_external_float64_ptr(mesh,(path+"/values").c_str(),
        nodes?field.second.data():&empty_value,nodes);
      copy_bytes+=field.second.size()*sizeof(double);
    }
    const auto cell_field=[&](const char* name,std::vector<std::int64_t>& values) {
      if(values.size()!=cells) return allocation_failure("device products","final cell field size differs",comm);
      const auto path=std::string("fields/")+name;
      conduit_node_set_path_char8_str(mesh,(path+"/association").c_str(),"element");
      conduit_node_set_path_char8_str(mesh,(path+"/topology").c_str(),"mesh");
      conduit_node_set_path_external_int64_ptr(mesh,(path+"/values").c_str(),
        values.empty()?&empty_cell:reinterpret_cast<conduit_int64*>(values.data()),values.size());
      return 0;
    };
    if(product.shape=="line") {
      cell_field("SeedIds",product.seed_ids);cell_field("IntegrationDirection",product.directions);
    }
    copy_bytes+=(nodes*3+product.connectivity.size()+product.seed_ids.size()+product.directions.size())*8;
  }
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_before");
#endif
  const double started=MPI_Wtime();
  check(catalyst_execute(params));
  // Previous storage stays alive until every consumer has taken this frame.
  if(retained) conduit_node_destroy(retained);
  retained=params;retained_products=std::move(products);
  std::printf("ASTR_INSITU_COMPACT_BRIDGE rank=%d step=%d final_geometry_bytes=%zu execute_inclusive=%.9g\n",
    rank,step,copy_bytes,MPI_Wtime()-started);
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_after");
#endif
  return 0;
} catch(const std::exception& error) {
  return allocation_failure("compact products",error.what(),MPI_Comm_f2c(fcomm));
} catch(...) {
  return allocation_failure("compact products","unknown exception",MPI_Comm_f2c(fcomm));
}

extern "C" int astr_insitu_mesh_finish()
try {
  if (!active) return 0;
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("finalize_before");
#endif
  auto end = conduit_node_create();
  check(catalyst_finalize(end));
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("finalize_after");
#endif
  conduit_node_destroy(end);
  conduit_node_destroy(retained);
  retained = nullptr;
  std::vector<double>().swap(coordinates);
  std::vector<double>().swap(primitive);
  std::vector<double>().swap(gradients);
  std::vector<double>().swap(means);
  std::vector<conduit_int64>().swap(plane_cells);
  std::vector<astr_insitu::CompactMesh>().swap(retained_products);
#ifdef ASTR_INSITU_DEVICE_RENDERING
  device_actors.clear();
  retained_device_products.clear();
  resident_pipeline.clear();
#endif
  check_mpi(MPI_Comm_free(&render_comm),"release render communicator",MPI_COMM_WORLD);
  active = false;
  return 0;
} catch (const std::exception& error) {
  return allocation_failure("finalize",error.what(),MPI_COMM_WORLD);
} catch (...) {
  return allocation_failure("finalize","unknown C++ exception",MPI_COMM_WORLD);
}
