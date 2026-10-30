#include <catalyst.h>
#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <exception>
#include <cmath>
#include "insitu_compact_mesh.h"
#include "insitu_product_dispatch.h"
#include "insitu_product_failure.h"
extern "C" int astr_insitu_runtime_product_count() __attribute__((weak));
extern "C" int astr_insitu_runtime_product_identity(int,char*,int,std::int64_t*,double*) __attribute__((weak));
extern "C" int astr_insitu_runtime_product_fields(int,int*,char*,int) __attribute__((weak));
extern "C" void astr_insitu_runtime_geometry_source(const int*,int,double*,char*,char*) __attribute__((weak));
#ifdef ASTR_INSITU_DEVICE_RENDERING
#include "insitu_device_vtk_mapper.h"
#include "insitu_streamline_seeds.h"
#include <vtkColorTransferFunction.h>
#include <vtkConduitSource.h>
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
using Real=astr_insitu::Real;
#ifdef ASTR_REAL32
constexpr auto set_external_real=conduit_node_set_path_external_float32_ptr;
#else
constexpr auto set_external_real=conduit_node_set_path_external_float64_ptr;
#endif
std::int64_t m12_color_mask=0;
std::array<Real,6> m12_color_ranges{};
bool active = false;
MPI_Comm render_comm = MPI_COMM_NULL;
conduit_node* retained = nullptr;
std::vector<Real> coordinates, primitive, gradients, means;
std::vector<conduit_int64> plane_cells;
std::vector<astr_insitu::CompactMesh> retained_products;
std::shared_ptr<astr_insitu::CompactMesh> retained_runtime_compact;
#ifdef ASTR_INSITU_DEVICE_RENDERING
std::map<std::string,vtkSmartPointer<vtkActor>> device_actors;
std::string resident_pipeline;
Json::Value preview_source;
conduit_node* detached_device_node=nullptr;
bool resident_inputs_detached=false;
bool resident_frame_open=false;
int resident_frame_step=0,resident_frame_groups=0;
double resident_frame_time=0.,resident_execute_seconds=0.;
std::string runtime_context,runtime_error;
int runtime_result=0;
#endif
Real empty_value = 0;
conduit_int64 empty_cell = 0;
int shape[3] = {0,0,0};
int selected_profile = 0;
bool native = false;
std::string output_directory, expected_uuid;
std::vector<astr_insitu::StreamlineView> streamline_candidates;
int allocation_failure(const char* stage, const char* detail, MPI_Comm comm)
{
  std::fprintf(stderr,"ASTR INSITU BRIDGE ERROR: %s: %s\n",stage,detail);
  MPI_Abort(comm,1);
  return 1;
}
}

extern "C" int astr_insitu_m12_color_range(int component,int store,Real* range) {
  if(component<0 || component>=3 || !range || (store!=0 && store!=1)) return -1;
  const std::int64_t bit=std::int64_t(1)<<component;
  if(store) {
    if(!std::isfinite(range[0]) || !std::isfinite(range[1]) || range[0]>=range[1]) return -1;
    std::copy_n(range,2,m12_color_ranges.begin()+2*component);m12_color_mask|=bit;
  } else {
    if(!(m12_color_mask&bit)) return 0;
    std::copy_n(m12_color_ranges.begin()+2*component,2,range);
  }
  return 1;
}

extern "C" int astr_insitu_m12_color_state(int restore,std::int64_t* mask,Real* ranges) {
  if(!mask || !ranges || (restore!=0 && restore!=1)) return 0;
  if(restore) {
    if(*mask<0 || *mask>7) return 0;
    for(int i=0;i<3;++i) {
      const Real a=ranges[2*i],b=ranges[2*i+1];
      if(!std::isfinite(a) || !std::isfinite(b) || ((*mask&(std::int64_t(1)<<i))?(a>=b):(a!=0. || b!=0.))) return 0;
    }
    m12_color_mask=*mask;std::copy_n(ranges,6,m12_color_ranges.begin());
  } else {
    *mask=m12_color_mask;std::copy(m12_color_ranges.begin(),m12_color_ranges.end(),ranges);
  }
  return 1;
}
namespace {
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

const std::vector<astr_insitu::StreamlineView>& astr_insitu::streamline_views() {
  return streamline_candidates;
}
bool astr_insitu::streamline_preview_active() {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  return !preview_source.isNull();
#else
  return false;
#endif
}
bool astr_insitu::retain_quantitative_product_fields() {
  if(!streamline_candidates.empty() || streamline_preview_active()) return true;
  if(const char* audit=std::getenv("ASTR_INSITU_RESIDENT_AUDIT"))
    if(std::string(audit)=="1") return true;
  for(const char* name:{"ASTR_INSITU_TEST_CURVE_Q_PREFIX","ASTR_INSITU_TEST_CURVE_TRACE_PREFIX",
                       "ASTR_INSITU_TEST_PLANE_PREFIX"})
    if(const char* prefix=std::getenv(name)) if(*prefix) return true;
  return false;
}

astr_insitu::RuntimeProductView astr_insitu::runtime_product_view(const char* id,const char* source,
    const char* definition,const char* directory,std::int64_t version,const char* labels,
    const int* integers,const double* reals) {
  RuntimeProductView view;
  view.id=id;view.source=source;view.definition=definition;view.directory=directory;view.version=version;
  for(int i=0;i<11;++i) {
    view.labels[i]=std::string(labels+32*i,32);
    while(!view.labels[i].empty() && view.labels[i].back()==' ') view.labels[i].pop_back();
  }
  std::copy_n(integers,7,view.integers.begin());std::copy_n(reals,31,view.reals.begin());
  if(view.labels[1]=="geometry") {
    if(!astr_insitu_runtime_product_count || !astr_insitu_runtime_product_identity ||
        !astr_insitu_runtime_product_fields || !astr_insitu_runtime_geometry_source)
      throw std::logic_error("Geometry export requires the native typed product registry");
    int selected=0;
    for(int i=1;i<=astr_insitu_runtime_product_count();++i) {
      std::int64_t state[5]{};double time=0.;char query=0;
      const int n=astr_insitu_runtime_product_identity(i,&query,0,state,&time);
      std::vector<char> name(n);
      astr_insitu_runtime_product_identity(i,name.data(),n,state,&time);
      if(view.id==name.data()) {selected=i;break;}
    }
    if(!selected) throw std::logic_error("Geometry product is absent from the native registry");
    int query=0;char blank=0;
    const int count=astr_insitu_runtime_product_fields(selected,&query,&blank,0);
    if(count<=0) throw std::logic_error("Geometry product has no declared native fields");
    std::vector<int> ids(count);std::vector<char> names(count*32),units(count*32);
    if(astr_insitu_runtime_product_fields(selected,ids.data(),names.data(),count)!=count)
      throw std::logic_error("Geometry field declaration changed during synchronous extraction");
    char system[16];
    astr_insitu_runtime_geometry_source(ids.data(),count,view.references.data(),system,units.data());
    auto text=[](const char* begin,int count) {
      std::string result(begin,count);
      while(!result.empty() && result.back()==' ') result.pop_back();
      return result;
    };
    view.unit_system=text(system,16);
    for(int i=0;i<count;++i) {
      view.fields.emplace_back(ids[i],text(names.data()+32*i,32));
      view.units.push_back(text(units.data()+32*i,32));
    }
  }
  return view;
}

extern "C" int astr_insitu_preview_source_set(const char* path,std::int64_t step,Real time,
    int fcomm) {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(active || !path || !*path || step<0 || !std::isfinite(time)) return 1;
  int ranks=0;
  if(MPI_Comm_size(MPI_Comm_f2c(fcomm),&ranks)!=MPI_SUCCESS) return 1;
  preview_source["checkpoint"]=path;
  preview_source["step"]=Json::Int64(step);preview_source["time"]=time;
  preview_source["ranks"]=ranks;
  return 0;
#else
  return 1;
#endif
}

extern "C" int astr_insitu_streamline_view_add(int index,const char* id,const char* velocity,
    const char* seeds,const char* color,const Real* values) {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(active || index!=static_cast<int>(streamline_candidates.size()) || index>=6 ||
      !id || !velocity || !seeds || !color || !values) return 1;
  astr_insitu::StreamlineView view;
  view.id=id;view.velocity=velocity;view.seeds=seeds;view.color=color;
  std::copy(values,values+19,view.values.begin());
  if(!std::all_of(view.values.begin(),view.values.end(),[](Real v){return std::isfinite(v);})) return 1;
  streamline_candidates.push_back(std::move(view));
  return 0;
#else
  return 1;
#endif
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
    int fcomm, int nx, int ny, int nz, int step, Real time,
    Real* xyz, Real* fields, Real* derived, int has_mean, Real* mean_fields, int profile)
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
    set_external_real(mesh, key.c_str(), xyz+d*count, count);
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
  auto field = [mesh,count](const char* name, Real* data) {
    const auto base = std::string("fields/")+name;
    conduit_node_set_path_char8_str(mesh, (base+"/association").c_str(), "vertex");
    conduit_node_set_path_char8_str(mesh, (base+"/topology").c_str(), "mesh");
    set_external_real(mesh, (base+"/values").c_str(), data, count);
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
    Real* mean_data=means.empty() ? &empty_value : means.data();
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
    Real* mean_data=means.empty() ? &empty_value : means.data();
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
      copied_values*sizeof(Real));
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
    int fcomm, int nx, int ny, int nz, int step, Real time,
    Real* xyz, Real* fields, Real* derived, int has_mean, Real* mean_fields)
{
  return mesh_execute(backend,script,fcomm,nx,ny,nz,step,time,xyz,fields,derived,has_mean,mean_fields,0);
}

extern "C" int astr_insitu_mesh_selected(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, Real time,
    Real* xyz, Real* velocity, Real* q, int profile)
{
  return mesh_execute(backend,script,fcomm,nx,ny,nz,step,time,xyz,velocity,q,0,nullptr,profile);
}

extern "C" int astr_insitu_mesh_wall_statistics(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, Real time,
    Real* xyz, Real* fields, Real* statistics, int profile)
{
  return mesh_execute(backend,script,fcomm,nx,ny,nz,step,time,xyz,fields,nullptr,2,statistics,profile);
}

#ifdef ASTR_INSITU_DEVICE_RENDERING
std::vector<double> astr_insitu::device_display_palette(const std::string& color_map) {
  if(color_map=="grayscale") {
    std::vector<double> table(4096*3);
    for(int i=0;i<4096;++i) for(int d=0;d<3;++d) table[3*i+d]=i/4095.;
    return table;
  }
  if(color_map!="diverging_blue_red") throw std::invalid_argument("Unsupported runtime display palette");
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

extern "C" void astr_insitu_release_runtime_actor(const char* name) {
  device_actors.erase(name);
}

namespace {
Json::Value runtime_catalog_snapshot() {
  if(!astr_insitu_runtime_product_count || !astr_insitu_runtime_product_identity)
    throw std::logic_error("The runtime renderer requires the native product registry");
  Json::Value catalog(Json::objectValue);
  for(int i=1;i<=astr_insitu_runtime_product_count();++i) {
    std::int64_t metadata[5]{};char query=0;double clock=0.;
    const int length=astr_insitu_runtime_product_identity(i,&query,0,metadata,&clock);
    if(length<1) throw std::logic_error("Invalid native runtime product identity");
    if(!metadata[2] || (metadata[1]!=1 && metadata[1]!=3)) continue;
    std::vector<char> id(length);
    if(astr_insitu_runtime_product_identity(i,id.data(),length,metadata,&clock)!=length)
      throw std::logic_error("Runtime product identity changed during synchronous rendering");
    auto& record=catalog[id.data()];
    record["version"]=Json::Int64(metadata[0]);record["applied_step"]=Json::Int64(metadata[4]);record["applied_time"]=clock;
  }
  return catalog;
}

std::string runtime_descriptor(const astr_insitu::RuntimeProductView* view,bool host=false) {
  Json::Value data;data["mode"]=view?"product":"catalog";data["catalog"]=runtime_catalog_snapshot();
  if(view) {
    data["processing_backend"]=host?"host":"device";
    data["id"]=view->id;data["source"]=view->source;data["version"]=Json::Int64(view->version);
    data["definition"]=view->definition;data["directory"]=view->directory;
    const auto& applied=data["catalog"][view->id];
    data["applied_step"]=applied["applied_step"];data["applied_time"]=applied["applied_time"];
    for(const auto& label:view->labels) data["labels"].append(label);
    for(const auto word:view->integers) data["integers"].append(word);
    for(const auto real:view->reals) data["reals"].append(real);
    if(view->labels[1]=="geometry") {
      for(std::size_t i=0;i<view->fields.size();++i) {
        data["fields"].append(view->fields[i].second);data["units"][view->fields[i].second]=view->units[i];
        data["point_fields"].append(view->fields[i].second);
      }
      data["unit_system"]=view->unit_system;
      for(double value:view->references) data["reference_state"].append(value);
      if(view->labels[0]=="streamlines") {
        if(host) {
          data["point_fields"].append("IntegrationTime");
          data["cell_fields"].append("ReasonForTermination");data["units"]["ReasonForTermination"]="1";
          data["units"]["IntegrationTime"]=view->unit_system=="si"?"s":"1";
        } else {
        for(const char* name:{"accepted_length","termination_status","untravelled_arc_length"}) data["point_fields"].append(name);
        data["units"]["termination_status"]="1";
        data["units"]["accepted_length"]=view->unit_system=="si"?"m":"1";
        data["units"]["untravelled_arc_length"]=data["units"]["accepted_length"];
        }
        for(const char* name:{"SeedIds","IntegrationDirection"}) {data["cell_fields"].append(name);data["units"][name]="1";}
      }
    }
    if(view->statistics_provider) {
      auto& statistics=data["statistics"];
      const bool wall=view->labels[0]=="wall_curve";
      const bool samples=!wall && view->statistics_provider>2;
      statistics["provider"]=wall?"wall-time":(samples?"solver-samples":"velocity-time");
      auto& window=statistics[samples?"sample_time_range":"requested_window"];
      window.append(view->statistics[0]);window.append(view->statistics[1]);
      if(samples) statistics["sample_count"]=Json::Int64(view->samples);
      else statistics["covered_duration"]=view->statistics[2];
      if(wall) statistics["accumulated_endpoints"]=Json::Int64(view->samples);
      statistics["weight"]=samples?"equal_sample_count":(view->source=="favre"?"density_time_integral":"time_integral");
      statistics["sampling_phase"]=samples?"existing_pre_step_RK1":"completed_step_clipped_time";
    }
  }
  Json::StreamWriterBuilder writer;writer["indentation"]="";
  return Json::writeString(writer,data);
}
}

extern "C" const char* astr_insitu_runtime_context() {return runtime_context.c_str();}
extern "C" void astr_insitu_runtime_failure(const char* reason) {
  runtime_result=1;runtime_error=reason;
}
extern "C" void astr_insitu_runtime_error(char* buffer,int capacity) {
  if(capacity>0) std::snprintf(buffer,capacity,"%s",runtime_error.c_str());
}

extern "C" int astr_insitu_runtime_catalog_cleanup(int fcomm,int step,Real time) try {
  if(!active) return 0;
  if(selected_profile!=8 && !(selected_profile==7 && resident_pipeline=="compatible"))
    return allocation_failure("runtime catalog","another renderer owns the session",MPI_Comm_f2c(fcomm));
  runtime_context=runtime_descriptor(nullptr);runtime_result=0;runtime_error.clear();
  auto params=conduit_node_create();
  conduit_node_set_path_int64(params,"catalyst/state/timestep",step);
  conduit_node_set_path_double(params,"catalyst/state/time",time);
  check(catalyst_execute(params));conduit_node_destroy(params);runtime_context.clear();
  if(const char* trace=std::getenv("ASTR_INSITU_MEMORY_TRACE")) if(std::string(trace)=="1")
    std::printf("ASTR_RUNTIME_CATALOG_CLEANUP step=%d actor_keys=%zu\n",step,device_actors.size());
  return runtime_result;
} catch(const std::exception& error) {
  return allocation_failure("runtime catalog cleanup",error.what(),MPI_Comm_f2c(fcomm));
}

extern "C" void astr_insitu_retire_resident_source(std::uintptr_t address) {
  auto source=vtkConduitSource::SafeDownCast(reinterpret_cast<vtkObjectBase*>(address));
  if(!source || !detached_device_node) {
    allocation_failure("resident retirement","invalid Conduit producer",render_comm);
    return;
  }
  // SetNode borrows its node; retain a real empty node until Catalyst finalizes.
  source->SetNode(detached_device_node);
  source->SetGlobalFieldsNode(nullptr);
  source->SetAssemblyNode(nullptr);
  if(auto output=source->GetOutputDataObject(0)) output->Initialize();
}

extern "C" void astr_insitu_retire_resident_views() {
  for(auto& item:device_actors)
    static_cast<astr_insitu::DirectDeviceMapper*>(item.second->GetMapper())->clear_view();
  resident_inputs_detached=true;
}

int astr_insitu::render_resident_products(const char* pipeline,const char* backend,const char* script,int fcomm,
    int step,Real time,const char* profile,std::vector<DeviceMesh> products,bool covered,bool complete,
    const RuntimeProductView* runtime)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  const std::string route=pipeline?pipeline:"";
  if(runtime) {runtime_context=runtime_descriptor(runtime);runtime_result=0;runtime_error.clear();}
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
    if(!streamline_candidates.empty()) {
      Json::Value candidates(Json::arrayValue);
      for(const auto& view:streamline_candidates) {
        Json::Value item;
        item["id"]=view.id;item["velocity"]=view.velocity;item["seeds"]=view.seeds;item["color"]=view.color;
        for(double value:view.values) item["values"].append(value);
        for(const auto& site:astr_insitu::tgv_streamline_seeds(view.seeds)) {
          Json::Value point(Json::arrayValue);
          for(double value:site) point.append(value);
          item["seed_sites"].append(point);
        }
        candidates.append(item);
      }
      Json::StreamWriterBuilder writer;writer["indentation"]="";
      Json::Value payload;payload["candidates"]=candidates;payload["source"]=preview_source;
      const auto text=Json::writeString(writer,payload);
      conduit_node_set_char8_str(conduit_node_append(args),text.c_str());
    }
    conduit_node_set_path_int64(init,"catalyst/mpi_comm",MPI_Comm_c2f(render_comm));
    check(catalyst_initialize(init));conduit_node_destroy(init);
    detached_device_node=conduit_node_create();
    selected_profile=8;active=true;resident_pipeline=route;
    astr_insitu_resource_check("initialized");
  }
  if(!resident_frame_open) {
    resident_frame_open=true;resident_frame_step=step;resident_frame_time=time;
    resident_frame_groups=0;resident_execute_seconds=0.;
    astr_insitu_resource_check("frame_before");
  } else if(resident_frame_step!=step || resident_frame_time!=time) {
    return allocation_failure("resident products","unfinished product group from another frame",comm);
  }
  ++resident_frame_groups;
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
    conduit_node_set_path_int64(mesh,"state/fields/frame_complete",complete);
    conduit_node_set_path_int64(mesh,"state/fields/group_member",1);
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
      else set_external_real(mesh,key.c_str(),&empty_value,0);
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
    if(route=="direct-device" && (!runtime || runtime->labels[0]!="wall_curve")) {
      auto& actor=device_actors[product.actor_key.empty()?product.name:product.actor_key];
      if(!actor) {
        actor=vtkSmartPointer<vtkActor>::New();
        vtkNew<astr_insitu::DirectDeviceMapper> mapper;
        actor->SetMapper(mapper);
      }
      static_cast<astr_insitu::DirectDeviceMapper*>(actor->GetMapper())->set_view(view);
    }
  }
  resident_inputs_detached=false;
  const double start=MPI_Wtime();
  nvtxRangePushA("ASTR_IS8_RESIDENT_RENDER");
  check(catalyst_execute(params));
  nvtxRangePop();
  if(!resident_inputs_detached)
    return allocation_failure("resident retirement","pipeline did not detach external arrays",comm);
  nvtxRangePushA("ASTR_IS8_RESIDENT_RETIRE");
  conduit_node_destroy(params);
  products.clear();
  nvtxRangePop();
  resident_execute_seconds+=MPI_Wtime()-start;
  if(const char* trace=std::getenv("ASTR_INSITU_MEMORY_TRACE")) if(std::string(trace)=="1") {
    std::printf("ASTR_INSITU_RESIDENT_GROUP rank=%d step=%d group=%d complete=%d arrays_detached=1 owners_retained=0\n",
      rank,step,resident_frame_groups,int(complete));
    if(complete) std::printf("ASTR_INSITU_RESIDENT_RETIRE rank=%d step=%d arrays_detached=1 owners_retained=0\n",rank,step);
  }
  if(complete) {
    std::printf("ASTR_INSITU_RESIDENT_BRIDGE rank=%d step=%d pipeline=%s geometry_host_bytes=0 execute_inclusive=%.9g\n",
      rank,step,route.c_str(),resident_execute_seconds);
    astr_insitu_resource_check("frame_after");
    resident_frame_open=false;
  }
  if(runtime) {
    const bool failed=consensus_product_failure(comm,runtime_error);
    runtime_context.clear();return failed?1:0;
  }
  return 0;
} catch(const std::exception& error) {
  return allocation_failure("resident products",error.what(),MPI_Comm_f2c(fcomm));
} catch(...) {
  return allocation_failure("resident products","unknown exception",MPI_Comm_f2c(fcomm));
}
#endif

std::shared_ptr<astr_insitu::CompactMesh> astr_insitu::retained_runtime_geometry() {
  return retained_runtime_compact;
}

static void initialize_compact_renderer(const char* backend,const char* script,int fcomm,const char* profile,bool runtime) {
  const auto comm=MPI_Comm_f2c(fcomm);
  if(!active) {
    check_mpi(MPI_Comm_dup(comm,&render_comm),"create compact render communicator",comm);
    auto init=conduit_node_create();
    conduit_node_set_path_char8_str(init,"catalyst_load/implementation","paraview");
    conduit_node_set_path_char8_str(init,"catalyst_load/search_paths/astr",backend);
    conduit_node_set_path_char8_str(init,"catalyst/scripts/probe/filename",script);
    auto args=conduit_node_fetch(init,"catalyst/scripts/probe/args");
    for(const char* value:{output_directory.c_str(),expected_uuid.c_str(),profile,runtime?"compatible":"device"})
      conduit_node_set_char8_str(conduit_node_append(args),value);
    conduit_node_set_path_int64(init,"catalyst/mpi_comm",MPI_Comm_c2f(render_comm));
    check(catalyst_initialize(init));conduit_node_destroy(init);
    selected_profile=7;active=true;
#ifdef ASTR_INSITU_DEVICE_RENDERING
    if(runtime) {resident_pipeline="compatible";detached_device_node=conduit_node_create();}
#endif
#ifdef ASTR_CUDA_RENDERER
    astr_insitu_resource_check("initialized");
#endif
  }
}

#ifdef ASTR_INSITU_DEVICE_RENDERING
extern "C" int astr_insitu_runtime_host_begin(const char* backend,const char* script,int fcomm,int step,Real time) try {
  if(!native || !backend || !script || (active && (selected_profile!=7 || resident_pipeline!="compatible")))
    return allocation_failure("host products","invalid compatible renderer session",MPI_Comm_f2c(fcomm));
  if(!active) {
    initialize_compact_renderer(backend,script,fcomm,"runtime_catalog",true);
  }
  return 0;
} catch(const std::exception& error) {
  return allocation_failure("host product initialization",error.what(),MPI_Comm_f2c(fcomm));
}
#endif

int astr_insitu::render_compact_products(const char* backend,const char* script,int fcomm,
    int step,Real time,const char* profile,bool covered,Real duration,
    Real window_start,Real window_end,std::vector<CompactMesh> products,
    const RuntimeProductView* runtime,std::shared_ptr<CompactMesh> frozen)
try {
  const auto comm=MPI_Comm_f2c(fcomm);
  if(!native || !backend || !script || !profile || (products.empty() && !frozen) || (active && selected_profile!=7))
    return allocation_failure("device products","invalid compact renderer session",comm);
#ifdef ASTR_INSITU_DEVICE_RENDERING
  const bool host=frozen && frozen->controls.count("host_source") && frozen->controls.at("host_source")!=0.;
  if(runtime) {runtime_context=runtime_descriptor(runtime,host);runtime_result=0;runtime_error.clear();}
#endif
  static_assert(sizeof(std::int64_t)==sizeof(conduit_int64),"Conduit geometry requires int64 connectivity");
  initialize_compact_renderer(backend,script,fcomm,profile,runtime!=nullptr);
  auto params=conduit_node_create();
  conduit_node_set_path_int64(params,"catalyst/state/timestep",step);
  conduit_node_set_path_double(params,"catalyst/state/time",time);
  int rank=0;check_mpi(MPI_Comm_rank(comm,&rank),"compact domain rank",comm);
  std::size_t copy_bytes=0;
  std::vector<CompactMesh*> selected;
  if(frozen) selected.push_back(frozen.get());
  else for(auto& product:products) selected.push_back(&product);
  for(auto* pointer:selected) {
    auto& product=*pointer;
    const auto nodes=product.coordinates[0].size();
    if(product.coordinates[1].size()!=nodes || product.coordinates[2].size()!=nodes)
      return allocation_failure("device products","coordinate component sizes differ",comm);
    const int arity=product.shape=="tri"?3:(product.shape=="quad"?4:(product.shape=="line"?2:0));
    if(!arity || product.connectivity.size()%arity)
      return allocation_failure("device products","invalid final connectivity",comm);
    const auto cells=product.connectivity.size()/arity;
    if(!runtime) for(auto index:product.connectivity) if(index<0 || std::uint64_t(index)>=nodes)
      return allocation_failure("device products","final vertex index out of range",comm);
    const auto base="catalyst/channels/"+product.name;
    conduit_node_set_path_char8_str(params,(base+"/type").c_str(),"mesh");
    auto mesh=conduit_node_fetch(params,(base+"/data").c_str());
    conduit_node_set_path_int64(mesh,"state/domain_id",rank);
    conduit_node_set_path_int64(mesh,"state/fields/device_products",1);
    for(const auto& control:product.controls)
      conduit_node_set_path_double(mesh,("state/fields/"+control.first).c_str(),control.second);
    conduit_node_set_path_int64(mesh,"state/fields/mean_covered",covered);
    conduit_node_set_path_double(mesh,"state/fields/statistics_duration",duration);
    conduit_node_set_path_double(mesh,"state/fields/statistics_window_start",window_start);
    conduit_node_set_path_double(mesh,"state/fields/statistics_window_end",window_end);
    conduit_node_set_path_char8_str(mesh,"coordsets/coords/type","explicit");
    const char* axes[]={"x","y","z"};
    for(int d=0;d<3;++d) {
      const auto path=std::string("coordsets/coords/values/")+axes[d];
      set_external_real(mesh,path.c_str(),
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
      set_external_real(mesh,(path+"/values").c_str(),
        nodes?field.second.data():&empty_value,nodes);
      copy_bytes+=field.second.size()*sizeof(Real);
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
    if(product.shape=="line" && (!runtime || runtime->labels[1]=="geometry")) {
      cell_field("SeedIds",product.seed_ids);cell_field("IntegrationDirection",product.directions);
    }
    for(auto& field:product.cell_fields) {
      if(cell_field(field.first.c_str(),field.second)) return 1;
      copy_bytes+=field.second.size()*sizeof(std::int64_t);
    }
    copy_bytes+=nodes*3*sizeof(Real)+
      (product.connectivity.size()+product.seed_ids.size()+product.directions.size())*sizeof(std::int64_t);
  }
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_before");
#endif
  const double started=MPI_Wtime();
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(runtime) resident_inputs_detached=false;
#endif
  check(catalyst_execute(params));
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(runtime) {
    if(!resident_inputs_detached)
      return allocation_failure("compatible retirement","pipeline did not detach external arrays",comm);
    conduit_node_destroy(params);
  } else
#endif
  {
    // The legacy consumer does not implement explicit per-group retirement.
    if(retained) conduit_node_destroy(retained);
    retained=params;retained_products=std::move(products);retained_runtime_compact=std::move(frozen);
  }
  std::printf("ASTR_INSITU_COMPACT_BRIDGE rank=%d step=%d final_geometry_bytes=%zu execute_inclusive=%.9g\n",
    rank,step,copy_bytes,MPI_Wtime()-started);
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_after");
#endif
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(runtime) {
    const bool failed=consensus_product_failure(comm,runtime_error);
    runtime_context.clear();return failed?1:0;
  }
#endif
  return 0;
} catch(const std::exception& error) {
  return allocation_failure("compact products",error.what(),MPI_Comm_f2c(fcomm));
} catch(...) {
  return allocation_failure("compact products","unknown exception",MPI_Comm_f2c(fcomm));
}

extern "C" int astr_insitu_mesh_finish()
try {
  if (!active) {m12_color_mask=0;m12_color_ranges.fill(0.);return 0;}
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(resident_frame_open) return allocation_failure("finalize","unfinished resident product frame",render_comm);
#endif
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("finalize_before");
#endif
  auto end = conduit_node_create();
  check(catalyst_finalize(end));
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("finalize_after");
#endif
  conduit_node_destroy(end);
  if(retained) conduit_node_destroy(retained);
  retained = nullptr;
  std::vector<Real>().swap(coordinates);
  std::vector<Real>().swap(primitive);
  std::vector<Real>().swap(gradients);
  std::vector<Real>().swap(means);
  std::vector<conduit_int64>().swap(plane_cells);
  std::vector<astr_insitu::CompactMesh>().swap(retained_products);
  retained_runtime_compact.reset();
#ifdef ASTR_INSITU_DEVICE_RENDERING
  device_actors.clear();
  if(detached_device_node) conduit_node_destroy(detached_device_node);
  detached_device_node=nullptr;
  streamline_candidates.clear();
  resident_pipeline.clear();
#endif
  check_mpi(MPI_Comm_free(&render_comm),"release render communicator",MPI_COMM_WORLD);
  active = false;
  m12_color_mask=0;m12_color_ranges.fill(0.);
  return 0;
} catch (const std::exception& error) {
  return allocation_failure("finalize",error.what(),MPI_COMM_WORLD);
} catch (...) {
  return allocation_failure("finalize","unknown C++ exception",MPI_COMM_WORLD);
}
