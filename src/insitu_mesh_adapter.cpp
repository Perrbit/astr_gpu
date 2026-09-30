#include <catalyst.h>
#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <exception>
#ifdef ASTR_CUDA_RENDERER
extern "C" int astr_insitu_map_current_cuda(int*,char*,int,char*,int);
extern "C" int astr_insitu_resource_check(const char*);
#endif

namespace {
bool active = false;
MPI_Comm render_comm = MPI_COMM_NULL;
conduit_node* retained = nullptr;
std::vector<double> coordinates, primitive, gradients, means;
int shape[3] = {0,0,0};
bool native = false;
std::string output_directory, expected_uuid;
int allocation_failure(const char* stage, const char* detail, MPI_Comm comm)
{
  std::fprintf(stderr,"ASTR INSITU BRIDGE ERROR: %s: %s\n",stage,detail);
  MPI_Abort(comm,1);
  return 1;
}
void check(catalyst_status status)
{
  int bad = status != catalyst_status_ok, any = 0;
  MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, render_comm);
  if (any) MPI_Abort(render_comm, 1);
}
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
  MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
  if (any) { std::fprintf(stderr,"ASTR native render mapping: %s\n",message); return 1; }
  const std::string ordinal=std::to_string(index);
  bad = setenv("VTK_EGL_DEVICE_INDEX",ordinal.c_str(),1) != 0;
  bad |= setenv("VTK_DEFAULT_OPENGL_WINDOW","vtkEGLRenderWindow",1) != 0;
  MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
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
extern "C" int astr_insitu_mesh_probe(const char* backend, const char* script,
    int fcomm, int nx, int ny, int nz, int step, double time,
    double* xyz, double* fields, double* derived, int has_mean, double* mean_fields)
try {
  MPI_Comm comm = MPI_Comm_f2c(fcomm);
  if (!active) {
  MPI_Comm_dup(comm, &render_comm);
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
  }
  conduit_node_set_path_int64(init, "catalyst/mpi_comm", MPI_Comm_c2f(render_comm));
  check(catalyst_initialize(init));
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("initialized");
#endif
  conduit_node_destroy(init);
  retained = conduit_node_create();
  shape[0]=nx; shape[1]=ny; shape[2]=nz;
  coordinates.resize(long(nx)*ny*nz*3);
  primitive.resize(long(nx)*ny*nz*11);
  gradients.resize(long(nx)*ny*nz*14);
  active = true;
  }
  if (shape[0]!=nx || shape[1]!=ny || shape[2]!=nz) MPI_Abort(comm, 1);
  std::copy(xyz, xyz+coordinates.size(), coordinates.begin());
  std::copy(fields, fields+primitive.size(), primitive.begin());
  std::copy(derived, derived+gradients.size(), gradients.begin());
  xyz=coordinates.data(); fields=primitive.data(); derived=gradients.data();
  auto params = retained;
  conduit_node_set_path_int64(params, "catalyst/state/timestep", step);
  conduit_node_set_path_double(params, "catalyst/state/time", time);
  conduit_node_set_path_char8_str(params, "catalyst/channels/grid/type", "mesh");
  auto mesh = conduit_node_fetch(params, "catalyst/channels/grid/data");
  int rank;
  MPI_Comm_rank(comm, &rank);
  conduit_node_set_path_int64(mesh, "state/domain_id", rank);
  conduit_node_set_path_char8_str(mesh, "coordsets/coords/type", "explicit");
  const long count = long(nx)*ny*nz;
  const char* axes[] = {"x", "y", "z"};
  for (int d=0; d<3; ++d) {
    const auto key = std::string("coordsets/coords/values/")+axes[d];
    conduit_node_set_path_external_float64_ptr(mesh, key.c_str(), xyz+d*count, count);
  }
  conduit_node_set_path_char8_str(mesh, "topologies/mesh/type", "structured");
  conduit_node_set_path_char8_str(mesh, "topologies/mesh/coordset", "coords");
  conduit_node_set_path_int64(mesh, "topologies/mesh/elements/dims/i", nx-1);
  conduit_node_set_path_int64(mesh, "topologies/mesh/elements/dims/j", ny-1);
  conduit_node_set_path_int64(mesh, "topologies/mesh/elements/dims/k", nz-1);
  auto field = [mesh,count](const char* name, double* data) {
    const auto base = std::string("fields/")+name;
    conduit_node_set_path_char8_str(mesh, (base+"/association").c_str(), "vertex");
    conduit_node_set_path_char8_str(mesh, (base+"/topology").c_str(), "mesh");
    conduit_node_set_path_external_float64_ptr(mesh, (base+"/values").c_str(), data, count);
  };
  const char* names[] = {"q1","q2","q3","q4","q5","rho","u","v","w","pressure","temperature"};
  for (int c=0; c<11; ++c) field(names[c], fields+c*count);
  const char* diagnostics[] = {"du_dx","dv_dx","dw_dx","du_dy","dv_dy","dw_dy",
      "du_dz","dv_dz","dw_dz","Q_rs","divergence","omega_x","omega_y","omega_z"};
  for (int c=0; c<14; ++c) field(diagnostics[c], derived+c*count);
  const char* mean_names[] = {"mean_u_reynolds","mean_v_reynolds","mean_w_reynolds",
    "mean_u_favre","mean_v_favre","mean_w_favre","statistics_duration"};
  if (has_mean) {
    means.assign(mean_fields,mean_fields+7*count);
    for (int c=0;c<7;++c) field(mean_names[c],means.data()+c*count);
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
  check(catalyst_execute(params));
#ifdef ASTR_CUDA_RENDERER
  astr_insitu_resource_check("frame_after");
#endif
  return 0;
} catch (const std::exception& error) {
  return allocation_failure("execute",error.what(),MPI_Comm_f2c(fcomm));
} catch (...) {
  return allocation_failure("execute","unknown C++ exception",MPI_Comm_f2c(fcomm));
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
  MPI_Comm_free(&render_comm);
  active = false;
  return 0;
} catch (const std::exception& error) {
  return allocation_failure("finalize",error.what(),MPI_COMM_WORLD);
} catch (...) {
  return allocation_failure("finalize","unknown C++ exception",MPI_COMM_WORLD);
}
