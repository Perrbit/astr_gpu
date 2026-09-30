#include <catalyst.h>
#include <mpi.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
int collective_bad(int local, MPI_Comm comm)
{
  int result = 0;
  if (MPI_Allreduce(&local, &result, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS)
    MPI_Abort(comm, 1);
  return result;
}
}

extern "C" int astr_catalyst_check(const char* path, int fortran_comm)
{
  static_assert(sizeof(MPI_Fint) == sizeof(int), "Unsupported MPI Fortran handle size");
  MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_comm));
  const char* prefer_env = std::getenv("CATALYST_IMPLEMENTATION_PREFER_ENV");
  if (collective_bad(prefer_env && *prefer_env ? 1 : 0, comm)) {
    std::fprintf(stderr, "ASTR INSITU: unset CATALYST_IMPLEMENTATION_PREFER_ENV; use the configuration path\n");
    return 1;
  }
  const std::string library = std::string(path) + "/libcatalyst-paraview.so";
  // Resolve dependencies on every rank before entering the backend's MPI initialization.
  void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) std::fprintf(stderr, "ASTR INSITU dlopen: %s\n", dlerror());
  if (collective_bad(handle ? 0 : 1, comm)) {
    if (handle) dlclose(handle);
    return 1;
  }
  auto params = conduit_node_create();
  conduit_node_set_path_char8_str(params, "catalyst_load/implementation", "paraview");
  conduit_node_set_path_char8_str(params, "catalyst_load/search_paths/astr", path);
  conduit_node_set_path_int64(params, "catalyst/mpi_comm", fortran_comm);
  int bad = collective_bad(catalyst_initialize(params) == catalyst_status_ok ? 0 : 1, comm);
  conduit_node_destroy(params);
  if (bad) return 2; // Caller aborts collectively; do not finalize partially initialized ranks.

  auto about = conduit_node_create();
  bad = collective_bad(catalyst_about(about) == catalyst_status_ok ? 0 : 1, comm);
  if (!bad) {
    const char* implementation = conduit_node_fetch_path_as_char8_str(about, "catalyst/implementation");
    bad = collective_bad(!implementation || std::strcmp(implementation, "paraview") != 0, comm);
  }
  conduit_node_destroy(about);
  auto end = conduit_node_create();
  int finalize_bad = collective_bad(catalyst_finalize(end) == catalyst_status_ok ? 0 : 1, comm);
  conduit_node_destroy(end);
  dlclose(handle);
  return bad || finalize_bad ? 3 : 0;
}
