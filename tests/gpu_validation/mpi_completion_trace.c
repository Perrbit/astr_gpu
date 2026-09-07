#include <nvtx3/nvToolsExt.h>

void astr_trace_mpi_completion(int completed)
{
    nvtxMarkA(completed ? "ASTR_MPI_TESTALL_COMPLETE" : "ASTR_MPI_TESTALL_PENDING");
}
