#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int rank, ranks, total;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) return 2;
    if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return 3;
    if (MPI_Allreduce(&rank, &total, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return 4;
    if (total != ranks * (ranks - 1) / 2) return 5;
    if (rank == 0) printf("MPI_ONLY_CONTEXT_PROBE ranks=%d sum=%d\n", ranks, total);
    return MPI_Finalize() == MPI_SUCCESS ? 0 : 6;
}
