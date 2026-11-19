#include <mpi.h>
#include <stdio.h>
#include <string.h>

static int probe_messages(const char *mode, int rank, int ranks)
{
    const int barrier = strcmp(mode, "barrier") == 0;
    const int null_zero = strcmp(mode, "null-zero") == 0;
    const int valid_zero = strcmp(mode, "valid-zero") == 0;
    const int host_byte = strcmp(mode, "host-byte") == 0;
    const int iterations = 32;
    if (ranks != 2 || !(barrier || null_zero || valid_zero || host_byte)) return 7;

    fprintf(stderr, "MPI_MESSAGE_PROBE mode=%s rank=%d phase=begin\n", mode, rank);
    fflush(stderr);
    for (int i = 0; i < iterations; ++i) {
        if (barrier) {
            if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) return 8;
        } else {
            const int peer = 1 - rank;
            const int count = host_byte ? 1 : 0;
            unsigned char send = (unsigned char)(rank * iterations + i);
            unsigned char recv = 255;
            MPI_Status status;
            if (MPI_Sendrecv(null_zero ? NULL : &send, count, MPI_BYTE, peer, 11,
                             null_zero ? NULL : &recv, count, MPI_BYTE, peer, 11,
                             MPI_COMM_WORLD, &status) != MPI_SUCCESS) return 9;
            int received;
            if (MPI_Get_count(&status, MPI_BYTE, &received) != MPI_SUCCESS || received != count ||
                status.MPI_SOURCE != peer || status.MPI_TAG != 11) return 10;
            if ((host_byte && recv != (unsigned char)(peer * iterations + i)) ||
                (!host_byte && recv != 255)) return 11;
        }
    }
    fprintf(stderr, "MPI_MESSAGE_PROBE mode=%s rank=%d phase=end iterations=%d verified=1\n",
            mode, rank, iterations);
    fflush(stderr);
    return 0;
}

int main(int argc, char **argv)
{
    int rank, ranks, total;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) return 2;
    if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS) return 3;
    if (argc != 1) {
        int status = argc == 2 ? probe_messages(argv[1], rank, ranks) : 7;
        if (status != 0) {
            fprintf(stderr, "MPI_MESSAGE_PROBE rank=%d failure=%d\n", rank, status);
            MPI_Abort(MPI_COMM_WORLD, status);
            return status;
        }
        return MPI_Finalize() == MPI_SUCCESS ? 0 : 6;
    }
    if (MPI_Allreduce(&rank, &total, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return 4;
    if (total != ranks * (ranks - 1) / 2) return 5;
    if (rank == 0) printf("MPI_ONLY_CONTEXT_PROBE ranks=%d sum=%d\n", ranks, total);
    return MPI_Finalize() == MPI_SUCCESS ? 0 : 6;
}
