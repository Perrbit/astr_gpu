/* Test-only LD_PRELOAD shim for OpenMPI's Fortran binding; never link into ASTR. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*allreduce_f)(const void *, void *, MPI_Fint *, MPI_Fint *,
                            MPI_Fint *, MPI_Fint *, MPI_Fint *);

void mpi_allreduce_(const void *send, void *recv, MPI_Fint *count,
                    MPI_Fint *type, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *err)
{
    static allreduce_f real_allreduce;
    static int injected;
    int values[5], rank;
    if (!real_allreduce) real_allreduce = (allreduce_f)dlsym(RTLD_NEXT, "mpi_allreduce_");
    if (!real_allreduce) abort();
    PMPI_Comm_rank(MPI_Comm_f2c(*comm), &rank);
    if (!injected && rank == 1 && *count == 5 &&
        MPI_Type_f2c(*type) == MPI_INTEGER && MPI_Op_f2c(*op) == MPI_MAX) {
        const int *original = send;
        for (int i = 0; i < 5; ++i) values[i] = original[i];
        values[0] = 99;
        injected = 1;
        fprintf(stderr, "AIR5_TEST_INJECT packed chemistry rank=1 status=99\n");
        real_allreduce(values, recv, count, type, op, comm, err);
    } else {
        real_allreduce(send, recv, count, type, op, comm, err);
    }
}
