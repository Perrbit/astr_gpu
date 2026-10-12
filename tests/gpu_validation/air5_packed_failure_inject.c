/* Test-only LD_PRELOAD shim for OpenMPI's Fortran binding; never link into ASTR. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*allreduce_f)(const void *, void *, MPI_Fint *, MPI_Fint *,
                            MPI_Fint *, MPI_Fint *, MPI_Fint *);

void mpi_allreduce_(const void *send, void *recv, MPI_Fint *count,
                    MPI_Fint *type, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *err)
{
    static allreduce_f real_allreduce;
    static int injected;
    int values[5], rank;
    const char *begin = getenv("AIR5_TEST_HALF_BEGIN");
    const char *end = getenv("AIR5_TEST_HALF_END");
    int match = *count == 5;
    if (begin && end) {
        uintptr_t caller = (uintptr_t)__builtin_return_address(0);
        if (getenv("AIR5_TEST_HALF_PIE")) {
            Dl_info image;
            if (!dladdr((void *)caller, &image)) abort();
            caller -= (uintptr_t)image.dli_fbase;
        }
        match = (*count == 1 || *count == 5) &&
            caller >= strtoull(begin, NULL, 16) && caller < strtoull(end, NULL, 16);
    }
    if (!real_allreduce) real_allreduce = (allreduce_f)dlsym(RTLD_NEXT, "mpi_allreduce_");
    if (!real_allreduce) abort();
    PMPI_Comm_rank(MPI_Comm_f2c(*comm), &rank);
    if (!injected && rank == 1 && match &&
        MPI_Type_f2c(*type) == MPI_INTEGER && MPI_Op_f2c(*op) == MPI_MAX) {
        const int *original = send;
        for (int i = 0; i < *count; ++i) values[i] = original[i];
        values[0] = 99;
        injected = 1;
        fprintf(stderr, "AIR5_TEST_INJECT chemistry rank=1 count=%d status=99\n", (int)*count);
        real_allreduce(values, recv, count, type, op, comm, err);
    } else {
        real_allreduce(send, recv, count, type, op, comm, err);
    }
}
