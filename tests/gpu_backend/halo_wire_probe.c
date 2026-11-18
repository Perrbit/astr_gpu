#include <mpi.h>
#include <stdint.h>

static int64_t counters[5];

static void record(int slot, int count, MPI_Datatype datatype, int peer)
{
#ifdef ASTR_REAL32
    const MPI_Datatype expected=MPI_FLOAT;
#else
    const MPI_Datatype expected=MPI_DOUBLE;
#endif
    int bytes=0;
    PMPI_Type_size(datatype,&bytes);
    ++counters[slot];
    if(peer!=MPI_PROC_NULL) counters[slot+2]+=(int64_t)count*bytes;
    if(datatype!=expected) ++counters[4];
}

int MPI_Isend(const void *buffer,int count,MPI_Datatype datatype,int peer,int tag,MPI_Comm comm,MPI_Request *request)
{
    record(0,count,datatype,peer);
    return PMPI_Isend(buffer,count,datatype,peer,tag,comm,request);
}

int MPI_Irecv(void *buffer,int count,MPI_Datatype datatype,int peer,int tag,MPI_Comm comm,MPI_Request *request)
{
    record(1,count,datatype,peer);
    return PMPI_Irecv(buffer,count,datatype,peer,tag,comm,request);
}

void astr_halo_wire_counts(int64_t values[5])
{
    for(int i=0;i<5;++i) values[i]=counters[i];
}
