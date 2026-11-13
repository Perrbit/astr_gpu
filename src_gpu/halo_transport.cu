#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <new>
#include <climits>
#include "device_state_internal.cuh"
#include "device_halo.h"
#include "halo_transport.h"

namespace astr {
struct HaloTransport {
    MPI_Comm communicator=MPI_COMM_NULL;
    int lower[3]{},upper[3]{},sizes[3]{},max_components=0;
    double *send_device[2]{},*recv_device[2]{},*send_host[2]{},*recv_host[2]{};
    MPI_Request requests[4]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
};

int release_transport(HaloTransport *transport)
{
    if (!transport) return 0;
    int status=0;
    for (int side=0;side<2;++side) {
        void *device[]={transport->send_device[side],transport->recv_device[side]};
        void *host[]={transport->send_host[side],transport->recv_host[side]};
        for (void *p:device) if (p) {
            const int freed=static_cast<int>(cudaFree(p));
            if (status==0) status=freed;
        }
        for (void *p:host) if (p) {
            const int freed=static_cast<int>(cudaFreeHost(p));
            if (status==0) status=freed;
        }
    }
    if (transport->communicator!=MPI_COMM_NULL) {
        const int freed=MPI_Comm_free(&transport->communicator);
        if (status==0) status=freed;
    }
    delete transport;
    return status;
}
}

namespace {
constexpr int threads=astr::block_threads;

__device__ void packet_point(size_t id,int width,int first,int n1,int n2,int axis,AstrDeviceView view,
                             int p[3],int &c,int &d,int &h)
{
    h=id%width+first;
    p[(axis+1)%3]=(id/width)%n1;
    p[(axis+2)%3]=(id/(static_cast<size_t>(width)*n1))%n2;
    const size_t component=id/(static_cast<size_t>(width)*n1*n2);
    c=component%view.extent[3]+1;d=component/view.extent[3]+1;
    return;
}

__global__ __launch_bounds__(threads)
void pack(AstrDeviceView view,int axis,int upper,int width,int first,int n1,int n2,size_t count,double *low,double *high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id<count) {
        int p[3]={0,0,0},c,d,h;
        packet_point(id,width,first,n1,n2,axis,view,p,c,d,h);
        p[axis]=h;
        low[id]=astr::at(view,p[0],p[1],p[2],c,d);
        p[axis]=upper-h;
        high[id]=astr::at(view,p[0],p[1],p[2],c,d);
    }
    return;
}

__global__ __launch_bounds__(threads)
void unpack(AstrDeviceView view,int axis,int upper,int width,int first,int n1,int n2,size_t count,
            const double *low,const double *high,int reconcile)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id<count) {
        int p[3]={0,0,0},c,d,h;
        packet_point(id,width,first,n1,n2,axis,view,p,c,d,h);
        if (h>0 || reconcile) {
            p[axis]=-h;
            double &left=astr::at(view,p[0],p[1],p[2],c,d);
            left=h>0 ? low[id] : .5*(left+low[id]);
            p[axis]=upper+h;
            double &right=astr::at(view,p[0],p[1],p[2],c,d);
            right=h>0 ? high[id] : .5*(right+high[id]);
        }
    }
    return;
}
}

extern "C" int astr_device_transport_prepare(void *handle,int communicator,const int lower[3],
    const int upper[3],const int sizes[3],int max_components)
{
    if (!handle || !lower || !upper || !sizes || max_components<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (s.transport || s.halo<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    size_t capacity=0;
    for (int axis=0;axis<3;++axis) {
        if (sizes[axis]<1 || s.upper[axis]<s.halo) return ASTR_DEVICE_INVALID_ARGUMENT;
        if (sizes[axis]>1) {
            if (lower[axis]==MPI_PROC_NULL || upper[axis]==MPI_PROC_NULL) return ASTR_DEVICE_INVALID_ARGUMENT;
            const size_t count=static_cast<size_t>(s.halo+1)*(s.upper[(axis+1)%3]+1)
                              *(s.upper[(axis+2)%3]+1)*max_components;
            if (count>INT_MAX) return ASTR_DEVICE_INVALID_ARGUMENT;
            if (count>capacity) capacity=count;
        }
    }
    s.transport=new (std::nothrow) astr::HaloTransport{};
    if (!s.transport) return static_cast<int>(cudaErrorMemoryAllocation);
    astr::HaloTransport &t=*s.transport;
    t.max_components=max_components;
    for (int axis=0;axis<3;++axis) { t.lower[axis]=lower[axis];t.upper[axis]=upper[axis];t.sizes[axis]=sizes[axis]; }
    if (capacity==0) return 0;
    int status=MPI_Comm_dup(MPI_Comm_f2c(static_cast<MPI_Fint>(communicator)),&t.communicator);
    if (status!=MPI_SUCCESS) return status;
    // Tags 0/1 belong only to this private communicator; Fortran owns topology.
    for (int side=0;side<2;++side) {
        cudaError_t err=cudaMalloc(&t.send_device[side],capacity*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&t.recv_device[side],capacity*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMallocHost(&t.send_host[side],capacity*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMallocHost(&t.recv_host[side],capacity*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    return 0;
}

extern "C" int astr_device_exchange_axis(void *handle,int field,int axis,int reconcile)
{
    if (!handle || field<0 || field>=ASTR_DEVICE_FIELD_COUNT || axis<0 || axis>2)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (!s.transport) return ASTR_DEVICE_NOT_ALLOCATED;
    astr::HaloTransport &t=*s.transport;
    if (t.sizes[axis]==1) return astr_device_periodic_axis(handle,field,axis,reconcile);
    const AstrDeviceView view=s.fields[field].view;
    if (!view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if (view.lower[axis]!=-s.halo || view.extent[3]*view.extent[4]>t.max_components)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int n1=s.upper[(axis+1)%3]+1,n2=s.upper[(axis+2)%3]+1;
    const int width=s.halo+(reconcile ? 1 : 0),first=reconcile ? 0 : 1;
    const size_t count=static_cast<size_t>(width)*n1*n2*view.extent[3]*view.extent[4],bytes=count*sizeof(double);
    pack<<<(count+threads-1)/threads,threads,0,s.stream>>>(view,axis,s.upper[axis],width,first,n1,n2,count,
        t.send_device[0],t.send_device[1]);
    int status=astr::finish_launch(s);
    if (status!=0) return status;
    for (int side=0;side<2;++side) {
        const cudaError_t err=cudaMemcpyAsync(t.send_host[side],t.send_device[side],bytes,cudaMemcpyDeviceToHost,s.stream);
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    cudaError_t err=cudaStreamSynchronize(s.stream);
    if (err!=cudaSuccess) return static_cast<int>(err);
    status=MPI_Irecv(t.recv_host[0],static_cast<int>(count),MPI_DOUBLE,t.lower[axis],1,t.communicator,&t.requests[0]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Irecv(t.recv_host[1],static_cast<int>(count),MPI_DOUBLE,t.upper[axis],0,t.communicator,&t.requests[1]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Isend(t.send_host[0],static_cast<int>(count),MPI_DOUBLE,t.lower[axis],0,t.communicator,&t.requests[2]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Isend(t.send_host[1],static_cast<int>(count),MPI_DOUBLE,t.upper[axis],1,t.communicator,&t.requests[3]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Waitall(4,t.requests,MPI_STATUSES_IGNORE);
    if (status!=MPI_SUCCESS) return status;
    for (int side=0;side<2;++side) {
        err=cudaMemcpyAsync(t.recv_device[side],t.recv_host[side],bytes,cudaMemcpyHostToDevice,s.stream);
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    unpack<<<(count+threads-1)/threads,threads,0,s.stream>>>(view,axis,s.upper[axis],width,first,n1,n2,count,
        t.recv_device[0],t.recv_device[1],reconcile);
    return astr::finish_launch(s);
}
