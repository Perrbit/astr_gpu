#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <new>
#include "compact_line_solver.h"
#include "device_state_internal.cuh"
#include "halo_transport.h"

namespace {
using Real=astr::numerical_real;
const MPI_Datatype compact_mpi_type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
constexpr int tile_threads = 128;

int finish_compact(astr::State &s)
{
    const cudaError_t status=cudaGetLastError();
    if(status!=cudaSuccess) return static_cast<int>(status);
    return s.compact_sync || s.debug_sync ? static_cast<int>(cudaStreamSynchronize(s.stream)) : 0;
}

struct CompactPlan {
    astr::State *parent = nullptr;
    AstrCompactFactors f{};
    int capacity = 0;
    bool registered = false;
    size_t bytes = 0;
    Real *rhs = nullptr, *solution = nullptr, *local = nullptr;
    Real *reduced[2]{};
};

template<class Plan,class T> int allocate(Plan &p, T *&destination, size_t count)
{
    const cudaError_t status = cudaMalloc(&destination,count*sizeof(T));
    if (status == cudaSuccess) p.bytes += count*sizeof(T);
    return static_cast<int>(status);
}

template<class Plan,class T> int copy(Plan &p, T *&destination, const T *source, size_t count)
{
    int status = allocate(p,destination,count);
    if (status != 0) return status;
    return static_cast<int>(cudaMemcpyAsync(destination,source,count*sizeof(T),
                                           cudaMemcpyHostToDevice,p.parent->stream));
}

int release(CompactPlan *p)
{
    int status = static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    void *buffers[] = {p->f.first,p->f.last,p->f.owner,p->f.local_levels,
        p->f.initial,p->f.left,p->f.right,p->f.inverse,p->f.edge_lower,p->f.edge_upper,
        p->f.response_left,p->f.response_right,p->f.reduced_initial,p->f.reduced_left,
        p->f.reduced_right,p->f.reduced_inverse,p->f.cyclic_response,
        p->rhs,p->solution,p->local,p->reduced[0],p->reduced[1]};
    for (void *buffer : buffers) {
        if (!buffer) continue;
        const int freed = static_cast<int>(cudaFree(buffer));
        if (status == 0) status = freed;
    }
    if (p->registered) --p->parent->compact_plans;
    delete p;
    return status;
}

__global__ void local_pcr(AstrCompactFactors f, const Real *rhs, Real *local, Real *reduced)
{
    __shared__ Real old[tile_threads], next[tile_threads];
    const int lane = threadIdx.x, tile = blockIdx.x, column = blockIdx.y;
    const int first = f.first[tile]-1, last = f.last[tile]-1, m = last-first-1;
    const size_t col = static_cast<size_t>(column)*f.n;
    Real value = 0.0;
    if (lane < m) value = rhs[first+1+lane+col]*f.initial[lane+f.tile_capacity*tile];
    old[lane] = value;
    __syncthreads();
    int stride = 1;
    for (int level = 0; level < f.local_levels[tile]; ++level) {
        value = 0.0;
        if (lane < m) {
            const size_t pos = lane+static_cast<size_t>(f.tile_capacity)*(level+f.levels*tile);
            value = old[lane];
            if (lane >= stride) value -= f.left[pos]*old[lane-stride];
            if (lane+stride < m) value -= f.right[pos]*old[lane+stride];
            value *= f.inverse[pos];
        }
        next[lane] = value;
        __syncthreads();
        old[lane] = next[lane];
        __syncthreads();
        stride *= 2;
    }
    if (lane < m) local[first+1+lane+col] = old[lane];
    if (lane == 0) {
        Real lo = rhs[first+col], hi = rhs[last+col];
        if (m > 0) {
            lo -= f.edge_upper[tile]*old[0];
            hi -= f.edge_lower[tile]*old[m-1];
        }
        const int k = 2*tile;
        const size_t base = static_cast<size_t>(column)*f.reduced_n;
        reduced[k+base] = lo*f.reduced_initial[k];
        reduced[k+1+base] = hi*f.reduced_initial[k+1];
    }
}

__global__ void reduced_pcr(AstrCompactFactors f, int level, int stride,
                            const Real *old, Real *next)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= f.reduced_n) return;
    const size_t col = static_cast<size_t>(blockIdx.y)*f.reduced_n;
    const size_t pos = i+static_cast<size_t>(f.reduced_n)*level;
    Real value = old[i+col];
    if (i >= stride) value -= f.reduced_left[pos]*old[i-stride+col];
    if (i+stride < f.reduced_n) value -= f.reduced_right[pos]*old[i+stride+col];
    next[i+col] = value*f.reduced_inverse[pos];
}

__global__ void reconstruct(AstrCompactFactors f, const Real *reduced,
                             const Real *local, Real *solution)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= f.n) return;
    const int tile = f.owner[i]-1, k = 2*tile;
    const size_t rcol = static_cast<size_t>(blockIdx.y)*f.reduced_n;
    const size_t col = static_cast<size_t>(blockIdx.y)*f.n;
    Real lo = reduced[k+rcol], hi = reduced[k+1+rcol];
    if (f.cyclic) {
        const Real d1 = f.correction[0]*reduced[f.reduced_n-1+rcol]+f.correction[1]*reduced[rcol];
        const Real d2 = f.correction[2]*reduced[f.reduced_n-1+rcol]+f.correction[3]*reduced[rcol];
        lo = lo-f.cyclic_response[k]*d1-f.cyclic_response[k+f.reduced_n]*d2;
        hi = hi-f.cyclic_response[k+1]*d1-f.cyclic_response[k+1+f.reduced_n]*d2;
    }
    if (i == f.first[tile]-1) solution[i+col] = lo;
    else if (i == f.last[tile]-1) solution[i+col] = hi;
    else solution[i+col] = local[i+col]-f.response_left[i]*lo-f.response_right[i]*hi;
}

AstrDeviceView matrix_view(Real *data, int n, int capacity)
{
    AstrDeviceView view{};
    view.data = data; view.rank = 2; view.element_bytes = sizeof(Real);
    int64_t stride = 1;
    for (int axis = 0; axis < 5; ++axis) {
        view.lower[axis] = 1;
        view.extent[axis] = axis == 0 ? n : (axis == 1 ? capacity : 1);
        view.stride[axis] = stride;
        stride *= view.extent[axis];
    }
    return view;
}
}

extern "C" int astr_compact_line_create(void *parent, const AstrCompactFactors *f,
                                       int capacity, void **handle, size_t *device_bytes)
{
    if (!parent || !f || !handle || !device_bytes || capacity < 1 || f->n < 2 ||
        f->tile_capacity < 4 || f->tile_capacity > tile_threads || f->tiles < 1 ||
        f->reduced_n != 2*f->tiles || f->levels < 1 || f->reduced_levels < 1)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    *handle = nullptr; *device_bytes = 0;
    int device;
    cudaDeviceProp properties;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return static_cast<int>(error);
    error = cudaGetDeviceProperties(&properties,device);
    if (error != cudaSuccess) return static_cast<int>(error);
    if (capacity > properties.maxGridSize[1] || f->tiles > properties.maxGridSize[0] ||
        tile_threads > properties.maxThreadsPerBlock) return ASTR_DEVICE_INVALID_ARGUMENT;
    CompactPlan *p = new (std::nothrow) CompactPlan{};
    if (!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->parent = static_cast<astr::State *>(parent); p->capacity = capacity;
    p->f.n=f->n; p->f.tile_capacity=f->tile_capacity; p->f.tiles=f->tiles;
    p->f.levels=f->levels; p->f.reduced_n=f->reduced_n;
    p->f.reduced_levels=f->reduced_levels; p->f.cyclic=f->cyclic;
    for (int i=0;i<4;++i) p->f.correction[i]=f->correction[i];
    int status=0;
    const size_t local_size=static_cast<size_t>(f->tile_capacity)*f->tiles;
    const size_t reduced_size=static_cast<size_t>(f->reduced_n)*f->reduced_levels;
#define COPY(member,count) if (status==0) status=copy(*p,p->f.member,f->member,count)
    COPY(first,f->tiles); COPY(last,f->tiles); COPY(owner,f->n); COPY(local_levels,f->tiles);
    COPY(initial,local_size); COPY(left,local_size*f->levels);
    COPY(right,local_size*f->levels); COPY(inverse,local_size*f->levels);
    COPY(edge_lower,f->tiles); COPY(edge_upper,f->tiles);
    COPY(response_left,f->n); COPY(response_right,f->n);
    COPY(reduced_initial,f->reduced_n); COPY(reduced_left,reduced_size);
    COPY(reduced_right,reduced_size); COPY(reduced_inverse,reduced_size);
    COPY(cyclic_response,2*static_cast<size_t>(f->reduced_n));
#undef COPY
    const size_t nodes=static_cast<size_t>(f->n)*capacity;
    const size_t endpoints=static_cast<size_t>(f->reduced_n)*capacity;
    if (status==0) status=allocate(*p,p->rhs,nodes);
    if (status==0) status=allocate(*p,p->solution,nodes);
    if (status==0) status=allocate(*p,p->local,nodes);
    if (status==0) status=allocate(*p,p->reduced[0],endpoints);
    if (status==0) status=allocate(*p,p->reduced[1],endpoints);
    if (status==0) status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    if (status!=0) { release(p); return status; }
    ++p->parent->compact_plans; p->registered=true;
    *handle=p; *device_bytes=p->bytes;
    return 0;
}

extern "C" int astr_compact_line_views(void *handle, AstrDeviceView *rhs, AstrDeviceView *solution)
{
    if (!handle || !rhs || !solution) return ASTR_DEVICE_INVALID_ARGUMENT;
    CompactPlan &p=*static_cast<CompactPlan *>(handle);
    *rhs=matrix_view(p.rhs,p.f.n,p.capacity);
    *solution=matrix_view(p.solution,p.f.n,p.capacity);
    return 0;
}

extern "C" int astr_compact_line_solve(void *handle, int nrhs)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    CompactPlan &p=*static_cast<CompactPlan *>(handle);
    if (nrhs<1 || nrhs>p.capacity) return ASTR_DEVICE_INVALID_ARGUMENT;
    local_pcr<<<dim3(p.f.tiles,nrhs),tile_threads,0,p.parent->stream>>>(p.f,p.rhs,p.local,p.reduced[0]);
    int status=finish_compact(*p.parent);
    if (status!=0) return status;
    int stride=1;
    for (int level=0;level<p.f.reduced_levels;++level) {
        reduced_pcr<<<dim3((p.f.reduced_n+tile_threads-1)/tile_threads,nrhs),tile_threads,0,p.parent->stream>>>(
            p.f,level,stride,p.reduced[level%2],p.reduced[(level+1)%2]);
        status=finish_compact(*p.parent);
        if (status!=0) return status;
        stride*=2;
    }
    reconstruct<<<dim3((p.f.n+tile_threads-1)/tile_threads,nrhs),tile_threads,0,p.parent->stream>>>(
        p.f,p.reduced[p.f.reduced_levels%2],p.local,p.solution);
    return finish_compact(*p.parent);
}

extern "C" int astr_compact_line_transfer(void *handle, int mode, Real *host, int nrhs)
{
    if (!handle || !host || mode<0 || mode>2) return ASTR_DEVICE_INVALID_ARGUMENT;
    CompactPlan &p=*static_cast<CompactPlan *>(handle);
    if (nrhs<1 || nrhs>p.capacity) return ASTR_DEVICE_INVALID_ARGUMENT;
    Real *device=mode==2 ? p.solution : p.rhs;
    const size_t bytes=static_cast<size_t>(p.f.n)*nrhs*sizeof(Real);
    const cudaError_t error=mode==0
        ? cudaMemcpyAsync(device,host,bytes,cudaMemcpyHostToDevice,p.parent->stream)
        : cudaMemcpyAsync(host,device,bytes,cudaMemcpyDeviceToHost,p.parent->stream);
    if (error!=cudaSuccess) return static_cast<int>(error);
    return static_cast<int>(cudaStreamSynchronize(p.parent->stream));
}

extern "C" int astr_compact_line_destroy(void **handle)
{
    if (!handle || !*handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    CompactPlan *p=static_cast<CompactPlan *>(*handle);
    *handle=nullptr;
    return release(p);
}

namespace {
struct InterfaceExchange {
    MPI_Request requests[4]{MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
    int nrhs=0,left=MPI_PROC_NULL,right=MPI_PROC_NULL;
    bool restore_only=false;
};

struct DistributedPlan {
    CompactPlan *local = nullptr;
    astr::State *parent = nullptr;
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank=0, ranks=1, levels=0, mode=ASTR_HALO_PINNED, nodes=0;
    bool cyclic=false, owns_local=false;
    size_t bytes=0, host_bytes=0;
    int64_t sent_bytes=0, staging_bytes=0;
    Real *response=nullptr, *coefficient=nullptr, *cyclic_response=nullptr, *restored=nullptr;
    Real *endpoint[2]{}, *low=nullptr, *high=nullptr, *first=nullptr, *last=nullptr;
    Real *host[3]{};
    Real correction[4]{};
    InterfaceExchange pending;
    int local_ready=0;
    cudaEvent_t phase_begin=nullptr,phase_end=nullptr;
    // Local GPU, PCR interface host, periodic endpoint host, backsolve GPU, restore exchange host.
    double phase_seconds[5]{};
};

int start_device_phase(DistributedPlan &p)
{
    return p.phase_begin ? static_cast<int>(cudaEventRecord(p.phase_begin,p.parent->stream)) : 0;
}

int end_device_phase(DistributedPlan &p,int phase,int status)
{
    if(status!=0 || !p.phase_begin) return status;
    cudaError_t error=cudaEventRecord(p.phase_end,p.parent->stream);
    if(error==cudaSuccess) error=cudaEventSynchronize(p.phase_end);
    float ms=0.0f;
    if(error==cudaSuccess) error=cudaEventElapsedTime(&ms,p.phase_begin,p.phase_end);
    if(error==cudaSuccess) p.phase_seconds[phase]+=1e-3*ms;
    return static_cast<int>(error);
}

__global__ void endpoints(int n,int nrhs,const Real *values,Real *endpoint)
{
    const int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=nrhs) return;
    const size_t col=static_cast<size_t>(j)*n;
    endpoint[2*j]=values[col];endpoint[2*j+1]=values[n-1+col];
}

__global__ void interface_pcr(int nrhs,int level,const Real *coefficient,const Real *old,
                              const Real *low,const Real *high,Real *next)
{
    const int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=nrhs) return;
    const Real *c=coefficient+12*level;
    for(int k=0;k<2;++k) next[2*j+k]=c[k]*old[2*j]+c[k+2]*old[2*j+1]
        -c[k+4]*low[2*j]-c[k+6]*low[2*j+1]-c[k+8]*high[2*j]-c[k+10]*high[2*j+1];
}

__global__ void reconstruct_global(int n,const Real *response,const Real *low,
                                    const Real *high,Real *values)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
    if(i<n) {
        const size_t pos=i+static_cast<size_t>(n)*j;
        values[pos]=values[pos]-response[i]*low[2*j+1]-response[i+n]*high[2*j];
    }
}

__global__ void cyclic_basis(int n,int rank,int ranks,Real lower,Real upper,Real *rhs)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
    if(i>=n) return;
    Real value=0;
    if(rank==0 && i==0 && j==0) value=lower;
    if(rank==ranks-1 && i==n-1 && j==1) value=upper;
    rhs[i+static_cast<size_t>(n)*j]=value;
}

__global__ void correct_global(int n,const Real *z,const Real *first,const Real *last,
                                Real h11,Real h12,Real h21,Real h22,Real *values)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
    if(i>=n) return;
    const Real c1=h11*last[2*j+1]+h12*first[2*j];
    const Real c2=h21*last[2*j+1]+h22*first[2*j];
    const size_t pos=i+static_cast<size_t>(n)*j;
    values[pos]=values[pos]-z[i]*c1-z[i+n]*c2;
}

__global__ void restore_nodes(int n,int nodes,const Real *solution,const Real *high,Real *values)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;
    if(i<nodes) values[i+static_cast<size_t>(nodes)*j]=i<n
        ? solution[i+static_cast<size_t>(n)*j] : high[2*j];
}

int post_exchange(DistributedPlan &p,const Real *send,int nrhs,int left,int right,InterfaceExchange &request)
{
    const size_t bytes=2*static_cast<size_t>(nrhs)*sizeof(Real);
    cudaError_t error=cudaSuccess;
    if(p.mode==ASTR_HALO_PINNED)
        error=cudaMemcpyAsync(p.host[0],send,bytes,cudaMemcpyDeviceToHost,p.parent->stream);
    if(error!=cudaSuccess) return static_cast<int>(error);
    error=cudaStreamSynchronize(p.parent->stream);
    if(error!=cudaSuccess) return static_cast<int>(error);
    Real *low=p.mode==ASTR_HALO_DEVICE ? p.low : p.host[1];
    Real *high=p.mode==ASTR_HALO_DEVICE ? p.high : p.host[2];
    const Real *source=p.mode==ASTR_HALO_DEVICE ? send : p.host[0];
    int status=MPI_SUCCESS;
    if(!request.restore_only) {
        status=MPI_Irecv(low,2*nrhs,compact_mpi_type,left,1,p.communicator,&request.requests[0]);
        if(status!=MPI_SUCCESS) return status;
    }
    status=MPI_Irecv(high,2*nrhs,compact_mpi_type,right,0,p.communicator,&request.requests[1]);
    if(status!=MPI_SUCCESS) return status;
    status=MPI_Isend(source,2*nrhs,compact_mpi_type,left,0,p.communicator,&request.requests[2]);
    if(status!=MPI_SUCCESS) return status;
    if(!request.restore_only) {
        status=MPI_Isend(source,2*nrhs,compact_mpi_type,right,1,p.communicator,&request.requests[3]);
        if(status!=MPI_SUCCESS) return status;
    }
    request.nrhs=nrhs;request.left=left;request.right=right;
    return 0;
}

int complete_exchange(DistributedPlan &p,InterfaceExchange &request)
{
    const int left=request.left,right=request.right;
    const size_t bytes=2*static_cast<size_t>(request.nrhs)*sizeof(Real);
    const int status=MPI_Waitall(4,request.requests,MPI_STATUSES_IGNORE);
    if(status!=MPI_SUCCESS) return status;
    request.nrhs=0;
    const bool has_left=left!=MPI_PROC_NULL,has_right=right!=MPI_PROC_NULL;
    const bool receive_low=has_left && !request.restore_only;
    if(p.mode==ASTR_HALO_PINNED) {
        // Both slots start at zero and stay mirrored; a null receive changes neither.
        if(receive_low) {
            const cudaError_t error=cudaMemcpyAsync(p.low,p.host[1],bytes,cudaMemcpyHostToDevice,p.parent->stream);
            if(error!=cudaSuccess) return static_cast<int>(error);
        }
        if(has_right) {
            const cudaError_t error=cudaMemcpyAsync(p.high,p.host[2],bytes,cudaMemcpyHostToDevice,p.parent->stream);
            if(error!=cudaSuccess) return static_cast<int>(error);
        }
        p.staging_bytes+=bytes*(1+receive_low+has_right);
    }
    p.sent_bytes+=bytes*(has_left+(has_right && !request.restore_only));
    return p.parent->compact_sync || p.parent->debug_sync ? static_cast<int>(cudaStreamSynchronize(p.parent->stream)) : 0;
}

int exchange(DistributedPlan &p,const Real *send,int nrhs,int left,int right,bool restore_only=false)
{
    InterfaceExchange request;
    request.restore_only=restore_only;
    const int status=post_exchange(p,send,nrhs,left,right,request);
    return status==0 ? complete_exchange(p,request) : status;
}

int broadcast(DistributedPlan &p,Real *values,int nrhs,int root)
{
    const size_t bytes=2*static_cast<size_t>(nrhs)*sizeof(Real);
    cudaError_t error=cudaSuccess;
    if(p.mode==ASTR_HALO_PINNED && p.rank==root)
        error=cudaMemcpyAsync(p.host[0],values,bytes,cudaMemcpyDeviceToHost,p.parent->stream);
    if(error!=cudaSuccess) return static_cast<int>(error);
    error=cudaStreamSynchronize(p.parent->stream);
    if(error!=cudaSuccess) return static_cast<int>(error);
    int status=MPI_Bcast(p.mode==ASTR_HALO_DEVICE ? values : p.host[0],2*nrhs,compact_mpi_type,root,p.communicator);
    if(status!=MPI_SUCCESS) return status;
    if(p.mode==ASTR_HALO_PINNED) {
        error=cudaMemcpyAsync(values,p.host[0],bytes,cudaMemcpyHostToDevice,p.parent->stream);
        if(error!=cudaSuccess) return static_cast<int>(error);
        p.staging_bytes+=bytes*(1+(p.rank==root));
    }
    if(p.rank==root) p.sent_bytes+=bytes*(p.ranks-1);
    return p.parent->compact_sync || p.parent->debug_sync ? static_cast<int>(cudaStreamSynchronize(p.parent->stream)) : 0;
}

int prepare_local(DistributedPlan &p,int nrhs)
{
    const int n=p.local->f.n;
    int status=start_device_phase(p);
    if(status!=0) return status;
    status=astr_compact_line_solve(p.local,nrhs);
    if(status!=0) return status;
    endpoints<<<(nrhs+127)/128,128,0,p.parent->stream>>>(n,nrhs,p.local->solution,p.endpoint[0]);
    return end_device_phase(p,0,finish_compact(*p.parent));
}

int solve_interfaces(DistributedPlan &p,int nrhs,InterfaceExchange *first=nullptr)
{
    const int n=p.local->f.n;
    int status;
    int stride=1;
    const double started=p.phase_begin ? MPI_Wtime() : 0.0;
    for(int level=0;level<p.levels;++level) {
        int left=p.rank-stride,right=p.rank+stride;
        if(left<0) left=MPI_PROC_NULL;
        if(right>=p.ranks) right=MPI_PROC_NULL;
        status=level==0 && first ? complete_exchange(p,*first) : exchange(p,p.endpoint[level%2],nrhs,left,right);
        if(status!=0) return status;
        interface_pcr<<<(nrhs+127)/128,128,0,p.parent->stream>>>(nrhs,level,p.coefficient,
            p.endpoint[level%2],p.low,p.high,p.endpoint[(level+1)%2]);
        status=finish_compact(*p.parent);
        if(status!=0) return status;
        stride*=2;
    }
    status=exchange(p,p.endpoint[p.levels%2],nrhs,p.rank==0 ? MPI_PROC_NULL : p.rank-1,
                    p.rank==p.ranks-1 ? MPI_PROC_NULL : p.rank+1);
    if(status!=0) return status;
    if(p.phase_begin) p.phase_seconds[1]+=MPI_Wtime()-started;
    status=start_device_phase(p);
    if(status!=0) return status;
    reconstruct_global<<<dim3((n+127)/128,nrhs),128,0,p.parent->stream>>>(
        n,p.response,p.low,p.high,p.local->solution);
    return end_device_phase(p,3,finish_compact(*p.parent));
}

int solve_base(DistributedPlan &p,int nrhs)
{
    const int status=prepare_local(p,nrhs);
    return status==0 ? solve_interfaces(p,nrhs) : status;
}

int global_endpoints(DistributedPlan &p,int nrhs)
{
    endpoints<<<(nrhs+127)/128,128,0,p.parent->stream>>>(p.local->f.n,nrhs,p.local->solution,p.first);
    int status=finish_compact(*p.parent);
    if(status!=0) return status;
    cudaError_t error=cudaMemcpyAsync(p.last,p.first,2*static_cast<size_t>(nrhs)*sizeof(Real),
                                     cudaMemcpyDeviceToDevice,p.parent->stream);
    if(error!=cudaSuccess) return static_cast<int>(error);
    status=broadcast(p,p.first,nrhs,0);
    if(status!=0) return status;
    return broadcast(p,p.last,nrhs,p.ranks-1);
}

int correct_periodic(DistributedPlan &p,int nrhs)
{
    if(!p.cyclic) return 0;
    const double started=p.phase_begin ? MPI_Wtime() : 0.0;
    int status=global_endpoints(p,nrhs);
    if(status!=0) return status;
    if(p.phase_begin) p.phase_seconds[2]+=MPI_Wtime()-started;
    status=start_device_phase(p);
    if(status!=0) return status;
    correct_global<<<dim3((p.local->f.n+127)/128,nrhs),128,0,p.parent->stream>>>(
        p.local->f.n,p.cyclic_response,p.first,p.last,p.correction[0],p.correction[1],
        p.correction[2],p.correction[3],p.local->solution);
    return end_device_phase(p,3,finish_compact(*p.parent));
}

int release(DistributedPlan *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    const cudaEvent_t events[]{p->phase_begin,p->phase_end};
    for(auto event:events) {
        if(!event) continue;
        const int freed=static_cast<int>(cudaEventDestroy(event));
        if(status==0) status=freed;
    }
    void *buffers[]={p->response,p->coefficient,p->cyclic_response,p->restored,
        p->endpoint[0],p->endpoint[1],p->low,p->high,p->first,p->last};
    for(void *buffer:buffers) {
        if(!buffer) continue;
        const int freed=static_cast<int>(cudaFree(buffer));
        if(status==0) status=freed;
    }
    for(Real *buffer:p->host) {
        if(!buffer) continue;
        const int freed=static_cast<int>(cudaFreeHost(buffer));
        if(status==0) status=freed;
    }
    if(p->owns_local) {
        const int freed=release(p->local);
        if(status==0) status=freed;
    }
    delete p;
    return status;
}
}

extern "C" int astr_compact_distributed_create(void **local,const AstrCompactDistributedFactors *f,
                                               void **handle,size_t *device_bytes,size_t *host_bytes)
{
    if(!local || !*local || !f || !handle || !device_bytes || !host_bytes ||
       (f->mode!=ASTR_HALO_PINNED && f->mode!=ASTR_HALO_DEVICE)) return ASTR_DEVICE_INVALID_ARGUMENT;
    CompactPlan &line=*static_cast<CompactPlan *>(*local);
    if(line.f.cyclic || line.capacity<2 || f->nodes<line.f.n || f->nodes>line.f.n+1)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    *handle=nullptr;*device_bytes=0;*host_bytes=0;
    DistributedPlan *p=new(std::nothrow) DistributedPlan{};
    if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->local=&line;p->parent=line.parent;p->mode=f->mode;p->nodes=f->nodes;p->cyclic=f->cyclic!=0;
    p->communicator=MPI_Comm_f2c(static_cast<MPI_Fint>(f->communicator));
    int status=MPI_Comm_rank(p->communicator,&p->rank);
    if(status==MPI_SUCCESS) status=MPI_Comm_size(p->communicator,&p->ranks);
    for(int stride=1;stride<p->ranks;stride*=2) ++p->levels;
    if(status==MPI_SUCCESS && p->levels!=f->levels) status=ASTR_DEVICE_INVALID_ARGUMENT;
    if(status==0) status=copy(*p,p->response,f->response,2*static_cast<size_t>(line.f.n));
    if(status==0 && p->levels>0) status=copy(*p,p->coefficient,f->coefficient,12*static_cast<size_t>(p->levels));
    if(status==0) status=allocate(*p,p->cyclic_response,2*static_cast<size_t>(line.f.n));
    if(status==0) status=allocate(*p,p->restored,static_cast<size_t>(p->nodes)*line.capacity);
    Real **packets[]={&p->endpoint[0],&p->endpoint[1],&p->low,&p->high,&p->first,&p->last};
    for(Real **packet:packets) {
        if(status==0) status=allocate(*p,*packet,2*static_cast<size_t>(line.capacity));
        if(status==0) status=static_cast<int>(cudaMemsetAsync(*packet,0,2*static_cast<size_t>(line.capacity)*sizeof(Real),p->parent->stream));
    }
    if(p->mode==ASTR_HALO_PINNED) for(Real *&buffer:p->host) {
        if(status==0) status=static_cast<int>(cudaMallocHost(&buffer,2*static_cast<size_t>(line.capacity)*sizeof(Real)));
        if(status==0) {
            p->host_bytes+=2*static_cast<size_t>(line.capacity)*sizeof(Real);
            for(int i=0;i<2*line.capacity;++i) buffer[i]=0;
        }
    }
    if(status==0) status=static_cast<int>(cudaMemsetAsync(p->cyclic_response,0,2*static_cast<size_t>(line.f.n)*sizeof(Real),p->parent->stream));
    if(status==0 && p->cyclic) {
        cyclic_basis<<<dim3((line.f.n+127)/128,2),128,0,p->parent->stream>>>(
            line.f.n,p->rank,p->ranks,f->edges[0],f->edges[1],line.rhs);
        status=finish_compact(*p->parent);
        if(status==0) status=solve_base(*p,2);
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(p->cyclic_response,line.solution,
            2*static_cast<size_t>(line.f.n)*sizeof(Real),cudaMemcpyDeviceToDevice,p->parent->stream));
        if(status==0) status=global_endpoints(*p,2);
        Real first[4],last[4];
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(first,p->first,sizeof(first),cudaMemcpyDeviceToHost,p->parent->stream));
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(last,p->last,sizeof(last),cudaMemcpyDeviceToHost,p->parent->stream));
        if(status==0) status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
        if(status==0) {
            const Real a=1+last[1],b=last[3],c=first[0],d=1+first[2],det=a*d-b*c;
            if(det==0 || !std::isfinite(det)) status=ASTR_DEVICE_INVALID_ARGUMENT;
            else { p->correction[0]=d/det;p->correction[1]=-b/det;p->correction[2]=-c/det;p->correction[3]=a/det; }
        }
    }
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    if(status!=0) { release(p);return status; }
    p->sent_bytes=0;p->staging_bytes=0;
    p->owns_local=true;*local=nullptr;*handle=p;
    *device_bytes=p->bytes+line.bytes;*host_bytes=p->host_bytes;
    return 0;
}

extern "C" int astr_compact_distributed_views(void *handle,AstrDeviceView *rhs,
                                              AstrDeviceView *solution,AstrDeviceView *nodes)
{
    if(!handle || !nodes) return ASTR_DEVICE_INVALID_ARGUMENT;
    DistributedPlan &p=*static_cast<DistributedPlan *>(handle);
    int status=astr_compact_line_views(p.local,rhs,solution);
    if(status==0) *nodes=matrix_view(p.restored,p.nodes,p.local->capacity);
    return status;
}

extern "C" int astr_compact_distributed_solve(void *handle,int nrhs)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    DistributedPlan &p=*static_cast<DistributedPlan *>(handle);
    if(nrhs<1 || nrhs>p.local->capacity || p.pending.nrhs || p.local_ready) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=solve_base(p,nrhs);
    return status==0 ? correct_periodic(p,nrhs) : status;
}

extern "C" int astr_compact_distributed_local(void *handle,int nrhs)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<DistributedPlan *>(handle);
    if(nrhs<1 || nrhs>p.local->capacity || p.ranks<2 || p.pending.nrhs || p.local_ready)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=prepare_local(p,nrhs);
    if(status==0) p.local_ready=nrhs;
    return status;
}

extern "C" int astr_compact_distributed_begin(void *handle,int nrhs)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<DistributedPlan *>(handle);
    if(nrhs<1 || p.local_ready!=nrhs || p.pending.nrhs) return ASTR_DEVICE_INVALID_ARGUMENT;
    const double started=p.phase_begin ? MPI_Wtime() : 0.0;
    const int status=post_exchange(p,p.endpoint[0],nrhs,p.rank==0 ? MPI_PROC_NULL : p.rank-1,
                                   p.rank==p.ranks-1 ? MPI_PROC_NULL : p.rank+1,p.pending);
    if(p.phase_begin) p.phase_seconds[1]+=MPI_Wtime()-started;
    if(status==0) p.local_ready=0;
    return status;
}

extern "C" int astr_compact_distributed_finish(void *handle,int nrhs)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<DistributedPlan *>(handle);
    if(nrhs<1 || p.pending.nrhs!=nrhs) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=solve_interfaces(p,nrhs,&p.pending);
    return status==0 ? correct_periodic(p,nrhs) : status;
}

extern "C" int astr_compact_distributed_restore(void *handle,int nrhs)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    DistributedPlan &p=*static_cast<DistributedPlan *>(handle);
    if(nrhs<1 || nrhs>p.local->capacity) return ASTR_DEVICE_INVALID_ARGUMENT;
    const double started=p.phase_begin ? MPI_Wtime() : 0.0;
    endpoints<<<(nrhs+127)/128,128,0,p.parent->stream>>>(p.local->f.n,nrhs,p.local->solution,p.endpoint[0]);
    int status=finish_compact(*p.parent);
    int left=p.rank-1,right=p.rank+1;
    if(p.cyclic) { left=(left+p.ranks)%p.ranks;right%=p.ranks; }
    else { if(left<0) left=MPI_PROC_NULL;if(right>=p.ranks) right=MPI_PROC_NULL; }
    // Only the right rank's first node is needed to restore our upper shared node.
    if(status==0) status=exchange(p,p.endpoint[0],nrhs,left,right,true);
    if(p.phase_begin) p.phase_seconds[4]+=MPI_Wtime()-started;
    if(status==0) status=start_device_phase(p);
    if(status==0) {
        restore_nodes<<<dim3((p.nodes+127)/128,nrhs),128,0,p.parent->stream>>>(
            p.local->f.n,p.nodes,p.local->solution,p.high,p.restored);
        status=end_device_phase(p,3,finish_compact(*p.parent));
    }
    return status;
}

extern "C" int astr_compact_distributed_transfer(void *handle,int mode,Real *host,int nrhs)
{
    if(!handle || !host || mode<0 || mode>3) return ASTR_DEVICE_INVALID_ARGUMENT;
    DistributedPlan &p=*static_cast<DistributedPlan *>(handle);
    if(nrhs<1 || nrhs>p.local->capacity) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(mode<3) return astr_compact_line_transfer(p.local,mode,host,nrhs);
    cudaError_t error=cudaMemcpyAsync(host,p.restored,static_cast<size_t>(p.nodes)*nrhs*sizeof(Real),
                                     cudaMemcpyDeviceToHost,p.parent->stream);
    if(error!=cudaSuccess) return static_cast<int>(error);
    return static_cast<int>(cudaStreamSynchronize(p.parent->stream));
}

extern "C" int astr_compact_distributed_metrics(void *handle,int64_t values[2],int reset)
{
    if(!handle || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    DistributedPlan &p=*static_cast<DistributedPlan *>(handle);
    values[0]=p.sent_bytes;values[1]=p.staging_bytes;
    if(reset) { p.sent_bytes=0;p.staging_bytes=0; }
    return 0;
}

extern "C" int astr_compact_distributed_phase_enable(void *handle)
{
    auto &p=*static_cast<DistributedPlan *>(handle);
    cudaError_t status=cudaEventCreate(&p.phase_begin);
    if(status==cudaSuccess) status=cudaEventCreate(&p.phase_end);
    return static_cast<int>(status);
}

extern "C" void astr_compact_distributed_phase_seconds(void *handle,double values[5])
{
    auto &p=*static_cast<DistributedPlan *>(handle);
    for(int i=0;i<5;++i) {
        values[i]=p.phase_seconds[i];
        p.phase_seconds[i]=0;
    }
}

extern "C" int astr_compact_distributed_destroy(void **handle)
{
    if(!handle || !*handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    DistributedPlan *p=static_cast<DistributedPlan *>(*handle);
    if(p->pending.nrhs) return ASTR_DEVICE_INVALID_ARGUMENT;
    *handle=nullptr;
    return release(p);
}
