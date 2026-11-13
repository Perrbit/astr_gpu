#include <cmath>
#include <climits>
#include "device_state_internal.cuh"
#include "fluid_explicit.h"

namespace {
constexpr int threads = astr::block_threads;
using astr::at;
struct Maximum { double value; int64_t key; };

__device__ Maximum largest(Maximum a, Maximum b)
{
    return b.value > a.value || (b.value == a.value && b.key < a.key) ? b : a;
}

__device__ void indices(size_t id, int nx, int ny, int &i, int &j, int &k)
{
    i = id%nx; j = (id/nx)%ny; k = id/(static_cast<size_t>(nx)*ny);
    return;
}

__global__ __launch_bounds__(threads)
void statistics_partial(AstrDeviceView q, AstrDeviceView rho, AstrDeviceView vel, AstrDeviceView tmp,
                        AstrDeviceView grad, int nx, int ny, size_t count, double re, double tc,
                        double tc1, double *partial)
{
    __shared__ double scratch[3][threads];
    const int lane = threadIdx.x;
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+lane;
    int i=0,j=0,k=0;
    double e=0,z=0,d=0;
    if (id < count) {
        indices(id,nx,ny,i,j,k);
        // Match the original unique-node sum: 1:im, 1:jm, 1:km.
        if (i > 0 && j > 0 && k > 0) {
            const double u=at(vel,i,j,k,1), v=at(vel,i,j,k,2), w=at(vel,i,j,k,3);
            const double a=at(grad,i,j,k,3,2)-at(grad,i,j,k,2,3);
            const double b=at(grad,i,j,k,1,3)-at(grad,i,j,k,3,1);
            const double c=at(grad,i,j,k,2,1)-at(grad,i,j,k,1,2);
            e=at(rho,i,j,k)*(u*u+v*v+w*w);
            z=at(rho,i,j,k)*(a*a+b*b+c*c);
            const double s11=at(grad,i,j,k,1,1),s22=at(grad,i,j,k,2,2),s33=at(grad,i,j,k,3,3);
            const double s12=.5*(at(grad,i,j,k,1,2)+at(grad,i,j,k,2,1));
            const double s13=.5*(at(grad,i,j,k,1,3)+at(grad,i,j,k,3,1));
            const double s23=.5*(at(grad,i,j,k,2,3)+at(grad,i,j,k,3,2));
            const double div=s11+s22+s33,t=at(tmp,i,j,k);
            const double mu=t*sqrt(t)*tc1/(t+tc)/re;
            d=2.0*mu*(s11*s11+s22*s22+s33*s33+2.0*(s12*s12+s13*s13+s23*s23)-div*div/3.0);
        }
    }
    scratch[0][lane]=e; scratch[1][lane]=z; scratch[2][lane]=d;
    __syncthreads();
    for (int s=threads/2;s>0;s/=2) {
        if (lane<s) for (int n=0;n<3;++n) scratch[n][lane]+=scratch[n][lane+s];
        __syncthreads();
    }
    if (lane==0) for (int n=0;n<3;++n) partial[13*blockIdx.x+n]=scratch[n][0];
    __syncthreads();
    for (int n=0;n<5;++n) {
        const double value=id<count ? at(q,i,j,k,n+1) : 0;
        scratch[0][lane]=id<count ? value : -INFINITY;
        scratch[1][lane]=id<count ? value : INFINITY;
        __syncthreads();
        for (int s=threads/2;s>0;s/=2) {
            if (lane<s) {
                scratch[0][lane]=fmax(scratch[0][lane],scratch[0][lane+s]);
                scratch[1][lane]=fmin(scratch[1][lane],scratch[1][lane+s]);
            }
            __syncthreads();
        }
        if (lane==0) {
            partial[13*blockIdx.x+3+n]=scratch[0][0];
            partial[13*blockIdx.x+8+n]=scratch[1][0];
        }
        __syncthreads();
    }
    return;
}

__global__ __launch_bounds__(threads)
void statistics_finish(const double *partial, size_t blocks, double *result)
{
    __shared__ double scratch[threads];
    const int n=blockIdx.x,lane=threadIdx.x;
    double value=n<3 ? 0 : (n<8 ? -INFINITY : INFINITY);
    for (size_t b=lane;b<blocks;b+=threads) {
        const double next=partial[13*b+n];
        value=n<3 ? value+next : (n<8 ? fmax(value,next) : fmin(value,next));
    }
    scratch[lane]=value;
    __syncthreads();
    for (int s=threads/2;s>0;s/=2) {
        if (lane<s) scratch[lane]=n<3 ? scratch[lane]+scratch[lane+s]
            : (n<8 ? fmax(scratch[lane],scratch[lane+s]) : fmin(scratch[lane],scratch[lane+s]));
        __syncthreads();
    }
    if (lane==0) result[n]=scratch[0];
    return;
}

__device__ void rates_at(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,int i,int j,int k,
                        int nondim,double mach,double gamma,double gas,double values[4],int *failure)
{
    const double temperature=at(tmp,i,j,k);
    const double sound=nondim ? sqrt(temperature)/mach : sqrt(gamma*gas*temperature);
    bool valid=isfinite(sound) && sound>0;
    for (int axis=1;axis<=3;++axis) {
        const double a=at(metric,i,j,k,axis,1),b=at(metric,i,j,k,axis,2),c=at(metric,i,j,k,axis,3);
        const double u=at(vel,i,j,k,1),v=at(vel,i,j,k,2),w=at(vel,i,j,k,3);
        values[axis-1]=fabs(u*a+v*b+w*c)+sound*sqrt(a*a+b*b+c*c);
        valid=valid && isfinite(a) && isfinite(b) && isfinite(c) && isfinite(u) && isfinite(v)
            && isfinite(w) && isfinite(values[axis-1]) && values[axis-1]>=0;
    }
    values[3]=values[0]+values[1]+values[2];
    if (!valid || !isfinite(values[3])) atomicCAS(failure,0,2);
    return;
}

__global__ __launch_bounds__(threads)
void cfl_partial(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,int nx,int ny,size_t count,
                 int nondim,double mach,double gamma,double gas,int gx,int gy,int ox,int oy,int oz,
                 Maximum *partial,int *failure)
{
    __shared__ Maximum scratch[4][threads];
    const int lane=threadIdx.x;
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+lane;
    double values[4]={0,0,0,0};
    int64_t key=INT64_MAX;
    if (id<count) {
        int i,j,k;
        indices(id,nx,ny,i,j,k);
        rates_at(vel,tmp,metric,i,j,k,nondim,mach,gamma,gas,values,failure);
        key=(static_cast<int64_t>(k+oz)*(gy+1)+j+oy)*(gx+1)+i+ox;
    }
    for (int n=0;n<4;++n) scratch[n][lane]={values[n],key};
    __syncthreads();
    for (int s=threads/2;s>0;s/=2) {
        if (lane<s) for (int n=0;n<4;++n) scratch[n][lane]=largest(scratch[n][lane],scratch[n][lane+s]);
        __syncthreads();
    }
    if (lane==0) for (int n=0;n<4;++n) partial[4*blockIdx.x+n]=scratch[n][0];
    return;
}

__global__ __launch_bounds__(threads)
void cfl_finish(const Maximum *partial,size_t blocks,double *values,int64_t *keys)
{
    __shared__ Maximum scratch[4][threads];
    const int lane=threadIdx.x;
    Maximum best[4]={{0,INT64_MAX},{0,INT64_MAX},{0,INT64_MAX},{0,INT64_MAX}};
    for (size_t b=lane;b<blocks;b+=threads)
        for (int n=0;n<4;++n) best[n]=largest(best[n],partial[4*b+n]);
    for (int n=0;n<4;++n) scratch[n][lane]=best[n];
    __syncthreads();
    for (int s=threads/2;s>0;s/=2) {
        if (lane<s) for (int n=0;n<4;++n) scratch[n][lane]=largest(scratch[n][lane],scratch[n][lane+s]);
        __syncthreads();
    }
    if (lane==0) for (int n=0;n<4;++n) { values[n]=scratch[n][0].value; keys[n]=scratch[n][0].key; }
    return;
}

__global__ __launch_bounds__(threads)
void cfl_plane(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,int nx,int nz,
               int nondim,double mach,double gamma,double gas,double *profile,int *failure)
{
    __shared__ double scratch[4][threads];
    const int lane=threadIdx.x,j=blockIdx.x;
    double best[4]={0,0,0,0};
    for (size_t id=lane;id<static_cast<size_t>(nx)*nz;id+=threads) {
        double values[4];
        rates_at(vel,tmp,metric,id%nx,j,id/nx,nondim,mach,gamma,gas,values,failure);
        for (int n=0;n<4;++n) best[n]=fmax(best[n],values[n]);
    }
    for (int n=0;n<4;++n) scratch[n][lane]=best[n];
    __syncthreads();
    for (int s=threads/2;s>0;s/=2) {
        if (lane<s) for (int n=0;n<4;++n) scratch[n][lane]=fmax(scratch[n][lane],scratch[n][lane+s]);
        __syncthreads();
    }
    if (lane==0) for (int n=0;n<4;++n) profile[4*j+n]=scratch[n][0];
    return;
}
}

extern "C" int astr_device_fluid_statistics(void *handle,double re,double tc,double tc1,double values[13])
{
    if (!handle || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (!s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3],blocks=(count+threads-1)/threads;
    if (!s.statistics_partial) {
        cudaError_t err=cudaMalloc(&s.statistics_partial,13*blocks*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&s.statistics_values,13*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    statistics_partial<<<blocks,threads,0,s.stream>>>(s.fields[0].view,s.fields[3].view,s.fields[4].view,
        s.fields[6].view,s.fields[8].view,s.upper[0]+1,s.upper[1]+1,count,re,tc,tc1,s.statistics_partial);
    int status=astr::finish_launch(s);
    if (status!=0) return status;
    statistics_finish<<<13,threads,0,s.stream>>>(s.statistics_partial,blocks,s.statistics_values);
    status=astr::finish_launch(s);
    if (status!=0) return status;
    const cudaError_t err=cudaMemcpyAsync(values,s.statistics_values,13*sizeof(double),cudaMemcpyDeviceToHost,s.stream);
    return err==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(err);
}

extern "C" int astr_device_fluid_cfl_diagnostics(void *handle,int nondim,double mach,double gamma,double gas,
    const int global[3],const int origin[3],int ny,double values[4],int64_t keys[4],double *profile)
{
    if (!handle || !global || !origin || !values || !keys || ny<0 || (ny>0 && !profile))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if (ny>0 && ny<origin[1]+s.upper[1]+1) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=s.fields[1].view.stride[3],blocks=(count+threads-1)/threads;
    cudaError_t err=cudaSuccess;
    if (!s.cfl_records) {
        err=cudaMalloc(&s.cfl_records,4*blocks*sizeof(Maximum));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&s.cfl_diagnostic_values,4*sizeof(double));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&s.cfl_diagnostic_keys,4*sizeof(int64_t));
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    cfl_partial<<<blocks,threads,0,s.stream>>>(s.fields[4].view,s.fields[6].view,s.fields[7].view,
        s.upper[0]+1,s.upper[1]+1,count,nondim,mach,gamma,gas,global[0],global[1],origin[0],origin[1],origin[2],
        static_cast<Maximum *>(s.cfl_records),s.fluid_failure);
    int status=astr::finish_launch(s);
    if (status!=0) return status;
    cfl_finish<<<1,threads,0,s.stream>>>(static_cast<const Maximum *>(s.cfl_records),blocks,
        s.cfl_diagnostic_values,s.cfl_diagnostic_keys);
    status=astr::finish_launch(s);
    if (status!=0) return status;
    if (ny>0) {
        if (!s.cfl_profile) {
            err=cudaMalloc(&s.cfl_profile,4*(s.upper[1]+1)*sizeof(double));
            if (err!=cudaSuccess) return static_cast<int>(err);
        }
        cfl_plane<<<s.upper[1]+1,threads,0,s.stream>>>(s.fields[4].view,s.fields[6].view,s.fields[7].view,
            s.upper[0]+1,s.upper[2]+1,nondim,mach,gamma,gas,s.cfl_profile,s.fluid_failure);
        status=astr::finish_launch(s);
        if (status!=0) return status;
        err=cudaMemcpyAsync(profile+4*origin[1],s.cfl_profile,4*(s.upper[1]+1)*sizeof(double),
                            cudaMemcpyDeviceToHost,s.stream);
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    err=cudaMemcpyAsync(values,s.cfl_diagnostic_values,4*sizeof(double),cudaMemcpyDeviceToHost,s.stream);
    if (err!=cudaSuccess) return static_cast<int>(err);
    err=cudaMemcpyAsync(keys,s.cfl_diagnostic_keys,4*sizeof(int64_t),cudaMemcpyDeviceToHost,s.stream);
    return err==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(err);
}
