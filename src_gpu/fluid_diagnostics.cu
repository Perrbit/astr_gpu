#include <cmath>
#include <climits>
#include "device_state_internal.cuh"
#include "fluid_explicit.h"
#ifndef ASTR_REAL32
#include "chemistry_properties_math.cuh"
#endif

namespace {
constexpr int threads = astr::block_threads;
using astr::at;
using Real=astr::numerical_real;
struct Maximum { Real value; int64_t key; };
constexpr int adaptive_threads=256;

__device__ Maximum largest(Maximum a, Maximum b)
{
    return b.value > a.value || (b.value == a.value && b.key < a.key) ? b : a;
}

__device__ void indices(size_t id, int nx, int ny, int &i, int &j, int &k)
{
    i = id%nx; j = (id/nx)%ny; k = id/(static_cast<size_t>(nx)*ny);
    return;
}

__global__ __launch_bounds__(adaptive_threads)
void adaptive_energy_partial(AstrDeviceView q,int im,int jm,size_t count,Real *partial)
{
    __shared__ Real sums[adaptive_threads];
    __shared__ int bad[adaptive_threads];
    const int lane=threadIdx.x;
    const size_t stride=static_cast<size_t>(gridDim.x)*blockDim.x;
    Real total=0;int invalid=0;
    for(size_t node=static_cast<size_t>(blockIdx.x)*blockDim.x+lane;node<count;node+=stride) {
        int i,j,k;indices(node,im,jm,i,j,k);
        const Real density=at(q,i,j,k,1);
        if(!isfinite(density)||density<=0) {invalid=1;continue;}
        const Real a=at(q,i,j,k,2),b=at(q,i,j,k,3),c=at(q,i,j,k,4);
        const Real value=(a*a+b*b+c*c)/density;
        if(!isfinite(value)||value<0) invalid=1;
        else total+=value;
    }
    sums[lane]=total;bad[lane]=invalid;__syncthreads();
    for(int s=adaptive_threads/2;s>0;s/=2) {
        if(lane<s) {sums[lane]+=sums[lane+s];bad[lane]|=bad[lane+s];}
        __syncthreads();
    }
    if(lane==0) partial[blockIdx.x]=bad[0] ? Real(-1) : sums[0];
}

__global__ __launch_bounds__(adaptive_threads)
void adaptive_energy_finish(const Real *partial,int blocks,Real *result)
{
    __shared__ Real sums[adaptive_threads];
    __shared__ int bad[adaptive_threads];
    const int lane=threadIdx.x;
    Real total=0;int invalid=0;
    for(int n=lane;n<blocks;n+=adaptive_threads) {
        const Real value=partial[n];
        if(!isfinite(value)||value<0) invalid=1;
        else total+=value;
    }
    sums[lane]=total;bad[lane]=invalid;__syncthreads();
    for(int s=adaptive_threads/2;s>0;s/=2) {
        if(lane<s) {sums[lane]+=sums[lane+s];bad[lane]|=bad[lane+s];}
        __syncthreads();
    }
    if(lane==0) {result[0]=sums[0];result[1]=static_cast<Real>(bad[0]);}
}

__global__ __launch_bounds__(threads)
void statistics_partial(AstrDeviceView q, AstrDeviceView rho, AstrDeviceView vel, AstrDeviceView tmp,
                        AstrDeviceView grad, int nx, int ny, size_t count, Real re, Real tc,
                        Real tc1, Real *partial)
{
    __shared__ Real scratch[3][threads];
    const int lane = threadIdx.x;
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+lane;
    int i=0,j=0,k=0;
    Real e=0,z=0,d=0;
    if (id < count) {
        indices(id,nx,ny,i,j,k);
        // Match the original unique-node sum: 1:im, 1:jm, 1:km.
        if (i > 0 && j > 0 && k > 0) {
            const Real u=at(vel,i,j,k,1), v=at(vel,i,j,k,2), w=at(vel,i,j,k,3);
            const Real a=astr::load_value(grad,i,j,k,3,2)-astr::load_value(grad,i,j,k,2,3);
            const Real b=astr::load_value(grad,i,j,k,1,3)-astr::load_value(grad,i,j,k,3,1);
            const Real c=astr::load_value(grad,i,j,k,2,1)-astr::load_value(grad,i,j,k,1,2);
            e=at(rho,i,j,k)*(u*u+v*v+w*w);
            z=at(rho,i,j,k)*(a*a+b*b+c*c);
            const Real s11=astr::load_value(grad,i,j,k,1,1),s22=astr::load_value(grad,i,j,k,2,2),s33=astr::load_value(grad,i,j,k,3,3);
            const Real s12=Real(.5)*(astr::load_value(grad,i,j,k,1,2)+astr::load_value(grad,i,j,k,2,1));
            const Real s13=Real(.5)*(astr::load_value(grad,i,j,k,1,3)+astr::load_value(grad,i,j,k,3,1));
            const Real s23=Real(.5)*(astr::load_value(grad,i,j,k,2,3)+astr::load_value(grad,i,j,k,3,2));
            const Real div=s11+s22+s33,t=at(tmp,i,j,k);
            const Real mu=t*sqrt(t)*tc1/(t+tc)/re;
            d=Real(2.0)*mu*(s11*s11+s22*s22+s33*s33+Real(2.0)*(s12*s12+s13*s13+s23*s23)-div*div/Real(3.0));
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
        const Real value=id<count ? at(q,i,j,k,n+1) : 0;
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
void statistics_finish(const Real *partial, size_t blocks, Real *result)
{
    __shared__ Real scratch[threads];
    const int n=blockIdx.x,lane=threadIdx.x;
    Real value=n<3 ? 0 : (n<8 ? -INFINITY : INFINITY);
    for (size_t b=lane;b<blocks;b+=threads) {
        const Real next=partial[13*b+n];
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

__global__ __launch_bounds__(threads)
void channel_partial(AstrDeviceView q,AstrDeviceView vel,AstrDeviceView coords,AstrDeviceView tmp,
                     AstrDeviceView grad,int nx,int ny,int jm,size_t count,Real re,Real tc,Real tc1,
                     int lower,int upper,Real *partial)
{
    __shared__ Real scratch[3][threads];
    const int lane=threadIdx.x;
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+lane;
    Real mass=0,shear=0,w2=0;
    if(id<count) {
        int i,j,k;indices(id,nx,ny,i,j,k);
        if(i>0 && k>0) {
            if(j>0) {
                const Real dy=at(coords,i,j,k)-at(coords,i,j-1,k);
                mass=Real(.5)*(at(q,i,j,k,2)+at(q,i,j-1,k,2))*dy;
                const Real w=at(vel,i,j,k,3);w2=w*w;
            }
            if((j==0 && lower) || (j==jm && upper)) {
                const Real t=at(tmp,i,j,k),mu=t*sqrt(t)*tc1/(t+tc)/re;
                shear=(j==0 ? Real(1.0) : -Real(1.0))*mu*at(grad,i,j,k,1,2);
            }
        }
    }
    scratch[0][lane]=mass;scratch[1][lane]=shear;scratch[2][lane]=w2;
    __syncthreads();
    for(int s=threads/2;s>0;s/=2) {
        if(lane<s) for(int n=0;n<3;++n) scratch[n][lane]+=scratch[n][lane+s];
        __syncthreads();
    }
    if(lane==0) for(int n=0;n<3;++n) partial[13*blockIdx.x+n]=scratch[n][0];
    return;
}

__global__ __launch_bounds__(threads)
void flat_partial(AstrDeviceView q,AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView grad,AstrDeviceView dtmp,
    AstrDeviceView volumes,AstrDeviceView areas,int nx,int ny,size_t count,Real re,Real tc,Real tc1,
    Real prandtl,Real c5,int lower_wall,Real *partial)
{
    __shared__ Real scratch[5][threads];
    const int lane=threadIdx.x;
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+lane;
    Real values[5]={};
    if(id<count) {
        int i,j,k;indices(id,nx,ny,i,j,k);
        if(i>0 && j>0 && k>0) {
            const Real momentum=Real(.125)*(at(q,i-1,j-1,k,2)+at(q,i,j-1,k,2)+at(q,i-1,j,k,2)+at(q,i,j,k,2)
                +at(q,i-1,j-1,k-1,2)+at(q,i,j-1,k-1,2)+at(q,i-1,j,k-1,2)+at(q,i,j,k-1,2));
            values[0]=momentum*at(volumes,i-1,j-1,k-1);
            const Real w=at(vel,i,j,k,3);values[2]=w*w;
        }
        if(lower_wall && j==0 && i>0 && k>0) {
            Real shear[4],heat[4];
            const int ii[4]={i,i-1,i,i-1},kk[4]={k,k,k-1,k-1};
            for(int c=0;c<4;++c) {
                const Real t=at(tmp,ii[c],0,kk[c]),mu=t*sqrt(t)*tc1/(t+tc)/re;
                shear[c]=mu*astr::load_value(grad,ii[c],0,kk[c],1,2);
                heat[c]=(mu/prandtl)/c5*astr::load_value(dtmp,ii[c],0,kk[c],2);
            }
            const Real area=at(areas,i-1,0,k-1);
            values[1]=Real(.25)*area*(shear[0]+shear[1]+shear[2]+shear[3]);
            values[3]=Real(.25)*area*(heat[0]+heat[1]+heat[2]+heat[3]);values[4]=area;
        }
    }
    for(int n=0;n<5;++n) scratch[n][lane]=values[n];
    __syncthreads();
    for(int s=threads/2;s>0;s/=2) {
        if(lane<s) for(int n=0;n<5;++n) scratch[n][lane]+=scratch[n][lane+s];
        __syncthreads();
    }
    if(lane==0) for(int n=0;n<5;++n) partial[13*blockIdx.x+n]=scratch[n][0];
    return;
}

__global__ __launch_bounds__(threads)
void flat_finish(const Real *partial,size_t blocks,Real *result)
{
    __shared__ Real scratch[threads];
    const int n=blockIdx.x,lane=threadIdx.x;
    Real value=0;
    for(size_t b=lane;b<blocks;b+=threads) value+=partial[13*b+n];
    scratch[lane]=value;__syncthreads();
    for(int s=threads/2;s>0;s/=2) {
        if(lane<s) scratch[lane]+=scratch[lane+s];
        __syncthreads();
    }
    if(lane==0) result[n]=scratch[0];
    return;
}

__device__ void rates_at(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,AstrDeviceView q,
                        bool reacting,int i,int j,int k,int nondim,Real mach,Real gamma,Real gas,Real values[4],int *failure)
{
    Real sound,u,v,w;
    int thermo_status=0;
#ifndef ASTR_REAL32
    if(reacting) {
        double state[11],velocity[3]={},css;
        for(int c=0;c<11;++c) state[c]=at(q,i,j,k,c+1);
        thermo_status=astr::air5::frozen_sound(state,css,velocity);
        sound=css;u=velocity[0];v=velocity[1];w=velocity[2];
    } else
#endif
    {
        const Real temperature=at(tmp,i,j,k);
        sound=nondim ? sqrt(temperature)/mach : sqrt(gamma*gas*temperature);
        u=at(vel,i,j,k,1);v=at(vel,i,j,k,2);w=at(vel,i,j,k,3);
    }
    bool valid=thermo_status==0 && isfinite(sound) && sound>0;
    for (int axis=1;axis<=3;++axis) {
        const Real a=at(metric,i,j,k,axis,1),b=at(metric,i,j,k,axis,2),c=at(metric,i,j,k,axis,3);
        values[axis-1]=fabs(u*a+v*b+w*c)+sound*sqrt(a*a+b*b+c*c);
        valid=valid && isfinite(a) && isfinite(b) && isfinite(c) && isfinite(u) && isfinite(v)
            && isfinite(w) && isfinite(values[axis-1]) && values[axis-1]>=0;
    }
    values[3]=values[0]+values[1]+values[2];
    if (!valid || !isfinite(values[3])) atomicCAS(failure,0,2);
    return;
}

__global__ __launch_bounds__(threads)
void cfl_partial(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,AstrDeviceView q,bool reacting,int nx,int ny,size_t count,
                 int nondim,Real mach,Real gamma,Real gas,int gx,int gy,int ox,int oy,int oz,
                 Maximum *partial,int *failure)
{
    __shared__ Maximum scratch[4][threads];
    const int lane=threadIdx.x;
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+lane;
    Real values[4]={0,0,0,0};
    int64_t key=INT64_MAX;
    if (id<count) {
        int i,j,k;
        indices(id,nx,ny,i,j,k);
        rates_at(vel,tmp,metric,q,reacting,i,j,k,nondim,mach,gamma,gas,values,failure);
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
void cfl_finish(const Maximum *partial,size_t blocks,Real *values,int64_t *keys)
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
void cfl_plane(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,AstrDeviceView q,bool reacting,int nx,int nz,
               int nondim,Real mach,Real gamma,Real gas,Real *profile,int *failure)
{
    __shared__ Real scratch[4][threads];
    const int lane=threadIdx.x,j=blockIdx.x;
    Real best[4]={0,0,0,0};
    for (size_t id=lane;id<static_cast<size_t>(nx)*nz;id+=threads) {
        Real values[4];
        rates_at(vel,tmp,metric,q,reacting,id%nx,j,id/nx,nondim,mach,gamma,gas,values,failure);
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

extern "C" int astr_device_fluid_adaptive_energy_sum(void *handle,Real values[2])
{
    if(!handle||!values) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    const size_t count=static_cast<size_t>(s.upper[0])*s.upper[1]*s.upper[2];
    const size_t required=(count+adaptive_threads-1)/adaptive_threads;
    const int blocks=required==0 ? 1 : (required>256 ? 256 : static_cast<int>(required));
    if(!s.statistics_partial) {
        const size_t statistic_blocks=(s.fields[ASTR_DEVICE_RHS].view.stride[3]+threads-1)/threads;
        // The existing 13-channel storage also covers the 256-thread scalar partials.
        cudaError_t status=cudaMalloc(&s.statistics_partial,13*statistic_blocks*sizeof(Real));
        if(status!=cudaSuccess) return static_cast<int>(status);
        status=cudaMalloc(&s.statistics_values,13*sizeof(Real));
        if(status!=cudaSuccess) return static_cast<int>(status);
    }
    adaptive_energy_partial<<<blocks,adaptive_threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.upper[0],s.upper[1],count,s.statistics_partial);
    int status=astr::finish_launch(s);
    if(status!=0) return status;
    adaptive_energy_finish<<<1,adaptive_threads,0,s.stream>>>(s.statistics_partial,blocks,s.statistics_values);
    status=astr::finish_launch(s);
    if(status!=0) return status;
    const cudaError_t copied=cudaMemcpyAsync(values,s.statistics_values,2*sizeof(Real),cudaMemcpyDeviceToHost,s.stream);
    return copied==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(copied);
}

extern "C" int astr_device_fluid_statistics(void *handle,Real re,Real tc,Real tc1,Real values[13])
{
    if (!handle || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (!s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3],blocks=(count+threads-1)/threads;
    if (!s.statistics_partial) {
        cudaError_t err=cudaMalloc(&s.statistics_partial,13*blocks*sizeof(Real));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&s.statistics_values,13*sizeof(Real));
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    const AstrDeviceView diagnostic=s.fields[ASTR_DEVICE_DIAGNOSTIC_DVELOCITY].view;
    statistics_partial<<<blocks,threads,0,s.stream>>>(s.fields[0].view,s.fields[3].view,s.fields[4].view,
        s.fields[6].view,diagnostic.data?diagnostic:s.fields[8].view,
        s.upper[0]+1,s.upper[1]+1,count,re,tc,tc1,s.statistics_partial);
    int status=astr::finish_launch(s);
    if (status!=0) return status;
    statistics_finish<<<13,threads,0,s.stream>>>(s.statistics_partial,blocks,s.statistics_values);
    status=astr::finish_launch(s);
    if (status!=0) return status;
    const cudaError_t err=cudaMemcpyAsync(values,s.statistics_values,13*sizeof(Real),cudaMemcpyDeviceToHost,s.stream);
    return err==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(err);
}

extern "C" int astr_device_fluid_channel_statistics(void *handle,Real re,Real tc,Real tc1,
                                                    int lower,int upper,Real values[3])
{
    if(!handle || !values || lower<0 || lower>1 || upper<0 || upper>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.gradients_ready || !s.fields[ASTR_DEVICE_COORDINATE_Y].view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3],blocks=(count+threads-1)/threads;
    // Reuse the scalar-statistics workspace; its layout holds 13 entries/block.
    if(!s.statistics_partial) {
        cudaError_t status=cudaMalloc(&s.statistics_partial,13*blocks*sizeof(Real));
        if(status!=cudaSuccess) return static_cast<int>(status);
        status=cudaMalloc(&s.statistics_values,13*sizeof(Real));
        if(status!=cudaSuccess) return static_cast<int>(status);
    }
    channel_partial<<<blocks,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_COORDINATE_Y].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_DVELOCITY].view,
        s.upper[0]+1,s.upper[1]+1,s.upper[1],count,re,tc,tc1,lower,upper,s.statistics_partial);
    int status=astr::finish_launch(s);
    if(status!=0) return status;
    statistics_finish<<<3,threads,0,s.stream>>>(s.statistics_partial,blocks,s.statistics_values);
    status=astr::finish_launch(s);
    if(status!=0) return status;
    const cudaError_t copied=cudaMemcpyAsync(values,s.statistics_values,3*sizeof(Real),cudaMemcpyDeviceToHost,s.stream);
    return copied==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(copied);
}

extern "C" int astr_device_fluid_flat_geometry_prepare(void *handle,int lower_wall)
{
    if(!handle || (lower_wall!=0 && lower_wall!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    int upper[3]={s.upper[0]-1,s.upper[1]-1,s.upper[2]-1};
    if(!s.fields[ASTR_DEVICE_CELL_VOLUME].view.data) {
        const int status=astr::allocate_field(s.fields[ASTR_DEVICE_CELL_VOLUME],upper,0,3);
        if(status!=0) return status;
    }
    if(lower_wall && !s.fields[ASTR_DEVICE_WALL_CELL_AREA].view.data) {
        upper[1]=0;
        return astr::allocate_field(s.fields[ASTR_DEVICE_WALL_CELL_AREA],upper,0,3);
    }
    return 0;
}

extern "C" int astr_device_fluid_flat_statistics(void *handle,Real re,Real tc,Real tc1,
    Real prandtl,Real c5,int lower_wall,Real values[5])
{
    if(!handle || !values || (lower_wall!=0 && lower_wall!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.gradients_ready || !s.fields[ASTR_DEVICE_CELL_VOLUME].view.data ||
       (lower_wall && !s.fields[ASTR_DEVICE_WALL_CELL_AREA].view.data)) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3],blocks=(count+threads-1)/threads;
    if(!s.statistics_partial) {
        cudaError_t status=cudaMalloc(&s.statistics_partial,13*blocks*sizeof(Real));
        if(status!=cudaSuccess) return static_cast<int>(status);
        status=cudaMalloc(&s.statistics_values,13*sizeof(Real));
        if(status!=cudaSuccess) return static_cast<int>(status);
    }
    const AstrDeviceView diagnostic=s.fields[ASTR_DEVICE_DIAGNOSTIC_DVELOCITY].view;
    const AstrDeviceView thermal=s.fields[ASTR_DEVICE_DIAGNOSTIC_DTEMPERATURE].view;
    flat_partial<<<blocks,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,diagnostic.data?diagnostic:s.fields[ASTR_DEVICE_DVELOCITY].view,
        thermal.data?thermal:s.fields[ASTR_DEVICE_DTEMPERATURE].view,
        s.fields[ASTR_DEVICE_CELL_VOLUME].view,s.fields[ASTR_DEVICE_WALL_CELL_AREA].view,s.upper[0]+1,s.upper[1]+1,count,
        re,tc,tc1,prandtl,c5,lower_wall,s.statistics_partial);
    int status=astr::finish_launch(s);
    if(status!=0) return status;
    flat_finish<<<5,threads,0,s.stream>>>(s.statistics_partial,blocks,s.statistics_values);
    status=astr::finish_launch(s);
    if(status!=0) return status;
    const cudaError_t copied=cudaMemcpyAsync(values,s.statistics_values,5*sizeof(Real),cudaMemcpyDeviceToHost,s.stream);
    return copied==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(copied);
}

extern "C" int astr_device_fluid_cfl_diagnostics(void *handle,int nondim,Real mach,Real gamma,Real gas,
    const int global[3],const int origin[3],int ny,Real values[4],int64_t keys[4],Real *profile)
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
        err=cudaMalloc(&s.cfl_diagnostic_values,4*sizeof(Real));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&s.cfl_diagnostic_keys,4*sizeof(int64_t));
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    cfl_partial<<<blocks,threads,0,s.stream>>>(s.fields[4].view,s.fields[6].view,s.fields[7].view,
        s.fields[0].view,s.air5_ready,s.upper[0]+1,s.upper[1]+1,count,nondim,mach,gamma,gas,global[0],global[1],origin[0],origin[1],origin[2],
        static_cast<Maximum *>(s.cfl_records),s.fluid_failure);
    int status=astr::finish_launch(s);
    if (status!=0) return status;
    cfl_finish<<<1,threads,0,s.stream>>>(static_cast<const Maximum *>(s.cfl_records),blocks,
        s.cfl_diagnostic_values,s.cfl_diagnostic_keys);
    status=astr::finish_launch(s);
    if (status!=0) return status;
    if (ny>0) {
        if (!s.cfl_profile) {
            err=cudaMalloc(&s.cfl_profile,4*(s.upper[1]+1)*sizeof(Real));
            if (err!=cudaSuccess) return static_cast<int>(err);
        }
        cfl_plane<<<s.upper[1]+1,threads,0,s.stream>>>(s.fields[4].view,s.fields[6].view,s.fields[7].view,
            s.fields[0].view,s.air5_ready,s.upper[0]+1,s.upper[2]+1,nondim,mach,gamma,gas,s.cfl_profile,s.fluid_failure);
        status=astr::finish_launch(s);
        if (status!=0) return status;
        err=cudaMemcpyAsync(profile+4*origin[1],s.cfl_profile,4*(s.upper[1]+1)*sizeof(Real),
                            cudaMemcpyDeviceToHost,s.stream);
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    err=cudaMemcpyAsync(values,s.cfl_diagnostic_values,4*sizeof(Real),cudaMemcpyDeviceToHost,s.stream);
    if (err!=cudaSuccess) return static_cast<int>(err);
    err=cudaMemcpyAsync(keys,s.cfl_diagnostic_keys,4*sizeof(int64_t),cudaMemcpyDeviceToHost,s.stream);
    return err==cudaSuccess ? astr_device_fluid_check(handle) : static_cast<int>(err);
}
