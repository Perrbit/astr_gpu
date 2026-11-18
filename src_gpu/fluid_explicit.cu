#include <cmath>
#include "device_state_internal.cuh"
#include "chemistry_transport.h"
#include "fluid_explicit.h"
#include "chemistry_statistics.h"
#include "halo_transport.h"
#include "high_wall_math.cuh"
#include "sbp63_math.cuh"

namespace {
using astr::State;
using astr::at;
using Real=astr::numerical_real;
constexpr int threads = astr::block_threads;
constexpr Real inv60 = Real(1) / Real(60);
struct PhysicalEnds {
    int low[3], high[3], upper[3];
};
struct FluxBox { PhysicalEnds ends; int first[3],last[3],inner_first[3],inner_last[3]; bool wall_band=false; };

size_t physical_count(const State &state)
{
    return state.fields[ASTR_DEVICE_RHS].view.stride[3];
}

__device__ void point(size_t id, int nx, int ny, int &i, int &j, int &k)
{
    i = id % nx;
    j = (id / nx) % ny;
    k = id / (static_cast<size_t>(nx) * ny);
}

__global__ __launch_bounds__(threads)
void primitive_corners_kernel(AstrDeviceView rho,AstrDeviceView vel,AstrDeviceView prs,AstrDeviceView tmp,
                              int nx,int ny,size_t count,int halo,int im,int jm,int km)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id<count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        i-=halo;j-=halo;k-=halo;
        if ((i<0 || i>im)+(j<0 || j>jm)+(k<0 || k>km)>1) {
            at(rho,i,j,k)=0.0;at(prs,i,j,k)=0.0;at(tmp,i,j,k)=0.0;
            for (int c=1;c<=3;++c) at(vel,i,j,k,c)=0.0;
        }
    }
    return;
}

__global__ __launch_bounds__(threads)
void initialize_exterior_kernel(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView vel,AstrDeviceView prs,
    AstrDeviceView tmp,int nx,int ny,size_t count,int halo,PhysicalEnds ends)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int i,j,k;point(id,nx,ny,i,j,k);i-=halo;j-=halo;k-=halo;
        const int p[3]={i,j,k};int outside=0;bool physical=false;
        for(int a=0;a<3;++a) {
            outside+=(p[a]<0 || p[a]>ends.upper[a]);
            physical=physical || (p[a]<0 && ends.low[a]) || (p[a]>ends.upper[a] && ends.high[a]);
        }
        if(physical || outside>1) for(int c=1;c<=5;++c) at(q,i,j,k,c)=0.0;
        if(physical) {
            at(rho,i,j,k)=0.0;at(prs,i,j,k)=0.0;at(tmp,i,j,k)=0.0;
            for(int c=1;c<=3;++c) at(vel,i,j,k,c)=0.0;
        }
    }
    return;
}

template<bool physical>
__global__ __launch_bounds__(threads)
void primitive_kernel(AstrDeviceView q, AstrDeviceView rho, AstrDeviceView vel,
                      AstrDeviceView prs, AstrDeviceView tmp, int nx, int ny,
                      size_t count, int halo, int im, int jm, int km, Real c2, Real c6, int *failure,PhysicalEnds ends)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        i -= halo; j -= halo; k -= halo;
        if(physical && ((i<0 && ends.low[0]) || (i>im && ends.high[0]) ||
           (j<0 && ends.low[1]) || (j>jm && ends.high[1]) || (k<0 && ends.low[2]) || (k>km && ends.high[2]))) return;
        const int outside = (i < 0 || i > im)+(j < 0 || j > jm)+(k < 0 || k > km);
        if (outside > 1) return;
        const Real den = at(q, i, j, k, 1);
        const Real u = at(q, i, j, k, 2) / den;
        const Real v = at(q, i, j, k, 3) / den;
        const Real w = at(q, i, j, k, 4) / den;
        const Real ke = Real(.5) * den * (u*u + v*v + w*w);
        const Real pressure = (at(q, i, j, k, 5) - ke) / c6;
        const Real temperature = pressure / den * c2;
        at(rho, i, j, k) = den;
        at(vel, i, j, k, 1) = u;
        at(vel, i, j, k, 2) = v;
        at(vel, i, j, k, 3) = w;
        at(prs, i, j, k) = pressure;
        at(tmp, i, j, k) = temperature;
        if (!(isfinite(den) && den > Real(0) && isfinite(u) && isfinite(v) && isfinite(w)
              && isfinite(pressure) && pressure > Real(0) && isfinite(temperature) && temperature > Real(0)))
            atomicCAS(failure, 0, 1);
    }
    return;
}

__global__ __launch_bounds__(threads)
void conservative_kernel(AstrDeviceView q, AstrDeviceView rho, AstrDeviceView vel,
                         AstrDeviceView tmp, int nx, int ny, size_t count,
                         Real coefficient, int *failure)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        const Real den = at(rho, i, j, k);
        const Real u = at(vel, i, j, k, 1);
        const Real v = at(vel, i, j, k, 2);
        const Real w = at(vel, i, j, k, 3);
        const Real temperature = at(tmp, i, j, k);
        const Real ke = Real(.5) * (u*u + v*v + w*w);
        const Real energy = den * (temperature * coefficient + ke);
        at(q, i, j, k, 1) = den;
        at(q, i, j, k, 2) = den * u;
        at(q, i, j, k, 3) = den * v;
        at(q, i, j, k, 4) = den * w;
        at(q, i, j, k, 5) = energy;
        if (!(isfinite(den) && den > Real(0) && isfinite(u) && isfinite(v) && isfinite(w)
              && isfinite(temperature) && temperature > Real(0) && isfinite(energy)))
            atomicCAS(failure, 0, 1);
    }
    return;
}

__device__ Real derivative6(const AstrDeviceView &field, int i, int j, int k, int component, int axis)
{
    const size_t center = astr::offset(field, i, j, k, component);
    const size_t s = field.stride[axis];
    const Real *v = static_cast<const Real *>(field.data);
    return Real(.75) * (v[center+s] - v[center-s])
         - Real(.15) * (v[center+2*s] - v[center-2*s])
         + inv60 * (v[center+3*s] - v[center-3*s]);
}

__device__ Real derivative6_basic(const AstrDeviceView &field,int i,int j,int k,int component,
                                    int axis,PhysicalEnds ends)
{
    const int position[3]={i,j,k};
    const int p=position[axis],last=ends.upper[axis];
    const bool left=ends.low[axis] && p<3,right=ends.high[axis] && p>last-3;
    if(!left && !right) return derivative6(field,i,j,k,component,axis);
    const int depth=left ? p : last-p;
    const int step=left ? 1 : -1;
    int base[3]={i,j,k};base[axis]=left ? 0 : last;
    Real v[5];
    const int count=depth==2 ? 5 : 3;
    for(int r=0;r<count;++r) {
        int q[3]={base[0],base[1],base[2]};q[axis]+=step*r;
        v[r]=at(field,q[0],q[1],q[2],component);
    }
    if(depth==0) return step*(-Real(.5)*v[2]+Real(2)*v[1]-Real(1.5)*v[0]);
    if(depth==1) return step*Real(.5)*(v[2]-v[0]);
    return step*((Real(2)/Real(3))*(v[3]-v[1])-(Real(1)/Real(12))*(v[4]-v[0]));
}

template<bool physical,bool thermal=true>
__global__ __launch_bounds__(threads)
void gradient_kernel(AstrDeviceView vel, AstrDeviceView tmp, AstrDeviceView metric,
                     AstrDeviceView dvel, AstrDeviceView dtmp, int nx, int ny, size_t count,PhysicalEnds ends)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        for (int a = 1; a <= 3; ++a) {
            const Real dx = physical ? derivative6_basic(vel,i,j,k,a,0,ends) : derivative6(vel,i,j,k,a,0);
            const Real dy = physical ? derivative6_basic(vel,i,j,k,a,1,ends) : derivative6(vel,i,j,k,a,1);
            const Real dz = physical ? derivative6_basic(vel,i,j,k,a,2,ends) : derivative6(vel,i,j,k,a,2);
            for (int b = 1; b <= 3; ++b)
                astr::store_value(dvel,i,j,k,dx*at(metric,i,j,k,1,b)
                    + dy*at(metric,i,j,k,2,b) + dz*at(metric,i,j,k,3,b),a,b);
        }
        if(thermal) {
            const Real dx = physical ? derivative6_basic(tmp,i,j,k,1,0,ends) : derivative6(tmp,i,j,k,1,0);
            const Real dy = physical ? derivative6_basic(tmp,i,j,k,1,1,ends) : derivative6(tmp,i,j,k,1,1);
            const Real dz = physical ? derivative6_basic(tmp,i,j,k,1,2,ends) : derivative6(tmp,i,j,k,1,2);
            for (int b = 1; b <= 3; ++b)
                astr::store_value(dtmp,i,j,k,dx*at(metric,i,j,k,1,b)
                    + dy*at(metric,i,j,k,2,b) + dz*at(metric,i,j,k,3,b),b);
        }
    }
    return;
}

// Global y rows and weak boundary lifts; x/z retain the existing centered D6.
__global__ void sbp63_gradient_kernel(AstrDeviceView velocity,AstrDeviceView temperature,AstrDeviceView metric,
    AstrDeviceView dvel,AstrDeviceView dtmp,int nx,int ny,size_t count,int total,int start,double tw1,double tw2)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int i,j,k;point(id,nx,ny,i,j,k);
    const int globalj=start+j;
    double gu[3][3]={},gt[3]={};
    for(int axis=0;axis<3;++axis) {
        double du[3]={},dt=0;
        for(int offset=-5;offset<=5;++offset) {
            const double w=astr::sbp63::axis_weight(axis,globalj,offset,total);
            if(w==0) continue;
            int p[3]={i,j,k};p[axis]+=offset;
            for(int c=0;c<3;++c) du[c]+=w*at(velocity,p[0],p[1],p[2],c+1);
            dt+=w*at(temperature,p[0],p[1],p[2]);
        }
        for(int d=0;d<3;++d) {
            const double m=at(metric,i,j,k,axis+1,d+1);
            for(int c=0;c<3;++c) gu[c][d]+=du[c]*m;
            gt[d]+=dt*m;
        }
    }
    if(globalj==0 || globalj==total) {
        const double sign=globalj==0 ? -1 : 1,wall_t=globalj==0 ? tw1 : tw2;
        double normal[3],magnitude=0;
        for(int d=0;d<3;++d) {normal[d]=at(metric,i,j,k,2,d+1);magnitude+=normal[d]*normal[d];}
        magnitude=sqrt(magnitude);
        const double inverse_length=magnitude/astr::sbp63::norm(globalj,total);
        for(int d=0;d<3;++d) {
            normal[d]=sign*normal[d]/magnitude;
            for(int c=0;c<3;++c) gu[c][d]+=inverse_length*(-at(velocity,i,j,k,c+1))*normal[d];
            gt[d]+=inverse_length*(wall_t-at(temperature,i,j,k))*normal[d];
        }
    }
    for(int d=0;d<3;++d) {
        for(int c=0;c<3;++c) at(dvel,i,j,k,c+1,d+1)=gu[c][d];
        at(dtmp,i,j,k,d+1)=gt[d];
    }
}

// Existing boundary_high_order_body.inc rows, including the matched D6 seventh moment.
__constant__ double high_wall_derivative_rows[3][8]={
    {-156.0/60,423.0/60,-639.0/60,715.0/60,-540.0/60,261.0/60,-73.0/60,9.0/60},
    {-9.0/60,-84.0/60,171.0/60,-135.0/60,85.0/60,-36.0/60,9.0/60,-1.0/60},
    {1.0/60,-17.0/60,-56.0/60,115.0/60,-65.0/60,29.0/60,-8.0/60,1.0/60}};

__global__ __launch_bounds__(threads)
void high_wall_gradient_correction(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView metric,
    AstrDeviceView dvel,AstrDeviceView dtmp,PhysicalEnds ends,int axis,int n1,size_t plane)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=6*plane) return;
    const int side=id/(3*plane),row=(id/plane)%3;
    if(side ? !ends.high[axis] : !ends.low[axis]) return;
    const size_t node=id%plane;
    int p[3]={};p[axis]=side ? ends.upper[axis]-row : row;
    p[axis==0 ? 1 : 0]=node%n1;p[axis==2 ? 1 : 2]=node/n1;
    for(int c=1;c<=4;++c) {
        const AstrDeviceView field=c==4 ? tmp : vel;
        const int component=c==4 ? 1 : c;
        const double center=at(field,p[0],p[1],p[2],component);
        double derivative=0;
        for(int r=0;r<8;++r) {
            int q[3]={p[0],p[1],p[2]};q[axis]=side ? ends.upper[axis]-r : r;
            derivative+=high_wall_derivative_rows[row][r]*(at(field,q[0],q[1],q[2],component)-center);
        }
        if(side) derivative=-derivative;
        const double correction=derivative-derivative6_basic(field,p[0],p[1],p[2],component,axis,ends);
        for(int b=1;b<=3;++b) {
            const double increment=correction*at(metric,p[0],p[1],p[2],axis+1,b);
            if(c==4) at(dtmp,p[0],p[1],p[2],b)+=increment;
            else at(dvel,p[0],p[1],p[2],c,b)+=increment;
        }
    }
}

__global__ __launch_bounds__(threads)
void cfl_partial_kernel(AstrDeviceView vel, AstrDeviceView tmp, AstrDeviceView metric,
                        int nx, int ny, size_t count, int nondimensional,
                        Real mach, Real gamma, Real rgas, Real *partial, int *failure)
{
    __shared__ Real values[3][threads];
    const int lane = threadIdx.x;
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
    Real rates[3] = {};
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        const Real temperature = at(tmp, i, j, k);
        const Real sound = nondimensional ? sqrt(temperature)/mach : sqrt(gamma*rgas*temperature);
        bool valid = isfinite(sound) && sound > Real(0);
        Real velocity[3];
        for (int b = 0; b < 3; ++b) {
            velocity[b] = at(vel,i,j,k,b+1);
            valid = valid && isfinite(velocity[b]);
        }
        for (int a = 0; a < 3; ++a) {
            Real projection = 0, norm = 0;
            for (int b = 0; b < 3; ++b) {
                const Real m = at(metric,i,j,k,a+1,b+1);
                valid = valid && isfinite(m);
                projection += velocity[b] * m;
                norm += m * m;
            }
            rates[a] = fabs(projection) + sound * sqrt(norm);
            valid = valid && isfinite(rates[a]) && rates[a] >= Real(0);
        }
        if (!valid) atomicCAS(failure, 0, 2);
    }
    for (int a = 0; a < 3; ++a) values[a][lane] = rates[a];
    __syncthreads();
    for (int stride = threads/2; stride > 0; stride /= 2) {
        if (lane < stride)
            for (int a = 0; a < 3; ++a) values[a][lane] = fmax(values[a][lane], values[a][lane+stride]);
        __syncthreads();
    }
    if (lane == 0)
        for (int a = 0; a < 3; ++a) partial[3*blockIdx.x+a] = values[a][0];
    return;
}

__global__ __launch_bounds__(threads)
void cfl_finish_kernel(const Real *partial, Real *values, size_t blocks)
{
    __shared__ Real reduced[3][threads];
    const int lane = threadIdx.x;
    Real best[3] = {};
    for (size_t block = lane; block < blocks; block += blockDim.x)
        for (int a = 0; a < 3; ++a) best[a] = fmax(best[a], partial[3*block+a]);
    for (int a = 0; a < 3; ++a) reduced[a][lane] = best[a];
    __syncthreads();
    for (int stride = threads/2; stride > 0; stride /= 2) {
        if (lane < stride)
            for (int a = 0; a < 3; ++a) reduced[a][lane] = fmax(reduced[a][lane], reduced[a][lane+stride]);
        __syncthreads();
    }
    if (lane == 0)
        for (int a = 0; a < 3; ++a) values[a] = reduced[a][0];
    return;
}

__device__ Real convective_flux(AstrDeviceView q, AstrDeviceView vel, AstrDeviceView prs,
                                  AstrDeviceView metric, AstrDeviceView jac,
                                  int i, int j, int k, int component, int direction)
{
    const Real u = at(metric,i,j,k,direction,1)*at(vel,i,j,k,1)
                   + at(metric,i,j,k,direction,2)*at(vel,i,j,k,2)
                   + at(metric,i,j,k,direction,3)*at(vel,i,j,k,3);
    const Real value = at(q,i,j,k,component), pressure = at(prs,i,j,k);
    if (component == 1) return at(jac,i,j,k)*value*u;
    if (component == 5) return at(jac,i,j,k)*(value+pressure)*u;
    return at(jac,i,j,k)*(value*u+at(metric,i,j,k,direction,component-1)*pressure);
}

__device__ Real viscous_flux(AstrDeviceView stress, AstrDeviceView heat,
                               AstrDeviceView metric, AstrDeviceView jac,
                               int i, int j, int k, int component, int direction)
{
    const int tensor[3][3] = {{1,2,3},{2,4,5},{3,5,6}};
    Real flux[3];
    for (int b = 0; b < 3; ++b)
        flux[b] = component == 5 ? astr::load_value(heat,i,j,k,b+1)
                                : astr::load_value(stress,i,j,k,tensor[component-2][b]);
    return (flux[0]*at(metric,i,j,k,direction,1)+flux[1]*at(metric,i,j,k,direction,2)
           +flux[2]*at(metric,i,j,k,direction,3))*at(jac,i,j,k);
}

__global__ void sbp63_diffusion_kernel(AstrDeviceView velocity,AstrDeviceView temperature,AstrDeviceView metric,
    AstrDeviceView jac,AstrDeviceView stress,AstrDeviceView heat,AstrDeviceView dtmp,AstrDeviceView rhs,
    int nx,int ny,size_t count,int total,int start,double tw1,double tw2,double re,double pr,double c5,double tc,double tc1)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int i,j,k;point(id,nx,ny,i,j,k);
    const int globalj=start+j;
    double result[5]={};
    for(int axis=0;axis<3;++axis) for(int offset=-5;offset<=5;++offset) {
        const double w=astr::sbp63::axis_weight(axis,globalj,offset,total);
        if(w==0) continue;
        int p[3]={i,j,k};p[axis]+=offset;
        for(int c=2;c<=5;++c) result[c-1]+=w*viscous_flux(stress,heat,metric,jac,p[0],p[1],p[2],c,axis+1);
    }
    if(globalj==0 || globalj==total) {
        const double sign=globalj==0 ? -1 : 1,wall_t=globalj==0 ? tw1 : tw2;
        double normal[3],magnitude=0,u[3],vn=0;
        for(int d=0;d<3;++d) {normal[d]=at(metric,i,j,k,2,d+1);magnitude+=normal[d]*normal[d];u[d]=at(velocity,i,j,k,d+1);}
        magnitude=sqrt(magnitude);
        const double h=astr::sbp63::norm(globalj,total),inverse_length=magnitude/h,area=at(jac,i,j,k)*magnitude;
        for(int d=0;d<3;++d) {normal[d]=sign*normal[d]/magnitude;vn+=u[d]*normal[d];}
        const double t=at(temperature,i,j,k),mu=t*sqrt(t)*tc1/(t+tc)/re,kappa=(mu/pr)/c5;
        const int tensor[3][3]={{1,2,3},{2,4,5},{3,5,6}};
        for(int c=0;c<3;++c) {
            double raw=0;
            for(int d=0;d<3;++d) raw+=astr::load_value(stress,i,j,k,tensor[c][d])*normal[d];
            const double wall=raw-mu*inverse_length*(u[c]+normal[c]*vn/3);
            result[c+1]+=area*(wall-raw)/h;
        }
        double heat_gradient=0,raw=0;
        for(int d=0;d<3;++d) {heat_gradient+=at(dtmp,i,j,k,d+1)*normal[d];raw+=astr::load_value(heat,i,j,k,d+1)*normal[d];}
        const double wall=kappa*heat_gradient-kappa*inverse_length*(t-wall_t);
        result[4]+=area*(wall-raw)/h;
    }
    for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)+=result[c];
}

template<bool viscous,bool physical=false>
__global__ __launch_bounds__(threads)
void divergence_kernel(AstrDeviceView q, AstrDeviceView vel, AstrDeviceView prs,
                       AstrDeviceView metric, AstrDeviceView jac, AstrDeviceView stress,
                       AstrDeviceView heat, AstrDeviceView rhs, int nx, int ny, size_t count, int axis,FluxBox box)
{
    const int lane = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+lane;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const int node[3]={i,j,k};
        if(physical) {
            // CPU convection uses the common body box; diffusion owns each axis separately.
            if(viscous) {
                if(node[axis]<box.first[axis] || node[axis]>box.last[axis]) return;
            } else if(i<box.first[0] || i>box.last[0] || j<box.first[1] || j>box.last[1] ||
                      k<box.first[2] || k>box.last[2]) return;
        }
        if(physical && box.wall_band && i>=box.inner_first[0] && i<=box.inner_last[0] &&
           j>=box.inner_first[1] && j<=box.inner_last[1] && k>=box.inner_first[2] && k<=box.inner_last[2]) return;
        for (int c = viscous ? 2 : 1; c <= 5; ++c) {
            const bool left=physical && box.ends.low[axis] && node[axis]<3;
            const bool right=physical && box.ends.high[axis] && node[axis]>box.ends.upper[axis]-3;
            Real df;
            if(left || right) {
                const int step=left ? 1 : -1;
                const int depth=left ? node[axis] : box.ends.upper[axis]-node[axis];
                Real v[5];
                for(int r=0;r<(depth==2 ? 5 : 3);++r) {
                    int p[3]={i,j,k};p[axis]=(left ? 0 : box.ends.upper[axis])+step*r;
                    v[r]=viscous ? viscous_flux(stress,heat,metric,jac,p[0],p[1],p[2],c,axis+1)
                        : convective_flux(q,vel,prs,metric,jac,p[0],p[1],p[2],c,axis+1);
                }
                if(depth==0) df=step*(-Real(.5)*v[2]+Real(2)*v[1]-Real(1.5)*v[0]);
                else if(depth==1) df=step*Real(.5)*(v[2]-v[0]);
                else df=step*((Real(2)/Real(3))*(v[3]-v[1])-(Real(1)/Real(12))*(v[4]-v[0]));
            } else {
            Real plus[3],minus[3];
            for (int r = 1; r <= 3; ++r) {
                int p[3] = {i,j,k}, m[3] = {i,j,k};
                p[axis] += r; m[axis] -= r;
                plus[r-1] = viscous
                    ? viscous_flux(stress,heat,metric,jac,p[0],p[1],p[2],c,axis+1)
                    : convective_flux(q,vel,prs,metric,jac,p[0],p[1],p[2],c,axis+1);
                minus[r-1] = viscous
                    ? viscous_flux(stress,heat,metric,jac,m[0],m[1],m[2],c,axis+1)
                    : convective_flux(q,vel,prs,metric,jac,m[0],m[1],m[2],c,axis+1);
            }
            df = Real(.75)*(plus[0]-minus[0])-Real(.15)*(plus[1]-minus[1])
                            +inv60*(plus[2]-minus[2]);
            }
            if (viscous) at(rhs,i,j,k,c) += df;
            else at(rhs,i,j,k,c) -= df;
        }
    }
    return;
}

__constant__ double high_wall_pressure_ghost[3][6]={
    {21.,-35./2,35./3,-21./4,7./5,-1./6},
    {224.,-210.,448./3,-70.,96./5,-7./3},
    {1134.,-1134.,840.,-405.,567./5,-14.}};

__device__ double high_wall_pressure_derivative(const double p[8],int row)
{
    const double weights[7]={-1./60,9./60,-45./60,0,45./60,-9./60,1./60};
    double value=0;
    for(int offset=-3;offset<=3;++offset) {
        const int j=row+offset;
        double delta;
        if(j<0) {
            double ghost_delta=0;
            for(int s=1;s<=6;++s) ghost_delta+=high_wall_pressure_ghost[-j-1][s-1]*(p[s]-p[0]);
            delta=(p[0]-p[row])+ghost_delta;
        } else delta=p[j]-p[row];
        value+=weights[offset+3]*delta;
    }
    return value;
}

__device__ double high_wall_adiabatic_pressure_derivative(const double p[8],int row)
{
    const double weights[2][5]={{-1.0/12,1,-1.0/3,1.0/12,-1.0/100},
        {-1,1.0/6,2.0/3,-1.0/8,1.0/75}};
    double value=0;
    if(row>0) for(int j=1;j<=5;++j) value+=weights[row-1][j-1]*(p[j]-p[0]);
    return value;
}

template<bool viscous>
__global__ __launch_bounds__(threads)
void high_wall_flux_correction(AstrDeviceView q,AstrDeviceView vel,AstrDeviceView prs,
    AstrDeviceView metric,AstrDeviceView jac,AstrDeviceView stress,AstrDeviceView heat,AstrDeviceView rhs,
    FluxBox box,int axis,int n1,size_t plane,bool adiabatic)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=6*plane) return;
    const int side=id/(3*plane),row=(id/plane)%3,sign=side ? -1 : 1;
    if(side ? !box.ends.high[axis] : !box.ends.low[axis]) return;
    const size_t node=id%plane;
    int p[3]={};p[axis]=side ? box.ends.upper[axis]-row : row;
    p[axis==0 ? 1 : 0]=node%n1;p[axis==2 ? 1 : 2]=node/n1;
    if(viscous) {
        if(p[axis]<box.first[axis] || p[axis]>box.last[axis]) return;
    } else for(int a=0;a<3;++a) if(p[a]<box.first[a] || p[a]>box.last[a]) return;
    double pressure[8];
    if(!viscous) for(int r=0;r<8;++r) {
        int v[3]={p[0],p[1],p[2]};v[axis]=side ? box.ends.upper[axis]-r : r;
        pressure[r]=at(jac,v[0],v[1],v[2])*(at(metric,v[0],v[1],v[2],axis+1,axis+1)*at(prs,v[0],v[1],v[2]));
    }
    for(int c=viscous ? 2 : 1;c<=5;++c) {
        double flux[8],high_value=0;
        for(int r=0;r<8;++r) {
            int v[3]={p[0],p[1],p[2]};v[axis]=side ? box.ends.upper[axis]-r : r;
            flux[r]=viscous ? viscous_flux(stress,heat,metric,jac,v[0],v[1],v[2],c,axis+1)
                : convective_flux(q,vel,prs,metric,jac,v[0],v[1],v[2],c,axis+1);
        }
        for(int r=0;r<8;++r) high_value+=high_wall_derivative_rows[row][r]*(flux[r]-flux[row]);
        if(!viscous && c==axis+2) {
            double pressure_basic=0;
            for(int r=0;r<8;++r) pressure_basic+=high_wall_derivative_rows[row][r]*(pressure[r]-pressure[row]);
            high_value+=(adiabatic ? high_wall_adiabatic_pressure_derivative(pressure,row) :
                high_wall_pressure_derivative(pressure,row))-pressure_basic;
        }
        const double low_value=row==0 ? -.5*flux[2]+2*flux[1]-1.5*flux[0]
            : (row==1 ? .5*(flux[2]-flux[0]) : (2./3)*(flux[3]-flux[1])-(1./12)*(flux[4]-flux[0]));
        const double correction=sign*(high_value-low_value);
        if(viscous) at(rhs,p[0],p[1],p[2],c)+=correction;
        else at(rhs,p[0],p[1],p[2],c)-=correction;
    }
}

__global__ void high_wall_rhs(AstrDeviceView q,AstrDeviceView metric,AstrDeviceView jac,AstrDeviceView rhs,
    PhysicalEnds ends,int axis,int n1,size_t plane,double lower_t,double upper_t,double c2,double c6,int kind)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=2*plane) return;
    const int side=id/plane,sign=side ? -1 : 1;
    if(side ? !ends.high[axis] : !ends.low[axis]) return;
    const size_t node=id%plane;
    int p[3]={};p[axis]=side ? ends.upper[axis] : 0;
    p[axis==0 ? 1 : 0]=node%n1;p[axis==2 ? 1 : 2]=node/n1;
    double flux[8],derivative=0;
    for(int r=0;r<8;++r) {
        int v[3]={p[0],p[1],p[2]};v[axis]=side ? ends.upper[axis]-r : r;
        flux[r]=at(jac,v[0],v[1],v[2])*at(q,v[0],v[1],v[2],axis+2)*at(metric,v[0],v[1],v[2],axis+1,axis+1);
    }
    for(int r=0;r<8;++r) derivative+=high_wall_derivative_rows[0][r]*(flux[r]-flux[0]);
    double density_rate=at(rhs,p[0],p[1],p[2],1)-sign*derivative;
    if(kind==421) {
        const double weights[3]={.75,-.15,1.0/60};
        for(int a=0;a<3;a+=2) {
            double divergence=0;
            for(int offset=1;offset<=3;++offset) {
                int v[3]={p[0],p[1],p[2]};v[a]+=offset;
                const double forward=at(jac,v[0],v[1],v[2])*at(q,v[0],v[1],v[2],a+2)*at(metric,v[0],v[1],v[2],a+1,a+1);
                v[a]-=2*offset;
                const double backward=at(jac,v[0],v[1],v[2])*at(q,v[0],v[1],v[2],a+2)*at(metric,v[0],v[1],v[2],a+1,a+1);
                divergence+=weights[offset-1]*(forward-backward);
            }
            density_rate-=divergence;
        }
    }
    double energy_rate=density_rate*(side ? upper_t : lower_t)/c2*c6;
    double velocity[3]={},velocity_rate[3]={};
    const double den=at(q,p[0],p[1],p[2],1),volume=at(jac,p[0],p[1],p[2]);
    if(kind!=41) {
        double wall[5],thermal[7],tangent[2][7];
        for(int c=0;c<5;++c) wall[c]=at(q,p[0],p[1],p[2],c+1);
        const double temperature=astr::high_wall::temperature(wall,c2,c6);
        for(int s=1;s<=7;++s) {
            int v[3]={p[0],p[1],p[2]};v[axis]=side ? ends.upper[axis]-s : s;
            double state[5],rate[5];
            for(int c=0;c<5;++c) {
                state[c]=at(q,v[0],v[1],v[2],c+1);
                rate[c]=at(rhs,v[0],v[1],v[2],c+1)/at(jac,v[0],v[1],v[2]);
            }
            thermal[s-1]=astr::high_wall::temperature_rate(state,rate,c2,c6);
            if(kind==421) for(int a=0;a<2;++a) {
                const int c=2*a+1;
                tangent[a][s-1]=(rate[c]-state[c]/state[0]*rate[0])/state[0];
            }
        }
        energy_rate=(temperature*density_rate+den*astr::high_wall::neumann7(thermal)*volume)*c6/c2;
        if(kind==421) {
            for(int c=0;c<3;++c) velocity[c]=wall[c+1]/den;
            velocity_rate[0]=astr::high_wall::neumann7(tangent[0]);
            velocity_rate[2]=astr::high_wall::neumann7(tangent[1]);
            energy_rate+=.5*(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2])*density_rate+
                den*(velocity[0]*velocity_rate[0]+velocity[1]*velocity_rate[1]+velocity[2]*velocity_rate[2])*volume;
        }
    }
    at(rhs,p[0],p[1],p[2],1)=density_rate;
    for(int c=0;c<3;++c) at(rhs,p[0],p[1],p[2],c+2)=velocity[c]*density_rate+den*velocity_rate[c]*volume;
    at(rhs,p[0],p[1],p[2],axis+2)=0;
    at(rhs,p[0],p[1],p[2],5)=energy_rate;
}

__global__ __launch_bounds__(threads)
void stress_kernel(AstrDeviceView vel, AstrDeviceView tmp, AstrDeviceView dvel,
                   AstrDeviceView dtmp, AstrDeviceView stress, AstrDeviceView heat,
                   int nx, int ny, size_t count, Real reynolds, Real prandtl,
                   Real c5, Real tc, Real tc1)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const Real t = at(tmp,i,j,k);
        const Real mu = t*sqrt(t)*tc1/(t+tc)/reynolds, twice = Real(2)*mu;
        const Real hcc = (mu/prandtl)/c5;
        const Real s11 = at(dvel,i,j,k,1,1), s22 = at(dvel,i,j,k,2,2), s33 = at(dvel,i,j,k,3,3);
        const Real skk = (s11+s22+s33)/Real(3);
        const Real tau[6] = {twice*(s11-skk),
            twice*(Real(.5)*(at(dvel,i,j,k,1,2)+at(dvel,i,j,k,2,1))),
            twice*(Real(.5)*(at(dvel,i,j,k,1,3)+at(dvel,i,j,k,3,1))), twice*(s22-skk),
            twice*(Real(.5)*(at(dvel,i,j,k,2,3)+at(dvel,i,j,k,3,2))), twice*(s33-skk)};
        for (int c = 0; c < 6; ++c) astr::store_value(stress,i,j,k,tau[c],c+1);
        const Real u = at(vel,i,j,k,1), v = at(vel,i,j,k,2), w = at(vel,i,j,k,3);
        astr::store_value(heat,i,j,k,hcc*at(dtmp,i,j,k,1)+tau[0]*u+tau[1]*v+tau[2]*w,1);
        astr::store_value(heat,i,j,k,hcc*at(dtmp,i,j,k,2)+tau[1]*u+tau[3]*v+tau[4]*w,2);
        astr::store_value(heat,i,j,k,hcc*at(dtmp,i,j,k,3)+tau[2]*u+tau[4]*v+tau[5]*w,3);
    }
    return;
}

__global__ __launch_bounds__(threads)
void uniform_force_kernel(AstrDeviceView q,AstrDeviceView jac,AstrDeviceView rhs,int nx,int ny,size_t count,
                          Real fx,Real fy,Real fz)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int i,j,k;point(id,nx,ny,i,j,k);
        const Real volume=at(jac,i,j,k),density=at(q,i,j,k,1);
        at(rhs,i,j,k,2)+=fx*volume;
        at(rhs,i,j,k,3)+=fy*volume;
        at(rhs,i,j,k,4)+=fz*volume;
        at(rhs,i,j,k,5)+=(fx*(at(q,i,j,k,2)/density)+fy*(at(q,i,j,k,3)/density)
                          +fz*(at(q,i,j,k,4)/density))*volume;
    }
    return;
}

__global__ __launch_bounds__(threads)
void rti_source_kernel(AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView jac,AstrDeviceView rhs,
                       int nx,int ny,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int i,j,k;point(id,nx,ny,i,j,k);
        const Real den=at(rho,i,j,k),volume=at(jac,i,j,k);
        at(rhs,i,j,k,3)+=den*volume;
        at(rhs,i,j,k,5)+=den*at(velocity,i,j,k,2)*volume;
    }
    return;
}

template<bool viscous> int divergence(State &s)
{
    const size_t count = physical_count(s);
    for (int axis = 0; axis < 3; ++axis) {
        const dim3 block = axis == 0 ? dim3(512,1,1) : (axis == 1 ? dim3(32,16,1) : dim3(64,1,8));
        divergence_kernel<viscous><<<(count+threads-1)/threads,block,0,s.stream>>>(
            s.fields[0].view,s.fields[4].view,s.fields[5].view,s.fields[7].view,
            s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_STRESS].view,
            s.fields[ASTR_DEVICE_HEAT_FLUX].view,s.fields[1].view,
            s.upper[0]+1,s.upper[1]+1,count,axis,FluxBox{});
        const int status = astr::finish_launch(s);
        if (status != 0) return status;
    }
    return 0;
}

__device__ Real filter_basic_end(AstrDeviceView source,int i,int j,int k,int component,int axis,int depth,int last,bool left)
{
    if(depth==0) return at(source,i,j,k,component);
    Real value=0;
    if(depth<=2) {
        const Real weights[2][7]={{Real(1)/64,Real(29)/32,Real(15)/64,-Real(5)/16,Real(15)/64,-Real(3)/32,Real(1)/64},
            {-Real(1)/64,Real(3)/32,Real(49)/64,Real(5)/16,-Real(15)/64,Real(3)/32,-Real(1)/64}};
        for(int m=0;m<=6;++m) {
            int p[3]={i,j,k};p[axis]=left ? m : last-m;
            value+=weights[depth-1][m]*at(source,p[0],p[1],p[2],component);
        }
    } else {
        const Real six[4]={Real(11)/32,Real(15)/64,-Real(3)/32,Real(1)/64};
        const Real eight[5]={Real(93)/256,Real(7)/32,-Real(7)/64,Real(1)/32,-Real(1)/256};
        const size_t center=astr::offset(source,i,j,k,component),stride=source.stride[axis];
        const Real *v=static_cast<const Real *>(source.data);
        for(int m=0;m<=depth;++m) value+=(depth==3 ? six[m] : eight[m])*(v[center-m*stride]+v[center+m*stride]);
    }
    return value;
}

__device__ double air5_linear_value(AstrDeviceView source,int i,int j,int k,int c,int axis,
                                    bool end,int depth,bool left)
{
    const double *v=static_cast<const double *>(source.data);
    const size_t stride=source.stride[axis];
    const double *p=v+astr::offset(source,i,j,k,c);
    if(end && depth==0) return *p;
    // Preserve the FP64 contraction of the original NVHPC directional kernels.
    if(end && depth<=2) {
        const ptrdiff_t step=left ? ptrdiff_t(stride) : -ptrdiff_t(stride);
        const double *edge=p-depth*step;
        if(depth==1) {
            double value=fma(1.0/64,edge[0],(29.0/32)*edge[step]);
            value=fma(15.0/64,edge[2*step],value);
            value=fma(-5.0/16,edge[3*step],value);
            value=fma(15.0/64,edge[4*step],value);
            value=fma(-3.0/32,edge[5*step],value);
            return fma(1.0/64,edge[6*step],value);
        }
        const double near=fma(3.0/32,edge[step],(-1.0/64)*edge[0]);
        const double main=fma(49.0/64,edge[2*step],(5.0/16)*edge[3*step]);
        const double middle=fma(-15.0/64,edge[4*step],near+main);
        return fma(-1.0/64,edge[6*step],fma(3.0/32,edge[5*step],middle));
    }
    const ptrdiff_t d=stride;
    const double pair1=p[d]+p[-d],pair2=p[2*d]+p[-2*d],pair3=p[3*d]+p[-3*d];
    if(end && depth==3)
        return fma(11.0/16,*p,(15.0/64)*pair1)+fma(1.0/64,pair3,(-3.0/32)*pair2);
    const double pair4=p[4*d]+p[-4*d];
    if(end)
        return fma(93.0/128,*p,(7.0/32)*pair1)
            +fma(-1.0/256,pair4,fma(1.0/32,pair3,(-7.0/64)*pair2));
    double value=fma(193.0/256,*p,(105.0/512)*pair1);
    value=fma(-15.0/128,pair2,value);
    value=fma(45.0/1024,pair3,value);
    value=fma(-5.0/512,pair4,value);
    return fma(1.0/1024,p[5*d]+p[-5*d],value);
}

template<int components,bool physical=false,bool air5=false>
__global__ __launch_bounds__(threads)
void filter_kernel(AstrDeviceView source, AstrDeviceView destination, int nx, int ny, size_t count, int axis,
                   int source_component,int destination_component,int lower,int upper,int last)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const size_t stride = source.stride[axis];
        const Real *v = static_cast<const Real *>(source.data);
        const int node[3]={i,j,k};
        const bool left=physical && lower && node[axis]<5;
        const bool right=physical && upper && node[axis]>last-5;
        for (int c = 0; c < components; ++c) {
            if(air5) {
                const double value=air5_linear_value(source,i,j,k,source_component+c,axis,left || right,
                    left ? node[axis] : last-node[axis],left);
                static_cast<double *>(destination.data)[astr::offset(destination,i,j,k,destination_component+c)]=value;
                continue;
            }
            if(left || right) {
                at(destination,i,j,k,destination_component+c)=filter_basic_end(source,i,j,k,source_component+c,axis,
                    left ? node[axis] : last-node[axis],last,left);
                continue;
            }
            const size_t p = astr::offset(source,i,j,k,source_component+c);
            at(destination,i,j,k,destination_component+c) = (Real(193)/Real(256))*v[p]+(Real(105)/Real(512))*(v[p+stride]+v[p-stride])
                -(Real(15)/Real(128))*(v[p+2*stride]+v[p-2*stride])
                +(Real(45)/Real(1024))*(v[p+3*stride]+v[p-3*stride])
                -(Real(5)/Real(512))*(v[p+4*stride]+v[p-4*stride])
                +(Real(1)/Real(1024))*(v[p+5*stride]+v[p-5*stride]);
        }
    }
    return;
}

__constant__ double high_wall_filter_a0[2][4]={{323.0/2898.0,701.0/2490.0,827.0/2383.0,2413.0/3032.0},
    {544.0/3743.0,-152.0/2883.0,1582.0/2067.0,1083.0/580.0}};
__constant__ double high_wall_filter_a2[2][4]={{-72.0/1639.0,1461.0/3958.0,174.0/3949.0,2413.0/3032.0},
    {-1093.0/2657.0,822.0/1711.0,121.0/3956.0,1083.0/580.0}};
__constant__ double wall_filter_difference[7]={1.0/64.0,-6.0/64.0,15.0/64.0,-20.0/64.0,
    15.0/64.0,-6.0/64.0,1.0/64.0};

template<bool air5=false>
__global__ __launch_bounds__(threads)
void high_wall_filter_rows(AstrDeviceView source,AstrDeviceView destination,int axis,int last,
    int lower,int upper,int source_component,int destination_component,int components,size_t plane,int width,int family)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=8*plane) return;
    const int side=id/(4*plane),row=(id/plane)%4+1;
    if((side==0 && !lower) || (side==1 && !upper)) return;
    if(side==1 && lower && last-row<=4) return;
    const size_t node=id%plane;
    int p[3]={};p[axis==0 ? 1 : 0]=node%width;p[axis==2 ? 1 : 2]=node/width;
    p[axis]=side==0 ? row : last-row;
    for(int c=0;c<components;++c) {
        const double center=at(source,p[0],p[1],p[2],source_component+c);
        double value=center;
        for(int j=0;j<=8;++j) {
            if(j==row) continue;
            const double b0=j<=6 ? wall_filter_difference[j] : 0;
            const double b1=j>=1 && j<=7 ? wall_filter_difference[j-1] : 0;
            const double b2=j>=2 ? wall_filter_difference[j-2] : 0;
            const double delta=high_wall_filter_a0[family][row-1]*(b0+b1)+high_wall_filter_a2[family][row-1]*(b1+b2)
                +(row%2 ? -b1 : b1);
            int q[3]={p[0],p[1],p[2]};q[axis]=side==0 ? j : last-j;
            const double diff=at(source,q[0],q[1],q[2],source_component+c)-center;
            value=air5 ? fma(delta,diff,value) : value+delta*diff;
        }
        at(destination,p[0],p[1],p[2],destination_component+c)=value;
    }
}

__global__ void air5_legacy_y_ghosts(AstrDeviceView q,AstrDeviceView work,int nx,int ny,int nz,int halo,
                                    int component,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count)return;
    const int i=id%nx,k=(id/nx)%nz,c=id/(static_cast<size_t>(nx)*nz)+1;
    for(int h=1;h<=halo;++h) {
        at(work,i,-h,k,c)=at(q,i,-h,k,component ? component : c);
        at(work,i,ny-1+h,k,c)=at(q,i,ny-1+h,k,component ? component : c);
    }
}

template<int components>
__global__ __launch_bounds__(threads)
void physical_copy_kernel(AstrDeviceView source, AstrDeviceView destination, int nx, int ny, size_t count,
                          int source_component,int destination_component)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        for (int c = 0; c < components; ++c)
            at(destination,i,j,k,destination_component+c) = at(source,i,j,k,source_component+c);
    }
    return;
}

template<bool save>
__global__ __launch_bounds__(threads)
void rk_kernel(AstrDeviceView q, AstrDeviceView saved, AstrDeviceView rhs, AstrDeviceView jac,
               int nx, int ny, size_t count, Real dt, Real a, Real b, Real c)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const Real volume = at(jac,i,j,k);
        for (int m = 1; m <= 5; ++m) {
            if (save) at(saved,i,j,k,m) = at(q,i,j,k,m)*volume;
            else at(q,i,j,k,m) = (a*at(saved,i,j,k,m)+b*at(q,i,j,k,m)*volume
                                +c*at(rhs,i,j,k,m)*dt)/volume;
        }
    }
    return;
}

template<bool save> int rk(State &s, Real dt, Real a, Real b, Real c)
{
    const size_t count = physical_count(s);
    rk_kernel<save><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[0].view,s.fields[2].view,s.fields[1].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.upper[0]+1,s.upper[1]+1,count,dt,a,b,c);
    return astr::finish_launch(s);
}
}

extern "C" int astr_device_fluid_filter_prepare(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (s.halo < 5 || s.fields[ASTR_DEVICE_FILTER_WORK].view.data) return ASTR_DEVICE_INVALID_ARGUMENT;
    return astr::allocate_field(s.fields[ASTR_DEVICE_FILTER_WORK],s.upper,s.halo,4,5);
}

extern "C" int astr_device_fluid_filter_prepare_scalar(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (s.halo < 5 || s.fields[ASTR_DEVICE_FILTER_WORK].view.data) return ASTR_DEVICE_INVALID_ARGUMENT;
    return astr::allocate_field(s.fields[ASTR_DEVICE_FILTER_WORK],s.upper,s.halo,3);
}

extern "C" int astr_device_fluid_filter_workspace_components(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    const AstrDeviceView &work = static_cast<State *>(handle)->fields[ASTR_DEVICE_FILTER_WORK].view;
    return work.data ? work.extent[3] : ASTR_DEVICE_NOT_ALLOCATED;
}

extern "C" int astr_device_air5_legacy_linear_prepare(void *handle,int scalar)
{
    if(!handle || (scalar!=0 && scalar!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    auto &work=s.fields[ASTR_DEVICE_FILTER_WORK];
    if(!s.air5_ready || s.halo<5 || work.view.data) return ASTR_DEVICE_INVALID_ARGUMENT;
    return astr::allocate_field(work,s.upper,s.halo,scalar ? 3 : 4,scalar ? 1 : 11,1,sizeof(double));
}

extern "C" int astr_device_air5_legacy_linear_axis(void *handle,int axis,int component,int lower,int upper)
{
    if(!handle || axis<0 || axis>2 || component<0 || component>11 ||
       lower<0 || lower>1 || upper<0 || upper>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    if(!s.air5_ready || !work.data || work.extent[3]!=(component ? 1 : 11))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=physical_count(s),grid=(count+threads-1)/threads;
    const int nx=s.upper[0]+1,ny=s.upper[1]+1,last=s.upper[axis];
    const AstrDeviceView src=axis==1 ? work : q,dst=axis==1 ? q : work;
    if(component)
        filter_kernel<1,true,true><<<grid,threads,0,s.stream>>>(src,dst,nx,ny,count,axis,
            axis==1 ? 1 : component,axis==1 ? component : 1,lower,upper,last);
    else
        filter_kernel<11,true,true><<<grid,threads,0,s.stream>>>(src,dst,nx,ny,count,axis,1,1,lower,upper,last);
    return astr::finish_launch(s);
}

extern "C" int astr_device_air5_legacy_linear_wall(void *handle,int axis,int component,int lower,int upper,int family)
{
    if(family<0 || family>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=astr_device_air5_legacy_linear_axis(handle,axis,component,lower,upper);
    if(status || (!lower && !upper)) return status;
    State &s=*static_cast<State *>(handle);
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    const int width=s.upper[axis==0 ? 1 : 0]+1;
    const size_t plane=physical_count(s)/(s.upper[axis]+1);
    high_wall_filter_rows<true><<<(8*plane+threads-1)/threads,threads,0,s.stream>>>(
        axis==1 ? work : q,axis==1 ? q : work,axis,s.upper[axis],lower,upper,
        axis==1 || !component ? 1 : component,axis==1 && component ? component : 1,
        component ? 1 : 11,plane,width,family);
    return astr::finish_launch(s);
}

extern "C" int astr_device_air5_legacy_linear_commit(void *handle,int component)
{
    if(!handle || component<0 || component>11) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    if(!s.air5_ready || !work.data || work.extent[3]!=(component ? 1 : 11))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=physical_count(s),grid=(count+threads-1)/threads;
    const int nx=s.upper[0]+1,ny=s.upper[1]+1;
    if(component)physical_copy_kernel<1><<<grid,threads,0,s.stream>>>(work,q,nx,ny,count,1,component);
    else physical_copy_kernel<11><<<grid,threads,0,s.stream>>>(work,q,nx,ny,count,1,1);
    return astr::finish_launch(s);
}

extern "C" int astr_device_air5_legacy_linear_y_ghosts(void *handle,int component)
{
    State &s=*static_cast<State *>(handle);
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    const int nx=s.upper[0]+1,ny=s.upper[1]+1,nz=s.upper[2]+1;
    const size_t count=static_cast<size_t>(nx)*nz*work.extent[3];
    air5_legacy_y_ghosts<<<(count+threads-1)/threads,threads,0,s.stream>>>(
        q,work,nx,ny,nz,s.halo,component,count);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_component_axis(void *handle,int axis,int component)
{
    if (!handle || axis<0 || axis>2 || component<1 || component>5) return ASTR_DEVICE_INVALID_ARGUMENT;
    if (astr_device_fluid_filter_workspace_components(handle)!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    const size_t count = physical_count(s);
    const AstrDeviceView q=s.fields[0].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    filter_kernel<1><<<(count+threads-1)/threads,threads,0,s.stream>>>(axis==1 ? work : q,axis==1 ? q : work,
        s.upper[0]+1,s.upper[1]+1,count,axis,axis==1 ? 1 : component,axis==1 ? component : 1,0,0,0);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_component_commit(void *handle,int component)
{
    if (!handle || component<1 || component>5) return ASTR_DEVICE_INVALID_ARGUMENT;
    if (astr_device_fluid_filter_workspace_components(handle)!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    const size_t count = physical_count(s);
    physical_copy_kernel<1><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[ASTR_DEVICE_FILTER_WORK].view,s.fields[0].view,s.upper[0]+1,s.upper[1]+1,count,1,component);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_axis(void *handle, int axis)
{
    if (!handle || axis < 0 || axis > 2) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (astr_device_fluid_filter_workspace_components(handle)!=5) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count = physical_count(s);
    const AstrDeviceView q = s.fields[0].view, work = s.fields[ASTR_DEVICE_FILTER_WORK].view;
    filter_kernel<5><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        axis == 1 ? work : q,axis == 1 ? q : work,s.upper[0]+1,s.upper[1]+1,count,axis,1,1,0,0,0);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_component_axis_basic(void *handle,int axis,int component,int lower,int upper)
{
    if(!handle || axis<0 || axis>2 || component<1 || component>5 || lower<0 || lower>1 || upper<0 || upper>1)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    if(astr_device_fluid_filter_workspace_components(handle)!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    // A single physical end can use valid halo data toward its MPI neighbour.
    if(lower && upper && s.upper[axis]<8) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=physical_count(s);
    const AstrDeviceView q=s.fields[0].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    filter_kernel<1,true><<<(count+threads-1)/threads,threads,0,s.stream>>>(axis==1 ? work : q,axis==1 ? q : work,
        s.upper[0]+1,s.upper[1]+1,count,axis,axis==1 ? 1 : component,axis==1 ? component : 1,lower,upper,s.upper[axis]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_axis_basic(void *handle,int axis,int lower,int upper)
{
    if(!handle || axis<0 || axis>2 || lower<0 || lower>1 || upper<0 || upper>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(astr_device_fluid_filter_workspace_components(handle)!=5) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(lower && upper && s.upper[axis]<8) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=physical_count(s);
    const AstrDeviceView q=s.fields[0].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    filter_kernel<5,true><<<(count+threads-1)/threads,threads,0,s.stream>>>(axis==1 ? work : q,axis==1 ? q : work,
        s.upper[0]+1,s.upper[1]+1,count,axis,1,1,lower,upper,s.upper[axis]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_axis_high_wall(void *handle,int axis,int component,int lower,int upper,int adiabatic)
{
    if(component<0 || component>5 || adiabatic<0 || adiabatic>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=component==0 ? astr_device_fluid_filter_axis_basic(handle,axis,lower,upper) :
        astr_device_fluid_filter_component_axis_basic(handle,axis,component,lower,upper);
    if(status!=0 || (!lower && !upper)) return status;
    State &s=*static_cast<State *>(handle);
    const AstrDeviceView q=s.fields[0].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    const int width=s.upper[axis==0 ? 1 : 0]+1;
    const size_t plane=physical_count(s)/(s.upper[axis]+1);
    high_wall_filter_rows<><<<(8*plane+threads-1)/threads,threads,0,s.stream>>>(axis==1 ? work : q,axis==1 ? q : work,
        axis,s.upper[axis],lower,upper,axis==1 || component==0 ? 1 : component,
        axis==1 && component!=0 ? component : 1,component==0 ? 5 : 1,plane,width,adiabatic);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_filter_commit(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (astr_device_fluid_filter_workspace_components(handle)!=5) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count = physical_count(s);
    physical_copy_kernel<5><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[ASTR_DEVICE_FILTER_WORK].view,s.fields[0].view,s.upper[0]+1,s.upper[1]+1,count,1,1);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_rk_save(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    return s.rhs_ready ? rk<true>(s,0,0,0,0) : ASTR_DEVICE_NOT_ALLOCATED;
}

extern "C" int astr_device_fluid_rk_update(void *handle, Real dt, Real a, Real b, Real c)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    return s.rhs_ready ? rk<false>(s,dt,a,b,c) : ASTR_DEVICE_NOT_ALLOCATED;
}

namespace {
int prepare_rhs(void *handle,int viscous,int flux_bytes)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready || (viscous && !s.gradients_ready)) return ASTR_DEVICE_NOT_ALLOCATED;
    if (s.rhs_ready || s.halo < 3) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status = astr::allocate_field(s.fields[ASTR_DEVICE_JACOBIAN],s.upper,s.halo,3);
    if (status != 0) return status;
    if (viscous) {
        status = astr::allocate_field(s.fields[ASTR_DEVICE_STRESS],s.upper,s.halo,4,6,1,flux_bytes);
        if (status != 0) return status;
        status = astr::allocate_field(s.fields[ASTR_DEVICE_HEAT_FLUX],s.upper,s.halo,4,3,1,flux_bytes);
        if (status != 0) return status;
    }
    s.rhs_ready = true;
    s.viscous_ready = viscous != 0;
    return 0;
}
}
extern "C" int astr_device_fluid_rhs_prepare(void *handle,int viscous)
{
    return prepare_rhs(handle,viscous,sizeof(Real));
}
extern "C" int astr_device_fluid_rhs_prepare_mixed_viscous(void *handle)
{
    return prepare_rhs(handle,1,sizeof(float));
}

extern "C" int astr_device_fluid_convection(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const cudaError_t cleared = cudaMemsetAsync(s.fields[ASTR_DEVICE_RHS].view.data,0,
                                               s.fields[ASTR_DEVICE_RHS].bytes,s.stream);
    if (cleared != cudaSuccess) return static_cast<int>(cleared);
    return divergence<false>(s);
}

extern "C" int astr_device_fluid_rhs_clear(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    return static_cast<int>(cudaMemsetAsync(s.fields[ASTR_DEVICE_RHS].view.data,0,s.fields[ASTR_DEVICE_RHS].bytes,s.stream));
}

static int flux_divergence_basic(void *handle,int viscous,const int low[3],const int high[3],
                                 const int first[3],const int last[3],bool wall_band)
{
    if(!handle || !low || !high || !first || !last || viscous<0 || viscous>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.rhs_ready || (viscous && !s.viscous_ready)) return ASTR_DEVICE_NOT_ALLOCATED;
    FluxBox box{};
    box.wall_band=wall_band;
    if(wall_band) {
        const int status=astr::me6_inner_bounds(s,box.inner_first,box.inner_last);
        if(status!=0) return status;
    }
    for(int a=0;a<3;++a) {
        if(low[a]<0 || low[a]>1 || high[a]<0 || high[a]>1 || first[a]<0 || last[a]>s.upper[a] || first[a]>last[a])
            return ASTR_DEVICE_INVALID_ARGUMENT;
        if((low[a] || high[a]) && s.upper[a]<4) return ASTR_DEVICE_INVALID_ARGUMENT;
        box.ends.low[a]=low[a];box.ends.high[a]=high[a];box.ends.upper[a]=s.upper[a];
        box.first[a]=first[a];box.last[a]=last[a];
    }
    const size_t count=physical_count(s);
    for(int axis=0;axis<3;++axis) {
        const dim3 block=axis==0 ? dim3(512,1,1) : (axis==1 ? dim3(32,16,1) : dim3(64,1,8));
        if(viscous) divergence_kernel<true,true><<<(count+threads-1)/threads,block,0,s.stream>>>(
            s.fields[0].view,s.fields[4].view,s.fields[5].view,s.fields[7].view,s.fields[10].view,
            s.fields[11].view,s.fields[12].view,s.fields[1].view,s.upper[0]+1,s.upper[1]+1,count,axis,box);
        else divergence_kernel<false,true><<<(count+threads-1)/threads,block,0,s.stream>>>(
            s.fields[0].view,s.fields[4].view,s.fields[5].view,s.fields[7].view,s.fields[10].view,
            s.fields[11].view,s.fields[12].view,s.fields[1].view,s.upper[0]+1,s.upper[1]+1,count,axis,box);
        const int status=astr::finish_launch(s);
        if(status!=0) return status;
    }
    return 0;
}

extern "C" int astr_device_fluid_flux_divergence_basic(void *handle,int viscous,const int low[3],const int high[3],
                                                       const int first[3],const int last[3])
{
    return flux_divergence_basic(handle,viscous,low,high,first,last,false);
}

extern "C" int astr_device_fluid_viscous_wall_band(void *handle,const int low[3],const int high[3],
                                                  const int first[3],const int last[3])
{
    return flux_divergence_basic(handle,1,low,high,first,last,true);
}

extern "C" int astr_device_fluid_flux_divergence_high_wall(void *handle,int viscous,int axis,
    const int low[3],const int high[3],const int first[3],const int last[3],int adiabatic)
{
    if(!handle || axis<0 || axis>2 || !low || !high || adiabatic<0 || adiabatic>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if((low[axis] || high[axis]) && (s.upper[axis]+s.halo<7 || (low[axis] && high[axis] && s.upper[axis]<7)))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=flux_divergence_basic(handle,viscous,low,high,first,last,false);
    if(status!=0 || (!low[axis] && !high[axis])) return status;
    FluxBox box{};
    for(int a=0;a<3;++a) {
        box.ends.low[a]=low[a];box.ends.high[a]=high[a];box.ends.upper[a]=s.upper[a];
        box.first[a]=first[a];box.last[a]=last[a];
    }
    const int n1=s.upper[axis==0 ? 1 : 0]+1;
    const size_t plane=static_cast<size_t>(n1)*(s.upper[axis==2 ? 1 : 2]+1);
    if(viscous) high_wall_flux_correction<true><<<(6*plane+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[0].view,s.fields[4].view,s.fields[5].view,s.fields[7].view,s.fields[10].view,
        s.fields[11].view,s.fields[12].view,s.fields[1].view,box,axis,n1,plane,adiabatic!=0);
    else high_wall_flux_correction<false><<<(6*plane+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[0].view,s.fields[4].view,s.fields[5].view,s.fields[7].view,s.fields[10].view,
        s.fields[11].view,s.fields[12].view,s.fields[1].view,box,axis,n1,plane,adiabatic!=0);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_viscous_high_open(void *handle,const int low[3],const int high[3])
{
    if(!handle||!low||!high) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    const int first[3]={0,0,0};
    int status=flux_divergence_basic(handle,1,low,high,first,s.upper,false);
    if(status) return status;
    FluxBox box{};
    for(int a=0;a<3;++a) {
        box.ends.low[a]=low[a];box.ends.high[a]=high[a];box.ends.upper[a]=s.upper[a];box.last[a]=s.upper[a];
    }
    for(int axis=0;axis<3;++axis) {
        if(!low[axis]&&!high[axis]) continue;
        const int n1=s.upper[axis==0 ? 1 : 0]+1;
        const size_t plane=static_cast<size_t>(n1)*(s.upper[axis==2 ? 1 : 2]+1);
        high_wall_flux_correction<true><<<(6*plane+threads-1)/threads,threads,0,s.stream>>>(
            s.fields[0].view,s.fields[4].view,s.fields[5].view,s.fields[7].view,s.fields[10].view,
            s.fields[11].view,s.fields[12].view,s.fields[1].view,box,axis,n1,plane,false);
        status=astr::finish_launch(s);if(status) return status;
    }
    return 0;
}

extern "C" int astr_device_fluid_wall_rhs(void *handle,int axis,const int low[3],const int high[3],
    const Real temperature[2],Real c2,Real c6,int kind)
{
    if(!handle || axis<0 || axis>2 || !low || !high || !temperature) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(kind!=41 && kind!=42 && kind!=421) return ASTR_DEVICE_INVALID_ARGUMENT;
    if((kind==42 && axis>1) || (kind==421 && axis!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!low[axis] && !high[axis]) return 0;
    if(s.upper[axis]+s.halo<7 || (low[axis] && high[axis] && s.upper[axis]<7)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(kind!=41 && s.upper[axis]<8) return ASTR_DEVICE_INVALID_ARGUMENT;
    PhysicalEnds ends{};
    for(int a=0;a<3;++a) { ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=s.upper[a]; }
    const int n1=s.upper[axis==0 ? 1 : 0]+1;
    const size_t plane=static_cast<size_t>(n1)*(s.upper[axis==2 ? 1 : 2]+1);
    // Adiabatic/slip state rates need more registers than the nodal flux kernels.
    constexpr int wall_threads=256;
    high_wall_rhs<<<(2*plane+wall_threads-1)/wall_threads,wall_threads,0,s.stream>>>(s.fields[0].view,s.fields[7].view,
        s.fields[10].view,s.fields[1].view,ends,axis,n1,plane,temperature[0],temperature[1],c2,c6,kind);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_viscous_flux(void *handle, Real reynolds, Real prandtl,
                                            Real c5, Real tc, Real tc1)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.viscous_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count = physical_count(s);
    stress_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.fields[4].view,s.fields[6].view,
        s.fields[8].view,s.fields[9].view,s.fields[ASTR_DEVICE_STRESS].view,s.fields[ASTR_DEVICE_HEAT_FLUX].view,
        s.upper[0]+1,s.upper[1]+1,count,reynolds,prandtl,c5,tc,tc1);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_viscous_divergence(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.viscous_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    return divergence<true>(s);
}

extern "C" int astr_device_fluid_prepare(void *handle, int gradients)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (s.fluid_ready) return ASTR_DEVICE_INVALID_ARGUMENT;
    for (int field = ASTR_DEVICE_RHO; field <= ASTR_DEVICE_METRIC; ++field) {
        const int rank = field == ASTR_DEVICE_METRIC ? 5 : (field == ASTR_DEVICE_VELOCITY ? 4 : 3);
        const int components = field == ASTR_DEVICE_METRIC || field == ASTR_DEVICE_VELOCITY ? 3 : 1;
        const int directions = field == ASTR_DEVICE_METRIC ? 3 : 1;
        const int status = astr::allocate_field(s.fields[field],s.upper,s.halo,rank,components,directions);
        if (status != 0) return status;
    }
    const AstrDeviceView rho=s.fields[ASTR_DEVICE_RHO].view;
    const size_t nodes=rho.stride[3];
    // Match the existing nonreacting CPU corner definition without filling
    // physical nodes or face halos before their actual reconstruction.
    primitive_corners_kernel<<<(nodes+threads-1)/threads,threads,0,s.stream>>>(rho,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,rho.extent[0],rho.extent[1],nodes,
        s.halo,s.upper[0],s.upper[1],s.upper[2]);
    const int initialized=astr::finish_launch(s);
    if (initialized!=0) return initialized;
    if (gradients) {
        int status = astr::allocate_field(s.fields[ASTR_DEVICE_DVELOCITY],s.upper,0,5,3,3);
        if (status != 0) return status;
        status = astr::allocate_field(s.fields[ASTR_DEVICE_DTEMPERATURE],s.upper,0,4,3);
        if (status != 0) return status;
    }
    cudaError_t status = cudaMalloc(&s.fluid_failure,sizeof(int));
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaMemsetAsync(s.fluid_failure,0,sizeof(int),s.stream);
    if (status != cudaSuccess) return static_cast<int>(status);
    s.gradients_ready = gradients != 0;
    s.fluid_ready = true;
    return 0;
}

extern "C" int astr_device_fluid_channel_geometry_prepare(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(s.fields[ASTR_DEVICE_COORDINATE_Y].view.data) return ASTR_DEVICE_INVALID_ARGUMENT;
    return astr::allocate_field(s.fields[ASTR_DEVICE_COORDINATE_Y],s.upper,0,3);
}

extern "C" int astr_device_fluid_initialize_exterior(void *handle,const int low[3],const int high[3])
{
    if(!handle || !low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    PhysicalEnds ends{};
    for(int a=0;a<3;++a) {
        if(low[a]<0 || low[a]>1 || high[a]<0 || high[a]>1) return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=s.upper[a];
    }
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view;
    const size_t count=q.stride[3];
    initialize_exterior_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(q,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        q.extent[0],q.extent[1],count,s.halo,ends);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_rti_source(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=physical_count(s);
    rti_source_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_RHS].view,
        s.upper[0]+1,s.upper[1]+1,count);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_uniform_force(void *handle,const Real force[3])
{
    if(!handle || !force) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=physical_count(s);
    uniform_force_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_RHS].view,s.upper[0]+1,s.upper[1]+1,count,
        force[0],force[1],force[2]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_primitives(void *handle, Real c2, Real c6)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count = physical_count(s);
    primitive_kernel<false><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[0].view,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view,
        s.upper[0]+1,s.upper[1]+1,count,0,s.upper[0],s.upper[1],s.upper[2],c2,c6,s.fluid_failure,PhysicalEnds{});
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_refresh_primitives(void *handle, Real c2, Real c6)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const AstrDeviceView q = s.fields[0].view;
    const size_t count = q.stride[3];
    primitive_kernel<false><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        q,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view,
        q.extent[0],q.extent[1],count,s.halo,s.upper[0],s.upper[1],s.upper[2],c2,c6,s.fluid_failure,PhysicalEnds{});
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_refresh_primitives_boundary(void *handle,Real c2,Real c6,const int low[3],const int high[3])
{
    if(!handle || !low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    PhysicalEnds ends{};
    for(int a=0;a<3;++a) {
        if(low[a]<0 || low[a]>1 || high[a]<0 || high[a]>1) return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=low[a];ends.high[a]=high[a];
    }
    const AstrDeviceView q=s.fields[0].view;
    const size_t count=q.stride[3];
    primitive_kernel<true><<<(count+threads-1)/threads,threads,0,s.stream>>>(q,s.fields[3].view,s.fields[4].view,
        s.fields[5].view,s.fields[6].view,q.extent[0],q.extent[1],count,s.halo,
        s.upper[0],s.upper[1],s.upper[2],c2,c6,s.fluid_failure,ends);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_conservative(void *handle, Real coefficient)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count = physical_count(s);
    conservative_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[0].view,s.fields[3].view,s.fields[4].view,s.fields[6].view,
        s.upper[0]+1,s.upper[1]+1,count,coefficient,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_gradient(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if (s.halo < 3) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count = physical_count(s);
    gradient_kernel<false><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[4].view,s.fields[6].view,s.fields[7].view,s.fields[8].view,s.fields[9].view,
        s.upper[0]+1,s.upper[1]+1,count,PhysicalEnds{});
    return astr::finish_launch(s);
}

static int basic_gradient(const State &s,const int low[3],const int high[3],PhysicalEnds &ends)
{
    if(!low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.halo<3) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) {
        if(low[a]<0 || low[a]>1 || high[a]<0 || high[a]>1) return ASTR_DEVICE_INVALID_ARGUMENT;
        if((low[a] || high[a]) && s.upper[a]<4) return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=s.upper[a];
    }
    return 0;
}

extern "C" int astr_device_fluid_gradient_basic_boundary(void *handle,const int low[3],const int high[3])
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    PhysicalEnds ends{};
    const int status=basic_gradient(s,low,high,ends);
    if(status) return status;
    const size_t count=physical_count(s);
    gradient_kernel<true><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[4].view,s.fields[6].view,s.fields[7].view,s.fields[8].view,s.fields[9].view,
        s.upper[0]+1,s.upper[1]+1,count,ends);
    return astr::finish_launch(s);
}

extern "C" int astr_device_air5_mean_refresh_velocity(void *handle,const int low[3],const int high[3])
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    PhysicalEnds ends{};
    int status=basic_gradient(s,low,high,ends);
    if(status) return status;
    const auto &velocity=s.fields[ASTR_DEVICE_VELOCITY];
    const auto &scratch=s.fields[ASTR_DEVICE_AIR5_MEAN_VELOCITY];
    if(!scratch.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    // Refresh diagnostic halos without modifying the checkpointed primitive cache.
    status=static_cast<int>(cudaMemcpyAsync(scratch.view.data,velocity.view.data,velocity.bytes,
        cudaMemcpyDeviceToDevice,s.stream));
    if(status) return status;
    for(int axis=0;axis<3;++axis) {
        status=astr_device_exchange_axis(handle,ASTR_DEVICE_AIR5_MEAN_VELOCITY,axis,0);
        if(status) return status;
    }
    const size_t count=physical_count(s);
    gradient_kernel<true,false><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        scratch.view,AstrDeviceView{},s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_DVELOCITY].view,
        AstrDeviceView{},s.upper[0]+1,s.upper[1]+1,count,ends);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_prepare_diagnostic_gradient(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const int fields[]={ASTR_DEVICE_DIAGNOSTIC_DVELOCITY,ASTR_DEVICE_DIAGNOSTIC_DTEMPERATURE};
    for(int c=0;c<2;++c) if(!s.fields[fields[c]].view.data) {
        const int status=astr::allocate_field(s.fields[fields[c]],s.upper,0,c==0?5:4,3,c==0?3:1,sizeof(float));
        if(status!=0) return status;
    }
    return 0;
}

extern "C" int astr_device_fluid_diagnostic_gradient(void *handle,const int low[3],const int high[3])
{
    if(!handle || !low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.fields[ASTR_DEVICE_DIAGNOSTIC_DVELOCITY].view.data ||
       !s.fields[ASTR_DEVICE_DIAGNOSTIC_DTEMPERATURE].view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.halo<3) return ASTR_DEVICE_INVALID_ARGUMENT;
    PhysicalEnds ends{};
    for(int a=0;a<3;++a) {
        if(low[a]<0 || low[a]>1 || high[a]<0 || high[a]>1) return ASTR_DEVICE_INVALID_ARGUMENT;
        if((low[a] || high[a]) && s.upper[a]<4) return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=s.upper[a];
    }
    const size_t count=physical_count(s);
    gradient_kernel<true><<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[4].view,s.fields[6].view,s.fields[7].view,s.fields[ASTR_DEVICE_DIAGNOSTIC_DVELOCITY].view,
        s.fields[ASTR_DEVICE_DIAGNOSTIC_DTEMPERATURE].view,s.upper[0]+1,s.upper[1]+1,count,ends);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_sbp63_gradient(void *handle,int total,int start,const Real temperature[2])
{
    if(!handle || !temperature || total<11 || start<0) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.halo<5 || start+s.upper[1]>total) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=physical_count(s);
    sbp63_gradient_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[4].view,s.fields[6].view,s.fields[7].view,
        s.fields[8].view,s.fields[9].view,s.upper[0]+1,s.upper[1]+1,count,total,start,temperature[0],temperature[1]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_sbp63_diffusion(void *handle,int total,int start,const Real temperature[2],
    Real re,Real pr,Real c5,Real tc,Real tc1)
{
    if(!handle || !temperature || total<11 || start<0) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.gradients_ready || !s.viscous_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.halo<5 || start+s.upper[1]>total) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=physical_count(s);
    sbp63_diffusion_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[4].view,s.fields[6].view,s.fields[7].view,
        s.fields[10].view,s.fields[11].view,s.fields[12].view,s.fields[9].view,s.fields[1].view,
        s.upper[0]+1,s.upper[1]+1,count,total,start,temperature[0],temperature[1],re,pr,c5,tc,tc1);
    return astr::finish_launch(s);
}

namespace {
int correct_high_gradient(astr::State &s,int axis,const int low[3],const int high[3])
{
    if(!low[axis]&&!high[axis]) return 0;
    PhysicalEnds ends{};
    for(int a=0;a<3;++a) { ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=s.upper[a]; }
    const int first=axis==0 ? 1 : 0,second=axis==2 ? 1 : 2,n1=s.upper[first]+1;
    const size_t plane=static_cast<size_t>(n1)*(s.upper[second]+1);
    high_wall_gradient_correction<<<(6*plane+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,
        s.fields[ASTR_DEVICE_DVELOCITY].view,s.fields[ASTR_DEVICE_DTEMPERATURE].view,ends,axis,n1,plane);
    return astr::finish_launch(s);
}
}

extern "C" int astr_device_fluid_gradient_high_wall(void *handle,int axis,const int low[3],const int high[3])
{
    if(!handle || axis<0 || axis>2 || !low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if((low[axis] || high[axis]) && (s.upper[axis]+s.halo<7 || (low[axis] && high[axis] && s.upper[axis]<7)))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=astr_device_fluid_gradient_basic_boundary(handle,low,high);
    return status ? status : correct_high_gradient(s,axis,low,high);
}

extern "C" int astr_device_fluid_gradient_high_open(void *handle,const int low[3],const int high[3])
{
    if(!handle||!low||!high) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    for(int axis=0;axis<3;++axis)
        if((low[axis]||high[axis])&&(s.upper[axis]+s.halo<7||(low[axis]&&high[axis]&&s.upper[axis]<7)))
            return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=astr_device_fluid_gradient_basic_boundary(handle,low,high);
    for(int axis=0;axis<3&&status==0;++axis) status=correct_high_gradient(s,axis,low,high);
    return status;
}

extern "C" int astr_device_fluid_check(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    int failure = 0;
    cudaError_t status = cudaMemcpyAsync(&failure,s.fluid_failure,sizeof(int),cudaMemcpyDeviceToHost,s.stream);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaStreamSynchronize(s.stream);
    if (status != cudaSuccess) return static_cast<int>(status);
    return failure == 0 ? 0 : (failure == 1 ? ASTR_DEVICE_NONPHYSICAL : ASTR_DEVICE_INVALID_SPECTRUM);
}

extern "C" int astr_device_fluid_cfl(void *handle, int nondimensional, Real mach,
                                    Real gamma, Real rgas, Real rates[3])
{
    if (!handle || !rates) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
#ifndef ASTR_REAL32
    if(s.air5_ready) {
        const int origin[3]={0,0,0};Real values[4];int64_t keys[4];
        const int status=astr_device_fluid_cfl_diagnostics(handle,nondimensional,mach,gamma,rgas,
            s.upper,origin,0,values,keys,nullptr);
        if(status==0) for(int a=0;a<3;++a) rates[a]=values[a];
        return status;
    }
#endif
    const size_t count = physical_count(s);
    const size_t blocks = (count+threads-1)/threads;
    if (!s.cfl_partial) {
        cudaError_t status = cudaMalloc(&s.cfl_partial,3*blocks*sizeof(Real));
        if (status != cudaSuccess) return static_cast<int>(status);
        status = cudaMalloc(&s.cfl_values,3*sizeof(Real));
        if (status != cudaSuccess) return static_cast<int>(status);
    }
    cfl_partial_kernel<<<blocks,threads,0,s.stream>>>(s.fields[4].view,s.fields[6].view,
        s.fields[7].view,s.upper[0]+1,s.upper[1]+1,count,nondimensional,mach,gamma,rgas,
        s.cfl_partial,s.fluid_failure);
    int status = astr::finish_launch(s);
    if (status != 0) return status;
    cfl_finish_kernel<<<1,threads,0,s.stream>>>(s.cfl_partial,s.cfl_values,blocks);
    status = astr::finish_launch(s);
    if (status != 0) return status;
    status = static_cast<int>(cudaMemcpyAsync(rates,s.cfl_values,3*sizeof(Real),cudaMemcpyDeviceToHost,s.stream));
    if (status != 0) return status;
    return astr_device_fluid_check(handle);
}
