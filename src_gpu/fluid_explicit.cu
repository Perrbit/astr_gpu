#include <cmath>
#include "device_state_internal.cuh"
#include "fluid_explicit.h"

namespace {
using astr::State;
using astr::at;
constexpr int threads = astr::block_threads;
constexpr double inv60 = 1.0 / 60.0;

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
void primitive_kernel(AstrDeviceView q, AstrDeviceView rho, AstrDeviceView vel,
                      AstrDeviceView prs, AstrDeviceView tmp, int nx, int ny,
                      size_t count, int halo, int im, int jm, int km, double c2, double c6, int *failure)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        i -= halo; j -= halo; k -= halo;
        const int outside = (i < 0 || i > im)+(j < 0 || j > jm)+(k < 0 || k > km);
        if (outside > 1) return;
        const double den = at(q, i, j, k, 1);
        const double u = at(q, i, j, k, 2) / den;
        const double v = at(q, i, j, k, 3) / den;
        const double w = at(q, i, j, k, 4) / den;
        const double ke = 0.5 * den * (u*u + v*v + w*w);
        const double pressure = (at(q, i, j, k, 5) - ke) / c6;
        const double temperature = pressure / den * c2;
        at(rho, i, j, k) = den;
        at(vel, i, j, k, 1) = u;
        at(vel, i, j, k, 2) = v;
        at(vel, i, j, k, 3) = w;
        at(prs, i, j, k) = pressure;
        at(tmp, i, j, k) = temperature;
        if (!(isfinite(den) && den > 0.0 && isfinite(u) && isfinite(v) && isfinite(w)
              && isfinite(pressure) && pressure > 0.0 && isfinite(temperature) && temperature > 0.0))
            atomicCAS(failure, 0, 1);
    }
    return;
}

__global__ __launch_bounds__(threads)
void conservative_kernel(AstrDeviceView q, AstrDeviceView rho, AstrDeviceView vel,
                         AstrDeviceView tmp, int nx, int ny, size_t count,
                         double coefficient, int *failure)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        const double den = at(rho, i, j, k);
        const double u = at(vel, i, j, k, 1);
        const double v = at(vel, i, j, k, 2);
        const double w = at(vel, i, j, k, 3);
        const double temperature = at(tmp, i, j, k);
        const double ke = 0.5 * (u*u + v*v + w*w);
        const double energy = den * (temperature * coefficient + ke);
        at(q, i, j, k, 1) = den;
        at(q, i, j, k, 2) = den * u;
        at(q, i, j, k, 3) = den * v;
        at(q, i, j, k, 4) = den * w;
        at(q, i, j, k, 5) = energy;
        if (!(isfinite(den) && den > 0.0 && isfinite(u) && isfinite(v) && isfinite(w)
              && isfinite(temperature) && temperature > 0.0 && isfinite(energy)))
            atomicCAS(failure, 0, 1);
    }
    return;
}

__device__ double derivative6(const AstrDeviceView &field, int i, int j, int k, int component, int axis)
{
    const size_t center = astr::offset(field, i, j, k, component);
    const size_t s = field.stride[axis];
    const double *v = static_cast<const double *>(field.data);
    return 0.75 * (v[center+s] - v[center-s])
         - 0.15 * (v[center+2*s] - v[center-2*s])
         + inv60 * (v[center+3*s] - v[center-3*s]);
}

__global__ __launch_bounds__(threads)
void gradient_kernel(AstrDeviceView vel, AstrDeviceView tmp, AstrDeviceView metric,
                     AstrDeviceView dvel, AstrDeviceView dtmp, int nx, int ny, size_t count)
{
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        for (int a = 1; a <= 3; ++a) {
            const double dx = derivative6(vel, i, j, k, a, 0);
            const double dy = derivative6(vel, i, j, k, a, 1);
            const double dz = derivative6(vel, i, j, k, a, 2);
            for (int b = 1; b <= 3; ++b)
                at(dvel, i, j, k, a, b) = dx*at(metric,i,j,k,1,b)
                    + dy*at(metric,i,j,k,2,b) + dz*at(metric,i,j,k,3,b);
        }
        const double dx = derivative6(tmp, i, j, k, 1, 0);
        const double dy = derivative6(tmp, i, j, k, 1, 1);
        const double dz = derivative6(tmp, i, j, k, 1, 2);
        for (int b = 1; b <= 3; ++b)
            at(dtmp, i, j, k, b) = dx*at(metric,i,j,k,1,b)
                + dy*at(metric,i,j,k,2,b) + dz*at(metric,i,j,k,3,b);
    }
    return;
}

__global__ __launch_bounds__(threads)
void cfl_partial_kernel(AstrDeviceView vel, AstrDeviceView tmp, AstrDeviceView metric,
                        int nx, int ny, size_t count, int nondimensional,
                        double mach, double gamma, double rgas, double *partial, int *failure)
{
    __shared__ double values[3][threads];
    const int lane = threadIdx.x;
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
    double rates[3] = {0.0, 0.0, 0.0};
    if (id < count) {
        int i, j, k;
        point(id, nx, ny, i, j, k);
        const double temperature = at(tmp, i, j, k);
        const double sound = nondimensional ? sqrt(temperature)/mach : sqrt(gamma*rgas*temperature);
        bool valid = isfinite(sound) && sound > 0.0;
        double velocity[3];
        for (int b = 0; b < 3; ++b) {
            velocity[b] = at(vel,i,j,k,b+1);
            valid = valid && isfinite(velocity[b]);
        }
        for (int a = 0; a < 3; ++a) {
            double projection = 0.0, norm = 0.0;
            for (int b = 0; b < 3; ++b) {
                const double m = at(metric,i,j,k,a+1,b+1);
                valid = valid && isfinite(m);
                projection += velocity[b] * m;
                norm += m * m;
            }
            rates[a] = fabs(projection) + sound * sqrt(norm);
            valid = valid && isfinite(rates[a]) && rates[a] >= 0.0;
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
void cfl_finish_kernel(const double *partial, double *values, size_t blocks)
{
    __shared__ double reduced[3][threads];
    const int lane = threadIdx.x;
    double best[3] = {0.0, 0.0, 0.0};
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

__device__ double convective_flux(AstrDeviceView q, AstrDeviceView vel, AstrDeviceView prs,
                                  AstrDeviceView metric, AstrDeviceView jac,
                                  int i, int j, int k, int component, int direction)
{
    const double u = at(metric,i,j,k,direction,1)*at(vel,i,j,k,1)
                   + at(metric,i,j,k,direction,2)*at(vel,i,j,k,2)
                   + at(metric,i,j,k,direction,3)*at(vel,i,j,k,3);
    const double value = at(q,i,j,k,component), pressure = at(prs,i,j,k);
    if (component == 1) return at(jac,i,j,k)*value*u;
    if (component == 5) return at(jac,i,j,k)*(value+pressure)*u;
    return at(jac,i,j,k)*(value*u+at(metric,i,j,k,direction,component-1)*pressure);
}

__device__ double viscous_flux(AstrDeviceView stress, AstrDeviceView heat,
                               AstrDeviceView metric, AstrDeviceView jac,
                               int i, int j, int k, int component, int direction)
{
    const int tensor[3][3] = {{1,2,3},{2,4,5},{3,5,6}};
    double flux[3];
    for (int b = 0; b < 3; ++b)
        flux[b] = component == 5 ? at(heat,i,j,k,b+1)
                                : at(stress,i,j,k,tensor[component-2][b]);
    return (flux[0]*at(metric,i,j,k,direction,1)+flux[1]*at(metric,i,j,k,direction,2)
           +flux[2]*at(metric,i,j,k,direction,3))*at(jac,i,j,k);
}

template<bool viscous>
__global__ __launch_bounds__(threads)
void divergence_kernel(AstrDeviceView q, AstrDeviceView vel, AstrDeviceView prs,
                       AstrDeviceView metric, AstrDeviceView jac, AstrDeviceView stress,
                       AstrDeviceView heat, AstrDeviceView rhs, int nx, int ny, size_t count, int axis)
{
    const int lane = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+lane;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        for (int c = viscous ? 2 : 1; c <= 5; ++c) {
            double plus[3],minus[3];
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
            const double df = .75*(plus[0]-minus[0])-.15*(plus[1]-minus[1])
                            +inv60*(plus[2]-minus[2]);
            if (viscous) at(rhs,i,j,k,c) += df;
            else at(rhs,i,j,k,c) -= df;
        }
    }
    return;
}

__global__ __launch_bounds__(threads)
void stress_kernel(AstrDeviceView vel, AstrDeviceView tmp, AstrDeviceView dvel,
                   AstrDeviceView dtmp, AstrDeviceView stress, AstrDeviceView heat,
                   int nx, int ny, size_t count, double reynolds, double prandtl,
                   double c5, double tc, double tc1)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const double t = at(tmp,i,j,k);
        const double mu = t*sqrt(t)*tc1/(t+tc)/reynolds, twice = 2.0*mu;
        const double hcc = (mu/prandtl)/c5;
        const double s11 = at(dvel,i,j,k,1,1), s22 = at(dvel,i,j,k,2,2), s33 = at(dvel,i,j,k,3,3);
        const double skk = (s11+s22+s33)/3.0;
        const double tau[6] = {twice*(s11-skk),
            twice*(.5*(at(dvel,i,j,k,1,2)+at(dvel,i,j,k,2,1))),
            twice*(.5*(at(dvel,i,j,k,1,3)+at(dvel,i,j,k,3,1))), twice*(s22-skk),
            twice*(.5*(at(dvel,i,j,k,2,3)+at(dvel,i,j,k,3,2))), twice*(s33-skk)};
        for (int c = 0; c < 6; ++c) at(stress,i,j,k,c+1) = tau[c];
        const double u = at(vel,i,j,k,1), v = at(vel,i,j,k,2), w = at(vel,i,j,k,3);
        at(heat,i,j,k,1) = hcc*at(dtmp,i,j,k,1)+tau[0]*u+tau[1]*v+tau[2]*w;
        at(heat,i,j,k,2) = hcc*at(dtmp,i,j,k,2)+tau[1]*u+tau[3]*v+tau[4]*w;
        at(heat,i,j,k,3) = hcc*at(dtmp,i,j,k,3)+tau[2]*u+tau[4]*v+tau[5]*w;
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
            s.upper[0]+1,s.upper[1]+1,count,axis);
        const int status = astr::finish_launch(s);
        if (status != 0) return status;
    }
    return 0;
}

template<int components>
__global__ __launch_bounds__(threads)
void filter_kernel(AstrDeviceView source, AstrDeviceView destination, int nx, int ny, size_t count, int axis,
                   int source_component,int destination_component)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const size_t stride = source.stride[axis];
        const double *v = static_cast<const double *>(source.data);
        for (int c = 0; c < components; ++c) {
            const size_t p = astr::offset(source,i,j,k,source_component+c);
            at(destination,i,j,k,destination_component+c) = (193.0/256.0)*v[p]+(105.0/512.0)*(v[p+stride]+v[p-stride])
                -(15.0/128.0)*(v[p+2*stride]+v[p-2*stride])
                +(45.0/1024.0)*(v[p+3*stride]+v[p-3*stride])
                -(5.0/512.0)*(v[p+4*stride]+v[p-4*stride])
                +(1.0/1024.0)*(v[p+5*stride]+v[p-5*stride]);
        }
    }
    return;
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
               int nx, int ny, size_t count, double dt, double a, double b, double c)
{
    const size_t id = static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id < count) {
        int i,j,k;
        point(id,nx,ny,i,j,k);
        const double volume = at(jac,i,j,k);
        for (int m = 1; m <= 5; ++m) {
            if (save) at(saved,i,j,k,m) = at(q,i,j,k,m)*volume;
            else at(q,i,j,k,m) = (a*at(saved,i,j,k,m)+b*at(q,i,j,k,m)*volume
                                +c*at(rhs,i,j,k,m)*dt)/volume;
        }
    }
    return;
}

template<bool save> int rk(State &s, double dt, double a, double b, double c)
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

extern "C" int astr_device_fluid_filter_component_axis(void *handle,int axis,int component)
{
    if (!handle || axis<0 || axis>2 || component<1 || component>5) return ASTR_DEVICE_INVALID_ARGUMENT;
    if (astr_device_fluid_filter_workspace_components(handle)!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    const size_t count = physical_count(s);
    const AstrDeviceView q=s.fields[0].view,work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    filter_kernel<1><<<(count+threads-1)/threads,threads,0,s.stream>>>(axis==1 ? work : q,axis==1 ? q : work,
        s.upper[0]+1,s.upper[1]+1,count,axis,axis==1 ? 1 : component,axis==1 ? component : 1);
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
        axis == 1 ? work : q,axis == 1 ? q : work,s.upper[0]+1,s.upper[1]+1,count,axis,1,1);
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

extern "C" int astr_device_fluid_rk_update(void *handle, double dt, double a, double b, double c)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    return s.rhs_ready ? rk<false>(s,dt,a,b,c) : ASTR_DEVICE_NOT_ALLOCATED;
}

extern "C" int astr_device_fluid_rhs_prepare(void *handle, int viscous)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready || (viscous && !s.gradients_ready)) return ASTR_DEVICE_NOT_ALLOCATED;
    if (s.rhs_ready || s.halo < 3) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status = astr::allocate_field(s.fields[ASTR_DEVICE_JACOBIAN],s.upper,s.halo,3);
    if (status != 0) return status;
    if (viscous) {
        status = astr::allocate_field(s.fields[ASTR_DEVICE_STRESS],s.upper,s.halo,4,6);
        if (status != 0) return status;
        status = astr::allocate_field(s.fields[ASTR_DEVICE_HEAT_FLUX],s.upper,s.halo,4,3);
        if (status != 0) return status;
    }
    s.rhs_ready = true;
    s.viscous_ready = viscous != 0;
    return 0;
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

extern "C" int astr_device_fluid_viscous_flux(void *handle, double reynolds, double prandtl,
                                             double c5, double tc, double tc1)
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

extern "C" int astr_device_fluid_primitives(void *handle, double c2, double c6)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count = physical_count(s);
    primitive_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[0].view,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view,
        s.upper[0]+1,s.upper[1]+1,count,0,s.upper[0],s.upper[1],s.upper[2],c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_refresh_primitives(void *handle, double c2, double c6)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const AstrDeviceView q = s.fields[0].view;
    const size_t count = q.stride[3];
    primitive_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(
        q,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view,
        q.extent[0],q.extent[1],count,s.halo,s.upper[0],s.upper[1],s.upper[2],c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fluid_conservative(void *handle, double coefficient)
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
    gradient_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(
        s.fields[4].view,s.fields[6].view,s.fields[7].view,s.fields[8].view,s.fields[9].view,
        s.upper[0]+1,s.upper[1]+1,count);
    return astr::finish_launch(s);
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

extern "C" int astr_device_fluid_cfl(void *handle, int nondimensional, double mach,
                                    double gamma, double rgas, double rates[3])
{
    if (!handle || !rates) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s = *static_cast<State *>(handle);
    if (!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count = physical_count(s);
    const size_t blocks = (count+threads-1)/threads;
    if (!s.cfl_partial) {
        cudaError_t status = cudaMalloc(&s.cfl_partial,3*blocks*sizeof(double));
        if (status != cudaSuccess) return static_cast<int>(status);
        status = cudaMalloc(&s.cfl_values,3*sizeof(double));
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
    status = static_cast<int>(cudaMemcpyAsync(rates,s.cfl_values,3*sizeof(double),cudaMemcpyDeviceToHost,s.stream));
    if (status != 0) return status;
    return astr_device_fluid_check(handle);
}
