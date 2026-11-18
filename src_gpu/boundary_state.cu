#include <cmath>
#include "boundary_state.h"
#include "fluid_explicit.h"
#include "device_state_internal.cuh"
#include "high_wall_math.cuh"
#include "nscbc_incident_math.cuh"
#include "high_open_math.cuh"

namespace {
using astr::at;
using Real=astr::numerical_real;
__global__ void wall_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,int axis,int side,int last,int n1,size_t count,Real tw,Real c2,Real c6,bool adiabatic,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int p[3]={};p[(axis+1)%3]=id%n1;p[(axis+2)%3]=id/n1;p[axis]=side ? last-1 : 1;
    const Real p1=at(pressure,p[0],p[1],p[2]);
    const Real t1=adiabatic ? at(temperature,p[0],p[1],p[2]) : 0;
    p[axis]=side ? last-2 : 2;
    const Real pw=(4*p1-at(pressure,p[0],p[1],p[2]))*(Real(1.0)/Real(3.0));
    if(adiabatic) tw=(4*t1-at(temperature,p[0],p[1],p[2]))*(Real(1.0)/Real(3.0));
    const Real den=pw/tw*c2;
    p[axis]=side ? last : 0;
    at(rho,p[0],p[1],p[2])=den;at(pressure,p[0],p[1],p[2])=pw;at(temperature,p[0],p[1],p[2])=tw;
    at(q,p[0],p[1],p[2],1)=den;at(q,p[0],p[1],p[2],5)=pw*c6;
    for(int c=1;c<=3;++c) { at(velocity,p[0],p[1],p[2],c)=0;at(q,p[0],p[1],p[2],c+1)=0; }
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0 && isfinite(tw) && tw>0)) atomicCAS(failure,0,1);
}

__global__ void slip_y_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView metric,const int8_t *mask,int code,int side,int im,int jm,
    size_t count,Real tw,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=id%(im+1),k=id/(im+1),j=side ? jm : 0,j1=side ? jm-1 : 1,j2=side ? jm-2 : 2;
    const bool slip=code==421 || (side==0 && mask[id]!=0);
    const Real pw=(4*at(pressure,i,j1,k)-at(pressure,i,j2,k))*(Real(1.0)/Real(3.0));
    Real v[3]={};
    if(slip) {
        tw=(4*at(temperature,i,j1,k)-at(temperature,i,j2,k))*(Real(1.0)/Real(3.0));
        Real normal[3];
        for(int c=0;c<3;++c) {
            v[c]=(4*at(velocity,i,j1,k,c+1)-at(velocity,i,j2,k,c+1))*(Real(1.0)/Real(3.0));
            normal[c]=at(metric,i,j,k,2,c+1);
        }
        const Real norm=sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
        for(int c=0;c<3;++c) normal[c]/=norm;
        const Real vn=v[0]*normal[0]+v[1]*normal[1]+v[2]*normal[2];
        for(int c=0;c<3;++c) v[c]-=vn*normal[c];
    }
    const Real den=pw/tw*c2;
    at(rho,i,j,k)=den;at(pressure,i,j,k)=pw;at(temperature,i,j,k)=tw;at(q,i,j,k,1)=den;
    for(int c=0;c<3;++c) {
        at(velocity,i,j,k,c+1)=v[c];at(q,i,j,k,c+2)=den*v[c];
    }
    at(q,i,j,k,5)=pw*c6+Real(.5)*den*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0 && isfinite(tw) && tw>0 &&
         isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]))) atomicCAS(failure,0,1);
}

template<bool outlet_only,bool forced>
__global__ void forced_lower_wall_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView speed,int im,size_t count,Real tw,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=outlet_only ? im : id%(im+1),k=outlet_only ? id : id/(im+1);
    const Real pw=(4*at(pressure,i,1,k)-at(pressure,i,2,k))*(Real(1.0)/3),den=pw/tw*c2;
    Real normal[3],norm=0;
    for(int c=0;c<3;++c) {normal[c]=at(metric,i,0,k,2,c+1);norm+=normal[c]*normal[c];}
    norm=sqrt(norm);
    const Real value=forced ? at(speed,i,0,k) : Real(0.0);
    Real u[3];
    for(int c=0;c<3;++c) {
        u[c]=value*(normal[c]/norm);
        at(velocity,i,0,k,c+1)=u[c];at(q,i,0,k,c+2)=den*u[c];
    }
    at(rho,i,0,k)=den;at(pressure,i,0,k)=pw;at(temperature,i,0,k)=tw;at(q,i,0,k,1)=den;
    at(q,i,0,k,5)=pw*c6+Real(.5)*den*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0 && isfinite(norm) && norm>0 && isfinite(value)))
        atomicCAS(failure,0,1);
}

__global__ void fixed_rti_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,int im,int j,size_t count,bool upper,Real c2,Real c6)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=id%(im+1),k=id/(im+1);
    const Real den=upper ? Real(1.0) : Real(2.0),pw=upper ? Real(2.5) : Real(1.0);
    at(rho,i,j,k)=den;at(pressure,i,j,k)=pw;at(temperature,i,j,k)=pw/den*c2;
    at(q,i,j,k,1)=den;at(q,i,j,k,5)=pw*c6;
    for(int c=1;c<=3;++c) { at(velocity,i,j,k,c)=0;at(q,i,j,k,c+1)=0; }
}

__global__ void ldc_lid_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,int im,int jm,size_t count,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=id%(im+1),k=id/(im+1);
    const Real pw=(4*at(pressure,i,jm-1,k)-at(pressure,i,jm-2,k))*(Real(1.0)/Real(3.0)),den=pw*c2;
    at(rho,i,jm,k)=den;at(pressure,i,jm,k)=pw;at(temperature,i,jm,k)=1;
    at(velocity,i,jm,k,1)=1;at(velocity,i,jm,k,2)=0;at(velocity,i,jm,k,3)=0;
    at(q,i,jm,k,1)=den;at(q,i,jm,k,2)=den;at(q,i,jm,k,3)=0;at(q,i,jm,k,4)=0;
    at(q,i,jm,k,5)=pw*c6+Real(.5)*den;
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0)) atomicCAS(failure,0,1);
}

__global__ void evolved_density_wall_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,int axis,int position,int n1,size_t count,
    Real tw,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int p[3]={};p[axis]=position;
    p[axis==0 ? 1 : 0]=id%n1;p[axis==2 ? 1 : 2]=id/n1;
    const Real den=at(q,p[0],p[1],p[2],1),pw=den*tw/c2;
    at(rho,p[0],p[1],p[2])=den;at(pressure,p[0],p[1],p[2])=pw;at(temperature,p[0],p[1],p[2])=tw;
    for(int c=1;c<=3;++c) { at(velocity,p[0],p[1],p[2],c)=0;at(q,p[0],p[1],p[2],c+1)=0; }
    at(q,p[0],p[1],p[2],5)=pw*c6;
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0)) atomicCAS(failure,0,1);
}

__global__ void evolved_adiabatic_wall_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,int axis,int side,int last,int n1,size_t count,
    bool slip,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int p[3]={};p[axis]=side ? last : 0;
    p[axis==0 ? 1 : 0]=id%n1;p[axis==2 ? 1 : 2]=id/n1;
    Real thermal[7],tangent[2][7];
    for(int s=1;s<=7;++s) {
        int v[3]={p[0],p[1],p[2]};v[axis]=side ? last-s : s;
        Real state[5];
        for(int c=0;c<5;++c) state[c]=at(q,v[0],v[1],v[2],c+1);
        thermal[s-1]=astr::high_wall::temperature(state,c2,c6);
        if(slip) {tangent[0][s-1]=state[1]/state[0];tangent[1][s-1]=state[3]/state[0];}
    }
    const Real tw=astr::high_wall::neumann7(thermal),den=at(q,p[0],p[1],p[2],1),pw=den*tw/c2;
    Real u[3]={};
    if(slip) {u[0]=astr::high_wall::neumann7(tangent[0]);u[2]=astr::high_wall::neumann7(tangent[1]);}
    at(rho,p[0],p[1],p[2])=den;at(pressure,p[0],p[1],p[2])=pw;at(temperature,p[0],p[1],p[2])=tw;
    for(int c=0;c<3;++c) {at(velocity,p[0],p[1],p[2],c+1)=u[c];at(q,p[0],p[1],p[2],c+2)=den*u[c];}
    at(q,p[0],p[1],p[2],5)=pw*c6+Real(.5)*den*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0 && isfinite(tw) && tw>0 &&
         isfinite(u[0]) && isfinite(u[2]))) atomicCAS(failure,0,1);
}

template<bool symmetric>
__global__ void extrapolate_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView rhs,AstrDeviceView metric,
    int axis,int side,int last,int n1,size_t count,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int p[3]={};p[(axis+1)%3]=id%n1;p[(axis+2)%3]=id/n1;p[axis]=side ? last-1 : 1;
    const Real p1=at(pressure,p[0],p[1],p[2]);
    const bool use_temperature=symmetric ? !(axis==2 && side==1) : axis==0;
    const Real therm1=use_temperature ? at(temperature,p[0],p[1],p[2]) : at(rho,p[0],p[1],p[2]);
    Real v[3];
    for(int c=0;c<3;++c) v[c]=4*at(velocity,p[0],p[1],p[2],c+1);
    p[axis]=side ? last-2 : 2;
    const Real pw=(4*p1-at(pressure,p[0],p[1],p[2]))*(Real(1.0)/Real(3.0));
    const Real therm2=use_temperature ? at(temperature,p[0],p[1],p[2]) : at(rho,p[0],p[1],p[2]);
    const Real extrapolated=(4*therm1-therm2)*(Real(1.0)/Real(3.0));
    const Real den=use_temperature ? pw/extrapolated*c2 : extrapolated;
    const Real tw=use_temperature ? extrapolated : pw/den*c2;
    for(int c=0;c<3;++c) v[c]=(v[c]-at(velocity,p[0],p[1],p[2],c+1))*(Real(1.0)/Real(3.0));
    p[axis]=side ? last : 0;
    if(symmetric) {
        Real n[3];
        for(int c=0;c<3;++c) n[c]=at(metric,p[0],p[1],p[2],axis+1,c+1);
        const Real norm=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
        for(int c=0;c<3;++c) n[c]/=norm;
        const Real normal_velocity=v[0]*n[0]+v[1]*n[1]+v[2]*n[2];
        for(int c=0;c<3;++c) v[c]-=normal_velocity*n[c];
    }
    at(rho,p[0],p[1],p[2])=den;at(pressure,p[0],p[1],p[2])=pw;at(temperature,p[0],p[1],p[2])=tw;
    at(q,p[0],p[1],p[2],1)=den;
    for(int c=0;c<3;++c) {
        at(velocity,p[0],p[1],p[2],c+1)=v[c];at(q,p[0],p[1],p[2],c+2)=den*v[c];
    }
    at(q,p[0],p[1],p[2],5)=pw*c6+Real(.5)*den*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    for(int c=1;c<=5;++c) at(rhs,p[0],p[1],p[2],c)=0;
    if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0 && isfinite(tw) && tw>0)) atomicCAS(failure,0,1);
}

__global__ void profile_inflow_state(AstrDeviceView target,AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,AstrDeviceView rhs,int ny,size_t count,
    bool mach_pressure,Real mach,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<count) {
        const int j=id%ny,k=id/ny;
        Real den=at(target,0,j,k,1),pw=at(target,0,j,k,5),tw=pw/den*c2;
        Real u[3];
        for(int c=0;c<3;++c) u[c]=at(target,0,j,k,c+2);
        if(mach_pressure) {
            const Real tp=at(target,0,j,k,6),css=sqrt(tp)/mach;
            const Real normal=u[0]*at(target,0,j,k,7)+u[1]*at(target,0,j,k,8)+u[2]*at(target,0,j,k,9);
            if(!(isfinite(normal) && normal>=Real(0.0))) atomicCAS(failure,0,1);
            if(normal/css<Real(1.0)) {
                pw=(Real(4.0)*at(pressure,1,j,k)-at(pressure,2,j,k))*(Real(1.0)/Real(3.0));
                tw=tp;den=pw/tw*c2;
            }
        }
        at(rho,0,j,k)=den;at(pressure,0,j,k)=pw;at(temperature,0,j,k)=tw;
        at(q,0,j,k,1)=den;
        for(int c=0;c<3;++c) { at(velocity,0,j,k,c+1)=u[c];at(q,0,j,k,c+2)=den*u[c]; }
        at(q,0,j,k,5)=pw*c6+Real(0.5)*den*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        for(int c=1;c<=5;++c) at(rhs,0,j,k,c)=Real(0.0);
        if(!(isfinite(den) && den>Real(0.0) && isfinite(pw) && pw>Real(0.0) && isfinite(tw) && tw>Real(0.0)))
            atomicCAS(failure,0,1);
    }
    return;
}

__device__ Real cubic_inflow(Real y1,Real y2,Real y3,Real y4,Real delta,Real x)
{
    const Real inv=Real(1.0)/delta,d3=delta*delta*delta;
    const Real a=(y4-3*y3+3*y2-y1)*(inv*inv*inv)/Real(6.0);
    const Real b=((y3-2*y2+y1)-6*a*d3)*(inv*inv)/Real(2.0);
    const Real c=(y2-y1-a*d3-b*delta*delta)*inv;
    return ((a*x+b)*x+c)*x+y1;
}

__global__ void dynamic_inflow_state(AstrDeviceView target,AstrDeviceView slices,AstrDeviceView q,AstrDeviceView rho,
    AstrDeviceView velocity,AstrDeviceView pressure,AstrDeviceView temperature,AstrDeviceView rhs,int ny,size_t count,
    int s0,int s1,int s2,int s3,Real delta,Real x,bool mach_pressure,Real mach,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int j=id%ny,k=id/ny;
    Real value[5];
    for(int c=0;c<5;++c) {
        const Real base=at(target,0,j,k,c==4 ? 6 : c+1);
        value[c]=base+cubic_inflow(at(slices,0,j,k,c+1,s0+1),at(slices,0,j,k,c+1,s1+1),
            at(slices,0,j,k,c+1,s2+1),at(slices,0,j,k,c+1,s3+1),delta,x);
    }
    // Retain the existing inlet prescription, including its negative-u rule.
    if(value[1]<0) value[1]=at(target,0,j,k,2);
    Real den=value[0],tw=value[4],pw=den*tw/c2;
    if(!(isfinite(den) && den>0 && isfinite(tw) && tw>0)) {atomicCAS(failure,0,1);return;}
    if(mach_pressure) {
        const Real vn=value[1]*at(target,0,j,k,7)+value[2]*at(target,0,j,k,8)+value[3]*at(target,0,j,k,9);
        if(!(isfinite(vn) && vn>=0)) {atomicCAS(failure,0,1);return;}
        if(vn/(sqrt(tw)/mach)<1) {
            pw=(4*at(pressure,1,j,k)-at(pressure,2,j,k))*(Real(1.0)/Real(3.0));den=pw/tw*c2;
        }
    }
    at(rho,0,j,k)=den;at(pressure,0,j,k)=pw;at(temperature,0,j,k)=tw;at(q,0,j,k,1)=den;
    Real kinetic=0;
    for(int c=0;c<3;++c) {
        const Real u=value[c+1];at(velocity,0,j,k,c+1)=u;at(q,0,j,k,c+2)=den*u;kinetic+=u*u;
    }
    at(q,0,j,k,5)=pw*c6+Real(.5)*den*kinetic;
    for(int c=1;c<=5;++c) at(rhs,0,j,k,c)=0;
    if(!(isfinite(pw) && pw>0 && isfinite(value[1]) && isfinite(value[2]) && isfinite(value[3]))) atomicCAS(failure,0,1);
}

__global__ void outflow21_state(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView rhs,int im,int ny,size_t count,
    Real mach,Real c2,Real c6,Real pressure_target,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<count) {
        const int j=id%ny,k=id/ny;
        Real normal[3],normal_velocity=Real(0.0),norm2=Real(0.0),u[3];
        for(int c=0;c<3;++c) {
            normal[c]=at(metric,im,j,k,1,c+1);norm2+=normal[c]*normal[c];
            normal_velocity+=normal[c]*at(velocity,im,j,k,c+1);
            u[c]=(Real(4.0)*at(velocity,im-1,j,k,c+1)-at(velocity,im-2,j,k,c+1))*(Real(1.0)/Real(3.0));
        }
        normal_velocity/=sqrt(norm2);
        Real tw=at(temperature,im,j,k),pw;
        const Real pe=(Real(4.0)*at(pressure,im-1,j,k)-at(pressure,im-2,j,k))*(Real(1.0)/Real(3.0));
        if(normal_velocity>=sqrt(tw)/mach) {
            pw=pe;tw=(Real(4.0)*at(temperature,im-1,j,k)-at(temperature,im-2,j,k))*(Real(1.0)/Real(3.0));
        } else {
            pw=pressure_target;
            const Real den_e=(Real(4.0)*at(rho,im-1,j,k)-at(rho,im-2,j,k))*(Real(1.0)/Real(3.0));
            const Real sound_e=(Real(4.0)*(sqrt(at(temperature,im-1,j,k))/mach)
                -sqrt(at(temperature,im-2,j,k))/mach)*(Real(1.0)/Real(3.0));
            u[0]+=(pe-pw)/den_e/sound_e;
        }
        const Real den=pw/tw*c2;
        at(rho,im,j,k)=den;at(pressure,im,j,k)=pw;at(temperature,im,j,k)=tw;
        at(q,im,j,k,1)=den;
        for(int c=0;c<3;++c) {at(velocity,im,j,k,c+1)=u[c];at(q,im,j,k,c+2)=den*u[c];}
        at(q,im,j,k,5)=pw*c6+Real(.5)*den*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        for(int c=1;c<=5;++c) at(rhs,im,j,k,c)=0;
        if(!(isfinite(den) && den>0 && isfinite(pw) && pw>0 && isfinite(tw) && tw>0)) atomicCAS(failure,0,1);
    }
    return;
}

template<bool high_order> __global__ void outflow_filter_y(AstrDeviceView q,AstrDeviceView plane,int im,int jm,int km,int npdcj)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t nodes=static_cast<size_t>(jm+1)*(km+1);
    if(id>=nodes*5) return;
    const int j=id%(jm+1),k=(id/(jm+1))%(km+1),c=id/nodes+1;
    const bool lower=npdcj==1 || npdcj==4,upper=npdcj==2 || npdcj==4;
    Real value;
    if(high_order&&((lower&&j<3)||(upper&&j>jm-3))) {
        const bool from_lower=lower&&j<3;const int row=from_lower ? j : jm-j;
        astr::numerical_real line[7];
        for(int n=0;n<7;++n) line[n]=at(q,im,from_lower ? n : jm-n,k,c);
        at(plane,0,j,k,c)=astr::high_open::filter6(line,row);
        return;
    }
    if(j==0 && lower) {
        // The low-order plate's imposed bottom-wall state is not filtered.
        value=at(q,im,j,k,c);
    } else if((lower && j==1) || (upper && j==jm-1)) {
        const int end=lower && j==1 ? 0 : jm,dir=lower && j==1 ? 1 : -1;
        value=(Real(1.0)/Real(16.0))*at(q,im,end,k,c)+Real(.75)*at(q,im,end+dir,k,c)
            +(Real(3.0)/Real(8.0))*at(q,im,end+2*dir,k,c)-Real(.25)*at(q,im,end+3*dir,k,c)
            +(Real(1.0)/Real(16.0))*at(q,im,end+4*dir,k,c);
    } else if(upper && j==jm) {
        value=(Real(15.0)/Real(16.0))*at(q,im,j,k,c)+Real(.25)*at(q,im,j-1,k,c)
            -(Real(3.0)/Real(8.0))*at(q,im,j-2,k,c)+Real(.25)*at(q,im,j-3,k,c)-(Real(1.0)/Real(16.0))*at(q,im,j-4,k,c);
    } else if((lower && j==2) || (upper && j==jm-2)) {
        value=(Real(5.0)/Real(8.0))*at(q,im,j,k,c)+Real(.25)*(at(q,im,j-1,k,c)+at(q,im,j+1,k,c))
            -(Real(1.0)/Real(16.0))*(at(q,im,j-2,k,c)+at(q,im,j+2,k,c));
    } else {
        value=(Real(11.0)/Real(16.0))*at(q,im,j,k,c)+(Real(15.0)/Real(64.0))*(at(q,im,j-1,k,c)+at(q,im,j+1,k,c))
            -(Real(3.0)/Real(32.0))*(at(q,im,j-2,k,c)+at(q,im,j+2,k,c))
            +(Real(1.0)/Real(64.0))*(at(q,im,j-3,k,c)+at(q,im,j+3,k,c));
    }
    at(plane,0,j,k,c)=value;
    return;
}

__global__ void outflow_filter_commit(AstrDeviceView q,AstrDeviceView plane,int im,int jm,int km)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t nodes=static_cast<size_t>(jm+1)*(km+1);
    if(id<nodes*5) {
        const int j=id%(jm+1),k=(id/(jm+1))%(km+1),c=id/nodes+1;
        at(q,im,j,k,c)=at(plane,0,j,k,c);
    }
    return;
}

__device__ Real top_sample(AstrDeviceView q,int i,int j,int k,int c,int axis,int index)
{
    return axis==0 ? at(q,index,j,k,c) : at(q,i,j,index,c);
}

__device__ Real top_filter_value(AstrDeviceView q,int i,int j,int k,int c,int axis,int index,int last,int ntype)
{
    const bool lower=ntype==1 || ntype==4,upper=ntype==2 || ntype==4;
    if((lower && index<2) || (upper && index>last-2)) {
        const int edge=lower && index<2 ? 0 : last,dir=edge==0 ? 1 : -1;
        const int row=edge==0 ? index : last-index;
        const Real f0=top_sample(q,i,j,k,c,axis,edge),f1=top_sample(q,i,j,k,c,axis,edge+dir),
            f2=top_sample(q,i,j,k,c,axis,edge+2*dir),f3=top_sample(q,i,j,k,c,axis,edge+3*dir),
            f4=top_sample(q,i,j,k,c,axis,edge+4*dir);
        if(row==0) return (Real(15.0)/Real(16.0))*f0+Real(.25)*f1-(Real(3.0)/Real(8.0))*f2+Real(.25)*f3-(Real(1.0)/Real(16.0))*f4;
        return (Real(1.0)/Real(16.0))*f0+Real(.75)*f1+(Real(3.0)/Real(8.0))*f2-Real(.25)*f3+(Real(1.0)/Real(16.0))*f4;
    }
    const Real f0=top_sample(q,i,j,k,c,axis,index),fm1=top_sample(q,i,j,k,c,axis,index-1),
        fp1=top_sample(q,i,j,k,c,axis,index+1),fm2=top_sample(q,i,j,k,c,axis,index-2),
        fp2=top_sample(q,i,j,k,c,axis,index+2);
    if((lower && index==2) || (upper && index==last-2))
        return (Real(5.0)/Real(8.0))*f0+Real(.25)*(fm1+fp1)-(Real(1.0)/Real(16.0))*(fm2+fp2);
    return (Real(11.0)/Real(16.0))*f0+(Real(15.0)/Real(64.0))*(fm1+fp1)-(Real(3.0)/Real(32.0))*(fm2+fp2)
        +(Real(1.0)/Real(64.0))*(top_sample(q,i,j,k,c,axis,index-3)+top_sample(q,i,j,k,c,axis,index+3));
}

__global__ void farfield_filter(AstrDeviceView q,AstrDeviceView plane,int im,int jm,int km,int axis,int ntype)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t nodes=static_cast<size_t>(im+1)*(km+1);
    if(id<nodes*5) {
        const int i=id%(im+1),k=(id/(im+1))%(km+1),c=id/nodes+1;
        const int index=axis==0 ? i : k,last=axis==0 ? im : km;
        at(plane,i,0,k,c)=top_filter_value(q,i,jm,k,c,axis,index,last,ntype);
    }
    return;
}

__global__ void farfield_filter_commit(AstrDeviceView q,AstrDeviceView plane,int im,int jm,int km)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t nodes=static_cast<size_t>(im+1)*(km+1);
    if(id<nodes*5) {
        const int i=id%(im+1),k=(id/(im+1))%(km+1),c=id/nodes+1;
        at(q,i,jm,k,c)=at(plane,i,0,k,c);
    }
    return;
}

__device__ void characteristic_matrices(Real density,const Real u[3],Real temperature,const Real ddi[3],
    Real gamma,Real mach,Real right[5][5],Real left[5][5])
{
    const Real css=sqrt(temperature)/mach;
    const Real norm=sqrt(ddi[0]*ddi[0]+ddi[1]*ddi[1]+ddi[2]*ddi[2]);
    const Real nx=ddi[0]/norm,ny=ddi[1]/norm,nz=ddi[2]/norm;
    const Real uu=u[0],vv=u[1],ww=u[2];
    const Real phi=Real(.5)*(gamma-1)*(uu*uu+vv*vv+ww*ww),vn=nx*uu+ny*vv+nz*ww;
    const Real a1=gamma-1,a2=1/(sqrt(Real(2.0))*density*css),a3=density/(css*sqrt(Real(2.0)));
    const Real a4=(phi+css*css)/(gamma-1),a5=1-phi/(css*css),a6=phi/(gamma-1);
    left[0][0]=nx*a5-(nz*vv-ny*ww)/density;
    left[0][1]=nx*a1*uu/(css*css);left[0][2]=nx*a1*vv/(css*css)+nz/density;
    left[0][3]=nx*a1*ww/(css*css)-ny/density;left[0][4]=-nx*a1/(css*css);
    left[1][0]=ny*a5-(nx*ww-nz*uu)/density;
    left[1][1]=ny*a1*uu/(css*css)-nz/density;left[1][2]=ny*a1*vv/(css*css);
    left[1][3]=ny*a1*ww/(css*css)+nx/density;left[1][4]=-ny*a1/(css*css);
    left[2][0]=nz*a5-(ny*uu-nx*vv)/density;
    left[2][1]=nz*a1*uu/(css*css)+ny/density;left[2][2]=nz*a1*vv/(css*css)-nx/density;
    left[2][3]=nz*a1*ww/(css*css);left[2][4]=-nz*a1/(css*css);
    left[3][0]=a2*(phi-css*vn);left[3][1]=-a2*(a1*uu-nx*css);
    left[3][2]=-a2*(a1*vv-ny*css);left[3][3]=-a2*(a1*ww-nz*css);left[3][4]=a1*a2;
    left[4][0]=a2*(phi+css*vn);left[4][1]=-a2*(a1*uu+nx*css);
    left[4][2]=-a2*(a1*vv+ny*css);left[4][3]=-a2*(a1*ww+nz*css);left[4][4]=a1*a2;
    right[0][0]=nx;right[0][1]=ny;right[0][2]=nz;right[0][3]=a3;right[0][4]=a3;
    right[1][0]=nx*uu;right[1][1]=ny*uu-nz*density;right[1][2]=nz*uu+ny*density;
    right[1][3]=a3*(uu+nx*css);right[1][4]=a3*(uu-nx*css);
    right[2][0]=nx*vv+nz*density;right[2][1]=ny*vv;right[2][2]=nz*vv-nx*density;
    right[2][3]=a3*(vv+ny*css);right[2][4]=a3*(vv-ny*css);
    right[3][0]=nx*ww-ny*density;right[3][1]=ny*ww+nx*density;right[3][2]=nz*ww;
    right[3][3]=a3*(ww+nz*css);right[3][4]=a3*(ww-nz*css);
    right[4][0]=nx*a6+density*(nz*vv-ny*ww);
    right[4][1]=ny*a6+density*(nx*ww-nz*uu);
    right[4][2]=nz*a6+density*(ny*uu-nx*vv);
    right[4][3]=a3*(a4+css*vn);right[4][4]=a3*(a4-css*vn);
}

__device__ void transformed_flux(AstrDeviceView q,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView jacobian,AstrDeviceView metric,int i,int j,int k,int axis,Real flux[5])
{
    Real un=0;
    for(int c=1;c<=3;++c) un+=at(metric,i,j,k,axis,c)*at(velocity,i,j,k,c);
    const Real jac=at(jacobian,i,j,k),p=at(pressure,i,j,k);
    flux[0]=jac*at(q,i,j,k,1)*un;
    for(int c=1;c<=3;++c) flux[c]=jac*(at(q,i,j,k,c+1)*un+at(metric,i,j,k,axis,c)*p);
    flux[4]=jac*(at(q,i,j,k,5)+p)*un;
}

__device__ bool physical_endpoint(int index,int last,int type)
{
    return (index==0 && (type==1 || type==4)) || (index==last && (type==2 || type==4));
}

__device__ void cartesian_flux(AstrDeviceView q,AstrDeviceView velocity,AstrDeviceView pressure,
    int i,int j,int k,int axis,Real flux[5])
{
    const Real un=at(velocity,i,j,k,axis),p=at(pressure,i,j,k);
    flux[0]=at(q,i,j,k,axis+1);
    for(int c=1;c<=3;++c) flux[c]=at(q,i,j,k,c+1)*un;
    flux[axis]+=p;flux[4]=(at(q,i,j,k,5)+p)*un;
}

__device__ void tangential_x_rhs(AstrDeviceView q,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView jacobian,AstrDeviceView metric,int i,int j,int k,int jm,int km,
    int npdcj,int npdck,bool include_lower_z,Real divergence[5])
{
    Real minus[5],plus[5];
    if(!physical_endpoint(j,jm,npdcj)) {
        transformed_flux(q,velocity,pressure,jacobian,metric,i,j-1,k,2,minus);
        transformed_flux(q,velocity,pressure,jacobian,metric,i,j+1,k,2,plus);
        for(int c=0;c<5;++c) divergence[c]+=Real(.5)*(plus[c]-minus[c]);
    }
    if((include_lower_z || !(j==0 && (npdcj==1 || npdcj==4))) && !physical_endpoint(k,km,npdck)) {
        transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k-1,3,minus);
        transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k+1,3,plus);
        for(int c=0;c<5;++c) divergence[c]+=Real(.5)*(plus[c]-minus[c]);
    }
}

__global__ void inflow12_rhs(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView jacobian,AstrDeviceView metric,AstrDeviceView rhs,
    int jm,int km,int npdcj,int npdck,Real gamma,Real mach,Real length,
    Real roinf,Real uinf,Real vinf,Real winf,Real pinf)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(jm+1)*(km+1)) return;
    const int j=id%(jm+1),k=id/(jm+1),i=0;
    Real ddi[3],u[3],norm2=0,normal=0;
    for(int c=0;c<3;++c) {
        ddi[c]=at(metric,i,j,k,1,c+1);u[c]=at(velocity,i,j,k,c+1);
        norm2+=ddi[c]*ddi[c];normal+=ddi[c]*u[c];
    }
    normal/=sqrt(norm2);
    const Real css=sqrt(at(temperature,i,j,k))/mach,jac=at(jacobian,i,j,k);
    Real right[5][5],left[5][5],f0[5],f1[5],rest[5]={},decs[5],lodi[5]={};
    characteristic_matrices(at(rho,i,j,k),u,at(temperature,i,j,k),ddi,gamma,mach,right,left);
    transformed_flux(q,velocity,pressure,jacobian,metric,0,j,k,1,f0);
    transformed_flux(q,velocity,pressure,jacobian,metric,1,j,k,1,f1);
    for(int c=0;c<5;++c) decs[c]=f1[c]-f0[c];
    for(int axis=1;axis<=3;++axis) {
        const Real derivative=at(metric,1,j,k,1,axis)*at(jacobian,1,j,k)-at(metric,0,j,k,1,axis)*jac;
        Real flux[5];cartesian_flux(q,velocity,pressure,0,j,k,axis,flux);
        for(int c=0;c<5;++c) rest[c]+=flux[c]*derivative;
    }
    const Real target[5]={roinf,roinf*uinf,roinf*vinf,roinf*winf,
        pinf/(gamma-1)+Real(.5)*roinf*(uinf*uinf+vinf*vinf+winf*winf)};
    const Real lambda[5]={normal,normal,normal,normal+css,normal-css};
    const Real tolerance=64*astr::incident::epsilon*fmax(css,fabs(normal)),rate=Real(.25)*css/length;
    for(int m=0;m<5;++m) {
        if(lambda[m]>tolerance) {
            for(int c=0;c<5;++c) lodi[m]+=left[m][c]*(at(q,0,j,k,c+1)-target[c]);
            lodi[m]*=rate;
        } else {
            for(int c=0;c<5;++c) lodi[m]+=left[m][c]*(decs[c]-rest[c]);
            lodi[m]/=jac;
        }
    }
    for(int c=0;c<5;++c) {
        Real value=0;
        for(int m=0;m<5;++m) value+=right[c][m]*lodi[m];
        decs[c]=value*jac+rest[c];
    }
    tangential_x_rhs(q,velocity,pressure,jacobian,metric,0,j,k,jm,km,npdcj,npdck,true,decs);
    for(int c=0;c<5;++c) at(rhs,0,j,k,c+1)-=decs[c];
    return;
}

__global__ void outflow22_rhs(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView jacobian,AstrDeviceView metric,AstrDeviceView rhs,
    int im,int jm,int km,int npdcj,int npdck,bool include_lower_z,
    Real gamma,Real mach,Real length,Real roinf,Real uinf,Real vinf,Real winf,Real pinf,Real pout)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(jm+1)*(km+1)) return;
    const int j=id%(jm+1),k=id/(jm+1),i=im;
    Real ddi[3],u[3],norm2=0,normal=0;
    for(int c=0;c<3;++c) {
        ddi[c]=at(metric,i,j,k,1,c+1);u[c]=at(velocity,i,j,k,c+1);
        norm2+=ddi[c]*ddi[c];normal+=ddi[c]*u[c];
    }
    normal/=sqrt(norm2);
    const Real density=at(rho,i,j,k),css=sqrt(at(temperature,i,j,k))/mach,jac=at(jacobian,i,j,k);
    Real right[5][5],left[5][5],f0[5],f1[5],f2[5],rest[5]={},decs[5],lodi[5]={};
    characteristic_matrices(density,u,at(temperature,i,j,k),ddi,gamma,mach,right,left);
    transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k,1,f0);
    transformed_flux(q,velocity,pressure,jacobian,metric,i-1,j,k,1,f1);
    transformed_flux(q,velocity,pressure,jacobian,metric,i-2,j,k,1,f2);
    for(int c=0;c<5;++c) decs[c]=Real(1.5)*f0[c]-2*f1[c]+Real(.5)*f2[c];
    for(int axis=1;axis<=3;++axis) {
        const Real derivative=Real(1.5)*at(metric,i,j,k,1,axis)*jac
            -2*at(metric,i-1,j,k,1,axis)*at(jacobian,i-1,j,k)
            +Real(.5)*at(metric,i-2,j,k,1,axis)*at(jacobian,i-2,j,k);
        Real flux[5];cartesian_flux(q,velocity,pressure,i,j,k,axis,flux);
        for(int c=0;c<5;++c) rest[c]+=flux[c]*derivative;
    }
    for(int m=0;m<5;++m) {
        for(int c=0;c<5;++c) lodi[m]+=left[m][c]*(decs[c]-rest[c]);
        lodi[m]/=jac;
    }
    const Real tolerance=64*astr::incident::epsilon*fmax(css,fabs(normal));
    if(normal-css < -tolerance) {
        const Real rate=Real(3.75)*fmax(Real(0.0),1-(normal/css)*(normal/css))*css/length;
        lodi[4]=rate*((at(pressure,i,j,k)-pinf*pout)/density/css)*sqrt(Real(2.0));
    }
    if(normal < -tolerance) {
        const Real target[5]={roinf*pout,roinf*pout*uinf,roinf*pout*vinf,roinf*pout*winf,
            pinf*pout/(gamma-1)+Real(.5)*roinf*pout*(uinf*uinf+vinf*vinf+winf*winf)};
        Real recovery[5]={};
        for(int m=0;m<5;++m) for(int c=0;c<5;++c) recovery[m]+=left[m][c]*(at(q,i,j,k,c+1)-target[c]);
        const Real rate=Real(.25)*(-normal)/length;
        for(int m=0;m<3;++m) lodi[m]=rate*recovery[m];
        lodi[4]+=rate*recovery[4];
        if(normal+css < -tolerance) lodi[3]=Real(.25)*(-normal-css)/length*recovery[3];
    }
    for(int c=0;c<5;++c) {
        Real value=0;
        for(int m=0;m<5;++m) value+=right[c][m]*lodi[m];
        decs[c]=value*jac+rest[c];
    }
    tangential_x_rhs(q,velocity,pressure,jacobian,metric,i,j,k,jm,km,npdcj,npdck,include_lower_z,decs);
    // The C++ RK update consumes negative divergence, as does CUDA Fortran.
    for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)-=decs[c];
}

__global__ void top_prediff_capture(AstrDeviceView rhs,AstrDeviceView before,int im,int jm,int km)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t nodes=static_cast<size_t>(im+1)*(km+1);
    if(id<nodes*5) {
        const int i=id%(im+1),k=(id/(im+1))%(km+1),c=id/nodes+1;
        at(before,i,0,k,c)=at(rhs,i,jm,k,c);
    }
    return;
}

__global__ void top_viscous_source(AstrDeviceView rhs,AstrDeviceView before,AstrDeviceView rho,
    AstrDeviceView velocity,AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView jacobian,
    int im,int jm,int km,Real gamma,Real mach,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(im+1)*(km+1)) return;
    const int i=id%(im+1),k=id/(im+1),j=jm;
    Real source[5],ddi[3],u[3],norm2=0,un=0;
    for(int c=0;c<5;++c) {
        source[c]=at(rhs,i,j,k,c+1)-at(before,i,0,k,c+1);
        if(!isfinite(source[c])) {
            atomicCAS(failure,0,1);
            return;
        }
    }
    for(int c=0;c<3;++c) {
        ddi[c]=at(metric,i,j,k,2,c+1);u[c]=at(velocity,i,j,k,c+1);
        norm2+=ddi[c]*ddi[c];un+=ddi[c]*u[c];
    }
    un/=sqrt(norm2);
    const Real css=sqrt(at(temperature,i,j,k))/mach,jac=at(jacobian,i,j,k);
    const Real lambda[5]={un,un,un,un+css,un-css};
    const Real tolerance=64*astr::incident::epsilon*fmax(css,fabs(un));
    Real right[5][5],left[5][5],projected[5]={};
    characteristic_matrices(at(rho,i,j,k),u,at(temperature,i,j,k),ddi,gamma,mach,right,left);
    for(int m=0;m<5;++m) if(lambda[m]<-tolerance) {
        for(int c=0;c<5;++c) projected[m]+=left[m][c]*source[c]/jac;
    }
    for(int c=0;c<5;++c) {
        Real correction=0;
        for(int m=0;m<5;++m) correction+=right[c][m]*projected[m];
        at(rhs,i,j,k,c+1)-=jac*correction;
    }
    return;
}

struct TopReference { Real q[5]; };

__global__ void top_outlet_corner(AstrDeviceView rhs,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView jacobian,
    int im,int jm,int km,Real gamma,Real mach)
{
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>km) return;
    Real ddi[3],u[3],norm2=0,normal=0,source[5],projected[5]={};
    for(int c=0;c<3;++c) {
        ddi[c]=at(metric,im,jm,k,2,c+1);u[c]=at(velocity,im,jm,k,c+1);
        norm2+=ddi[c]*ddi[c];normal+=ddi[c]*u[c];
    }
    normal/=sqrt(norm2);
    const Real css=sqrt(at(temperature,im,jm,k))/mach,jac=at(jacobian,im,jm,k);
    const Real lambda[5]={normal,normal,normal,normal+css,normal-css};
    const Real tolerance=64*astr::incident::epsilon*fmax(css,fabs(normal));
    Real right[5][5],left[5][5];
    characteristic_matrices(at(rho,im,jm,k),u,at(temperature,im,jm,k),ddi,gamma,mach,right,left);
    for(int c=0;c<5;++c) source[c]=at(rhs,im,jm,k,c+1);
    for(int m=0;m<5;++m) if(lambda[m]<-tolerance)
        for(int c=0;c<5;++c) projected[m]+=left[m][c]*source[c]/jac;
    for(int c=0;c<5;++c) {
        Real correction=0;
        for(int m=0;m<5;++m) correction+=right[c][m]*projected[m];
        at(rhs,im,jm,k,c+1)=source[c]-jac*correction;
    }
    return;
}

__global__ __launch_bounds__(astr::block_threads)
void top_mach2_partial(AstrDeviceView velocity,AstrDeviceView temperature,AstrDeviceView metric,
    int im,int jm,size_t count,Real mach,Real *result,int *failure)
{
    __shared__ Real scratch[astr::block_threads];
    const int lane=threadIdx.x;
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+lane;
    Real value=0;
    if(id<count) {
        const int i=id%(im+1),k=id/(im+1);
        Real norm2=0,un=0;
        for(int c=1;c<=3;++c) {
            const Real m=at(metric,i,jm,k,2,c);
            norm2+=m*m;un+=m*at(velocity,i,jm,k,c);
        }
        const Real css=sqrt(at(temperature,i,jm,k))/mach;
        value=un*un*(Real(1.0)/norm2)/css/css;
        if(!(isfinite(value) && value>=0 && norm2>0 && css>0)) atomicCAS(failure,0,1);
    }
    scratch[lane]=value;__syncthreads();
    for(int stride=astr::block_threads/2;stride>0;stride/=2) {
        if(lane<stride) scratch[lane]=fmax(scratch[lane],scratch[lane+stride]);
        __syncthreads();
    }
    if(lane==0) {
#ifdef ASTR_REAL32
        atomicMax(reinterpret_cast<unsigned int *>(result),__float_as_uint(scratch[0]));
#else
        auto *bits=reinterpret_cast<unsigned long long *>(result);
        unsigned long long previous=atomicCAS(bits,0ULL,0ULL);
        while(__longlong_as_double(previous)<scratch[0]) {
            const unsigned long long assumed=previous;
            previous=atomicCAS(bits,assumed,static_cast<unsigned long long>(__double_as_longlong(scratch[0])));
            if(previous==assumed) break;
        }
#endif
    }
    return;
}

__global__ void farfield52_rhs(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView jacobian,AstrDeviceView metric,AstrDeviceView rhs,AstrDeviceView target_plane,
    int im,int jm,int km,int npdci,int npdck,int policy,bool shock,bool keep_corner,
    Real gamma,Real mach,Real length,Real pressure_target,TopReference target,Real max_mach2)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(im+1)*(km+1)) return;
    const int i=id%(im+1),j=jm,k=id/(im+1);
    const bool left_end=i==0 && (npdci==1 || npdci==4),right_end=i==im && (npdci==2 || npdci==4);
    if(policy==2 && keep_corner && left_end) return;
    Real ddi[3],u[3],norm2=0,normal=0;
    for(int c=0;c<3;++c) {
        ddi[c]=at(metric,i,j,k,2,c+1);u[c]=at(velocity,i,j,k,c+1);
        norm2+=ddi[c]*ddi[c];normal+=ddi[c]*u[c];
    }
    normal/=sqrt(norm2);
    const Real css=sqrt(at(temperature,i,j,k))/mach,jac=at(jacobian,i,j,k),density=at(rho,i,j,k);
    Real right[5][5],left[5][5],f0[5],f1[5],f2[5],rest[5]={},decs[5],lodi[5]={};
    characteristic_matrices(density,u,at(temperature,i,j,k),ddi,gamma,mach,right,left);
    transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k,2,f0);
    transformed_flux(q,velocity,pressure,jacobian,metric,i,j-1,k,2,f1);
    transformed_flux(q,velocity,pressure,jacobian,metric,i,j-2,k,2,f2);
    for(int c=0;c<5;++c) decs[c]=Real(1.5)*f0[c]-2*f1[c]+Real(.5)*f2[c];
    if(policy==2 && keep_corner && right_end) {
        for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)-=decs[c];
        return;
    }
    for(int axis=1;axis<=3;++axis) {
        const Real derivative=Real(1.5)*at(metric,i,j,k,2,axis)*jac
            -2*at(metric,i,j-1,k,2,axis)*at(jacobian,i,j-1,k)
            +Real(.5)*at(metric,i,j-2,k,2,axis)*at(jacobian,i,j-2,k);
        Real flux[5];cartesian_flux(q,velocity,pressure,i,j,k,axis,flux);
        for(int c=0;c<5;++c) rest[c]+=flux[c]*derivative;
    }
    for(int m=0;m<5;++m) {
        for(int c=0;c<5;++c) lodi[m]+=left[m][c]*(decs[c]-rest[c]);
        lodi[m]/=jac;
    }
    const Real lambda[5]={normal,normal,normal,normal+css,normal-css};
    const Real tolerance=policy==1 && shock ? Real(0.0) : 64*astr::incident::epsilon*fmax(css,fabs(normal));
    if(policy==2) {
        Real transverse[5]={};
        if(left_end) {
            transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k,1,f0);
            transformed_flux(q,velocity,pressure,jacobian,metric,i+1,j,k,1,f1);
            for(int c=0;c<5;++c) transverse[c]+=f1[c]-f0[c];
        } else if(right_end) {
            transformed_flux(q,velocity,pressure,jacobian,metric,i-1,j,k,1,f0);
            transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k,1,f1);
            for(int c=0;c<5;++c) transverse[c]+=f1[c]-f0[c];
        } else {
            transformed_flux(q,velocity,pressure,jacobian,metric,i-1,j,k,1,f0);
            transformed_flux(q,velocity,pressure,jacobian,metric,i+1,j,k,1,f1);
            for(int c=0;c<5;++c) transverse[c]+=Real(.5)*(f1[c]-f0[c]);
        }
        if(!physical_endpoint(k,km,npdck)) {
            transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k-1,3,f0);
            transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k+1,3,f1);
            for(int c=0;c<5;++c) transverse[c]+=Real(.5)*(f1[c]-f0[c]);
        }
        for(int c=0;c<5;++c) rest[c]+=transverse[c];
        for(int m=0;m<5;++m) if(lambda[m]<-tolerance) {
            Real source=0;
            for(int c=0;c<5;++c) source+=left[m][c]*rest[c];
            lodi[m]=-source/jac;
        }
    } else if(policy!=3) {
        Real delta[5],characteristic[5]={};
        for(int c=0;c<5;++c) delta[c]=at(q,i,j,k,c+1)-(shock ? at(target_plane,i,0,k,c+1) : target.q[c]);
        for(int m=0;m<5;++m) for(int c=0;c<5;++c) characteristic[m]+=left[m][c]*delta[c];
        if(policy==1) {
            const Real rate=normal>=-tolerance ? fmax(Real(0.0),Real(.25)*(1-max_mach2)*css/length) : Real(.25)*css/length;
            for(int m=0;m<5;++m) if(lambda[m]<-tolerance) lodi[m]=rate*characteristic[m];
        } else if(normal < -tolerance) {
            for(int m=0;m<5;++m) if(lambda[m]<-tolerance) lodi[m]=Real(.25)*css/length*characteristic[m];
        } else if(lambda[4]<-tolerance) {
            const Real rate=Real(.25)*fmax(Real(0.0),1-(normal/css)*(normal/css))*css/length;
            lodi[4]=rate*((at(pressure,i,j,k)-pressure_target)/density/css)*sqrt(Real(2.0));
        }
    }
    for(int c=0;c<5;++c) {
        Real value=0;
        for(int m=0;m<5;++m) value+=right[c][m]*lodi[m];
        decs[c]=value*jac+rest[c];
    }
    if(policy!=2 && !physical_endpoint(i,im,npdci)) {
        transformed_flux(q,velocity,pressure,jacobian,metric,i-1,j,k,1,f0);
        transformed_flux(q,velocity,pressure,jacobian,metric,i+1,j,k,1,f1);
        for(int c=0;c<5;++c) decs[c]+=Real(.5)*(f1[c]-f0[c]);
        if(!physical_endpoint(k,km,npdck)) {
            transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k-1,3,f0);
            transformed_flux(q,velocity,pressure,jacobian,metric,i,j,k+1,3,f1);
            for(int c=0;c<5;++c) decs[c]+=Real(.5)*(f1[c]-f0[c]);
        }
    }
    for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)-=decs[c];
    return;
}

__global__ void incident_complete_rhs(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView jacobian,
    AstrDeviceView rhs,AstrDeviceView exterior,int im,int jm,int km,int npdci,Real gamma,Real mach,int *error)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=(im+1)*(km+1)) return;
    const int i=id%(im+1),k=id/(im+1);
    if(i==0&&(npdci==1||npdci==4)) {
        for(int c=1;c<=5;++c) at(rhs,i,jm,k,c)=0;
        return;
    }
    Real ddi[3],u[3],right_matrix[5][5],left_matrix[5][5];
    for(int c=0;c<3;++c) { ddi[c]=at(metric,i,jm,k,2,c+1);u[c]=at(velocity,i,jm,k,c+1); }
    if(i==im&&(npdci==2||npdci==4)&&u[0]<sqrt(at(temperature,i,jm,k))/mach) {
        atomicMax(error,4);return;
    }
    characteristic_matrices(at(rho,i,jm,k),u,at(temperature,i,jm,k),ddi,gamma,mach,right_matrix,left_matrix);
    using Real=astr::numerical_real;
    Real right[25],left[25],face[5],external[5],normal_metric[3],rate[5],lambda[5],reference_rate[5]={},local_rhs[5];
    for(int c=0;c<5;++c) {
        face[c]=at(q,i,jm,k,c+1);external[c]=astr::load_value(exterior,i,0,k,c+1);
        local_rhs[c]=at(rhs,i,jm,k,c+1);
        for(int m=0;m<5;++m) { right[c+5*m]=right_matrix[c][m];left[c+5*m]=left_matrix[c][m]; }
    }
    for(int c=0;c<3;++c) normal_metric[c]=ddi[c];
    bool incoming[5];
    int status=astr::incident::incoming_rate(face,external,Real(gamma),normal_metric,
        astr::load_value(exterior,i,0,k,6),left,reference_rate,rate,lambda,incoming);
    if(status==0&&(incoming[0]||incoming[1]||incoming[2])) {
        Real states[15],vel[9],p[3],jac[3],metrics[9],normal_lodi[5];
        for(int l=0;l<3;++l) {
            const int j=jm-l;
            for(int c=0;c<5;++c) states[c+5*l]=at(q,i,j,k,c+1);
            for(int c=0;c<3;++c) { vel[c+3*l]=at(velocity,i,j,k,c+1);metrics[c+3*l]=at(metric,i,j,k,2,c+1); }
            p[l]=at(pressure,i,j,k);jac[l]=at(jacobian,i,j,k);
        }
        status=astr::incident::normal_characteristics(states,vel,p,jac,metrics,left,normal_lodi);
        if(status==0) astr::incident::convective_compatibility(local_rhs,at(jacobian,i,jm,k),left,normal_lodi,incoming,rate);
    }
    if(status==0) status=astr::incident::apply_complete_rhs(local_rhs,at(jacobian,i,jm,k),right,left,incoming,rate);
    if(status!=0) { atomicMax(error,status);return; }
    for(int c=0;c<5;++c) at(rhs,i,jm,k,c+1)=local_rhs[c];
}

__global__ void high_open_lower_wall(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView speed,
    int im,size_t count,int code,bool forced,bool outlet_only,Real tw,Real c2,Real c6,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=outlet_only ? im : id%(im+1),k=outlet_only ? id : id/(im+1);
    Real line[7],u[3]={};
    for(int s=0;s<7;++s) line[s]=at(pressure,i,s+1,k);
    const Real pw=astr::high_wall::neumann7(line);
    if(code==421) {
        for(int s=0;s<7;++s) line[s]=at(temperature,i,s+1,k);
        tw=astr::high_wall::neumann7(line);
        for(int c=0;c<3;++c) {
            for(int s=0;s<7;++s) line[s]=at(velocity,i,s+1,k,c+1);
            u[c]=astr::high_wall::neumann7(line);
        }
    }
    Real normal[3],norm=0,vn=0;
    for(int c=0;c<3;++c) {normal[c]=at(metric,i,0,k,2,c+1);norm+=normal[c]*normal[c];}
    norm=sqrt(norm);
    for(int c=0;c<3;++c) {normal[c]/=norm;vn+=u[c]*normal[c];}
    const Real value=forced ? at(speed,i,0,k) : 0,den=pw/tw*c2;
    for(int c=0;c<3;++c) {
        u[c]=u[c]-vn*normal[c]+value*normal[c];
        at(velocity,i,0,k,c+1)=u[c];at(q,i,0,k,c+2)=den*u[c];
    }
    at(rho,i,0,k)=den;at(pressure,i,0,k)=pw;at(temperature,i,0,k)=tw;at(q,i,0,k,1)=den;
    at(q,i,0,k,5)=pw*c6+Real(.5)*den*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    if(!(isfinite(den)&&den>0&&isfinite(pw)&&pw>0&&isfinite(tw)&&tw>0&&isfinite(norm)&&norm>0))
        atomicCAS(failure,0,1);
}

__device__ void high_open_basis(AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView temperature,
    AstrDeviceView metric,int i,int j,int k,int axis,Real gamma,Real mach,
    astr::numerical_real right[25],astr::numerical_real left[25])
{
    Real u[3],ddi[3],r[5][5],l[5][5];
    for(int c=0;c<3;++c) {u[c]=at(velocity,i,j,k,c+1);ddi[c]=at(metric,i,j,k,axis+1,c+1);}
    characteristic_matrices(at(rho,i,j,k),u,at(temperature,i,j,k),ddi,gamma,mach,r,l);
    for(int c=0;c<5;++c) for(int m=0;m<5;++m) {right[c+5*m]=r[c][m];left[c+5*m]=l[c][m];}
}

__device__ void high_open_normal(AstrDeviceView q,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView jacobian,AstrDeviceView metric,int i,int j,int k,int axis,
    astr::numerical_real derivative[5],astr::numerical_real rest[5])
{
    using Real=astr::numerical_real;
    Real states[40],vel[24],prs[8],jac[8],metrics[24];
    for(int s=0;s<8;++s) {
        int p[3]={i,j,k};p[axis]-=s;
        for(int c=0;c<5;++c) states[c+5*s]=at(q,p[0],p[1],p[2],c+1);
        for(int c=0;c<3;++c) {vel[c+3*s]=at(velocity,p[0],p[1],p[2],c+1);metrics[c+3*s]=at(metric,p[0],p[1],p[2],axis+1,c+1);}
        prs[s]=at(pressure,p[0],p[1],p[2]);jac[s]=at(jacobian,p[0],p[1],p[2]);
    }
    astr::high_open::normal_terms6(states,vel,prs,jac,metrics,derivative,rest);
}

__global__ void high_open_complete_rhs(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView jacobian,
    AstrDeviceView rhs,AstrDeviceView exterior,int im,int jm,int km,int lower_x,int lower_y,int upper_x,int upper_y,
    Real gamma,Real mach,Real length,Real roinf,Real uinf,Real vinf,Real winf,Real pinf,Real pout,int *error)
{
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    const int right_nodes=upper_x*(jm+1)*(km+1);
    int i,j,k;
    if(id<right_nodes) {i=im;j=id%(jm+1);k=id/(jm+1);}
    else {
        id-=right_nodes;
        if(!upper_y||id>=(im+1)*(km+1)) return;
        i=id%(im+1);j=jm;k=id/(im+1);
        if(upper_x&&i==im) return;
    }
    if((lower_x&&i==0)||(lower_y&&j==0)) return;
    using Real=astr::numerical_real;
    Real right[25],left[25],derivative[5],rest[5],lodi[5],corrected[5],residual[5],local_rhs[5],metric_line[3];
    const Real jac=at(jacobian,i,j,k),css=sqrt(at(temperature,i,j,k))/mach;
    Real u[3];for(int c=0;c<3;++c) u[c]=at(velocity,i,j,k,c+1);
    if(upper_x&&i==im) {
        high_open_basis(rho,velocity,temperature,metric,i,j,k,0,gamma,mach,right,left);
        high_open_normal(q,velocity,pressure,jacobian,metric,i,j,k,0,derivative,rest);
        for(int c=0;c<5;++c) residual[c]=derivative[c]-rest[c];
        for(int m=0;m<5;++m) {lodi[m]=astr::incident::row_dot(left,m,residual)/jac;corrected[m]=lodi[m];}
        Real un=0,norm2=0;
        for(int c=0;c<3;++c) {const Real a=at(metric,i,j,k,1,c+1);un+=a*u[c];norm2+=a*a;}
        un/=sqrt(norm2);
        const Real tolerance=Real(64)*astr::incident::epsilon*fmax(css,fabs(un));
        if(un-css < -tolerance) {
            const Real scale=Real(Real(3.75))*fmax(Real(0),Real(1)-(un/css)*(un/css))*css/Real(length);
            corrected[4]=scale*(at(pressure,i,j,k)-Real(pinf)*Real(pout))/at(rho,i,j,k)/css*sqrt(Real(2));
        }
        if(un < -tolerance) {
            const Real target[5]={Real(roinf*pout),Real(roinf*pout*uinf),Real(roinf*pout*vinf),Real(roinf*pout*winf),
                Real(pinf*pout/(gamma-1)+Real(.5)*roinf*pout*(uinf*uinf+vinf*vinf+winf*winf))};
            Real dq[5];for(int c=0;c<5;++c) residual[c]=at(q,i,j,k,c+1)-target[c];
            for(int m=0;m<5;++m) dq[m]=astr::incident::row_dot(left,m,residual);
            const Real recovery=Real(Real(.25))*(-un)/Real(length);
            for(int m=0;m<3;++m) corrected[m]=recovery*dq[m];
            corrected[4]+=recovery*dq[4];
            if(un+css < -tolerance) corrected[3]=Real(Real(.25))*(-un-css)/Real(length)*dq[3];
        }
        for(int c=0;c<5;++c) residual[c]=lodi[c]-corrected[c];
        for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)+=jac*astr::incident::row_dot(right,c,residual);
    }
    if(!upper_y||j!=jm) return;
    if(upper_x&&i==im&&u[0]<css) {atomicMax(error,4);return;}
    high_open_basis(rho,velocity,temperature,metric,i,j,k,1,gamma,mach,right,left);
    Real primitive[40],external[5],face[5],rate[5],lambda[5],zero[5]={};bool incoming[5];
    primitive[0]=at(rho,i,j,k);primitive[4]=at(pressure,i,j,k);
    for(int c=0;c<3;++c) {primitive[c+1]=u[c];metric_line[c]=at(metric,i,j,k,2,c+1);}
    for(int s=1;s<8;++s) for(int c=0;c<5;++c) primitive[c+5*s]=astr::load_value(exterior,i,0,k,5*(s-1)+c+1);
    for(int c=0;c<5;++c) {face[c]=at(q,i,j,k,c+1);external[c]=astr::load_value(exterior,i,0,k,36+c);}
    const Real hn=astr::load_value(exterior,i,0,k,41);
    int status=astr::incident::incoming_rate(face,external,Real(gamma),metric_line,hn,left,zero,rate,lambda,incoming);
    if(status==0) {
        high_open_normal(q,velocity,pressure,jacobian,metric,i,j,k,1,derivative,rest);
        for(int c=0;c<5;++c) {residual[c]=derivative[c]-rest[c];local_rhs[c]=at(rhs,i,j,k,c+1);}
        for(int m=0;m<5;++m) lodi[m]=astr::incident::row_dot(left,m,residual)/jac;
        astr::high_open::incoming_rate7(primitive,Real(gamma),hn,left,lambda,incoming,local_rhs,jac,lodi,
            astr::load_value(exterior,i,0,k,42)!=0,rate);
        status=astr::incident::apply_complete_rhs(local_rhs,jac,right,left,incoming,rate);
        if(status==0) for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)=local_rhs[c];
    }
    if(status) atomicMax(error,status);
}

__global__ void high_open_zero_prescribed(AstrDeviceView rhs,int im,int jm,int km,int lower_x,int lower_y,int width)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=width*(km+1)) return;
    const int a=id%width,k=id/width;
    for(int c=1;c<=5;++c) {
        if(lower_x&&a<=jm) at(rhs,0,a,k,c)=0;
        if(lower_y&&a<=im) at(rhs,a,0,k,c)=0;
    }
}

}

extern "C" int astr_device_nscbc_high_open_prepare(void *parent,const Real *target,size_t bytes)
{
    if(!parent||(!target&&bytes)||(target&&!bytes)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!s.nscbc_incident_status) {
        const cudaError_t allocated=cudaMalloc(&s.nscbc_incident_status,sizeof(int));
        if(allocated!=cudaSuccess) return static_cast<int>(allocated);
    }
    if(!target) return 0;
    const size_t expected=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1)*42*sizeof(Real);
    if(bytes!=expected) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.nscbc_top_target.view.data) {
        const int upper[3]={s.upper[0],0,s.upper[2]};
        const int status=astr::allocate_field(s.nscbc_top_target,upper,0,4,42,1,sizeof(Real));
        if(status) return status;
    }
    if(s.nscbc_top_target.bytes!=expected) return ASTR_DEVICE_INVALID_ARGUMENT;
    const cudaError_t copied=cudaMemcpyAsync(s.nscbc_top_target.view.data,target,bytes,cudaMemcpyHostToDevice,s.stream);
    return copied==cudaSuccess ? static_cast<int>(cudaStreamSynchronize(s.stream)) : static_cast<int>(copied);
}

extern "C" int astr_device_nscbc_high_open_complete(void *parent,const int low[3],const int high[3],
    Real gamma,Real mach,Real length,Real roinf,Real uinf,Real vinf,Real winf,
    Real pinf,Real pout,int *boundary_status)
{
    if(!parent||!low||!high||!boundary_status) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready||!s.rhs_ready||!s.nscbc_incident_status||(high[1]&&!s.nscbc_top_target.view.data))
        return ASTR_DEVICE_NOT_ALLOCATED;
    const cudaError_t zeroed=cudaMemsetAsync(s.nscbc_incident_status,0,sizeof(int),s.stream);
    if(zeroed!=cudaSuccess) return static_cast<int>(zeroed);
    const size_t count=(high[0]*static_cast<size_t>(s.upper[1]+1)+high[1]*static_cast<size_t>(s.upper[0]+1))*(s.upper[2]+1);
    if(count) {
        high_open_complete_rhs<<<(count+127)/128,128,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
            s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
            s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_RHS].view,
            s.nscbc_top_target.view,s.upper[0],s.upper[1],s.upper[2],low[0],low[1],high[0],high[1],
            gamma,mach,length,roinf,uinf,vinf,winf,pinf,pout,s.nscbc_incident_status);
        const int status=astr::finish_launch(s);if(status) return status;
    }
    if(low[0]||low[1]) {
        const int width=s.upper[0]>s.upper[1] ? s.upper[0]+1 : s.upper[1]+1;
        const size_t nodes=static_cast<size_t>(width)*(s.upper[2]+1);
        high_open_zero_prescribed<<<(nodes+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_RHS].view,
            s.upper[0],s.upper[1],s.upper[2],low[0],low[1],width);
        const int status=astr::finish_launch(s);if(status) return status;
    }
    const cudaError_t copied=cudaMemcpyAsync(boundary_status,s.nscbc_incident_status,sizeof(int),cudaMemcpyDeviceToHost,s.stream);
    return copied==cudaSuccess ? static_cast<int>(cudaStreamSynchronize(s.stream)) : static_cast<int>(copied);
}

extern "C" int astr_device_high_open_wall_y(void *parent,int code,int forcing,int outlet_only,Real tw,Real c2,Real c6)
{
    if(!parent||(code!=41&&code!=421)||forcing<0||forcing>1||outlet_only<0||outlet_only>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready||(forcing&&!s.lower_wall_velocity.view.data)) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=static_cast<size_t>(outlet_only ? 1 : s.upper[0]+1)*(s.upper[2]+1);
    high_open_lower_wall<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.lower_wall_velocity.view,s.upper[0],count,code,forcing!=0,outlet_only!=0,
        tw,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_incident_prepare(void *parent,const Real *target,size_t bytes)
{
    if(!parent||!target) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t expected=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1)*6*sizeof(Real);
    if(bytes!=expected) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.nscbc_top_target.view.data) {
        const int upper[3]={s.upper[0],0,s.upper[2]};
        const int status=astr::allocate_field(s.nscbc_top_target,upper,0,4,6,1,sizeof(Real));
        if(status) return status;
    }
    if(s.nscbc_top_target.bytes!=expected) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.nscbc_incident_status) {
        const cudaError_t allocated=cudaMalloc(&s.nscbc_incident_status,sizeof(int));
        if(allocated!=cudaSuccess) return static_cast<int>(allocated);
    }
    const cudaError_t copied=cudaMemcpyAsync(s.nscbc_top_target.view.data,target,bytes,cudaMemcpyHostToDevice,s.stream);
    return copied==cudaSuccess ? static_cast<int>(cudaStreamSynchronize(s.stream)) : static_cast<int>(copied);
}

extern "C" int astr_device_nscbc_incident_complete(void *parent,int npdci,Real gamma,Real mach,int *boundary_status)
{
    if(!parent||!boundary_status||npdci<1||npdci>4||!(gamma>1&&mach>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready||!s.rhs_ready||!s.nscbc_incident_status||!s.nscbc_top_target.view.data)
        return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const cudaError_t zeroed=cudaMemsetAsync(s.nscbc_incident_status,0,sizeof(int),s.stream);
    if(zeroed!=cudaSuccess) return static_cast<int>(zeroed);
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    incident_complete_rhs<<<(count+127)/128,128,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_RHS].view,
        s.nscbc_top_target.view,s.upper[0],s.upper[1],s.upper[2],npdci,gamma,mach,s.nscbc_incident_status);
    int status=astr::finish_launch(s);
    if(status) return status;
    const cudaError_t copied=cudaMemcpyAsync(boundary_status,s.nscbc_incident_status,sizeof(int),cudaMemcpyDeviceToHost,s.stream);
    return copied==cudaSuccess ? static_cast<int>(cudaStreamSynchronize(s.stream)) : static_cast<int>(copied);
}

extern "C" int astr_device_nscbc_top_target_upload(void *parent,const Real *target,size_t bytes)
{
    if(!parent || !target) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t expected=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1)*5*sizeof(Real);
    if(bytes!=expected) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.nscbc_top_target.view.data) {
        const int upper[3]={s.upper[0],0,s.upper[2]};
        const int status=astr::allocate_field(s.nscbc_top_target,upper,0,4,5);
        if(status!=0) return status;
    }
    const cudaError_t copied=cudaMemcpyAsync(s.nscbc_top_target.view.data,target,bytes,cudaMemcpyHostToDevice,s.stream);
    return copied==cudaSuccess ? static_cast<int>(cudaStreamSynchronize(s.stream)) : static_cast<int>(copied);
}

extern "C" int astr_device_nscbc_top_prediff_capture(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    s.nscbc_top_prediff_ready=false;
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!s.nscbc_top_prediff.view.data) {
        const int upper[3]={s.upper[0],0,s.upper[2]};
        const int status=astr::allocate_field(s.nscbc_top_prediff,upper,0,4,5);
        if(status!=0) return status;
    }
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1)*5;
    top_prediff_capture<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_RHS].view,s.nscbc_top_prediff.view,
        s.upper[0],s.upper[1],s.upper[2]);
    const int status=astr::finish_launch(s);
    s.nscbc_top_prediff_ready=status==0;
    return status;
}

extern "C" int astr_device_nscbc_top_viscous_source(void *parent,Real gamma,Real mach)
{
    if(!parent || !(gamma>1 && mach>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!s.nscbc_top_prediff_ready) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    top_viscous_source<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_RHS].view,s.nscbc_top_prediff.view,
        s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.upper[0],s.upper[1],s.upper[2],gamma,mach,s.fluid_failure);
    const int status=astr::finish_launch(s);
    if(status==0) s.nscbc_top_prediff_ready=false;
    return status;
}

extern "C" int astr_device_nscbc_top_filter_prepare(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.nscbc_top_plane.view.data) return 0;
    const int upper[3]={s.upper[0],0,s.upper[2]};
    return astr::allocate_field(s.nscbc_top_plane,upper,0,4,5);
}

extern "C" int astr_device_nscbc_top_filter(void *parent,int axis,int ntype)
{
    if(!parent || (axis!=0 && axis!=2) || ntype<1 || ntype>4) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.nscbc_top_plane.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if((ntype!=3 && s.upper[axis]<4) || (ntype!=4 && s.halo<3)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1)*5;
    farfield_filter<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.nscbc_top_plane.view,
        s.upper[0],s.upper[1],s.upper[2],axis,ntype);
    const int status=astr::finish_launch(s);
    if(status!=0) return status;
    farfield_filter_commit<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.nscbc_top_plane.view,
        s.upper[0],s.upper[1],s.upper[2]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_top_mach2(void *parent,Real mach,Real *value)
{
    if(!parent || !value || !(mach>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!s.nscbc_top_mach) {
        const cudaError_t allocated=cudaMalloc(&s.nscbc_top_mach,sizeof(Real));
        if(allocated!=cudaSuccess) return static_cast<int>(allocated);
    }
    const cudaError_t zeroed=cudaMemsetAsync(s.nscbc_top_mach,0,sizeof(Real),s.stream);
    if(zeroed!=cudaSuccess) return static_cast<int>(zeroed);
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    top_mach2_partial<<<(count+astr::block_threads-1)/astr::block_threads,astr::block_threads,0,s.stream>>>(
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,
        s.upper[0],s.upper[1],count,mach,s.nscbc_top_mach,s.fluid_failure);
    const int status=astr::finish_launch(s);
    if(status!=0) return status;
    const cudaError_t copied=cudaMemcpyAsync(value,s.nscbc_top_mach,sizeof(Real),cudaMemcpyDeviceToHost,s.stream);
    return copied==cudaSuccess ? astr_device_fluid_check(parent) : static_cast<int>(copied);
}

extern "C" int astr_device_nscbc_top_rhs(void *parent,int npdci,int npdck,int policy,int shock,int keep_corner,
    Real gamma,Real mach,Real length,Real pressure_target,const Real target[5],Real max_mach2)
{
    if(!parent || !target || npdci<1 || npdci>4 || npdck<1 || npdck>4 || policy<0 || policy>3 ||
        shock<0 || shock>1 || (shock && policy!=1 && policy!=3) || !(gamma>1 && mach>0 && length>0))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready || (shock && policy!=3 && !s.nscbc_top_target.view.data)) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    TopReference reference{};
    for(int c=0;c<5;++c) reference.q[c]=target[c];
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    farfield52_rhs<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_RHS].view,
        s.nscbc_top_target.view,s.upper[0],s.upper[1],s.upper[2],npdci,npdck,policy,shock!=0,keep_corner!=0,
        gamma,mach,length,pressure_target,reference,max_mach2);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_outlet_prepare(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.nscbc_outlet_plane.view.data) return 0;
    const int upper[3]={0,s.upper[1],s.upper[2]};
    return astr::allocate_field(s.nscbc_outlet_plane,upper,0,4,5);
}

extern "C" int astr_device_nscbc_top_outlet_corner(void *parent,Real gamma,Real mach)
{
    if(!parent || !(gamma>1 && mach>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    top_outlet_corner<<<(s.upper[2]+256)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_RHS].view,
        s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.upper[0],s.upper[1],s.upper[2],gamma,mach);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_outlet_filter(void *parent,int npdcj)
{
    if(!parent || npdcj<1 || npdcj>4) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.nscbc_outlet_plane.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if((npdcj!=3 && s.upper[1]<4) || (npdcj!=4 && s.halo<3)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1)*5;
    outflow_filter_y<false><<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.nscbc_outlet_plane.view,s.upper[0],s.upper[1],s.upper[2],npdcj);
    const int status=astr::finish_launch(s);
    if(status!=0) return status;
    outflow_filter_commit<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.nscbc_outlet_plane.view,s.upper[0],s.upper[1],s.upper[2]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_outlet_filter_high_open(void *parent,int npdcj)
{
    if(!parent||npdcj<1||npdcj>4) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready||!s.nscbc_outlet_plane.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1)*5;
    outflow_filter_y<true><<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.nscbc_outlet_plane.view,s.upper[0],s.upper[1],s.upper[2],npdcj);
    const int status=astr::finish_launch(s);if(status) return status;
    outflow_filter_commit<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.nscbc_outlet_plane.view,s.upper[0],s.upper[1],s.upper[2]);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_outlet_x(void *parent,int npdcj,int npdck,int include_lower_z,
    Real gamma,Real mach,Real length,Real roinf,Real uinf,Real vinf,Real winf,Real pinf,Real pout)
{
    if(!parent || npdcj<1 || npdcj>4 || npdck<1 || npdck>4 || !(gamma>1 && mach>0 && length>0))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    outflow22_rhs<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_RHS].view,
        s.upper[0],s.upper[1],s.upper[2],npdcj,npdck,include_lower_z!=0,
        gamma,mach,length,roinf,uinf,vinf,winf,pinf,pout);
    return astr::finish_launch(s);
}

extern "C" int astr_device_nscbc_inlet_x(void *parent,int npdcj,int npdck,Real gamma,Real mach,Real length,
    Real roinf,Real uinf,Real vinf,Real winf,Real pinf)
{
    if(!parent || npdcj<1 || npdcj>4 || npdck<1 || npdck>4 || !(gamma>1 && mach>0 && length>0))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    inflow12_rhs<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_RHS].view,
        s.upper[1],s.upper[2],npdcj,npdck,gamma,mach,length,roinf,uinf,vinf,winf,pinf);
    return astr::finish_launch(s);
}

static int wall_basic_state(void *parent,int axis,int side,Real tw,Real c2,Real c6,bool adiabatic)
{
    if(!parent || axis<0 || axis>2 || side<0 || side>1 || !(tw>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[axis]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int n1=s.upper[(axis+1)%3]+1;
    const size_t count=static_cast<size_t>(n1)*(s.upper[(axis+2)%3]+1);
    wall_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        axis,side,s.upper[axis],n1,count,tw,c2,c6,adiabatic,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_wall41_basic_state(void *parent,int axis,int side,Real tw,Real c2,Real c6)
{
    return wall_basic_state(parent,axis,side,tw,c2,c6,false);
}

extern "C" int astr_device_wall42_basic_state(void *parent,int axis,int side,Real c2,Real c6)
{
    if(axis>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    return wall_basic_state(parent,axis,side,1,c2,c6,true);
}

extern "C" int astr_device_slip_mask_upload(void *parent,const int8_t *mask,size_t bytes)
{
    if(!parent || !mask) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.slip_lower_mask) {
        const cudaError_t status=cudaMalloc(&s.slip_lower_mask,bytes);
        if(status!=cudaSuccess) return static_cast<int>(status);
    }
    const cudaError_t copied=cudaMemcpyAsync(s.slip_lower_mask,mask,bytes,cudaMemcpyHostToDevice,s.stream);
    if(copied!=cudaSuccess) return static_cast<int>(copied);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_slip_wall_y(void *parent,int code,int side,Real tw,Real c2,Real c6)
{
    if(!parent || (code!=411 && code!=421) || side<0 || side>1 || (code==411 && !(tw>0)))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || (code==411 && side==0 && !s.slip_lower_mask)) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    slip_y_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.slip_lower_mask,code,side,s.upper[0],s.upper[1],count,tw,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_fixed_rti_y(void *parent,int side,Real c2,Real c6)
{
    if(!parent || side<0 || side>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    fixed_rti_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.upper[0],side ? s.upper[1] : 0,count,side!=0,c2,c6);
    return astr::finish_launch(s);
}

extern "C" int astr_device_ldc_top_lid_y(void *parent,Real c2,Real c6)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    ldc_lid_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.upper[0],s.upper[1],count,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_wall41_evolved_density_state(void *parent,int axis,int side,Real tw,Real c2,Real c6)
{
    if(!parent || axis<0 || axis>2 || side<0 || side>1 || !(tw>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const int first=axis==0 ? 1 : 0,second=axis==2 ? 1 : 2,n1=s.upper[first]+1;
    const size_t count=static_cast<size_t>(n1)*(s.upper[second]+1);
    evolved_density_wall_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,axis,side ? s.upper[axis] : 0,n1,count,tw,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_wall_forcing_prepare(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.lower_wall_velocity.view.data) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int upper[3]={s.upper[0],0,s.upper[2]};
    return astr::allocate_field(s.lower_wall_velocity,upper,0,3);
}

extern "C" int astr_device_wall_forcing_upload(void *parent,const Real *velocity,size_t bytes)
{
    if(!parent || !velocity) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.lower_wall_velocity.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=s.lower_wall_velocity.bytes) return ASTR_DEVICE_INVALID_ARGUMENT;
    cudaError_t status=cudaMemcpyAsync(s.lower_wall_velocity.view.data,velocity,bytes,cudaMemcpyHostToDevice,s.stream);
    if(status!=cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_wall41_forcing_y(void *parent,Real tw,Real c2,Real c6)
{
    if(!parent || !(tw>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.lower_wall_velocity.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    forced_lower_wall_state<false,true><<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.lower_wall_velocity.view,s.upper[0],count,tw,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_wall41_outlet_y(void *parent,int forcing,Real tw,Real c2,Real c6)
{
    if(!parent || forcing<0 || forcing>1 || !(tw>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || (forcing && !s.lower_wall_velocity.view.data)) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=s.upper[2]+1;
    if(forcing) {
        forced_lower_wall_state<true,true><<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
            s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
            s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,s.lower_wall_velocity.view,
            s.upper[0],count,tw,c2,c6,s.fluid_failure);
    } else {
        forced_lower_wall_state<true,false><<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
            s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
            s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,s.lower_wall_velocity.view,
            s.upper[0],count,tw,c2,c6,s.fluid_failure);
    }
    return astr::finish_launch(s);
}

extern "C" int astr_device_wall_adiabatic_evolved_state(void *parent,int axis,int side,int slip,Real c2,Real c6)
{
    if(!parent || axis<0 || axis>1 || side<0 || side>1 || slip<0 || slip>1 || (slip && axis!=1))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[axis]<8) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int first=axis==0 ? 1 : 0,second=axis==2 ? 1 : 2,n1=s.upper[first]+1;
    const size_t count=static_cast<size_t>(n1)*(s.upper[second]+1);
    evolved_adiabatic_wall_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,axis,side,s.upper[axis],n1,count,slip!=0,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

static int extrapolate_basic(void *parent,int axis,int side,Real c2,Real c6,bool symmetric)
{
    if(!parent || axis<0 || axis>2 || side<0 || side>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[axis]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int n1=s.upper[(axis+1)%3]+1;
    const size_t count=static_cast<size_t>(n1)*(s.upper[(axis+2)%3]+1);
    if(symmetric) extrapolate_state<true><<<(count+255)/256,256,0,s.stream>>>(
        s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_METRIC].view,axis,side,s.upper[axis],n1,count,c2,c6,s.fluid_failure);
    else extrapolate_state<false><<<(count+255)/256,256,0,s.stream>>>(
        s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_METRIC].view,axis,side,s.upper[axis],n1,count,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_zeroextrap_basic(void *parent,int axis,int side,Real c2,Real c6)
{
    return extrapolate_basic(parent,axis,side,c2,c6,false);
}

extern "C" int astr_device_symmetry_basic(void *parent,int axis,int side,Real c2,Real c6)
{
    return extrapolate_basic(parent,axis,side,c2,c6,true);
}

extern "C" int astr_device_profile_inflow_prepare(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &state=*static_cast<astr::State *>(parent);
    if(!state.fluid_ready || !state.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(state.fields[ASTR_DEVICE_INFLOW_TARGET].view.data) return 0;
    const int plane[3]={0,state.upper[1],state.upper[2]};
    return astr::allocate_field(state.fields[ASTR_DEVICE_INFLOW_TARGET],plane,0,4,9);
}

extern "C" int astr_device_profile_inflow_x(void *parent,int mach_pressure,Real mach,Real c2,Real c6)
{
    if(!parent || (mach_pressure!=0 && mach_pressure!=1) || !(mach>Real(0.0))) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fields[ASTR_DEVICE_INFLOW_TARGET].view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    profile_inflow_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_INFLOW_TARGET].view,
        s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_RHS].view,
        s.upper[1]+1,count,mach_pressure!=0,mach,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_inflow_slices_prepare(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fields[ASTR_DEVICE_INFLOW_TARGET].view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.inflow_slices.view.data) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int upper[3]={0,s.upper[1],s.upper[2]};
    return astr::allocate_field(s.inflow_slices,upper,0,5,5,4);
}

static int transfer_inflow_slice(void *parent,int slot,Real *values,size_t bytes,cudaMemcpyKind kind)
{
    if(!parent || !values || slot<0 || slot>3) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.inflow_slices.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=s.inflow_slices.bytes/4) return ASTR_DEVICE_INVALID_ARGUMENT;
    Real *device=static_cast<Real *>(s.inflow_slices.view.data)+slot*s.inflow_slices.view.stride[4];
    const cudaError_t status=kind==cudaMemcpyHostToDevice ?
        cudaMemcpyAsync(device,values,bytes,kind,s.stream) : cudaMemcpyAsync(values,device,bytes,kind,s.stream);
    if(status!=cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_inflow_slice_upload(void *parent,int slot,const Real *values,size_t bytes)
{
    return transfer_inflow_slice(parent,slot,const_cast<Real *>(values),bytes,cudaMemcpyHostToDevice);
}

extern "C" int astr_device_inflow_slice_download(void *parent,int slot,Real *values,size_t bytes)
{
    return transfer_inflow_slice(parent,slot,values,bytes,cudaMemcpyDeviceToHost);
}

extern "C" int astr_device_dynamic_inflow_x(void *parent,const int slots[4],Real time0,Real delta,Real time,
    int mach_pressure,Real mach,Real c2,Real c6)
{
    if(!parent || !slots || !(delta>0) || !(mach>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int n=0;n<4;++n) if(slots[n]<0 || slots[n]>3) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.inflow_slices.view.data || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    dynamic_inflow_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_INFLOW_TARGET].view,s.inflow_slices.view,
        s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_RHS].view,
        s.upper[1]+1,count,slots[0],slots[1],slots[2],slots[3],delta,time-time0,mach_pressure!=0,mach,c2,c6,s.fluid_failure);
    return astr::finish_launch(s);
}

extern "C" int astr_device_farfield51_upper_basic(void *parent,Real c2,Real c6)
{
    return extrapolate_basic(parent,1,1,c2,c6,false);
}

extern "C" int astr_device_outflow21_x_basic(void *parent,Real mach,Real c2,Real c6,Real pressure_target)
{
    if(!parent || !(mach>Real(0.0))) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || !s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    outflow21_state<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_RHS].view,s.upper[0],s.upper[1]+1,count,
        mach,c2,c6,pressure_target,s.fluid_failure);
    return astr::finish_launch(s);
}
