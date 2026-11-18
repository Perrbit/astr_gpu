#include <cmath>
#include "device_state_internal.cuh"
#include "upwind_operator.h"
#include "shock_sensor_internal.cuh"
#include "high_open_math.cuh"

namespace {
using astr::State;
using astr::at;
using Real=astr::numerical_real;
constexpr int threads=astr::block_threads;
struct Box { int low[3],high[3],upper[3],first[3],last[3]; };

__device__ Real minmod2(Real a,Real b)
{
    if(a>Real(0.0) && b>Real(0.0)) return fmin(fabs(a),fabs(b));
    if(a<Real(0.0) && b<Real(0.0)) return -fmin(fabs(a),fabs(b));
    return Real(0.0);
}

__device__ Real minmod4(Real a,Real b,Real c,Real d)
{
    const Real value=fmin(fmin(fabs(a),fabs(b)),fmin(fabs(c),fabs(d)));
    if(a>Real(0.0) && b>Real(0.0) && c>Real(0.0) && d>Real(0.0)) return value;
    if(a<Real(0.0) && b<Real(0.0) && c<Real(0.0) && d<Real(0.0)) return -value;
    return Real(0.0);
}

__device__ Real limit_mp(const Real *u,Real linear,int center,bool active)
{
    if(!active) return linear;
    const Real dm=u[center-2]-Real(2.0)*u[center-1]+u[center];
    const Real d0=u[center-1]-Real(2.0)*u[center]+u[center+1];
    const Real dp=u[center]-Real(2.0)*u[center+1]+u[center+2];
    const Real dhm=minmod4(Real(4.0)*dm-d0,Real(4.0)*d0-dm,dm,d0);
    const Real dhp=minmod4(Real(4.0)*d0-dp,Real(4.0)*dp-d0,d0,dp);
    const Real ul=u[center]+Real(4.0)*(u[center]-u[center-1]);
    const Real av=Real(0.5)*(u[center]+u[center+1]);
    const Real md=av-Real(0.5)*dhp;
    const Real lc=u[center]+Real(0.5)*(u[center]-u[center-1])+Real(1.333333333333333)*dhm;
    const Real minimum=fmax(fmin(fmin(u[center],u[center+1]),md),fmin(fmin(u[center],ul),lc));
    const Real maximum=fmin(fmax(fmax(u[center],u[center+1]),md),fmax(fmax(u[center],ul),lc));
    return linear+minmod2(minimum-linear,maximum-linear);
}

__device__ Real reconstruct(const Real f[8],int position,int axis,Box box,int scheme,Real weight,bool shock)
{
    const int end=box.upper[axis];
    if((box.low[axis] && position==0) || (box.high[axis] && position==end-1))
        return Real(0.5)*(f[3]+f[4]);
    if((box.low[axis] && position==1) || (box.high[axis] && position==end-2))
        return (-f[2]+Real(5.0)*f[3]+Real(2.0)*f[4])/Real(6.0);
    const bool fifth=(box.low[axis] && position==2) || (box.high[axis] && position==end-3);
    Real linear;
    if(fifth) {
        if(scheme==3) linear=(Real(2.0)*f[1]-Real(13.0)*f[2]+Real(47.0)*f[3]+Real(27.0)*f[4]-Real(3.0)*f[5])/Real(60.0);
        else {
            const Real v=Real(1.666666666666667e-2)*weight;
            linear=(-v+Real(3.333333333333333e-2))*f[1]+(Real(5.0)*v-Real(2.166666666666667e-1))*f[2]
                +(-Real(10.0)*v+Real(7.833333333333333e-1))*f[3]+(Real(10.0)*v+Real(0.45))*f[4]
                +(-Real(5.0)*v-Real(5.e-2))*f[5]+v*f[6];
        }
    } else if(scheme==3) {
        linear=(-Real(3.0)*f[0]+Real(25.0)*f[1]-Real(101.0)*f[2]+Real(319.0)*f[3]+Real(214.0)*f[4]-Real(38.0)*f[5]+Real(4.0)*f[6])/Real(420.0);
    } else {
        const Real v=Real(3.571428571428571e-3)*weight;
        linear=(v-Real(7.142857142857143e-3))*f[0]+(-Real(7.0)*v+Real(5.952380952380952e-2))*f[1]
            +(Real(21.0)*v-Real(0.240476190476190))*f[2]+(-Real(35.0)*v+Real(0.759523809523809))*f[3]
            +(Real(35.0)*v+Real(0.509523809523809))*f[4]+(-Real(21.0)*v-Real(9.047619047619047e-2))*f[5]
            +(Real(7.0)*v+Real(9.523809523809525e-3))*f[6]-v*f[7];
    }
    const Real mp=f[3]+minmod2(f[4]-f[3],Real(4.0)*(f[3]-f[2]));
    return limit_mp(f,linear,3,(scheme==5 && !fifth) ? shock : (linear-f[3])*(linear-mp)>=Real(1.e-10));
}

__device__ void split(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView jac,const int p[3],int axis,int sign,
    Real gamma,Real mach,bool metric_consistent,Real f[5])
{
    const int i=p[0],j=p[1],k=p[2];
    const Real den=at(rho,i,j,k),prs=at(pressure,i,j,k),jloc=at(jac,i,j,k);
    Real u[3],dx[3];
    for(int c=0;c<3;++c) { u[c]=at(velocity,i,j,k,c+1);dx[c]=at(metric,i,j,k,axis+1,c+1); }
    const Real uu=dx[0]*u[0]+dx[1]*u[1]+dx[2]*u[2];
    const Real inverse_norm=Real(1.0)/sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
    const Real css=sqrt(at(temperature,i,j,k))/mach,csa=css/inverse_norm,lmach=uu/csa;
    if(lmach>=Real(1.0) || lmach<=-Real(1.0)) {
        if((lmach>=Real(1.0) && sign<0) || (lmach<=-Real(1.0) && sign>0)) {
            for(int c=0;c<5;++c) f[c]=Real(0.0);
        } else {
            f[0]=jloc*at(q,i,j,k,1)*uu;
            for(int c=0;c<3;++c) f[c+1]=jloc*(at(q,i,j,k,c+2)*uu+dx[c]*prs);
            f[4]=jloc*(at(q,i,j,k,5)+prs)*uu;
        }
        return;
    }
    const Real eps=metric_consistent ? Real(0.04)/inverse_norm : Real(0.04);
    const Real l1=uu,l4=uu+csa,l5=uu-csa;
    const Real lp1=Real(0.5)*(l1+sqrt(l1*l1+eps*eps));
    const Real lp4=Real(0.5)*(l4+sqrt(l4*l4+eps*eps));
    const Real lp5=Real(0.5)*(l5+sqrt(l5*l5+eps*eps));
    const Real v1=sign>0 ? lp1 : l1-lp1;
    const Real v2=sign>0 ? lp4-lp5 : (l4-lp4)-(l5-lp5);
    const Real v3=sign>0 ? Real(2.0)*lp1-lp4-lp5 : Real(2.0)*(l1-lp1)-(l4-lp4)-(l5-lp5);
    const Real gm2=Real(0.5)/gamma,v4=v1-v3*gm2,jro=jloc*den;
    f[0]=jro*v4;
    for(int c=0;c<3;++c) f[c+1]=jro*(v4*u[c]+v2*css*(dx[c]*inverse_norm)*gm2);
    const Real fhi=Real(0.5)*(gamma-Real(1.0))*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    f[4]=jloc*(v1*at(q,i,j,k,5)+den*(v2*uu*inverse_norm*css*gm2-v3*(fhi+css*css)*gm2/(gamma-Real(1.0))));
    return;
}

__device__ void roe(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView metric,const int left[3],int axis,Real gamma,Real lev[5][5],Real rev[5][5])
{
    int right[3]={left[0],left[1],left[2]};++right[axis];
    const int i=left[0],j=left[1],k=left[2],ir=right[0],jr=right[1],kr=right[2];
    const Real dl=at(rho,i,j,k),dr=at(rho,ir,jr,kr),w=sqrt(dl)/(sqrt(dl)+sqrt(dr)),w1=Real(1.0)-w;
    Real u[3],n[3],average[3];
    for(int c=0;c<3;++c) {
        u[c]=w*at(velocity,i,j,k,c+1)+w1*at(velocity,ir,jr,kr,c+1);
        average[c]=Real(0.5)*(at(metric,i,j,k,axis+1,c+1)+at(metric,ir,jr,kr,axis+1,c+1));
    }
    const Real kinetic=Real(0.5)*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    const Real hl=(at(q,i,j,k,5)+at(pressure,i,j,k))/dl;
    const Real hr=(at(q,ir,jr,kr,5)+at(pressure,ir,jr,kr))/dr;
    const Real enthalpy=w*hl+w1*hr,css=sqrt((gamma-Real(1.0))*(enthalpy-kinetic)),rcs=Real(1.0)/css;
    const Real inv=Real(1.0)/sqrt(average[0]*average[0]+average[1]*average[1]+average[2]*average[2]);
    for(int c=0;c<3;++c) n[c]=average[c]*inv;
    const Real ugp=u[0]*n[0]+u[1]*n[1]+u[2]*n[2];
    const Real b1=(gamma-Real(1.0))/(css*css),b2=Real(1.0)+Real(2.0)*b1*kinetic-b1*enthalpy;
    for(int r=0;r<5;++r) for(int c=0;c<5;++c) { lev[r][c]=Real(0.0);rev[r][c]=Real(0.0); }
    lev[0][0]=Real(0.5)*(b2+ugp*rcs);lev[4][0]=Real(0.5)*(b2-ugp*rcs);
    lev[0][4]=lev[4][4]=Real(0.5)*b1;
    rev[0][0]=rev[0][1]=rev[0][4]=Real(1.0);
    rev[4][0]=enthalpy-ugp*css;rev[4][4]=enthalpy+ugp*css;
    lev[1][0]=Real(1.0)-b2;lev[1][4]=-b1;rev[4][1]=enthalpy-Real(1.0)/b1;
    for(int c=0;c<3;++c) {
        lev[0][c+1]=-Real(0.5)*(b1*u[c]+n[c]*rcs);lev[4][c+1]=-Real(0.5)*(b1*u[c]-n[c]*rcs);
        lev[1][c+1]=b1*u[c];
        rev[c+1][0]=u[c]-css*n[c];rev[c+1][1]=u[c];rev[c+1][4]=u[c]+css*n[c];
    }
    const int basis=fabs(average[0])>Real(1.e-12) ? 0 : (fabs(average[1])>Real(1.e-12) ? 1 : 2);
    const Real inverse_basis=Real(1.0)/n[basis];
    int row=2;
    for(int tangent=0;tangent<3;++tangent) if(tangent!=basis) {
        lev[row][0]=(ugp*n[tangent]-u[tangent])*inverse_basis;
        for(int c=0;c<3;++c) lev[row][c+1]=c==basis ? -n[tangent]
            : ((c==tangent ? Real(1.0) : Real(0.0))-n[tangent]*n[c])*inverse_basis;
        rev[basis+1][row]=-n[tangent];rev[tangent+1][row]=n[basis];
        rev[4][row]=u[tangent]*n[basis]-u[basis]*n[tangent];++row;
    }
    return;
}

template<bool high_order,bool positivity_faces> __global__ __launch_bounds__(threads)
void interface_kernel(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView metric,AstrDeviceView jac,const int8_t *mask,const int8_t *faces,
    const int8_t *critical,AstrDeviceView flux,
    Box box,int axis,int scheme,bool characteristic,bool round_positive,Real gamma,Real mach,Real weight,bool metric_consistent,
    int nx,int ny,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int p[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
        --p[axis];
        bool active=true;
        for(int a=0;a<3;++a) active=active && p[a]>=box.first[a]-(a==axis ? 1 : 0) && p[a]<=box.last[a];
        if(!active) return;
        if(positivity_faces && ((box.low[axis] && p[axis]<0) ||
            (box.high[axis] && p[axis]>=box.upper[axis]))) return;
        if(high_order&&box.high[axis]&&p[axis]==box.upper[axis]) return;
        bool shock=true;
        if(mask) shock=astr::shock_interface_active(mask,faces,box.upper,p,axis);
        bool high_dissipation=false;
        if(critical) {
            int left[3]={p[0],p[1],p[2]},right[3]={p[0],p[1],p[2]};++right[axis];
            if(box.low[axis]) left[axis]=max(0,left[axis]);
            if(box.high[axis]) right[axis]=min(box.upper[axis],right[axis]);
            high_dissipation=critical[astr::offset(q,left[0],left[1],left[2])]!=0 ||
                critical[astr::offset(q,right[0],right[1],right[2])]!=0;
        }
        int boundary_row=-1,boundary_side=0;
        if(high_order) {
            const int type=box.low[axis] ? (box.high[axis] ? 4 : 1) : (box.high[axis] ? 2 : 3);
            astr::high_open::interface_position(p[axis],box.upper[axis],type,boundary_row,boundary_side);
        }
        const bool biased=boundary_row>=0&&!shock&&!high_dissipation;
        Real plus[5][8],minus[5][8],lev[5][5],rev[5][5],result[5];
        const bool decompose=characteristic && shock;
        if(decompose) roe(q,rho,velocity,pressure,metric,p,axis,gamma,lev,rev);
        for(int n=0;n<8;++n) {
            int lo[3]={p[0],p[1],p[2]},hi[3]={p[0],p[1],p[2]};
            lo[axis]+=n-3;hi[axis]+=4-n;
            if(box.low[axis]) {lo[axis]=max(0,lo[axis]);hi[axis]=max(0,hi[axis]);}
            if(box.high[axis]) {lo[axis]=min(box.upper[axis],lo[axis]);hi[axis]=min(box.upper[axis],hi[axis]);}
            if(biased) lo[axis]=hi[axis]=boundary_side==1 ? n : box.upper[axis]-n;
            Real fp[5],fm[5];
            split(q,rho,velocity,pressure,temperature,metric,jac,lo,axis,1,gamma,mach,metric_consistent,fp);
            split(q,rho,velocity,pressure,temperature,metric,jac,hi,axis,-1,gamma,mach,metric_consistent,fm);
            for(int m=0;m<5;++m) {
                plus[m][n]=decompose ? lev[m][0]*fp[0]+lev[m][1]*fp[1]+lev[m][2]*fp[2]+lev[m][3]*fp[3]+lev[m][4]*fp[4] : fp[m];
                minus[m][n]=decompose ? lev[m][0]*fm[0]+lev[m][1]*fm[1]+lev[m][2]*fm[2]+lev[m][3]*fm[3]+lev[m][4]*fm[4] : fm[m];
            }
        }
        for(int m=0;m<5;++m) {
            if(biased) {
                astr::numerical_real fp[8],fm[8];
                for(int n=0;n<8;++n) {fp[n]=plus[m][n];fm[n]=minus[m][n];}
                result[m]=astr::high_open::mp_flux6(fp,boundary_row,boundary_side==1)+
                    astr::high_open::mp_flux6(fm,boundary_row,boundary_side==-1);
                continue;
            }
            Real positive=high_dissipation ? plus[m][3] : reconstruct(plus[m],p[axis],axis,box,scheme,weight,shock);
            // Preserve the old split/fused MP1 contract: round F+ before adding F-.
            if(round_positive) positive=static_cast<Real>(static_cast<float>(positive));
            result[m]=positive+(high_dissipation ? minus[m][3] : reconstruct(minus[m],p[axis],axis,box,scheme,weight,shock));
        }
        for(int m=0;m<5;++m) astr::store_value(flux,p[0],p[1],p[2],decompose
            ? rev[m][0]*result[0]+rev[m][1]*result[1]+rev[m][2]*result[2]+rev[m][3]*result[3]+rev[m][4]*result[4] : result[m],m+1);
    }
    return;
}

template<bool high_order,bool upper_boundary_only=false> __global__ __launch_bounds__(threads)
void difference_kernel(AstrDeviceView flux,AstrDeviceView rhs,AstrDeviceView q,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView jacobian,AstrDeviceView metric,Box box,int axis,
    int nx,int ny,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int p[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
        for(int a=0;a<3;++a) if(p[a]<box.first[a] || p[a]>box.last[a]) return;
        if(upper_boundary_only && !((box.high[0] && p[0]==box.upper[0]) ||
            (box.high[1] && p[1]==box.upper[1]))) return;
        if(high_order&&box.high[axis]&&p[axis]==box.upper[axis]) {
            using Real=astr::numerical_real;
            Real states[40],vel[24],prs[8],jac[8],metrics[24],derivative[5],rest[5];
            for(int n=0;n<8;++n) {
                int node[3]={p[0],p[1],p[2]};node[axis]-=n;
                for(int c=0;c<5;++c) states[c+5*n]=at(q,node[0],node[1],node[2],c+1);
                for(int c=0;c<3;++c) {
                    vel[c+3*n]=at(velocity,node[0],node[1],node[2],c+1);
                    metrics[c+3*n]=at(metric,node[0],node[1],node[2],axis+1,c+1);
                }
                prs[n]=at(pressure,node[0],node[1],node[2]);jac[n]=at(jacobian,node[0],node[1],node[2]);
            }
            astr::high_open::normal_terms6(states,vel,prs,jac,metrics,derivative,rest);
            for(int c=0;c<5;++c) at(rhs,p[0],p[1],p[2],c+1)-=derivative[c];
            return;
        }
        int left[3]={p[0],p[1],p[2]};--left[axis];
        for(int c=1;c<=5;++c) at(rhs,p[0],p[1],p[2],c)+=astr::load_value(flux,left[0],left[1],left[2],c)
            -astr::load_value(flux,p[0],p[1],p[2],c);
    }
    return;
}
}

namespace {
int prepare(void *handle,int storage)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state=*static_cast<State *>(handle);
    if(!state.fluid_ready || !state.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(state.halo<4) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int element_bytes=storage==0 ? sizeof(Real) : sizeof(float);
    const auto &view=state.fields[ASTR_DEVICE_INTERFACE_FLUX].view;
    if(view.data) return view.element_bytes==element_bytes && state.upwind_storage==storage
        ? 0 : ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=astr::allocate_field(state.fields[ASTR_DEVICE_INTERFACE_FLUX],state.upper,1,4,5,1,element_bytes);
    if(!status) state.upwind_storage=storage;
    return status;
}
}

extern "C" int astr_device_upwind_prepare(void *handle) {return prepare(handle,0);}
extern "C" int astr_device_upwind_prepare_mixed_flux(void *handle) {return prepare(handle,1);}
extern "C" int astr_device_upwind_prepare_mixed_characteristic_flux(void *handle) {return prepare(handle,2);}

namespace {
template<bool high_order,bool faces_only> int launch_axis(State &s,Box box,int axis,int reconstruction,int characteristic,
    Real gamma,Real mach,Real weight,int metric_consistent)
{
    int shape[3]={s.upper[0]+1,s.upper[1]+1,s.upper[2]+1};++shape[axis];
    const size_t faces=static_cast<size_t>(shape[0])*shape[1]*shape[2];
    interface_kernel<high_order,faces_only><<<(faces+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.shock_mask,s.shock_faces_ready ? s.shock_faces : nullptr,s.critical_mask,s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,
        box,axis,reconstruction,characteristic!=0,s.upwind_storage==1,gamma,mach,weight,
        metric_consistent!=0,shape[0],shape[1],faces);
    const int status=astr::finish_launch(s);
    if(status) return status;
    if(faces_only) return 0;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    difference_kernel<high_order><<<(count+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_METRIC].view,
        box,axis,s.upper[0]+1,s.upper[1]+1,count);
    return astr::finish_launch(s);
}

int apply_axis(void *handle,int axis,int reconstruction,int characteristic,
    Real gamma,Real mach,Real weight,int metric_consistent,
    const int low[3],const int high[3],const int first[3],const int last[3],bool high_order,bool faces_only=false)
{
    if(!handle || axis<0 || axis>2 || (reconstruction!=3 && reconstruction!=5) ||
       (characteristic!=0 && characteristic!=1) || (metric_consistent!=0 && metric_consistent!=1) ||
       !std::isfinite(gamma) || gamma<=Real(1.0) || !std::isfinite(mach) || mach<=Real(0.0) || !std::isfinite(weight) ||
       !low || !high || !first || !last) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);Box box;
    for(int a=0;a<3;++a) {
        if((low[a]!=0 && low[a]!=1) || (high[a]!=0 && high[a]!=1) || first[a]<0 || last[a]>s.upper[a] || first[a]>last[a])
            return ASTR_DEVICE_INVALID_ARGUMENT;
        box.low[a]=low[a];box.high[a]=high[a];box.upper[a]=s.upper[a];box.first[a]=first[a];box.last[a]=last[a];
    }
    if(!s.fields[ASTR_DEVICE_INTERFACE_FLUX].view.data || ((characteristic || reconstruction==5) && !s.shock_mask))
        return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upwind_storage==1) {
        if(characteristic || reconstruction!=3) return ASTR_DEVICE_INVALID_ARGUMENT;
        for(int a=0;a<3;++a) if(low[a] || high[a]) return ASTR_DEVICE_INVALID_ARGUMENT;
    }
    if(s.upwind_storage==2 && (!characteristic || reconstruction!=3)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(faces_only) return high_order ? launch_axis<true,true>(s,box,axis,reconstruction,characteristic,gamma,mach,weight,metric_consistent)
        : launch_axis<false,true>(s,box,axis,reconstruction,characteristic,gamma,mach,weight,metric_consistent);
    return high_order ? launch_axis<true,false>(s,box,axis,reconstruction,characteristic,gamma,mach,weight,metric_consistent)
        : launch_axis<false,false>(s,box,axis,reconstruction,characteristic,gamma,mach,weight,metric_consistent);
}
}

extern "C" int astr_device_upwind_axis(void *handle,int axis,int reconstruction,int characteristic,
    Real gamma,Real mach,Real weight,int metric_consistent,
    const int low[3],const int high[3],const int first[3],const int last[3])
{
    return apply_axis(handle,axis,reconstruction,characteristic,gamma,mach,weight,metric_consistent,low,high,first,last,false);
}

extern "C" int astr_device_upwind_high_order_axis(void *handle,int axis,Real gamma,Real mach,Real weight,
    const int low[3],const int high[3],const int first[3],const int last[3])
{
    return apply_axis(handle,axis,5,1,gamma,mach,weight,0,low,high,first,last,true);
}

extern "C" int astr_device_upwind_positivity_faces(void *handle,int axis,int high_order,Real gamma,Real mach,Real weight,
    const int low[3],const int high[3])
{
    if(!handle || (high_order!=0 && high_order!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    const int first[3]={0,0,0};
    return apply_axis(handle,axis,5,1,gamma,mach,weight,0,low,high,first,s.upper,high_order!=0,true);
}

extern "C" int astr_device_upwind_high_order_boundary_rhs(void *handle,int axis,const int low[3],const int high[3])
{
    if(!handle || axis<0 || axis>2 || !low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);Box box{};
    for(int a=0;a<3;++a) {
        box.upper[a]=s.upper[a];box.low[a]=low[a];box.high[a]=high[a];
        box.first[a]=low[a];box.last[a]=s.upper[a];
    }
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    difference_kernel<true,true><<<(count+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_METRIC].view,
        box,axis,s.upper[0]+1,s.upper[1]+1,count);
    return astr::finish_launch(s);
}
