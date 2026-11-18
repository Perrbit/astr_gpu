#pragma once
#include "device_state_internal.cuh"
#include "chemistry_properties_math.cuh"
#include "shock_sensor_internal.cuh"

// AIR5 directional face candidates. No cell budget or limiter is applied here.
namespace astr { namespace air5 {
struct FaceFields {
    AstrDeviceView q,rho,velocity,pressure,species,metric,jacobian;
    AstrDeviceView stress,heat,species_flux,ev_flux,species_energy;
};
inline FaceFields face_fields(const State &s)
{
    return {s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_AIR5_SPECIES].view,s.fields[ASTR_DEVICE_METRIC].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_STRESS].view,s.fields[ASTR_DEVICE_HEAT_FLUX].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES_FLUX].view,s.fields[ASTR_DEVICE_AIR5_EV_FLUX].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX].view};
}
__device__ inline double face_value(AstrDeviceView field,const int p[3],int c=1,int d=1)
{return static_cast<const double *>(field.data)[astr::offset(field,p[0],p[1],p[2],c,d)];}
struct ConvectiveLine {
    FaceFields fields;int p[3],axis;
    bool fused=false;
    bool right=false;
    bool raw=false;
    bool mp_energy=false;
    __device__ void load(int index,double flux[11],double *normal_out=nullptr) const
    {
        int x[3]={p[0],p[1],p[2]};x[axis]=index;
        double normal=0;
        for(int i=1;i<=3;++i) {
            const int d=fused?4-i:i;
            const double m=face_value(fields.metric,x,axis+1,d),u=face_value(fields.velocity,x,d);
            normal=fused?fma(m,u,normal):normal+m*u;
        }
        if(normal_out)*normal_out=normal;
        const double jac=face_value(fields.jacobian,x),pressure=face_value(fields.pressure,x);
        flux[0]=jac*face_value(fields.q,x,1)*normal;
        for(int c=1;c<=3;++c) {
            const double q=face_value(fields.q,x,c+1),m=face_value(fields.metric,x,axis+1,c);
            flux[c]=jac*(fused?fma(m,pressure,q*normal):q*normal+m*pressure);
        }
        flux[4]=jac*(face_value(fields.q,x,5)+pressure)*normal;
        for(int c=5;c<11;++c)flux[c]=jac*face_value(fields.q,x,c+1)*normal;
    }
    __device__ double radius(int index,int &status) const
    {
        int x[3]={p[0],p[1],p[2]};x[axis]=index;
        const double rho=face_value(fields.rho,x),pressure=face_value(fields.pressure,x);
        if(rho<=0 || pressure<=0) {status=invalid_composition;return 0.;}
        double r=0,cv=0,normal=0,norm=0;
        for(int s=0;s<5;++s) {
            const double y=face_value(fields.species,x,s+1);
            r=fused?fma(y,gas_constant(s),r):r+y*gas_constant(s);
            cv=fused?fma(y,cv_tr(s),cv):cv+y*cv_tr(s);
        }
        if(r<=0 || cv<=0) {status=invalid_composition;return 0.;}
        for(int d=1;d<=3;++d) {
            const double metric=face_value(fields.metric,x,axis+1,d);
            const double u=face_value(fields.velocity,x,d);
            normal=fused?fma(metric,u,normal):normal+metric*u;
            norm=fused?fma(metric,metric,norm):norm+metric*metric;
        }
        return fabs(normal)+sqrt((1.+r/cv)*pressure/rho)*sqrt(norm);
    }
    __device__ void split(int index,double alpha,double plus[11],double minus[11],bool low_order=false) const
    {
        int x[3]={p[0],p[1],p[2]};x[axis]=index;
        double flux[11],normal;load(index,flux,&normal);
        const double jac=face_value(fields.jacobian,x);
        for(int c=0;c<11;++c) {
            const double state=face_value(fields.q,x,c+1);
            const double dissipative=alpha*jac*state;
            if(fused && mp_energy && c==4) {
                // Preserve the old contractions for each consumer of the split.
                const double enthalpy=jac*(state+face_value(fields.pressure,x));
                plus[c]=.5*(right?fma(normal,enthalpy,dissipative):fma(alpha*jac,state,flux[c]));
                if(raw || (right && !low_order))minus[c]=.5*fma(normal,enthalpy,-dissipative);
                else if(right || !low_order)minus[c]=.5*fma(-alpha*jac,state,flux[c]);
                else minus[c]=.5*(flux[c]-dissipative);
                continue;
            }
            // Match the old MP vibrational contractions in the raw and right-face paths.
            if(fused && mp_energy && right && c==10)plus[c]=.5*fma(normal,jac*state,dissipative);
            else plus[c]=.5*(fused?fma(alpha*jac,state,flux[c]):flux[c]+alpha*jac*state);
            if(fused && mp_energy && c==10 && (raw || (right && !low_order)))
                minus[c]=.5*fma(normal,jac*state,-dissipative);
            else {
                const bool fuse_penalty=fused && (!mp_energy || (!raw && c>=1 && c<=3 && (low_order || !right)));
                minus[c]=.5*(fuse_penalty?fma(-alpha*jac,state,flux[c]):flux[c]-dissipative);
            }
        }
    }
};
struct DiffusiveLine {
    FaceFields fields;int p[3],axis,channel;
    bool fused=false;
    __device__ void load(int index,double flux[11],bool scaled=true) const
    {
        int x[3]={p[0],p[1],p[2]};x[axis]=index;
        const double jac=scaled?face_value(fields.jacobian,x):1.;double metric[3];
        for(int d=0;d<3;++d)metric[d]=face_value(fields.metric,x,axis+1,d+1);
        for(int c=0;c<11;++c)flux[c]=0;
        const int tensor[3][3]={{1,2,3},{2,4,5},{3,5,6}};
        if(channel==0)for(int c=0;c<3;++c) {
            double value=0;for(int d=0;d<3;++d) {
                const double term=face_value(fields.stress,x,tensor[c][d]);
                value=fused ? fma(term,metric[d],value) : value+term*metric[d];
            }
            flux[c+1]=jac*value;
        }
        double heat=0,ev=0;
        for(int d=0;d<3;++d) {
            const double h=face_value(channel==0?fields.heat:fields.species_energy,x,d+1);
            const double v=face_value(channel==0?fields.ev_flux:fields.species_energy,x,channel==0?d+1:d+4);
            heat=fused ? fma(h,metric[d],heat) : heat+h*metric[d];
            ev=fused ? fma(v,metric[d],ev) : ev+v*metric[d];
        }
        flux[4]=jac*heat;flux[10]=jac*ev;
        for(int s=0;s<5;++s) {
            double value=0;for(int d=0;d<3;++d) {
                const double term=face_value(fields.species_flux,x,s+1,d+1);
                value=fused ? fma(term,metric[d],value) : value+term*metric[d];
            }
            flux[5+s]=jac*value;
        }
    }
};
template<bool fused=false,class Line> __device__ void centered_face(const Line &line,int face,int last,int ntype,double flux[11])
{
    double values[6][11];
    const bool lower=(ntype==1 || ntype==4) && face<=1;
    const bool upper=(ntype==2 || ntype==4) && face>=last-2;
    const int first=lower?0:(upper?last-5:face-2);
    for(int r=0;r<6;++r)line.load(first+r,values[r]);
    for(int c=0;c<11;++c) {
        const double middle=values[2][c]+values[3][c],near=values[1][c]+values[4][c],outer=values[0][c]+values[5][c];
        const double anchor=fused?fma(1./60.,outer,fma(37.*(1./60.),middle,(-2./15.)*near)):
            (37.*(1./60.))*middle-(2./15.)*near+(1./60.)*outer;
        if(lower) {
            const double d0=-.5*values[2][c]+2.*values[1][c]-1.5*values[0][c];
            const double d1=.5*(values[2][c]-values[0][c]);
            const double d2=(2./3.)*(values[3][c]-values[1][c])-(1./12.)*(values[4][c]-values[0][c]);
            flux[c]=face==1?anchor-d2:(face==0?anchor-d2-d1:anchor-d2-d1-d0);
        } else if(upper) {
            const double d0=(2./3.)*(values[4][c]-values[2][c])-(1./12.)*(values[5][c]-values[1][c]);
            const double d1=.5*(values[5][c]-values[3][c]);
            const double d2=.5*values[3][c]-2.*values[4][c]+1.5*values[5][c];
            const int depth=face-(last-3);
            flux[c]=depth==1?anchor+d0:(depth==2?anchor+d0+d1:anchor+d0+d1+d2);
        } else flux[c]=anchor;
    }
}
__device__ inline double face_minmod2(double a,double b)
{
    if(a>0 && b>0)return fmin(fabs(a),fabs(b));
    if(a<0 && b<0)return -fmin(fabs(a),fabs(b));
    return 0.;
}
__device__ inline double face_minmod4(double a,double b,double c,double d)
{
    const double m=fmin(fmin(fabs(a),fabs(b)),fmin(fabs(c),fabs(d)));
    if(a>0 && b>0 && c>0 && d>0)return m;
    if(a<0 && b<0 && c<0 && d<0)return -m;
    return 0.;
}
__device__ inline double face_polynomial(const double v[7],int order,bool fused=false)
{
    if(order==3)return (fused?fma(5.,v[1],-v[0])+2.*v[2]:-v[0]+5.*v[1]+2.*v[2])/6.;
    const int center=order==5?2:3;
    double linear;
    if(fused) {
        if(order==5)linear=fma(-3.,v[4],fma(27.,v[3],fma(47.,v[2],fma(-13.,v[1],2.*v[0]))))/60.;
        else {
            double f=fma(319.,v[3],fma(-101.,v[2],fma(25.,v[1],-3.*v[0])));
            linear=fma(4.,v[6],fma(-38.,v[5],fma(214.,v[4],f)))/420.;
        }
    } else linear=order==5?(2.*v[0]-13.*v[1]+47.*v[2]+27.*v[3]-3.*v[4])/60.:
        (-3.*v[0]+25.*v[1]-101.*v[2]+319.*v[3]+214.*v[4]-38.*v[5]+4.*v[6])/420.;
    const double mp=v[center]+face_minmod2(v[center+1]-v[center],4.*(v[center]-v[center-1]));
    if(linear>=fmin(v[center],mp) && linear<=fmax(v[center],mp))return linear;
    const double dm=v[center-2]-2.*v[center-1]+v[center],d0=v[center-1]-2.*v[center]+v[center+1];
    const double dp=v[center]-2.*v[center+1]+v[center+2];
    const double dhm=face_minmod4(4.*dm-d0,4.*d0-dm,dm,d0),dhp=face_minmod4(4.*d0-dp,4.*dp-d0,d0,dp);
    const double upper_limit=v[center]+4.*(v[center]-v[center-1]),average=.5*(v[center]+v[center+1]);
    const double median=average-.5*dhp,base=v[center]+.5*(v[center]-v[center-1]);
    const double curvature=fused?fma(4./3.,dhm,base):base+(4./3.)*dhm;
    const double lower=fmax(fmin(fmin(v[center],v[center+1]),median),fmin(fmin(v[center],upper_limit),curvature));
    const double upper=fmin(fmax(fmax(v[center],v[center+1]),median),fmax(fmax(v[center],upper_limit),curvature));
    return linear+face_minmod2(lower-linear,upper-linear);
}
__device__ inline void llf_face(const ConvectiveLine &line,int face,int order,double flux[11],int &status,bool low_order=false)
{
    double alpha=0;
    for(int r=face-(order-1)/2;r<=face+(order+1)/2;++r)alpha=fmax(alpha,line.radius(r,status));
    if(alpha<=0) {status=invalid_composition;return;}
    double plus[7][11],minus[7][11],unused[11];
    for(int r=0;r<order;++r) {
        line.split(face+r-(order-1)/2,alpha,plus[r],unused,low_order);
        line.split(face+(order+1)/2-r,alpha,unused,minus[r],low_order);
    }
    for(int c=0;c<11;++c) {
        if(order==1)flux[c]=plus[0][c]+minus[0][c];
        else {
            double p[7],m[7];for(int r=0;r<order;++r) {p[r]=plus[r][c];m[r]=minus[r][c];}
            flux[c]=face_polynomial(p,order,line.fused)+face_polynomial(m,order,line.fused);
        }
    }
}
__device__ inline bool air5_shock_face(const int8_t *mask,const int upper[3],const int p[3],int axis,int ntype)
{
    const int face=p[axis],last=upper[axis];int left[3]={p[0],p[1],p[2]},right[3]={p[0],p[1],p[2]};
    if(face<0 || (face==0 && (ntype==2 || ntype==3)))left[axis]=right[axis]=0;
    else if(face>=last || (face==last-1 && (ntype==1 || ntype==3)))left[axis]=right[axis]=last;
    else right[axis]=face+1;
    return mask[shock_node_offset(upper,left)]!=0 || mask[shock_node_offset(upper,right)]!=0;
}
template<bool fused=false>
__device__ inline void convective_face(const ConvectiveLine &line,int face,int last,int ntype,bool shock,
    bool low_order,bool symmetric,double flux[11],int &status)
{
    if(low_order)llf_face(line,face,1,flux,status,true);
    else if(shock) {
        int order=7;
        if(((ntype==1 || ntype==4) && face==0)||((ntype==2 || ntype==4) && face==last-1))order=1;
        else if(((ntype==1 || ntype==4) && face==1)||((ntype==2 || ntype==4) && face==last-2))order=3;
        else if(((ntype==1 || ntype==4) && face==2)||((ntype==2 || ntype==4) && face==last-3))order=5;
        llf_face(line,face,order,flux,status);
    } else centered_face<fused>(line,face,last,ntype,flux);
    if(status || symmetric)return;
    if(shock && !low_order) {
        double sum=0;for(int c=6;c<=9;++c)sum+=flux[c];flux[5]=flux[0]-sum;
    } else {
        flux[5]=flux[0];for(int c=6;c<=9;++c)flux[5]-=flux[c];
    }
}
}}
