#pragma once
#include "chemistry_properties_math.cuh"

namespace astr { namespace air5 { namespace characteristic {
__device__ inline void y_flux_differential(const double q[11],double pressure,const double dpdq[11],
    const double dq[11],double df[11])
{
    const double v=q[2]/q[0],dv=(dq[2]-v*dq[0])/q[0];double dp=0.;
    for(int c=0;c<11;++c)dp+=dpdq[c]*dq[c];
    for(int c=0;c<11;++c)df[c]=v*dq[c]+dv*q[c];
    df[2]+=dp;df[4]+=v*dp+pressure*dv;
}
__device__ inline bool normal_convection(const double q0[11],const double q1[11],const double q2[11],
    double spacing,double pressure,const double dpdq[11],double normal[11],bool first_order)
{
    for(int c=0;c<11;++c)normal[c]=0.;
    bool valid=spacing>0. && finite(spacing) && q0[0]>0.;
    for(int c=0;c<11;++c)valid=valid && finite(q0[c]) && finite(q1[c]) && finite(q2[c]);
    if(!valid)return false;
    double gradient[11];for(int c=0;c<11;++c)
        gradient[c]=first_order?(q0[c]-q1[c])/spacing:(1.5*(q0[c]-q1[c])-.5*(q1[c]-q2[c]))/spacing;
    y_flux_differential(q0,pressure,dpdq,gradient,normal);
    for(int c=0;c<11;++c)valid=valid && finite(normal[c]);return valid;
}
__device__ inline void acoustic_amplitudes(const double difference[11],const double dpdq[11],
    double v,double sound,double a2,double amplitude[2])
{
    double dp=0.;for(int c=0;c<11;++c)dp+=dpdq[c]*difference[c];
    const double dm=sound*(difference[2]-v*difference[0]);
    amplitude[0]=(dp-dm)/(2.*a2);amplitude[1]=(dp+dm)/(2.*a2);
}
__device__ inline bool transport_rhs(const double q[11],const double target[11],double pressure,double sound,
    const double dpdq[11],const double normal[11],const double remaining[11],double rate,double rhs[11],double reference_sound)
{
    for(int c=0;c<11;++c)rhs[c]=0.;
    bool valid=finite(pressure) && finite(sound) && finite(rate) && finite(reference_sound);
    for(int c=0;c<11;++c)valid=valid && finite(q[c]) && finite(target[c]) && finite(dpdq[c]) && finite(normal[c]) && finite(remaining[c]);
    if(!valid || !(q[0]>0. && target[0]>0. && pressure>0. && sound>0. && rate>=0. && reference_sound>0.))return false;
    const double u[3]={q[1]/q[0],q[2]/q[0],q[3]/q[0]};
    const double convective_rate=(fmax(-u[1],0.)/reference_sound)*rate;
    if(!finite(u[0]) || !finite(u[1]) || !finite(u[2]) || !finite(convective_rate))return false;
    const double a2=sound*sound;double waves[2][11],difference[11],amplitude[2];
    for(int mode=0;mode<2;++mode) {
        const int sign=2*mode-1;waves[mode][0]=1.;
        for(int d=0;d<3;++d)waves[mode][d+1]=u[d];waves[mode][2]=u[1]+sign*sound;
        waves[mode][4]=(q[4]+pressure)/q[0]+sign*sound*u[1];
        for(int c=5;c<11;++c)waves[mode][c]=q[c]/q[0];
    }
    // The outgoing normal part and the incoming target relaxation are independent.
    const bool convective_in=u[1]<0.,acoustic_in[2]={u[1]-sound<0.,u[1]+sound<0.};
    for(int c=0;c<11;++c)rhs[c]=convective_in?remaining[c]:remaining[c]-normal[c];
    if(acoustic_in[0]!=convective_in || acoustic_in[1]!=convective_in) {
        acoustic_amplitudes(normal,dpdq,u[1],sound,a2,amplitude);
        for(int mode=0;mode<2;++mode)if(acoustic_in[mode]!=convective_in) {
            const double scale=convective_in?-amplitude[mode]:amplitude[mode];
            for(int c=0;c<11;++c)rhs[c]+=waves[mode][c]*scale;
        }
    }
    if(convective_in && convective_rate!=0.) {
        for(int c=0;c<11;++c)difference[c]=-convective_rate*(q[c]-target[c]);
        acoustic_amplitudes(difference,dpdq,u[1],sound,a2,amplitude);
        for(int c=0;c<11;++c)rhs[c]+=difference[c]-waves[0][c]*amplitude[0]-waves[1][c]*amplitude[1];
    }
    if(rate!=0. && (acoustic_in[0] || acoustic_in[1])) {
        for(int c=0;c<11;++c)difference[c]=-rate*(q[c]-target[c]);
        acoustic_amplitudes(difference,dpdq,u[1],sound,a2,amplitude);
        for(int mode=0;mode<2;++mode)if(acoustic_in[mode])
            for(int c=0;c<11;++c)rhs[c]+=waves[mode][c]*amplitude[mode];
    }
    for(int c=0;c<11;++c)valid=valid && finite(rhs[c]);
    return valid;
}
// Same frozen thermodynamic differential assembled by the original top kernel.
__device__ inline int pressure_differential(const double q[11],bool preserve,double &pressure,double &sound,double gradient[11])
{
    double rho,u[3],t,y[5],tv;
    const int status=conservative_to_primitive(q,preserve,rho,u,t,y,tv,pressure);if(status)return status;
    double gas[5],cv[5],r=0.,cv_mix=0.;
    for(int s=0;s<5;++s) {gas[s]=gas_constant(s);cv[s]=cv_tr(s);r+=y[s]*gas[s];cv_mix+=y[s]*cv[s];}
    const double beta=r/cv_mix;sound=sqrt((1.+beta)*pressure/rho);
    gradient[0]=.5*beta*(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
    for(int d=0;d<3;++d)gradient[d+1]=-beta*u[d];gradient[4]=beta;
    for(int s=0;s<5;++s)gradient[s+5]=(gas[s]-beta*cv[s])*t-beta*data::formation_energy[s];
    gradient[10]=-beta;return ok;
}
}}}
