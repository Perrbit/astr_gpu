#pragma once
#include "chemistry_properties_math.cuh"

namespace astr { namespace air5 { namespace nr4_fixed {
struct Wide {double high,low; __device__ Wide(double a=0.,double b=0.):high(a),low(b) {}};
__device__ inline Wide two_sum(double a,double b)
{
    const double hi=a+b,v=hi-a;
    return {hi,(a-(hi-v))+(b-v)};
}
__device__ inline Wide operator+(Wide a,Wide b)
{
    const Wide h=two_sum(a.high,b.high);
    return two_sum(h.high,h.low+(a.low+b.low));
}
__device__ inline Wide operator-(Wide a,Wide b) {return a+Wide(-b.high,-b.low);}
__device__ inline Wide operator*(Wide a,Wide b)
{
    const double hi=a.high*b.high;
    return two_sum(hi,(fma(a.high,b.high,-hi)+a.high*b.low)+a.low*b.high);
}
__device__ inline int sign(Wide a)
{
    const double nonzero=a.high!=0.?a.high:a.low;
    return (nonzero>0.)-(nonzero<0.);
}

// Test-only direction selector. EOS acceptance, pressure and acoustic
// amplitudes remain the responsibility of the unchanged characteristic code.
__device__ inline int classify(const double q[11],bool preserve,bool incoming[2],int &temperature_region)
{
    Wide cr,gr,cy,gy,formation,momentum_squared;
    double y[5];
    for(int s=0;s<5;++s)y[s]=q[5+s]/q[0];
    if(!preserve) {
        double supplied=0.;for(int s=1;s<5;++s)supplied+=y[s];
        y[0]=1.-supplied;
    }
    for(int s=0;s<5;++s) {
        const double g=gas_constant(s),c=cv_tr(s);
        cr=cr+Wide(q[5+s])*c;gr=gr+Wide(q[5+s])*g;
        cy=cy+Wide(y[s])*c;gy=gy+Wide(y[s])*g;
        formation=formation+Wide(q[5+s])*data::formation_energy[s];
    }
    for(int d=1;d<4;++d)momentum_squared=momentum_squared+Wide(q[d])*q[d];
    const Wide enthalpy=Wide(q[4])-Wide(q[10])-formation;
    const Wide thermal=Wide(2.)*q[0]*enthalpy-momentum_squared;
    const Wide denominator=Wide(2.)*q[0]*cr;
    temperature_region=0;
    if(sign(thermal-denominator*data::temperature_min)<0)temperature_region=-1;
    else if(sign(denominator*data::temperature_max-thermal)<0)temperature_region=1;
    Wide indicator;
    if(temperature_region==0)indicator=Wide(2.)*q[2]*q[2]*cr*cy-gr*(cy+gy)*thermal;
    else {
        const double wall=temperature_region<0?data::temperature_min:data::temperature_max;
        indicator=Wide(q[2])*q[2]*cy-Wide(q[0])*gr*(cy+gy)*wall;
    }
    const int side=sign(indicator);
    incoming[0]=q[2]<=0.||side<0;
    incoming[1]=q[2]<0.&&side>0;
    return side;
}
}}}
