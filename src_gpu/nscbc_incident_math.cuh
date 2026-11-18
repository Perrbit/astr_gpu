#pragma once

#include "astr_precision.h"
#include <cmath>
#include <limits>

namespace astr { namespace incident {
using Real = numerical_real;
constexpr Real epsilon=std::numeric_limits<Real>::epsilon();

// Matrices and normal-line columns retain Fortran column-major ordering.
__device__ inline Real row_dot(const Real matrix[25],int row,const Real vector[5])
{
    Real result=0;
    for(int n=0;n<5;++n) result+=matrix[row+5*n]*vector[n];
    return result;
}

__device__ inline bool admissible(const Real q[5])
{
    for(int n=0;n<5;++n) if(!isfinite(q[n])) return false;
    if(q[0]<=0) return false;
    return q[4]>Real(.5)*(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])/q[0];
}

__device__ inline int incoming_rate(const Real face[5],const Real external[5],Real gamma,
    const Real metric[3],Real hn,const Real left[25],const Real reference_rate[5],
    Real rate[5],Real lambda[5],bool incoming[5])
{
    for(int m=0;m<5;++m) { incoming[m]=false;lambda[m]=0;rate[m]=reference_rate[m]; }
    if(!isfinite(gamma)||!isfinite(hn)) return 1;
    for(int n=0;n<3;++n) if(!isfinite(metric[n])) return 1;
    if(gamma<=1||hn<=0) return 1;
    const Real norm=sqrt(metric[0]*metric[0]+metric[1]*metric[1]+metric[2]*metric[2]);
    if(norm<=0||!admissible(face)||!admissible(external)) return 1;
    Real u[3],ug[3],du[3],speed2=0,external_speed2=0,normal_speed=0,work=0;
    for(int n=0;n<3;++n) {
        u[n]=face[n+1]/face[0];ug[n]=external[n+1]/external[0];du[n]=u[n]-ug[n];
        speed2+=u[n]*u[n];external_speed2+=ug[n]*ug[n];
        normal_speed+=metric[n]*u[n];work+=u[n]*du[n];
    }
    const Real pressure=(gamma-1)*(face[4]-Real(.5)*face[0]*speed2);
    const Real external_pressure=(gamma-1)*(external[4]-Real(.5)*external[0]*external_speed2);
    const Real css=sqrt(gamma*pressure/face[0]),un=normal_speed/norm;
    for(int m=0;m<3;++m) lambda[m]=un;
    lambda[3]=un+css;lambda[4]=un-css;
    const Real tolerance=Real(64)*epsilon*fmax(css,fabs(un));
    const Real drho=face[0]-external[0];
    Real increment[5];increment[0]=drho;
    for(int n=0;n<3;++n) increment[n+1]=u[n]*drho+face[0]*du[n];
    increment[4]=(pressure-external_pressure)/(gamma-1)+Real(.5)*speed2*drho+face[0]*work;
    for(int m=0;m<5;++m) {
        incoming[m]=lambda[m]<-tolerance;
        if(incoming[m]) rate[m]=reference_rate[m]-fabs(lambda[m])/hn*row_dot(left,m,increment);
    }
    for(int m=0;m<5;++m) if(!isfinite(rate[m])) return 2;
    return 0;
}

__device__ inline int normal_characteristics(const Real states[15],const Real velocity[9],
    const Real pressure[3],const Real jacobian[3],const Real metric[9],const Real left[25],Real lodi[5])
{
    for(int m=0;m<5;++m) lodi[m]=0;
    for(int j=0;j<3;++j) if(jacobian[j]<=0) return 1;
    Real flux[15],cartesian[15],delta_metric[3],derivative[5];
    for(int j=0;j<3;++j) {
        Real un=0;
        for(int n=0;n<3;++n) un+=metric[n+3*j]*velocity[n+3*j];
        flux[5*j]=jacobian[j]*states[5*j]*un;
        for(int n=0;n<3;++n) {
            flux[n+1+5*j]=jacobian[j]*(states[n+1+5*j]*un+metric[n+3*j]*pressure[j]);
            cartesian[n+1+5*j]=states[n+1]*velocity[j];
        }
        flux[4+5*j]=jacobian[j]*(states[4+5*j]+pressure[j])*un;
        cartesian[5*j]=states[j+1];cartesian[j+1+5*j]+=pressure[0];
        cartesian[4+5*j]=(states[4]+pressure[0])*velocity[j];
        delta_metric[j]=Real(1.5)*metric[j]*jacobian[0]-Real(2)*metric[j+3]*jacobian[1]+
            Real(.5)*metric[j+6]*jacobian[2];
    }
    for(int m=0;m<5;++m) {
        Real geometric=0;
        for(int j=0;j<3;++j) geometric+=cartesian[m+5*j]*delta_metric[j];
        derivative[m]=Real(1.5)*flux[m]-Real(2)*flux[m+5]+Real(.5)*flux[m+10]-geometric;
    }
    for(int m=0;m<5;++m) lodi[m]=row_dot(left,m,derivative)/jacobian[0];
    for(int m=0;m<5;++m) if(!isfinite(lodi[m])) return 2;
    return 0;
}

__device__ inline void convective_compatibility(const Real rhs[5],Real jacobian,const Real left[25],
    const Real normal_lodi[5],const bool incoming[5],Real rate[5])
{
    // Only entropy/shear retain transverse transport through a zero normal speed.
    for(int m=0;m<3;++m)
        if(incoming[m]) rate[m]=rate[m]+row_dot(left,m,rhs)/jacobian+normal_lodi[m];
}

__device__ inline int apply_complete_rhs(Real rhs[5],Real jacobian,const Real right[25],
    const Real left[25],const bool incoming[5],const Real rate[5])
{
    if(!isfinite(jacobian)||jacobian<=0) return 1;
    Real correction[5]={},trial[5];
    for(int m=0;m<5;++m)
        if(incoming[m]) correction[m]=rate[m]-row_dot(left,m,rhs)/jacobian;
    for(int m=0;m<5;++m) trial[m]=rhs[m]+jacobian*row_dot(right,m,correction);
    for(int m=0;m<5;++m) if(!isfinite(trial[m])) return 2;
    for(int m=0;m<5;++m) rhs[m]=trial[m];
    return 0;
}
}}
