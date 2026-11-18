#pragma once

#include "nscbc_incident_math.cuh"

namespace astr { namespace high_open {
using Real=numerical_real;

__device__ inline Real derivative6(const Real f[8],int row)
{
    // Match the interior D6 seventh moment, including the second derivative chain.
    const Real weights[3][8]={
        {Real(-156)/60,Real(423)/60,Real(-639)/60,Real(715)/60,Real(-540)/60,Real(261)/60,Real(-73)/60,Real(9)/60},
        {Real(-9)/60,Real(-84)/60,Real(171)/60,Real(-135)/60,Real(85)/60,Real(-36)/60,Real(9)/60,Real(-1)/60},
        {Real(1)/60,Real(-17)/60,Real(-56)/60,Real(115)/60,Real(-65)/60,Real(29)/60,Real(-8)/60,Real(1)/60}};
    Real derivative=0;
    for(int j=0;j<8;++j) derivative+=weights[row][j]*(f[j]-f[row]);
    return derivative;
}

__device__ inline void interface_position(int node,int dim,int ntype,int &row,int &side)
{
    row=-1;side=0;
    if((ntype==1||ntype==4)&&node>=0&&node<3) { row=node;side=1; }
    else if((ntype==2||ntype==4)&&node>=dim-3&&node<dim) { row=dim-1-node;side=-1; }
}

__device__ inline Real mp_flux6(const Real f[8],int row,bool incoming)
{
    // These are FD numerical-flux moments, not midpoint interpolation weights.
    const Real weights[3][8]={
        {Real(60)/420,Real(669)/420,Real(-591)/420,Real(459)/420,Real(-241)/420,Real(74)/420,Real(-10)/420,0},
        {Real(-10)/420,Real(130)/420,Real(459)/420,Real(-241)/420,Real(109)/420,Real(-31)/420,Real(4)/420,0},
        {Real(4)/420,Real(-38)/420,Real(214)/420,Real(319)/420,Real(-101)/420,Real(25)/420,Real(-3)/420,0}};
    const Real difference[8]={-1,7,-21,35,-35,21,-7,1};
    const Real strength[3]={Real(.125),Real(.25),Real(.125)};
    Real value=f[row],correction=0;
    for(int j=0;j<8;++j) {
        value+=weights[row][j]*(f[j]-f[row]);
        correction+=difference[j]*(f[j]-f[row]);
    }
    if(incoming) value-=strength[row]*correction;
    return value;
}

__device__ inline Real filter6(const Real f[7],int row)
{
    Real value=f[row];
    if(row==0) return value;
    const Real difference[7]={1,-6,15,-20,15,-6,1};
    Real residual=0;
    for(int j=0;j<7;++j) residual+=difference[j]*(f[j]-f[row]);
    if(row==2) residual=-residual;
    return value+residual/Real(64);
}

__device__ inline void normal_terms6(const Real state[40],const Real velocity[24],const Real pressure[8],
    const Real jacobian[8],const Real metric[24],Real derivative[5],Real rest[5])
{
    // The line runs from an upper physical face inward; preserve the outward sign.
    Real flux[40],cartesian[15],delta_metric[3],line[8];
    for(int s=0;s<8;++s) {
        Real un=0;for(int d=0;d<3;++d) un+=metric[d+3*s]*velocity[d+3*s];
        flux[5*s]=jacobian[s]*state[5*s]*un;
        for(int d=0;d<3;++d) flux[d+1+5*s]=jacobian[s]*(state[d+1+5*s]*un+metric[d+3*s]*pressure[s]);
        flux[4+5*s]=jacobian[s]*(state[4+5*s]+pressure[s])*un;
    }
    for(int d=0;d<3;++d) {
        cartesian[5*d]=state[d+1];
        for(int m=0;m<3;++m) cartesian[m+1+5*d]=state[m+1]*velocity[d];
        cartesian[d+1+5*d]+=pressure[0];
        cartesian[4+5*d]=(state[4]+pressure[0])*velocity[d];
        for(int s=0;s<8;++s) line[s]=metric[d+3*s]*jacobian[s];
        delta_metric[d]=-derivative6(line,0);
    }
    for(int m=0;m<5;++m) {
        for(int s=0;s<8;++s) line[s]=flux[m+5*s];
        derivative[m]=-derivative6(line,0);rest[m]=0;
        for(int d=0;d<3;++d) rest[m]+=cartesian[m+5*d]*delta_metric[d];
    }
}

__device__ inline void incoming_rate7(const Real primitive[40],Real gamma,Real hn,const Real left[25],
    const Real lambda[5],const bool incoming[5],const Real rhs[5],Real jacobian,
    const Real normal_lodi[5],bool crosses_front,Real rate[5])
{
    const Real weights[7]={Real(2940)/420,Real(-4410)/420,Real(4900)/420,
        Real(-3675)/420,Real(1764)/420,Real(-490)/420,Real(60)/420};
    Real derivative[5]={},dq[5],speed2=0,work=0;
    for(int m=0;m<5;++m) {
        if(crosses_front) derivative[m]=primitive[m+5]-primitive[m];
        else for(int j=1;j<8;++j) derivative[m]+=weights[j-1]*(primitive[m+5*j]-primitive[m]);
        derivative[m]/=hn;
    }
    dq[0]=derivative[0];
    for(int m=0;m<3;++m) {
        dq[m+1]=primitive[m+1]*derivative[0]+primitive[0]*derivative[m+1];
        speed2+=primitive[m+1]*primitive[m+1];work+=primitive[m+1]*derivative[m+1];
    }
    dq[4]=derivative[4]/(gamma-1)+Real(.5)*speed2*derivative[0]+primitive[0]*work;
    for(int m=0;m<5;++m) {
        const Real projected_rhs=incident::row_dot(left,m,rhs)/jacobian;
        rate[m]=projected_rhs;
        if(incoming[m]) rate[m]=projected_rhs+normal_lodi[m]-lambda[m]*incident::row_dot(left,m,dq);
    }
}
}}
