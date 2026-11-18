#pragma once

// Algebra shared by high-wall state projection and same-stage wall RHS.
namespace astr { namespace high_wall {
template<class Real>
__device__ inline Real temperature(const Real q[5],Real c2,Real c6)
{
    const Real u=q[1]/q[0],v=q[2]/q[0],w=q[3]/q[0];
    return (q[4]/q[0]-Real(.5)*(u*u+v*v+w*w))*c2/c6;
}

template<class Real>
__device__ inline Real temperature_rate(const Real q[5],const Real rhs[5],Real c2,Real c6)
{
    const Real u=q[1]/q[0],v=q[2]/q[0],w=q[3]/q[0];
    return (rhs[4]-(u*rhs[1]+v*rhs[2]+w*rhs[3])+
        (u*u+v*v+w*w-q[4]/q[0])*rhs[0])*c2/(c6*q[0]);
}

template<class Real>
__device__ inline Real neumann7(const Real line[7])
{
    const Real weights[8]={-Real(1089)/420,Real(2940)/420,-Real(4410)/420,Real(4900)/420,
        -Real(3675)/420,Real(1764)/420,-Real(490)/420,Real(60)/420};
    Real residual=0;
    for(int j=1;j<=7;++j) residual-=weights[j]*(line[j-1]-line[0]);
    return line[0]+residual/weights[0];
}
}}
