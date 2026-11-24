#pragma once
#include <cmath>
#include "astr_precision.h"
#include "keep_w1_tensor_math.h"

namespace astr {
namespace w1 {
struct SolverState {
    TensorGrid grid;
    const numerical_real *velocity,*temperature,*metric,*jac;
    double re,pr,c5,tc,tc1;

    ASTR_W1_FUNCTION int halo_size() const {
        return normal_stride(grid)*(grid.n[1]+2*grid.halo)*(grid.n[2]+2*grid.halo);
    }
    ASTR_W1_FUNCTION double value(int node,int component) const {
        return component==3 ? temperature[node] : velocity[node+component*halo_size()];
    }
    ASTR_W1_FUNCTION double viscosity(int node) const {
        const double t=temperature[node];
        return (t*sqrt(t)*tc1/(t+tc))/re;
    }
    ASTR_W1_FUNCTION double conductivity(int,double mu) const { return (mu/pr)/c5; }
    ASTR_W1_FUNCTION double jacobian(int node) const { return 1./metric[node+4*halo_size()]; }
    ASTR_W1_FUNCTION double volume(int node) const { return jac[node]; }
};

// Component-major storage is compatible with the existing CPU and GPU halo exchanges.
struct SolverFlux {
    numerical_real *data;
    int volume;
    ASTR_W1_FUNCTION double value(int node,int axis,int m) const { return data[node+(4*axis+m)*volume]; }
    ASTR_W1_FUNCTION double viscosity(int node) const { return data[node+12*volume]; }
    ASTR_W1_FUNCTION double conductivity(int node) const { return data[node+13*volume]; }
    ASTR_W1_FUNCTION void store(int node,const TensorFlux &flux) const {
        for(int a=0;a<3;++a) for(int m=0;m<4;++m) data[node+(4*a+m)*volume]=flux.value[a][m];
        data[node+12*volume]=flux.mu;
        data[node+13*volume]=flux.kappa;
    }
};

// Each packet contains wall-inward x/z planes of (u,v,w,T,mu,kappa,J_y).
struct EndSamples {
    const numerical_real *lower,*upper;
    int low_width,high_width,last,nx,nz;
    ASTR_W1_FUNCTION const numerical_real *sample(int i,int j,int k) const {
        if(j<low_width) return lower+7*(i+nx*(k+nz*j));
        if(last-j<high_width) return upper+7*(i+nx*(k+nz*(last-j)));
        return nullptr;
    }
};

template<typename Samples=NoEndSamples>
ASTR_W1_FUNCTION void accumulate_w1(const TensorTable &table,SolverState state,SolverFlux flux,
    numerical_real *rhs,int node,int count,Samples samples={})
{
    const int nx=state.grid.n[0],ny=state.grid.n[1];
    const int i=node%nx,j=(node/nx)%ny,k=node/(nx*ny);
    const auto result=tensor_rhs(table,state.grid,state,flux,i,j,k,samples);
    const double volume=state.volume(tensor_offset(state.grid,i,j,k));
    for(int m=0;m<4;++m) rhs[node+(m+1)*count]+=volume*result.rhs[m];
}
} // namespace w1
} // namespace astr
