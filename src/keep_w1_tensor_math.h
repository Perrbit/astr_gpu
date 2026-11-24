#pragma once
#include "keep_w1_normal_math.h"

namespace astr {
namespace w1 {

struct TensorTable { NormalTable normal; double dleft[6][9]; };
struct TensorGrid {
    int n[3]; double inv_h[3],tw[2]; bool duplicate;
    int halo=0,first_y=0,total_y=0;
    bool split[3]{};
};
struct TensorState {
    const double *u[4], *mu, *kappa, *jac_y;
    ASTR_W1_FUNCTION double value(int node,int component) const { return u[component][node]; }
    ASTR_W1_FUNCTION double viscosity(int node) const { return mu[node]; }
    ASTR_W1_FUNCTION double conductivity(int node,double) const { return kappa[node]; }
    ASTR_W1_FUNCTION double jacobian(int node) const { return jac_y[node]; }
};
struct TensorFlux { double value[3][4],mu,kappa; };
struct TensorResult { double rhs[4], wall_flux[4]; };

struct PackedTensorFlux {
    const TensorFlux *data;
    ASTR_W1_FUNCTION double value(int node,int axis,int m) const { return data[node].value[axis][m]; }
    ASTR_W1_FUNCTION double viscosity(int node) const { return data[node].mu; }
    ASTR_W1_FUNCTION double conductivity(int node) const { return data[node].kappa; }
};

ASTR_W1_FUNCTION int normal_last(TensorGrid g) { return g.total_y ? g.total_y : g.n[1]-1; }
ASTR_W1_FUNCTION int normal_stride(TensorGrid g) { return g.n[0]+2*g.halo; }

ASTR_W1_FUNCTION int tensor_offset(TensorGrid g,int i,int j,int k)
{
    return i+g.halo+normal_stride(g)*(j+g.halo+(g.n[1]+2*g.halo)*(k+g.halo));
}

ASTR_W1_FUNCTION int periodic_index(int i,int n)
{
    int wrapped=i%n;
    return wrapped<0 ? wrapped+n : wrapped;
}

ASTR_W1_FUNCTION int neighbor(TensorGrid g,int i,int j,int k,int axis,int offset)
{
    if(axis==0) i=g.split[0] ? i+offset : periodic_index(i+offset,g.n[0]-int(g.duplicate));
    else if(axis==2) k=g.split[2] ? k+offset : periodic_index(k+offset,g.n[2]-int(g.duplicate));
    else j+=offset;
    return tensor_offset(g,i,j,k);
}

ASTR_W1_FUNCTION double normal_weight(const TensorTable &table,int last,int j,int p)
{
    if(j<6) return p<9 ? table.dleft[j][p] : 0;
    if(last-j<6) return last-p<9 ? -table.dleft[last-j][last-p] : 0;
    const double centred[7]={-1./60,3./20,-3./4,0,3./4,-3./20,1./60};
    return centred[p-j+3];
}

ASTR_W1_FUNCTION void normal_range(int last,int j,int &first,int &end)
{
    first=j<6 ? 0 : (last-j<6 ? last-8 : j-3);
    end=j<6 ? 8 : (last-j<6 ? last : j+3);
}

template<typename State>
ASTR_W1_FUNCTION double gradient(const TensorTable &table,TensorGrid g,State s,double jac_y,
    int i,int j,int k,int axis,int component)
{
    if(axis==1) {
        int first,end;
        const int global=g.first_y+j,last=normal_last(g);
        normal_range(last,global,first,end);
        double sum=0,local=s.value(tensor_offset(g,i,j,k),component);
        for(int p=first;p<=end;++p)
            sum+=normal_weight(table,last,global,p)*(s.value(tensor_offset(g,i,p-g.first_y,k),component)-local);
        return sum*g.inv_h[1]/jac_y;
    }
    const double centred[3]={.75,-.15,1./60};
    double sum=0;
    for(int q=1;q<=3;++q)
        sum+=centred[q-1]*(s.value(neighbor(g,i,j,k,axis,q),component)-s.value(neighbor(g,i,j,k,axis,-q),component));
    return sum*g.inv_h[axis];
}

template<typename State>
ASTR_W1_FUNCTION TensorFlux tensor_flux(const TensorTable &table,TensorGrid grid,State s,int i,int j,int k)
{
    const int n=tensor_offset(grid,i,j,k);
    double grad[4][3];
    for(int m=0;m<4;++m) for(int a=0;a<3;++a)
        grad[m][a]=gradient(table,grid,s,s.jacobian(n),i,j,k,a,m);
    const double div=grad[0][0]+grad[1][1]+grad[2][2];
    TensorFlux flux{};
    flux.mu=s.viscosity(n);
    flux.kappa=s.conductivity(n,flux.mu);
    for(int a=0;a<3;++a) {
        for(int m=0;m<3;++m) {
            double value=grad[m][a]+grad[a][m]-(m==a ? (2./3)*div : 0);
            // The paired SBP D2 below owns all purely normal second derivatives.
            if(a==1) value-=(m==1 ? 4./3 : 1.)*grad[m][1];
            flux.value[a][m]=flux.mu*value;
            flux.value[a][3]+=s.value(n,m)*flux.value[a][m];
        }
        if(a!=1) flux.value[a][3]+=flux.kappa*grad[3][a];
    }
    return flux;
}

template<typename State,typename Flux>
ASTR_W1_FUNCTION double tangential_jump(TensorGrid g,State s,Flux flux,int i,int j,int k,int axis,int component)
{
    const double centred[3]={.75,-.15,1./60};
    double sum=0;
    for(int q=1;q<=3;++q) {
        const int p=neighbor(g,i,j,k,axis,q),m=neighbor(g,i,j,k,axis,-q);
        sum+=centred[q-1]*(flux.viscosity(p)*s.value(p,component)-flux.viscosity(m)*s.value(m,component));
    }
    return sum*g.inv_h[axis];
}

struct NoEndSamples {
    ASTR_W1_FUNCTION const double *sample(int,int,int) const { return nullptr; }
};

template<typename State,typename Flux,typename EndSamples> struct TensorLine {
    State s;
    Flux flux;
    EndSamples samples;
    int base,stride,component,first_y,i,k;
    ASTR_W1_FUNCTION int offset(int j) const { return base+(j-first_y)*stride; }
    ASTR_W1_FUNCTION double value(int j) const {
        const auto *p=samples.sample(i,j,k);
        return p ? p[component] : s.value(offset(j),component);
    }
    ASTR_W1_FUNCTION double jacobian(int j) const {
        const auto *p=samples.sample(i,j,k);
        return p ? p[6] : s.jacobian(offset(j));
    }
    ASTR_W1_FUNCTION double coefficient(int j) const {
        const auto *p=samples.sample(i,j,k);
        if(p) return p[component==3 ? 5 : 4]/p[6];
        const int n=offset(j);
        return (component==3 ? flux.conductivity(n) : flux.viscosity(n))/s.jacobian(n);
    }
};

template<typename State,typename Flux,typename EndSamples=NoEndSamples>
ASTR_W1_FUNCTION TensorResult tensor_rhs(const TensorTable &table,TensorGrid grid,State s,
    Flux flux,int i,int j,int k,EndSamples samples={})
{
    TensorResult result{};
    const int n=tensor_offset(grid,i,j,k),last=normal_last(grid),base=tensor_offset(grid,i,0,k);
    const int stride=normal_stride(grid),global=grid.first_y+j;
    const double centred[3]={.75,-.15,1./60};
    const int distance=global<last-global ? global : last-global;
    const double inv_y=grid.inv_h[1]/s.jacobian(n);
    int first,end;
    normal_range(last,global,first,end);
    for(int m=0;m<4;++m) {
        for(int axis=0;axis<3;axis+=2)
            for(int q=1;q<=3;++q)
                result.rhs[m]+=grid.inv_h[axis]*centred[q-1]*(flux.value(neighbor(grid,i,j,k,axis,q),axis,m)-
                                                            flux.value(neighbor(grid,i,j,k,axis,-q),axis,m));
        for(int p=first;p<=end;++p)
            result.rhs[m]+=inv_y*normal_weight(table,last,global,p)*
                           (flux.value(tensor_offset(grid,i,p-grid.first_y,k),1,m)-flux.value(n,1,m));
        const double factor=m==1 ? 4./3 : 1.;
        const TensorLine<State,Flux,EndSamples> line{s,flux,samples,base,stride,m,grid.first_y,i,k};
        const auto pure=normal_line_point(table.normal,line,last,global,grid.inv_h[1],m==3 ? grid.tw[0] : 0.,
                                          m==3 ? grid.tw[1] : 0.,m==3 ? 1. : 2./factor);
        result.rhs[m]+=factor*pure.rhs;
        result.wall_flux[m]=factor*pure.wall_flux;
        if(m!=3) result.rhs[3]+=factor*pure.stationary_wall_work;
    }
    if(distance==0) {
        const double sign=global==0 ? -1. : 1.;
        const double wall_scale=sign*inv_y/table.normal.h0[0];
        for(int m=0;m<3;++m) result.wall_flux[m]+=sign*flux.value(n,1,m);
        // Transpose of the omitted tangential part of the normal traction.
        result.rhs[0]+=(2./3)*wall_scale*tangential_jump(grid,s,flux,i,j,k,0,1);
        result.rhs[1]-=wall_scale*(tangential_jump(grid,s,flux,i,j,k,0,0)+tangential_jump(grid,s,flux,i,j,k,2,2));
        result.rhs[2]+=(2./3)*wall_scale*tangential_jump(grid,s,flux,i,j,k,2,1);
        result.rhs[3]-=wall_scale*flux.value(n,1,3);
    }
    return result;
}

} // namespace w1
} // namespace astr
