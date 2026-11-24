#include "keep_w1_solver_math.h"
#include "keep_w1_tables.h"

extern "C" int astr_keep_w1_missing_y(int first,int span,int last,int halo,const int *received,int *required)
{
    using namespace astr::w1;
    const auto &table=solver_table.normal;
    for(int j=first;j<=first+span;++j) {
        int lo,hi;
        normal_range(last,j,lo,hi);
        const auto s=normal_stencil(last,j);
        const int halo_lo=first-halo,halo_hi=first+span+halo;
        // The SAT borrowing bound accesses wall and first five interior coefficients.
        if(j==0 && hi<5) hi=5;
        if(j==last && lo>last-5) lo=last-5;
        bool missing=lo<halo_lo || hi>halo_hi;
        for(int t=table.begin[s.row];t<table.begin[s.row+1];++t) {
            const int a=s.origin+s.direction*table.value_index[t],b=s.origin+s.direction*table.coefficient_index[t];
            const int points[2]={a,b};
            for(int p : points) {
                if(p<lo) lo=p;
                if(p>hi) hi=p;
                if((p<halo_lo || p>halo_hi) && p>=received[0] && last-p>=received[1]) missing=true;
            }
        }
        if(missing) {
            required[0]=lo;required[1]=hi;
            return j;
        }
    }
    return -1;
}

extern "C" void astr_keep_w1_cpu(int stage,int im,int jm,int km,int halo,int first_y,int total_y,const int *split,
    const astr_numerical_real *velocity,
    const astr_numerical_real *temperature,const astr_numerical_real *metric,const astr_numerical_real *jac,
    const double *inv,const double *wall,double re,double pr,double c5,double tc,double tc1,
    astr_numerical_real *scratch,astr_numerical_real *rhs,const int *width,
    const astr_numerical_real *lower,const astr_numerical_real *upper)
{
    using namespace astr::w1;
    const TensorGrid grid{{im+1,jm+1,km+1},{inv[0],inv[1],inv[2]},{wall[0],wall[1]},true,
                          halo,first_y,total_y,{split[0]!=0,split[1]!=0,split[2]!=0}};
    const SolverState state{grid,velocity,temperature,metric,jac,re,pr,c5,tc,tc1};
    const SolverFlux flux{scratch,state.halo_size()};
    const int count=(im+1)*(jm+1)*(km+1);
    if(stage==0) {
        for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i)
            flux.store(tensor_offset(grid,i,j,k),tensor_flux(solver_table,grid,state,i,j,k));
    } else if(width[0] || width[1]) {
        const EndSamples samples{lower,upper,width[0],width[1],total_y,im+1,km+1};
        for(int n=0;n<count;++n) accumulate_w1(solver_table,state,flux,rhs,n,count,samples);
    } else {
        for(int n=0;n<count;++n) accumulate_w1(solver_table,state,flux,rhs,n,count);
    }
}
