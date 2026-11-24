#pragma once

#ifndef ASTR_W1_FUNCTION
#define ASTR_W1_FUNCTION inline
#endif

namespace astr {
namespace w1 {

// Nine physical-end rows and one centered row, in computational coordinates.
constexpr int normal_rows=10;
constexpr int normal_terms=485;
struct NormalTable {
    int begin[normal_rows+1], value_index[normal_terms], coefficient_index[normal_terms];
    double term[normal_terms], h0[6], borrowing;
};

struct NormalResult {
    double rhs, stationary_wall_work, wall_flux;
};

struct NormalStencil { int row,origin,direction; };

ASTR_W1_FUNCTION NormalStencil normal_stencil(int last,int i)
{
    const int distance=i<last-i ? i : last-i;
    const int row=distance<9 ? distance : 9;
    const int origin=row==9 ? i : (i<last-i ? 0 : last);
    return {row,origin,row==9 || origin==0 ? 1 : -1};
}

ASTR_W1_FUNCTION double trace_weight(int j)
{
    const double s[5]={-25.0/12.0,4.0,-3.0,4.0/3.0,-.25};
    return s[j];
}

// One field on a tensor-grid line: b=c/J_y. Values at the wall are not overwritten.
template<typename Line>
ASTR_W1_FUNCTION NormalResult normal_line_point(const NormalTable &table,Line line,int last,int i,double inv_h,
    double wall_low,double wall_high,double raw_bound_scale)
{
    const int distance=i<last-i ? i : last-i;
    const auto stencil=normal_stencil(last,i);
    const int row=stencil.row,origin=stencil.origin,direction=stencil.direction;
    double strong=0,work=0;
    const double fi=line.value(i);
    for(int r=table.begin[row];r<table.begin[row+1];++r) {
        const double value=line.value(origin+direction*table.value_index[r]);
        const double scale=table.term[r]*line.coefficient(origin+direction*table.coefficient_index[r]);
        const double delta=scale*(value-fi);
        strong+=delta;
        work+=delta*(.5*(value+fi));
    }
    const double physical_scale=inv_h*inv_h/line.jacobian(i);
    const double norm=distance<6 ? table.h0[distance] : 1.0;
    NormalResult result{strong*physical_scale,work*physical_scale,0};
    if(distance<5) {
        const int wall=origin;
        const double fw=line.value(wall),bw=line.coefficient(wall);
        const double jump=fw-(wall==0 ? wall_low : wall_high);
        result.rhs-=physical_scale*bw*trace_weight(distance)*jump/norm;
        if(distance==0) {
            double bmin=bw,flux=0,work_flux=0;
            for(int j=0;j<6;++j) {
                const int p=wall+direction*j;
                const double bp=line.coefficient(p);
                if(bp<bmin) bmin=bp;
                if(j<5) {
                    const double f=line.value(p),df=f-fw;
                    flux-=bw*trace_weight(j)*df;
                    work_flux-=bw*trace_weight(j)*df*(.5*(f+fw));
                }
            }
            const double penalty=2*(raw_bound_scale*bw/norm+table.borrowing*bw*bw/bmin);
            result.rhs-=physical_scale*penalty*jump/norm;
            result.wall_flux=inv_h*(flux-penalty*jump);
            result.stationary_wall_work-=physical_scale*work_flux/norm;
        }
    }
    return result;
}

struct StridedNormalLine {
    const double *f,*b,*jac;
    int stride;
    ASTR_W1_FUNCTION double value(int j) const { return f[j*stride]; }
    ASTR_W1_FUNCTION double coefficient(int j) const { return b[j*stride]; }
    ASTR_W1_FUNCTION double jacobian(int j) const { return jac[j*stride]; }
};

ASTR_W1_FUNCTION NormalResult normal_point(const NormalTable &table,const double *f,const double *b,
    const double *jac,int stride,int last,int i,double inv_h,double wall_low,double wall_high)
{
    return normal_line_point(table,StridedNormalLine{f,b,jac,stride},last,i,inv_h,wall_low,wall_high,1.);
}

} // namespace w1
} // namespace astr
