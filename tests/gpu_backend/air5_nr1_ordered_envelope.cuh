#pragma once

namespace astr { namespace air5 { namespace nr1_budget {
struct Enclosure {
    double lo,hi;
    __device__ Enclosure(double a=0.):lo(a),hi(a) {}
    __device__ Enclosure(double a,double b):lo(a),hi(b) {}
};

__device__ inline Enclosure add(Enclosure a,Enclosure b)
{return {__dadd_rd(a.lo,b.lo),__dadd_ru(a.hi,b.hi)};}
__device__ inline Enclosure neg(Enclosure a) {return {-a.hi,-a.lo};}
__device__ inline Enclosure absolute(Enclosure a)
{return {fmax(0.,fmax(a.lo,-a.hi)),fmax(fabs(a.lo),fabs(a.hi))};}
__device__ inline Enclosure mul(Enclosure a,Enclosure b)
{
    return {fmin(fmin(__dmul_rd(a.lo,b.lo),__dmul_rd(a.lo,b.hi)),fmin(__dmul_rd(a.hi,b.lo),__dmul_rd(a.hi,b.hi))),
            fmax(fmax(__dmul_ru(a.lo,b.lo),__dmul_ru(a.lo,b.hi)),fmax(__dmul_ru(a.hi,b.lo),__dmul_ru(a.hi,b.hi)))};
}
__device__ inline Enclosure divide_positive(Enclosure a,double b)
{return {__ddiv_rd(a.lo,b),__ddiv_ru(a.hi,b)};}

struct BoundedRow {Enclosure a[5],b;};

__device__ inline bool contract_row(const BoundedRow &row,const Enclosure box[5],Enclosure result[5])
{
    double minimum[5];
    for(int s=0;s<5;++s)minimum[s]=mul(row.a[s],box[s]).lo;
    // Do not subtract one term from a previously rounded sum of minima.
    for(int s=0;s<5;++s) {
        double tail=0.;
        for(int i=0;i<5;++i)if(i!=s)tail=__dadd_rd(tail,minimum[i]);
        const double rhs=__dsub_ru(row.b.hi,tail);
        if(rhs<minimum[s])return false;
        const Enclosure a=row.a[s];
        if(a.lo>0.)result[s].hi=fmin(result[s].hi,__ddiv_ru(rhs,rhs>=0.?a.lo:a.hi));
        if(a.hi<0.)result[s].lo=fmax(result[s].lo,__ddiv_rd(rhs,rhs>=0.?a.hi:a.lo));
        if(result[s].lo>result[s].hi)return false;
    }
    return true;
}

__device__ inline double round_error(double magnitude)
{
    if(magnitude==0. || !finite(magnitude))return magnitude;
    const int exponent=ilogb(magnitude)-53;
    return scalbn(1.,exponent<-1074?-1074:exponent);
}

__device__ inline Enclosure division_error(double a,double b,double rounded)
{
    // The directed FMA encloses a - rounded*b without first rounding the product.
    return divide_positive({__fma_rd(-rounded,b,a),__fma_ru(-rounded,b,a)},b);
}

struct BoundedSum {Enclosure value;double error;};
__device__ inline void accumulate(BoundedSum &sum,Enclosure term,double error)
{
    const Enclosure a=add(sum.value,{-sum.error,sum.error}),b=add(term,{-error,error});
    const Enclosure positive=a.lo>0.?a:b,negative=a.lo>0.?b:a;
    const bool exact_add=(a.lo==0. && a.hi==0.) || (b.lo==0. && b.hi==0.) ||
        (positive.lo>0. && negative.hi<0. && __dmul_rd(2.,positive.lo)>=-negative.lo &&
         __dmul_rd(-2.,negative.hi)>=positive.hi);
    sum.value=add(sum.value,term);
    sum.error=__dadd_ru(sum.error,error);
    if(!exact_add)sum.error=__dadd_ru(sum.error,round_error(__dadd_ru(absolute(sum.value).hi,sum.error)));
}

__device__ inline double ordered_budget_guard(const Enclosure interval[5],const double error[5],
    Enclosure target,double target_error,double tolerance)
{
    constexpr double eps=64.*DBL_EPSILON;
    BoundedSum sum{{0.},0.},magnitude{{0.},0.};
    for(int s=0;s<5;++s) {
        accumulate(sum,interval[s],error[s]);accumulate(magnitude,absolute(interval[s]),error[s]);
    }
    accumulate(sum,neg(target),target_error);
    const double budget_max=__dadd_ru(tolerance,__dmul_ru(eps,__dadd_ru(magnitude.value.hi,magnitude.error)));
    const double budget_error=__dadd_ru(__dmul_ru(eps,magnitude.error),round_error(budget_max));
    // EPS is a power of two; account separately for underflow in the product.
    const double underflow=__dmul_rd(eps,__dsub_rd(magnitude.value.lo,magnitude.error))>=DBL_MIN?0.:scalbn(1.,-1074);
    return __dadd_ru(sum.error,__dadd_ru(budget_error,underflow));
}

__device__ __noinline__ bool ordered_envelope(const double l[5],const double h[5],double sf,const double gas[5],
    double sw,double mass,double moment,const double g[5],const double targets[2],double tol,
    int orthant,Enclosure region[5],double guard[2])
{
    constexpr double eps=64.*DBL_EPSILON;
    const Enclosure t[2]={divide_positive(mass,sf),divide_positive(divide_positive(moment,sw),sf)};
    Enclosure w[5],box[5];BoundedRow rows[4];
    for(int s=0;s<5;++s) {
        w[s]=divide_positive(gas[s],sw);
        box[s]=region[s]=divide_positive({l[s],h[s]},sf);
        const double sign=(orthant&(1<<s))?-1.:1.;
        for(int r=0;r<4;++r) {
            const double side=(r&1)?-1.:1.;
            const Enclosure coeff=r<2?Enclosure(1.):w[s];
            rows[r].a[s]=add(mul(side,coeff),mul(-sign*eps,absolute(coeff)));
        }
    }
    for(int r=0;r<4;++r) {
        rows[r].b=add((r&1)?neg(t[r/2]):t[r/2],tol);
        if(!contract_row(rows[r],box,region))return false;
    }
    // Positive row combinations preserve the inequalities and eliminate one
    // component exactly; contract all rows from the same original box.
    for(int i=0;i<2;++i)for(int j=2;j<4;++j)for(int p=0;p<5;++p) {
        const Enclosure a=rows[i].a[p],b=rows[j].a[p];
        if(!((a.lo>0. && b.hi<0.) || (a.hi<0. && b.lo>0.)))continue;
        const Enclosure u=absolute(b),v=absolute(a);BoundedRow row;
        row.b=add(mul(u,rows[i].b),mul(v,rows[j].b));
        for(int s=0;s<5;++s)if(s!=p)row.a[s]=add(mul(u,rows[i].a[s]),mul(v,rows[j].a[s]));
        if(!contract_row(row,box,region))return false;
    }
    double e_mass[5],e_moment[5];Enclosure weighted[5];
    for(int s=0;s<5;++s) {
        const double value=absolute(region[s]).hi;
        if(l[s]==h[s])e_mass[s]=absolute(division_error(l[s],sf,l[s]/sf)).hi;
        else {
            const double physical=fmin(__dmul_ru(value,sf),fmax(fabs(l[s]),fabs(h[s])));
            const double e=__ddiv_ru(round_error(physical),sf);
            e_mass[s]=__dadd_ru(e,round_error(__dadd_ru(value,e)));
        }
        weighted[s]=mul(w[s],region[s]);
        const double abs_g=fabs(g[s]);
        const double prod=__dmul_ru(abs_g,__dadd_ru(value,e_mass[s]));
        const double round=abs_g==1.?0.:round_error(prod);
        const double half_err=abs_g==.5?scalbn(1.,-1074):round;
        const double weight_error=absolute(division_error(gas[s],sw,g[s])).hi;
        e_moment[s]=__dadd_ru(__dadd_ru(__dmul_ru(abs_g,e_mass[s]),__dmul_ru(weight_error,value)),half_err);
    }
    const double mid=moment/sw;
    const Enclosure moment_delta=add(divide_positive(division_error(moment,sw,mid),sf),division_error(mid,sf,targets[1]));
    guard[0]=ordered_budget_guard(region,e_mass,t[0],absolute(division_error(mass,sf,targets[0])).hi,tol);
    guard[1]=ordered_budget_guard(weighted,e_moment,t[1],absolute(moment_delta).hi,tol);
    return true;
}
}}}
