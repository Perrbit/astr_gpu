#pragma once
#include "chemistry_flux_math.cuh"
#include "air5_nr1_exact_box.cuh"
#include "air5_nr1_ordered_envelope.cuh"

// Experimental component. The solver and the C ABI do not use this projector.
namespace astr { namespace air5 { namespace nr1_budget {
struct Wide { double hi,lo; __device__ Wide(double h=0.,double l=0.):hi(h),lo(l) {} };

__device__ inline Wide sum_word(double a,double b)
{
    const double hi=a+b,v=hi-a;
    return {hi,(a-(hi-v))+(b-v)};
}
__device__ inline Wide operator+(Wide a,Wide b)
{
    const Wide s=sum_word(a.hi,b.hi),t=sum_word(a.lo,b.lo),m=sum_word(s.lo,t.hi);
    const Wide h=sum_word(s.hi,m.hi);
    return sum_word(h.hi,(h.lo+m.lo)+t.lo);
}
__device__ inline Wide operator-(Wide a,Wide b) {return a+Wide(-b.hi,-b.lo);}
__device__ inline Wide operator*(Wide a,Wide b)
{
    const double hi=a.hi*b.hi;
    return sum_word(hi,(fma(a.hi,b.hi,-hi)+a.hi*b.lo)+a.lo*b.hi);
}
__device__ inline Wide operator/(Wide a,Wide b)
{
    const double q=a.hi/b.hi;
    return sum_word(q,(a-b*q).hi/b.hi);
}
__device__ inline bool less(Wide a,Wide b) {return a.hi<b.hi || (a.hi==b.hi && a.lo<b.lo);}
__device__ inline Wide abs_word(Wide x) {return less(x,0.)?Wide()-x:x;}

struct Band { Wide a[5],bound; };

__device__ inline Wide measure(const Band &band,const Wide x[5])
{
    Wide sum;
    for(int s=0;s<5;++s)sum=sum+band.a[s]*x[s];
    return sum;
}

__device__ inline bool final_accepted(const double x[5],double scale,const double g[5],
    double mass,double moment,double tolerance)
{
    constexpr double eps=64.*DBL_EPSILON;
    double v[5],moment_abs=0.;
    for(int s=0;s<5;++s) {v[s]=x[s]/scale;moment_abs+=fabs(g[s]*v[s]);}
    return fabs(sum5(v)-mass)<=tolerance+eps*abs_sum5(v) &&
           fabs(dot5(g,v)-moment)<=tolerance+eps*moment_abs;
}

__device__ inline bool return_ranges_disjoint(const double lo[5],const double hi[5],double scale,
    const double g[5],const double targets[2],double tolerance)
{
    constexpr double eps=64.*DBL_EPSILON;
    double lower[2]={},upper[2]={},magnitude[2]={};
    // Monotonic FP64 division, products and ordered sums enclose every
    // rounded box return, including its largest permitted residual budget.
    for(int s=0;s<5;++s) {
        const double a=lo[s]/scale,b=hi[s]/scale,p=g[s]*a,q=g[s]*b;
        lower[0]+=a;upper[0]+=b;magnitude[0]+=fmax(fabs(a),fabs(b));
        lower[1]+=fmin(p,q);upper[1]+=fmax(p,q);magnitude[1]+=fmax(fabs(p),fabs(q));
    }
    for(int j=0;j<2;++j) {
        const double distance=fmax(lower[j]-targets[j],targets[j]-upper[j]);
        if(distance>tolerance+eps*magnitude[j])return true;
    }
    return false;
}

__device__ inline bool multipliers(const Band bands[4],int a,int b,int free,const Wide x[5],Wide &l1,Wide &l2)
{
    l1=l2=0.;
    if(a<0)return true;
    Wide aa,ab,bb;
    for(int s=0;s<5;++s)if(free&(1<<s)) {
        const Wide u=bands[a].a[s],v=b<0?Wide():bands[b].a[s];
        aa=aa+u*u;ab=ab+u*v;bb=bb+v*v;
    }
    if(!less(0.,aa))return false;
    const Wide r1=measure(bands[a],x)-bands[a].bound;
    if(b<0)l1=r1/aa;
    else {
        const Wide det=aa*bb-ab*ab;
        if(!less(0.,det))return false;
        const Wide r2=measure(bands[b],x)-bands[b].bound;
        l1=(bb*r1-ab*r2)/det;l2=(aa*r2-ab*r1)/det;
    }
    return !less(l1,0.) && !less(l2,0.);
}

__device__ inline int project(const double c[5],const double lo[5],const double hi[5],
    double mass,const double gas[5],double moment,double output[5])
{
    constexpr double eps=64.*DBL_EPSILON;
    for(int s=0;s<5;++s)output[s]=c[s];
    if(!finite5(c) || !finite5(lo) || !finite5(hi) || !finite5(gas) || !finite(mass) || !finite(moment))return nonfinite;
    double sf=fabs(mass),sw=0.;bool inside=true;
    for(int s=0;s<5;++s) {
        if(lo[s]>hi[s])return infeasible_flux;
        inside=inside && c[s]>=lo[s] && c[s]<=hi[s];
        sf=fmax(sf,fmax(fabs(c[s]),fmax(fabs(lo[s]),fabs(hi[s]))));sw=fmax(sw,fabs(gas[s]));
    }
    if(sw==0)return moment!=0?infeasible_flux:close_species_flux(c,lo,hi,mass,output);
    if(sf==0)return moment==0?ok:infeasible_flux;
    const double targets[2]={mass/sf,(moment/sw)/sf};
    if(!finite(targets[1]))return nonconverged;
    double g[5],cn[5];Wide coeff[2][5],origin[5];
    for(int s=0;s<5;++s) {
        cn[s]=c[s]/sf;g[s]=gas[s]/sw;origin[s]=Wide(c[s])/sf;
        coeff[0][s]=1.;coeff[1][s]=Wide(gas[s])/sw;
    }
    const double tolerance=eps*fmax(fmax(fabs(targets[0]),fabs(targets[1])),abs_sum5(cn));
    if(inside && fabs(sum5(cn)-targets[0])<=tolerance && fabs(dot5(g,cn)-targets[1])<=tolerance)return ok;
    if(strict_feasible(lo,hi,mass,gas,moment))
        return close_species_flux_moment(c,lo,hi,mass,gas,moment,output);
    if(return_ranges_disjoint(lo,hi,sf,g,targets,tolerance))return infeasible_flux;
    const Wide exact_target[2]={Wide(mass)/sf,(Wide(moment)/sw)/sf};
    bool found=false;Wide best_cost;
    for(int orthant=0;orthant<32;++orthant) {
        double l[5],h[5];bool intersects=true;Band bands[4];
        for(int s=0;s<5;++s) {
            const bool negative=(orthant&(1<<s))!=0;
            if((negative && lo[s]>=0.) || (!negative && hi[s]<0.)) {intersects=false;break;}
            l[s]=negative?lo[s]:fmax(lo[s],0.);h[s]=negative?fmin(hi[s],0.):hi[s];
            for(int row=0;row<4;++row) {
                const double direction=row%2?-1.:1.;
                const Wide w=coeff[row/2][s];
                bands[row].a[s]=w*direction-eps*abs_word(w)*(negative?-1.:1.);
            }
        }
        if(!intersects)continue;
        Enclosure region[5];double guard[2];
        if(!ordered_envelope(l,h,sf,gas,sw,mass,moment,g,targets,tolerance,orthant,region,guard))continue;
        for(int row=0;row<4;++row) {
            const double direction=row%2?-1.:1.;
            bands[row].bound=exact_target[row/2]*direction+(Wide(tolerance)-guard[row/2]);
        }
        Wide low[5],high[5];for(int s=0;s<5;++s) {low[s]=Wide(l[s])/sf;high[s]=Wide(h[s])/sf;}
        for(int box=0;box<243;++box) {
            int mask=box,free=0;bool valid=true;Wide base[5];double fixed[5];
            for(int s=0;s<5;++s) {
                const int kind=mask%3;mask/=3;
                if(l[s]==h[s] && kind!=1) {valid=false;break;}
                if(kind==0) {free|=1<<s;base[s]=origin[s];}
                else {base[s]=kind==1?low[s]:high[s];fixed[s]=kind==1?l[s]:h[s];}
            }
            if(!valid)continue;
            for(int mass_row=-1;mass_row<2;++mass_row)for(int moment_row=1;moment_row<4;++moment_row) {
                const int a=mass_row>=0?mass_row:(moment_row>=2?moment_row:-1);
                const int b=mass_row>=0 && moment_row>=2?moment_row:-1;
                Wide l1,l2;
                if(!multipliers(bands,a,b,free,base,l1,l2))continue;
                Wide x[5];bool feasible=true;
                for(int s=0;s<5;++s) {
                    x[s]=base[s];
                    if(free&(1<<s)) {
                        if(a>=0)x[s]=x[s]-l1*bands[a].a[s];
                        if(b>=0)x[s]=x[s]-l2*bands[b].a[s];
                        if(less(x[s],low[s]) || less(high[s],x[s]))feasible=false;
                    }
                }
                if(!feasible)continue;
                // Selected planes were solved algebraically. The rounded return is gated below.
                for(int row=0;row<4;++row)if(row!=a && row!=b && less(bands[row].bound,measure(bands[row],x)))feasible=false;
                if(!feasible)continue;
                double rounded[5];Wide cost;
                for(int s=0;s<5;++s) {
                    rounded[s]=(free&(1<<s))?(x[s]*sf).hi:fixed[s];
                    if(rounded[s]<l[s] || rounded[s]>h[s])feasible=false;
                    const Wide delta=Wide(rounded[s])/sf-origin[s];cost=cost+delta*delta;
                }
                if(!feasible || (found && !less(cost,best_cost)) ||
                   !final_accepted(rounded,sf,g,targets[0],targets[1],tolerance))continue;
                found=true;best_cost=cost;for(int s=0;s<5;++s)output[s]=rounded[s];
            }
        }
    }
    return found?ok:nonconverged;
}
}}}
