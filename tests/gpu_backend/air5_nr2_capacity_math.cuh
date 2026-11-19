#pragma once
#include "chemistry_filter_math.cuh"

namespace astr { namespace air5 { namespace nr2_capacity {
__device__ inline int project(const double raw[5],const double lower[5],const double upper[5],double mass,double flux[5])
{
    for(int s=0;s<5;++s)flux[s]=raw[s];
    if(!finite5(raw)||!finite5(lower)||!finite5(upper)||!finite(mass))return nonfinite;
    double scale=fabs(mass);bool inside=true;
    for(int s=0;s<5;++s) {
        if(lower[s]>upper[s])return infeasible_flux;
        inside=inside && raw[s]>=lower[s] && raw[s]<=upper[s];
        scale=fmax(scale,fmax(fabs(raw[s]),fmax(fabs(lower[s]),fabs(upper[s]))));
    }
    if(scale==0.)return ok;
    double norm[5],lo[5],hi[5],clipped[5],expansion[6];int length=0;
    const double target=mass/scale;
    if(target==0. && mass!=0.)return nonconverged;
    filter::grow_expansion(expansion,length,mass);
    for(int s=0;s<5;++s) {
        norm[s]=raw[s]/scale;lo[s]=lower[s]/scale;hi[s]=upper[s]/scale;
        if((lo[s]==0. && lower[s]!=0.) || (hi[s]==0. && upper[s]!=0.))return nonconverged;
        clipped[s]=bounded(raw[s],lower[s],upper[s]);
        filter::grow_expansion(expansion,length,-clipped[s]);
    }
    if(target<sum5(lo)||target>sum5(hi))return infeasible_flux;
    const double tol=32.*DBL_EPSILON*fmax(fabs(target),abs_sum5(norm));
    if(inside && fabs(sum5(norm)-target)<=tol)return ok;
    if(!filter::mass_in_exact_bounds(mass,lower,upper))return infeasible_flux;

    double residual=0.;
    for(int i=0;i<length;++i)residual+=expansion[i];
    double room[5],result[5];
    if(residual!=0.) {
        const double *end=residual>0.?upper:lower;
        for(int s=0;s<5;++s)room[s]=(end[s]-clipped[s])/scale;
        const double share=(residual/scale)/sum5(room);
        if(!finite(share))return nonconverged;
        // Exact feasibility bounds the share by one; use the corresponding
        // representable endpoint when the quotient rounds to/past one.
        for(int s=0;s<5;++s)result[s]=share>=1.?end[s]:fma(share,end[s]-clipped[s],clipped[s]);
    } else for(int s=0;s<5;++s)result[s]=clipped[s];
    for(int s=0;s<5;++s) {
        if(!finite(result[s]) || result[s]<lower[s] || result[s]>upper[s])return nonconverged;
        norm[s]=result[s]/scale;
    }
    if(fabs(sum5(norm)-target)>tol)return nonconverged;
    for(int s=0;s<5;++s)flux[s]=result[s];
    return ok;
}
}}}
