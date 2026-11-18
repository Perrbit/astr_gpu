#pragma once

namespace astr { namespace air5 {
// Returns carry for the separately computed shared high-part mean.
__device__ inline double compensated_mean(double left,double right,double cl,double cr)
{
    const double total=left+right,virtual_part=total-left;
    const double error=(left-(total-virtual_part))+(right-virtual_part);
    return .5*(cl+cr)-.5*error;
}
__device__ inline void compensated_normalize(double &high,double &carry)
{
    const double total=high-carry,virtual_part=total-high;
    const double error=(high-(total-virtual_part))+(-carry-virtual_part);
    high=total;carry=-error;
}
// Keep the original ordered IEEE operations; do not reassociate these residuals.
__device__ inline void compensated_add(double &high,double &carry,double increment)
{
    const double adjusted=increment-carry,updated=high+adjusted;
    carry=(updated-high)-adjusted;
    high=updated;
}
__device__ inline void compensated_rk(double &high,double &carry,double origin,double origin_carry,
                                      double a,double b,double increment)
{
    const double difference=origin-high,virtual_part=difference-origin;
    const double error=(origin-(difference-virtual_part))-(high+virtual_part);
    carry=a*origin_carry+b*carry;
    compensated_add(high,carry,a*difference);
    compensated_add(high,carry,a*error);
    compensated_add(high,carry,increment);
}
// closure is the zero-based species index; q stores species in components 6:10.
__device__ inline void compensated_filter_projection(double high[11],double carry[11],
                                                       const double species[5],int closure)
{
    const int c=5+closure;
    for(int s=0;s<5;++s) if(s!=closure && high[5+s]!=species[s]) {
        high[5+s]=species[s];carry[5+s]=0.;
    }
    high[c]=high[0];carry[c]=carry[0];
    for(int s=0;s<5;++s) if(s!=closure) {
        const int m=5+s;
        const double total=high[c]-high[m],virtual_part=total-high[c];
        const double error=(high[c]-(total-virtual_part))-(high[m]+virtual_part);
        carry[c]=(carry[c]-carry[m])-error;high[c]=total;
    }
    for(int m=0;m<11;++m) {
        const double total=high[m]-carry[m],virtual_part=total-high[m];
        const double error=(high[m]-(total-virtual_part))+(-carry[m]-virtual_part);
        high[m]=total;carry[m]=-error;
    }
}
}}
