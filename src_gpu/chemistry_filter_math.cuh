#pragma once
#include "chemistry_flux_math.cuh"

namespace astr { namespace air5 { namespace filter {
// Original IEEE expansions: compile with contraction and flushing disabled.
constexpr double unit_roundoff=.5*DBL_EPSILON;
constexpr double sign_gamma=32.*unit_roundoff/(1.-32.*unit_roundoff);
__device__ inline double up(double x) {return nextafter(x,static_cast<double>(INFINITY));}
__device__ inline double down(double x) {return nextafter(x,-static_cast<double>(INFINITY));}
__device__ inline double pressure_floor()
{
    double factor=0;for(int s=0;s<5;++s)factor=fmax(factor,data::cv_tr_factor[s]);
    return factor*data::pressure_min;
}
__device__ inline void two_product(double a,double b,double &high,double &low)
{
    constexpr double splitter=134217729.;high=a*b;
    const double split_a=splitter*a,split_b=splitter*b;
    const double ah=split_a-(split_a-a),al=a-ah,bh=split_b-(split_b-b),bl=b-bh;
    const double error1=high-ah*bh,error2=error1-al*bh,error3=error2-ah*bl;
    low=al*bl-error3;
}
__device__ inline void grow_expansion(double *expansion,int &length,double term)
{
    double q=term;int out=0;
    for(int i=0;i<length;++i) {
        const double b=expansion[i],total=q+b,virtual_b=total-q,virtual_a=total-virtual_b;
        const double error_b=b-virtual_b,error_a=q-virtual_a,error=error_a+error_b;
        if(error!=0)expansion[out++]=error;q=total;
    }
    if(q!=0 || out==0)expansion[out++]=q;
    length=out;
}
__device__ inline void add_product(double *expansion,int &length,double a,double b)
{
    double high,low;two_product(a,b,high,low);
    grow_expansion(expansion,length,low);grow_expansion(expansion,length,high);
}
__device__ inline bool vibration_sign(const double state[11],const double floor[5])
{
    double expansion[32];int length=0;grow_expansion(expansion,length,state[10]);
    for(int s=0;s<5;++s)add_product(expansion,length,-state[5+s],floor[s]);
    return expansion[length-1]>=0;
}
__device__ inline bool mass_in_exact_bounds(double mass,const double lower[5],const double upper[5])
{
    double low[32],high[32];int nl=0,nh=0;
    grow_expansion(low,nl,mass);grow_expansion(high,nh,-mass);
    for(int s=0;s<5;++s) {grow_expansion(low,nl,-lower[s]);grow_expansion(high,nh,upper[s]);}
    return low[nl-1]>=0 && high[nh-1]>=0;
}
__device__ inline double feasible_mass(const double candidate[5],const double lower[5],const double upper[5],double raw)
{
    double mass=fmax(sum5(lower),fmin(sum5(upper),raw));
    for(;;) {
        double magnitude=fabs(mass);for(int s=0;s<5;++s)magnitude=fmax(magnitude,fmax(fabs(candidate[s]),fmax(fabs(lower[s]),fabs(upper[s]))));
        if(magnitude==0)return mass;
        double lo[5],hi[5];for(int s=0;s<5;++s) {lo[s]=lower[s]/magnitude;hi[s]=upper[s]/magnitude;}
        const double target=mass/magnitude;
        if(mass_in_exact_bounds(mass,lower,upper) && mass_in_exact_bounds(target,lo,hi) && target>=sum5(lo) && target<=sum5(hi))return mass;
        mass=nextafter(mass,-copysign(static_cast<double>(INFINITY),mass));
    }
}
__device__ inline bool thermal_sign(const double state[11],const double floor[5],double offset)
{
    double thermal[32],result[32];int length=0,result_length=0;
    grow_expansion(thermal,length,state[4]);grow_expansion(thermal,length,-state[10]);grow_expansion(thermal,length,-offset);
    for(int s=0;s<5;++s)add_product(thermal,length,-state[5+s],floor[s]);
    const double twice_rho=2.*state[0];
    for(int i=0;i<length;++i)add_product(result,result_length,twice_rho,thermal[i]);
    for(int i=1;i<4;++i)add_product(result,result_length,-state[i],state[i]);
    return result[result_length-1]>=0;
}
__device__ inline bool lower_energy_admissible(const double state[11],const double tr_floor[5],const double v_floor[5])
{
    double thermal=state[4]-state[10],vibration=state[10];
    double tr_scale=fabs(state[4])+fabs(state[10]),v_scale=fabs(state[10]);
    for(int s=0;s<5;++s) {
        double term=state[5+s]*tr_floor[s];thermal-=term;tr_scale+=fabs(term);
        term=state[5+s]*v_floor[s];vibration-=term;v_scale+=fabs(term);
    }
    const double kinetic=((state[1]*state[1]+state[2]*state[2])+state[3]*state[3])/(2.*state[0]);
    const double margin=thermal-kinetic;tr_scale+=fabs(kinetic);
    const bool vibration_ok=fabs(vibration)>up(sign_gamma*v_scale)?vibration>0:vibration_sign(state,v_floor);
    if(!vibration_ok)return false;
    return fabs(margin)>up(sign_gamma*tr_scale)?margin>0:thermal_sign(state,tr_floor,0.);
}
__device__ inline bool thermal_margin_admissible(const double state[11],const double floor[5],double offset)
{
    double residual=(state[4]-state[10])-offset,scale=fabs(state[4])+fabs(state[10])+offset;
    for(int s=0;s<5;++s) {const double term=state[5+s]*floor[s];residual-=term;scale+=fabs(term);}
    const double kinetic=((state[1]*state[1]+state[2]*state[2])+state[3]*state[3])/(2.*state[0]);
    residual-=kinetic;scale+=fabs(kinetic);
    return fabs(residual)>up(sign_gamma*scale)?residual>0:thermal_sign(state,floor,offset);
}
__device__ inline bool pressure_admissible(const double state[11])
{
    return thermal_margin_admissible(state,data::formation_energy,pressure_floor());
}
__device__ inline void state_magnitude(const double high[11],const double carry[11],double magnitude[11])
{
    for(int s=0;s<11;++s) {magnitude[s]=fabs(high[s])+fabs(carry[s]);if(magnitude[s]>0)magnitude[s]=up(magnitude[s]);}
}
__device__ __forceinline__ bool pair_box_safe(const double state[11],const double magnitude[11],const double flux[11],bool use_carry,
    const double tr_floor[5],const double v_floor[5])
{
    const double commit=up((use_carry?13.:2.)*unit_roundoff/(1.-(use_carry?13.:2.)*unit_roundoff));
    const double pair=use_carry?up(3.*unit_roundoff/(1.-3.*unit_roundoff)):unit_roundoff;
    const double coefficient=up(commit+pair);double worst[11],center[11],radius[11];
    for(int s=0;s<11;++s) {
        center[s]=state[s]+2.*flux[s];worst[s]=center[s];
        const double scale=magnitude[s]+2.*fabs(flux[s]);
        radius[s]=scale>0?up(coefficient*up(scale)):0.;
        if(radius[s]==0)continue;
        if(s==0 || s==4)worst[s]=down(center[s]-radius[s]);
        else if(s<4)worst[s]=up(fabs(center[s])+radius[s]);
        else if(s<10)worst[s]=data::formation_energy[s-5]>=0?up(center[s]+radius[s]):down(center[s]-radius[s]);
        else worst[s]=up(center[s]+radius[s]);
    }
    if(worst[0]<=0 || !pressure_admissible(worst))return false;
    // Pressure and the two temperature floors have different minimizing corners.
    for(int s=5;s<10;++s)if(radius[s]>0)
        worst[s]=tr_floor[s-5]>=0?up(center[s]+radius[s]):down(center[s]-radius[s]);
    if(!thermal_margin_admissible(worst,tr_floor,0.))return false;
    if(radius[10]>0)worst[10]=down(center[10]-radius[10]);
    for(int s=5;s<10;++s)if(radius[s]>0)worst[s]=up(center[s]+radius[s]);
    return vibration_sign(worst,v_floor);
}
__device__ inline bool represented_pressure_sign(const double high[11],const double carry[11])
{
    double thermal[128],result[128];int length=0,result_length=0;
    grow_expansion(thermal,length,high[4]);grow_expansion(thermal,length,-carry[4]);
    grow_expansion(thermal,length,-high[10]);grow_expansion(thermal,length,carry[10]);grow_expansion(thermal,length,-pressure_floor());
    for(int s=0;s<5;++s) {
        add_product(thermal,length,-high[5+s],data::formation_energy[s]);
        add_product(thermal,length,carry[5+s],data::formation_energy[s]);
    }
    for(int i=0;i<length;++i) {
        add_product(result,result_length,2.*high[0],thermal[i]);add_product(result,result_length,-2.*carry[0],thermal[i]);
    }
    for(int s=1;s<4;++s) {
        add_product(result,result_length,-high[s],high[s]);add_product(result,result_length,2.*high[s],carry[s]);
        add_product(result,result_length,-carry[s],carry[s]);
    }
    return result[result_length-1]>=0;
}
__device__ inline bool pressure_represented_admissible(const double high[11],const double carry[11])
{
    double state[11],magnitude[11];for(int s=0;s<11;++s)state[s]=high[s]-carry[s];
    if(state[0]<=0)return false;state_magnitude(high,carry,magnitude);
    const double floor=pressure_floor();double residual=(state[4]-state[10])-floor,scale=magnitude[4]+magnitude[10]+floor;
    for(int s=0;s<5;++s) {residual-=state[5+s]*data::formation_energy[s];scale+=magnitude[5+s]*fabs(data::formation_energy[s]);}
    const double kinetic=((state[1]*state[1]+state[2]*state[2])+state[3]*state[3])/(2.*state[0]);
    residual-=kinetic;scale+=fabs(kinetic);
    const double gamma=64.*unit_roundoff/(1.-64.*unit_roundoff);
    return fabs(residual)>up(gamma*scale)?residual>0:represented_pressure_sign(high,carry);
}
__device__ inline bool lower_admissible(const double state[11],const double tr_floor[5],const double v_floor[5])
{
    for(int c=0;c<11;++c)if(!finite(state[c]))return false;
    if(state[0]<=0)return false;for(int s=0;s<5;++s)if(state[5+s]<0)return false;
    return lower_energy_admissible(state,tr_floor,v_floor) && pressure_admissible(state);
}
__device__ inline void face_bounds(const double left[11],const double right[11],const double high[11],
    double lower[5],double upper[5],double &mass)
{
    for(int s=0;s<5;++s) {
        double half=.5*left[5+s];if(half>0)half=down(half);lower[s]=-half;
        half=.5*right[5+s];if(half>0)half=down(half);upper[s]=half;
    }
    mass=feasible_mass(high+5,lower,upper,high[0]);
}
__device__ __forceinline__ bool pair_admissible(const double left[11],const double right[11],const double flux[11],
    const double tr_floor[5],const double v_floor[5],const double lm[11],const double rm[11],bool use_carry)
{
    double a[11],b[11],negative[11];
    for(int s=0;s<11;++s) {negative[s]=-flux[s];a[s]=left[s]+2.*flux[s];b[s]=right[s]-2.*flux[s];}
    return lower_admissible(a,tr_floor,v_floor) && lower_admissible(b,tr_floor,v_floor) &&
        pair_box_safe(left,lm,flux,use_carry,tr_floor,v_floor) && pair_box_safe(right,rm,negative,use_carry,tr_floor,v_floor);
}
__device__ __forceinline__ int thermal_limit(const double left[11],const double right[11],const double candidate[11],
    const double tr_floor[5],const double v_floor[5],const double lm[11],const double rm[11],bool use_carry,
    double flux[11],double &ratio)
{
    for(int s=0;s<11;++s)flux[s]=candidate[s];ratio=1.;
    if(!lower_admissible(left,tr_floor,v_floor) || !lower_admissible(right,tr_floor,v_floor))return invalid_composition;
    const double zero[11]={};
    if(!pair_box_safe(left,lm,zero,use_carry,tr_floor,v_floor) || !pair_box_safe(right,rm,zero,use_carry,tr_floor,v_floor)) {
        for(int s=0;s<11;++s)flux[s]=0.;ratio=0.;return ok;
    }
    if(pair_admissible(left,right,flux,tr_floor,v_floor,lm,rm,use_carry))return ok;
    double lower=0.,upper=1.;
    for(int iteration=0;iteration<ratio_iterations;++iteration) {
        const double middle=.5*(lower+upper);for(int s=0;s<11;++s)flux[s]=middle*candidate[s];
        if(pair_admissible(left,right,flux,tr_floor,v_floor,lm,rm,use_carry))lower=middle;else upper=middle;
    }
    ratio=lower/(1.+transport_gamma);if(ratio>0)ratio=down(ratio);
    for(int s=0;s<11;++s)flux[s]=ratio*candidate[s];
    return pair_admissible(left,right,flux,tr_floor,v_floor,lm,rm,use_carry)?ok:invalid_composition;
}
// Stencils use the original Fortran layout (component,offset=-4:5).
__device__ inline void high_face(const double stencil[110],const double carry[110],int global_face,int intervals,
    bool periodic,bool use_carry,double high[11])
{
    constexpr double coefficients[5]={105./512.,-15./128.,45./1024.,-5./512.,1./1024.};
    constexpr int difference4[5]={1,-4,6,-4,1};
    for(int s=0;s<11;++s)high[s]=0.;
    if(periodic) {
        for(int s=0;s<11;++s)for(int m=1;m<=5;++m) {
            double pairs=0;
            for(int r=0;r<m;++r) {
                double pair=stencil[11*(r+5)+s]-stencil[11*(4-r)+s];
                if(use_carry)pair-=carry[11*(r+5)+s]-carry[11*(4-r)+s];pairs+=pair;
            }
            high[s]+=coefficients[m-1]*pairs;
        }
    } else {
        if(global_face<=0 || global_face>=intervals-1)return;
        const int first=global_face-4>1?global_face-4:1,last=global_face<intervals-6?global_face:intervals-6;
        for(int s=0;s<11;++s) {
            for(int row=first;row<=last;++row) {
                double difference5=0.;
                for(int k=0;k<=4;++k) {
                    const int r=row+k-global_face;
                    double pair=stencil[11*(r+5)+s]-stencil[11*(r+4)+s];
                    if(use_carry)pair-=carry[11*(r+5)+s]-carry[11*(r+4)+s];difference5+=difference4[k]*pair;
                }
                high[s]+=difference4[global_face-row]*difference5;
            }
            high[s]*=1./1024.;
        }
    }
}
}}}
