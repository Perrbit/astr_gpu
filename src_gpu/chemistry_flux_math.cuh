#pragma once
#include "chemistry_properties_math.cuh"

// Fixed AIR5 bounded interface projection and original RK energy budgets.
// These are flux constraints, not clipping or renormalization of a cell state.
namespace astr { namespace air5 {
constexpr int ratio_iterations=53;
constexpr double transport_gamma=(128.*(.5*DBL_EPSILON))/(1.-128.*(.5*DBL_EPSILON));
__device__ inline double sum5(const double a[5])
{double value=0;for(int s=0;s<5;++s)value+=a[s];return value;}
__device__ inline double dot5(const double a[5],const double b[5])
{double value=0;for(int s=0;s<5;++s)value+=a[s]*b[s];return value;}
__device__ inline double abs_sum5(const double a[5])
{double value=0;for(int s=0;s<5;++s)value+=fabs(a[s]);return value;}
__device__ inline double bounded(double x,double lo,double hi) {return fmax(lo,fmin(hi,x));}
__device__ inline bool finite5(const double a[5])
{for(int s=0;s<5;++s)if(!finite(a[s]))return false;return true;}
__device__ inline double interior_ratio(double value)
{
    double result=bounded(value,0.,1.);
    if(result>0 && result<1)result=nextafter(result,-static_cast<double>(INFINITY));
    return result;
}
__device__ inline double transport_species_ratio(double base,double negative,double scale)
{
    if(negative>=0)return 1.;
    const double reserve=scale>0?nextafter(transport_gamma*scale,static_cast<double>(INFINITY)):0.;
    return interior_ratio(fmax(0.,base-reserve)/(-negative+reserve));
}
__device__ inline int close_species_flux(const double candidate[5],const double lower[5],const double upper[5],
    double mass,double flux[5])
{
    for(int s=0;s<5;++s)flux[s]=candidate[s];
    if(!finite5(candidate)||!finite5(lower)||!finite5(upper)||!finite(mass))return nonfinite;
    double magnitude=fabs(mass);
    for(int s=0;s<5;++s) {
        if(lower[s]>upper[s])return infeasible_flux;
        magnitude=fmax(magnitude,fmax(fabs(candidate[s]),fmax(fabs(lower[s]),fabs(upper[s]))));
    }
    if(magnitude==0)return ok;
    double c[5],lo[5],hi[5],trial[5];const double target=mass/magnitude;
    for(int s=0;s<5;++s) {
        c[s]=candidate[s]/magnitude;lo[s]=lower[s]/magnitude;hi[s]=upper[s]/magnitude;
        if((lo[s]==0 && lower[s]!=0)||(hi[s]==0 && upper[s]!=0))return nonconverged;
    }
    if(target==0 && mass!=0)return nonconverged;
    if(target<sum5(lo)||target>sum5(hi))return infeasible_flux;
    const double tolerance=32.*DBL_EPSILON*fmax(fabs(target),abs_sum5(c));
    bool inside=true;
    for(int s=0;s<5;++s)inside=inside && candidate[s]>=lower[s] && candidate[s]<=upper[s];
    if(inside && fabs(sum5(c)-target)<=tolerance)return ok;
    double left=lo[0]-c[0],right=hi[0]-c[0],middle,residual;
    for(int s=1;s<5;++s) {left=fmin(left,lo[s]-c[s]);right=fmax(right,hi[s]-c[s]);}
    int iteration=0;
    for(;;) {
        ++iteration;middle=.5*left+.5*right;
        for(int s=0;s<5;++s)trial[s]=bounded(c[s]+middle,lo[s],hi[s]);
        residual=sum5(trial)-target;
        if(residual==0 || middle==left || middle==right)break;
        if(iteration>=2*ratio_iterations && fabs(residual)<=tolerance)break;
        if(residual<0)left=middle;else right=middle;
    }
    if(fabs(residual)>tolerance)return nonconverged;
    for(int s=0;s<5;++s)trial[s]=bounded(trial[s]*magnitude,lower[s],upper[s]);
    double scaled=0;for(int s=0;s<5;++s)scaled+=trial[s]/magnitude;scaled-=target;
    while(fabs(scaled)>tolerance || fabs(residual)>tolerance) {
        if(middle==left || middle==right)return nonconverged;
        if(fabs(scaled)>tolerance)residual=scaled;
        if(residual<0)left=middle;else right=middle;
        middle=.5*left+.5*right;
        for(int s=0;s<5;++s)trial[s]=bounded(c[s]+middle,lo[s],hi[s]);
        residual=sum5(trial)-target;
        scaled=0;
        for(int s=0;s<5;++s) {trial[s]=bounded(trial[s]*magnitude,lower[s],upper[s]);scaled+=trial[s]/magnitude;}
        scaled-=target;
    }
    for(int s=0;s<5;++s)flux[s]=trial[s];
    return ok;
}
__device__ inline int close_species_flux_moment(const double candidate[5],const double lower[5],const double upper[5],
    double mass,const double weights[5],double moment,double flux[5])
{
    for(int s=0;s<5;++s)flux[s]=candidate[s];
    if(!finite5(candidate)||!finite5(lower)||!finite5(upper)||!finite5(weights)||!finite(mass)||!finite(moment))return nonfinite;
    double scale_w=0,scale_f=fabs(mass);
    for(int s=0;s<5;++s) {
        if(lower[s]>upper[s])return infeasible_flux;
        scale_w=fmax(scale_w,fabs(weights[s]));
        scale_f=fmax(scale_f,fmax(fabs(candidate[s]),fmax(fabs(lower[s]),fabs(upper[s]))));
    }
    if(scale_w==0)return moment!=0?infeasible_flux:close_species_flux(candidate,lower,upper,mass,flux);
    if(scale_f==0)return moment==0?ok:infeasible_flux;
    double c[5],lo[5],hi[5],g[5],dlo[5],dhi[5],best[5];bool inside=true;
    const double target_mass=mass/scale_f,target_moment=(moment/scale_w)/scale_f;
    if(!finite(target_moment))return nonconverged;
    for(int s=0;s<5;++s) {
        c[s]=candidate[s]/scale_f;lo[s]=lower[s]/scale_f;hi[s]=upper[s]/scale_f;g[s]=weights[s]/scale_w;
        if((lo[s]==0 && lower[s]!=0)||(hi[s]==0 && upper[s]!=0))return nonconverged;
        inside=inside && candidate[s]>=lower[s] && candidate[s]<=upper[s];
    }
    const double tolerance=64.*DBL_EPSILON*fmax(fmax(fabs(target_mass),fabs(target_moment)),abs_sum5(c));
    if(inside && fabs(sum5(c)-target_mass)<=tolerance && fabs(dot5(g,c)-target_moment)<=tolerance)return ok;
    for(int s=0;s<5;++s) {dlo[s]=(lower[s]-candidate[s])/scale_f;dhi[s]=(upper[s]-candidate[s])/scale_f;}
    double delta_mass=(mass-sum5(candidate))/scale_f;
    double delta_moment=((moment-dot5(weights,candidate))/scale_w)/scale_f;
    if(fabs(delta_mass)<=tolerance)delta_mass=0;
    if(fabs(delta_moment)<=tolerance)delta_moment=0;
    bool found=false;double best_cost=DBL_MAX;
    for(int active=0;active<243;++active) {
        int digits=active,nfree=0;bool free[5],skip=false;double trial[5],mean=0,variance=0;
        for(int s=0;s<5;++s) {
            const int kind=digits%3;digits/=3;free[s]=kind==0;
            trial[s]=kind==0?0.:(kind==1?dlo[s]:dhi[s]);
            if(free[s]) {++nfree;mean+=g[s];}
        }
        if(nfree>0) {
            mean/=nfree;
            for(int s=0;s<5;++s)if(free[s])variance+=(g[s]-mean)*(g[s]-mean);
            for(int refinement=0;refinement<3;++refinement) {
                const double r1=delta_mass-sum5(trial),r2=delta_moment-dot5(g,trial);
                double weighted_abs=0;for(int s=0;s<5;++s)weighted_abs+=fabs(g[s]*trial[s]);
                const double mass_tol=fmax(tolerance,64.*DBL_EPSILON*(fabs(delta_mass)+abs_sum5(trial)));
                const double moment_tol=fmax(tolerance,64.*DBL_EPSILON*(fabs(delta_moment)+weighted_abs));
                double coefficient=0;
                if(variance>0)coefficient=(r2-mean*r1)/variance;
                else if(fabs(r2-mean*r1)>moment_tol+fabs(mean)*mass_tol) {skip=true;break;}
                for(int s=0;s<5;++s)if(free[s])trial[s]=trial[s]+r1/nfree+coefficient*(g[s]-mean);
            }
        }
        if(skip)continue;
        for(int s=0;s<5;++s)if(trial[s]<dlo[s] || trial[s]>dhi[s])skip=true;
        if(skip)continue;
        double weighted_abs=0,cost=0;
        for(int s=0;s<5;++s) {weighted_abs+=fabs(g[s]*trial[s]);cost+=trial[s]*trial[s];}
        const double mass_tol=fmax(tolerance,64.*DBL_EPSILON*(fabs(delta_mass)+abs_sum5(trial)));
        const double moment_tol=fmax(tolerance,64.*DBL_EPSILON*(fabs(delta_moment)+weighted_abs));
        if(fabs(sum5(trial)-delta_mass)>mass_tol || fabs(dot5(g,trial)-delta_moment)>moment_tol)continue;
        if(!found || cost<best_cost) {for(int s=0;s<5;++s)best[s]=trial[s];best_cost=cost;found=true;}
    }
    if(!found)return infeasible_flux;
    double scaled[5],mass_abs=0,moment_abs=0;
    for(int s=0;s<5;++s) {
        best[s]=bounded(candidate[s]+best[s]*scale_f,lower[s],upper[s]);scaled[s]=best[s]/scale_f;
        mass_abs+=fabs(scaled[s]);moment_abs+=fabs(g[s]*scaled[s]);
    }
    if(fabs(sum5(scaled)-target_mass)>tolerance+64.*DBL_EPSILON*mass_abs ||
       fabs(dot5(g,scaled)-target_moment)>tolerance+64.*DBL_EPSILON*moment_abs)return nonconverged;
    for(int s=0;s<5;++s)flux[s]=best[s];
    return ok;
}
// The original driver computes the floor energies on the CPU once and uploads
// them. Do not recompute them on the device at a rounded energy boundary.
template<bool fused=false>
__device__ inline double vibrational_excess(double ev,const double species[5],const double floor[5])
{for(int s=0;s<5;++s)ev=fused?fma(-floor[s],species[s],ev):ev-floor[s]*species[s];return ev;}
template<bool fused=false>
__device__ inline double translational_margin(const double q[11])
{
    if(q[0]<=0)return -DBL_MAX;
    double thermal=q[4]-q[10];
    for(int s=0;s<5;++s) {
        if(fused)thermal=fma(-q[5+s],fma(data::temperature_min,cv_tr(s),data::formation_energy[s]),thermal);
        else thermal-=q[5+s]*(data::formation_energy[s]+data::temperature_min*cv_tr(s));
    }
    const double twice_kinetic=fused ? fma(q[3],q[3],fma(q[2],q[2],q[1]*q[1])) :
        q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
    return thermal-twice_kinetic/(2.*q[0]);
}
template<bool fused=false>
__device__ inline bool state_admissible(const double q[11],const double floor[5])
{
    for(int s=0;s<5;++s)if(q[5+s]<0)return false;
    return q[0]>0 && vibrational_excess<fused>(q[10],q+5,floor)>=0 && translational_margin<fused>(q)>=0;
}
__device__ inline int constraint_mask(const double q[11],const double floor[5],int &species_mask)
{
    int mask=0;species_mask=0;
    for(int s=0;s<5;++s)if(q[5+s]<0)species_mask|=1<<s;
    if(species_mask)mask|=1;
    if(vibrational_excess(q[10],q+5,floor)<0)mask|=2;
    if(q[0]<=0)mask|=4;
    if(translational_margin(q)<0)mask|=8;
    return mask;
}
template<bool fused=false>
__device__ inline bool face_state_admissible(const double q[11],const double floor[5],bool species_budget)
{
    return species_budget?q[0]>0 && vibrational_excess<fused>(q[10],q+5,floor)>=0 && translational_margin<fused>(q)>=0
                         :state_admissible<fused>(q,floor);
}
template<bool fused=false>
__device__ inline double admissible_face_ratio(const double base[11],const double correction[11],bool diagnostics,
    int &mask,int &species_mask,bool species_budget,const double floor[5])
{
    double trial[11];mask=0;species_mask=0;
    for(int c=0;c<11;++c)trial[c]=fused ? fma(6.,correction[c],base[c]) : base[c]+6.*correction[c];
    if(face_state_admissible<fused>(trial,floor,species_budget))return 1.;
    double lower=0,upper=1;
    for(int iteration=0;iteration<ratio_iterations;++iteration) {
        const double middle=.5*(lower+upper);
        for(int c=0;c<11;++c)trial[c]=fused ? fma(6.*middle,correction[c],base[c]) : base[c]+6.*middle*correction[c];
        if(face_state_admissible<fused>(trial,floor,species_budget))lower=middle;else upper=middle;
    }
    if(diagnostics) {
        for(int c=0;c<11;++c)trial[c]=fused ? fma(6.*upper,correction[c],base[c]) : base[c]+6.*upper*correction[c];
        mask=constraint_mask(trial,floor,species_mask);
        if(species_budget) {mask&=~1;species_mask=0;}
    }
    return interior_ratio(lower);
}
__device__ inline int symmetric_face_trial(const double high[11],const double low[11],const double base_left[11],
    const double base_right[11],const double ql[11],const double qr[11],double cdt,bool preserve_all,
    bool check_left,bool check_right,const double floor[5],double flux[11],double &accepted_beta)
{
    for(int c=0;c<11;++c)flux[c]=high[c];accepted_beta=-1.;
    double density,u[3],t,y[5],tv,p;
    int status=conservative_to_primitive(ql,preserve_all,density,u,t,y,tv,p);if(status)return status;
    double tf=.5*t,tvf=.5*tv,uf[3];for(int d=0;d<3;++d)uf[d]=.5*u[d];
    status=conservative_to_primitive(qr,preserve_all,density,u,t,y,tv,p);if(status)return status;
    tf+=.5*t;tvf+=.5*tv;for(int d=0;d<3;++d)uf[d]+=.5*u[d];
    const double kinetic=.5*(uf[0]*uf[0]+uf[1]*uf[1]+uf[2]*uf[2]);
    double gas[5],ev[5],energy[5],lower[5],upper[5],best[11];
    for(int s=0;s<5;++s) {
        gas[s]=gas_constant(s);ev[s]=species_ev(s,tvf);energy[s]=data::formation_energy[s]+cv_tr(s)*tf+ev[s]+kinetic;
        const double left=128.*DBL_EPSILON*(fabs(base_left[5+s])+6.*cdt*(2.*fabs(low[5+s])+fabs(high[5+s])));
        const double right=128.*DBL_EPSILON*(fabs(base_right[5+s])+6.*cdt*(2.*fabs(low[5+s])+fabs(high[5+s])));
        lower[s]=low[5+s]-fmax(0.,base_right[5+s]-right)/(6.*cdt);
        upper[s]=low[5+s]+fmax(0.,base_left[5+s]-left)/(6.*cdt);
    }
    double beta_lo=0,beta_hi=1,beta=1;
    for(int iteration=0;iteration<=ratio_iterations+2;++iteration) {
        double trial[11],species[5],delta[5];for(int c=0;c<11;++c)trial[c]=low[c]+beta*(high[c]-low[c]);
        if(!check_left) {
            const double error=128.*DBL_EPSILON*(fabs(trial[0])+abs_sum5(lower));
            for(int s=0;s<5;++s) {double bound=0;for(int r=0;r<5;++r)if(r!=s)bound+=lower[r];upper[s]=trial[0]-bound+error;}
        } else if(!check_right) {
            const double error=128.*DBL_EPSILON*(fabs(trial[0])+abs_sum5(upper));
            for(int s=0;s<5;++s) {double bound=0;for(int r=0;r<5;++r)if(r!=s)bound+=upper[r];lower[s]=trial[0]-bound-error;}
        }
        for(int s=0;s<5;++s)
            if(high[5+s]==0 && low[5+s]==0 && base_left[5+s]==0 && base_right[5+s]==0 && ql[5+s]==0 && qr[5+s]==0)
                lower[s]=upper[s]=0.;
        const int projection=close_species_flux_moment(trial+5,lower,upper,trial[0],gas,dot5(gas,trial+5),species);
        bool admissible=projection==ok;
        if(admissible) {
            for(int s=0;s<5;++s) {delta[s]=species[s]-trial[5+s];trial[5+s]=species[s];}
            trial[10]+=dot5(ev,delta);trial[4]+=dot5(energy,delta);
            double left[11],right[11];
            for(int c=0;c<11;++c) {left[c]=base_left[c]-6.*cdt*(trial[c]-low[c]);right[c]=base_right[c]+6.*cdt*(trial[c]-low[c]);}
            if(check_left)admissible=admissible && state_admissible(left,floor);
            if(check_right)admissible=admissible && state_admissible(right,floor);
        }
        if(iteration==ratio_iterations+2) {
            if(!admissible)return 201;
            for(int c=0;c<11;++c)flux[c]=trial[c];accepted_beta=beta;return ok;
        }
        if(iteration==0 && admissible) {for(int c=0;c<11;++c)flux[c]=trial[c];accepted_beta=beta;return ok;}
        if(iteration==0) {beta=0;continue;}
        if(iteration==1 && !admissible)return projection==ok?200:100+projection;
        if(admissible) {for(int c=0;c<11;++c)best[c]=trial[c];beta_lo=beta;}else beta_hi=beta;
        beta=.5*(beta_lo+beta_hi);
        if(iteration==ratio_iterations+1)beta=beta_lo*(1.-transport_gamma);
    }
    for(int c=0;c<11;++c)flux[c]=best[c];accepted_beta=beta_lo;return ok;
}
}}
