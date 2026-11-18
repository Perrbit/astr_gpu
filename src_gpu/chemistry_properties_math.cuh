#pragma once
#include <cfloat>
#include <cmath>
#include "chemistry_air5_data.h"

// Fixed FP64 AIR5 equations from chemistry_core/flow_state/transport_gpu.cuf.
// Indices in local vectors are zero-based; matrices retain Fortran ordering.
namespace astr { namespace air5 {
enum Status {
    ok=0, invalid_mechanism=1, invalid_density=2, invalid_composition=3,
    out_of_domain=4, no_vibrational_capacity=5, nonconverged=6,
    nonfinite=7, linear_failure=8, invalid_timestep=9, invalid_tolerance=10,
    step_limit=11, invalid_source_mode=12, infeasible_flux=13
};

__device__ inline bool finite(double x) { return x==x && fabs(x)<=DBL_MAX; }
__device__ inline int validate_species(const double z[5])
{
    double density=0;
    for(int s=0;s<5;++s) {
        if(!finite(z[s])) return nonfinite;
        if(z[s]<0) return invalid_composition;
        density+=z[s];
    }
    return density<=DBL_MIN ? invalid_composition : ok;
}
__device__ inline int validate_state(double rho,const double z[5])
{
    if(!finite(rho)) return nonfinite;
    if(rho<=0) return invalid_density;
    int status=validate_species(z);
    if(status!=ok) return status;
    double density=0;
    for(int s=0;s<5;++s) density+=z[s];
    return fabs(density-rho)>1.e-10*fmax(rho,1.) ? invalid_composition : ok;
}
__device__ inline bool pressure_in_domain(double p)
{
    const double tolerance=64.*DBL_EPSILON*data::pressure_max;
    return finite(p) && p>=data::pressure_min-tolerance && p<=data::pressure_max+tolerance;
}
__device__ inline double gas_constant(int s) { return data::ru/data::molar_mass[s]; }
__device__ inline double cv_tr(int s) { return data::cv_tr_factor[s]*gas_constant(s); }
__device__ inline double species_ev(int s,double tv)
{
    const double theta=data::theta_v[s];
    return theta>0 ? gas_constant(s)*theta/(exp(theta/tv)-1.) : 0.;
}
__device__ inline double species_cvv(int s,double tv)
{
    const double theta=data::theta_v[s];
    if(theta<=0) return 0;
    const double x=theta/tv,e=exp(x);
    return gas_constant(s)*x*x*e/((e-1.)*(e-1.));
}
__device__ inline double species_vibrational_difference(int s,double t,double tv)
{
    const double theta=data::theta_v[s];
    if(theta<=0) return 0;
    const double a=theta/t,b=theta/tv,delta=theta*((t-tv)/(t*tv));
    return gas_constant(s)*theta*exp(a)*expm1(delta)/(expm1(a)*expm1(b));
}
__device__ inline void species_vibrational_pair(int s,double tv,double &energy,double &cv)
{
    energy=0;cv=0;
    if(data::theta_v[s]<=0) return;
    const double x=data::theta_v[s]/tv,e=exp(x),denominator=e-1.,r=gas_constant(s);
    energy=r*data::theta_v[s]/denominator;
    cv=r*x*x*e/(denominator*denominator);
}
__device__ inline int ev_from_tv(const double z[5],double tv,double &ev)
{
    ev=0;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(!finite(tv)) return nonfinite;
    if(tv<data::temperature_min || tv>data::temperature_max) return out_of_domain;
    double vibrating=0;
    for(int s=0;s<5;++s) if(data::theta_v[s]>0) vibrating+=z[s];
    if(vibrating<=DBL_MIN) return no_vibrational_capacity;
    for(int s=0;s<5;++s) ev+=z[s]*species_ev(s,tv);
    return ok;
}
__device__ inline int tv_from_ev(const double z[5],double ev,double &tv)
{
    tv=data::temperature_min;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(!finite(ev)) return nonfinite;
    double vibrating=0;
    for(int s=0;s<5;++s) if(data::theta_v[s]>0) vibrating+=z[s];
    if(vibrating<=DBL_MIN) return no_vibrational_capacity;
    double ev_lower,ev_upper,ev_trial;
    status=ev_from_tv(z,data::temperature_min,ev_lower);
    if(status!=ok) return status;
    status=ev_from_tv(z,data::temperature_max,ev_upper);
    if(status!=ok) return status;
    const double tolerance=8.*DBL_EPSILON*fmax(fmax(fabs(ev_lower),fabs(ev)),1.);
    if(ev<ev_lower-tolerance || ev>ev_upper+tolerance) return out_of_domain;
    if(fabs(ev-ev_lower)<=tolerance) { tv=data::temperature_min;return ok; }
    if(fabs(ev-ev_upper)<=tolerance) { tv=data::temperature_max;return ok; }
    double lower=data::temperature_min,upper=data::temperature_max;
    for(int iteration=0;iteration<96;++iteration) {
        tv=.5*(lower+upper);
        status=ev_from_tv(z,tv,ev_trial);
        if(status!=ok) return status;
        if(fabs(ev_trial-ev)<=tolerance) return ok;
        if(ev_trial<ev) lower=tv;else upper=tv;
        if(upper-lower<=8.*DBL_EPSILON*fmax(tv,1.)) return ok;
    }
    return nonconverged;
}
__device__ inline int q5_from_state(double rho,const double momentum[3],const double z[5],
                                   double ev,double t,double &q5)
{
    q5=0;
    if(!finite(rho)||!finite(ev)||!finite(t)) return nonfinite;
    for(int d=0;d<3;++d) if(!finite(momentum[d])) return nonfinite;
    if(rho<=0) return invalid_density;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(ev<0 || t<data::temperature_min || t>data::temperature_max) return out_of_domain;
    double cv=0,formation=0;
    for(int s=0;s<5;++s) {
        cv+=z[s]*cv_tr(s);
        formation+=z[s]*data::formation_energy[s];
    }
    const double kinetic=(momentum[0]*momentum[0]+momentum[1]*momentum[1]+
        momentum[2]*momentum[2])/(2.*rho);
    q5=kinetic+cv*t+ev+formation;
    return ok;
}
__device__ inline int temperature_from_q5(double rho,const double momentum[3],const double z[5],
                                         double ev,double q5,double &t)
{
    t=data::temperature_min;
    if(!finite(rho)||!finite(ev)||!finite(q5)) return nonfinite;
    for(int d=0;d<3;++d) if(!finite(momentum[d])) return nonfinite;
    if(rho<=0) return invalid_density;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(ev<0) return out_of_domain;
    double cv=0,formation=0;
    for(int s=0;s<5;++s) {
        cv+=z[s]*cv_tr(s);
        formation+=z[s]*data::formation_energy[s];
    }
    if(cv<=DBL_MIN) return invalid_composition;
    const double kinetic=(momentum[0]*momentum[0]+momentum[1]*momentum[1]+
        momentum[2]*momentum[2])/(2.*rho);
    t=(q5-kinetic-ev-formation)/cv;
    if(!finite(t)) return nonfinite;
    const double tolerance=64.*DBL_EPSILON*data::temperature_max;
    if(t<data::temperature_min-tolerance || t>data::temperature_max+tolerance) return out_of_domain;
    t=fmin(fmax(t,data::temperature_min),data::temperature_max);
    return ok;
}
__device__ inline int temperature_derivatives(const double z[5],double t,double dz[5],double &dev)
{
    for(int s=0;s<5;++s) dz[s]=0;
    dev=0;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(!finite(t)) return nonfinite;
    if(t<data::temperature_min || t>data::temperature_max) return out_of_domain;
    double cv=0;
    for(int s=0;s<5;++s) cv+=z[s]*cv_tr(s);
    if(cv<=DBL_MIN) return invalid_composition;
    for(int s=0;s<5;++s) dz[s]=-(data::formation_energy[s]+cv_tr(s)*t)/cv;
    dev=-1./cv;
    return ok;
}
__device__ inline int tv_derivatives(const double z[5],double tv,double dz[5],double &dev)
{
    for(int s=0;s<5;++s) dz[s]=0;
    dev=0;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(!finite(tv)) return nonfinite;
    if(tv<data::temperature_min || tv>data::temperature_max) return out_of_domain;
    double cv=0;
    for(int s=0;s<5;++s) cv+=z[s]*species_cvv(s,tv);
    if(cv<=DBL_MIN) return no_vibrational_capacity;
    for(int s=0;s<5;++s) dz[s]=-species_ev(s,tv)/cv;
    dev=1./cv;
    return ok;
}
__device__ inline int pressure(const double z[5],double t,double &p)
{
    p=0;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(!finite(t)) return nonfinite;
    if(t<data::temperature_min || t>data::temperature_max) return out_of_domain;
    for(int s=0;s<5;++s) p+=z[s]*gas_constant(s)*t;
    return ok;
}
__device__ inline int frozen_sound(const double q[11],double &sound,double velocity[3])
{
    sound=-1.;
    double temperature;
    const int status=temperature_from_q5(q[0],q+1,q+5,q[10],q[4],temperature);
    if(status!=ok) return status;
    double rden=0,cvden=0;
    for(int s=0;s<5;++s) {rden+=q[5+s]*gas_constant(s);cvden+=q[5+s]*cv_tr(s);}
    for(int d=0;d<3;++d) velocity[d]=q[1+d]/q[0];
    sound=sqrt((1.+rden/cvden)*rden*temperature/q[0]);
    return ok;
}
__device__ inline int primitive_to_conservative(double rho,const double velocity[3],double t,
                                               const double y[5],double tv,double q[11])
{
    for(int c=0;c<11;++c) q[c]=0;
    if(!finite(rho)||!finite(t)||!finite(tv)) return nonfinite;
    if(rho<=0) return invalid_density;
    double sum=0,z[5],momentum[3],ev;
    for(int d=0;d<3;++d) if(!finite(velocity[d])) return nonfinite;
    for(int s=0;s<5;++s) {
        if(!finite(y[s])) return nonfinite;
        if(y[s]<0) return invalid_composition;
        sum+=y[s];z[s]=rho*y[s];
    }
    if(fabs(sum-1.)>64.*DBL_EPSILON) return invalid_composition;
    for(int d=0;d<3;++d) momentum[d]=rho*velocity[d];
    int status=validate_state(rho,z);
    if(status!=ok) return status;
    status=ev_from_tv(z,tv,ev);
    if(status!=ok) return status;
    status=q5_from_state(rho,momentum,z,ev,t,q[4]);
    if(status!=ok) return status;
    q[0]=rho;q[10]=ev;
    for(int d=0;d<3;++d) q[d+1]=momentum[d];
    for(int s=0;s<5;++s) q[s+5]=z[s];
    return ok;
}
__device__ inline int conservative_to_primitive(const double q[11],bool preserve_all,
    double &rho,double velocity[3],double &t,double y[5],double &tv,double &p)
{
    rho=0;t=0;tv=0;p=0;
    for(int d=0;d<3;++d) velocity[d]=0;
    for(int s=0;s<5;++s) y[s]=0;
    for(int c=0;c<11;++c) if(!finite(q[c])) return nonfinite;
    rho=q[0];
    const double *z=q+5,*momentum=q+1,ev=q[10];
    int status=validate_state(rho,z);
    if(status!=ok) return status;
    for(int d=0;d<3;++d) velocity[d]=momentum[d]/rho;
    if(preserve_all) {
        for(int s=0;s<5;++s) y[s]=z[s]/rho;
    } else {
        double sum=0;
        for(int s=1;s<5;++s) { y[s]=z[s]/rho;sum+=y[s]; }
        y[0]=1.-sum;
    }
    if(y[0]<0) return invalid_composition;
    status=temperature_from_q5(rho,momentum,z,ev,q[4],t);
    if(status!=ok) return status;
    status=tv_from_ev(z,ev,tv);
    if(status!=ok) return status;
    return pressure(z,t,p);
}
__device__ inline int transport_properties(double t,double tv,double p,const double y[5],
    double &viscosity,double &ktr,double &kv,double mus[5],double binary[25],double mixture[5],
    double *evs=nullptr)
{
    viscosity=0;ktr=0;kv=0;
    for(int s=0;s<5;++s) { mus[s]=0;mixture[s]=0; }
    for(int i=0;i<25;++i) binary[i]=0;
    if(!finite(t)||!finite(tv)||!finite(p)) return nonfinite;
    if(t<=0 || tv<=0 || !pressure_in_domain(p)) return out_of_domain;
    double fraction_sum=0,mole_sum=0,mole[5],ks[5],kvs[5],phi[25];
    for(int s=0;s<5;++s) {
        if(!finite(y[s])) return nonfinite;
        if(y[s]<0) return invalid_composition;
        fraction_sum+=y[s];mole_sum+=y[s]/data::molar_mass[s];
    }
    if(fabs(fraction_sum-1.)>128.*DBL_EPSILON || mole_sum<=0) return invalid_composition;
    for(int s=0;s<5;++s) mole[s]=(y[s]/data::molar_mass[s])/mole_sum;
    const double logt=log(t);
    for(int s=0;s<5;++s) {
        mus[s]=.1*exp(data::blottner[3*s]*logt*logt+data::blottner[3*s+1]*logt+
                     data::blottner[3*s+2]);
        ks[s]=(data::atom_count[2*s]+data::atom_count[2*s+1]==1?3.75:4.75)*mus[s]*gas_constant(s);
        double cv;
        if(evs) species_vibrational_pair(s,tv,evs[s],cv);else cv=species_cvv(s,tv);
        kvs[s]=mus[s]*cv;
    }
    for(int s=0;s<5;++s) {
        for(int r=0;r<5;++r) {
            const double x=1.+sqrt(mus[s]/mus[r])*pow(data::molar_mass[r]/data::molar_mass[s],.25);
            phi[s+5*r]=x*x/sqrt(8.*(1.+data::molar_mass[s]/data::molar_mass[r]));
        }
        double denominator=0;
        for(int r=0;r<5;++r) denominator+=mole[r]*phi[s+5*r];
        viscosity+=mole[s]*mus[s]/denominator;
        ktr+=mole[s]*ks[s]/denominator;
        kv+=mole[s]*kvs[s]/denominator;
    }
    for(int s=0;s<5;++s) for(int r=s;r<5;++r) {
        const double *b=data::binary_diffusion+4*(s+5*r);
        binary[s+5*r]=10.1325/p*exp(b[3])*pow(t,b[0]*logt*logt+b[1]*logt+b[2]);
        binary[r+5*s]=binary[s+5*r];
    }
    for(int s=0;s<5;++s) {
        double numerator=0,denominator=0;
        for(int r=0;r<5;++r) if(r!=s) {
            numerator+=mole[r];denominator+=mole[r]/binary[s+5*r];
        }
        if(numerator==0) mixture[s]=0;
        else if(denominator>0) mixture[s]=numerator/denominator;
        else return invalid_composition;
    }
    if(!finite(viscosity)||!finite(ktr)||!finite(kv)) return nonfinite;
    for(int s=0;s<5;++s) {
        if(!finite(mixture[s])) return nonfinite;
        for(int r=0;r<5;++r) if(!finite(binary[s+5*r])) return nonfinite;
    }
    return ok;
}
__device__ inline int diffusive_flux(double rho,const double velocity[3],double t,double tv,double p,
    const double y[5],const double grad_velocity[9],const double grad_t[3],const double grad_tv[3],
    const double grad_y[15],double momentum[9],double species[15],double energy[3],double evflux[3],
    double *enthalpy_out=nullptr,double *ev_out=nullptr)
{
    for(int i=0;i<9;++i) momentum[i]=0;
    for(int i=0;i<15;++i) species[i]=0;
    for(int d=0;d<3;++d) { energy[d]=0;evflux[d]=0; }
    if(!finite(rho)||rho<=0) return nonfinite;
    for(int d=0;d<3;++d) if(!finite(velocity[d])||!finite(grad_t[d])||!finite(grad_tv[d])) return nonfinite;
    for(int i=0;i<9;++i) if(!finite(grad_velocity[i])) return nonfinite;
    for(int i=0;i<15;++i) if(!finite(grad_y[i])) return nonfinite;
    double mu,ktr,kv,mus[5],binary[25],mixture[5],ev[5],enthalpy[5];
    int status=transport_properties(t,tv,p,y,mu,ktr,kv,mus,binary,mixture,ev);
    if(status!=ok) return status;
    const double div=grad_velocity[0]+grad_velocity[4]+grad_velocity[8];
    for(int j=0;j<3;++j) for(int i=0;i<3;++i) {
        momentum[i+3*j]=mu*(grad_velocity[i+3*j]+grad_velocity[j+3*i]);
        if(i==j) momentum[i+3*j]-=(2./3.)*mu*div;
    }
    for(int s=0;s<5;++s) enthalpy[s]=cv_tr(s)*t+ev[s]+data::formation_energy[s]+gas_constant(s)*t;
    for(int d=0;d<3;++d) {
        double correction=0,raw[5],work=0,es=0,evs=0;
        for(int s=0;s<5;++s) { raw[s]=-rho*mixture[s]*grad_y[s+5*d];correction+=raw[s]; }
        for(int s=0;s<5;++s) {
            species[s+5*d]=raw[s]-y[s]*correction;
            es+=enthalpy[s]*species[s+5*d];evs+=ev[s]*species[s+5*d];
        }
        for(int i=0;i<3;++i) work+=momentum[i+3*d]*velocity[i];
        energy[d]=work+ktr*grad_t[d]+kv*grad_tv[d]-es;
        evflux[d]=kv*grad_tv[d]-evs;
    }
    if(enthalpy_out) for(int s=0;s<5;++s) enthalpy_out[s]=enthalpy[s];
    if(ev_out) for(int s=0;s<5;++s) ev_out[s]=ev[s];
    return ok;
}
}}
