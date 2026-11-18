#pragma once
#include "chemistry_properties_math.cuh"

namespace astr { namespace air5 { namespace boundary {
__device__ inline int similarity_farfield(const double base[11],double origin,double x,double result[11])
{
    for(int c=0;c<11;++c)result[c]=0.;
    const double density=base[0];if(density<=0. || origin<=0. || x<0.)return invalid_density;
    const double scale=sqrt(origin/(origin+x)),momentum=base[2]*scale;
    for(int c=0;c<11;++c)result[c]=base[c];
    result[2]=momentum;result[4]=base[4]+(momentum*momentum-base[2]*base[2])/(2.*density);return ok;
}
__device__ inline int hbl_outflow(const double one[11],const double two[11],bool preserve,double result[11])
{
    double high[11],trial[11],rho,u[3],t,y[5],tv,p;
    for(int c=0;c<11;++c)result[c]=high[c]=(4.*one[c]-two[c])/3.;
    int status=conservative_to_primitive(result,preserve,rho,u,t,y,tv,p);if(status==ok)return ok;
    for(int c=0;c<11;++c)result[c]=one[c];
    status=conservative_to_primitive(result,preserve,rho,u,t,y,tv,p);
    if(status!=ok) {for(int c=0;c<11;++c)result[c]=0.;return status;}
    for(int c=0;c<11;++c)trial[c]=high[c];bool species_limited=false;
    constexpr double inward=1.-256.*0x1p-52;
    for(int c=6;c<=9;++c)if(high[c]<0.) {
        species_limited=true;
        double theta=one[c]<=0.?0.:(one[c]/(one[c]-high[c]))*inward;
        theta=fmax(0.,fmin(1.,theta));trial[c]=fma(theta,high[c]-one[c],one[c]);
    }
    if(species_limited) {
        // Original device path closes the dependent species by successive subtraction.
        trial[5]=high[0];for(int c=6;c<=9;++c)trial[5]-=trial[c];
        status=conservative_to_primitive(trial,preserve,rho,u,t,y,tv,p);
        if(status==ok) {for(int c=0;c<11;++c)result[c]=trial[c];return ok;}
    }
    double upper=1.;
    for(int c=5;c<=9;++c)if(high[c]<0.)upper=one[c]<=0.?0.:fmin(upper,one[c]/(one[c]-high[c]));
    upper=fmax(0.,fmin(1.,upper));if(upper<1.)upper*=inward;
    // Original NVHPC CPU/device code fuses these blends, including near cancellation.
    for(int c=0;c<11;++c)trial[c]=fma(upper,high[c]-one[c],one[c]);
    status=conservative_to_primitive(trial,preserve,rho,u,t,y,tv,p);
    if(status==ok) {for(int c=0;c<11;++c)result[c]=trial[c];return ok;}
    double lower=0.;
    for(int iteration=0;iteration<64;++iteration) {
        const double theta=.5*(lower+upper);
        for(int c=0;c<11;++c)trial[c]=fma(theta,high[c]-one[c],one[c]);
        status=conservative_to_primitive(trial,preserve,rho,u,t,y,tv,p);
        if(status==ok) {lower=theta;for(int c=0;c<11;++c)result[c]=trial[c];}
        else upper=theta;
    }
    return ok;
}
__device__ inline int pressure_outlet(const double inner[11],double target,bool preserve,double result[11])
{
    for(int c=0;c<11;++c)result[c]=0.;
    if(!pressure_in_domain(target))return out_of_domain;
    double rho,u[3],temperature,y[5],tv,pressure;
    int status=conservative_to_primitive(inner,preserve,rho,u,temperature,y,tv,pressure);if(status)return status;
    if(u[0]<=0)return out_of_domain;
    double r=0,cv=0;
    for(int s=0;s<5;++s) {r+=y[s]*gas_constant(s);cv+=y[s]*cv_tr(s);}
    const double gamma=1.+r/cv,c=sqrt(gamma*pressure/rho);
    if(u[0]>=c || target==pressure) {for(int m=0;m<11;++m)result[m]=inner[m];return ok;}
    const double dp=target-pressure,new_rho=rho+dp/(c*c);
    if(new_rho<=0)return invalid_density;
    u[0]-=dp/(rho*c);
    if(u[0]<=0 || u[0]>=sqrt(gamma*target/new_rho))return out_of_domain;
    temperature=target/(new_rho*r);
    return primitive_to_conservative(new_rho,u,temperature,y,tv,result);
}
__device__ inline int isothermal_wall(const double one[11],const double two[11],double wall_t,bool preserve,double result[11])
{
    for(int c=0;c<11;++c)result[c]=0.;
    double rho_one,rho_two,u_one[3],u_two[3],t_one,t_two,y_one[5],y_two[5],tv_one,tv_two,p_one,p_two;
    int status=conservative_to_primitive(one,preserve,rho_one,u_one,t_one,y_one,tv_one,p_one);if(status)return status;
    status=conservative_to_primitive(two,preserve,rho_two,u_two,t_two,y_two,tv_two,p_two);if(status)return status;
    const double wall_p=(4.*p_one-p_two)/3.;if(wall_p<=0)return invalid_density;
    double r=0.;for(int s=0;s<5;++s)r+=(y_one[s]*data::ru)/data::molar_mass[s];
    const double wall_rho=wall_p/(r*wall_t),velocity[3]={};
    return primitive_to_conservative(wall_rho,velocity,wall_t,y_one,wall_t,result);
}
}}}
