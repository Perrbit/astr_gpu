#pragma once
#include "chemistry_properties_math.cuh"

// Existing fixed AIR5 equations. Local matrices retain first-index-contiguous storage.
namespace astr { namespace air5 {
enum SourceMode { coupled=0, chemical=1, vt_only=2, frozen=3 };
struct ActivityData {
    double production[5],loss[5],energy[2],temperatures[2];
};

template<bool derivatives>
__device__ inline int vt_source(const double z[5],double t,double tv,const double dt[6],
                                const double dtv[6],double &source,double *ds)
{
    source=0;
    if(derivatives) for(int v=0;v<6;++v) ds[v]=0;
    int status=validate_species(z);
    if(status!=ok) return status;
    if(!finite(t)||!finite(tv)) return nonfinite;
    if(derivatives) for(int v=0;v<6;++v) if(!finite(dt[v])||!finite(dtv[v])) return nonfinite;
    if(t<data::temperature_min || t>data::temperature_max || tv<data::temperature_min ||
       tv>data::temperature_max) return out_of_domain;
    double p;
    status=pressure(z,t,p);
    if(status!=ok) return status;
    if(!pressure_in_domain(p)||!data::relaxation_uses_relaxing_species_density) return out_of_domain;
    double concentration[5],mole[5],sum=0,gas_density=0;
    for(int r=0;r<5;++r) {
        concentration[r]=z[r]/data::molar_mass[r];sum+=concentration[r];
        if(derivatives) gas_density+=z[r]*gas_constant(r);
    }
    for(int r=0;r<5;++r) mole[r]=concentration[r]/sum;
    constexpr double avogadro=6.02214076e23,patm=101325.,pi=3.1415926535897932384626433832795;
    for(int m=0;m<5;++m) {
        if(data::theta_v[m]<=0 || z[m]<=0) continue;
        double rate[5],park[5],mw[5],density_derivative[5];
        for(int r=0;r<5;++r) {
            const int index=m+5*r;
            const double mw_time=patm/p*exp(data::mw_a[index]*(pow(t,-1./3.)-data::mw_b[index])-18.42);
            if(!finite(mw_time)) return nonfinite;
            if(mw_time<=0) return out_of_domain;
            if(derivatives) { park[r]=0;mw[r]=1;density_derivative[r]=0; }
            if(data::collision_sigma0[index]>0) {
                const double sigma=data::collision_sigma0[index]*pow(t,data::collision_sigma_power[index]);
                const double reduced_mass=data::molar_mass[m]*data::molar_mass[r]/
                    (data::molar_mass[m]+data::molar_mass[r]);
                const double speed=sqrt(8.*data::ru*t/(pi*reduced_mass));
                const double coefficient=((avogadro/data::molar_mass[m])*sigma)*speed;
                if(!finite(sigma)||!finite(speed)||!finite(coefficient)) return nonfinite;
                const double collision_rate=coefficient*z[m],product=mw_time*collision_rate,denominator=1.+product;
                if(!finite(collision_rate)||!finite(product)||!finite(denominator)) return nonfinite;
                rate[r]=collision_rate/denominator;
                if(derivatives) {
                    park[r]=1./denominator;mw[r]=product/denominator;
                    density_derivative[r]=(coefficient/denominator)/denominator;
                }
            } else rate[r]=1./mw_time;
            if(!finite(rate[r])) return nonfinite;
            if(derivatives && !finite(density_derivative[r])) return nonfinite;
        }
        double inverse_time=0;
        for(int r=0;r<5;++r) inverse_time+=mole[r]*rate[r];
        const double difference=species_vibrational_difference(m,t,tv);
        source+=z[m]*difference*inverse_time;
        if(derivatives) for(int v=0;v<6;++v) {
            double dp=gas_density*dt[v];
            if(v<5) dp+=gas_constant(v)*t;
            if(!finite(dp)) return nonfinite;
            double inverse_derivative=0;
            for(int r=0;r<5;++r) {
                double dx=0;
                if(v<5) {
                    dx=-mole[r]/(data::molar_mass[v]*sum);
                    if(v==r) dx+=1./(data::molar_mass[v]*sum);
                }
                const double dlog_mw=-dp/p-(data::mw_a[m+5*r]*pow(t,-1./3.)/3.)*(dt[v]/t);
                const double dlog_collision=(data::collision_sigma_power[m+5*r]+.5)*(dt[v]/t);
                double dr=rate[r]*(park[r]*dlog_collision-mw[r]*dlog_mw);
                if(v==m) dr+=density_derivative[r];
                if(!finite(dlog_mw)||!finite(dlog_collision)||!finite(dr)) return nonfinite;
                inverse_derivative=inverse_derivative+dx*rate[r]+mole[r]*dr;
            }
            const double de=species_cvv(m,t)*dt[v]-species_cvv(m,tv)*dtv[v];
            if(v==m) ds[v]+=difference*inverse_time;
            ds[v]+=z[m]*(de*inverse_time+difference*inverse_derivative);
        }
    }
    if(!finite(source)) return nonfinite;
    if(derivatives) for(int v=0;v<6;++v) if(!finite(ds[v])) return nonfinite;
    return ok;
}

__device__ inline double mass_action(const double c[5],const int *stoichiometry)
{
    double value=1;
    for(int s=0;s<5;++s) for(int exponent=0;exponent<stoichiometry[s];++exponent) value*=c[s];
    return value;
}
__device__ inline double mass_action_derivative(const double c[5],const int *stoichiometry,int v)
{
    if(v>=5 || stoichiometry[v]==0) return 0;
    double value=1;
    for(int s=0;s<5;++s) for(int exponent=0;exponent<stoichiometry[s]-(s==v);++exponent) value*=c[s];
    return double(stoichiometry[v])*value/data::molar_mass[v];
}
__device__ inline void preferential_factor(int s,double t,double &factor,double &derivative)
{
    const double *a=data::psi_coefficient+5*s;
    const double exponent=a[0]/t+a[1]+a[2]*log(t)+a[3]*t+a[4]*t*t;
    const double raw=exp(exponent),dr=raw*(-a[0]/(t*t)+a[2]/t+a[3]+2.*a[4]*t);
    if(raw<=data::psi_bounds[2*s]) { factor=data::psi_bounds[2*s];derivative=0; }
    else if(raw>=data::psi_bounds[2*s+1]) { factor=data::psi_bounds[2*s+1];derivative=0; }
    else { factor=raw;derivative=dr; }
}

template<bool with_jacobian,bool activity=false>
__device__ inline int instantaneous_source(double rho,const double momentum[3],double q5,
    const double state[6],double source[6],double *jacobian,double progress[12],int mode,ActivityData *diagnostics=nullptr)
{
    for(int v=0;v<6;++v) source[v]=0;
    for(int r=0;r<12;++r) progress[r]=0;
    if(with_jacobian) for(int i=0;i<36;++i) jacobian[i]=0;
    if(activity) {
        for(int s=0;s<5;++s) { diagnostics->production[s]=0;diagnostics->loss[s]=0; }
        for(int i=0;i<2;++i) { diagnostics->energy[i]=0;diagnostics->temperatures[i]=0; }
    }
    if(mode<coupled || mode>frozen) return invalid_source_mode;
    int status=validate_species(state);
    if(status!=ok) return status;
    for(int v=0;v<6;++v) if(!finite(state[v])) return nonfinite;
    if(state[5]<0) return out_of_domain;
    double t,tv,p,dt[6],dtv[6];
    status=temperature_from_q5(rho,momentum,state,state[5],q5,t);
    if(status!=ok) return status;
    status=tv_from_ev(state,state[5],tv);
    if(status!=ok) return status;
    status=pressure(state,t,p);
    if(status!=ok) return status;
    if(!pressure_in_domain(p)) return out_of_domain;
    if(activity) { diagnostics->temperatures[0]=t;diagnostics->temperatures[1]=tv; }
    if(mode==frozen) return ok;
    if(with_jacobian) {
        status=temperature_derivatives(state,t,dt,dt[5]);
        if(status!=ok) return status;
        status=tv_derivatives(state,tv,dtv,dtv[5]);
        if(status!=ok) return status;
    }
    if(mode==coupled || mode==chemical) {
        double concentration[5];
        for(int s=0;s<5;++s) concentration[s]=state[s]/data::molar_mass[s];
        for(int r=0;r<12;++r) {
            const int *reactant=data::nu_reactant+5*r,*product=data::nu_product+5*r;
            double third_body=1;
            if(data::has_third_body[r]) {
                third_body=0;
                for(int s=0;s<5;++s) third_body+=data::third_body_efficiency[s+5*r]*concentration[s];
            }
            const double fp=mass_action(concentration,reactant),rp=mass_action(concentration,product);
            const double tr=data::rate_uses_tv[r]?sqrt(t*tv):t;
            const double kf=data::arrhenius_a[r]*pow(tr,data::arrhenius_b[r])*exp(-data::activation_temperature[r]/tr);
            const double z=10000./t,*kc=data::ln_kc_coefficient+5*r;
            const double ln_kc=kc[0]/z+kc[1]+kc[2]*log(z)+kc[3]*z+kc[4]*z*z;
            const double kr=data::arrhenius_a[r]*pow(t,data::arrhenius_b[r])*exp(-data::activation_temperature[r]/t)/exp(ln_kc);
            const double forward=kf*third_body*fp,reverse=kr*third_body*rp,rate=forward-reverse;
            progress[r]=rate;
            double drate[6];
            if(with_jacobian) {
                const double dln_kc_dt=(-kc[0]/(z*z)+kc[2]/z+kc[3]+2.*kc[4]*z)*(-z/t);
                for(int v=0;v<6;++v) {
                    const double dtr=data::rate_uses_tv[r]?.5*tr*(dt[v]/t+dtv[v]/tv):dt[v];
                    const double dkf=kf*(data::arrhenius_b[r]/tr+data::activation_temperature[r]/(tr*tr))*dtr;
                    const double dln_kc=dln_kc_dt*dt[v];
                    const double dkr=kr*((data::arrhenius_b[r]/t+data::activation_temperature[r]/(t*t))*dt[v]-dln_kc);
                    const double dthird=data::has_third_body[r] && v<5 ?
                        data::third_body_efficiency[v+5*r]/data::molar_mass[v] : 0.;
                    const double dfp=mass_action_derivative(concentration,reactant,v);
                    const double drp=mass_action_derivative(concentration,product,v);
                    const double df=dkf*third_body*fp+kf*dthird*fp+kf*third_body*dfp;
                    const double db=dkr*third_body*rp+kr*dthird*rp+kr*third_body*drp;
                    drate[v]=df-db;
                }
            }
            for(int s=0;s<5;++s) {
                if(activity) {
                    diagnostics->production[s]+=data::molar_mass[s]*(double(product[s])*forward+double(reactant[s])*reverse);
                    diagnostics->loss[s]+=data::molar_mass[s]*(double(reactant[s])*forward+double(product[s])*reverse);
                }
                const int nu=product[s]-reactant[s];
                if(nu==0) continue;
                const double mass=data::molar_mass[s]*double(nu);
                source[s]+=mass*rate;
                if(with_jacobian) for(int v=0;v<6;++v) jacobian[s+6*v]+=mass*drate[v];
                if(data::psi_bounds[2*s+1]<=0) continue;
                double energy,psi,dpsi;
                if(data::rate_uses_tv[r]) {
                    preferential_factor(s,t,psi,dpsi);
                    energy=psi*gas_constant(s)*data::activation_temperature[r];
                } else energy=species_ev(s,tv);
                source[5]+=mass*rate*energy;
                if(with_jacobian) for(int v=0;v<6;++v) {
                    const double de=data::rate_uses_tv[r]?dpsi*dt[v]*gas_constant(s)*data::activation_temperature[r]
                        :species_cvv(s,tv)*dtv[v];
                    jacobian[5+6*v]+=mass*(drate[v]*energy+rate*de);
                }
            }
        }
    }
    if(activity) diagnostics->energy[0]=source[5];
    if(mode==coupled || mode==vt_only) {
        double vt,dvt[6];
        status=vt_source<with_jacobian>(state,t,tv,dt,dtv,vt,dvt);
        if(status!=ok) return status;
        if(activity) diagnostics->energy[1]=vt;
        source[5]+=vt;
        if(with_jacobian) for(int v=0;v<6;++v) jacobian[5+6*v]+=dvt[v];
    }
    for(int v=0;v<6;++v) {
        if(!finite(source[v])) return nonfinite;
        if(with_jacobian) for(int s=0;s<6;++s) if(!finite(jacobian[s+6*v])) return nonfinite;
    }
    for(int r=0;r<12;++r) if(!finite(progress[r])) return nonfinite;
    return ok;
}

__device__ inline double activity_index(double rho,const double state[6],double duration,
                                       const ActivityData &d,double components[9])
{
    for(int i=0;i<9;++i) components[i]=DBL_MAX;
    if(!finite(rho)||!finite(duration)||rho<=0 || duration<=0) return DBL_MAX;
    for(int i=0;i<6;++i) if(!finite(state[i])) return DBL_MAX;
    for(int i=0;i<2;++i) if(!finite(d.energy[i])||!finite(d.temperatures[i])) return DBL_MAX;
    const double t=d.temperatures[0],tv=d.temperatures[1];
    if(t<=0 || tv<=0) return DBL_MAX;
    double cv=0,cvv=0,thermal=0,vibrational=0;
    for(int s=0;s<5;++s) {
        if(!finite(d.production[s])||!finite(d.loss[s]) || d.production[s]<0 || d.loss[s]<0) return DBL_MAX;
        const double scale=1.e-12+1.e-6*fabs(state[s]/rho);
        components[s]=((duration*fmax(d.production[s],d.loss[s]))/rho)/scale;
        cv+=state[s]*cv_tr(s);cvv+=state[s]*species_cvv(s,tv);
        const double exchange=d.production[s]+d.loss[s];
        thermal+=fabs(data::formation_energy[s]+cv_tr(s)*t)*exchange;
        vibrational+=fabs(species_ev(s,tv))*exchange;
    }
    if(!finite(cv)||!finite(cvv)||cv<=0 || cvv<=0) return DBL_MAX;
    const double ev=state[5]/rho,energy_scale=1.e-10*fmax(fabs(ev),1.)+1.e-7*fabs(ev);
    for(int i=0;i<2;++i) components[5+i]=((duration*fabs(d.energy[i]))/rho)/energy_scale;
    thermal=thermal+fabs(d.energy[0])+fabs(d.energy[1]);
    vibrational=vibrational+fabs(d.energy[0])+fabs(d.energy[1]);
    components[7]=((duration*thermal)/cv)/(1.e-5+1.e-7*fabs(t));
    components[8]=((duration*vibrational)/cvv)/(1.e-5+1.e-7*fabs(tv));
    double index=0;
    for(int i=0;i<9;++i) { if(!finite(components[i])) return DBL_MAX;index=fmax(index,components[i]); }
    return index;
}

__device__ inline int lu_factor_6(const double matrix[36],double lu[36],int pivots[6])
{
    double scale[6];
    for(int row=0;row<6;++row) {
        pivots[row]=row;double row_max=0;
        for(int col=0;col<6;++col) {
            const double value=matrix[row+6*col];lu[row+6*col]=value;
            if(!finite(value)) return nonfinite;
            row_max=fmax(row_max,fabs(value));
        }
        if(row_max==0) return linear_failure;
        scale[row]=row_max;
    }
    for(int col=0;col<5;++col) {
        int pivot=col;double best=fabs(lu[col+6*col])/scale[col];
        for(int row=col+1;row<6;++row) {
            const double scaled=fabs(lu[row+6*col])/scale[row];
            if(scaled>best) { pivot=row;best=scaled; }
        }
        if(best<=64.*DBL_EPSILON) return linear_failure;
        pivots[col]=pivot;
        if(pivot!=col) {
            for(int item=0;item<6;++item) {
                double value=lu[col+6*item];lu[col+6*item]=lu[pivot+6*item];lu[pivot+6*item]=value;
            }
            double value=scale[col];scale[col]=scale[pivot];scale[pivot]=value;
        }
        for(int row=col+1;row<6;++row) {
            lu[row+6*col]=lu[row+6*col]/lu[col+6*col];
            for(int item=col+1;item<6;++item) lu[row+6*item]=lu[row+6*item]-lu[row+6*col]*lu[col+6*item];
        }
    }
    if(fabs(lu[35])/scale[5]<=64.*DBL_EPSILON) return linear_failure;
    for(int row=0;row<6;++row) for(int col=0;col<6;++col) if(!finite(lu[row+6*col])) return linear_failure;
    return ok;
}
__device__ inline int lu_solve_6(const double lu[36],const int pivots[6],const double rhs[6],double solution[6])
{
    for(int row=0;row<6;++row) {
        solution[row]=0;
        if(pivots[row]<0 || pivots[row]>=6) return linear_failure;
        if(!finite(rhs[row])) return nonfinite;
        for(int col=0;col<6;++col) if(!finite(lu[row+6*col])) return nonfinite;
    }
    for(int row=0;row<6;++row) solution[row]=rhs[row];
    for(int col=0;col<5;++col) if(pivots[col]!=col) {
        double value=solution[col];solution[col]=solution[pivots[col]];solution[pivots[col]]=value;
    }
    for(int row=1;row<6;++row) {
        double value=0;
        for(int item=0;item<row;++item) value+=lu[row+6*item]*solution[item];
        solution[row]-=value;
    }
    for(int row=5;row>=0;--row) {
        if(lu[row+6*row]==0) {
            for(int i=0;i<6;++i) solution[i]=0;
            return linear_failure;
        }
        double value=0;
        for(int item=row+1;item<6;++item) value+=lu[row+6*item]*solution[item];
        solution[row]=(solution[row]-value)/lu[row+6*row];
    }
    for(int row=0;row<6;++row) if(!finite(solution[row])) {
        for(int i=0;i<6;++i) solution[i]=0;
        return nonfinite;
    }
    return ok;
}
}}
