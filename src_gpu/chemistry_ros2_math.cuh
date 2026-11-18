#pragma once
#include "chemistry_kinetics_math.cuh"
#include "chemistry_compensation_math.cuh"

namespace astr { namespace air5 {
// Same binary64 coefficient construction as chemistry_ros2.
constexpr double ros2_gamma=1.707106781186547524400844362104849039;
constexpr double ros2_a21=1./ros2_gamma,ros2_c21=-2./ros2_gamma;
constexpr double ros2_m1=3./(2.*ros2_gamma),ros2_m2=1./(2.*ros2_gamma);
constexpr double ros2_e1=ros2_m2,ros2_e2=ros2_m2;
struct Ros2Counts { int accepted=0,rejected=0,rhs=0,jacobian=0; };

__device__ inline int validate_ros2_candidate(double rho,const double momentum[3],double q5,const double state[6])
{
    int status=validate_state(rho,state);
    if(status!=ok) return status;
    if(!finite(state[5])) return nonfinite;
    if(state[5]<0) return out_of_domain;
    double t,tv,p;
    status=temperature_from_q5(rho,momentum,state,state[5],q5,t);
    if(status!=ok) return status;
    status=tv_from_ev(state,state[5],tv);
    if(status!=ok) return status;
    status=pressure(state,t,p);
    if(status!=ok) return status;
    return pressure_in_domain(p)?ok:out_of_domain;
}
__device__ inline int ros2_fixed_step(double rho,const double momentum[3],double q5,const double state[6],
    double step,double candidate[6],double estimate[6],bool &retryable,int &rhs_evaluations,
    int &jacobian_evaluations,int mode,bool source_only,double *compensation=nullptr)
{
    for(int v=0;v<6;++v) { candidate[v]=state[v];estimate[v]=0; }
    retryable=false;rhs_evaluations=0;jacobian_evaluations=0;
    if(compensation) for(int v=0;v<6;++v) if(!finite(compensation[v])) return nonfinite;
    if(!finite(step)||step<=0) return invalid_timestep;
    if(mode==frozen) return validate_ros2_candidate(rho,momentum,q5,state);
    double source[6],jacobian[36],progress[12],matrix[36],lu[36],stage[6],stage_source[6],k1[6],k2[6],rhs[6],carry[6];
    int pivots[6];
    int status=instantaneous_source<true>(rho,momentum,q5,state,source,jacobian,progress,mode);
    rhs_evaluations=1;jacobian_evaluations=1;
    if(status!=ok) return status;
    retryable=true;
    for(int row=0;row<6;++row) {
        for(int col=0;col<6;++col) matrix[row+6*col]=-step*ros2_gamma*jacobian[row+6*col];
        matrix[row+6*row]+=1.;rhs[row]=step*ros2_gamma*source[row];
    }
    status=lu_factor_6(matrix,lu,pivots);
    if(status!=ok) return status;
    status=lu_solve_6(lu,pivots,rhs,k1);
    if(status!=ok) return status;
    for(int v=0;v<6;++v) stage[v]=state[v]+ros2_a21*k1[v];
    status=source_only?instantaneous_source<false>(rho,momentum,q5,stage,stage_source,nullptr,progress,mode):
        instantaneous_source<true>(rho,momentum,q5,stage,stage_source,jacobian,progress,mode);
    rhs_evaluations=2;
    if(status!=ok) return status;
    for(int v=0;v<6;++v) rhs[v]=step*ros2_gamma*stage_source[v]+ros2_gamma*ros2_c21*k1[v];
    status=lu_solve_6(lu,pivots,rhs,k2);
    if(status!=ok) return status;
    for(int v=0;v<6;++v) {
        candidate[v]=state[v]+ros2_m1*k1[v]+ros2_m2*k2[v];
        if(compensation) {
            candidate[v]=state[v];carry[v]=compensation[v];
            compensated_add(candidate[v],carry[v],ros2_m1*k1[v]);
            compensated_add(candidate[v],carry[v],ros2_m2*k2[v]);
        }
        estimate[v]=ros2_e1*k1[v]+ros2_e2*k2[v];
        if(!finite(candidate[v])||!finite(estimate[v])) {
            for(int i=0;i<6;++i) { candidate[i]=state[i];estimate[i]=0; }
            return nonfinite;
        }
    }
    status=validate_ros2_candidate(rho,momentum,q5,candidate);
    if(status!=ok) for(int i=0;i<6;++i) { candidate[i]=state[i];estimate[i]=0; }
    if(status==ok && compensation) for(int i=0;i<6;++i) compensation[i]=carry[i];
    return status;
}
__device__ inline double ros2_weighted_error(const double state[6],const double candidate[6],
    const double estimate[6],double rtol,const double atol[6])
{
    double scaled[6],maximum=0;
    for(int v=0;v<6;++v) {
        scaled[v]=fabs(estimate[v])/(atol[v]+rtol*fmax(fabs(state[v]),fabs(candidate[v])));
        maximum=fmax(maximum,scaled[v]);
    }
    if(!finite(maximum)) return DBL_MAX;
    if(maximum==0) return 0;
    double squares=0;
    for(int v=0;v<6;++v) { const double ratio=scaled[v]/maximum;squares+=ratio*ratio; }
    return maximum*sqrt(squares/6.);
}
__device__ inline int ros2_advance(double rho,const double momentum[3],double q5,const double state[6],
    double duration,double initial_step,double rtol,const double atol[6],int max_attempts,
    double final_state[6],double &suggested_step,Ros2Counts &counts,int mode,bool source_only,
    double *compensation=nullptr,double *history=nullptr,const double *activity_threshold=nullptr,bool *skipped=nullptr)
{
    for(int v=0;v<6;++v) final_state[v]=state[v];
    if(skipped) *skipped=false;
    suggested_step=0;counts=Ros2Counts{};
    if(!finite(duration)||!finite(initial_step)||duration<=0 || initial_step<=0) return invalid_timestep;
    if(history && (!finite(*history)||*history<0)) return invalid_timestep;
    if(!finite(rtol)) return invalid_tolerance;
    for(int v=0;v<6;++v) if(!finite(atol[v])||atol[v]<=0) return invalid_tolerance;
    if(rtol<=0) return invalid_tolerance;
    if(max_attempts<=0) return step_limit;
    double current_carry[6],trial_carry[6];
    if(compensation) for(int v=0;v<6;++v) {
        if(!finite(compensation[v])) return nonfinite;
        current_carry[v]=compensation[v];
    }
    int status=validate_ros2_candidate(rho,momentum,q5,state);
    if(status!=ok) return status;
    if(activity_threshold) {
        if(!finite(*activity_threshold)||*activity_threshold<0) return invalid_tolerance;
        if(*activity_threshold>0) {
            double source[6],progress[12],components[9];ActivityData data;
            status=instantaneous_source<false,true>(rho,momentum,q5,state,source,nullptr,progress,mode,&data);
            counts.rhs=1;
            if(status!=ok) return status;
            const double activity=activity_index(rho,state,duration,data,components);
            if(activity<DBL_MAX && activity<=*activity_threshold) {
                if(skipped) *skipped=true;
                suggested_step=duration;
                return ok;
            }
        }
    }
    if(mode==frozen) {
        suggested_step=duration;if(history) *history=duration;
        return ok;
    }
    double current[6];for(int v=0;v<6;++v) current[v]=state[v];
    double elapsed=0,step=fmin(initial_step,duration);
    if(history && *history>0) step=fmin(*history,duration);
    bool previous_rejected=false;
    for(int attempt=0;attempt<max_attempts;++attempt) {
        const double remaining=duration-elapsed,unclipped_step=step;
        step=fmin(step,remaining);
        if(step<=0 || elapsed+step==elapsed) { suggested_step=step;return step_limit; }
        if(compensation) for(int v=0;v<6;++v) trial_carry[v]=current_carry[v];
        double candidate[6],estimate[6];bool retryable;int rhs,jac;
        status=ros2_fixed_step(rho,momentum,q5,current,step,candidate,estimate,retryable,rhs,jac,
                              mode,source_only,compensation?trial_carry:nullptr);
        counts.rhs+=rhs;counts.jacobian+=jac;
        if(status==ok) {
            const double error=ros2_weighted_error(current,candidate,estimate,rtol,atol);
            if(error<=1.) {
                for(int v=0;v<6;++v) { current[v]=candidate[v];if(compensation) current_carry[v]=trial_carry[v]; }
                elapsed+=step;++counts.accepted;
                double factor=error==0?5.:fmin(5.,fmax(.2,.9/sqrt(error)));
                if(previous_rejected) factor=fmin(1.,factor);
                suggested_step=step*factor;previous_rejected=false;
                if(elapsed>=duration) {
                    for(int v=0;v<6;++v) { final_state[v]=current[v];if(compensation) compensation[v]=current_carry[v]; }
                    if(history) {
                        *history=fmin(suggested_step,duration);
                        if(step<unclipped_step) *history=fmin(unclipped_step,duration);
                    }
                    return ok;
                }
                step=history?suggested_step:fmin(suggested_step,duration-elapsed);
                continue;
            }
            const double factor=finite(error)?fmax(.2,fmin(1.,.9/sqrt(error))):.2;
            step*=factor;
        } else if(retryable) step=.5*step;
        else { suggested_step=step;return status; }
        ++counts.rejected;suggested_step=step;previous_rejected=true;
    }
    return step_limit;
}
}}
