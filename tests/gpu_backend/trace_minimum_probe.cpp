#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "insitu_rk45_device.h"

namespace {
using Real=astr_insitu::IntegrationReal;
struct Tangent {
    bool curved;
    bool operator()(const Real *p,Real,Real *v) const {
        if(curved) {const Real r=std::sqrt(p[0]*p[0]+p[1]*p[1]);v[0]=-p[1]/r;v[1]=p[0]/r;v[2]=0;}
        else {v[0]=1;v[1]=v[2]=0;}
        return true;
    }
};
struct Everywhere {bool operator()(const Real *) const {return true;}};
struct DiscardVertex {void operator()(int,const Real *,Real,Real) const {}};
}

int main()
{
    static_assert(sizeof(Real)==8,"This audit uses the current FP64 streamline path");
    constexpr Real minimum=1e-5,maximum=.02,tolerance=1e-8,target=2;
    constexpr int budget=10000;
    for(const bool curved:{false,true}) for(const int direction:{-1,1}) for(const Real initial:{Real(.002),minimum}) {
        astr_insitu::TraceState state{{curved?Real(1):Real(0),0,0},direction*initial,0,0,0,astr_insitu::TraceActive};
        while(state.status==astr_insitu::TraceActive)
            astr_insitu::trace_segment(Tangent{curved},Everywhere{},DiscardVertex{},state,128,
                minimum,maximum,tolerance,target,budget);
        const Real signed_length=direction*state.length;
        const Real error=curved ? std::fmax(std::abs(state.point[0]-std::cos(signed_length)),
            std::abs(state.point[1]-std::sin(signed_length))) : std::abs(state.point[0]-signed_length);
        std::printf("TRACE_MINIMUM_AUDIT curved=%d direction=%d initial=%.17g status=%d steps=%d length=%.17g suggested=%.17g error=%.17g position_error=%.17g\n",
            curved,direction,initial,state.status,state.steps,state.length,state.suggested,state.error,error);
        if(!std::isfinite(state.length) || error>2e-10) return 1;
        if((state.status!=astr_insitu::TraceLength && !astr_insitu::trace_subminimum_stop(state,target,minimum)) ||
            std::abs(state.length-target)>2e-10 || state.steps>=budget) return 2;
    }
    std::printf("TRACE_MINIMUM_RECOVERY_CONFIRMED constant_and_circle_forward_and_backward_reach_target=1\n");
    return 0;
}
