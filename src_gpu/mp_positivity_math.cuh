#pragma once
#include <cfloat>
#include <cmath>

namespace astr {
// Keep both production kinetic-energy orderings: either can lose positivity
// before the other when velocity, momentum or their squares approach overflow.
template<class Real> __device__ bool mp_state_measures(const Real q[5],Real &kinetic,Real &primitive_kinetic)
{
    for(int c=0;c<5;++c) if(!isfinite(q[c])) return false;
    if(q[0]<=Real(0)) return false;
    Real velocity[3],products[3],squared[3];
    for(int c=0;c<3;++c) {
        velocity[c]=q[c+1]/q[0];
        if(!isfinite(velocity[c])) return false;
    }
    for(int c=0;c<3;++c) {
        products[c]=Real(.5)*q[c+1]*velocity[c];
        if(!isfinite(products[c])) return false;
    }
    kinetic=(products[0]+products[1])+products[2];
    if(!isfinite(kinetic)) return false;
    for(int c=0;c<3;++c) {
        squared[c]=velocity[c]*velocity[c];
        if(!isfinite(squared[c])) return false;
    }
    primitive_kinetic=Real(.5)*q[0]*((squared[0]+squared[1])+squared[2]);
    if(!isfinite(primitive_kinetic)) return false;
    return q[4]>kinetic && q[4]>primitive_kinetic;
}

template<class Real> __device__ bool mp_admissible(const Real q[5])
{
    Real kinetic,primitive_kinetic;
    return mp_state_measures(q,kinetic,primitive_kinetic);
}

template<class Real> __device__ bool mp_interior_state(const Real base[5],const Real correction[5],
    Real theta,Real base_internal)
{
    constexpr Real epsilon=sizeof(Real)==sizeof(float) ? Real(FLT_EPSILON) : Real(DBL_EPSILON);
    constexpr Real u=Real(.5)*epsilon;
    constexpr Real gamma2=Real(2)*u/(Real(1)-Real(2)*u);
    constexpr Real gamma16=Real(16)*u/(Real(1)-Real(16)*u);
    Real trial[5],kinetic,primitive_kinetic;
    for(int c=0;c<5;++c) trial[c]=base[c]+theta*correction[c];
    if(!mp_state_measures(trial,kinetic,primitive_kinetic)) return false;
    const Real density_margin=fmin(Real(.5)*base[0],gamma2*fabs(base[0])+gamma2*fabs(theta*correction[0]));
    const Real energy_margin=fmin(Real(.5)*base_internal,gamma16*fabs(base[4])
        +gamma16*fabs(theta*correction[4])+gamma16*fmax(kinetic,primitive_kinetic));
    return trial[0]>=density_margin && trial[4]-kinetic>=energy_margin && trial[4]-primitive_kinetic>=energy_margin;
}

template<class Real> __device__ Real mp_face_budget(const Real base[5],const Real correction[5])
{
    Real kinetic,primitive_kinetic;
    if(!mp_state_measures(base,kinetic,primitive_kinetic)) return Real(0);
    const Real base_internal=fmin(base[4]-kinetic,base[4]-primitive_kinetic);
    if(mp_interior_state(base,correction,Real(1),base_internal)) return Real(1);
    Real lower=0,upper=1;
    for(int iteration=0;iteration<64;++iteration) {
        const Real middle=Real(.5)*(lower+upper);
        if(middle==lower || middle==upper) break;
        if(mp_interior_state(base,correction,middle,base_internal)) lower=middle;
        else upper=middle;
    }
    Real trial[5];
    for(int c=0;c<5;++c) trial[c]=base[c]+lower*correction[c];
    return mp_admissible(trial) ? lower : Real(-1);
}
}
