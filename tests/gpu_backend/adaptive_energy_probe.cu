#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include "fluid_explicit.h"

using Real=astr_numerical_real;
void check(int status)
{
    if(status) throw std::runtime_error("adaptive component device call failed");
}
void require(bool value,const char *message)
{
    if(!value) throw std::runtime_error(message);
}

int main()
{
    check(astr_device_bind(0));
    for(int sync=0;sync<=1;++sync) {
        const int nx=9,ny=7,nz=5,h=2;
        void *state=nullptr;
        check(astr_device_state_create(0,nx,ny,nz,h,sync,&state));
        check(astr_device_fluid_prepare(state,0));
        AstrDeviceView view{};check(astr_device_state_view(state,ASTR_DEVICE_Q,&view));
        const size_t nodes=view.stride[3];
        std::vector<Real> q(nodes*5),saved(q.size());
        auto at=[&](int i,int j,int k,int c)->Real& {
            return q[(i+h)+(j+h)*view.stride[1]+(k+h)*view.stride[2]+c*nodes];
        };
        for(int c=0;c<5;++c) for(int k=-h;k<=nz+h;++k)
            for(int j=-h;j<=ny+h;++j) for(int i=-h;i<=nx+h;++i)
                at(i,j,k,c)=c==0 ? Real(2) : (c==1 ? Real(i+20) :
                    (c==2 ? Real(.5)*Real(j+20) : (c==3 ? Real(.25)*Real(k+20) : Real(3000))));
        auto square_sum=[](int n) {return 400*n+20*n*(n-1)+(n-1)*n*(2*n-1)/6;};
        const Real expected=Real(.5)*(ny*nz*square_sum(nx)+Real(.25)*nx*nz*square_sum(ny)
            +Real(.0625)*nx*ny*square_sum(nz));
        const auto base=q;
        for(int scenario=0;scenario<6;++scenario) {
            q=base;
            if(scenario==1) at(0,0,0,0)=0;
            if(scenario==2) at(0,0,0,1)=std::numeric_limits<Real>::quiet_NaN();
            if(scenario==3) at(0,0,0,1)=std::numeric_limits<Real>::max()/Real(4);
            if(scenario==4) at(nx,ny,nz,0)=0; // Duplicate upper node is outside this diagnostic's sum.
            if(scenario==5) at(0,0,0,0)=-1;
            check(astr_device_state_upload(state,ASTR_DEVICE_Q,q.data(),q.size()*sizeof(Real)));
            Real values[2]{};check(astr_device_fluid_adaptive_energy_sum(state,values));
            const bool valid=scenario==0||scenario==4;
            require(values[1]==Real(!valid),"adaptive invalid-node classification");
            if(valid) require(values[0]==expected,"adaptive unique-node polynomial sum");
            check(astr_device_state_download(state,ASTR_DEVICE_Q,saved.data(),saved.size()*sizeof(Real)));
            require(std::memcmp(q.data(),saved.data(),q.size()*sizeof(Real))==0,"adaptive observer changed q words");
        }
        check(astr_device_state_destroy(&state));
        check(astr_device_state_create(0,0,0,0,0,sync,&state));
        check(astr_device_fluid_prepare(state,0));
        Real empty[2]{};check(astr_device_fluid_adaptive_energy_sum(state,empty));
        require(empty[0]==0&&empty[1]==0,"adaptive empty owned-node sum");
        check(astr_device_state_destroy(&state));
        std::printf("ADAPTIVE_ENERGY_COMPONENT_PASS bytes=%zu sync=%d scenarios=7\n",sizeof(Real),sync);
    }
}
