#include "device_state.h"
#include "fluid_explicit.h"
#include "upwind_operator.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(int status,const char *operation)
{
    if(status) throw std::runtime_error(std::string(operation)+": "+std::to_string(status));
}
template<class T> void read(std::ifstream &stream,T *data,size_t count)
{
    if(!stream.read(reinterpret_cast<char *>(data),count*sizeof(T))) throw std::runtime_error("truncated replay input");
}
template<class T> void write(const std::string &path,const std::vector<T> &data)
{
    std::ofstream stream(path,std::ios::binary);
    if(!stream.write(reinterpret_cast<const char *>(data.data()),data.size()*sizeof(T)))
        throw std::runtime_error("replay write failed: "+path);
}
}

int main(int argc,char **argv)
{
    try {
        if(argc!=3) throw std::runtime_error("usage: mp1_replay_probe input.bin output_prefix");
        std::ifstream input(argv[1],std::ios::binary);
        int32_t header[4];read(input,header,4);
        const int *upper=header,halo=header[3];
        const size_t nodes=size_t(upper[0]+2*halo+1)*(upper[1]+2*halo+1)*(upper[2]+2*halo+1);
        const size_t physical=size_t(upper[0]+1)*(upper[1]+1)*(upper[2]+1);
        std::vector<double> q(5*nodes),p(6*nodes),metric(9*nodes),jac(nodes),observed(q.size()),rhs(5*physical);
        read(input,q.data(),q.size());read(input,p.data(),p.size());
        read(input,metric.data(),metric.size());read(input,jac.data(),jac.size());
        void *state=nullptr;
        check(astr_device_state_create(0,upper[0],upper[1],upper[2],halo,1,&state),"create replay state");
        check(astr_device_fluid_prepare(state,0),"prepare primitives");
        check(astr_device_fluid_rhs_prepare(state,0),"prepare RHS");
        check(astr_device_upwind_prepare_mixed_flux(state),"prepare MP1");
        auto upload=[&](int field,const double *values,size_t count) {
            check(astr_device_state_upload(state,field,values,count*sizeof(double)),"upload frozen phase");
        };
        upload(ASTR_DEVICE_Q,q.data(),q.size());upload(ASTR_DEVICE_RHO,p.data(),nodes);
        upload(ASTR_DEVICE_VELOCITY,p.data()+nodes,3*nodes);
        upload(ASTR_DEVICE_PRESSURE,p.data()+4*nodes,nodes);upload(ASTR_DEVICE_TEMPERATURE,p.data()+5*nodes,nodes);
        upload(ASTR_DEVICE_METRIC,metric.data(),metric.size());upload(ASTR_DEVICE_JACOBIAN,jac.data(),jac.size());
        check(astr_device_fluid_rhs_clear(state),"clear replay RHS");
        AstrDeviceView flux{};check(astr_device_state_view(state,ASTR_DEVICE_INTERFACE_FLUX,&flux),"interface view");
        if(flux.element_bytes!=sizeof(float)) throw std::runtime_error("replay did not select FP32 interface storage");
        const size_t count=size_t(flux.extent[0])*flux.extent[1]*flux.extent[2]*flux.extent[3];
        std::vector<float> interfaces(count);
        const int ends[3]={0,0,0},first[3]={0,0,0};
        for(int axis=0;axis<3;++axis) {
            std::fill(interfaces.begin(),interfaces.end(),std::numeric_limits<float>::quiet_NaN());
            check(astr_device_state_upload(state,ASTR_DEVICE_INTERFACE_FLUX,interfaces.data(),count*sizeof(float)),"seed padding");
            check(astr_device_upwind_axis(state,axis,3,0,1.4,.1,.3,0,ends,ends,first,upper),"replay original MP7 axis");
            check(astr_device_state_download(state,ASTR_DEVICE_INTERFACE_FLUX,interfaces.data(),count*sizeof(float)),"download interfaces");
            write(std::string(argv[2])+".axis"+std::to_string(axis)+".f32",interfaces);
        }
        check(astr_device_state_download(state,ASTR_DEVICE_RHS,rhs.data(),rhs.size()*sizeof(double)),"download RHS");
        check(astr_device_state_download(state,ASTR_DEVICE_Q,observed.data(),observed.size()*sizeof(double)),"check input preservation");
        if(std::memcmp(q.data(),observed.data(),q.size()*sizeof(double))) throw std::runtime_error("MP1 replay changed frozen q");
        write(std::string(argv[2])+".rhs.f64",rhs);
        check(astr_device_state_destroy(&state),"release replay state");
        std::cout<<"MP1_REPLAY_PASS physical_nodes="<<physical<<" authority_bytes_unchanged=1\n";
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
