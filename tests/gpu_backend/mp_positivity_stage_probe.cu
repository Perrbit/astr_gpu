#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "device_state_internal.cuh"
#include "fluid_explicit.h"
#include "halo_transport.h"
#include "upwind_operator.h"
#include "shock_sensor.h"
#include "shock_sensor_internal.cuh"
#include "mp_positivity_operator.h"

namespace {
using Real=astr::numerical_real;
void require(bool ok,const char *message) {if(!ok) throw std::runtime_error(message);}
void check(int status) {if(status) throw std::runtime_error("MP component status="+std::to_string(status));}
template<class T> std::vector<T> read(std::ifstream &file,size_t count)
{
    std::vector<T> values(count);file.read(reinterpret_cast<char *>(values.data()),count*sizeof(T));
    require(file.good(),"stage packet length");return values;
}
size_t product(const int n[3]) {return static_cast<size_t>(n[0])*n[1]*n[2];}
size_t cube(const int p[3],const int lower[3],const int extent[3])
{
    return p[0]-lower[0]+static_cast<size_t>(extent[0])*(p[1]-lower[1]+static_cast<size_t>(extent[1])*(p[2]-lower[2]));
}
size_t offset(const AstrDeviceView &view,const int p[3],int c=1,int d=1)
{
    size_t index=0;for(int a=0;a<3;++a) index+=(p[a]-view.lower[a])*view.stride[a];
    return index+(c-view.lower[3])*view.stride[3]+(d-view.lower[4])*view.stride[4];
}
AstrDeviceView view(void *state,int field) {AstrDeviceView result{};check(astr_device_state_view(state,field,&result));return result;}
std::vector<Real> download(void *state,int field)
{
    const auto v=view(state,field);size_t n=1;for(int a=0;a<5;++a)n*=v.extent[a];
    std::vector<Real> result(n);check(astr_device_state_download(state,field,result.data(),n*sizeof(Real)));return result;
}
void upload(void *state,int field,const std::vector<Real> &values)
{check(astr_device_state_upload(state,field,values.data(),values.size()*sizeof(Real)));}
double relative(const std::vector<Real> &actual,const std::vector<Real> &expected,size_t nodes,int components)
{
    require(actual.size()==expected.size(),"comparison shape");double maximum=0;
    for(int c=0;c<components;++c) {
        double error=0,scale=1;
        for(size_t i=c*nodes;i<(c+1)*nodes;++i) {
            require(std::isfinite(actual[i]) && std::isfinite(expected[i]),"nonfinite compared state");
            error=std::max(error,std::abs(double(actual[i])-double(expected[i])));scale=std::max(scale,std::abs(double(expected[i])));
        }
        maximum=std::max(maximum,error/scale);
    }
    return maximum;
}
}

int main(int argc,char **argv)
{
    int rank=0;
    try {
        require(argc==4 || argc==5,"expected packet directory, case 0/1, pinned/device, optional diagnostic directory");
        int devices=0;check(astr_device_count(&devices));require(devices>0,"GPU required");
        const char *local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");require(local,"MPI local rank missing");
        const int device=std::atoi(local)%devices;check(astr_device_bind(device));
        check(MPI_Init(&argc,&argv));check(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
        int ranks=0;check(MPI_Comm_size(MPI_COMM_WORLD,&ranks));
        const int kind=std::stoi(argv[2]);require(kind==0 || kind==1,"case identity");
        const std::string transport=argv[3];require(transport=="pinned" || transport=="device","transport identity");
        const int mode=transport=="device" ? ASTR_HALO_DEVICE : ASTR_HALO_PINNED;
        char name[96];std::snprintf(name,sizeof(name),"/outdat/mp_stage.case%d.rank%08d.bin",kind,rank);
        std::ifstream input(std::string(argv[1])+name,std::ios::binary);
        const auto h=read<int32_t>(input,39);const auto parameters=read<Real>(input,5);
        require(h[4]==5 && h[36]==kind && h[17]*h[18]*h[19]==ranks,"packet dimensions/topology");
        const int upper[3]={h[0],h[1],h[2]},halo=h[3],zero[3]={0,0,0},minus1[3]={-1,-1,-1};
        int physical[3],full[3],faces[3],mask_shape[3],mask_lower[3],periodic[3],low[3],high[3],first[3],last[3];
        for(int a=0;a<3;++a) {
            physical[a]=upper[a]+1;full[a]=upper[a]+2*halo+1;faces[a]=upper[a]+2;
            mask_lower[a]=h[30+a];mask_shape[a]=h[33+a]-h[30+a]+1;
            periodic[a]=h[20+a]==3;
            low[a]=h[20+a]==1 || h[20+a]==4;high[a]=h[20+a]==2 || h[20+a]==4;
            first[a]=h[5+2*a];last[a]=h[6+2*a];
        }
        const size_t nf=product(full),np=product(physical),na=product(faces);
        const auto q=read<Real>(input,5*nf),pv=read<Real>(input,6*nf),metric=read<Real>(input,9*nf),jac=read<Real>(input,nf);
        const auto expected_high=read<Real>(input,15*na),delta=read<Real>(input,15*na),saved=read<Real>(input,5*np);
        const auto rhs=read<Real>(input,5*np),expected_ratio=read<Real>(input,nf),expected_rhs=read<Real>(input,5*np);
        const auto mask=read<int8_t>(input,product(mask_shape));require(input.peek()==std::char_traits<char>::eof(),"packet trailing data");
        double face_error=0,delta_error=0,ratio_error=0,rhs_error=0;
        int limited_value=0;
        for(int debug=0;debug<2;++debug) {
            void *state=nullptr;check(astr_device_state_create(device,upper[0],upper[1],upper[2],halo,debug,&state));
            check(astr_device_fluid_prepare(state,1));check(astr_device_fluid_rhs_prepare(state,0));
            check(astr_device_upwind_prepare(state));check(astr_device_shock_prepare(state));
            AstrDeviceView absent{};
            require(astr_device_state_view(state,ASTR_DEVICE_MP_DELTA,&absent)==ASTR_DEVICE_NOT_ALLOCATED &&
                astr_device_state_view(state,ASTR_DEVICE_MP_RATIO,&absent)==ASTR_DEVICE_NOT_ALLOCATED,"disabled MP scratch was allocated");
            const void *authority=view(state,ASTR_DEVICE_Q).data;
            check(astr_device_transport_prepare(state,int(MPI_Comm_c2f(MPI_COMM_WORLD)),h.data()+23,h.data()+26,h.data()+17,
                6,mode,0,periodic));
            check(astr_device_mp_positivity_prepare(state,int(MPI_Comm_c2f(MPI_COMM_WORLD)),h.data()+23,h.data()+26,mode));
            require(view(state,ASTR_DEVICE_Q).data==authority,"MP preparation replaced the authority q buffer");
            upload(state,ASTR_DEVICE_Q,q);upload(state,ASTR_DEVICE_METRIC,metric);upload(state,ASTR_DEVICE_JACOBIAN,jac);
            upload(state,ASTR_DEVICE_SAVE,saved);
            for(auto slot:std::array<std::array<int,3>,4>{{{{ASTR_DEVICE_RHO,0,1}},{{ASTR_DEVICE_VELOCITY,1,3}},
                {{ASTR_DEVICE_PRESSURE,4,1}},{{ASTR_DEVICE_TEMPERATURE,5,1}}}}) {
                std::vector<Real> values(pv.begin()+slot[1]*nf,pv.begin()+(slot[1]+slot[2])*nf);upload(state,slot[0],values);
            }
            auto &s=*static_cast<astr::State *>(state);
            std::vector<int8_t> local_mask(np),mask_faces(astr::shock_faces_size(upper),0);
            for(int k=0;k<=upper[2];++k) for(int j=0;j<=upper[1];++j) for(int i=0;i<=upper[0];++i) {
                const int p[3]={i,j,k};local_mask[cube(p,zero,physical)]=mask[cube(p,mask_lower,mask_shape)];
            }
            check(cudaMemcpy(s.shock_mask,local_mask.data(),np,cudaMemcpyHostToDevice));
            if(h[29]) {
                check(cudaMalloc(&s.shock_faces,mask_faces.size()));
                for(int a=0;a<3;++a) for(int side=0;side<2;++side) {
                    if(side==0 ? low[a] : high[a]) continue;
                    for(int v=0;v<=upper[(a+2)%3];++v) for(int u=0;u<=upper[(a+1)%3];++u) {
                        int p[3]={};p[(a+1)%3]=u;p[(a+2)%3]=v;p[a]=side==0 ? -1 : upper[a]+1;
                        mask_faces[astr::shock_face_offset(upper,a,side,p)]=mask[cube(p,mask_lower,mask_shape)];
                    }
                }
                check(cudaMemcpy(s.shock_faces,mask_faces.data(),mask_faces.size(),cudaMemcpyHostToDevice));s.shock_faces_ready=true;
            }
            if(kind==0) {
                check(astr_device_fluid_rhs_clear(state));check(astr_device_mp_positivity_begin(state));
                for(int a=0;a<3;++a) {
                    check(astr_device_mp_positivity_axis(state,a,0,parameters[0],parameters[1],parameters[2],low,high,first,last));
                    const auto flux=download(state,ASTR_DEVICE_INTERFACE_FLUX);const auto fv=view(state,ASTR_DEVICE_INTERFACE_FLUX);
                    for(int k=(a==2 ? -1 : 0);k<=upper[2];++k)
                    for(int j=(a==1 ? -1 : 0);j<=upper[1];++j)
                    for(int i=(a==0 ? -1 : 0);i<=upper[0];++i) {
                        const int p[3]={i,j,k};
                        for(int c=0;c<5;++c) {
                            const double expected=expected_high[cube(p,minus1,faces)+na*(c+5*a)];
                            const double actual=flux[offset(fv,p,c+1)];require(std::isfinite(actual)&&std::isfinite(expected),"face finite");
                            face_error=std::max(face_error,std::abs(actual-expected)/std::max(1.,std::abs(expected)));
                        }
                    }
                }
                check(astr_device_mp_positivity_faces_complete(state));
                const auto observed=download(state,ASTR_DEVICE_MP_DELTA);const auto dv=view(state,ASTR_DEVICE_MP_DELTA);
                for(int a=0;a<3;++a)
                for(int k=(a==2 ? -1 : 0);k<=upper[2];++k)
                for(int j=(a==1 ? -1 : 0);j<=upper[1];++j)
                for(int i=(a==0 ? -1 : 0);i<=upper[0];++i) {
                    const int p[3]={i,j,k};
                    for(int c=0;c<5;++c) {
                        const double expected=delta[cube(p,minus1,faces)+na*(c+5*a)],actual=observed[offset(dv,p,c+1,a+1)];
                        require(std::isfinite(actual)&&std::isfinite(expected),"delta finite");
                        delta_error=std::max(delta_error,std::abs(actual-expected)/std::max(1.,std::abs(expected)));
                    }
                }
                if(sizeof(Real)==8 && (face_error>2e-10 || delta_error>2e-10))
                    std::cerr<<std::setprecision(17)<<"MP_STAGE_FACE_DIAGNOSTIC rank="<<rank<<" debug="<<debug
                        <<" high="<<face_error<<" delta="<<delta_error<<'\n';
                if(sizeof(Real)==8) require(face_error<=2e-10 && delta_error<=2e-10,"frozen physical face/delta mismatch");
            }
            // Both implementations receive identical complete-stage inputs.
            const auto dv=view(state,ASTR_DEVICE_MP_DELTA);
            size_t nd=1;for(int a=0;a<5;++a)nd*=dv.extent[a];
            std::vector<Real> input_delta(nd,std::numeric_limits<Real>::quiet_NaN());
            for(int a=0;a<3;++a)
            for(int k=(a==2 ? -1 : 0);k<=upper[2];++k)
            for(int j=(a==1 ? -1 : 0);j<=upper[1];++j)
            for(int i=(a==0 ? -1 : 0);i<=upper[0];++i) {
                const int p[3]={i,j,k};
                for(int c=0;c<5;++c) input_delta[offset(dv,p,c+1,a+1)]=delta[cube(p,minus1,faces)+na*(c+5*a)];
            }
            upload(state,ASTR_DEVICE_MP_DELTA,input_delta);upload(state,ASTR_DEVICE_RHS,rhs);
            int limited=0;check(astr_device_mp_positivity_limit(state,parameters[3],1./3.,2./3.,2./3.,3,first,last,&limited));
            limited_value=limited;
            if(sizeof(Real)==8) require(limited==h[38],"limited rank-node count mismatch");
            if(kind==1) require(limited>0,"stress did not activate");
            const auto actual_ratio=download(state,ASTR_DEVICE_MP_RATIO),actual_rhs=download(state,ASTR_DEVICE_RHS);
            ratio_error=std::max(ratio_error,relative(actual_ratio,expected_ratio,nf,1));
            rhs_error=std::max(rhs_error,relative(actual_rhs,expected_rhs,np,5));
            if(argc==5) {
                char leaf[128];std::snprintf(leaf,sizeof(leaf),"/cpp_stage.case%d.debug%d.rank%08d.bin",kind,debug,rank);
                std::ofstream diagnostic(std::string(argv[4])+leaf,std::ios::binary);
                diagnostic.write(reinterpret_cast<const char *>(h.data()),h.size()*sizeof(int32_t));
                diagnostic.write(reinterpret_cast<const char *>(actual_ratio.data()),nf*sizeof(Real));
                diagnostic.write(reinterpret_cast<const char *>(actual_rhs.data()),5*np*sizeof(Real));
                require(diagnostic.good(),"diagnostic write");
            }
            if(sizeof(Real)==8 && (ratio_error>2e-10 || rhs_error>2e-10))
                std::cerr<<std::setprecision(17)<<"MP_STAGE_RATIO_DIAGNOSTIC rank="<<rank<<" case="<<kind
                    <<" debug="<<debug<<" ratio="<<ratio_error<<" rhs="<<rhs_error<<'\n';
            if(sizeof(Real)==8) require(ratio_error<=2e-10 && rhs_error<=2e-10,"complete-stage ratio/RHS mismatch");
            const auto unchanged=download(state,ASTR_DEVICE_Q);
            require(std::memcmp(unchanged.data(),q.data(),q.size()*sizeof(Real))==0,"limiting changed the pre-update q buffer");
            check(astr_device_fluid_rk_update(state,parameters[3],1./3.,2./3.,2./3.));
            const auto updated=download(state,ASTR_DEVICE_Q);const auto qv=view(state,ASTR_DEVICE_Q);
            for(int k=0;k<=upper[2];++k) for(int j=0;j<=upper[1];++j) for(int i=0;i<=upper[0];++i) {
                const int p[3]={i,j,k};double local[5];
                for(int c=0;c<5;++c) {local[c]=updated[offset(qv,p,c+1)];require(std::isfinite(local[c]),"RK nonfinite");}
                const double internal=local[4]-.5*(local[1]*local[1]+local[2]*local[2]+local[3]*local[3])/local[0];
                require(local[0]>0 && std::isfinite(internal) && internal>0,"RK inadmissible");
            }
            check(astr_device_state_destroy(&state));
        }
        std::cout<<std::setprecision(17)<<"CPP_MP_STAGE_PASS rank="<<rank<<" case="<<kind<<" transport="<<transport
            <<" limited="<<limited_value<<" face="<<face_error<<" delta="<<delta_error<<" ratio="<<ratio_error<<" rhs="<<rhs_error<<'\n';
        std::cout<<"CPP_MP_STAGE_STORAGE real_bytes="<<sizeof(Real)<<" reference_limited="<<h[38]<<'\n';
        check(MPI_Finalize());
    } catch(const std::exception &error) {
        std::cerr<<"CPP_MP_STAGE_FAILURE rank="<<rank<<" "<<error.what()<<'\n';
        MPI_Abort(MPI_COMM_WORLD,1);
        return 1;
    }
}
