#include "device_state.h"
#include "chemistry_boundary.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include "chemistry_transport.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int im=4,jm=3,km=2;
void require(bool valid,const char *what) {if(!valid) {std::fprintf(stderr,"%s\n",what);std::exit(1);}}
void check(int status) {if(status) {std::fprintf(stderr,"device status=%d\n",status);std::exit(1);}}
size_t count(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t at(AstrDeviceView v,int i,int j,int k,int c=0)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+c*v.stride[3];}
std::vector<double> read(const char *path,int n)
{
    FILE *f=std::fopen(path,"rb");require(f,"reference open");int size;
    require(std::fread(&size,4,1,f)==1 && size==n,"reference rows");std::vector<int> status(n);std::vector<double> q(22*n);
    require(std::fread(status.data(),4,n,f)==n && std::fread(q.data(),8,q.size(),f)==q.size(),"reference data");
    for(int s:status)require(s==0,"reference rejected state");require(std::fclose(f)==0,"reference close");return q;
}
}
int main(int argc,char **argv)
{
    if(argc!=5 && argc!=6)return 2;const bool sample=argc==6 && std::strcmp(argv[5],"sample")==0;
    if(argc==6 && !sample)return 2;FILE *f=std::fopen(argv[1],"rb");require(f,"input open");int n;double floor[5];
    require(std::fread(&n,4,1,f)==1 && n>0 && std::fread(floor,8,5,f)==5,"input header");std::vector<double> input(96*n);
    require(std::fread(input.data(),8,input.size(),f)==input.size(),"input rows");require(std::fclose(f)==0,"input close");
    const auto cpu=read(argv[2],n),gpu=read(argv[3],n);int calls=0,failure_checks=0;double maximum_ratio=0.;
    for(int debug=0;debug<2;++debug) {
        void *handle=nullptr;check(astr_device_state_create(0,im,jm,km,5,debug,&handle));check(astr_device_air5_prepare(handle));
        check(astr_device_air5_geometry_prepare(handle));check(astr_device_air5_faces_prepare(handle));
        std::vector<double> spacing((im+1)*(km+1),.125),x_top(spacing.size(),0.);
        if(debug)check(astr_device_air5_symmetric_prepare(handle,floor,1));
        else check(astr_device_air5_characteristic_top_prepare(handle,spacing.data(),spacing.size()));
        AstrDeviceView high;check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,&high));void *address=high.data;
        if(debug)check(astr_device_air5_characteristic_top_prepare(handle,spacing.data(),spacing.size()));
        else check(astr_device_air5_symmetric_prepare(handle,floor,1));
        check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,&high));require(high.data==address,"shared high plane reallocated");
        const AstrAir5ChemistryOptions options{1,0,0,1,1,32,0,1e-9,1e-13,0.};check(astr_device_air5_chemistry_prepare(handle,&options));
        AstrDeviceView q,rhs,jac,carry;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));check(astr_device_state_view(handle,ASTR_DEVICE_RHS,&rhs));
        check(astr_device_state_view(handle,ASTR_DEVICE_JACOBIAN,&jac));check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&carry));
        std::vector<double> before(count(q),-777.),authority(count(q)),low(count(carry),1e-30),after_low(count(carry)),
            residual(count(rhs),-91.),after_rhs(count(rhs)),plane(count(high),-77.),after_high(count(high)),geometry(count(jac));
        auto prepare=[&](const double *x) {
            std::fill(before.begin(),before.end(),-777.);
            for(int k=0;k<=km;++k)for(int j=0;j<=jm;++j)for(int i=0;i<=im;++i)for(int c=0;c<11;++c)
                before[at(q,i,j,k,c)]=j==jm-1?x[55+c]:j==jm-2?x[66+c]:x[c];
            std::fill(geometry.begin(),geometry.end(),x[94]);std::fill(spacing.begin(),spacing.end(),x[88]);
            double incident[22];for(int c=0;c<22;++c)incident[c]=x[11+c%11];
            check(astr_device_air5_top_boundary_prepare(handle,x+11,incident,1.,.5,1,x_top.data(),x_top.size()));
            check(astr_device_air5_characteristic_top_prepare(handle,spacing.data(),spacing.size()));
        };
        auto upload=[&]() {
            check(astr_device_state_upload(handle,ASTR_DEVICE_Q,before.data(),before.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_RHS,residual.data(),residual.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,plane.data(),plane.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_JACOBIAN,geometry.data(),geometry.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CARRY,low.data(),low.size()*8));
        };
        auto observe=[&]() {
            check(astr_device_state_download(handle,ASTR_DEVICE_Q,authority.data(),authority.size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,after_low.data(),after_low.size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_RHS,after_rhs.data(),after_rhs.size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,after_high.data(),after_high.size()*8));
            require(std::memcmp(before.data(),authority.data(),before.size()*8)==0,"top RHS changed q");
            require(std::memcmp(low.data(),after_low.data(),low.size()*8)==0,"top RHS changed carry");
        };
        for(int row=0;row<n;++row) {
            if(sample && row%64!=0)continue;const double *x=input.data()+96*row;prepare(x);
            for(int first=0;first<2;++first) {
                upload();check(astr_device_air5_characteristic_top_rhs(handle,first,static_cast<int>(x[93]),x[91],x[92]));
                int physics;check(astr_device_air5_check(handle,&physics));require(physics==0,"top RHS physics status");observe();
                for(int k=0;k<=km;++k)for(int j=0;j<=jm;++j)for(int i=0;i<=im;++i)for(int c=0;c<11;++c) {
                    const size_t index=at(rhs,i,j,k,c);
                    if(j==jm && i>=first)for(const auto *ref:{&cpu,&gpu}) {
                        for(int group=0;group<2;++group) {
                            const double actual=group==0?after_high[at(high,i,0,k,c)]:after_rhs[index];double scale=1.;
                            for(int d=0;d<11;++d)scale=fmax(scale,fabs((*ref)[22*row+group*11+d]));
                            const double value=(*ref)[22*row+group*11+c],ratio=fabs(actual-value)/(1e-13*scale+5e-12*fabs(value));
                            if(!std::isfinite(actual) || !(ratio<=1.)) {std::fprintf(stderr,"row%d first%d c%d group%d ratio%.17g\n",row,first,c,group,ratio);return 1;}
                            maximum_ratio=fmax(maximum_ratio,ratio);
                        }
                    } else require(std::memcmp(&after_rhs[index],&residual[index],8)==0,"non-top/owned inlet RHS changed");
                    if(j==0 && i<first) {
                        const size_t a=at(high,i,0,k,c);require(std::memcmp(&after_high[a],&plane[a],8)==0,"inlet-owned high plane changed");
                    }
                }
                ++calls;
            }
        }
        const double *x=input.data();
        for(int kind=0;kind<4;++kind) {
            prepare(x);
            if(kind==0)before[at(q,2,jm,1,0)]=-1.;
            if(kind==1) {spacing[2+(im+1)]=0.;check(astr_device_air5_characteristic_top_prepare(handle,spacing.data(),spacing.size()));}
            if(kind==2)geometry[at(jac,2,jm,1)]=0.;
            if(kind==3)before[at(q,0,jm,1,0)]=-1.;
            upload();check(astr_device_air5_characteristic_top_rhs(handle,1,static_cast<int>(x[93]),x[91],x[92]));
            int status;check(astr_device_air5_check(handle,&status));require(status==(kind==0?2:kind==3?0:7),"failed top status/ownership");observe();
            const int failed_i=kind==3?0:2;for(int c=0;c<11;++c) {
                const size_t r=at(rhs,failed_i,jm,1,c),h=at(high,failed_i,0,1,c);
                require(std::memcmp(&after_rhs[r],&residual[r],8)==0 && std::memcmp(&after_high[h],&plane[h],8)==0,"failed/skipped point committed RHS");
            }
            ++failure_checks;
        }
        check(astr_device_state_destroy(&handle));
    }
    f=std::fopen(argv[4],"w");require(f,"report open");
    std::fprintf(f,"{\"calls\":%d,\"failure_checks\":%d,\"maximum_original_component_budget_ratio\":%.17g,\"q_carry_untouched\":true,\"shared_high_plane_stable\":true,\"sample\":%s}\n",calls,failure_checks,maximum_ratio,sample?"true":"false");
    require(std::fclose(f)==0,"report close");std::printf("AIR5_CHARACTERISTIC_FIELD_PASS calls=%d failures=%d ratio=%.12g\n",calls,failure_checks,maximum_ratio);return 0;
}
