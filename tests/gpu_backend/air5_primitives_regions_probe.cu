#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace {
constexpr int upper[3]={8,7,6},halo=5;
const int fields[6]={ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE,ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES};
const int first[15]={0,0,0, -5,0,0, 0,0,7, -5,-5,0, 1,2,3};
const int last[15]={8,7,6, -1,7,6, 8,7,11, -1,-1,6, 1,2,3};
void require(bool valid,const char *label) {if(!valid) {std::fprintf(stderr,"%s\n",label);std::exit(1);}}
void check(int status) {if(status) {std::fprintf(stderr,"device status%d\n",status);std::exit(1);}}
size_t size(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t at(AstrDeviceView v,int i,int j,int k,int c=0)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+c*v.stride[3];}
double marker(size_t field,size_t index)
{const uint64_t bits=UINT64_C(0x7ff8000000000000)|(field<<40)|index;double value;std::memcpy(&value,&bits,8);return value;}
bool same(double a,double b) {return std::memcmp(&a,&b,8)==0;}
bool covered(int i,int j,int k)
{
    const int p[3]={i,j,k};int outside=0;for(int a=0;a<3;++a)outside+=p[a]<0 || p[a]>upper[a];if(outside>1)return false;
    for(int r=0;r<5;++r) {bool active=true;for(int a=0;a<3;++a)active=active && p[a]>=first[3*r+a] && p[a]<=last[3*r+a];if(active)return true;}
    return false;
}
}
int main(int argc,char **argv)
{
    if(argc!=2)return 2;FILE *f=std::fopen(argv[1],"rb");require(f,"input open");double base[11];
    require(std::fread(base,8,11,f)==11,"base data");require(std::fclose(f)==0,"input close");int failure_checks=0,shell_checks=0;
    for(int debug=0;debug<2;++debug)for(int preserve=0;preserve<2;++preserve) {
        void *handle=nullptr;check(astr_device_state_create(0,upper[0],upper[1],upper[2],halo,debug,&handle));check(astr_device_air5_prepare(handle));
        const AstrAir5ChemistryOptions options{1,1,0,1,preserve,32,0,1e-9,1e-13,0.};check(astr_device_air5_chemistry_prepare(handle,&options));
        AstrDeviceView q,carry,history;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));
        check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&carry));check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_HISTORY,&history));
        std::vector<double> initial(size(q)),input(size(q)),actual(size(q)),low(size(carry),1e-30),low_after(size(carry)),steps(size(history),1e-7),steps_after(size(history));
        low[0]=-0.;steps[0]=-0.;std::vector<AstrDeviceView> views(6);std::vector<std::vector<double>> poison(6),batch(6),observed(6);
        for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)for(int c=0;c<11;++c) {
            const int outside=(i<0 || i>upper[0])+(j<0 || j>upper[1])+(k<0 || k>upper[2]);
            initial[at(q,i,j,k,c)]=outside>1?marker(7,at(q,i,j,k,c)):base[c]*(1.+(i+2*j+3*k)/1024.);
        }
        for(int field=0;field<6;++field) {
            check(astr_device_state_view(handle,fields[field],&views[field]));poison[field].resize(size(views[field]));
            for(size_t i=0;i<poison[field].size();++i)poison[field][i]=marker(field+1,i);
            batch[field].resize(poison[field].size());observed[field].resize(poison[field].size());
        }
        auto initialize=[&]() {
            check(astr_device_state_upload(handle,ASTR_DEVICE_Q,input.data(),input.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CARRY,low.data(),low.size()*8));
            check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_HISTORY,steps.data(),steps.size()*8));
            for(int field=0;field<6;++field)check(astr_device_state_upload(handle,fields[field],poison[field].data(),poison[field].size()*8));
        };
        auto snapshots=[&](std::vector<std::vector<double>> &cache) {
            for(int field=0;field<6;++field)check(astr_device_state_download(handle,fields[field],cache[field].data(),cache[field].size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual.data(),actual.size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,low_after.data(),low_after.size()*8));
            check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_HISTORY,steps_after.data(),steps_after.size()*8));
            require(std::memcmp(input.data(),actual.data(),input.size()*8)==0,"primitive batch changed q words");
            require(std::memcmp(low.data(),low_after.data(),low.size()*8)==0,"primitive batch changed carry words");
            require(std::memcmp(steps.data(),steps_after.data(),steps.size()*8)==0,"primitive batch changed history words");
        };
        input=initial;initialize();check(astr_device_air5_primitives_regions(handle,5,first,last,preserve));int status;
        check(astr_device_air5_check(handle,&status));require(status==0,"valid primitive batch status");snapshots(batch);
        for(int field=0;field<6;++field)for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)
            for(int c=0;c<views[field].extent[3];++c) {
                const size_t index=at(views[field],i,j,k,c);
                if(covered(i,j,k))require(std::isfinite(batch[field][index]),"valid covered cache was not produced");
                else require(same(batch[field][index],poison[field][index]),"unselected/corner cache changed");
            }
        initialize();for(int r=0;r<5;++r) {check(astr_device_air5_primitives(handle,first+3*r,last+3*r,preserve));check(astr_device_air5_check(handle,&status));require(status==0,"single-region reference status");}
        snapshots(observed);
        for(int field=0;field<6;++field)require(std::memcmp(batch[field].data(),observed[field].data(),batch[field].size()*8)==0,"batch versus sequential cache words");
        const int shell_low[][3]={{1,1,1},{2,1,3},{2,4,5}},shell_high[][3]={{7,6,5},{6,4,4},{2,4,5}};
        for(int scenario=0;scenario<3;++scenario)for(int invalid_shell=0;invalid_shell<2;++invalid_shell) {
            const int *lo=shell_low[scenario],*hi=shell_high[scenario];
            input=initial;
            input[at(q,lo[0],lo[1],lo[2])]=-1.;
            if(invalid_shell)input[at(q,upper[0],0,0)]=-1.;
            initialize();check(astr_device_air5_primitives_shell(handle,lo,hi,preserve));
            check(astr_device_air5_check(handle,&status));require(status==(invalid_shell?2:0),"shell conversion status");
            snapshots(observed);
            for(int field=0;field<6;++field)for(int k=-halo;k<=upper[2]+halo;++k)
            for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)
            for(int c=0;c<views[field].extent[3];++c) {
                const size_t index=at(views[field],i,j,k,c);
                const bool physical=i>=0 && i<=upper[0] && j>=0 && j<=upper[1] && k>=0 && k<=upper[2];
                const bool recovered=i>=lo[0] && i<=hi[0] && j>=lo[1] && j<=hi[1] && k>=lo[2] && k<=hi[2];
                const bool failed=invalid_shell && i==upper[0] && j==0 && k==0;
                const double expected=physical && !recovered && !failed?batch[field][index]:poison[field][index];
                require(same(expected,observed[field][index]),"primitive shell coverage or failure write");
            }
            ++shell_checks;
        }
        for(int scenario=0;scenario<4;++scenario) {
            input=initial;
            if(scenario==0)input[at(q,0,0,0)]=-1.;
            if(scenario==1)input[at(q,2,2,upper[2]+1)]=-1.;
            if(scenario==2) {input[at(q,0,0,0,4)]=-1e20;input[at(q,-1,1,1)]=-1.;}
            if(scenario==3) {input[at(q,0,0,0,1)]=marker(9,1);input[at(q,-1,1,1)]=-1.;}
            initialize();check(astr_device_air5_primitives_regions(handle,5,first,last,preserve));
            check(astr_device_air5_check(handle,&status));require(status==(scenario<2?2:scenario==2?4:7),"earlier/multiple region failure lost");snapshots(observed);
            for(int field=0;field<6;++field)for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)
                for(int c=0;c<views[field].extent[3];++c) {
                    const bool failed=(scenario==0 && i==0 && j==0 && k==0)||(scenario==1 && i==2 && j==2 && k==upper[2]+1)||
                        (scenario>=2 && ((i==0 && j==0 && k==0)||(i==-1 && j==1 && k==1)));
                    const size_t index=at(views[field],i,j,k,c);
                    if(!covered(i,j,k)||failed)require(same(observed[field][index],poison[field][index]),"failed/non-target/corner cache overwritten");
                    else require(same(observed[field][index],batch[field][index]),"later valid region did not refresh same q");
                }
            ++failure_checks;
        }
        // A separate partial observation phase clears old status, but does not
        // repair or advance the invalid q left by the prior fixture.
        check(astr_device_air5_primitives_regions(handle,1,first+12,last+12,preserve));
        check(astr_device_air5_check(handle,&status));require(status==0,"new phase did not clear prior failure");snapshots(observed);
        check(astr_device_state_destroy(&handle));
    }
    std::printf("AIR5_PRIMITIVES_REGIONS_PASS contexts=4 regions=5 failure_checks=%d sequential_words=exact q_carry_history=unchanged\n",failure_checks);
    std::printf("AIR5_PRIMITIVE_SHELL_PASS cases=%d asymmetric_and_singleton=checked\n",shell_checks);return 0;
}
