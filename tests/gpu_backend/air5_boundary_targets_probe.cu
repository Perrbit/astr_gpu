#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include "chemistry_boundary.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int upper[3]={8,7,6},halo=5;
int rank_id=0;
void require(bool condition,const char *label)
{
    if(!condition) {std::fprintf(stderr,"rank%d %s\n",rank_id,label);MPI_Abort(MPI_COMM_WORLD,1);std::exit(1);}
}
void check(int code) {require(code==0,"device/MPI operation failed");}
size_t count(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t offset(AstrDeviceView v,int i,int j,int k,int c)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+c*v.stride[3];}
struct Reference {int n;std::vector<int> status;std::vector<double> q;};
Reference read_reference(const char *path,int n)
{
    FILE *f=std::fopen(path,"rb");require(f!=nullptr,"missing frozen reference");Reference r;
    require(std::fread(&r.n,4,1,f)==1 && r.n==n,"reference count");r.status.resize(n);r.q.resize(11*n);
    require(std::fread(r.status.data(),4,n,f)==n && std::fread(r.q.data(),8,r.q.size(),f)==r.q.size(),"reference data");
    require(std::fclose(f)==0,"reference close");return r;
}
}
int main(int argc,char **argv)
{
    if(argc!=9)return 2;
    const char *local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)return 2;
    const int devices=std::atoi(argv[8]);int available;
    if(cudaGetDeviceCount(&available)!=cudaSuccess || devices<1 || devices>available)return 2;
    // Four local test ranks may explicitly share the two available GPUs.
    const int device=std::atoi(local)%devices;check(astr_device_bind(device));
    check(MPI_Init(&argc,&argv));check(MPI_Comm_rank(MPI_COMM_WORLD,&rank_id));int ranks;check(MPI_Comm_size(MPI_COMM_WORLD,&ranks));
    int dims[3]={std::atoi(argv[5]),std::atoi(argv[6]),std::atoi(argv[7])},periodic[3]={0,0,0},coord[3];
    require(dims[0]>0 && dims[1]>0 && dims[2]>0 && dims[0]*dims[1]*dims[2]==ranks,"topology");
    MPI_Comm cart;check(MPI_Cart_create(MPI_COMM_WORLD,3,dims,periodic,0,&cart));check(MPI_Cart_coords(cart,rank_id,3,coord));
    FILE *f=std::fopen(argv[1],"rb");require(f!=nullptr,"input open");int n;
    require(std::fread(&n,4,1,f)==1 && n>0,"input count");std::vector<double> input(32*n);
    require(std::fread(input.data(),8,input.size(),f)==input.size(),"input data");require(std::fclose(f)==0,"input close");
    const Reference cpu=read_reference(argv[2],n),gpu=read_reference(argv[3],n);
    int calls=0,failures=0;double maximum_ratio=0.;
    for(int preserve=0;preserve<2;++preserve)for(int debug=0;debug<2;++debug) {
        int success=-1,failure=-1;
        for(int row=0;row<n;++row) {
            const double *x=input.data()+32*row;
            if(x[31]!=0 || static_cast<int>(x[23])!=preserve || x[22]<=0)continue;
            if(success<0 && cpu.status[row]==0 && gpu.status[row]==0 && std::memcmp(x,cpu.q.data()+11*row,88)!=0)success=row;
            if(failure<0 && cpu.status[row]!=0 && cpu.status[row]==gpu.status[row])failure=row;
        }
        require(success>=0 && failure>=0,"missing pressure success/failure cases");const double *base=input.data()+32*success;
        void *handle=nullptr;check(astr_device_state_create(device,upper[0],upper[1],upper[2],halo,debug,&handle));
        check(astr_device_air5_prepare(handle));const AstrAir5ChemistryOptions options{1,0,0,1,1,32,0,1e-9,1e-13,0.};
        check(astr_device_air5_chemistry_prepare(handle,&options));AstrDeviceView q,carry;
        check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&carry));
        require(q.element_bytes==8 && carry.element_bytes==8,"AIR5 field precision");
        std::vector<double> initial(count(q)),actual(count(q)),low(count(carry),1e-30),after_low(count(carry));
        for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)
            for(int c=0;c<11;++c) {
                const size_t at=offset(q,i,j,k,c);initial[at]=-777.-static_cast<double>(at)/1024.;
                if(i>=0 && i<=upper[0] && j>=0 && j<=upper[1] && k>=0 && k<=upper[2])
                    initial[at]=base[c]*(1.+(coord[0]*upper[0]+i+2*(coord[1]*upper[1]+j)+3*(coord[2]*upper[2]+k))/1024.);
            }
        for(int update=0;update<2;++update) {
            double left[11],right[11];std::vector<double> profile((upper[1]+1)*11);
            for(int c=0;c<11;++c) {
                left[c]=base[c]*(update?1.5:.5);right[c]=base[c]*(update?.75:1.25);
                for(int j=0;j<=upper[1];++j)profile[j+(upper[1]+1)*c]=base[c]*(1.+(coord[1]*upper[1]+j+update*16)/512.);
            }
            const std::vector<double> saved_left(left,left+11),saved_right(right,right+11),saved_profile=profile;
            check(astr_device_air5_fixed_boundary_prepare(handle,left,right));
            check(astr_device_air5_profile_boundary_prepare(handle,profile.data(),profile.size()));
            // The prepare call owns a copy, not the lifetime of these host arrays.
            for(int c=0;c<11;++c)left[c]=right[c]=NAN;for(double &v:profile)v=NAN;
            for(int mode=0;mode<6;++mode) {
                const bool left_owner=coord[0]==0 && mode!=4,right_owner=coord[0]==dims[0]-1 && mode!=4 && mode!=5;
                const int row=mode==3?failure:success;const double *x=input.data()+32*row;
                std::vector<double> before=initial,expected=initial;
                if(mode==2 || mode==3)for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)
                    for(int c=0;c<11;++c)before[offset(q,upper[0]-1,j,k,c)]=expected[offset(q,upper[0]-1,j,k,c)]=x[c];
                check(astr_device_state_upload(handle,ASTR_DEVICE_Q,before.data(),before.size()*8));
                check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CARRY,low.data(),low.size()*8));
                if(mode==5) {
                    // Clear only the previous test status; profile copying itself does not reset it.
                    check(astr_device_air5_fixed_boundary_apply(handle,0,0,0,0.,preserve));
                    if(left_owner)check(astr_device_air5_profile_inflow_lower_x(handle));
                } else check(astr_device_air5_fixed_boundary_apply(handle,left_owner,right_owner,mode!=0,mode>=2?x[22]:0.,preserve));
                int physics;check(astr_device_air5_check(handle,&physics));
                const int expected_status=mode==3 && right_owner?cpu.status[row]:0;
                require(physics==expected_status,"pressure failure status");if(physics)++failures;
                for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int c=0;c<11;++c) {
                    if(left_owner)for(int i=-halo;i<=0;++i)
                        expected[offset(q,i,j,k,c)]=mode==5?saved_profile[j+(upper[1]+1)*c]:saved_left[c];
                    if(right_owner && expected_status==0)for(int i=upper[0];i<=upper[0]+halo;++i)
                        expected[offset(q,i,j,k,c)]=mode==0?saved_right[c]:mode==1?before[offset(q,upper[0]-1,j,k,c)]:cpu.q[11*row+c];
                }
                check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual.data(),actual.size()*8));
                check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,after_low.data(),after_low.size()*8));
                require(std::memcmp(low.data(),after_low.data(),low.size()*8)==0,"boundary changed carry");
                for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)
                    for(int c=0;c<11;++c) {
                        const size_t at=offset(q,i,j,k,c);
                        if(mode==2 && right_owner && i>=upper[0] && j>=0 && j<=upper[1] && k>=0 && k<=upper[2]) {
                            for(const Reference *r:{&cpu,&gpu}) {
                                double scale=1.;for(int d=0;d<11;++d)scale=fmax(scale,fabs(r->q[11*row+d]));
                                const double ref=r->q[11*row+c],ratio=fabs(actual[at]-ref)/(1e-13*scale+5e-12*fabs(ref));
                                require(std::isfinite(actual[at]) && ratio<=1.,"pressure state budget");maximum_ratio=fmax(maximum_ratio,ratio);
                            }
                        } else require(std::memcmp(&actual[at],&expected[at],8)==0,"target/untouched word mismatch");
                    }
                ++calls;
            }
        }
        check(astr_device_state_destroy(&handle));
    }
    int total_calls,total_failures;double global_ratio;
    check(MPI_Reduce(&calls,&total_calls,1,MPI_INT,MPI_SUM,0,cart));check(MPI_Reduce(&failures,&total_failures,1,MPI_INT,MPI_SUM,0,cart));
    check(MPI_Reduce(&maximum_ratio,&global_ratio,1,MPI_DOUBLE,MPI_MAX,0,cart));
    if(rank_id==0) {
        FILE *out=std::fopen(argv[4],"w");require(out!=nullptr,"report open");
        std::fprintf(out,"{\"ranks\":%d,\"calls\":%d,\"expected_pressure_failures\":%d,\"maximum_original_component_budget_ratio\":%.17g,\"raw_and_untouched_words_exact\":true,\"carry_unchanged\":true}\n",ranks,total_calls,total_failures,global_ratio);
        require(std::fclose(out)==0,"report close");
        std::printf("AIR5_BOUNDARY_TARGETS_PASS ranks=%d calls=%d expected_failures=%d ratio=%.12g\n",ranks,total_calls,total_failures,global_ratio);
    }
    check(MPI_Comm_free(&cart));check(MPI_Finalize());return 0;
}
