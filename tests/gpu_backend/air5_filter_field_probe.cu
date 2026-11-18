#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include "chemistry_coupling.h"
#include "chemistry_boundary.h"
#include "halo_transport.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int prepare_rk_probe(void *,const int[3],const int[2],int,int,int,int,int,int,const double[5],int,int,int *);

void check(int code) {if(code){std::fprintf(stderr,"operation %d\n",code);MPI_Abort(MPI_COMM_WORLD,code);std::exit(1);}}
size_t count(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t offset(AstrDeviceView v,int i,int j,int k,int c)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+c*v.stride[3];}
int main(int argc,char **argv)
{
    if(argc!=11 && argc!=13 && argc!=14)return 2;
    const char *local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)return 2;
    const int devices=std::atoi(argv[10]);if(devices<1)return 2;const int device=std::atoi(local)%devices;
    if(astr_device_bind(device))return 3;
    check(MPI_Init(&argc,&argv));int rank,nranks;check(MPI_Comm_rank(MPI_COMM_WORLD,&rank));check(MPI_Comm_size(MPI_COMM_WORLD,&nranks));
    int sizes[3]={std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5])};if(sizes[0]*sizes[1]*sizes[2]!=nranks)check(1);
    const int scalar=std::atoi(argv[6]),compensated=std::atoi(argv[7]),mode=std::atoi(argv[8]),debug=std::atoi(argv[9]);
    // Stage modes: 1 legacy, 2 conservative, 3 unfiltered reuse, 4 later unfiltered stage.
    const int stage=argc==14 ? std::atoi(argv[13]) : 0;
    const bool legacy=argc==13 || stage==1;
    const int wall_axis=argc>=13 ? std::atoi(argv[11]) : -1,family=argc>=13 ? std::atoi(argv[12]) : 0;
    FILE *file=std::fopen(argv[1],"rb");if(!file)check(2);int global[3],periodic[3];double tr[5],v[5];
    if(std::fread(global,4,3,file)!=3 || std::fread(periodic,4,3,file)!=3 || std::fread(tr,8,5,file)!=5 || std::fread(v,8,5,file)!=5)check(2);
    std::vector<double> table(22*static_cast<size_t>(global[0]+1)*(global[1]+1)*(global[2]+1));
    if(std::fread(table.data(),8,table.size(),file)!=table.size())check(2);std::fclose(file);
    MPI_Comm cart;check(MPI_Cart_create(MPI_COMM_WORLD,3,sizes,periodic,0,&cart));
    int coordinates[3],low[3],high[3],upper[3],first[3];check(MPI_Cart_coords(cart,rank,3,coordinates));
    for(int a=0;a<3;++a) {check(MPI_Cart_shift(cart,a,1,&low[a],&high[a]));if(global[a]%sizes[a])check(1);upper[a]=global[a]/sizes[a];first[a]=upper[a]*coordinates[a];}
    void *handle=nullptr;check(astr_device_state_create(device,upper[0],upper[1],upper[2],5,debug,&handle));
    check(astr_device_air5_prepare(handle));
    if(compensated) {const AstrAir5ChemistryOptions options{1,0,0,1,1,32,0,1e-9,1e-13,0.};check(astr_device_air5_chemistry_prepare(handle,&options));}
    check(astr_device_air5_geometry_prepare(handle));check(astr_device_air5_rk_prepare(handle));
    check(astr_device_transport_prepare(handle,static_cast<int>(MPI_Comm_c2f(cart)),low,high,sizes,11,mode,1,periodic));
    if(legacy)check(astr_device_air5_legacy_linear_prepare(handle,scalar));
    else if(stage<3)check(astr_device_air5_filter_prepare(handle,tr,v,global,first,periodic,scalar));
    if(stage && !periodic[0])
        check(astr_device_air5_fixed_boundary_prepare(handle,table.data(),table.data()+22*global[0]));
    std::vector<double> original_q;
    for(int f=0;f<(compensated?2:1);++f) {
        const int field=f==0?ASTR_DEVICE_Q:ASTR_DEVICE_AIR5_CARRY;AstrDeviceView view;check(astr_device_state_view(handle,field,&view));
        std::vector<double> data(count(view),-777.);
        for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int i=0;i<=upper[0];++i) {
            const size_t at=22*(first[0]+i+static_cast<size_t>(global[0]+1)*(first[1]+j+static_cast<size_t>(global[1]+1)*(first[2]+k)));
            for(int c=0;c<11;++c)data[offset(view,i,j,k,c)]=table[at+11*f+c];
        }
        check(astr_device_state_upload(handle,field,data.data(),data.size()*8));if(f==0)original_q=data;
    }
    if(stage<3) {
        AstrDeviceView work;check(astr_device_state_view(handle,ASTR_DEVICE_FILTER_WORK,&work));
        if(work.extent[3]!=(scalar?1:11))check(5);
    }
    const int caches[]={ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE,
                        ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES};
    if(stage)for(int cache:caches) {
        AstrDeviceView view;check(astr_device_state_view(handle,cache,&view));
        std::vector<double> initial(count(view),-999.);
        check(astr_device_state_upload(handle,cache,initial.data(),initial.size()*8));
    }
    const int edges[]={low[0]==MPI_PROC_NULL,high[0]==MPI_PROC_NULL};
    int physics=0;for(int step=0;step<(stage<3 ? 2 : 1);++step) {
        if(stage) {
            int calls=-1;
            physics=prepare_rk_probe(handle,upper,edges,MPI_Comm_c2f(cart),compensated,stage==4 ? 2 : step+1,stage<3,
                                     1,stage==2,v,wall_axis,family,&calls);
            if(calls!=(stage==3 || physics ? 0 : 1))check(8);
        }
        else if(legacy)check(astr_device_air5_legacy_filter_apply(handle,v,wall_axis,family,&physics));
        else check(astr_device_air5_filter_apply(handle,1,&physics));
        if(physics)break;
    }
    if(legacy && nranks>1) {
        double metrics[6];int local_status;
        check(astr_device_transport_metrics(handle,metrics,0));
        check(astr_device_air5_check(handle,&local_status));
        if(metrics[4]<=0. || (mode==0 ? metrics[5]<=0. : metrics[5]!=0.))check(7);
        std::printf("rank=%d mpi_bytes=%.0f host_staging_bytes=%.0f local_status=%d global_status=%d\n",
                    rank,metrics[4],metrics[5],local_status,physics);
    }
    char output[2048];std::snprintf(output,sizeof(output),"%s.rank%d.bin",argv[2],rank);file=std::fopen(output,"wb");if(!file)check(2);
    const int header[4]={upper[0],upper[1],upper[2],physics};if(std::fwrite(header,4,4,file)!=4)check(2);
    for(int f=0;f<2;++f) {
        std::vector<double> physical(11*static_cast<size_t>(upper[0]+1)*(upper[1]+1)*(upper[2]+1),0.);
        if(f==0 || compensated) {
            const int field=f==0?ASTR_DEVICE_Q:ASTR_DEVICE_AIR5_CARRY;AstrDeviceView view;check(astr_device_state_view(handle,field,&view));
            std::vector<double> data(count(view));check(astr_device_state_download(handle,field,data.data(),data.size()*8));size_t at=0;
            for(int c=0;c<11;++c)for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int i=0;i<=upper[0];++i)physical[at++]=data[offset(view,i,j,k,c)];
            if(!legacy && !stage && f==0)for(int c=0;c<11;++c)for(int k=-5;k<=upper[2]+5;++k)for(int j=-5;j<=upper[1]+5;++j)for(int i=-5;i<=upper[0]+5;++i)
                if(i<0 || i>upper[0] || j<0 || j>upper[1] || k<0 || k>upper[2])
                    if(data[offset(view,i,j,k,c)]!=original_q[offset(view,i,j,k,c)])check(6);
        }
        if(std::fwrite(physical.data(),8,physical.size(),file)!=physical.size())check(2);
    }
    if(legacy || stage) {
        AstrDeviceView view;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&view));
        std::vector<double> padded(count(view));check(astr_device_state_download(handle,ASTR_DEVICE_Q,padded.data(),padded.size()*8));
        if(std::fwrite(padded.data(),8,padded.size(),file)!=padded.size())check(2);
    }
    if(stage)for(int cache:caches) {
        AstrDeviceView view;check(astr_device_state_view(handle,cache,&view));
        std::vector<double> field(count(view));check(astr_device_state_download(handle,cache,field.data(),field.size()*8));
        if(std::fwrite(field.data(),8,field.size(),file)!=field.size())check(2);
    }
    if(std::fclose(file))check(2);
    check(astr_device_state_destroy(&handle));check(MPI_Comm_free(&cart));check(MPI_Finalize());return physics?4:0;
}
