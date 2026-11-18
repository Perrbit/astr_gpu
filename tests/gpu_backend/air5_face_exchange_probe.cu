#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include "halo_transport.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
int wire_mode=-1,wire_calls=0,wire_bad=0,rank_id=0;
void check(int code)
{if(code) {std::fprintf(stderr,"rank%d operation status%d\n",rank_id,code);MPI_Abort(MPI_COMM_WORLD,code);std::exit(1);}}
void require(bool condition,const char *label)
{if(!condition) {std::fprintf(stderr,"rank%d %s\n",rank_id,label);MPI_Abort(MPI_COMM_WORLD,1);std::exit(1);}}
size_t count(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t offset(AstrDeviceView v,int i,int j,int k,int c)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3];}
double value(int rank,int round,int axis,int i,int j,int k,int c)
{return (rank+1)*100000000.+round*10000000.+axis*1000000.+c*10000.+i*100.+j*10.+k;}
void pointer_kind(const void *p)
{
    cudaPointerAttributes attributes{};
    const int status=static_cast<int>(cudaPointerGetAttributes(&attributes,p));
    if(status || attributes.type!=(wire_mode==ASTR_HALO_DEVICE?cudaMemoryTypeDevice:cudaMemoryTypeHost))++wire_bad;
}
}

extern "C" int MPI_Sendrecv(const void *send,int ns,MPI_Datatype ts,int peer_s,int tag_s,
    void *recv,int nr,MPI_Datatype tr,int peer_r,int tag_r,MPI_Comm comm,MPI_Status *status)
{
    if(wire_mode>=0) {
        ++wire_calls;
        if(ns!=nr || ns<=0 || ts!=MPI_DOUBLE || tr!=MPI_DOUBLE || tag_s<6 || tag_s>11 || tag_r!=tag_s)++wire_bad;
        if(peer_s!=MPI_PROC_NULL)pointer_kind(send);
        if(peer_r!=MPI_PROC_NULL)pointer_kind(recv);
    }
    return PMPI_Sendrecv(send,ns,ts,peer_s,tag_s,recv,nr,tr,peer_r,tag_r,comm,status);
}

int main(int argc,char **argv)
{
    if(argc!=4 && argc!=5)return 2;
    int first_mode=0,last_mode=2;
    if(argc==5) {
        if(std::strcmp(argv[4],"pinned")==0)first_mode=0;
        else if(std::strcmp(argv[4],"device")==0)first_mode=1;
        else return 2;
        last_mode=first_mode+1;
    }
    const char *local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)return 2;
    const int device=std::atoi(local);
    if(astr_device_bind(device))return 3;
    check(MPI_Init(&argc,&argv));check(MPI_Comm_rank(MPI_COMM_WORLD,&rank_id));
    int nranks;check(MPI_Comm_size(MPI_COMM_WORLD,&nranks));
    int sizes[3]={std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3])};
    require(sizes[0]>0 && sizes[1]>0 && sizes[2]>0 && sizes[0]*sizes[1]*sizes[2]==nranks,"topology size mismatch");
    const int upper[3]={8,7,6},halo=5;const double floors[5]={};
    for(int layout=0;layout<2;++layout) {
        int periodic[3]={layout==0,layout==0,layout==0};MPI_Comm cart;
        check(MPI_Cart_create(MPI_COMM_WORLD,3,sizes,periodic,0,&cart));
        int low[3],high[3];for(int a=0;a<3;++a)check(MPI_Cart_shift(cart,a,1,&low[a],&high[a]));
        for(int mode=first_mode;mode<last_mode;++mode)for(int debug=0;debug<2;++debug) {
            wire_mode=mode;wire_bad=wire_calls=0;
            void *handle=nullptr;check(astr_device_state_create(device,upper[0],upper[1],upper[2],halo,debug,&handle));
            check(astr_device_air5_prepare(handle));check(astr_device_air5_geometry_prepare(handle));
            check(astr_device_air5_faces_prepare(handle));check(astr_device_air5_symmetric_prepare(handle,floors,0));
            check(astr_device_transport_prepare(handle,static_cast<int>(MPI_Comm_c2f(cart)),low,high,sizes,11,mode,1,periodic));
            AstrDeviceView faces,q;check(astr_device_state_view(handle,ASTR_DEVICE_INTERFACE_FLUX,&faces));
            check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));
            std::vector<double> authority(count(q),rank_id+.125);
            check(astr_device_state_upload(handle,ASTR_DEVICE_Q,authority.data(),authority.size()*8));
            AstrDeviceView base;check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,&base));
            std::vector<double> budget(count(base)),expected_budget,actual_budget(count(base));
            for(int c=1;c<=11;++c)for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i)
                budget[offset(base,i,j,k,c)]=value(rank_id,0,0,i,j,k,c);
            expected_budget=budget;check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,budget.data(),budget.size()*8));
            for(int a=0;a<3;++a) {
                for(int c=1;c<=11;++c)for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int i=0;i<=upper[0];++i) {
                    const int origin[3]={i,j,k};if(origin[a]!=0)continue;
                    for(int h=1;h<=halo;++h) {
                        int dst[3]={i,j,k},src[3]={i,j,k};
                        if(low[a]!=MPI_PROC_NULL) {
                            dst[a]=-h;src[a]=upper[a]-h;
                            expected_budget[offset(base,dst[0],dst[1],dst[2],c)]=value(low[a],0,0,src[0],src[1],src[2],c);
                        }
                        if(high[a]!=MPI_PROC_NULL) {
                            dst[a]=upper[a]+h;src[a]=h;
                            expected_budget[offset(base,dst[0],dst[1],dst[2],c)]=value(high[a],0,0,src[0],src[1],src[2],c);
                        }
                    }
                }
                check(astr_device_exchange_axis(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,a,0));
                check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,actual_budget.data(),actual_budget.size()*8));
                require(std::memcmp(actual_budget.data(),expected_budget.data(),budget.size()*8)==0,"11-component budget halo changed shared/physical/corner bits");
            }
            double cleared_metrics[6];check(astr_device_transport_metrics(handle,cleared_metrics,1));
            void *slots[4];size_t capacity;check(astr_device_air5_symmetric_buffers(handle,slots,&capacity));
            double expected_sent=0.;int expected_calls=0;
            for(int round=0;round<2;++round)for(int a=0;a<3;++a) {
                std::vector<double> initial(count(faces)),expected,actual(count(faces));
                for(int c=1;c<=12;++c)for(int k=-1;k<=upper[2];++k)for(int j=-1;j<=upper[1];++j)for(int i=-1;i<=upper[0];++i)
                    initial[offset(faces,i,j,k,c)]=value(rank_id,round,a,i,j,k,c);
                expected=initial;check(astr_device_state_upload(handle,ASTR_DEVICE_INTERFACE_FLUX,initial.data(),initial.size()*8));
                for(int c=1;c<=12;++c)for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int i=0;i<=upper[0];++i) {
                    int p[3]={i,j,k};if(p[a]!=0)continue;
                    if(low[a]!=MPI_PROC_NULL) {
                        int source[3]={i,j,k};source[a]=upper[a]-1;p[a]=-1;
                        expected[offset(faces,p[0],p[1],p[2],c)]=value(low[a],round,a,source[0],source[1],source[2],c);
                    }
                    if(high[a]!=MPI_PROC_NULL) {
                        p[a]=upper[a];expected[offset(faces,p[0],p[1],p[2],c)]=value(high[a],round,a,i,j,k,c);
                    }
                }
                check(astr_device_air5_symmetric_exchange(handle,a+1));
                check(astr_device_state_download(handle,ASTR_DEVICE_INTERFACE_FLUX,actual.data(),actual.size()*8));
                require(std::memcmp(actual.data(),expected.data(),actual.size()*8)==0,"owned/received/interior/physical/corner face bits differ");
                void *after[4];size_t after_capacity;check(astr_device_air5_symmetric_buffers(handle,after,&after_capacity));
                require(capacity==after_capacity && std::memcmp(slots,after,sizeof(slots))==0,"face slots replaced during reuse");
                if(sizes[a]>1) {
                    expected_calls+=2;
                    const int peers=(low[a]!=MPI_PROC_NULL)+(high[a]!=MPI_PROC_NULL);
                    expected_sent+=peers*12.*(upper[(a+1)%3]+1)*(upper[(a+2)%3]+1)*8.;
                }
            }
            std::vector<double> actual_q(authority.size());
            check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual_q.data(),actual_q.size()*8));
            require(std::memcmp(authority.data(),actual_q.data(),authority.size()*8)==0,"communication changed authoritative q");
            double metrics[6];check(astr_device_transport_metrics(handle,metrics,0));
            require(metrics[4]==expected_sent && metrics[5]==(mode==ASTR_HALO_PINNED?2.*expected_sent:0.),"wire/staging byte accounting mismatch");
            require(wire_calls==expected_calls && wire_bad==0,"actual MPI datatype/tag/pointer evidence mismatch");
            check(astr_device_state_destroy(&handle));
            std::printf("AIR5_FACE_EXCHANGE_PASS rank=%d layout=%s mode=%s debug=%d calls=%d sent_bytes=%.0f staging_bytes=%.0f base11_halo_exact full_face_bits_exact q_unchanged slots_stable\n",
                rank_id,layout==0?"periodic":"physical",mode==0?"pinned":"device",debug,wire_calls,metrics[4],metrics[5]);
            wire_mode=-1;
        }
        check(MPI_Comm_free(&cart));
    }
    check(MPI_Finalize());return 0;
}
