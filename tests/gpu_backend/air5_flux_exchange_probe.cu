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
#include <cstdint>
#include <vector>

namespace {
constexpr int upper[3]={8,7,6},halo=5;
struct Message {int axis,first,components,peer,tag;bool high;};
std::vector<Message> sends,recvs;
int rank_id=0,round_id=0,wire_mode=-1,send_count=0,recv_count=0,wire_bad=0,progress_calls=0;
void check(int code)
{if(code) {std::fprintf(stderr,"rank%d operation status%d\n",rank_id,code);MPI_Abort(MPI_COMM_WORLD,code);std::exit(1);}}
void require(bool condition,const char *label)
{if(!condition) {std::fprintf(stderr,"rank%d %s\n",rank_id,label);MPI_Abort(MPI_COMM_WORLD,1);std::exit(1);}}
size_t count(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t offset(AstrDeviceView v,int i,int j,int k,int c,int d=1)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3]+(d-1)*v.stride[4];}
double value(int rank,int round,int channel,const int p[3])
{return (rank+1)*100000000.+round*10000000.+channel*100000.+p[0]*100.+p[1]*10.+p[2];}
void pointer_kind(const void *p)
{
    cudaPointerAttributes attributes{};
    const auto status=cudaPointerGetAttributes(&attributes,p);
    if(status!=cudaSuccess || attributes.type!=(wire_mode==ASTR_HALO_DEVICE?cudaMemoryTypeDevice:cudaMemoryTypeHost))++wire_bad;
}
bool signature(int n,MPI_Datatype datatype,int peer,int tag,const Message &m)
{
    const size_t expected=static_cast<size_t>(m.components)*halo*(upper[(m.axis+1)%3]+1)*(upper[(m.axis+2)%3]+1);
    return n==expected && datatype==MPI_DOUBLE && peer==m.peer && tag==m.tag;
}
void inspect_send(const void *buffer,int n,const Message &m)
{
    if(m.peer==MPI_PROC_NULL)return;
    std::vector<double> packet(n);
    if(wire_mode==ASTR_HALO_DEVICE)check(static_cast<int>(cudaMemcpy(packet.data(),buffer,n*8,cudaMemcpyDeviceToHost)));
    else std::memcpy(packet.data(),buffer,n*8);
    int dims[3]={upper[0]+1,upper[1]+1,upper[2]+1};dims[m.axis]=halo;
    for(size_t id=0;id<packet.size();++id) {
        size_t at=id;int p[3];p[0]=at%dims[0];at/=dims[0];p[1]=at%dims[1];at/=dims[1];p[2]=at%dims[2];
        const int channel=m.first+at/dims[2];p[m.axis]+=m.high?upper[m.axis]-halo:1;
        const double expected=value(rank_id,round_id,channel,p);
        if(packet[id]!=expected) {++wire_bad;break;}
    }
}
}
extern "C" int MPI_Isend(const void *buffer,int n,MPI_Datatype datatype,int peer,int tag,MPI_Comm comm,MPI_Request *request)
{
    if(wire_mode>=0) {
        if(send_count>=sends.size())++wire_bad;
        else {
            const auto m=sends[send_count];if(!signature(n,datatype,peer,tag,m))++wire_bad;
            pointer_kind(buffer);inspect_send(buffer,n,m);
        }
        ++send_count;
    }
    return PMPI_Isend(buffer,n,datatype,peer,tag,comm,request);
}
extern "C" int MPI_Irecv(void *buffer,int n,MPI_Datatype datatype,int peer,int tag,MPI_Comm comm,MPI_Request *request)
{
    if(wire_mode>=0) {
        if(recv_count>=recvs.size())++wire_bad;
        else {if(!signature(n,datatype,peer,tag,recvs[recv_count]))++wire_bad;pointer_kind(buffer);}
        ++recv_count;
    }
    return PMPI_Irecv(buffer,n,datatype,peer,tag,comm,request);
}
extern "C" int MPI_Testall(int n,MPI_Request *requests,int *complete,MPI_Status *statuses)
{if(wire_mode>=0)++progress_calls;return PMPI_Testall(n,requests,complete,statuses);}

int main(int argc,char **argv)
{
    if(argc!=5 && argc!=6)return 2;
    int first_backend=0,last_backend=2;
    if(argc==6) {
        if(std::strcmp(argv[5],"pinned")==0)first_backend=0;
        else if(std::strcmp(argv[5],"device")==0)first_backend=1;
        else return 2;
        last_backend=first_backend+1;
    }
    const char *local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)return 2;
    const int devices=std::atoi(argv[4]);int available;
    if(cudaGetDeviceCount(&available)!=cudaSuccess || devices<1 || devices>available)return 3;
    // Explicit test-only sharing, e.g. four component ranks on two local GPUs.
    const int device=std::atoi(local)%devices;if(astr_device_bind(device))return 3;
    check(MPI_Init(&argc,&argv));check(MPI_Comm_rank(MPI_COMM_WORLD,&rank_id));
    int nranks;check(MPI_Comm_size(MPI_COMM_WORLD,&nranks));
    int sizes[3]={std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3])};
    require(sizes[0]>0 && sizes[1]>0 && sizes[2]>0 && sizes[0]*sizes[1]*sizes[2]==nranks,"topology mismatch");
    const int ids[5]={ASTR_DEVICE_STRESS,ASTR_DEVICE_HEAT_FLUX,ASTR_DEVICE_AIR5_SPECIES_FLUX,
        ASTR_DEVICE_AIR5_EV_FLUX,ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX};
    for(int layout=0;layout<2;++layout) {
        int periodic[3]={layout==0,layout==0,layout==0};MPI_Comm cart;
        check(MPI_Cart_create(MPI_COMM_WORLD,3,sizes,periodic,0,&cart));
        int low[3],high[3];for(int a=0;a<3;++a)check(MPI_Cart_shift(cart,a,1,&low[a],&high[a]));
        for(int backend=first_backend;backend<last_backend;++backend)for(int mode=0;mode<3;++mode)for(int layered=0;layered<2;++layered) {
            const int debug=(mode+layered+layout)%2,channels=layered?33:27;
            void *handle=nullptr;check(astr_device_state_create(device,upper[0],upper[1],upper[2],halo,debug,&handle));
            check(astr_device_air5_prepare(handle));check(astr_device_air5_geometry_prepare(handle));
            check(astr_device_air5_transport_prepare(handle,layered));
            check(astr_device_transport_prepare(handle,static_cast<int>(MPI_Comm_c2f(cart)),low,high,sizes,mode==0?6:1,backend,1,periodic));
            check(astr_device_air5_flux_transport_prepare(handle,mode));
            AstrDeviceView q;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));
            std::vector<double> authority(count(q),rank_id+.125);
            check(astr_device_state_upload(handle,ASTR_DEVICE_Q,authority.data(),authority.size()*8));
            if(!layered) {AstrDeviceView unused;require(astr_device_state_view(handle,ids[4],&unused)==ASTR_DEVICE_NOT_ALLOCATED,"unused energy storage allocated");}
            sends.clear();recvs.clear();double expected_sent=0;
            const int groups[7]={6,3,5,5,5,3,6};int first=0;
            for(int group=0;group<(mode==0?(layered?7:6):1);++group) {
                const int components=mode==0?groups[group]:channels;
                for(int a=0;a<3;++a)if(sizes[a]>1) {
                    const int tag=20+2*a;
                    sends.push_back({a,first,components,low[a],tag,false});sends.push_back({a,first,components,high[a],tag+1,true});
                    recvs.push_back({a,first,components,high[a],tag,false});recvs.push_back({a,first,components,low[a],tag+1,true});
                    expected_sent+=2.*((low[a]!=MPI_PROC_NULL)+(high[a]!=MPI_PROC_NULL))*components*halo*(upper[(a+1)%3]+1)*(upper[(a+2)%3]+1)*8.;
                }
                first+=components;
            }
            int total_progress=0;
            for(round_id=0;round_id<2;++round_id) {
                std::vector<std::vector<double>> expected;
                first=0;
                for(int f=0;f<(layered?5:4);++f) {
                    AstrDeviceView v;check(astr_device_state_view(handle,ids[f],&v));std::vector<double> initial(count(v));
                    for(int d=1;d<=v.extent[4];++d)for(int c=1;c<=v.extent[3];++c)
                        for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i) {
                            const int p[3]={i,j,k},channel=first+c-1+(d-1)*v.extent[3];
                            const size_t at=offset(v,i,j,k,c,d);initial[at]=value(rank_id,round_id,channel,p);
                            if((i<0 || i>upper[0])+(j<0 || j>upper[1])+(k<0 || k>upper[2])>1) {
                                const uint64_t word=UINT64_C(0x7ff8000000000000)|at;std::memcpy(&initial[at],&word,8);
                            }
                        }
                    check(astr_device_state_upload(handle,ids[f],initial.data(),initial.size()*8));
                    auto reference=initial;
                    for(int d=1;d<=v.extent[4];++d)for(int c=1;c<=v.extent[3];++c)for(int a=0;a<3;++a)
                        for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int i=0;i<=upper[0];++i) {
                            int dst[3]={i,j,k},src[3]={i,j,k};if(dst[a]!=0)continue;
                            const int channel=first+c-1+(d-1)*v.extent[3];
                            for(int h=1;h<=halo;++h) {
                                if(low[a]!=MPI_PROC_NULL) {dst[a]=-h;src[a]=upper[a]-h;reference[offset(v,dst[0],dst[1],dst[2],c,d)]=value(low[a],round_id,channel,src);}
                                if(high[a]!=MPI_PROC_NULL) {dst[a]=upper[a]+h;src[a]=h;reference[offset(v,dst[0],dst[1],dst[2],c,d)]=value(high[a],round_id,channel,src);}
                            }
                        }
                    expected.push_back(std::move(reference));first+=v.extent[3]*v.extent[4];
                }
                send_count=recv_count=wire_bad=progress_calls=0;wire_mode=backend;
                check(astr_device_air5_flux_exchange(handle));wire_mode=-1;
                total_progress+=progress_calls;
                require(send_count==sends.size() && recv_count==recvs.size() && wire_bad==0,"MPI group/packet/order/pointer mismatch");
                for(int f=0;f<(layered?5:4);++f) {
                    AstrDeviceView v;check(astr_device_state_view(handle,ids[f],&v));std::vector<double> actual(count(v));
                    check(astr_device_state_download(handle,ids[f],actual.data(),actual.size()*8));
                    require(std::memcmp(actual.data(),expected[f].data(),actual.size()*8)==0,"flux physical/shared/halo/corner bits differ");
                }
            }
            std::vector<double> actual_q(authority.size());check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual_q.data(),actual_q.size()*8));
            require(std::memcmp(actual_q.data(),authority.data(),authority.size()*8)==0,"communication changed q");
            double metrics[6];check(astr_device_transport_metrics(handle,metrics,0));
            require(metrics[4]==expected_sent && metrics[5]==(backend==0?2.*expected_sent:0.),"copy byte accounting mismatch");
            check(astr_device_state_destroy(&handle));
            std::printf("AIR5_FLUX_EXCHANGE_PASS rank=%d layout=%s backend=%s mode=%d channels=%d debug=%d sent_bytes=%.0f staging_bytes=%.0f mpi_progress_calls=%d packets_exact fields_exact q_unchanged\n",
                rank_id,layout==0?"periodic":"physical",backend==0?"pinned":"device",mode,channels,debug,metrics[4],metrics[5],total_progress);
        }
        check(MPI_Comm_free(&cart));
    }
    check(MPI_Finalize());return 0;
}
