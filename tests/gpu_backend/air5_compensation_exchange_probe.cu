#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include "chemistry_transport.h"
#include "chemistry_compensation_math.cuh"
#include "halo_transport.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" void air5_reference_mean(double,double,double,double,double *);
extern "C" void air5_reference_normalize(double *,double *);
extern "C" void air5_reference_primitive(const double *,int,double *,int *);
namespace {
constexpr int upper[3]={8,7,6},halo=5;
int rank_id=0,watch_axis=-1,wire_mode=0,sends=0,payload_checks=0,wire_errors=0,lower_peer[3],upper_peer[3];
AstrDeviceView qview,cview;
const std::vector<double> *wire_q=nullptr,*wire_c=nullptr;
size_t size(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t at(AstrDeviceView v,const int p[3],int c=0)
{return (p[0]-v.lower[0])*v.stride[0]+(p[1]-v.lower[1])*v.stride[1]+(p[2]-v.lower[2])*v.stride[2]+c*v.stride[3];}
void require(bool valid,const char *label)
{if(!valid) {std::fprintf(stderr,"rank%d %s\n",rank_id,label);MPI_Abort(MPI_COMM_WORLD,1);std::exit(1);}}
void check(int code) {if(code) {std::fprintf(stderr,"rank%d operation%d\n",rank_id,code);MPI_Abort(MPI_COMM_WORLD,1);std::exit(1);}}
void exact(const std::vector<double> &a,const std::vector<double> &b,const char *label)
{if(std::memcmp(a.data(),b.data(),a.size()*8)) {for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],8)) {std::fprintf(stderr,"word%zu actual=%a ref=%a\n",i,a[i],b[i]);break;}require(false,label);}}
void point(size_t id,int axis,int p[3],int &c)
{const int u=axis==0?1:0,v=axis==2?1:2;p[0]=p[1]=p[2]=0;p[u]=id%(upper[u]+1);id/=upper[u]+1;p[v]=id%(upper[v]+1);c=id/(upper[v]+1);}
__global__ void scalar_math(const double *x,double *out,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;x+=4*i;out+=3*i;
    out[0]=astr::air5::compensated_mean(x[0],x[1],x[2],x[3]);out[1]=x[0];out[2]=x[2];
    astr::air5::compensated_normalize(out[1],out[2]);
}
int check_scalar_math()
{
    const double sub=std::nextafter(0.,1.);
    const double input[][4]={{1.,std::ldexp(1.,-53),0.,0.},{1.,-1.,std::ldexp(1.,-60),-std::ldexp(1.,-60)},
        {1e6,std::nextafter(1e6,INFINITY),1e-12,-2e-12},{-12345.,6789.,2e-13,-1e-13},
        {1.,1.,sub,sub},{1.,1.,-sub,-sub},{1.,-1.,sub,-sub},{1.,1.,-0.,0.},
        {std::ldexp(1.,-100),std::ldexp(1.,-101),std::ldexp(1.,-155),std::ldexp(1.,-155)}};
    const int n=sizeof(input)/sizeof(input[0]);double *x,*out;std::vector<double> actual(3*n),ref(3*n);
    check(cudaMalloc(&x,sizeof(input)));check(cudaMalloc(&out,actual.size()*8));check(cudaMemcpy(x,input,sizeof(input),cudaMemcpyHostToDevice));
    scalar_math<<<1,32>>>(x,out,n);check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(actual.data(),out,actual.size()*8,cudaMemcpyDeviceToHost));check(cudaFree(x));check(cudaFree(out));
    for(int i=0;i<n;++i) {air5_reference_mean(input[i][0],input[i][1],input[i][2],input[i][3],&ref[3*i]);
        ref[3*i+1]=input[i][0];ref[3*i+2]=input[i][2];air5_reference_normalize(&ref[3*i+1],&ref[3*i+2]);}
    exact(actual,ref,"scalar compensation math differs from original CPU source");return n;
}
}
extern "C" int MPI_Isend(const void *data,int n,MPI_Datatype type,int peer,int tag,MPI_Comm comm,MPI_Request *request)
{
    if(watch_axis>=0) {
        const int a=watch_axis,u=a==0?1:0,v=a==2?1:2,area=(upper[u]+1)*(upper[v]+1),side=tag-(42+2*a);
        if((side!=0 && side!=1) || type!=MPI_DOUBLE || n!=22*area || peer!=(side==0?lower_peer[a]:upper_peer[a])) {
            std::fprintf(stderr,"rank%d signature axis%d side%d count%d peer%d tag%d\n",rank_id,a,side,n,peer,tag);++wire_errors;
        }
        cudaPointerAttributes attr{};const int status=cudaPointerGetAttributes(&attr,data);
        if(status!=cudaSuccess || attr.type!=(wire_mode==ASTR_HALO_DEVICE?cudaMemoryTypeDevice:cudaMemoryTypeHost)) {
            std::fprintf(stderr,"rank%d pointer status%d type%d mode%d peer%d\n",rank_id,status,static_cast<int>(attr.type),wire_mode,peer);++wire_errors;
        }
        // MPI_PROC_NULL does not transfer payload, and the pinned host send
        // slot is deliberately not staged for this absent neighbor.
        if(peer!=MPI_PROC_NULL && (side==0 || side==1)) {
            ++payload_checks;
            std::vector<double> packet(n);if(wire_mode==ASTR_HALO_DEVICE)check(cudaMemcpy(packet.data(),data,n*8,cudaMemcpyDeviceToHost));
            else std::memcpy(packet.data(),data,n*8);
            for(size_t id=0;id<packet.size();++id) {
                int p[3],channel;point(id,a,p,channel);p[a]=side*upper[a];
                const double expected=channel<11?(*wire_q)[at(qview,p,channel)]:(*wire_c)[at(cview,p,channel-11)];
                if(std::memcmp(&packet[id],&expected,8)) {std::fprintf(stderr,"rank%d payload axis%d peer%d tag%d word%zu actual%a expected%a\n",rank_id,a,peer,tag,id,packet[id],expected);++wire_errors;break;}
            }
        }
        ++sends;
    }
    return PMPI_Isend(data,n,type,peer,tag,comm,request);
}
int main(int argc,char **argv)
{
    if(argc!=7)return 2;const char *local=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)return 2;
    const int devices=std::atoi(argv[5]);if(devices<1)return 2;check(astr_device_bind(std::atoi(local)%devices));
    check(MPI_Init(&argc,&argv));check(MPI_Comm_rank(MPI_COMM_WORLD,&rank_id));int ranks;check(MPI_Comm_size(MPI_COMM_WORLD,&ranks));
    int dims[3]={std::atoi(argv[2]),std::atoi(argv[3]),std::atoi(argv[4])},periodic[3]={0,0,1};
    require(dims[0]*dims[1]*dims[2]==ranks,"topology");
    if(std::strcmp(argv[6],"pinned")==0)wire_mode=ASTR_HALO_PINNED;else if(std::strcmp(argv[6],"device")==0)wire_mode=ASTR_HALO_DEVICE;else return 2;
    MPI_Comm cart;check(MPI_Cart_create(MPI_COMM_WORLD,3,dims,periodic,0,&cart));
    for(int a=0;a<3;++a)check(MPI_Cart_shift(cart,a,1,&lower_peer[a],&upper_peer[a]));
    FILE *file=std::fopen(argv[1],"rb");require(file,"base input");double base[11];require(std::fread(base,8,11,file)==11,"base data");require(std::fclose(file)==0,"base close");
    const int math_cases=check_scalar_math();int phases=0,packet_checks=0;double max_cache_ratio=0.;
    for(int compensated=0;compensated<2;++compensated)for(int debug=0;debug<2;++debug) {
        void *handle=nullptr;check(astr_device_state_create(std::atoi(local)%devices,upper[0],upper[1],upper[2],halo,debug,&handle));
        check(astr_device_air5_prepare(handle));
        if(compensated) {const AstrAir5ChemistryOptions options{1,0,0,1,debug,32,0,1e-9,1e-13,0.};check(astr_device_air5_chemistry_prepare(handle,&options));}
        check(astr_device_state_view(handle,ASTR_DEVICE_Q,&qview));
        if(compensated)check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&cview));
        else {cview={};require(astr_device_air5_compensation_axis(handle,1,0)==ASTR_DEVICE_NOT_ALLOCATED,"unprepared carry not rejected");}
        check(astr_device_transport_prepare(handle,static_cast<int>(MPI_Comm_c2f(cart)),lower_peer,upper_peer,dims,11,wire_mode,1,periodic));
        std::vector<double> q(size(qview)),actual(q.size()),carry(compensated?size(cview):0),actual_c(carry.size());
        for(size_t i=0;i<q.size();++i)q[i]=-777.-i/1048576.;
        for(int k=0;k<=upper[2];++k)for(int j=0;j<=upper[1];++j)for(int i=0;i<=upper[0];++i)for(int c=0;c<11;++c) {
            const int p[3]={i,j,k};q[at(qview,p,c)]=base[c]*(1.+(i+2*j+3*k+rank_id*.125)/1024.);
            if(compensated)carry[at(cview,p,c)]=std::ldexp(base[c],-55)*((i+j+k+c+rank_id)%2?1.:-1.);
        }
        check(astr_device_state_upload(handle,ASTR_DEVICE_Q,q.data(),q.size()*8));
        if(compensated)check(astr_device_state_upload(handle,ASTR_DEVICE_AIR5_CARRY,carry.data(),carry.size()*8));
        std::vector<double> gathered_q(q.size()*ranks),gathered_c(carry.size()*ranks);
        auto gather=[&]() {check(MPI_Allgather(q.data(),q.size(),MPI_DOUBLE,gathered_q.data(),q.size(),MPI_DOUBLE,cart));
            if(compensated)check(MPI_Allgather(carry.data(),carry.size(),MPI_DOUBLE,gathered_c.data(),carry.size(),MPI_DOUBLE,cart));};
        auto observe=[&](const char *phase) {
            check(astr_device_state_download(handle,ASTR_DEVICE_Q,actual.data(),actual.size()*8));exact(actual,q,phase);
            if(compensated) {check(astr_device_state_download(handle,ASTR_DEVICE_AIR5_CARRY,actual_c.data(),actual_c.size()*8));exact(actual_c,carry,phase);}
            ++phases;
        };
        auto q_exchange=[&](int a,int reconcile) {
            gather();const int u=a==0?1:0,v=a==2?1:2;const size_t plane=static_cast<size_t>(upper[u]+1)*(upper[v]+1)*11;
            for(int side=0;side<2;++side) {
                const int peer=side?upper_peer[a]:lower_peer[a];if(peer==MPI_PROC_NULL)continue;
                const double *remote=gathered_q.data()+peer*q.size();
                for(size_t id=0;id<plane;++id) {int p[3],c;point(id,a,p,c);
                    for(int h=reconcile?0:1;h<=halo;++h) {
                        int source[3]={p[0],p[1],p[2]};source[a]=side?h:upper[a]-h;p[a]=side?upper[a]+h:-h;
                        const size_t index=at(qview,p,c);q[index]=h==0?.5*q[index]+.5*remote[at(qview,source,c)]:remote[at(qview,source,c)];
                    }
                }
            }
            check(astr_device_exchange_axis(handle,ASTR_DEVICE_Q,a,reconcile));observe(reconcile?"q shared/halo":"final halo-only");
        };
        for(int a=0;a<3;++a) {
            const int u=a==0?1:0,v=a==2?1:2;const size_t plane=static_cast<size_t>(upper[u]+1)*(upper[v]+1)*11;
            if(compensated) {
                gather();const auto before_q=q,before_c=carry;wire_q=&before_q;wire_c=&before_c;sends=0;watch_axis=a;
                check(astr_device_air5_compensation_axis(handle,a+1,0));watch_axis=-1;
                require(sends==(dims[a]>1?2:0) && wire_errors==0,"q/carry wire signature/words/pointer kind");packet_checks+=sends;
                for(int side=0;side<2;++side) {
                    const int peer=side?upper_peer[a]:lower_peer[a];if(peer==MPI_PROC_NULL)continue;
                    const double *rq=gathered_q.data()+peer*q.size(),*rc=gathered_c.data()+peer*carry.size();
                    for(size_t id=0;id<plane;++id) {int p[3],c;point(id,a,p,c);p[a]=side*upper[a];int remote[3]={p[0],p[1],p[2]};remote[a]=(1-side)*upper[a];
                        const size_t index=at(cview,p,c);air5_reference_mean(before_q[at(qview,p,c)],rq[at(qview,remote,c)],before_c[index],rc[at(cview,remote,c)],&carry[index]);}
                }
                observe("carry mean before q average");
            }
            q_exchange(a,1);
            if(compensated) {
                for(int side=0;side<2;++side)if((side?upper_peer[a]:lower_peer[a])!=MPI_PROC_NULL)
                    for(size_t id=0;id<plane;++id) {int p[3],c;point(id,a,p,c);p[a]=side*upper[a];air5_reference_normalize(&q[at(qview,p,c)],&carry[at(cview,p,c)]);}
                check(astr_device_air5_compensation_axis(handle,a+1,1));observe("normalized q/carry faces");
            }
        }
        for(int a=0;a<3;++a)q_exchange(a,0);
        std::vector<int> first{0,0,0},last{upper[0],upper[1],upper[2]};
        for(int a=0;a<3;++a)for(int side=0;side<2;++side)if((side?upper_peer[a]:lower_peer[a])!=MPI_PROC_NULL) {
            int lo[3]={0,0,0},hi[3]={upper[0],upper[1],upper[2]};lo[a]=side?upper[a]+1:-halo;hi[a]=side?upper[a]+halo:-1;
            first.insert(first.end(),lo,lo+3);last.insert(last.end(),hi,hi+3);
        }
        check(astr_device_air5_primitives_regions(handle,first.size()/3,first.data(),last.data(),debug));
        int physics;check(astr_device_air5_check(handle,&physics));require(physics==0,"physical/final halo primitive batch");
        const int fields[6]={ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE,ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES};
        const int widths[6]={1,3,1,1,1,5},starts[6]={0,1,4,5,6,7};std::vector<AstrDeviceView> views(6);std::vector<std::vector<double>> cache(6);
        for(int f=0;f<6;++f) {check(astr_device_state_view(handle,fields[f],&views[f]));cache[f].resize(size(views[f]));check(astr_device_state_download(handle,fields[f],cache[f].data(),cache[f].size()*8));}
        for(int k=-halo;k<=upper[2]+halo;++k)for(int j=-halo;j<=upper[1]+halo;++j)for(int i=-halo;i<=upper[0]+halo;++i) {
            const int p[3]={i,j,k};int outside=0,axis=-1;for(int a=0;a<3;++a)if(p[a]<0 || p[a]>upper[a]) {++outside;axis=a;}
            if(outside>1 || (outside==1 && (p[axis]<0?lower_peer[axis]:upper_peer[axis])==MPI_PROC_NULL))continue;
            double state[11],ref[12];for(int c=0;c<11;++c)state[c]=q[at(qview,p,c)];int status;air5_reference_primitive(state,debug,ref,&status);require(status==0,"CPU primitive oracle");
            for(int f=0;f<6;++f) {double scale=1.;for(int c=0;c<widths[f];++c)scale=fmax(scale,fabs(ref[starts[f]+c]));
                for(int c=0;c<widths[f];++c) {const double value=cache[f][at(views[f],p,c)],expected=ref[starts[f]+c];
                    const double ratio=fabs(value-expected)/(1e-13*scale+5e-12*fabs(expected));require(std::isfinite(value)&&ratio<=1.,"final same-phase cache budget");max_cache_ratio=fmax(max_cache_ratio,ratio);}}
        }
        observe("primitive refresh leaves q/carry unchanged");check(astr_device_state_destroy(&handle));
    }
    std::printf("AIR5_COMPENSATION_EXCHANGE_PASS rank=%d mode=%d math=%d phases=%d packets=%d payloads=%d cache_ratio=%.17g\n",rank_id,wire_mode,math_cases,phases,packet_checks,payload_checks,max_cache_ratio);
    check(MPI_Comm_free(&cart));check(MPI_Finalize());return 0;
}
