#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <new>
#include <climits>
#include "device_state_internal.cuh"
#include "device_halo.h"
#include "halo_transport.h"
#include "keep_w1_operator.h"
#include "shock_sensor.h"
#include "shock_sensor_internal.cuh"
#include "crashfix_operator.h"
#include "chemistry_transport.h"
#include "chemistry_properties.h"
#include "chemistry_compensation_math.cuh"

namespace astr {
struct Air5FluxSlot {
    double *device[4]{};
    double *host=nullptr;
    size_t capacity=0;
};
struct HaloTransport {
    MPI_Comm communicator=MPI_COMM_NULL;
    int lower[3]{},upper[3]{},sizes[3]{},periodic[3]{},max_components=0,mode=ASTR_HALO_PINNED;
    bool active=false;
    int field=0,secondary=-1,axis=0,reconcile=0;
    bool timing=false,axis_timing=false;
    double metrics[6]{};
    double axis_metrics[3][5][7]{};
    numerical_real *send_device[2]{},*recv_device[2]{},*send_host[2]{},*recv_host[2]{};
    double *air5_face_host[4]{};
    double *air5_filter_host=nullptr;
    int air5_flux_mode=-1,air5_flux_components=0;
    bool air5_flux_ready=false;
    Air5FluxSlot air5_flux_slots[3];
    cudaStream_t air5_flux_pack_stream=nullptr;
    cudaEvent_t air5_flux_fields_ready=nullptr;
    MPI_Request requests[4]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
};

bool transport_in_flight(const HaloTransport *transport)
{
    return transport && transport->active;
}

int release_transport(HaloTransport *transport)
{
    if (!transport) return 0;
    if (transport->active) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=0;
    for (int side=0;side<2;++side) {
        void *device[]={transport->send_device[side],transport->recv_device[side]};
        void *host[]={transport->send_host[side],transport->recv_host[side]};
        for (void *p:device) if (p) {
            const int freed=static_cast<int>(cudaFree(p));
            if (status==0) status=freed;
        }
        for (void *p:host) if (p) {
            const int freed=static_cast<int>(cudaFreeHost(p));
            if (status==0) status=freed;
        }
    }
    for(double *p:transport->air5_face_host)if(p) {
        const int freed=static_cast<int>(cudaFreeHost(p));
        if(status==0)status=freed;
    }
    if(transport->air5_filter_host) {
        const int freed=static_cast<int>(cudaFreeHost(transport->air5_filter_host));if(status==0)status=freed;
    }
    if(transport->air5_flux_pack_stream) {
        const int waited=static_cast<int>(cudaStreamSynchronize(transport->air5_flux_pack_stream));
        if(status==0)status=waited;
        const int freed=static_cast<int>(cudaStreamDestroy(transport->air5_flux_pack_stream));
        if(status==0)status=freed;
    }
    if(transport->air5_flux_fields_ready) {
        const int freed=static_cast<int>(cudaEventDestroy(transport->air5_flux_fields_ready));
        if(status==0)status=freed;
    }
    for(auto &slot:transport->air5_flux_slots) {
        for(double *p:slot.device)if(p) {
            const int freed=static_cast<int>(cudaFree(p));if(status==0)status=freed;
        }
        if(slot.host) {const int freed=static_cast<int>(cudaFreeHost(slot.host));if(status==0)status=freed;}
    }
    if (transport->communicator!=MPI_COMM_NULL) {
        const int freed=MPI_Comm_free(&transport->communicator);
        if (status==0) status=freed;
    }
    delete transport;
    return status;
}
}

namespace {
constexpr int threads=astr::block_threads;
using Packet=astr::numerical_real;
const MPI_Datatype packet_mpi_type=sizeof(Packet)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
int halo_group(int field,int secondary)
{
    if(secondary>=0)return 4;
    switch(field) {
        case ASTR_DEVICE_Q:return 0;
        case ASTR_DEVICE_FILTER_WORK:return 1;
        case ASTR_DEVICE_STRESS:return 2;
        case ASTR_DEVICE_HEAT_FLUX:return 3;
        default:return -1;
    }
}

void record_halo_metric(astr::HaloTransport &t,int index,double value)
{
    t.metrics[index]+=value;
    if(t.axis_timing) {
        const int group=halo_group(t.field,t.secondary);
        if(group>=0)t.axis_metrics[t.axis][group][index]+=value;
    }
}

int components(const AstrDeviceView &view)
{
    return view.extent[3]*view.extent[4];
}

struct MaskBox { int upper[3]; };
struct Air5FluxFields {AstrDeviceView stress,heat,species,ev,energy;};
struct Air5FluxBox {int upper[3],halo;};
struct CompensationBox {int axis,upper,u,v,n1,n2;};
__device__ int compensation_point(size_t id,CompensationBox box,int p[3])
{
    p[0]=p[1]=p[2]=0;p[box.u]=id%box.n1;p[box.v]=(id/box.n1)%box.n2;
    return id/(static_cast<size_t>(box.n1)*box.n2);
}
__global__ void compensation_pack(AstrDeviceView q,AstrDeviceView carry,CompensationBox box,size_t count,double *low,double *high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int channel=compensation_point(id,box,p),component=channel%11+1;
    const AstrDeviceView view=channel<11?q:carry;const auto *data=static_cast<const double *>(view.data);
    low[id]=data[astr::offset(view,p[0],p[1],p[2],component)];p[box.axis]=box.upper;
    high[id]=data[astr::offset(view,p[0],p[1],p[2],component)];
}
__global__ void compensation_mean_remote(AstrDeviceView carry,CompensationBox box,size_t count,
    const double *low,const double *high,const double *minus,const double *plus,bool apply_low,bool apply_high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int c=compensation_point(id,box,p)+1;
    if(apply_low)static_cast<double *>(carry.data)[astr::offset(carry,p[0],p[1],p[2],c)]=
        astr::air5::compensated_mean(low[id],minus[id],low[count+id],minus[count+id]);
    if(apply_high) {p[box.axis]=box.upper;static_cast<double *>(carry.data)[astr::offset(carry,p[0],p[1],p[2],c)]=
        astr::air5::compensated_mean(high[id],plus[id],high[count+id],plus[count+id]);}
}
__global__ void compensation_mean_local(AstrDeviceView q,AstrDeviceView carry,CompensationBox box,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int c=compensation_point(id,box,p)+1;const size_t low=astr::offset(carry,p[0],p[1],p[2],c);
    const double ql=static_cast<const double *>(q.data)[astr::offset(q,p[0],p[1],p[2],c)];p[box.axis]=box.upper;
    const size_t high=astr::offset(carry,p[0],p[1],p[2],c);
    const double qr=static_cast<const double *>(q.data)[astr::offset(q,p[0],p[1],p[2],c)],
        cl=static_cast<const double *>(carry.data)[low],cr=static_cast<const double *>(carry.data)[high];
    static_cast<double *>(carry.data)[low]=astr::air5::compensated_mean(ql,qr,cl,cr);
    static_cast<double *>(carry.data)[high]=astr::air5::compensated_mean(qr,ql,cr,cl);
}
__global__ void compensation_normalize(AstrDeviceView q,AstrDeviceView carry,CompensationBox box,size_t count,bool low,bool high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int c=compensation_point(id,box,p)+1;
    for(int side=0;side<2;++side) {
        if((side==0 && !low)||(side==1 && !high))continue;p[box.axis]=side*box.upper;
        auto &value=static_cast<double *>(q.data)[astr::offset(q,p[0],p[1],p[2],c)];
        auto &residual=static_cast<double *>(carry.data)[astr::offset(carry,p[0],p[1],p[2],c)];
        astr::air5::compensated_normalize(value,residual);
    }
}

Air5FluxFields air5_flux_fields(astr::State &s)
{
    return {s.fields[ASTR_DEVICE_STRESS].view,s.fields[ASTR_DEVICE_HEAT_FLUX].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES_FLUX].view,s.fields[ASTR_DEVICE_AIR5_EV_FLUX].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX].view};
}
__device__ double &air5_flux_value(Air5FluxFields f,const int p[3],int channel)
{
    AstrDeviceView v;int c,d=1;
    if(channel<6) {v=f.stress;c=channel+1;}
    else if(channel<9) {v=f.heat;c=channel-5;}
    else if(channel<24) {v=f.species;c=(channel-9)%5+1;d=(channel-9)/5+1;}
    else if(channel<27) {v=f.ev;c=channel-23;}
    else {v=f.energy;c=channel-26;}
    return static_cast<double *>(v.data)[astr::offset(v,p[0],p[1],p[2],c,d)];
}
__device__ int air5_flux_point(size_t id,Air5FluxBox box,int axis,int first,int p[3])
{
    int dims[3]={box.upper[0]+1,box.upper[1]+1,box.upper[2]+1};dims[axis]=box.halo;
    p[0]=id%dims[0];id/=dims[0];p[1]=id%dims[1];id/=dims[1];p[2]=id%dims[2];
    return first+id/dims[2];
}
__global__ void air5_flux_pack(Air5FluxFields fields,Air5FluxBox box,int axis,int first,
    size_t count,double *low,double *high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int channel=air5_flux_point(id,box,axis,first,p),h=p[axis];
    p[axis]=h+1;low[id]=air5_flux_value(fields,p,channel);
    p[axis]=box.upper[axis]-box.halo+h;high[id]=air5_flux_value(fields,p,channel);
}
__global__ void air5_flux_unpack(Air5FluxFields fields,Air5FluxBox box,int axis,int first,
    size_t count,const double *plus,const double *minus,bool low,bool high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int channel=air5_flux_point(id,box,axis,first,p),h=p[axis];
    if(low) {p[axis]=-box.halo+h;air5_flux_value(fields,p,channel)=minus[id];}
    if(high) {p[axis]=box.upper[axis]+1+h;air5_flux_value(fields,p,channel)=plus[id];}
}
__global__ void air5_flux_periodic(Air5FluxFields fields,Air5FluxBox box,int axis,int first,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    int p[3];const int channel=air5_flux_point(id,box,axis,first,p),h=p[axis];
    p[axis]=h+1;const double high=air5_flux_value(fields,p,channel);
    p[axis]=box.upper[axis]-box.halo+h;const double low=air5_flux_value(fields,p,channel);
    p[axis]=-box.halo+h;air5_flux_value(fields,p,channel)=low;
    p[axis]=box.upper[axis]+1+h;air5_flux_value(fields,p,channel)=high;
}

int allocate_air5_flux_slot(astr::Air5FluxSlot &slot,size_t count,int mode)
{
    slot.capacity=count;
    // Pass an allocation base pointer for each DCU device MPI message.
    for(double *&p:slot.device) {
        const int status=static_cast<int>(cudaMalloc(&p,count*sizeof(double)));if(status)return status;
    }
    return mode==ASTR_HALO_PINNED?static_cast<int>(cudaMallocHost(&slot.host,4*count*sizeof(double))):0;
}

int exchange_air5_flux_group(astr::State &s,int first,int components)
{
    auto &t=*s.transport;const auto fields=air5_flux_fields(s);
    const Air5FluxBox box{{s.upper[0],s.upper[1],s.upper[2]},s.halo};
    bool prepared[3]{};
    if(t.air5_flux_pack_stream) {
        int status=static_cast<int>(cudaEventRecord(t.air5_flux_fields_ready,s.stream));if(status)return status;
        status=static_cast<int>(cudaStreamWaitEvent(t.air5_flux_pack_stream,t.air5_flux_fields_ready,0));if(status)return status;
    }
    for(int a=0;a<3;++a) {
        const size_t count=static_cast<size_t>(components)*s.halo*(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1);
        const size_t bytes=count*sizeof(double);
        if(t.sizes[a]==1) {
            if(t.periodic[a]) {
                air5_flux_periodic<<<(count+255)/256,256,0,s.stream>>>(fields,box,a,first,count);
                const int status=astr::finish_launch(s);if(status)return status;
            }
            continue;
        }
        double *device[4],*host[4];
        if(t.air5_flux_mode==ASTR_AIR5_FLUX_SEPARATE) {
            device[0]=reinterpret_cast<double *>(t.send_device[0]);device[1]=reinterpret_cast<double *>(t.send_device[1]);
            device[2]=reinterpret_cast<double *>(t.recv_device[1]);device[3]=reinterpret_cast<double *>(t.recv_device[0]);
            host[0]=reinterpret_cast<double *>(t.send_host[0]);host[1]=reinterpret_cast<double *>(t.send_host[1]);
            host[2]=reinterpret_cast<double *>(t.recv_host[1]);host[3]=reinterpret_cast<double *>(t.recv_host[0]);
        } else {
            auto &slot=t.air5_flux_slots[a];
            for(int i=0;i<4;++i) {device[i]=slot.device[i];host[i]=slot.host?slot.host+i*slot.capacity:nullptr;}
        }
        const double started=t.timing?MPI_Wtime():0;
        if(!prepared[a]) {
            air5_flux_pack<<<(count+255)/256,256,0,s.stream>>>(fields,box,a,first,count,device[0],device[1]);
            const int status=astr::finish_launch(s);if(status)return status;
        }
        void *wire[4];
        for(int i=0;i<4;++i)wire[i]=t.mode==ASTR_HALO_DEVICE?device[i]:host[i];
        const bool low=t.lower[a]!=MPI_PROC_NULL,high=t.upper[a]!=MPI_PROC_NULL;
        int status=0;
        if(t.mode==ASTR_HALO_PINNED) {
            if(low) {status=static_cast<int>(cudaMemcpyAsync(wire[0],device[0],bytes,cudaMemcpyDeviceToHost,s.stream));if(status)return status;}
            if(high) {status=static_cast<int>(cudaMemcpyAsync(wire[1],device[1],bytes,cudaMemcpyDeviceToHost,s.stream));if(status)return status;}
        }
        status=static_cast<int>(cudaStreamSynchronize(s.stream));if(status)return status;
        const double ready=t.timing?MPI_Wtime():0;
        const int tag=20+2*a;
        MPI_Request requests[4]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
        status=MPI_Irecv(wire[2],static_cast<int>(count),MPI_DOUBLE,t.upper[a],tag,t.communicator,&requests[0]);if(status)return status;
        status=MPI_Irecv(wire[3],static_cast<int>(count),MPI_DOUBLE,t.lower[a],tag+1,t.communicator,&requests[1]);if(status)return status;
        status=MPI_Isend(wire[0],static_cast<int>(count),MPI_DOUBLE,t.lower[a],tag,t.communicator,&requests[2]);if(status)return status;
        status=MPI_Isend(wire[1],static_cast<int>(count),MPI_DOUBLE,t.upper[a],tag+1,t.communicator,&requests[3]);if(status)return status;
        const double posted=t.timing?MPI_Wtime():0;
        if(t.air5_flux_pack_stream) {
            int next=a+1;while(next<3 && t.sizes[next]==1)++next;
            if(next<3) {
                auto &next_slot=t.air5_flux_slots[next];
                const size_t next_count=static_cast<size_t>(components)*s.halo*(s.upper[(next+1)%3]+1)*(s.upper[(next+2)%3]+1);
                air5_flux_pack<<<(next_count+255)/256,256,0,t.air5_flux_pack_stream>>>(fields,box,next,first,next_count,next_slot.device[0],next_slot.device[1]);
                status=static_cast<int>(cudaGetLastError());
                while(status==0) {
                    const auto progress=cudaStreamQuery(t.air5_flux_pack_stream);
                    if(progress==cudaSuccess)break;
                    if(progress!=cudaErrorNotReady) {status=static_cast<int>(progress);break;}
                    int complete=0;status=MPI_Testall(4,requests,&complete,MPI_STATUSES_IGNORE);
                    if(complete)break;
                }
                if(status==0)status=static_cast<int>(cudaStreamSynchronize(t.air5_flux_pack_stream));
                if(status) {MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);return status;}
                prepared[next]=true;
            }
        }
        status=MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);if(status)return status;
        const double received=t.timing?MPI_Wtime():0;
        if(t.mode==ASTR_HALO_PINNED) {
            if(high) {status=static_cast<int>(cudaMemcpyAsync(device[2],wire[2],bytes,cudaMemcpyHostToDevice,s.stream));if(status)return status;}
            if(low) {status=static_cast<int>(cudaMemcpyAsync(device[3],wire[3],bytes,cudaMemcpyHostToDevice,s.stream));if(status)return status;}
        }
        air5_flux_unpack<<<(count+255)/256,256,0,s.stream>>>(fields,box,a,first,count,device[2],device[3],low,high);
        status=astr::finish_launch(s);
        if(t.timing) {
            t.metrics[0]+=ready-started;t.metrics[1]+=posted-ready;t.metrics[2]+=received-posted;t.metrics[3]+=MPI_Wtime()-received;
            t.metrics[4]+=(low+high)*static_cast<double>(bytes);
            if(t.mode==ASTR_HALO_PINNED)t.metrics[5]+=2.*(low+high)*bytes;
        }
        if(status)return status;
    }
    return 0;
}

__global__ __launch_bounds__(threads)
void critical_local(int8_t *mask,AstrDeviceView view,MaskBox box,int axis,int width,int n1,
    size_t count,bool shared)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    const int h=id%width;const size_t transverse=id/width;
    int p[3]={};p[(axis+1)%3]=transverse%n1;p[(axis+2)%3]=transverse/n1;
    if(shared) {
        const size_t low=astr::offset(view,p[0],p[1],p[2]);
        p[axis]=box.upper[axis];const size_t high=astr::offset(view,p[0],p[1],p[2]);
        const int8_t value=static_cast<int8_t>(mask[low]!=0 || mask[high]!=0);
        mask[low]=value;mask[high]=value;
    } else {
        p[axis]=box.upper[axis]-width+h;const int8_t low=mask[astr::offset(view,p[0],p[1],p[2])];
        p[axis]=h+1;const int8_t high=mask[astr::offset(view,p[0],p[1],p[2])];
        p[axis]=-width+h;mask[astr::offset(view,p[0],p[1],p[2])]=low;
        p[axis]=box.upper[axis]+h+1;mask[astr::offset(view,p[0],p[1],p[2])]=high;
    }
}

__global__ __launch_bounds__(threads)
void critical_pack(const int8_t *mask,AstrDeviceView view,MaskBox box,int axis,int width,int n1,
    size_t count,bool shared,int8_t *low,int8_t *high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    const int h=id%width;const size_t transverse=id/width;
    int p[3]={};p[(axis+1)%3]=transverse%n1;p[(axis+2)%3]=transverse/n1;
    p[axis]=shared ? 0 : h+1;low[id]=mask[astr::offset(view,p[0],p[1],p[2])];
    p[axis]=shared ? box.upper[axis] : box.upper[axis]-width+h;
    high[id]=mask[astr::offset(view,p[0],p[1],p[2])];
}

__global__ __launch_bounds__(threads)
void critical_unpack(int8_t *mask,AstrDeviceView view,MaskBox box,int axis,int width,int n1,
    size_t count,bool shared,const int8_t *low,const int8_t *high,bool lower,bool upper)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    const int h=id%width;const size_t transverse=id/width;
    int p[3]={};p[(axis+1)%3]=transverse%n1;p[(axis+2)%3]=transverse/n1;
    if(lower) {
        p[axis]=shared ? 0 : -width+h;
        const size_t at=astr::offset(view,p[0],p[1],p[2]);
        mask[at]=shared ? static_cast<int8_t>(mask[at]!=0 || low[id]!=0) : low[id];
    }
    if(upper) {
        p[axis]=shared ? box.upper[axis] : box.upper[axis]+h+1;
        const size_t at=astr::offset(view,p[0],p[1],p[2]);
        mask[at]=shared ? static_cast<int8_t>(mask[at]!=0 || high[id]!=0) : high[id];
    }
}

__global__ __launch_bounds__(threads)
void pack_mask(const int8_t *mask,MaskBox box,int axis,int n1,size_t count,int8_t *low,int8_t *high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int p[3]={};p[(axis+1)%3]=id%n1;p[(axis+2)%3]=id/n1;p[axis]=1;
        low[id]=mask[astr::shock_node_offset(box.upper,p)];
        p[axis]=box.upper[axis]-1;high[id]=mask[astr::shock_node_offset(box.upper,p)];
    }
    return;
}

__global__ __launch_bounds__(threads)
void unpack_mask(const int8_t *mask,int8_t *faces,MaskBox box,int axis,int n1,size_t count,
    const int8_t *low,const int8_t *high,int lower_rule,int upper_rule)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int p[3]={};p[(axis+1)%3]=id%n1;p[(axis+2)%3]=id/n1;
        p[axis]=lower_rule==2 ? box.upper[axis]-1 : 0;
        faces[astr::shock_face_offset(box.upper,axis,0,p)]=lower_rule==1 ? low[id]
            : mask[astr::shock_node_offset(box.upper,p)];
        p[axis]=upper_rule==2 ? 1 : box.upper[axis];
        faces[astr::shock_face_offset(box.upper,axis,1,p)]=upper_rule==1 ? high[id]
            : mask[astr::shock_node_offset(box.upper,p)];
    }
    return;
}

__device__ void packet_point(size_t id,int width,int first,int n1,int n2,int axis,AstrDeviceView view,
                             int p[3],int &c,int &d,int &h)
{
    h=id%width+first;
    p[(axis+1)%3]=(id/width)%n1;
    p[(axis+2)%3]=(id/(static_cast<size_t>(width)*n1))%n2;
    const size_t component=id/(static_cast<size_t>(width)*n1*n2);
    c=component%view.extent[3]+1;d=component/view.extent[3]+1;
    return;
}

__global__ __launch_bounds__(threads)
void pack(AstrDeviceView view,int axis,int upper,int width,int first,int n1,int n2,size_t count,Packet *low,Packet *high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id<count) {
        int p[3]={0,0,0},c,d,h;
        packet_point(id,width,first,n1,n2,axis,view,p,c,d,h);
        p[axis]=h;
        low[id]=astr::load_value(view,p[0],p[1],p[2],c,d);
        p[axis]=upper-h;
        high[id]=astr::load_value(view,p[0],p[1],p[2],c,d);
    }
    return;
}

__global__ __launch_bounds__(threads)
void unpack(AstrDeviceView view,int axis,int upper,int width,int first,int n1,int n2,size_t count,
            const Packet *low,const Packet *high,bool owner_copy,bool apply_low,bool apply_high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if (id<count) {
        int p[3]={0,0,0},c,d,h;
        packet_point(id,width,first,n1,n2,axis,view,p,c,d,h);
        if(apply_low) {
            p[axis]=-h;
            astr::store_value(view,p[0],p[1],p[2],h>0 || owner_copy ? low[id]
                : Packet(.5)*(astr::load_value(view,p[0],p[1],p[2],c,d)+low[id]),c,d);
        }
        if(apply_high) {
            p[axis]=upper+h;
            astr::store_value(view,p[0],p[1],p[2],h>0 || owner_copy ? high[id]
                : Packet(.5)*(astr::load_value(view,p[0],p[1],p[2],c,d)+high[id]),c,d);
        }
    }
    return;
}

// Below 2: existing halo-only/average operations. Owner-copy encodes the selected
// interface sides and the subset that receives the authoritative value.
constexpr int owner_copy_flag=16;
int halo_peer(const astr::HaloTransport &transport,int axis,int side,int operation)
{
    if((operation&owner_copy_flag) && !(operation&(1<<side))) return MPI_PROC_NULL;
    return side==0 ? transport.lower[axis] : transport.upper[axis];
}
}

extern "C" int astr_device_transport_prepare(void *handle,int communicator,const int lower[3],
    const int upper[3],const int sizes[3],int max_components,int mode,int timing,const int periodic[3])
{
    if (!handle || !lower || !upper || !sizes || !periodic || max_components<1 ||
        (mode!=ASTR_HALO_PINNED && mode!=ASTR_HALO_DEVICE)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (s.transport || s.halo<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    size_t capacity=0;
    for (int axis=0;axis<3;++axis) {
        if (sizes[axis]<1 || s.upper[axis]<s.halo) return ASTR_DEVICE_INVALID_ARGUMENT;
        if (periodic[axis]!=0 && periodic[axis]!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
        if (sizes[axis]>1) {
            if (periodic[axis] && (lower[axis]==MPI_PROC_NULL || upper[axis]==MPI_PROC_NULL)) return ASTR_DEVICE_INVALID_ARGUMENT;
            const size_t count=static_cast<size_t>(s.halo+1)*(s.upper[(axis+1)%3]+1)
                              *(s.upper[(axis+2)%3]+1)*max_components;
            if (count>INT_MAX) return ASTR_DEVICE_INVALID_ARGUMENT;
            if (count>capacity) capacity=count;
        }
    }
    s.transport=new (std::nothrow) astr::HaloTransport{};
    if (!s.transport) return static_cast<int>(cudaErrorMemoryAllocation);
    astr::HaloTransport &t=*s.transport;
    t.max_components=max_components;
    t.mode=mode;
    t.timing=timing!=0;
    t.axis_timing=timing==2;
    for (int axis=0;axis<3;++axis) {
        t.lower[axis]=lower[axis];t.upper[axis]=upper[axis];t.sizes[axis]=sizes[axis];t.periodic[axis]=periodic[axis];
    }
    if (capacity==0) return 0;
    int status=MPI_Comm_dup(MPI_Comm_f2c(static_cast<MPI_Fint>(communicator)),&t.communicator);
    if (status!=MPI_SUCCESS) return status;
    // Tags 0/1 belong only to this private communicator; Fortran owns topology.
    for (int side=0;side<2;++side) {
        cudaError_t err=cudaMalloc(&t.send_device[side],capacity*sizeof(Packet));
        if (err!=cudaSuccess) return static_cast<int>(err);
        err=cudaMalloc(&t.recv_device[side],capacity*sizeof(Packet));
        if (err!=cudaSuccess) return static_cast<int>(err);
        if (mode==ASTR_HALO_PINNED) {
            err=cudaMallocHost(&t.send_host[side],capacity*sizeof(Packet));
            if (err!=cudaSuccess) return static_cast<int>(err);
            err=cudaMallocHost(&t.recv_host[side],capacity*sizeof(Packet));
            if (err!=cudaSuccess) return static_cast<int>(err);
        }
    }
    return 0;
}

static int exchange_begin(void *handle,int field,int secondary,int axis,int reconcile)
{
    if (!handle || field<0 || field>=ASTR_DEVICE_FIELD_COUNT || axis<0 || axis>2)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (!s.transport) return ASTR_DEVICE_NOT_ALLOCATED;
    astr::HaloTransport &t=*s.transport;
    if (t.active) return ASTR_DEVICE_INVALID_ARGUMENT;
    const bool owner_copy=(reconcile&owner_copy_flag)!=0;
    if(owner_copy ? (secondary>=0 || t.periodic[axis]) : (reconcile!=0 && reconcile!=1))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const AstrDeviceView view=s.fields[field].view;
    if (!view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    const AstrDeviceView other=secondary>=0?s.fields[secondary].view:AstrDeviceView{};
    if (secondary>=0 && !other.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if (t.sizes[axis]==1) {
        int status=t.periodic[axis] ? astr_device_periodic_axis(handle,field,axis,reconcile) : 0;
        if (status==0 && t.periodic[axis] && secondary>=0)
            status=astr_device_periodic_axis(handle,secondary,axis,reconcile);
        if (status!=0) return status;
        t.field=field;t.secondary=secondary;t.axis=axis;t.reconcile=reconcile;t.active=true;
        return 0;
    }
    if (view.lower[axis]!=-s.halo || (secondary>=0 && other.lower[axis]!=-s.halo) ||
        components(view)+components(other)>t.max_components)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int n1=s.upper[(axis+1)%3]+1,n2=s.upper[(axis+2)%3]+1;
    const int width=owner_copy ? 1 : s.halo+(reconcile ? 1 : 0),first=reconcile ? 0 : 1;
    const int lower=halo_peer(t,axis,0,reconcile),upper=halo_peer(t,axis,1,reconcile);
    const size_t face=static_cast<size_t>(width)*n1*n2;
    const size_t first_count=face*components(view),second_count=face*components(other);
    const size_t count=first_count+second_count,bytes=count*sizeof(Packet);
    const double started=t.timing ? MPI_Wtime() : 0;
    pack<<<(first_count+threads-1)/threads,threads,0,s.stream>>>(view,axis,s.upper[axis],width,first,n1,n2,first_count,
        t.send_device[0],t.send_device[1]);
    int status=astr::finish_launch(s);
    if (status!=0) return status;
    if (secondary>=0) {
        pack<<<(second_count+threads-1)/threads,threads,0,s.stream>>>(other,axis,s.upper[axis],width,first,n1,n2,
            second_count,t.send_device[0]+first_count,t.send_device[1]+first_count);
        status=astr::finish_launch(s);
        if (status!=0) return status;
    }
    for (int side=0;t.mode==ASTR_HALO_PINNED && side<2;++side) {
        if ((side==0 ? lower : upper)==MPI_PROC_NULL) continue;
        const cudaError_t err=cudaMemcpyAsync(t.send_host[side],t.send_device[side],bytes,cudaMemcpyDeviceToHost,s.stream);
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    cudaError_t err=cudaStreamSynchronize(s.stream);
    if (err!=cudaSuccess) return static_cast<int>(err);
    const double posted=t.timing ? MPI_Wtime() : 0;
    // This wait also completes the previous end's unpack before MPI may
    // overwrite the fixed receive slots. No requests are posted before it.
    t.field=field;t.secondary=secondary;t.axis=axis;t.reconcile=reconcile;t.active=true;
    Packet **recv=t.mode==ASTR_HALO_DEVICE ? t.recv_device : t.recv_host;
    Packet **send=t.mode==ASTR_HALO_DEVICE ? t.send_device : t.send_host;
    status=MPI_Irecv(recv[0],static_cast<int>(count),packet_mpi_type,lower,1,t.communicator,&t.requests[0]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Irecv(recv[1],static_cast<int>(count),packet_mpi_type,upper,0,t.communicator,&t.requests[1]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Isend(send[0],static_cast<int>(count),packet_mpi_type,lower,0,t.communicator,&t.requests[2]);
    if (status!=MPI_SUCCESS) return status;
    status=MPI_Isend(send[1],static_cast<int>(count),packet_mpi_type,upper,1,t.communicator,&t.requests[3]);
    if (status!=MPI_SUCCESS) return status;
    if (t.timing) {
        record_halo_metric(t,1,MPI_Wtime()-posted);
        record_halo_metric(t,0,posted-started);
        const int peers=(lower!=MPI_PROC_NULL)+(upper!=MPI_PROC_NULL);
        record_halo_metric(t,4,peers*static_cast<double>(bytes));
        if (t.mode==ASTR_HALO_PINNED) record_halo_metric(t,5,2.0*peers*bytes);
        if (t.axis_timing) {
            const int group=halo_group(field,secondary);
            if(group>=0)++t.axis_metrics[axis][group][6];
        }
    }
    return 0;
}

static int exchange_end(void *handle,int field,int secondary,int axis,int reconcile)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if (!s.transport) return ASTR_DEVICE_NOT_ALLOCATED;
    astr::HaloTransport &t=*s.transport;
    if (!t.active || field!=t.field || secondary!=t.secondary || axis!=t.axis || reconcile!=t.reconcile)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    if (t.sizes[axis]==1) { t.active=false;return 0; }
    const AstrDeviceView view=s.fields[field].view;
    const AstrDeviceView other=secondary>=0?s.fields[secondary].view:AstrDeviceView{};
    const int n1=s.upper[(axis+1)%3]+1,n2=s.upper[(axis+2)%3]+1;
    const bool owner_copy=(reconcile&owner_copy_flag)!=0;
    const int width=owner_copy ? 1 : s.halo+(reconcile ? 1 : 0),first=reconcile ? 0 : 1;
    const int lower=halo_peer(t,axis,0,reconcile),upper=halo_peer(t,axis,1,reconcile);
    const size_t face=static_cast<size_t>(width)*n1*n2;
    const size_t first_count=face*components(view),second_count=face*components(other);
    const size_t bytes=(first_count+second_count)*sizeof(Packet);
    const double started=t.timing ? MPI_Wtime() : 0;
    int status=MPI_Waitall(4,t.requests,MPI_STATUSES_IGNORE);
    if (status!=MPI_SUCCESS) return status;
    const double received=t.timing ? MPI_Wtime() : 0;
    if (t.timing) record_halo_metric(t,2,received-started);
    for (int side=0;t.mode==ASTR_HALO_PINNED && side<2;++side) {
        if ((side==0 ? lower : upper)==MPI_PROC_NULL) continue;
        const cudaError_t err=cudaMemcpyAsync(t.recv_device[side],t.recv_host[side],bytes,cudaMemcpyHostToDevice,s.stream);
        if (err!=cudaSuccess) return static_cast<int>(err);
    }
    unpack<<<(first_count+threads-1)/threads,threads,0,s.stream>>>(view,axis,s.upper[axis],width,first,n1,n2,first_count,
        t.recv_device[0],t.recv_device[1],owner_copy,lower!=MPI_PROC_NULL && (!owner_copy || (reconcile&4)),
        upper!=MPI_PROC_NULL && (!owner_copy || (reconcile&8)));
    status=astr::finish_launch(s);
    if (status==0 && secondary>=0) {
        unpack<<<(second_count+threads-1)/threads,threads,0,s.stream>>>(other,axis,s.upper[axis],width,first,n1,n2,
            second_count,t.recv_device[0]+first_count,t.recv_device[1]+first_count,false,
            t.lower[axis]!=MPI_PROC_NULL,t.upper[axis]!=MPI_PROC_NULL);
        status=astr::finish_launch(s);
    }
    if (t.timing) record_halo_metric(t,3,MPI_Wtime()-received);
    if (status==0) t.active=false;
    return status;
}

extern "C" int astr_device_exchange_begin(void *handle,int field,int axis,int reconcile)
{
    return exchange_begin(handle,field,-1,axis,reconcile);
}

extern "C" int astr_device_exchange_end(void *handle,int field,int axis,int reconcile)
{
    return exchange_end(handle,field,-1,axis,reconcile);
}

extern "C" int astr_device_viscous_exchange_begin(void *handle,int axis)
{
    return exchange_begin(handle,ASTR_DEVICE_STRESS,ASTR_DEVICE_HEAT_FLUX,axis,0);
}

extern "C" int astr_device_viscous_exchange_end(void *handle,int axis)
{
    return exchange_end(handle,ASTR_DEVICE_STRESS,ASTR_DEVICE_HEAT_FLUX,axis,0);
}

extern "C" int astr_device_exchange_axis(void *handle,int field,int axis,int reconcile)
{
    const int status=astr_device_exchange_begin(handle,field,axis,reconcile);
    return status==0 ? astr_device_exchange_end(handle,field,axis,reconcile) : status;
}

extern "C" int astr_device_exchange_owned_plane(void *handle,int field,int axis,int peers,int receivers)
{
    const int operation=owner_copy_flag|peers|(receivers<<2);
    const int status=exchange_begin(handle,field,-1,axis,operation);
    return status==0 ? exchange_end(handle,field,-1,axis,operation) : status;
}

namespace {
__global__ void pack_keep_w1_end(AstrDeviceView velocity,AstrDeviceView temperature,AstrDeviceView metric,
    AstrDeviceView flux,int nx,int nz,int span,bool upper,size_t points,Packet *packet)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=points) return;
    const int i=id%nx,k=(id/nx)%nz,p=id/(static_cast<size_t>(nx)*nz);
    const int j=upper ? span-p : p;
    for(int m=0;m<3;++m) packet[7*id+m]=astr::at(velocity,i,j,k,m+1);
    packet[7*id+3]=astr::at(temperature,i,j,k);
    packet[7*id+4]=astr::at(flux,i,j,k,13);
    packet[7*id+5]=astr::at(flux,i,j,k,14);
    packet[7*id+6]=1./astr::at(metric,i,j,k,2,2);
}
}

int astr::exchange_keep_w1_ends(State &s,const int send_width[2],const int recv_width[2],const void *received[2])
{
    auto &t=*s.transport;
    const int nx=s.upper[0]+1,nz=s.upper[2]+1;
    const size_t area=static_cast<size_t>(nx)*nz;
    const bool pinned=t.mode==ASTR_HALO_PINNED;
    for(int side=0;side<2;++side) if(send_width[side]) {
        // At most three end planes of seven values fit the fixed W1 halo slots.
        const size_t points=area*send_width[side];
        pack_keep_w1_end<<<(points+threads-1)/threads,threads,0,s.stream>>>(
            s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,
            s.fields[ASTR_DEVICE_KEEP_W1_FLUX].view,nx,nz,s.upper[1],side==1,points,t.send_device[side]);
        int status=finish_launch(s);
        if(status!=0) return status;
        if(pinned) {
            status=static_cast<int>(cudaMemcpyAsync(t.send_host[side],t.send_device[side],
                points*7*sizeof(Packet),cudaMemcpyDeviceToHost,s.stream));
            if(status!=0) return status;
        }
    }
    // Retire regular unpack (and any prior packet consumer) before MPI overwrites the slots.
    int status=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status!=0) return status;
    for(int side=0;side<2;++side) if(send_width[side] || recv_width[side]) {
        const int sends=static_cast<int>(area*7*send_width[side]);
        const int recvs=static_cast<int>(area*7*recv_width[side]);
        const int dest=sends ? (side==0 ? t.upper[1] : t.lower[1]) : MPI_PROC_NULL;
        const int source=recvs ? (side==0 ? t.lower[1] : t.upper[1]) : MPI_PROC_NULL;
        const int tag=61+side;
        status=MPI_Sendrecv(pinned ? t.send_host[side] : t.send_device[side],sends,packet_mpi_type,dest,tag,
            pinned ? t.recv_host[side] : t.recv_device[side],recvs,packet_mpi_type,source,tag,t.communicator,MPI_STATUS_IGNORE);
        if(status!=MPI_SUCCESS) return status;
        if(recvs && pinned) {
            status=static_cast<int>(cudaMemcpyAsync(t.recv_device[side],t.recv_host[side],
                static_cast<size_t>(recvs)*sizeof(Packet),cudaMemcpyHostToDevice,s.stream));
            if(status!=0) return status;
        }
        if(recvs) received[side]=t.recv_device[side];
    }
    return 0;
}

extern "C" int astr_device_air5_compensation_axis(void *handle,int axis,int normalize)
{
    if(!handle || axis<1 || axis>3 || normalize<0 || normalize>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.air5_ready || !s.transport ||
        !s.fields[ASTR_DEVICE_AIR5_CARRY].view.data)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &t=*s.transport;if(t.active)return ASTR_DEVICE_INVALID_ARGUMENT;const int a=axis-1;
    const bool local=t.sizes[a]==1 && t.periodic[a],low=local || t.lower[a]!=MPI_PROC_NULL,high=local || t.upper[a]!=MPI_PROC_NULL;
    if(!low && !high)return 0;
    CompensationBox box{a,s.upper[a],a==0?1:0,a==2?1:2,0,0};box.n1=s.upper[box.u]+1;box.n2=s.upper[box.v]+1;
    const size_t count=11*static_cast<size_t>(box.n1)*box.n2;
    const auto q=s.fields[ASTR_DEVICE_Q].view,carry=s.fields[ASTR_DEVICE_AIR5_CARRY].view;
    if(normalize) {
        compensation_normalize<<<(count+255)/256,256,0,s.stream>>>(q,carry,box,count,low,high);return astr::finish_launch(s);
    }
    if(local) {
        compensation_mean_local<<<(count+255)/256,256,0,s.stream>>>(q,carry,box,count);return astr::finish_launch(s);
    }
    // Each existing slot holds (hm+1)*max_components values per plane: enough
    // for q11+carry11 when hm>=1 and the transport was prepared for q11.
    if(t.max_components<11)return ASTR_DEVICE_INVALID_ARGUMENT;
    double *device[4]={reinterpret_cast<double *>(t.send_device[0]),reinterpret_cast<double *>(t.send_device[1]),
        reinterpret_cast<double *>(t.recv_device[0]),reinterpret_cast<double *>(t.recv_device[1])};
    double *host[4]={reinterpret_cast<double *>(t.send_host[0]),reinterpret_cast<double *>(t.send_host[1]),
        reinterpret_cast<double *>(t.recv_host[0]),reinterpret_cast<double *>(t.recv_host[1])};
    void *wire[4];for(int i=0;i<4;++i)wire[i]=t.mode==ASTR_HALO_DEVICE?device[i]:host[i];
    const size_t bytes=2*count*sizeof(double);const double started=t.timing?MPI_Wtime():0.;
    compensation_pack<<<(2*count+255)/256,256,0,s.stream>>>(q,carry,box,2*count,device[0],device[1]);
    int status=astr::finish_launch(s);if(status)return status;
    if(t.mode==ASTR_HALO_PINNED)for(int side=0;side<2;++side)if(side==0?low:high) {
        status=static_cast<int>(cudaMemcpyAsync(wire[side],device[side],bytes,cudaMemcpyDeviceToHost,s.stream));if(status)return status;
    }
    // Also retires the previous unpack before MPI can overwrite fixed recv slots.
    status=static_cast<int>(cudaStreamSynchronize(s.stream));if(status)return status;
    const double ready=t.timing?MPI_Wtime():0.;const int tag=42+2*a;
    MPI_Request requests[4]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
    status=MPI_Irecv(wire[2],static_cast<int>(2*count),MPI_DOUBLE,t.lower[a],tag+1,t.communicator,&requests[0]);if(status)return status;
    status=MPI_Irecv(wire[3],static_cast<int>(2*count),MPI_DOUBLE,t.upper[a],tag,t.communicator,&requests[1]);if(status)return status;
    status=MPI_Isend(wire[0],static_cast<int>(2*count),MPI_DOUBLE,t.lower[a],tag,t.communicator,&requests[2]);if(status)return status;
    status=MPI_Isend(wire[1],static_cast<int>(2*count),MPI_DOUBLE,t.upper[a],tag+1,t.communicator,&requests[3]);if(status)return status;
    const double posted=t.timing?MPI_Wtime():0.;status=MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);if(status)return status;
    const double completed=t.timing?MPI_Wtime():0.;
    if(t.mode==ASTR_HALO_PINNED)for(int side=0;side<2;++side)if(side==0?low:high) {
        status=static_cast<int>(cudaMemcpyAsync(device[2+side],wire[2+side],bytes,cudaMemcpyHostToDevice,s.stream));if(status)return status;
    }
    compensation_mean_remote<<<(count+255)/256,256,0,s.stream>>>(carry,box,count,device[0],device[1],device[2],device[3],low,high);
    status=astr::finish_launch(s);
    if(t.timing) {
        t.metrics[0]+=ready-started;t.metrics[1]+=posted-ready;t.metrics[2]+=completed-posted;t.metrics[3]+=MPI_Wtime()-completed;
        const int peers=static_cast<int>(low)+static_cast<int>(high);t.metrics[4]+=peers*static_cast<double>(bytes);
        if(t.mode==ASTR_HALO_PINNED)t.metrics[5]+=2.*peers*bytes;
    }
    return status;
}

extern "C" int astr_device_air5_symmetric_exchange(void *handle,int axis)
{
    if(!handle || axis<1 || axis>3)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.transport)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &t=*s.transport;if(t.active)return ASTR_DEVICE_INVALID_ARGUMENT;
    const int a=axis-1;
    void *device[4];size_t capacity;
    int status=astr_device_air5_symmetric_buffers(handle,device,&capacity);if(status)return status;
    const size_t count=12*static_cast<size_t>(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1);
    if(count>INT_MAX)return ASTR_DEVICE_INVALID_ARGUMENT;
    const bool plus=t.sizes[a]==1?t.periodic[a]!=0:t.upper[a]!=MPI_PROC_NULL;
    const bool minus=t.sizes[a]==1?t.periodic[a]!=0:t.lower[a]!=MPI_PROC_NULL;
    const double started=t.timing?MPI_Wtime():0;
    status=astr_device_air5_symmetric_plane(handle,axis,0,plus,minus);if(status)return status;
    if(t.sizes[a]==1) {
        if(t.periodic[a]) {status=astr_device_air5_symmetric_copy_local_planes(handle,axis);if(status)return status;}
        return astr_device_air5_symmetric_plane(handle,axis,1,plus,minus);
    }
    const size_t bytes=count*sizeof(double);
    if(t.mode==ASTR_HALO_PINNED && !t.air5_face_host[0]) {
        for(int slot=0;slot<4;++slot) {
            status=static_cast<int>(cudaMallocHost(&t.air5_face_host[slot],capacity*sizeof(double)));
            if(status)return status;
        }
    }
    void *wire[4];
    for(int slot=0;slot<4;++slot)wire[slot]=t.mode==ASTR_HALO_DEVICE?device[slot]:t.air5_face_host[slot];
    if(t.mode==ASTR_HALO_PINNED) {
        if(minus) {status=static_cast<int>(cudaMemcpyAsync(wire[0],device[0],bytes,cudaMemcpyDeviceToHost,s.stream));if(status)return status;}
        if(plus) {status=static_cast<int>(cudaMemcpyAsync(wire[1],device[1],bytes,cudaMemcpyDeviceToHost,s.stream));if(status)return status;}
    }
    // MPI is not stream-aware. Also finish the prior unpack before reusing slots.
    status=static_cast<int>(cudaStreamSynchronize(s.stream));if(status)return status;
    const double ready=t.timing?MPI_Wtime():0;
    // This communicator is private to the context. Tags0/1 are the ordinary
    // halo protocol; tags6..11 are the serialized AIR5 owned-face protocol.
    const int tag=6+2*a;
    status=MPI_Sendrecv(wire[0],static_cast<int>(count),MPI_DOUBLE,t.lower[a],tag,
        wire[2],static_cast<int>(count),MPI_DOUBLE,t.upper[a],tag,t.communicator,MPI_STATUS_IGNORE);
    if(status!=MPI_SUCCESS)return status;
    status=MPI_Sendrecv(wire[1],static_cast<int>(count),MPI_DOUBLE,t.upper[a],tag+1,
        wire[3],static_cast<int>(count),MPI_DOUBLE,t.lower[a],tag+1,t.communicator,MPI_STATUS_IGNORE);
    if(status!=MPI_SUCCESS)return status;
    const double received=t.timing?MPI_Wtime():0;
    if(t.mode==ASTR_HALO_PINNED) {
        if(plus) {status=static_cast<int>(cudaMemcpyAsync(device[2],wire[2],bytes,cudaMemcpyHostToDevice,s.stream));if(status)return status;}
        if(minus) {status=static_cast<int>(cudaMemcpyAsync(device[3],wire[3],bytes,cudaMemcpyHostToDevice,s.stream));if(status)return status;}
    }
    status=astr_device_air5_symmetric_plane(handle,axis,1,plus,minus);
    if(t.timing) {
        t.metrics[0]+=ready-started;t.metrics[2]+=received-ready;t.metrics[3]+=MPI_Wtime()-received;
        t.metrics[4]+=(plus+minus)*static_cast<double>(bytes);
        if(t.mode==ASTR_HALO_PINNED)t.metrics[5]+=2.*(plus+minus)*bytes;
    }
    return status;
}

extern "C" int astr_device_air5_flux_transport_prepare(void *handle,int mode)
{
    if(!handle || mode<ASTR_AIR5_FLUX_SEPARATE || mode>ASTR_AIR5_FLUX_PACKED_OVERLAP)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.transport || !s.air5_ready || !s.viscous_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &t=*s.transport;
    if(t.active || t.air5_flux_mode!=-1 || (mode==ASTR_AIR5_FLUX_SEPARATE && t.max_components<6))return ASTR_DEVICE_INVALID_ARGUMENT;
    t.air5_flux_mode=mode;
    t.air5_flux_components=s.fields[ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX].view.data?33:27;
    int distributed=0;
    for(int a=0;a<3;++a)if(t.sizes[a]>1) {
        ++distributed;
        const size_t area=static_cast<size_t>(s.halo)*(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1);
        const size_t count=area*(mode==ASTR_AIR5_FLUX_SEPARATE?6:t.air5_flux_components);
        if(count>INT_MAX)return ASTR_DEVICE_INVALID_ARGUMENT;
        if(mode!=ASTR_AIR5_FLUX_SEPARATE) {
            const int status=allocate_air5_flux_slot(t.air5_flux_slots[a],count,t.mode);if(status)return status;
        }
    }
    if(mode==ASTR_AIR5_FLUX_PACKED_OVERLAP && distributed>1) {
        int status=static_cast<int>(cudaStreamCreateWithFlags(&t.air5_flux_pack_stream,cudaStreamNonBlocking));if(status)return status;
        status=static_cast<int>(cudaEventCreateWithFlags(&t.air5_flux_fields_ready,cudaEventDisableTiming));if(status)return status;
    }
    t.air5_flux_ready=true;return 0;
}

extern "C" int astr_device_air5_flux_exchange(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.transport || !s.transport->air5_flux_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &t=*s.transport;if(t.active)return ASTR_DEVICE_INVALID_ARGUMENT;
    if(t.air5_flux_mode!=ASTR_AIR5_FLUX_SEPARATE)return exchange_air5_flux_group(s,0,t.air5_flux_components);
    const int components[7]={6,3,5,5,5,3,6};int first=0;
    for(int group=0;group<(t.air5_flux_components==33?7:6);++group) {
        const int status=exchange_air5_flux_group(s,first,components[group]);if(status)return status;
        first+=components[group];
    }
    return 0;
}

extern "C" int astr_device_air5_filter_exchange(void *handle,int axis,int phase)
{
    if(!handle || axis<1 || axis>3 || phase<0 || phase>2)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.transport)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &t=*s.transport;if(t.active)return ASTR_DEVICE_INVALID_ARGUMENT;
    void *device[8];size_t capacity,max_area;
    int status=astr_device_air5_filter_buffers(handle,device,&capacity,&max_area);if(status)return status;
    const int a=axis-1;const size_t area=static_cast<size_t>(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1);
    const size_t count=phase==0?22*(s.halo+1)*area:(phase==1?11:22)*area;
    if(count>INT_MAX)return ASTR_DEVICE_INVALID_ARGUMENT;
    const int send[2]={phase==0?0:(phase==1?4:6),1},receive[2]={phase==0?3:(phase==1?5:7),2};
    const int low=t.lower[a],high=t.upper[a],pairs=phase==0?2:1;
    const int destination[2]={phase==1?high:low,high},source[2]={phase==1?low:high,low};
    if(t.sizes[a]==1) {
        if(t.periodic[a])for(int pair=0;pair<pairs;++pair) {
            status=static_cast<int>(cudaMemcpyAsync(device[receive[pair]],device[send[pair]],count*sizeof(double),cudaMemcpyDeviceToDevice,s.stream));if(status)return status;
        }
        return 0;
    }
    if(t.mode==ASTR_HALO_PINNED && !t.air5_filter_host) {
        status=static_cast<int>(cudaMallocHost(&t.air5_filter_host,(4*capacity+66*max_area)*sizeof(double)));if(status)return status;
    }
    void *wire[8];size_t offset=0;
    for(int slot=0;slot<8;++slot) {
        wire[slot]=t.mode==ASTR_HALO_DEVICE?device[slot]:t.air5_filter_host+offset;
        offset+=slot<4?capacity:(slot<6?11:22)*max_area;
    }
    const size_t bytes=count*sizeof(double);const double started=t.timing?MPI_Wtime():0;
    for(int pair=0;t.mode==ASTR_HALO_PINNED && pair<pairs;++pair)if(destination[pair]!=MPI_PROC_NULL) {
        status=static_cast<int>(cudaMemcpyAsync(wire[send[pair]],device[send[pair]],bytes,cudaMemcpyDeviceToHost,s.stream));if(status)return status;
    }
    status=static_cast<int>(cudaStreamSynchronize(s.stream));if(status)return status;
    const double ready=t.timing?MPI_Wtime():0;const int tag=30+4*a+(phase==0?0:phase+1);
    for(int pair=0;pair<pairs;++pair) {
        status=MPI_Sendrecv(wire[send[pair]],static_cast<int>(count),MPI_DOUBLE,destination[pair],tag+pair,
            wire[receive[pair]],static_cast<int>(count),MPI_DOUBLE,source[pair],tag+pair,t.communicator,MPI_STATUS_IGNORE);if(status)return status;
    }
    const double received=t.timing?MPI_Wtime():0;
    for(int pair=0;t.mode==ASTR_HALO_PINNED && pair<pairs;++pair)if(source[pair]!=MPI_PROC_NULL) {
        status=static_cast<int>(cudaMemcpyAsync(device[receive[pair]],wire[receive[pair]],bytes,cudaMemcpyHostToDevice,s.stream));if(status)return status;
    }
    if(t.timing) {
        t.metrics[0]+=ready-started;t.metrics[2]+=received-ready;t.metrics[3]+=MPI_Wtime()-received;
        for(int pair=0;pair<pairs;++pair) {
            if(destination[pair]!=MPI_PROC_NULL)t.metrics[4]+=bytes;
            if(t.mode==ASTR_HALO_PINNED)t.metrics[5]+=((destination[pair]!=MPI_PROC_NULL)+(source[pair]!=MPI_PROC_NULL))*static_cast<double>(bytes);
        }
    }
    return 0;
}

extern "C" int astr_device_air5_filter_apply(void *handle,int preserve,int *physics_status)
{
    if(!handle || !physics_status || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.transport || !s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &t=*s.transport;if(t.active)return ASTR_DEVICE_INVALID_ARGUMENT;*physics_status=0;
    auto release_status=[&]() {
        int local;int status=astr_device_air5_check(handle,&local);if(status)return status;
        if(t.communicator==MPI_COMM_NULL) {*physics_status=local;return 0;}
        return MPI_Allreduce(&local,physics_status,1,MPI_INT,MPI_MAX,t.communicator);
    };
    for(int a=0;a<3;++a) {
        const int axis=a+1;
        const bool low=t.sizes[a]==1?t.periodic[a]!=0:t.lower[a]!=MPI_PROC_NULL;
        const bool high=t.sizes[a]==1?t.periodic[a]!=0:t.upper[a]!=MPI_PROC_NULL;
        int status=astr_device_air5_filter_pack(handle,axis);if(status)return status;
        status=astr_device_air5_filter_exchange(handle,axis,0);if(status)return status;
        status=astr_device_air5_filter_faces(handle,axis,high);if(status)return status;
        status=release_status();if(status || *physics_status)return status;
        status=astr_device_air5_filter_exchange(handle,axis,1);if(status)return status;
        status=astr_device_air5_filter_commit(handle,axis,low,high);if(status)return status;
        status=astr_device_air5_filter_alias(handle,axis,0);if(status)return status;
        status=astr_device_air5_filter_exchange(handle,axis,2);if(status)return status;
        if(high) {status=astr_device_air5_filter_alias(handle,axis,1);if(status)return status;}
        status=astr_device_air5_filter_check(handle,preserve);if(status)return status;
        status=release_status();if(status || *physics_status)return status;
    }
    return 0;
}

extern "C" int astr_device_air5_legacy_filter_apply(void *handle,const double floor[5],int wall_axis,int family,int *physics_status)
{
    if(!handle || !floor || !physics_status || wall_axis<-1 || wall_axis>2 || family<0 || family>1)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const auto &work=s.fields[ASTR_DEVICE_FILTER_WORK].view;
    if(!s.transport || !s.air5_ready || !work.data)return ASTR_DEVICE_NOT_ALLOCATED;
    const auto &t=*s.transport;
    if(t.active || s.air5_filter)return ASTR_DEVICE_INVALID_ARGUMENT;
    *physics_status=0;
    auto linear=[&]() {
        auto apply_axis=[&](int axis,int component) {
            const int lower=!t.periodic[axis] && t.lower[axis]==MPI_PROC_NULL;
            const int upper=!t.periodic[axis] && t.upper[axis]==MPI_PROC_NULL;
            return axis==wall_axis
                ? astr_device_air5_legacy_linear_wall(handle,axis,component,lower,upper,family)
                : astr_device_air5_legacy_linear_axis(handle,axis,component,lower,upper);
        };
        const int first=work.extent[3]==1 ? 1 : 0,last=first ? 11 : 0;
        int status=astr_device_exchange_axis(handle,ASTR_DEVICE_Q,0,0);if(status)return status;
        for(int c=first;c<=last;++c) {
            status=apply_axis(0,c);if(status)return status;
            status=t.sizes[1]==1 && !t.periodic[1] ? astr_device_air5_legacy_linear_y_ghosts(handle,c)
                : astr_device_exchange_axis(handle,ASTR_DEVICE_FILTER_WORK,1,0);
            if(status)return status;
            status=apply_axis(1,c);if(status)return status;
        }
        // Components are independent. Exchange the updated q z-faces once,
        // not the old and new components separately for each scalar pass.
        status=astr_device_exchange_axis(handle,ASTR_DEVICE_Q,2,0);if(status)return status;
        for(int c=first;c<=last;++c) {
            status=apply_axis(2,c);if(status)return status;
            status=astr_device_air5_legacy_linear_commit(handle,c);if(status)return status;
        }
        return 0;
    };
    const bool compensated=s.fields[ASTR_DEVICE_AIR5_CARRY].view.data!=nullptr;
    int status=0;
    if(compensated) {
        status=astr_device_air5_legacy_filter_compensation(handle,0);if(status)return status;
        status=astr_device_air5_legacy_filter_compensation(handle,1);if(status)return status;
        status=linear();if(status)return status;
        status=astr_device_air5_legacy_filter_compensation(handle,1);if(status)return status;
    }
    status=astr_device_air5_legacy_filter_save(handle);if(status)return status;
    status=linear();if(status)return status;
    if(compensated) {
        status=astr_device_air5_legacy_filter_compensation(handle,0);if(status)return status;
    }
    status=astr_device_air5_legacy_filter_repair(handle,floor);if(status)return status;
    int local=0;
    status=astr_device_air5_check(handle,&local);if(status)return status;
    if(t.communicator==MPI_COMM_NULL) {*physics_status=local;return 0;}
    return MPI_Allreduce(&local,physics_status,1,MPI_INT,MPI_MAX,t.communicator);
}

extern "C" int astr_device_shock_mask_exchange(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.transport || !s.shock_mask) return ASTR_DEVICE_NOT_ALLOCATED;
    astr::HaloTransport &t=*s.transport;
    if(t.active) return ASTR_DEVICE_INVALID_ARGUMENT;
    s.shock_faces_ready=false;
    if(!s.shock_faces) {
        const cudaError_t status=cudaMalloc(&s.shock_faces,astr::shock_faces_size(s.upper));
        if(status!=cudaSuccess) return static_cast<int>(status);
    }
    MaskBox box{};
    for(int a=0;a<3;++a) box.upper[a]=s.upper[a];
    for(int a=0;a<3;++a) {
        const int n1=s.upper[(a+1)%3]+1;
        const size_t count=static_cast<size_t>(n1)*(s.upper[(a+2)%3]+1);
        int8_t *recv[2]={reinterpret_cast<int8_t *>(t.recv_device[0]),reinterpret_cast<int8_t *>(t.recv_device[1])};
        int lower_rule=0,upper_rule=0;
        const double started=t.timing ? MPI_Wtime() : 0;
        if(t.sizes[a]==1) {
            if(t.periodic[a]) lower_rule=upper_rule=2;
        } else {
            int8_t *send[2]={reinterpret_cast<int8_t *>(t.send_device[0]),reinterpret_cast<int8_t *>(t.send_device[1])};
            pack_mask<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.shock_mask,box,a,n1,count,send[0],send[1]);
            int status=astr::finish_launch(s);
            if(status!=0) return status;
            for(int side=0;t.mode==ASTR_HALO_PINNED && side<2;++side) {
                if((side==0 ? t.lower[a] : t.upper[a])==MPI_PROC_NULL) continue;
                const cudaError_t err=cudaMemcpyAsync(t.send_host[side],send[side],count,cudaMemcpyDeviceToHost,s.stream);
                if(err!=cudaSuccess) return static_cast<int>(err);
            }
            const cudaError_t ready=cudaStreamSynchronize(s.stream);
            if(ready!=cudaSuccess) return static_cast<int>(ready);
            const double posted=t.timing ? MPI_Wtime() : 0;
            if(t.timing) t.metrics[0]+=posted-started;
            void *low_send=t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(send[0]) : t.send_host[0];
            void *high_send=t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(send[1]) : t.send_host[1];
            void *low_recv=t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(recv[0]) : t.recv_host[0];
            void *high_recv=t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(recv[1]) : t.recv_host[1];
            // Reuse the private communicator only while the generic slots are idle.
            status=MPI_Sendrecv(low_send,static_cast<int>(count),MPI_BYTE,t.lower[a],0,
                high_recv,static_cast<int>(count),MPI_BYTE,t.upper[a],0,t.communicator,MPI_STATUS_IGNORE);
            if(status!=MPI_SUCCESS) return status;
            status=MPI_Sendrecv(high_send,static_cast<int>(count),MPI_BYTE,t.upper[a],1,
                low_recv,static_cast<int>(count),MPI_BYTE,t.lower[a],1,t.communicator,MPI_STATUS_IGNORE);
            if(status!=MPI_SUCCESS) return status;
            const double received=t.timing ? MPI_Wtime() : 0;
            if(t.timing) {
                t.metrics[2]+=received-posted;
                const int peers=(t.lower[a]!=MPI_PROC_NULL)+(t.upper[a]!=MPI_PROC_NULL);
                t.metrics[4]+=peers*static_cast<double>(count);
                if(t.mode==ASTR_HALO_PINNED) t.metrics[5]+=2.0*peers*count;
            }
            for(int side=0;t.mode==ASTR_HALO_PINNED && side<2;++side) {
                if((side==0 ? t.lower[a] : t.upper[a])==MPI_PROC_NULL) continue;
                const cudaError_t err=cudaMemcpyAsync(recv[side],t.recv_host[side],count,cudaMemcpyHostToDevice,s.stream);
                if(err!=cudaSuccess) return static_cast<int>(err);
            }
            lower_rule=t.lower[a]==MPI_PROC_NULL ? 0 : 1;
            upper_rule=t.upper[a]==MPI_PROC_NULL ? 0 : 1;
            if(t.timing) t.metrics[3]+=MPI_Wtime()-received;
        }
        unpack_mask<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.shock_mask,s.shock_faces,box,a,n1,count,
            recv[0],recv[1],lower_rule,upper_rule);
        const int status=astr::finish_launch(s);
        if(status!=0) return status;
    }
    s.shock_faces_ready=true;
    return 0;
}

extern "C" int astr_device_crashfix_history_exchange(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.transport || !s.critical_mask) return ASTR_DEVICE_NOT_ALLOCATED;
    astr::HaloTransport &t=*s.transport;
    if(t.active) return ASTR_DEVICE_INVALID_ARGUMENT;
    MaskBox box{};for(int a=0;a<3;++a) box.upper[a]=s.upper[a];
    for(int phase=0;phase<2;++phase) for(int a=0;a<3;++a) {
        if(t.sizes[a]==1 && !t.periodic[a]) continue;
        const bool shared=phase==0;
        const int width=shared ? 1 : s.halo,n1=s.upper[(a+1)%3]+1;
        const size_t count=static_cast<size_t>(width)*n1*(s.upper[(a+2)%3]+1);
        if(t.sizes[a]==1) {
            critical_local<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.critical_mask,s.fields[0].view,
                box,a,width,n1,count,shared);
            const int status=astr::finish_launch(s);if(status!=0) return status;
            continue;
        }
        int8_t *send[2]={reinterpret_cast<int8_t *>(t.send_device[0]),reinterpret_cast<int8_t *>(t.send_device[1])};
        int8_t *recv[2]={reinterpret_cast<int8_t *>(t.recv_device[0]),reinterpret_cast<int8_t *>(t.recv_device[1])};
        critical_pack<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.critical_mask,s.fields[0].view,
            box,a,width,n1,count,shared,send[0],send[1]);
        int status=astr::finish_launch(s);if(status!=0) return status;
        {
            const double started=t.timing ? MPI_Wtime() : 0;
            for(int side=0;t.mode==ASTR_HALO_PINNED && side<2;++side) {
                if((side==0 ? t.lower[a] : t.upper[a])==MPI_PROC_NULL) continue;
                const cudaError_t err=cudaMemcpyAsync(t.send_host[side],send[side],count,cudaMemcpyDeviceToHost,s.stream);
                if(err!=cudaSuccess) return static_cast<int>(err);
            }
            const cudaError_t ready=cudaStreamSynchronize(s.stream);
            if(ready!=cudaSuccess) return static_cast<int>(ready);
            const double posted=t.timing ? MPI_Wtime() : 0;
            if(t.timing) t.metrics[0]+=posted-started;
            void *sends[2]={t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(send[0]) : t.send_host[0],
                t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(send[1]) : t.send_host[1]};
            void *recvs[2]={t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(recv[0]) : t.recv_host[0],
                t.mode==ASTR_HALO_DEVICE ? static_cast<void *>(recv[1]) : t.recv_host[1]};
            status=MPI_Sendrecv(sends[0],static_cast<int>(count),MPI_BYTE,t.lower[a],0,
                recvs[1],static_cast<int>(count),MPI_BYTE,t.upper[a],0,t.communicator,MPI_STATUS_IGNORE);
            if(status!=MPI_SUCCESS) return status;
            status=MPI_Sendrecv(sends[1],static_cast<int>(count),MPI_BYTE,t.upper[a],1,
                recvs[0],static_cast<int>(count),MPI_BYTE,t.lower[a],1,t.communicator,MPI_STATUS_IGNORE);
            if(status!=MPI_SUCCESS) return status;
            const double received=t.timing ? MPI_Wtime() : 0;
            if(t.timing) {
                const int peers=(t.lower[a]!=MPI_PROC_NULL)+(t.upper[a]!=MPI_PROC_NULL);
                t.metrics[2]+=received-posted;t.metrics[4]+=peers*static_cast<double>(count);
                if(t.mode==ASTR_HALO_PINNED) t.metrics[5]+=2.0*peers*count;
            }
            for(int side=0;t.mode==ASTR_HALO_PINNED && side<2;++side) {
                if((side==0 ? t.lower[a] : t.upper[a])==MPI_PROC_NULL) continue;
                const cudaError_t err=cudaMemcpyAsync(recv[side],t.recv_host[side],count,cudaMemcpyHostToDevice,s.stream);
                if(err!=cudaSuccess) return static_cast<int>(err);
            }
            if(t.timing) t.metrics[3]+=MPI_Wtime()-received;
        }
        critical_unpack<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.critical_mask,s.fields[0].view,
            box,a,width,n1,count,shared,recv[0],recv[1],t.lower[a]!=MPI_PROC_NULL,t.upper[a]!=MPI_PROC_NULL);
        status=astr::finish_launch(s);if(status!=0) return status;
    }
    return 0;
}

extern "C" int astr_device_transport_metrics(void *handle,double values[6],int reset)
{
    if (!handle || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::HaloTransport *t=static_cast<astr::State *>(handle)->transport;
    if (!t) return ASTR_DEVICE_NOT_ALLOCATED;
    if (t->active) return ASTR_DEVICE_INVALID_ARGUMENT;
    for (int n=0;n<6;++n) { values[n]=t->metrics[n];if (reset) t->metrics[n]=0; }
    return 0;
}

extern "C" int astr_device_transport_axis_metrics(void *handle,double values[3][5][7])
{
    if (!handle || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *t=static_cast<astr::State *>(handle)->transport;
    if (!t) return ASTR_DEVICE_NOT_ALLOCATED;
    if (t->active || !t->axis_timing) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int axis=0;axis<3;++axis)for(int group=0;group<5;++group)for(int i=0;i<7;++i) {
        values[axis][group][i]=t->axis_metrics[axis][group][i];
        t->axis_metrics[axis][group][i]=0.;
    }
    return 0;
}
