#include <new>
#include "device_state_internal.cuh"
#include "chemistry_coupling.h"
#include "chemistry_ros2_math.cuh"

namespace astr {
struct Air5ChemistryPlan {
    AstrAir5ChemistryOptions options{};
    AstrAir5ChemistryReport *report=nullptr;
    int64_t total=0;
};
int release_air5_chemistry(Air5ChemistryPlan *plan)
{
    if(!plan) return 0;
    const int status=static_cast<int>(cudaFree(plan->report));
    delete plan;
    return status;
}
}
namespace {
using astr::at;
using astr::State;
struct ChemistryBox {int first[3],size[3];size_t count;};

__global__ void begin_kernel(AstrAir5ChemistryReport *report)
{
    *report=AstrAir5ChemistryReport{};
    for(int i=0;i<3;++i) report->minimum_bits[i]=__double_as_longlong(DBL_MAX);
}
__device__ void record_failure(AstrAir5ChemistryReport *report,int i,int j,int k,int status,int phase,
    const double q[11],const double carry[6],bool compensated)
{
    atomicMax(&report->status,status);
    if(atomicCAS(report->failure,0,1)!=0) return;
    report->failure[1]=i;report->failure[2]=j;report->failure[3]=k;
    report->failure[4]=status;report->failure[5]=phase;
    for(int c=0;c<11;++c) report->failure_q[c]=q[c];
    for(int c=0;c<6;++c) report->failure_carry[c]=compensated?carry[c]:0.;
}
__device__ void minimum_bits(int64_t *address,double value)
{
    auto *word=reinterpret_cast<unsigned long long *>(address);
    unsigned long long old=*word;
    const long long incoming=__double_as_longlong(value);
    while(incoming<static_cast<long long>(old)) {
        const unsigned long long previous=old;
        old=atomicCAS(word,previous,static_cast<unsigned long long>(incoming));
        if(old==previous) break;
    }
}
__device__ void maximum_bits(int64_t *address,double value)
{
    auto *word=reinterpret_cast<unsigned long long *>(address);
    unsigned long long old=*word;
    const long long incoming=__double_as_longlong(value);
    while(incoming>static_cast<long long>(old)) {
        const unsigned long long previous=old;
        old=atomicCAS(word,previous,static_cast<unsigned long long>(incoming));
        if(old==previous) break;
    }
}

template<int diagnostics>
__global__ void half_step_kernel(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView vel,AstrDeviceView t,
    AstrDeviceView tv,AstrDeviceView p,AstrDeviceView y,AstrDeviceView carry,AstrDeviceView history,
    ChemistryBox box,double duration,AstrAir5ChemistryOptions options,bool collect_flag,AstrAir5ChemistryReport *report)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=box.count) return;
    const int i=id%box.size[0]+box.first[0],j=(id/box.size[0])%box.size[1]+box.first[1];
    const int k=id/(static_cast<size_t>(box.size[0])*box.size[1])+box.first[2];
    double local_q[11],input[6],final_state[6],trial_carry[6],original_carry[6],atol[6],suggested,trial_history=0;
    for(int c=0;c<11;++c) local_q[c]=at(q,i,j,k,c+1);
    for(int c=0;c<6;++c) {
        input[c]=local_q[5+c];
        original_carry[c]=options.compensated?at(carry,i,j,k,6+c):0.;trial_carry[c]=original_carry[c];
        atol[c]=options.atol_factor*(c<5?local_q[0]:fmax(fabs(input[5]),1.));
    }
    if(options.history_enabled) trial_history=at(history,i,j,k);
    astr::air5::Ros2Counts counts;bool skipped;
    int status=astr::air5::ros2_advance(local_q[0],local_q+1,local_q[4],input,duration,duration,
        options.rtol,atol,200000,final_state,suggested,counts,options.source_mode,options.source_only!=0,
        options.compensated?trial_carry:nullptr,options.history_enabled?&trial_history:nullptr,&options.activity_threshold,&skipped);
    if(status!=0) {
        record_failure(report,i,j,k,status,1,local_q,original_carry,options.compensated!=0);
        return;
    }
    for(int c=0;c<6;++c) local_q[5+c]=final_state[c];
    double density,velocity[3],temperature,vibration,pressure,fractions[5];
    status=astr::air5::conservative_to_primitive(local_q,options.preserve_all!=0,density,velocity,
                                               temperature,fractions,vibration,pressure);
    if(status!=0) {
        // Keep the captured failure state at the input phase, as in the original kernel.
        for(int c=0;c<11;++c) local_q[c]=at(q,i,j,k,c+1);
        record_failure(report,i,j,k,status,2,local_q,original_carry,options.compensated!=0);
        return;
    }
    for(int c=0;c<6;++c) {
        at(q,i,j,k,6+c)=final_state[c];
        if(options.compensated) at(carry,i,j,k,6+c)=trial_carry[c];
    }
    if(options.history_enabled) at(history,i,j,k)=trial_history;
    at(rho,i,j,k)=density;at(t,i,j,k)=temperature;at(tv,i,j,k)=vibration;at(p,i,j,k)=pressure;
    for(int c=0;c<3;++c) at(vel,i,j,k,c+1)=velocity[c];
    for(int c=0;c<5;++c) at(y,i,j,k,c+1)=fractions[c];
    const bool collect=diagnostics<0?collect_flag:diagnostics!=0;
    if(collect) {
        const int values[4]={counts.accepted,counts.rejected,counts.rhs,counts.jacobian};
        for(int c=0;c<4;++c) atomicMax(report->max_counts+c,values[c]);
        double minimum=final_state[0];for(int c=1;c<5;++c) minimum=fmin(minimum,final_state[c]);
        minimum_bits(report->minimum_bits,minimum);minimum_bits(report->minimum_bits+1,temperature);
        minimum_bits(report->minimum_bits+2,vibration);
        maximum_bits(report->maximum_bits,temperature);maximum_bits(report->maximum_bits+1,vibration);
        if(skipped) atomicAdd(reinterpret_cast<unsigned long long *>(&report->skipped),1ULL);
    }
}
}

extern "C" int astr_device_air5_chemistry_prepare(void *handle,const AstrAir5ChemistryOptions *options)
{
    if(!handle || !options || options->block_size<=0) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.air5_ready || s.air5_chemistry) return ASTR_DEVICE_INVALID_ARGUMENT;
    s.air5_chemistry=new(std::nothrow) astr::Air5ChemistryPlan{};
    if(!s.air5_chemistry) return static_cast<int>(cudaErrorMemoryAllocation);
    auto &plan=*s.air5_chemistry;plan.options=*options;
    int status=static_cast<int>(cudaMalloc(&plan.report,sizeof(AstrAir5ChemistryReport)));
    if(status!=0) return status;
    if(options->compensated) {
        status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_CARRY],s.upper,0,4,11);
        if(status!=0) return status;
        status=static_cast<int>(cudaMemsetAsync(s.fields[ASTR_DEVICE_AIR5_CARRY].view.data,0,
                                               s.fields[ASTR_DEVICE_AIR5_CARRY].bytes,s.stream));
        if(status!=0) return status;
    }
    if(options->history_enabled) {
        status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_HISTORY],s.upper,0,3);
        if(status!=0) return status;
        status=static_cast<int>(cudaMemsetAsync(s.fields[ASTR_DEVICE_AIR5_HISTORY].view.data,0,
                                               s.fields[ASTR_DEVICE_AIR5_HISTORY].bytes,s.stream));
        if(status!=0) return status;
    }
    return astr_device_air5_chemistry_begin(handle);
}
extern "C" int astr_device_air5_chemistry_begin(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.air5_chemistry) return ASTR_DEVICE_NOT_ALLOCATED;
    s.air5_chemistry->total=0;
    begin_kernel<<<1,1,0,s.stream>>>(s.air5_chemistry->report);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_chemistry_region(void *handle,const int first[3],const int last[3],double duration,int collect)
{
    if(!handle || !first || !last) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.air5_chemistry) return ASTR_DEVICE_NOT_ALLOCATED;
    ChemistryBox box{};box.count=1;
    for(int d=0;d<3;++d) {
        if(first[d]<0 || last[d]>s.upper[d] || first[d]>last[d]) return ASTR_DEVICE_INVALID_ARGUMENT;
        box.first[d]=first[d];box.size[d]=last[d]-first[d]+1;box.count*=box.size[d];
    }
    auto &plan=*s.air5_chemistry;plan.total+=box.count;
    const dim3 block(plan.options.block_size),grid((box.count+block.x-1)/block.x);
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view,rho=s.fields[ASTR_DEVICE_RHO].view,
        vel=s.fields[ASTR_DEVICE_VELOCITY].view,t=s.fields[ASTR_DEVICE_TEMPERATURE].view,
        tv=s.fields[ASTR_DEVICE_AIR5_TV].view,p=s.fields[ASTR_DEVICE_PRESSURE].view,
        y=s.fields[ASTR_DEVICE_AIR5_SPECIES].view,carry=s.fields[ASTR_DEVICE_AIR5_CARRY].view,
        history=s.fields[ASTR_DEVICE_AIR5_HISTORY].view;
    if(!plan.options.specialized_dispatch)
        half_step_kernel<-1><<<grid,block,0,s.stream>>>(q,rho,vel,t,tv,p,y,carry,history,box,duration,plan.options,collect!=0,plan.report);
    else if(collect)
        half_step_kernel<1><<<grid,block,0,s.stream>>>(q,rho,vel,t,tv,p,y,carry,history,box,duration,plan.options,true,plan.report);
    else
        half_step_kernel<0><<<grid,block,0,s.stream>>>(q,rho,vel,t,tv,p,y,carry,history,box,duration,plan.options,false,plan.report);
    // Preserve workspace allocation flags across the ordered chemistry stage.
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_chemistry_status(void *handle,int *status)
{
    if(!handle || !status) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.air5_chemistry) return ASTR_DEVICE_NOT_ALLOCATED;
    const int result=static_cast<int>(cudaMemcpyAsync(status,&s.air5_chemistry->report->status,sizeof(int),
                                                      cudaMemcpyDeviceToHost,s.stream));
    return result!=0?result:static_cast<int>(cudaStreamSynchronize(s.stream));
}
extern "C" int astr_device_air5_chemistry_report(void *handle,AstrAir5ChemistryReport *report)
{
    if(!handle || !report) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.air5_chemistry) return ASTR_DEVICE_NOT_ALLOCATED;
    int status=static_cast<int>(cudaMemcpyAsync(report,s.air5_chemistry->report,sizeof(*report),cudaMemcpyDeviceToHost,s.stream));
    if(status!=0) return status;
    status=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) report->total=s.air5_chemistry->total;
    return status;
}
