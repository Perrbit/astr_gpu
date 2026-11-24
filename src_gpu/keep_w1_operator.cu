#include "device_state_internal.cuh"
#include "halo_transport.h"
#include "keep_w1_operator.h"
#define ASTR_W1_FUNCTION __device__ __forceinline__
#include "keep_w1_solver_math.h"
#include "keep_w1_tables.h"

using namespace astr::w1;

namespace {
constexpr int threads=128;

__global__ void build_flux(const TensorTable *table,SolverState state,SolverFlux flux,int count)
{
    const int id=blockIdx.x*threads+threadIdx.x;
    if(id>=count) return;
    const int nx=state.grid.n[0],ny=state.grid.n[1];
    const int i=id%nx,j=(id/nx)%ny,k=id/(nx*ny);
    flux.store(tensor_offset(state.grid,i,j,k),tensor_flux(*table,state.grid,state,i,j,k));
}

template<typename Samples>
__global__ void add_rhs(const TensorTable *table,SolverState state,SolverFlux flux,astr_numerical_real *rhs,int count,
    Samples samples)
{
    const int id=blockIdx.x*threads+threadIdx.x;
    if(id<count) accumulate_w1(*table,state,flux,rhs,id,count,samples);
}
}

extern "C" int astr_keep_w1_prepare(void *handle)
{
    auto &parent=*static_cast<astr::State *>(handle);
    auto status=cudaMalloc(&parent.keep_w1_table,sizeof(TensorTable));
    if(status!=cudaSuccess) return static_cast<int>(status);
    status=cudaMemcpyAsync(parent.keep_w1_table,&solver_table,sizeof(TensorTable),cudaMemcpyHostToDevice,parent.stream);
    if(status!=cudaSuccess) return static_cast<int>(status);
    return astr::allocate_field(parent.fields[ASTR_DEVICE_KEEP_W1_FLUX],parent.upper,parent.halo,4,14);
}

extern "C" int astr_keep_w1_diffusion(void *handle,int first_y,int total_y,const int *split,
    const double *inv,const double *wall,double re,double pr,double c5,
    double tc,double tc1,const int *send_width,const int *recv_width)
{
    auto &parent=*static_cast<astr::State *>(handle);
    const auto &field=parent.fields[ASTR_DEVICE_KEEP_W1_FLUX].view;
    if(!field.data) return ASTR_DEVICE_NOT_ALLOCATED;
    const TensorGrid grid{{parent.upper[0]+1,parent.upper[1]+1,parent.upper[2]+1},
                          {inv[0],inv[1],inv[2]},{wall[0],wall[1]},true,
                          parent.halo,first_y,total_y,{split[0]!=0,split[1]!=0,split[2]!=0}};
    const SolverState state{grid,
        static_cast<const astr_numerical_real *>(parent.fields[ASTR_DEVICE_VELOCITY].view.data),
        static_cast<const astr_numerical_real *>(parent.fields[ASTR_DEVICE_TEMPERATURE].view.data),
        static_cast<const astr_numerical_real *>(parent.fields[ASTR_DEVICE_METRIC].view.data),
        static_cast<const astr_numerical_real *>(parent.fields[ASTR_DEVICE_JACOBIAN].view.data),re,pr,c5,tc,tc1};
    const int count=grid.n[0]*grid.n[1]*grid.n[2],blocks=(count+threads-1)/threads;
    const auto *table=static_cast<const TensorTable *>(parent.keep_w1_table);
    const SolverFlux work{static_cast<astr_numerical_real *>(field.data),static_cast<int>(field.stride[3])};
    build_flux<<<blocks,threads,0,parent.stream>>>(table,state,work,count);
    auto status=astr::finish_launch(parent);
    if(status!=0) return status;
    for(int axis=0;axis<3;++axis) if(grid.split[axis]) {
        status=astr_device_exchange_axis(handle,ASTR_DEVICE_KEEP_W1_FLUX,axis,0);
        if(status!=0) return status;
    }
    auto *rhs=static_cast<astr_numerical_real *>(parent.fields[ASTR_DEVICE_RHS].view.data);
    const void *received[2]{};
    if(send_width[0] || send_width[1] || recv_width[0] || recv_width[1]) {
        status=astr::exchange_keep_w1_ends(parent,send_width,recv_width,received);
        if(status!=0) return status;
    }
    if(recv_width[0] || recv_width[1]) {
        const EndSamples samples{static_cast<const astr_numerical_real *>(received[0]),
            static_cast<const astr_numerical_real *>(received[1]),recv_width[0],recv_width[1],total_y,grid.n[0],grid.n[2]};
        add_rhs<<<blocks,threads,0,parent.stream>>>(table,state,work,rhs,count,samples);
    } else {
        add_rhs<<<blocks,threads,0,parent.stream>>>(table,state,work,rhs,count,NoEndSamples{});
    }
    return astr::finish_launch(parent);
}
