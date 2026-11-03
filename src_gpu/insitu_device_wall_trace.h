#ifndef ASTR_INSITU_DEVICE_WALL_TRACE_H
#define ASTR_INSITU_DEVICE_WALL_TRACE_H

#include "insitu_device_streamlines.h"
#include <viskores/cont/ArrayHandleConstant.h>

namespace astr_insitu {
struct SampleWallTraceShear : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,ExecObject,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  template<class Grid>
  VISKORES_EXEC void operator()(const TraceVertex& point,const Grid& grid,
      NativeVector& shear,viskores::Int32& status) const {
    const Real p[3]={point[0],point[1],point[2]};Real value[3]{};
    const auto sampled=grid.sample(p,value);
    status=sampled!=WallSample::valid && sampled!=WallSample::zero;
    shear=NativeVector(value[0],value[1],value[2]);
  }
};

struct DropWallSingletons : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldInOut);
  using ExecutionSignature=void(_1);
  VISKORES_EXEC void operator()(viskores::Int32& count) const {
    if(count==1) count=0; // A terminal seed is metadata, not a drawable line cell.
  }
};

struct WallTracePartition {
  NativeVector lower,upper;
  viskores::Vec<bool,3> last;
  bool active;
};

// The caller supplies only the small, globally ordered physical seed packet.
// All field/interpolation/line arrays stay on the device. The shared driver
// owns continuation and geometry; its internal vector slot contains wall shear.
template<class Stages=DirectTraceStage>
inline DeviceStreamlines trace_wall_device(const WallLengthGrid* grid,
    const std::vector<NativeVector>& seeds,const WallTracePartition& partition,
    const TraceConfiguration& parameters,MPI_Comm parent,Stages stages={},bool image_only=true,bool retain_metadata=false) {
  parameters.validate();
  if(seeds.empty() || seeds.size()>std::size_t(std::numeric_limits<int>::max()/2))
    throw std::invalid_argument("Wall seed count is empty or exceeds particle identity range");
  if(partition.active!=(grid!=nullptr)) throw std::invalid_argument("Wall partition/field presence differs");
  const int seed_count=int(seeds.size()),particles=seed_count*(parameters.direction==0?2:1);
  PrivateTraceComm private_comm(parent);
  ParticleMessageType particle_type;
  try {
    const auto comm=private_comm.value;
    int ranks=0;trace_mpi(MPI_Comm_size(comm,&ranks));
    std::array<Real,7> local_bounds{partition.lower[0],partition.upper[0],partition.lower[2],partition.upper[2],
      Real(partition.last[0]),Real(partition.last[2]),Real(partition.active)};
    auto bounds=stages("wall-ownership",[&] {return std::vector<Real>(7*ranks);});
    trace_mpi(MPI_Allgather(local_bounds.data(),7,numerical_mpi_type(),bounds.data(),7,numerical_mpi_type(),comm));
    DeviceStreamlines result;
    stages("wall-seeds",[&] {
      result.minimum_step=parameters.minimum_step;result.maximum_length=parameters.maximum_length;
      result.seed_count=seed_count;result.seed_layout="wall-index-list";result.particles=particles;
      result.final_state.resize(particles);result.directions.resize(particles);result.subminimum_stop.resize(particles,false);
      for(int i=0;i<particles;++i) {
        const auto& seed=seeds[i%seed_count];
        const int sign=parameters.direction<0?-1:(i<seed_count?1:-1);
        result.directions[i]=sign;
        result.final_state[i]=ParticleState(seed[0],seed[1],seed[2],Real(sign)*parameters.initial_step,
          Real(0),Real(0),0,TraceActive);
      }
    });
    const auto locate=[&](DeviceStreamlines& output,std::vector<int>& owners) {
      for(int i=0;i<particles;++i) {
        const auto& state=output.final_state[i];
        if(state.status!=TraceActive && state.status!=TraceTransfer) continue;
        int owner=-1;
        for(int r=0;r<ranks;++r) {
          const auto* b=bounds.data()+7*r;
          if(b[6]==Real(0)) continue;
          const bool inside=state[0]>=b[0] && (state[0]<b[1] || (b[4]!=Real(0) && state[0]==b[1])) &&
            state[2]>=b[2] && (state[2]<b[3] || (b[5]!=Real(0) && state[2]==b[3]));
          if(!inside) continue;
          if(owner>=0) throw CollectiveTraceError("Wall particle has duplicate cell owners");
          owner=r;
        }
        if(owner<0) throw CollectiveTraceError("Wall particle has no physical cell owner");
        owners[i]=owner;
      }
    };
    const auto device=viskores::cont::DeviceAdapterTagCuda{};
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(device);
    if(grid) grid->RequireDeviceOnly();
    viskores::cont::Invoker invoke(device);
    const auto advance=[&](const auto& input,auto& geometry,auto& output,auto& counts,int chunk) {
      if(!grid) {
        viskores::cont::Algorithm::Copy(device,input,output);
        viskores::cont::Algorithm::Copy(device,viskores::cont::ArrayHandleConstant<viskores::Int32>(0,particles),counts);
        synchronize_device_stage("EmptyWallTracePartition");
        return;
      }
      invoke(RK45WallTraceWorklet(chunk,parameters.minimum_step,parameters.maximum_step,parameters.tolerance,
        parameters.maximum_length,parameters.maximum_steps,partition.lower,partition.upper,partition.last),
        input,*grid,geometry,output,counts);
      synchronize_device_stage("RK45WallTrace");
      invoke(DropWallSingletons{},counts);
      synchronize_device_stage("DropWallTraceSingletons");
    };
    const auto sample=[&](const auto& points,bool,auto& values,auto& status) {
      if(!points.GetNumberOfValues()) {values.Allocate(0);status.Allocate(0);return;}
      if(!grid) throw std::logic_error("Empty wall partition produced nonempty geometry");
      invoke(SampleWallTraceShear{},points,*grid,values,status);
    };
    const auto terminal=[](const TraceState& state,bool& subminimum) {
      subminimum=false;
      switch(state.status) {
        case TraceActive:case TraceTransfer:case TraceLength:case TraceSteps:
        case WallZero:case WallStageZero:case WallAccuracyLimited:case WallPhysicalExit:case WallRegionExit:return true;
        default:return false; // Nonfinite and missing interpolation support are product failures.
      }
    };
    const TraceDriverSettings settings{parameters,std::numeric_limits<std::uint64_t>::max(),true,true,
      image_only,false,retain_metadata};
    drive_device_trace(result,comm,particle_type.value,settings,locate,advance,sample,
      [&] {if(grid) grid->RequireDeviceOnly();},terminal,stages);
    return result;
  } catch(const CollectiveTraceError&) {
    particle_type.close();private_comm.close();throw;
  }
}
} // namespace astr_insitu
#endif
