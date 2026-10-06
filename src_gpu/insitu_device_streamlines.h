#ifndef ASTR_INSITU_DEVICE_STREAMLINES_H
#define ASTR_INSITU_DEVICE_STREAMLINES_H

#include "insitu_streamline_worklet.h"
#include <nvtx3/nvToolsExt.h>
#include "insitu_device_array.h"
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/filter/flow/worklet/Field.h>
#include <viskores/filter/flow/worklet/GridEvaluators.h>
#include <mpi.h>
#include <array>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include <exception>

namespace astr_insitu {
using TraceVertex=viskores::Vec<double,5>;

struct PackComponentHalo : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id nodes;
  VISKORES_CONT explicit PackComponentHalo(viskores::Id n):nodes(n) {}
  template <class Portal>
  VISKORES_EXEC void operator()(viskores::Id i,const Portal& input,viskores::Vec3f& v) const {
    for(int d=0;d<3;++d) v[d]=input.Get(i+d*nodes);
  }
};

struct ValidTraceVertex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id capacity;
  VISKORES_CONT explicit ValidTraceVertex(viskores::Id n):capacity(n) {}
  template <class Portal>
  VISKORES_EXEC void operator()(viskores::Id i,const Portal& counts,viskores::UInt8& valid) const {
    valid=(i%capacity)<counts.Get(i/capacity);
  }
};

struct SampleTraceVelocity : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,ExecObject,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  template <class Grid>
  VISKORES_EXEC void operator()(const TraceVertex& point,const Grid& grid,
      viskores::Vec3f& velocity,viskores::Int32& status) const {
    viskores::VecVariable<viskores::Vec3f,2> values;
    status=grid.Evaluate(viskores::Vec3f(point[0],point[1],point[2]),0.,values).CheckFail();
    velocity=status?viskores::Vec3f(NAN):values[0];
  }
};

struct CheckTraceCount : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  viskores::Id capacity;
  VISKORES_CONT explicit CheckTraceCount(viskores::Id n):capacity(n) {}
  VISKORES_EXEC void operator()(viskores::Int32 n,viskores::Id& bad) const {
    bad=n<0 || n>capacity;
  }
};
struct TraceParticleId : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  viskores::Id capacity;
  VISKORES_CONT explicit TraceParticleId(viskores::Id n):capacity(n) {}
  VISKORES_EXEC void operator()(viskores::Id i,viskores::Id& particle) const {particle=i/capacity;}
};
struct ResidentLineIndex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5);
  viskores::Id chunk,base;
  VISKORES_CONT ResidentLineIndex(viskores::Id n,viskores::Id b):chunk(n),base(b) {}
  template<class Counts,class Offsets>
  VISKORES_EXEC void operator()(viskores::Id i,const Counts& counts,const Offsets& offsets,
      viskores::Id& index,viskores::UInt8& valid) const {
    const auto particle=i/(2*chunk),edge=(i/2)%chunk;
    valid=edge+1<counts.Get(particle);
    index=valid?base+offsets.Get(particle)+edge+i%2:0;
  }
};
struct PackResidentTrace : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldIn,FieldOut,FieldOut,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,_7);
  VISKORES_EXEC void operator()(const TraceVertex& p,const viskores::Vec3f& v,
      viskores::Int32 status,viskores::Vec3f& xyz,double& u,double& q,viskores::Id& bad) const {
    xyz=viskores::Vec3f(p[0],p[1],p[2]);u=v[0];q=0.;bad=status!=0;
    for(int d=0;d<5;++d) bad+=!viskores::IsFinite(p[d]);
    for(int d=0;d<3;++d) bad+=!viskores::IsFinite(v[d]);
  }
};
struct ResidentTraceChunk {
  viskores::cont::ArrayHandle<TraceVertex> accepted;
  viskores::cont::ArrayHandle<viskores::Vec3f> xyz,velocity;
  viskores::cont::ArrayHandle<double> u,q;
  viskores::cont::ArrayHandle<viskores::Id> indices,particle;
};

template<class T>
inline void append_device_chunk(const viskores::cont::ArrayHandle<T>& source,
    viskores::cont::ArrayHandle<T>& destination,viskores::Id offset) {
  if(source.GetNumberOfValues() && !viskores::cont::Algorithm::CopySubRange(
      viskores::cont::DeviceAdapterTagCuda{},source,0,source.GetNumberOfValues(),destination,offset))
    throw std::runtime_error("Resident trajectory concatenation failed");
}

struct DeviceTraceSegment {
  int seed=0,direction=1;
  std::vector<TraceVertex> points;
  std::vector<viskores::Vec3f> velocity;
  std::vector<viskores::Vec3f> integrating_velocity;
};
struct DeviceStreamlines {
  std::vector<DeviceTraceSegment> segments;
  std::array<RK45TraceWorklet::State,32> final_state;
  std::array<bool,32> subminimum_stop{};
  int rounds=0,transfers=0,particles=32;
  std::uint64_t geometry_bytes=0;
  double minimum_step=.01*(2.*std::acos(-1.)/32.);
  viskores::cont::DataSet resident_geometry;
  std::uint64_t control_read_bytes=0;
};

inline void trace_mpi(int status) {
  if(status!=MPI_SUCCESS) throw std::runtime_error("Private streamline MPI failed");
}

struct PrivateTraceComm {
  MPI_Comm value=MPI_COMM_NULL;
  explicit PrivateTraceComm(MPI_Comm parent) {trace_mpi(MPI_Comm_dup(parent,&value));}
  ~PrivateTraceComm() {
    // The caller aborts MPI on exceptions; never wait in a destructor before it.
    if(value!=MPI_COMM_NULL && std::uncaught_exceptions()==0 && MPI_Comm_free(&value)!=MPI_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD,1);
  }
  PrivateTraceComm(const PrivateTraceComm&)=delete;
  PrivateTraceComm& operator=(const PrivateTraceComm&)=delete;
};

inline viskores::cont::ArrayHandle<viskores::Vec3f> pack_component_halo(
    double* pointer,const viskores::Id3& dimensions) {
  const auto nodes=dimensions[0]*dimensions[1]*dimensions[2];
  auto input=borrow_cuda_array(pointer,3*nodes);
  viskores::cont::ArrayHandle<viskores::Vec3f> output;
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  invoke(PackComponentHalo(nodes),viskores::cont::ArrayHandleIndex(nodes),input,output);
  synchronize_device_stage("PackComponentHalo");
  require_device_only(input); require_device_only(output);
  return output;
}

// Inputs are interleaved device halos including three private layers. Only
// small particle states and GPU-compacted final trajectories reach the host.
inline DeviceStreamlines trace_tgv_device(
    const viskores::cont::ArrayHandle<viskores::Vec3f>& trace_velocity,
    const viskores::cont::ArrayHandle<viskores::Vec3f>& color_velocity,
    const viskores::Id3& extent,const viskores::Id3& offset,MPI_Comm comm,
    std::uint64_t host_budget,bool constant=false,int constant_axis=0,bool forward_only=false,
    bool sample_trace_vector=false,int global_cells=32,bool resident=false) {
  constexpr int chunk=128;
  const int particles=forward_only?16:32;
  using State=RK45TraceWorklet::State;
  if(global_cells!=32 && global_cells!=256)
    throw std::invalid_argument("Unsupported bounded TGV streamline resolution");
  const double pi=std::acos(-1.),h=2.*pi/global_cells;
  PrivateTraceComm private_comm(comm);
  comm=private_comm.value;
  int rank=0,ranks=0;
  trace_mpi(MPI_Comm_rank(comm,&rank)); trace_mpi(MPI_Comm_size(comm,&ranks));
  if(ranks<1 || ranks>2 || constant_axis<0 || constant_axis>2 || host_budget<1024*1024)
    throw std::invalid_argument("Invalid bounded device streamline configuration");
  const viskores::Id3 dims(extent[0]+7,extent[1]+7,extent[2]+7);
  for(int d=0;d<3;++d)
    if(extent[d]<1 || offset[d]<0 || extent[d]+offset[d]>global_cells)
      throw std::invalid_argument("Invalid TGV streamline partition");
  if(trace_velocity.GetNumberOfValues()!=dims[0]*dims[1]*dims[2] ||
     color_velocity.GetNumberOfValues()!=trace_velocity.GetNumberOfValues())
    throw std::invalid_argument("Device streamline halo size differs");
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  auto grid=viskores::cont::DataSetBuilderUniform().Create(dims,
    viskores::Vec3f((offset[0]-3)*h,(offset[1]-3)*h,(offset[2]-3)*h),viskores::Vec3f(h));
  using Field=viskores::worklet::flow::VelocityField<viskores::cont::ArrayHandle<viskores::Vec3f>>;
  using Grid=viskores::worklet::flow::GridEvaluator<Field>;
  LengthGrid<Grid> evaluate(Grid(grid,Field(trace_velocity)),viskores::Vec3f(0.),viskores::Vec3f(2.*pi));
  Grid color_grid(grid,Field(color_velocity));
  Grid trace_grid(grid,Field(trace_velocity));
  viskores::Vec3f lower(offset[0]*h,offset[1]*h,offset[2]*h);
  viskores::Vec3f upper((offset[0]+extent[0])*h,(offset[1]+extent[1])*h,(offset[2]+extent[2])*h);
  viskores::Vec<bool,3> last;
  double bounds[6],all_bounds[12];
  for(int d=0;d<3;++d) {bounds[d]=lower[d];bounds[d+3]=upper[d];last[d]=offset[d]+extent[d]==global_cells;}
  trace_mpi(MPI_Allgather(bounds,6,MPI_DOUBLE,all_bounds,6,MPI_DOUBLE,comm));
  auto owner=[&](const State& s) {
    for(int r=0;r<ranks;++r) {
      bool inside=true;
      for(int d=0;d<3;++d) inside&=s[d]>=all_bounds[6*r+d] &&
        (s[d]<all_bounds[6*r+d+3] || (all_bounds[6*r+d+3]==2.*pi && s[d]==2.*pi));
      if(inside) return r;
    }
    throw std::runtime_error("Accepted trajectory has no physical-domain owner");
  };
  DeviceStreamlines result;
  result.minimum_step=.01*h;
  result.particles=particles;
  auto& state=result.final_state;
  for(auto& particle:state) particle=State(0.,0.,0.,0.,0.,0.,0.,TraceLength);
  for(int i=0;i<particles;++i) {
    viskores::Vec3f seed(pi/2.,pi/8.+(i%16)*(6.*pi/8.)/15.,pi/4.);
    if(constant && constant_axis!=0) {
      const auto original=seed;
      for(int d=0;d<3;++d) seed[d]=original[(d-constant_axis+3)%3];
    }
    // Both directions share the approved sixteen seeds, not reversed seed sets.
    state[i]=State(seed[0],seed[1],seed[2],(i<16?.1:-.1)*h,0.,0.,0.,TraceActive);
  }
  std::array<State,32> local;
  std::array<double,32*8> wire;
  std::vector<double> incoming(particles*8*ranks);
  std::vector<ResidentTraceChunk> chunks;
  viskores::Id resident_points=0,resident_indices=0;
  for(;result.rounds<1000;++result.rounds) {
    int active=0;
    for(int i=0;i<particles;++i) {
      local[i]=state[i];
      const bool running=state[i][7]==TraceActive || state[i][7]==TraceTransfer;
      active+=running;
      local[i][7]=running && owner(state[i])==rank?TraceActive:TraceTransfer;
    }
    if(!active) break;
    auto input=viskores::cont::make_ArrayHandle(local.data(),particles,viskores::CopyFlag::On);
    viskores::cont::ArrayHandle<State> output;
    viskores::cont::ArrayHandle<viskores::Int32> counts;
    viskores::cont::ArrayHandle<TraceVertex> geometry,compact;
    viskores::cont::ArrayHandle<viskores::UInt8> valid;
    geometry.Allocate(particles*(chunk+1));
    nvtxRangePushA("ASTR_IS8_DEVICE_TRACE_AND_COMPACT");
    invoke(RK45TraceWorklet(chunk,.01*h,.5*h,1e-8,pi,100000,lower,upper,last),
      input,evaluate,geometry,output,counts);
    synchronize_device_stage("RK45Trace");
    invoke(ValidTraceVertex(chunk+1),viskores::cont::ArrayHandleIndex(particles*(chunk+1)),counts,valid);
    synchronize_device_stage("ValidTraceVertex");
    viskores::cont::Algorithm::CopyIf(geometry,valid,compact);
    synchronize_device_stage("CompactTrace");
    viskores::cont::ArrayHandle<viskores::Vec3f> colors;
    viskores::cont::ArrayHandle<viskores::Int32> sample_status;
    invoke(SampleTraceVelocity{},compact,color_grid,colors,sample_status);
    synchronize_device_stage("SampleTraceVelocity");
    viskores::cont::ArrayHandle<viskores::Vec3f> trace_vectors;
    viskores::cont::ArrayHandle<viskores::Int32> trace_status;
    if(sample_trace_vector) {
      invoke(SampleTraceVelocity{},compact,trace_grid,trace_vectors,trace_status);
      synchronize_device_stage("SampleIntegratingVelocity");
    }
    require_device_only(trace_velocity); require_device_only(color_velocity);
    nvtxRangePop();
    if(resident) {
      nvtxRangePushA("ASTR_IS8_RESIDENT_TRACE_PACK");
      viskores::cont::ArrayHandle<viskores::Id> bad;
      invoke(CheckTraceCount(chunk+1),counts,bad);
      synchronize_device_stage("CheckTraceCount");
      if(viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},bad,viskores::Id(0)))
        throw std::runtime_error("Invalid resident trajectory count");
      viskores::cont::ArrayHandle<viskores::Int32> offsets;
      const auto total=viskores::cont::Algorithm::ScanExclusive(viskores::cont::DeviceAdapterTagCuda{},counts,offsets);
      if(total!=compact.GetNumberOfValues()) throw std::runtime_error("Resident trajectory compaction mismatch");
      ResidentTraceChunk data;
      data.accepted=compact;
      data.velocity=sample_trace_vector?trace_vectors:colors;
      invoke(PackResidentTrace{},compact,data.velocity,sample_trace_vector?trace_status:sample_status,
        data.xyz,data.u,data.q,bad);
      synchronize_device_stage("PackResidentTrace");
      if(viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},bad,viskores::Id(0)))
        throw std::runtime_error("Nonfinite or out-of-halo resident trajectory");
      viskores::cont::ArrayHandle<viskores::Id> raw_indices,raw_particle;
      viskores::cont::ArrayHandle<viskores::UInt8> valid_indices;
      invoke(ResidentLineIndex(chunk,resident_points),viskores::cont::ArrayHandleIndex(particles*chunk*2),
        counts,offsets,raw_indices,valid_indices);
      synchronize_device_stage("ResidentLineIndex");
      invoke(TraceParticleId(chunk+1),viskores::cont::ArrayHandleIndex(particles*(chunk+1)),raw_particle);
      synchronize_device_stage("TraceParticleId");
      viskores::cont::Algorithm::CopyIf(viskores::cont::DeviceAdapterTagCuda{},raw_indices,valid_indices,data.indices);
      synchronize_device_stage("CompactResidentLineIndex");
      viskores::cont::Algorithm::CopyIf(viskores::cont::DeviceAdapterTagCuda{},raw_particle,valid,data.particle);
      synchronize_device_stage("CompactResidentParticleId");
      require_device_only(data.xyz);require_device_only(data.velocity);require_device_only(data.indices);
      resident_points+=data.xyz.GetNumberOfValues();resident_indices+=data.indices.GetNumberOfValues();
      result.geometry_bytes+=data.xyz.GetNumberOfValues()*(sizeof(TraceVertex)+sizeof(viskores::Vec3f)*2+
        2*sizeof(double)+sizeof(viskores::Id))+data.indices.GetNumberOfValues()*sizeof(viskores::Id);
      chunks.push_back(std::move(data));
      nvtxRangePop();
      nvtxRangePushA("ASTR_IS8_TRACE_CONTROL_READ");
      const auto states=output.ReadPortal();
      for(int i=0;i<particles;++i) for(int d=0;d<8;++d) wire[i*8+d]=states.Get(i)[d];
      result.control_read_bytes+=particles*sizeof(State);
      nvtxRangePop();
    } else {
    nvtxRangePushA("ASTR_IS8_COMPACT_TRACE_READ");
    const auto sizes=counts.ReadPortal(); const auto states=output.ReadPortal();
    const auto points=compact.ReadPortal(); const auto vectors=colors.ReadPortal();
    const auto sampled=sample_status.ReadPortal();
    const auto integrator_vectors=trace_vectors.ReadPortal();
    const auto integrator_status=trace_status.ReadPortal();
    viskores::Id position=0;
    for(int i=0;i<particles;++i) {
      const auto n=sizes.Get(i);
      if(n<0 || n>chunk+1) throw std::runtime_error("Invalid compact trajectory size");
      for(int d=0;d<8;++d) wire[i*8+d]=states.Get(i)[d];
      if(n) {
        const std::uint64_t bytes=static_cast<std::uint64_t>(n)*(sample_trace_vector?11:8)*sizeof(double);
        if(bytes>host_budget-result.geometry_bytes)
          throw std::runtime_error("Device trajectory compact geometry exceeds host budget");
        DeviceTraceSegment segment;
        segment.seed=i%16; segment.direction=i<16?1:-1;
        segment.points.reserve(n); segment.velocity.reserve(n);
        for(int j=0;j<n;++j,++position) {
          if(sampled.Get(position)) throw std::runtime_error("Final trajectory color outside halo");
          const auto point=points.Get(position);
          const auto velocity=vectors.Get(position);
          for(int d=0;d<5;++d) if(!std::isfinite(point[d])) throw std::runtime_error("Nonfinite trajectory vertex");
          for(int d=0;d<3;++d) if(!std::isfinite(velocity[d])) throw std::runtime_error("Nonfinite trajectory color");
          segment.points.push_back(point);segment.velocity.push_back(velocity);
          if(sample_trace_vector) {
            const auto vector=integrator_vectors.Get(position);
            if(integrator_status.Get(position)) throw std::runtime_error("Final integrating velocity outside halo");
            for(int d=0;d<3;++d) if(!std::isfinite(vector[d])) throw std::runtime_error("Nonfinite integrating velocity");
            segment.integrating_velocity.push_back(vector);
          }
        }
        result.geometry_bytes+=bytes;result.segments.push_back(std::move(segment));
      }
    }
    if(position!=compact.GetNumberOfValues()) throw std::runtime_error("Trajectory compaction mismatch");
    nvtxRangePop();
    }
    trace_mpi(MPI_Allgather(wire.data(),particles*8,MPI_DOUBLE,incoming.data(),particles*8,MPI_DOUBLE,comm));
    for(int i=0;i<particles;++i) {
      const bool running=state[i][7]==TraceActive || state[i][7]==TraceTransfer;
      if(!running) continue;
      const int previous_owner=owner(state[i]);
      for(int d=0;d<8;++d) state[i][d]=incoming[previous_owner*particles*8+i*8+d];
      const int status=static_cast<int>(state[i][7]);
      const TraceState terminal{{state[i][0],state[i][1],state[i][2]},state[i][3],state[i][4],state[i][5],
        static_cast<int>(state[i][6]),status};
      result.subminimum_stop[i]=trace_subminimum_stop(terminal,pi,.01*h);
      if(status==TraceTransfer && rank==previous_owner) ++result.transfers;
      if(status!=TraceActive && status!=TraceTransfer && status!=TraceLength && status!=Outside &&
         !result.subminimum_stop[i])
        throw std::runtime_error("Device trajectory unexpected termination status="+std::to_string(status)+
          " remaining="+std::to_string(pi-state[i][4]));
    }
  }
  if(result.rounds==1000) throw std::runtime_error("Device trajectory collective round limit exceeded");
  if(resident && resident_points) {
    nvtxRangePushA("ASTR_IS8_RESIDENT_TRACE_CONCATENATE");
    ResidentTraceChunk all;
    all.xyz.Allocate(resident_points);all.velocity.Allocate(resident_points);
    all.accepted.Allocate(resident_points);all.u.Allocate(resident_points);all.q.Allocate(resident_points);
    all.particle.Allocate(resident_points);all.indices.Allocate(resident_indices);
    viskores::Id point_base=0,index_base=0;
    for(const auto& data:chunks) {
      append_device_chunk(data.xyz,all.xyz,point_base);append_device_chunk(data.velocity,all.velocity,point_base);
      append_device_chunk(data.accepted,all.accepted,point_base);append_device_chunk(data.u,all.u,point_base);
      append_device_chunk(data.q,all.q,point_base);append_device_chunk(data.particle,all.particle,point_base);
      append_device_chunk(data.indices,all.indices,index_base);
      point_base+=data.xyz.GetNumberOfValues();index_base+=data.indices.GetNumberOfValues();
    }
    synchronize_device_stage("ConcatenateResidentTrace");
    viskores::cont::CellSetSingleType<> cells;
    cells.Fill(resident_points,viskores::CELL_SHAPE_LINE,2,all.indices);
    result.resident_geometry.SetCellSet(cells);
    result.resident_geometry.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",all.xyz));
    result.resident_geometry.AddPointField("velocity",all.velocity);
    result.resident_geometry.AddPointField("u",all.u);result.resident_geometry.AddPointField("Q_rs",all.q);
    result.resident_geometry.AddPointField("particle",all.particle);
    result.resident_geometry.AddPointField("accepted",all.accepted);
    require_device_only(all.xyz);require_device_only(all.velocity);require_device_only(all.indices);
    nvtxRangePop();
  }
  return result;
}
} // namespace astr_insitu
#endif
