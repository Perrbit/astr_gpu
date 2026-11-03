#define ASTR_WALL_TRACE_NO_MAIN
#include "insitu_wall_trace_probe.cpp"
#include "insitu_device_wall_trace.h"
#include <viskores/cont/Initialize.h>
#include <sstream>

namespace {
struct DeviceFields {
  std::vector<R*> pointers;
  ~DeviceFields() {for(auto* pointer:pointers) cudaFree(pointer);}
  viskores::cont::ArrayHandle<R> upload(const std::vector<R>& values) {
    R* pointer=nullptr;
    if(cudaMalloc(&pointer,values.size()*sizeof(R))!=cudaSuccess)
      throw std::runtime_error("Cannot allocate wall probe field");
    pointers.push_back(pointer);
    if(cudaMemcpy(pointer,values.data(),values.size()*sizeof(R),cudaMemcpyHostToDevice)!=cudaSuccess)
      throw std::runtime_error("Cannot upload wall probe field");
    return astr_insitu::borrow_cuda_array(pointer,values.size());
  }
};

int state_owner(const Input& in,const astr_insitu::ParticleState& state,const int* topology) {
  int index[2]{};
  for(int d=0;d<2;++d) {
    const R* axis=d==0?in.x:in.z;
    const int partitions=topology[d==0?0:2];
    index[d]=-1;
    for(int p=0;p<partitions;++p) {
      const R lower=axis[p*16/partitions],upper=axis[(p+1)*16/partitions];
      const R coordinate=state[d==0?0:2];
      if(coordinate>=lower && (coordinate<upper || (p+1==partitions && coordinate==upper))) index[d]=p;
    }
    if(index[d]<0) throw std::runtime_error("Accepted wall particle has no rectangular owner");
  }
  return index[0]+topology[0]*topology[1]*index[1]; // Only the bottom-y rank owns wall cells.
}

void check_shared_case(const Input& in,int mode,const astr_insitu::WallLengthGrid* grid,
    const astr_insitu::WallTracePartition& partition,R& worst,R& arc_offset,int& transfers) {
  using namespace astr_insitu;
  TraceConfiguration settings{std::abs(in.initial),in.minimum,in.maximum,in.tolerance,in.target,in.maximum_steps,0};
  std::vector<Result> reference(2);
  for(int direction=0;direction<2;++direction) {
    Input signed_input=in;signed_input.initial=direction==0?settings.initial_step:-settings.initial_step;
    run_one(signed_input,reference[direction]);
  }
  DeviceStreamlines result;
  try {
    result=trace_wall_device(grid,{NativeVector(in.seed[0],in.seed[1],in.seed[2])},partition,
      settings,MPI_COMM_WORLD,DirectTraceStage{},false,true);
  } catch(const CollectiveTraceError& error) {
    const std::string expected="status="+std::to_string(mode==12?int(WallSupportExit):int(Nonfinite))+" ";
    if((mode==12 || mode==14) && std::string(error.what()).find(expected)!=std::string::npos) return;
    throw;
  }
  if(mode==12 || mode==14) throw std::runtime_error("Invalid wall support/field was accepted by the shared driver");
  if(result.seed_count!=1 || result.particles!=2 || result.directions!=std::vector<int>{1,-1} ||
     result.control_read_bytes!=std::uint64_t(result.rounds)*2*sizeof(ParticleState))
    throw std::runtime_error("Wall shared-driver seed/control identity differs");
  std::vector<R> local_steps;
  R error=R(0);
  for(int i=0;i<2;++i) {
    const auto& state=result.final_state[i];const auto& cpu=reference[i].state;
    if(state.status!=cpu.status || state.steps!=cpu.steps) {
      std::fprintf(stderr,"Shared wall termination case=%d particle=%d GPU=%d/%d CPU=%d/%d\n",
        mode,i,state.status,state.steps,cpu.status,cpu.steps);
      throw std::runtime_error("Wall shared-driver state differs from CPU");
    }
    for(int d=0;d<3;++d) error=std::fmax(error,std::abs(state[d]-cpu.point[d]));
    error=std::fmax(error,std::abs(state[4]-cpu.length));
  }
  if(result.resident_geometry.GetNumberOfCoordinateSystems()) {
    auto points=result.resident_geometry.GetPointField("accepted").GetData().AsArrayHandle<
      viskores::cont::ArrayHandle<TraceVertex>>();
    auto ids=result.resident_geometry.GetPointField("particle").GetData().AsArrayHandle<
      viskores::cont::ArrayHandle<viskores::Id>>();
    require_device_only(points);require_device_only(ids);
    // This is the explicitly marked test oracle, never the production rendering route.
    const auto values=points.ReadPortal();const auto particles=ids.ReadPortal();
    for(viskores::Id i=0;i<points.GetNumberOfValues();++i) {
      const auto id=particles.Get(i);
      const auto point=values.Get(i);
      if(id<0 || id>1) throw std::runtime_error("Invalid shared wall seed identity");
      const auto& cpu=reference[id];
      int closest=0;
      for(int j=1;j<cpu.count;++j)
        if(std::abs(cpu.points[j][3]-point[3])<std::abs(cpu.points[closest][3]-point[3])) closest=j;
      arc_offset=std::fmax(arc_offset,std::abs(point[3]-cpu.points[closest][3]));
      const R sign=R(result.directions[id]),arc=point[3];
      R exact[3]={in.seed[0],in.seed[1],in.seed[2]};
      if(mode==7 || mode==17) {
        exact[0]=R(.5)+R(.25)*std::cos(sign*arc/R(.25));
        exact[2]=R(.5)+R(.25)*std::sin(sign*arc/R(.25));
      } else {
        const R dx=mode==2 || mode==3?R(0):(mode==4 || mode==5 || mode==16?R(.6):(mode==1?R(-1):R(1)));
        const R dz=mode==2?R(1):(mode==3?R(-1):(mode==4 || mode==5 || mode==16?R(.8):R(0)));
        exact[0]+=sign*dx*arc;exact[2]+=sign*dz*arc;
      }
      for(int d=0;d<3;++d) {
        if(!std::isfinite(point[d])) throw std::runtime_error("Invalid shared wall point");
        error=std::fmax(error,std::abs(point[d]-exact[d]));
      }
      if(point[4]!=R(0)) {
        if(i==0 || particles.Get(i-1)!=id) throw std::runtime_error("Wall accepted step lost its segment predecessor");
        const auto previous=values.Get(i-1);
        const R before[3]={previous[0],previous[1],previous[2]};R next[3],actual=R(0),estimate=R(0);
        WallSample failure=WallSample::valid;
        const auto full=grid_for(in);
        if(cash_karp_step(WallStageEvaluator<Grid>{full,&failure},before,next,R(0),point[4],actual,estimate)!=StepOk ||
           estimate>settings.tolerance) throw std::runtime_error("CPU rejected the GPU's actual accepted wall step");
        for(int d=0;d<3;++d) error=std::fmax(error,std::abs(point[d]-next[d]));
        error=std::fmax(error,std::abs(point[3]-previous[3]-std::abs(actual)));
        local_steps.insert(local_steps.end(),{R(id),point[3],std::abs(actual)});
      }
    }
  }
  int ranks=0;trace_mpi(MPI_Comm_size(MPI_COMM_WORLD,&ranks));
  const int local_count=int(local_steps.size());std::vector<int> counts(ranks),offsets(ranks);
  trace_mpi(MPI_Allgather(&local_count,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD));
  int total=0;for(int r=0;r<ranks;++r) {offsets[r]=total;total+=counts[r];}
  std::vector<R> global_steps(total);
  trace_mpi(MPI_Allgatherv(local_steps.data(),local_count,numerical_mpi_type(),global_steps.data(),counts.data(),
    offsets.data(),numerical_mpi_type(),MPI_COMM_WORLD));
  for(int id=0;id<2;++id) {
    std::vector<std::array<R,2>> steps;
    for(int i=0;i<total;i+=3) if(global_steps[i]==R(id)) steps.push_back({global_steps[i+1],global_steps[i+2]});
    std::sort(steps.begin(),steps.end());
    if(int(steps.size())!=result.final_state[id].steps) throw std::runtime_error("Wall geometry lost or repeated accepted steps");
    R previous=R(0);
    for(const auto& step:steps) {
      if(!(step[0]>previous)) throw std::runtime_error("Wall geometry repeats an accepted arc identity");
      error=std::fmax(error,std::abs(step[0]-previous-step[1]));previous=step[0];
    }
    error=std::fmax(error,std::abs(previous-result.final_state[id][4]));
  }
  if(!std::isfinite(error) || error>R(2e-10)) throw std::runtime_error("Shared wall geometry differs from CPU case="+std::to_string(mode));
  worst=std::fmax(worst,error);transfers+=result.transfers;
}

void check_case(int mode,const int* topology,int rank,int ranks,R& worst,R& arc_offset,int& transfers,bool shared) {
  using namespace astr_insitu;
  Input in;initialize(in,mode);
  Result cpu{};run_one(in,cpu);
  R analytic_error=R(0),case_error=R(0);
  if(!analytic(in,cpu,mode,analytic_error)) throw std::runtime_error("Invalid CPU wall oracle case="+std::to_string(mode));
  const int ix=rank%topology[0],iy=(rank/topology[0])%topology[1],iz=rank/(topology[0]*topology[1]);
  const int offset_x=ix*16/topology[0],offset_z=iz*16/topology[2];
  const int end_x=(ix+1)*16/topology[0],end_z=(iz+1)*16/topology[2];
  const int begin_x=std::max(0,offset_x-3),begin_z=std::max(0,offset_z-3);
  const int finish_x=std::min(16,end_x+3),finish_z=std::min(16,end_z+3);
  std::vector<R> hx(in.x+begin_x,in.x+finish_x+1),hz(in.z+begin_z,in.z+finish_z+1);
  std::vector<R> ux,uz;
  for(int k=begin_z;k<=finish_z;++k) for(int i=begin_x;i<=finish_x;++i) {
    ux.push_back(in.tx[k*nodes+i]);uz.push_back(in.tz[k*nodes+i]);
  }
  DeviceFields allocations;
  viskores::cont::ArrayHandle<R> x,z,tx,tz;
  std::unique_ptr<WallLengthGrid> grid;
  if(iy==0) {
    x=allocations.upload(hx);z=allocations.upload(hz);tx=allocations.upload(ux);tz=allocations.upload(uz);
    grid=std::make_unique<WallLengthGrid>(x,z,tx,tz,in.seed[1],
      std::array<R,4>{R(0),R(1),R(0),R(1)},std::array<R,4>{R(0),in.region_upper,R(0),R(1)});
  }
  if(shared) {
    const WallTracePartition partition{NativeVector(in.x[offset_x],in.seed[1],in.z[offset_z]),
      NativeVector(in.x[end_x],in.seed[1],in.z[end_z]),
      viskores::Vec<bool,3>(ix+1==topology[0],true,iz+1==topology[2]),iy==0};
    check_shared_case(in,mode,grid.get(),partition,worst,arc_offset,transfers);
    if(grid) grid->RequireDeviceOnly();
    return;
  }
  ParticleMessageType datatype;
  ParticleState state(in.seed[0],in.seed[1],in.seed[2],in.initial,R(0),R(0),0,TraceActive);
  std::vector<ParticleState> received(ranks);
  std::vector<int> local_hits(cpu.count,0),global_hits(cpu.count,0);
  const auto device=viskores::cont::DeviceAdapterTagCuda{};
  viskores::cont::Invoker invoke(device);
  while(state.status==TraceActive || state.status==TraceTransfer) {
    const auto previous=state;
    const int owner=state_owner(in,state,topology);
    auto wire=state;wire.status=TraceTransfer;
    if(owner==rank) {
      if(!grid) throw std::runtime_error("An empty upper-y rank owns a wall particle");
      wire.status=TraceActive;
      auto input=viskores::cont::make_ArrayHandle(&wire,1,viskores::CopyFlag::On);
      viskores::cont::ArrayHandle<ParticleState> output;
      viskores::cont::ArrayHandle<viskores::Int32> counts;
      viskores::cont::ArrayHandle<TraceVertex> geometry,compact;
      viskores::cont::ArrayHandle<viskores::UInt8> valid;
      constexpr int chunk=32;
      geometry.Allocate(chunk+1);
      invoke(RK45WallTraceWorklet(chunk,in.minimum,in.maximum,in.tolerance,in.target,in.maximum_steps,
        NativeVector(in.x[offset_x],in.seed[1],in.z[offset_z]),
        NativeVector(in.x[end_x],in.seed[1],in.z[end_z]),
        viskores::Vec<bool,3>(ix+1==topology[0],true,iz+1==topology[2])),
        input,*grid,geometry,output,counts);
      synchronize_device_stage("WallTraceComponent");
      invoke(ValidTraceVertex(chunk+1),viskores::cont::ArrayHandleIndex(chunk+1),counts,valid);
      synchronize_device_stage("WallTraceComponentValidity");
      viskores::cont::Algorithm::CopyIf(device,geometry,valid,compact);
      synchronize_device_stage("WallTraceComponentCompaction");
      require_device_only(x);require_device_only(z);require_device_only(tx);require_device_only(tz);
      require_device_only(compact);
      // Explicit small analytical oracle only. Production geometry may not use this readback.
      const auto values=compact.ReadPortal();
      for(viskores::Id i=0;i<compact.GetNumberOfValues();++i) {
        const int index=previous.steps+int(i);
        if(index>=cpu.count) throw std::runtime_error("GPU accepted more wall points than the CPU oracle");
        ++local_hits[index];const auto value=values.Get(i);
        for(int d=0;d<5;++d) {
          if(!std::isfinite(value[d])) throw std::runtime_error("Invalid wall geometry oracle value");
          // A chunk's seam point is not a newly accepted RK step.
          if(i==0 && d==4) {
            if(value[d]!=R(0)) throw std::runtime_error("Wall segment starts with a fabricated accepted step");
            continue;
          }
          case_error=std::fmax(case_error,std::abs(value[d]-cpu.points[index][d]));
        }
      }
      wire=output.ReadPortal().Get(0);
      transfers+=wire.status==TraceTransfer;
      require_device_only(x);require_device_only(z);require_device_only(tx);require_device_only(tz);
    }
    trace_mpi(MPI_Allgather(&wire,1,datatype.value,received.data(),1,datatype.value,MPI_COMM_WORLD));
    state=received[owner];
    if(state.steps==previous.steps && (state.status==TraceActive || state.status==TraceTransfer))
      throw std::runtime_error("Wall component routing made no progress");
  }
  if(state.status!=cpu.state.status || state.steps!=cpu.state.steps)
    throw std::runtime_error("Wall component terminal status or step count differs from CPU case="+std::to_string(mode)+
      " status="+std::to_string(state.status)+" cpu_status="+std::to_string(cpu.state.status)+
      " steps="+std::to_string(state.steps)+" cpu_steps="+std::to_string(cpu.state.steps));
  for(int d=0;d<3;++d) case_error=std::fmax(case_error,std::abs(state[d]-cpu.state.point[d]));
  case_error=std::fmax(case_error,std::abs(state[4]-cpu.state.length));
  if(case_error>R(2e-10)) {
    std::fprintf(stderr,"Wall trajectory differs case=%d rank=%d maxabs=%.17g\n",mode,rank,double(case_error));
    throw std::runtime_error("Wall component same-discrete gate failed");
  }
  worst=std::fmax(worst,std::fmax(case_error,analytic_error));
  trace_mpi(MPI_Allreduce(local_hits.data(),global_hits.data(),cpu.count,MPI_INT,MPI_SUM,MPI_COMM_WORLD));
  for(int count:global_hits) if(count<1) throw std::runtime_error("Missing accepted wall point at an MPI seam");
}
}

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank=0,ranks=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  try {
    int topology[3]={1,1,1};
    if((argc!=2 && argc!=3) || std::sscanf(argv[1],"%d,%d,%d",topology,topology+1,topology+2)!=3 ||
       topology[0]*topology[1]*topology[2]!=ranks) throw std::runtime_error("Invalid wall probe topology");
    for(int d=0;d<3;++d) if(topology[d]<1 || topology[d]>2) throw std::runtime_error("Use the approved synthetic topologies");
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<1 || cudaSetDevice(rank%devices)!=cudaSuccess)
      throw std::runtime_error("Cannot bind the wall component GPU");
    viskores::cont::Initialize(argc,argv);
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
    if(viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
      viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    R worst=R(0),global_worst=R(0),arc_offset=R(0),global_arc_offset=R(0);int transfers=0,total_transfers=0;
    const bool shared=argc==3 && std::string(argv[2])=="shared";
    for(int mode=0;mode<cases;++mode) check_case(mode,topology,rank,ranks,worst,arc_offset,transfers,shared);
    MPI_Allreduce(&worst,&global_worst,1,astr_insitu::numerical_mpi_type(),MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(&arc_offset,&global_arc_offset,1,astr_insitu::numerical_mpi_type(),MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(&transfers,&total_transfers,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(!std::isfinite(global_worst) || global_worst>R(2e-10)) throw std::runtime_error("Wall component accuracy gate failed");
    if(rank==0) std::printf("PASS wall %s cases=%d ranks=%d error=%.17g transfers=%d source_host_mirror=0 oracle_geometry_readback=1 adaptive_arc_offset=%.17g\n",
      shared?"shared-driver":"worklet",cases,ranks,double(global_worst),total_transfers,double(global_arc_offset));
    MPI_Finalize();return 0;
  } catch(const std::exception& error) {
    std::fprintf(stderr,"Wall worklet rank=%d failed: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
