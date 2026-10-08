#include <mpi.h>
#include <nvml.h>
#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "insitu_shared_allocation_budget.h"
#ifdef ASTR_INSITU_DEVICE_RENDERING
#include <cuda_runtime_api.h>
#include "insitu_allocation_budget.h"
#endif

extern "C" int astr_insitu_map_current_cuda(int*,char*,int,char*,int);

namespace {
using Bytes=unsigned long long;
MPI_Comm node=MPI_COMM_NULL, world=MPI_COMM_NULL;
void* library=nullptr;
nvmlDevice_t device=nullptr;
std::vector<int> pids;
std::string report_path;
Bytes host_base=0,device_base=0,host_peak=0,device_peak=0;
Bytes host_limit=0,device_limit=0,reserve=0;
bool active=false;
bool allocation_single_rank_device=false;
MPI_Win allocation_window=MPI_WIN_NULL;
astr_insitu::SharedAllocationBudget* allocation_pool=nullptr;
astr_insitu::SharedAllocationBudget* shared_allocation=nullptr;
int allocation_groups=0,node_rank=-1;
std::map<const void*,Bytes> cuda_allocations,graphics_allocations;
Bytes allocation_baseline_free=0,allocation_calls=0,largest_allocation=0;
Bytes temporary_calls=0,graphics_allocation_calls=0;
using Query=nvmlReturn_t (*)(nvmlDevice_t,unsigned int*,nvmlProcessInfo_v2_t*);
Query compute_query=nullptr,graphics_query=nullptr;
nvmlReturn_t (*memory_query)(nvmlDevice_t,nvmlMemory_t*)=nullptr;
nvmlReturn_t (*shutdown_nvml)()=nullptr;

void require(bool condition,const char* message)
{
  if(!condition) throw std::runtime_error(message);
}
void mpi(int code) { require(code==MPI_SUCCESS,"resource observation MPI failure"); }
template<class T> T symbol(const char* name)
{
  auto address=dlsym(library,name);
  require(address!=nullptr,name);
  return reinterpret_cast<T>(address);
}
Bytes checked_sum(Bytes a,Bytes b)
{
  require(b<=std::numeric_limits<Bytes>::max()-a,"resource byte count overflow");
  return a+b;
}
Bytes rss()
{
  std::ifstream input("/proc/self/statm");
  Bytes virtual_pages=0,resident_pages=0;
  require(bool(input>>virtual_pages>>resident_pages),"cannot read process RSS");
  const long page=sysconf(_SC_PAGESIZE);
  require(page>0 && resident_pages<=std::numeric_limits<Bytes>::max()/Bytes(page),"invalid RSS size");
  return resident_pages*Bytes(page);
}
void processes(Query query,std::map<unsigned int,Bytes>& usage)
{
  unsigned int count=0;
  auto status=query(device,&count,nullptr);
  require(status==NVML_SUCCESS || status==NVML_ERROR_INSUFFICIENT_SIZE,"cannot query GPU processes");
  for(int attempt=0;attempt<4;++attempt) {
    std::vector<nvmlProcessInfo_v2_t> entries(count+16);
    count=static_cast<unsigned int>(entries.size());
    status=query(device,&count,entries.data());
    if(status==NVML_ERROR_INSUFFICIENT_SIZE) continue;
    require(status==NVML_SUCCESS && count<=entries.size(),"cannot read GPU processes");
    for(unsigned int i=0;i<count;++i) {
      const auto& item=entries[i];
      if(std::find(pids.begin(),pids.end(),int(item.pid))==pids.end()) continue;
      require(item.usedGpuMemory!=NVML_VALUE_NOT_AVAILABLE,"GPU process memory unavailable");
      usage[item.pid]=std::max(usage[item.pid],Bytes(item.usedGpuMemory));
    }
    return;
  }
  throw std::runtime_error("GPU process list did not stabilize");
}
void observe(Bytes& host,Bytes& gpu,Bytes& free)
{
  std::vector<Bytes> values(pids.size());
  const Bytes local=rss();
  mpi(MPI_Allgather(&local,1,MPI_UNSIGNED_LONG_LONG,values.data(),1,MPI_UNSIGNED_LONG_LONG,node));
  host=0;
  for(auto value:values) host=checked_sum(host,value);
  std::map<unsigned int,Bytes> usage;
  processes(compute_query,usage);
  processes(graphics_query,usage);
  require(!usage.empty(),"no job GPU memory visible to NVML");
  gpu=0;
  for(const auto& item:usage) gpu=checked_sum(gpu,item.second);
  nvmlMemory_t memory{};
  require(memory_query(device,&memory)==NVML_SUCCESS,"cannot query GPU free memory");
  free=memory.free;
}
void record(const char* stage,Bytes host,Bytes gpu,Bytes free)
{
  host_peak=std::max(host_peak,host);
  device_peak=std::max(device_peak,gpu);
  const Bytes dh=host>host_base?host-host_base:0;
  const Bytes dg=gpu>device_base?gpu-device_base:0;
  std::ofstream output(report_path,std::ios::app);
  require(bool(output),"cannot open resource observation report");
  output<<stage<<','<<host<<','<<gpu<<','<<free<<','<<dh<<','<<dg<<','
        <<host_peak<<','<<device_peak<<'\n';
  output.close();
  require(bool(output),"cannot write resource observation report");
  require(dh<=host_limit,"observed node host increment exceeds budget");
  require(dg<=device_limit,"observed physical GPU increment exceeds budget");
  require(free>=reserve,"observed device free memory below reserve");
  // Every rank must finish checking before any rank proceeds to the next phase.
  mpi(MPI_Barrier(world));
}
int fail(const char* message)
{
  std::fprintf(stderr,"ASTR INSITU RESOURCE OBSERVER ERROR: %s\n",message);
  MPI_Abort(world==MPI_COMM_NULL?MPI_COMM_WORLD:world,1);
  return 1;
}
}

extern "C" int astr_insitu_resource_begin(int fcomm,std::int64_t host_bytes,
    std::int64_t device_bytes,std::int64_t reserve_bytes,int shared_fixture,const char* output_directory)
try {
  world=MPI_Comm_f2c(fcomm);
  require(!active && host_bytes>0 && device_bytes>0 && reserve_bytes>=0,"invalid observer configuration");
  host_limit=host_bytes; device_limit=device_bytes; reserve=reserve_bytes;
  mpi(MPI_Comm_split_type(world,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node));
  int count=0,rank=0,pid=int(getpid());
  mpi(MPI_Comm_size(node,&count));
  mpi(MPI_Comm_rank(world,&rank));
  mpi(MPI_Comm_rank(node,&node_rank));
  pids.resize(count);
  mpi(MPI_Allgather(&pid,1,MPI_INT,pids.data(),1,MPI_INT,node));
  char suffix[80];
  std::snprintf(suffix,sizeof(suffix),"/resources.rank%08d.csv",rank);
  report_path=std::string(output_directory)+suffix;
  require(!std::ifstream(report_path).good(),"resource report already exists");
  library=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);
  require(library!=nullptr,"NVML runtime unavailable");
  auto initialize=symbol<nvmlReturn_t (*)()>("nvmlInit_v2");
  shutdown_nvml=symbol<nvmlReturn_t (*)()>("nvmlShutdown");
  require(initialize()==NVML_SUCCESS,"NVML initialization failed");
  auto by_uuid=symbol<nvmlReturn_t (*)(const char*,nvmlDevice_t*)>("nvmlDeviceGetHandleByUUID");
  char uuid[37]={},message[256]={};
  int index=-1;
  require(astr_insitu_map_current_cuda(&index,uuid,sizeof(uuid),message,sizeof(message))==0,message);
  const auto identity=std::string("GPU-")+uuid;
  require(by_uuid(identity.c_str(),&device)==NVML_SUCCESS,"NVML current CUDA UUID lookup failed");
  compute_query=symbol<Query>("nvmlDeviceGetComputeRunningProcesses_v2");
  graphics_query=symbol<Query>("nvmlDeviceGetGraphicsRunningProcesses_v2");
  memory_query=symbol<decltype(memory_query)>("nvmlDeviceGetMemoryInfo");
  Bytes free=0;
  observe(host_base,device_base,free);
  // Share one baseline for ranks assigned to the same physical GPU.
  std::vector<char> identities(count*37);
  std::vector<Bytes> bases(count);
  mpi(MPI_Allgather(uuid,37,MPI_CHAR,identities.data(),37,MPI_CHAR,node));
  mpi(MPI_Allgather(&device_base,1,MPI_UNSIGNED_LONG_LONG,bases.data(),1,MPI_UNSIGNED_LONG_LONG,node));
  for(int i=0;i<count;++i)
    if(std::string(identities.data()+37*i)==uuid) device_base=std::min(device_base,bases[i]);
  int device_ranks=0;
  for(int i=0;i<count;++i) device_ranks+=std::string(identities.data()+37*i)==uuid;
  allocation_single_rank_device=device_ranks==1;
  if(shared_fixture && device_ranks>1) {
    int world_ranks=0;
    mpi(MPI_Comm_size(world,&world_ranks));
    std::map<std::string,int> groups;
    for(int i=0;i<count;++i) ++groups[std::string(identities.data()+37*i)];
    require(world_ranks==4 && count==4 && groups.size()==2 && device_ranks==2,
      "shared allocation admission requires the approved four-rank/two-GPU fixture");
    auto lifecycle=reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT,"viskores_astr_allocation_lifecycle_version"));
    auto graphics_lifecycle=reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT,"vtk_astr_buffer_allocation_lifecycle_version"));
    require(lifecycle && graphics_lifecycle && lifecycle()==1 && graphics_lifecycle()==1,
      "shared allocation lifecycle dependency patch missing");
    allocation_groups=int(groups.size());
    int group=0;
    for(const auto& item:groups) { if(item.first==uuid) break; ++group; }
    void* local=nullptr;
    const MPI_Aint bytes=node_rank==0?MPI_Aint(sizeof(*allocation_pool)*allocation_groups):0;
    mpi(MPI_Win_allocate_shared(bytes,1,MPI_INFO_NULL,node,&local,&allocation_window));
    MPI_Aint shared_size=0;int displacement=0;
    mpi(MPI_Win_shared_query(allocation_window,0,&shared_size,&displacement,&local));
    require(shared_size==MPI_Aint(sizeof(*allocation_pool)*allocation_groups),"shared allocation window size");
    int* model=nullptr,flag=0;
    mpi(MPI_Win_get_attr(allocation_window,MPI_WIN_MODEL,&model,&flag));
    require(flag && model && *model==MPI_WIN_UNIFIED,"shared allocation window requires unified memory model");
    allocation_pool=static_cast<astr_insitu::SharedAllocationBudget*>(local);
    mpi(MPI_Win_lock_all(0,allocation_window));
    if(node_rank==0) for(int i=0;i<allocation_groups;++i)
      astr_insitu::initialize_shared_allocation_budget(allocation_pool[i]);
    mpi(MPI_Win_sync(allocation_window));
    mpi(MPI_Barrier(node));
    mpi(MPI_Win_sync(allocation_window));
    shared_allocation=&allocation_pool[group];
    std::printf("ASTR_INSITU_SHARED_ALLOCATION rank=%d device_ranks=%d budget_bytes=%llu atomic_reservation=1\n",
      rank,device_ranks,device_limit);
  }
#ifdef ASTR_INSITU_DEVICE_RENDERING
  std::size_t cuda_free=0,cuda_total=0;
  require(cudaMemGetInfo(&cuda_free,&cuda_total)==cudaSuccess,"cannot query allocation baseline");
  allocation_baseline_free=cuda_free;
  mpi(MPI_Allgather(&allocation_baseline_free,1,MPI_UNSIGNED_LONG_LONG,bases.data(),1,
    MPI_UNSIGNED_LONG_LONG,node));
  for(int i=0;i<count;++i)
    if(std::string(identities.data()+37*i)==uuid)
      allocation_baseline_free=std::max(allocation_baseline_free,bases[i]);
  allocation_calls=0;largest_allocation=0;
  temporary_calls=0;graphics_allocation_calls=0;
#endif
  std::ofstream output(report_path);
  output<<"# device="<<identity<<" host_limit="<<host_limit<<" device_limit="<<device_limit
        <<" reserve="<<reserve<<"\n"
        <<"stage,node_rss_bytes,job_device_bytes,device_free_bytes,host_increment_bytes,device_increment_bytes,"
        <<"node_rss_peak_bytes,job_device_peak_bytes\n";
  output.close();
  require(bool(output),"cannot initialize resource observation report");
  active=true;
  record("baseline",host_base,device_base,free);
  return 0;
} catch(const std::exception& error) { return fail(error.what()); }
  catch(...) { return fail("unknown observer initialization exception"); }

extern "C" int astr_insitu_resource_check(const char* stage)
try {
  if(!active) return 0;
  Bytes host=0,gpu=0,free=0;
  observe(host,gpu,free);
  record(stage,host,gpu,free);
  return 0;
} catch(const std::exception& error) { return fail(error.what()); }
  catch(...) { return fail("unknown resource observation exception"); }

extern "C" int astr_insitu_allocation_guard_active() {
  return active && (allocation_single_rank_device || shared_allocation) ? 1 : 0;
}

extern "C" void astr_insitu_device_allocation_preflight(std::size_t bytes,const char* source)
try {
#ifdef ASTR_INSITU_DEVICE_RENDERING
  if(!active) return;
  astr_insitu::SharedAllocationLock lock(shared_allocation);
  std::size_t free=0,total=0;
  // Also check this job's NVML allocation total: another process freeing memory
  // must not increase the permitted job increment above its configured budget.
  std::map<unsigned int,Bytes> usage;
  processes(compute_query,usage);processes(graphics_query,usage);
  require(!usage.empty(),"allocation preflight job memory unavailable");
  Bytes job_bytes=0;
  for(const auto& item:usage) job_bytes=checked_sum(job_bytes,item.second);
  const auto job_increment=job_bytes>device_base?job_bytes-device_base:0;
  if(cudaMemGetInfo(&free,&total)!=cudaSuccess ||
      !astr_insitu::admit_device_allocation(bytes,free,allocation_baseline_free,device_limit,reserve) ||
      job_increment>device_limit || bytes>device_limit-job_increment) {
    std::fprintf(stderr,"ASTR INSITU ALLOCATION REFUSED source=%s requested=%zu free=%zu "
      "baseline_free=%llu budget=%llu reserve=%llu\n",source?source:"unknown",bytes,free,
      allocation_baseline_free,device_limit,reserve);
    fail("device allocation would exceed budget/reserve");
    return;
  }
  if(shared_allocation && bytes) {
    const auto used=allocation_baseline_free>free?allocation_baseline_free-free:0;
    const auto actual=std::max(used,job_increment);
    const auto available=std::min(device_limit-actual,Bytes(free)-reserve);
    // Live and pending reservations are counted in addition to observed usage,
    // so concurrent ranks cannot admit the same remaining headroom.
    if(!astr_insitu::reserve_shared_allocation(*shared_allocation,bytes,available)) {
      std::fprintf(stderr,"ASTR INSITU SHARED ALLOCATION REFUSED requested=%zu held=%llu available=%llu\n",
        bytes,static_cast<Bytes>(shared_allocation->held),available);
      throw std::runtime_error("shared device allocation would exceed budget/reserve");
    }
  }
  ++allocation_calls;largest_allocation=std::max(largest_allocation,Bytes(bytes));
  if(source && std::strstr(source,"Thrust")) ++temporary_calls;
  if(source && std::strstr(source,"OpenGL")) ++graphics_allocation_calls;
#else
  (void)bytes;(void)source;
#endif
} catch(const std::exception& error) { fail(error.what()); }
  catch(...) { fail("unknown allocation preflight exception"); }

extern "C" void astr_insitu_device_allocation_commit(const void* pointer,std::size_t bytes,const char* source)
try {
  if(!active || !shared_allocation || !bytes) return;
  astr_insitu::SharedAllocationLock lock(*shared_allocation);
  if(!pointer) { astr_insitu::release_shared_allocation(*shared_allocation,bytes);return; }
  auto& allocations=source && std::strstr(source,"OpenGL")?graphics_allocations:cuda_allocations;
  auto previous=allocations.find(pointer);
  if(previous!=allocations.end()) {
    require(&allocations==&graphics_allocations,"duplicate CUDA allocation identity");
    astr_insitu::release_shared_allocation(*shared_allocation,previous->second);
    previous->second=bytes;
  } else allocations.emplace(pointer,bytes);
} catch(const std::exception& error) { fail(error.what()); }
  catch(...) { fail("unknown allocation commit exception"); }

extern "C" void astr_insitu_device_allocation_release(const void* pointer,const char* source)
try {
  if(!active || !shared_allocation || !pointer) return;
  astr_insitu::SharedAllocationLock lock(*shared_allocation);
  auto& allocations=source && std::strstr(source,"OpenGL")?graphics_allocations:cuda_allocations;
  const auto found=allocations.find(pointer);
  if(found==allocations.end()) return; // Pre-observer or unguarded buffers have no reservation.
  astr_insitu::release_shared_allocation(*shared_allocation,found->second);
  allocations.erase(found);
} catch(const std::exception& error) { fail(error.what()); }
  catch(...) { fail("unknown allocation release exception"); }

extern "C" int astr_insitu_resource_finish()
try {
  if(!active) return 0;
  astr_insitu_resource_check("session_released");
  std::printf("ASTR_INSITU_ALLOCATION_PREFLIGHT calls=%llu largest_request_bytes=%llu "
    "temporary_calls=%llu graphics_calls=%llu\n",allocation_calls,largest_allocation,
    temporary_calls,graphics_allocation_calls);
  if(shared_allocation) {
    { astr_insitu::SharedAllocationLock lock(*shared_allocation);
      std::printf("ASTR_INSITU_SHARED_ALLOCATION_END held=%llu peak=%llu admissions=%llu releases=%llu refusals=%llu\n",
        static_cast<Bytes>(shared_allocation->held),static_cast<Bytes>(shared_allocation->peak),
        static_cast<Bytes>(shared_allocation->admissions),static_cast<Bytes>(shared_allocation->releases),
        static_cast<Bytes>(shared_allocation->refusals)); }
    active=false;shared_allocation=nullptr;
    mpi(MPI_Barrier(node));
    if(node_rank==0) for(int i=0;i<allocation_groups;++i)
      astr_insitu::destroy_shared_allocation_budget(allocation_pool[i]);
    mpi(MPI_Win_unlock_all(allocation_window));
    mpi(MPI_Win_free(&allocation_window));
    allocation_pool=nullptr;allocation_groups=0;
    cuda_allocations.clear();graphics_allocations.clear();
  }
  require(shutdown_nvml()==NVML_SUCCESS,"NVML shutdown failed");
  dlclose(library); library=nullptr;
  mpi(MPI_Comm_free(&node));
  active=false;
  return 0;
} catch(const std::exception& error) { return fail(error.what()); }
  catch(...) { return fail("unknown resource observer finalization exception"); }
