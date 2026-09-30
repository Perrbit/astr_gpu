#include <mpi.h>
#include <nvml.h>
#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::int64_t device_bytes,std::int64_t reserve_bytes,const char* output_directory)
try {
  world=MPI_Comm_f2c(fcomm);
  require(!active && host_bytes>0 && device_bytes>0 && reserve_bytes>=0,"invalid observer configuration");
  host_limit=host_bytes; device_limit=device_bytes; reserve=reserve_bytes;
  mpi(MPI_Comm_split_type(world,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node));
  int count=0,rank=0,pid=int(getpid());
  mpi(MPI_Comm_size(node,&count));
  mpi(MPI_Comm_rank(world,&rank));
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

extern "C" int astr_insitu_resource_finish()
try {
  if(!active) return 0;
  astr_insitu_resource_check("session_released");
  require(shutdown_nvml()==NVML_SUCCESS,"NVML shutdown failed");
  dlclose(library); library=nullptr;
  mpi(MPI_Comm_free(&node));
  active=false;
  return 0;
} catch(const std::exception& error) { return fail(error.what()); }
  catch(...) { return fail("unknown resource observer finalization exception"); }
