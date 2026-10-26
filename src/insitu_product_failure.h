#pragma once
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

extern "C" int astr_insitu_product_disable(const char*) __attribute__((weak));
extern "C" int astr_insitu_product_device_health();

namespace astr_insitu {
// Only local work belongs here. Distributed algorithms need their own matching
// rendezvous before this boundary; MPI/CUDA context failures remain fatal.
template<class Operation>
bool local_product(MPI_Comm comm,const char* name,int step,double time,Operation operation) {
  std::string error;
  int rank=0;
  if(MPI_Comm_rank(comm,&rank)!=MPI_SUCCESS) MPI_Abort(comm,1);
  try {
#ifdef ASTR_BUILD_TESTING
    if(const char* fault=std::getenv("ASTR_INSITU_TEST_PRODUCT_FAIL"))
      if(std::string(fault)==std::string(name)+":"+std::to_string(step)+":"+std::to_string(rank))
        throw std::runtime_error("injected rank-local product extraction failure");
#endif
    operation();
  } catch(const std::exception& caught) { error=caught.what(); }
  int bad=!error.empty(),any=0;
  if(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm)!=MPI_SUCCESS) MPI_Abort(comm,1);
  if(!any) return true;
  bad=astr_insitu_product_device_health()!=0;
  if(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm)!=MPI_SUCCESS || any) {
    std::fprintf(stderr,"ASTR INSITU unrecoverable CUDA failure after product %s\n",name);
    MPI_Abort(comm,1);
  }
  for(const char* kind:{"image","geometry"}) {
    const auto id=std::string(name)+"."+kind;
    if(astr_insitu_product_disable && astr_insitu_product_disable(id.c_str())) MPI_Abort(comm,1);
  }
  std::fprintf(stderr,"ASTR_INSITU_PRODUCT_DISABLED rank=%d product=%s step=%d time=%.17g reason=%s\n",
    rank,name,step,time,error.empty()?"product failed on another rank":error.c_str());
  return false;
}
}
