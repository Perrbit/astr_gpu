#include "insitu_vortex_math.h"

static_assert(sizeof(astr_insitu::VortexReal)==8,"The TB3 mathematical probe is FP64");

extern "C" int astr_vortex_math_cpu(int count,const double* gradients,int requested,double* values,int* status,int* work) {
  if(count<0) return 1;
  for(int i=0;i<count;++i) {
    const auto result=astr_insitu::vortex_values(gradients+9*i,(requested&1)!=0,(requested&2)!=0);
    values[2*i]=result.lambda2;values[2*i+1]=result.lambda_ci;
    status[2*i]=result.lambda2_status;status[2*i+1]=result.lambda_ci_status;
    work[2*i]=result.jacobi_sweeps;work[2*i+1]=result.swirl_iterations;
  }
  return 0;
}
