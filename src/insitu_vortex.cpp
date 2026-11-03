#include "insitu_vortex_math.h"

extern "C" int astr_insitu_vortex_point(const astr::numerical_real* gradient,int requested,
    astr::numerical_real* values) {
  const auto result=astr_insitu::vortex_values(gradient,(requested&1)!=0,(requested&2)!=0);
  values[0]=result.lambda2;values[1]=result.lambda_ci;
  if((requested&1) && result.lambda2_status!=astr_insitu::VortexOk) return result.lambda2_status;
  if((requested&2) && result.lambda_ci_status!=astr_insitu::VortexOk) return result.lambda_ci_status;
  return 0;
}
