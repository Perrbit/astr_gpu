#include "astr_precision.h"

extern "C" astr::numerical_real astr_precision_affine(astr::numerical_real x) {
  return astr::numerical_real{3} * x + astr::numerical_real{2};
}

extern "C" int astr_precision_bytes() {
  return sizeof(astr::numerical_real);
}
