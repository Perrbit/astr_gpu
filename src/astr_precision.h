#pragma once

#ifdef ASTR_REAL32
typedef float astr_numerical_real;
#else
typedef double astr_numerical_real;
#endif

#ifdef __cplusplus
namespace astr {
using numerical_real = astr_numerical_real;
}
#endif
