#ifndef ASTR_INSITU_VORTEX_MATH_H
#define ASTR_INSITU_VORTEX_MATH_H

#include "astr_precision.h"
#include <cmath>
#include <climits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ASTR_VORTEX_EXEC __host__ __device__
#else
#define ASTR_VORTEX_EXEC
#endif

namespace astr_insitu {
using VortexReal=astr::numerical_real;
enum VortexStatus { VortexOk=0,VortexNonfinite=1,VortexNoConvergence=2,VortexRange=3,VortexNotRequested=4 };
struct VortexValues {
  VortexReal lambda2,lambda_ci;
  int lambda2_status,lambda_ci_status,jacobi_sweeps,swirl_iterations;
};

ASTR_VORTEX_EXEC inline VortexReal imaginary_pair(VortexReal a,VortexReal b,VortexReal c,VortexReal d) {
  using R=VortexReal;
  const R p=R(.5)*a-R(.5)*d;
  int ep=0,eb=0,ec=0;
  const R mp=std::frexp(p,&ep),mb=std::frexp(b,&eb),mc=std::frexp(c,&ec);
  int scale=p!=R(0)?2*ep:INT_MIN;
  if(b!=R(0) && c!=R(0) && eb+ec>scale) scale=eb+ec;
  if(scale==INT_MIN) return R(0);
  const R pp=mp*mp,bc=mb*mc;
  const R x=mp==R(0)?R(0):std::ldexp(pp,2*ep-scale);
  const R y=mb==R(0) || mc==R(0)?R(0):std::ldexp(bc,eb+ec-scale);
  const R sum=x+y,v=sum-x;
  const R error=(x-(sum-v))+(y-v)+std::ldexp(std::fma(mp,mp,-pp),2*ep-scale)+
    std::ldexp(std::fma(mb,mc,-bc),eb+ec-scale);
  const R discriminant=sum+error;
  if(discriminant>=R(0)) return R(0);
  int half=scale/2;if(scale<0 && scale%2) --half;
  return std::ldexp(std::sqrt(std::ldexp(-discriminant,scale-2*half)),half);
}

// Cyclic Jacobi on a scaled real symmetric 3x3 matrix; no eigenvectors needed.
ASTR_VORTEX_EXEC inline int symmetric_middle(VortexReal a[3][3],VortexReal& middle,int& sweeps) {
  using R=VortexReal;
  for(sweeps=0;sweeps<32;++sweeps) {
    if(a[0][1]==R(0) && a[0][2]==R(0) && a[1][2]==R(0)) {
      R x=a[0][0],y=a[1][1],z=a[2][2],t;
      if(x>y) {t=x;x=y;y=t;}
      if(y>z) {t=y;y=z;z=t;}
      if(x>y) {t=x;x=y;y=t;}
      middle=y;return VortexOk;
    }
    for(int p=0;p<2;++p) for(int q=p+1;q<3;++q) {
      const R off=a[p][q];
      if(off==R(0)) continue;
      const R app=a[p][p],aqq=a[q][q],magnitude=std::abs(off);
      if(std::abs(app)+magnitude==std::abs(app) && std::abs(aqq)+magnitude==std::abs(aqq)) {
        a[p][q]=a[q][p]=R(0);continue;
      }
      const R difference=aqq-app,radius=std::hypot(difference,R(2)*off);
      const R t=R(2)*off/(difference+std::copysign(radius,difference));
      const R cosine=R(1)/std::hypot(R(1),t),sine=t*cosine,change=t*off;
      if(t!=R(0) && change==R(0)) return VortexRange;
      a[p][p]=app-change;a[q][q]=aqq+change;a[p][q]=a[q][p]=R(0);
      for(int k=0;k<3;++k) if(k!=p && k!=q) {
        const R x=a[k][p],y=a[k][q];
        a[k][p]=a[p][k]=cosine*x-sine*y;
        a[k][q]=a[q][k]=sine*x+cosine*y;
      }
    }
  }
  return VortexNoConvergence;
}

// M = sym(A^2) = S^2 + Omega^2. Exponent-scaled products avoid first forming
// huge strain/rotation squares which subsequently cancel in a shear field.
ASTR_VORTEX_EXEC inline int vortex_lambda2(const VortexReal* gradient,VortexReal& value,int& sweeps) {
  using R=VortexReal;
  if(gradient[1]==R(0) && gradient[2]==R(0) && gradient[3]==R(0) && gradient[5]==R(0) &&
     gradient[6]==R(0) && gradient[7]==R(0)) {
    R x=std::abs(gradient[0]),y=std::abs(gradient[4]),z=std::abs(gradient[8]),swap;
    if(x>y) {swap=x;x=y;y=swap;}if(y>z) {swap=y;y=z;z=swap;}if(x>y) {swap=x;x=y;y=swap;}
    int exponent=0;const R mantissa=std::frexp(y,&exponent);
    value=std::ldexp(mantissa*mantissa,2*exponent);sweeps=0;
    return !std::isfinite(value) || (y!=R(0) && value==R(0))?VortexRange:VortexOk;
  }
  R mantissa[3][3];int exponent[3][3];
  for(int i=0;i<3;++i) for(int j=0;j<3;++j)
    mantissa[i][j]=std::frexp(gradient[i+3*j],&exponent[i][j]);
  int scale=INT_MIN;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
    if(mantissa[i][k]!=R(0) && mantissa[k][j]!=R(0)) {
      const int e=exponent[i][k]+exponent[k][j];if(e>scale) scale=e;
    }
  if(scale==INT_MIN) {value=R(0);sweeps=0;return VortexOk;}
  R matrix[3][3],largest=R(0);
  for(int i=0;i<3;++i) for(int j=i;j<3;++j) {
    R sum=R(0);
    for(int k=0;k<3;++k) for(int transpose=0;transpose<2;++transpose) {
      const int row=transpose?j:i,column=transpose?i:j;
      const R product=mantissa[row][k]*mantissa[k][column];
      if(product==R(0)) continue;
      const R term=std::ldexp(product,exponent[row][k]+exponent[k][column]-scale);
      if(term==R(0)) return VortexRange;
      sum+=term;
    }
    matrix[i][j]=matrix[j][i]=R(.5)*sum;
    largest=std::fmax(largest,std::abs(matrix[i][j]));
  }
  if(largest==R(0)) {value=R(0);sweeps=0;return VortexOk;}
  int normalization=0;std::frexp(largest,&normalization);
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) matrix[i][j]=std::ldexp(matrix[i][j],-normalization);
  R middle=R(0);
  const int status=symmetric_middle(matrix,middle,sweeps);
  if(status!=VortexOk) return status;
  value=std::ldexp(middle,scale+normalization);
  if(!std::isfinite(value) || (middle!=R(0) && value==R(0))) return VortexRange;
  return VortexOk;
}

// Two local FP64 components retain cancellation information in the cubic
// coefficients. No field, gradient array or persistent state changes precision.
struct VortexWide {
  VortexReal hi,lo;
  ASTR_VORTEX_EXEC VortexWide(VortexReal value=0,VortexReal tail=0):hi(value),lo(tail) {}
  ASTR_VORTEX_EXEC bool zero() const {return hi==0 && lo==0;}
  ASTR_VORTEX_EXEC bool finite() const {return std::isfinite(hi) && std::isfinite(lo);}
  ASTR_VORTEX_EXEC int sign() const {return hi>0?1:hi<0?-1:lo>0?1:lo<0?-1:0;}
};
ASTR_VORTEX_EXEC inline VortexWide wide_normalize(VortexReal hi,VortexReal lo) {
  const VortexReal sum=hi+lo;
  return {sum,lo-(sum-hi)};
}
ASTR_VORTEX_EXEC inline VortexWide operator+(VortexWide a,VortexWide b) {
  const VortexReal sum=a.hi+b.hi,v=sum-a.hi;
  return wide_normalize(sum,(a.hi-(sum-v))+(b.hi-v)+a.lo+b.lo);
}
ASTR_VORTEX_EXEC inline VortexWide operator-(VortexWide a) {return {-a.hi,-a.lo};}
ASTR_VORTEX_EXEC inline VortexWide operator-(VortexWide a,VortexWide b) {return a+(-b);}
ASTR_VORTEX_EXEC inline VortexWide operator*(VortexWide a,VortexWide b) {
  if(a.zero() || b.zero()) return {};
  const VortexReal product=a.hi*b.hi;
  if(product==0 && a.hi!=0 && b.hi!=0) return {VortexReal(NAN),0};
  return wide_normalize(product,std::fma(a.hi,b.hi,-product)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo);
}
ASTR_VORTEX_EXEC inline VortexWide operator/(VortexWide a,VortexWide b) {
  const VortexReal first=a.hi/b.hi;
  if(!std::isfinite(first) || (first==0 && a.hi!=0)) return {VortexReal(NAN),0};
  const auto remainder=a-b*VortexWide(first);
  return wide_normalize(first,(remainder.hi+remainder.lo)/b.hi);
}
ASTR_VORTEX_EXEC inline VortexWide wide_scale(VortexWide a,int exponent) {
  return {std::ldexp(a.hi,exponent),std::ldexp(a.lo,exponent)};
}
ASTR_VORTEX_EXEC inline int exponent_ceiling(int exponent,int divisor) {
  return exponent>0?(exponent+divisor-1)/divisor:exponent/divisor;
}

ASTR_VORTEX_EXEC inline int vortex_lambda_ci(const VortexReal* gradient,VortexReal& value,int& iterations) {
  using R=VortexReal;using W=VortexWide;
  iterations=0;
  // Exact block-triangular structure factors the characteristic polynomial.
  for(int isolated=0;isolated<3;++isolated) {
    bool row=true,column=true;
    for(int j=0;j<3;++j) if(j!=isolated) {
      row=row && gradient[isolated+3*j]==R(0);
      column=column && gradient[j+3*isolated]==R(0);
    }
    if(row || column) {
      const int i=(isolated+1)%3,j=(isolated+2)%3;
      value=imaginary_pair(gradient[i+3*i],gradient[i+3*j],gradient[j+3*i],gradient[j+3*j]);
      return std::isfinite(value)?VortexOk:VortexRange;
    }
  }
  R largest=R(0);
  for(int i=0;i<9;++i) largest=std::fmax(largest,std::abs(gradient[i]));
  int scale=0;std::frexp(largest,&scale);
  W a[3][3];
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
    const R normalized=std::ldexp(gradient[i+3*j],-scale);
    if(gradient[i+3*j]!=R(0) && normalized==R(0)) return VortexRange;
    a[i][j]=W(normalized);
  }
  // Center by the compensated trace, not an arbitrary diagonal: a traceless
  // nilpotent input must not acquire cancelling O(1) cubic coefficients.
  const W shift=(a[0][0]+a[1][1]+a[2][2])/W(R(3));
  for(int i=0;i<3;++i) a[i][i]=a[i][i]-shift;
  largest=R(0);for(int i=0;i<3;++i) for(int j=0;j<3;++j) largest=std::fmax(largest,std::abs(a[i][j].hi));
  if(largest==R(0)) {value=R(0);return VortexOk;}
  int extra=0;std::frexp(largest,&extra);
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) a[i][j]=wide_scale(a[i][j],-extra);
  const W trace=-(a[0][0]+a[1][1]+a[2][2]);
  const W quadratic=a[0][0]*a[1][1]-a[0][1]*a[1][0]+a[0][0]*a[2][2]-a[0][2]*a[2][0]+
    a[1][1]*a[2][2]-a[1][2]*a[2][1];
  const W constant=-(a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-
    a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]));
  W p=(W(R(3))*quadratic-trace*trace)/W(R(3));
  W q=(W(R(2))*trace*trace*trace-W(R(9))*trace*quadratic+W(R(27))*constant)/W(R(27));
  if(!p.finite() || !q.finite()) return VortexRange;
  if(p.zero() && q.zero()) {value=R(0);return VortexOk;}
  // Scale the polynomial variable, not just the matrix. This avoids p^3
  // underflow when a very weak pair accompanies a large real shear.
  int root_scale=INT_MIN,e=0;
  if(!p.zero()) {std::frexp(std::abs(p.hi),&e);root_scale=exponent_ceiling(e,2);}
  if(!q.zero()) {
    std::frexp(std::abs(q.hi),&e);const int candidate=exponent_ceiling(e,3);
    if(candidate>root_scale) root_scale=candidate;
  }
  p=wide_scale(p,-2*root_scale);q=wide_scale(q,-3*root_scale);
  const W third=p/W(R(3)),half=q/W(R(2));
  const W discriminant=half*half+third*third*third;
  if(!discriminant.finite()) return VortexRange;
  if(discriminant.sign()<=0) {value=R(0);return VortexOk;}
  const R radical=std::sqrt(discriminant.hi+discriminant.lo);
  const R u=std::cbrt(-(q.hi+q.lo)*R(.5)-std::copysign(radical,q.hi+q.lo));
  if(u==R(0) || !std::isfinite(u)) return VortexRange;
  W root(u-(p.hi+p.lo)/(R(3)*u));
  for(iterations=0;iterations<3;++iterations) {
    const W derivative=W(R(3))*root*root+p;
    if(derivative.zero()) return VortexNoConvergence;
    root=root-(root*root*root+p*root+q)/derivative;
    if(!root.finite()) return VortexRange;
  }
  const W imaginary_squared=(W(R(3))*root*root+W(R(4))*p)/W(R(4));
  if(!imaginary_squared.finite() || imaginary_squared.sign()<=0) return VortexNoConvergence;
  const R imaginary=std::sqrt(imaginary_squared.hi+imaginary_squared.lo);
  value=std::ldexp(imaginary,scale+extra+root_scale);
  return !std::isfinite(value) || (imaginary!=R(0) && value==R(0))?VortexRange:VortexOk;
}

// Input is A_ij=du_i/dx_j, column-major, matching the Fortran gradient packet.
ASTR_VORTEX_EXEC inline VortexValues vortex_values(const VortexReal* gradient,bool need_lambda2=true,bool need_lambda_ci=true) {
  using R=VortexReal;
  const R unavailable=R(NAN);
  VortexValues out{unavailable,unavailable,VortexNotRequested,VortexNotRequested,0,0};
  for(int i=0;i<9;++i) if(!std::isfinite(gradient[i])) {
    if(need_lambda2) out.lambda2_status=VortexNonfinite;
    if(need_lambda_ci) out.lambda_ci_status=VortexNonfinite;
    return out;
  }
  if(need_lambda2) out.lambda2_status=vortex_lambda2(gradient,out.lambda2,out.jacobi_sweeps);
  if(need_lambda_ci) out.lambda_ci_status=vortex_lambda_ci(gradient,out.lambda_ci,out.swirl_iterations);
  if(out.lambda2_status!=VortexOk) out.lambda2=unavailable;
  if(out.lambda_ci_status!=VortexOk) out.lambda_ci=unavailable;
  return out;
}
} // namespace astr_insitu
#undef ASTR_VORTEX_EXEC
#endif
