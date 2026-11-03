#pragma once
#include <viskores/Math.h>

namespace astr_insitu {
enum class WallArrowStatus { valid, zero, invalid };

template<class Real>
VISKORES_EXEC_CONT WallArrowStatus wall_arrow(const viskores::Vec<Real,3>& origin,
    Real tx,Real tz,Real length,viskores::Vec<Real,3> points[6],Real& magnitude,Real sx=Real(1.),Real sz=Real(1.)) {
  if(!viskores::IsFinite(tx) || !viskores::IsFinite(tz) || !viskores::IsFinite(length) || length<=Real(0.))
    return WallArrowStatus::invalid;
  if(!viskores::IsFinite(sx) || !viskores::IsFinite(sz) || sx<=Real(0.) || sz<=Real(0.))
    return WallArrowStatus::invalid;
  for(int d=0;d<3;++d) if(!viskores::IsFinite(origin[d])) return WallArrowStatus::invalid;
  const Real scale=viskores::Max(viskores::Abs(tx),viskores::Abs(tz));
  magnitude=Real(0.);
  if(scale==Real(0.)) return WallArrowStatus::zero;
  const Real x=tx/scale,z=tz/scale,norm=viskores::Sqrt(x*x+z*z);
  magnitude=scale*norm;
  if(!viskores::IsFinite(magnitude)) return WallArrowStatus::invalid;
  const viskores::Vec<Real,3> direction(x/norm,Real(0.),z/norm);
  // Only the arrowhead is a display annotation; the shaft keeps its physical direction and length.
  const viskores::Vec<Real,3> normal(-direction[2]*sz/sx,Real(0.),direction[0]*sx/sz);
  const auto tip=origin+length*direction;
  if(tip==origin) return WallArrowStatus::invalid;
  points[0]=origin;points[1]=tip;
  points[2]=tip;points[3]=tip-length*Real(.25)*direction+length*Real(.12)*normal;
  points[4]=tip;points[5]=tip-length*Real(.25)*direction-length*Real(.12)*normal;
  for(int p=0;p<6;++p)
    for(int d=0;d<3;++d) if(!viskores::IsFinite(points[p][d])) return WallArrowStatus::invalid;
  return WallArrowStatus::valid;
}
}
