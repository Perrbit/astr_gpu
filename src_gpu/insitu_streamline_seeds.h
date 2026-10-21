#ifndef ASTR_INSITU_STREAMLINE_SEEDS_H
#define ASTR_INSITU_STREAMLINE_SEEDS_H
#include "astr_precision.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace astr_insitu {
using Real=astr::numerical_real;
// Case presets supply physical coordinates; integrators do not choose seed sites.
inline std::vector<std::array<Real,3>> tgv_streamline_seeds(const std::string& layout) {
  const Real pi=std::acos(-Real(1.));
  std::vector<std::array<Real,3>> seeds;
  if(layout=="line16") {
    seeds.reserve(16);
    for(int i=0;i<16;++i) seeds.push_back({pi/Real(2.),pi/Real(8.)+i*(Real(6.)*pi/Real(8.))/Real(15.),pi/Real(4.)});
  } else if(layout=="tgv-stratified") {
    seeds.reserve(256);
    for(int k=0;k<4;++k) for(int j=0;j<8;++j) for(int i=0;i<8;++i)
      seeds.push_back({(i+Real(.5))*(Real(2.)*pi/Real(8.)),(j+Real(.5))*(Real(2.)*pi/Real(8.)),(k+Real(.5))*(Real(2.)*pi/Real(4.))});
  } else if(layout=="bl-layered64" || layout=="bl-layered256") {
    const int count=layout=="bl-layered64"?16:64;
    seeds.reserve(4*count);
    for(Real y:{Real(.5),Real(1.),Real(2.),Real(5.)}) for(int k=0;k<count;++k)
      seeds.push_back({Real(54.),y,(k+Real(.5))*Real(90.)/count});
  } else throw std::invalid_argument("Unsupported TGV streamline seed layout");
  return seeds;
}
} // namespace astr_insitu
#endif
