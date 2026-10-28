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
inline std::vector<std::array<Real,3>> tgv_streamline_seeds(const std::string& layout,
    const std::array<Real,6>* physical_bounds=nullptr) {
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
  } else if(layout=="sbli-layered16") {
    if(!physical_bounds) throw std::invalid_argument("SBLI seeds require physical domain bounds");
    const auto& b=*physical_bounds;
    for(int d=0;d<3;++d) if(!std::isfinite(b[2*d]) || !std::isfinite(b[2*d+1]) || b[2*d]>=b[2*d+1])
      throw std::invalid_argument("Invalid SBLI seed domain");
    for(int j=0;j<16;++j) {
      const Real eta=(j+Real(.5))/Real(16.);
      seeds.push_back({b[0]+Real(.01)*(b[1]-b[0]),b[2]+eta*eta*(b[3]-b[2]),(b[4]+b[5])/Real(2.)});
    }
  } else throw std::invalid_argument("Unsupported TGV streamline seed layout");
  return seeds;
}
} // namespace astr_insitu
#endif
