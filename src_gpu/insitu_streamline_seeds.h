#ifndef ASTR_INSITU_STREAMLINE_SEEDS_H
#define ASTR_INSITU_STREAMLINE_SEEDS_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace astr_insitu {
// Case presets supply physical coordinates; integrators do not choose seed sites.
inline std::vector<std::array<double,3>> tgv_streamline_seeds(const std::string& layout) {
  const double pi=std::acos(-1.);
  std::vector<std::array<double,3>> seeds;
  if(layout=="line16") {
    seeds.reserve(16);
    for(int i=0;i<16;++i) seeds.push_back({pi/2.,pi/8.+i*(6.*pi/8.)/15.,pi/4.});
  } else if(layout=="tgv-stratified") {
    seeds.reserve(256);
    for(int k=0;k<4;++k) for(int j=0;j<8;++j) for(int i=0;i<8;++i)
      seeds.push_back({(i+.5)*(2.*pi/8.),(j+.5)*(2.*pi/8.),(k+.5)*(2.*pi/4.)});
  } else if(layout=="bl-layered64") {
    seeds.reserve(64);
    for(double y:{.5,1.,2.,5.}) for(int k=0;k<16;++k)
      seeds.push_back({54.,y,90./32.+k*(90.-2.*90./32.)/15.});
  } else throw std::invalid_argument("Unsupported TGV streamline seed layout");
  return seeds;
}
} // namespace astr_insitu
#endif
