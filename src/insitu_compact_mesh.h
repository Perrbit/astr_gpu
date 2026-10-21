#ifndef ASTR_INSITU_COMPACT_MESH_H
#define ASTR_INSITU_COMPACT_MESH_H

#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "astr_precision.h"

namespace astr_insitu {
using Real=astr::numerical_real;
// Host-owned final geometry only; no solver or three-dimensional field buffer.
struct CompactMesh {
  std::string name,shape;
  std::vector<Real> coordinates[3];
  std::vector<std::int64_t> connectivity,seed_ids,directions;
  std::map<std::string,std::vector<Real>> point_fields;
};
// A backend-neutral, borrowed device view. No graphics handles cross this API.
struct DeviceDrawView {
  const float* positions=nullptr;
  const unsigned char* colors=nullptr;
  const unsigned int* indices=nullptr;
  std::size_t points=0,cells=0;
  int arity=3,device=-1;
  double bounds[6]={0.,0.,0.,0.,0.,0.};
};
struct DeviceMesh {
  std::string name;
  DeviceDrawView draw;
  std::shared_ptr<void> owner;
  std::map<std::string,double> controls;
};
struct StreamlineView {
  std::string id,velocity,seeds,color;
  std::array<Real,19> values;
};
const std::vector<StreamlineView>& streamline_views();
bool streamline_preview_active();
bool retain_quantitative_product_fields();
std::vector<double> device_display_palette();
DeviceMesh build_device_plane(int fcomm,int step,Real time,const int global[3],
  const int local[3],const int offset[3],Real* coordinates,Real* velocity,
  const Real origin[3],const Real normal[3],std::int64_t budget,
  std::int64_t reserve,std::int64_t retained,std::int64_t host_budget,bool speed_colors=false,Real speed_max=1);
int render_resident_products(const char* pipeline,const char* backend,const char* script,int fcomm,
  int step,Real time,const char* profile,std::vector<DeviceMesh> products,bool covered=false,bool complete=true);
int render_compact_products(const char* backend,const char* script,int fcomm,
  int step,Real time,const char* profile,bool covered,Real duration,
  Real window_start,Real window_end,std::vector<CompactMesh> products);
}
#endif
