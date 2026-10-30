#ifndef ASTR_INSITU_COMPACT_MESH_H
#define ASTR_INSITU_COMPACT_MESH_H

#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "astr_precision.h"
namespace viskores {namespace cont {class DataSet;}}

namespace astr_insitu {
using Real=astr::numerical_real;
// Host-owned final geometry only; no solver or three-dimensional field buffer.
struct CompactMesh {
  std::string name,shape;
  std::vector<Real> coordinates[3];
  std::vector<std::int64_t> connectivity,seed_ids,directions;
  std::map<std::string,std::vector<Real>> point_fields;
  std::map<std::string,std::vector<std::int64_t>> cell_fields;
  std::map<std::string,double> controls;
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
  std::string name,actor_key;
  DeviceDrawView draw;
  std::shared_ptr<void> owner;
  std::map<std::string,double> controls;
};
struct RuntimeProductView {
  std::string id,source,definition,directory;
  std::int64_t version=0;
  std::array<std::string,11> labels;
  std::array<int,7> integers;
  std::array<double,31> reals;
  int statistics_provider=0;
  std::int64_t samples=0;
  std::array<double,3> statistics{{0.,0.,0.}};
  std::int64_t host_budget=0;
  std::vector<std::pair<int,std::string>> fields;
  std::vector<std::string> units;
  std::array<double,12> references{};
  std::string unit_system;
};
RuntimeProductView runtime_product_view(const char*,const char*,const char*,const char*,std::int64_t,
  const char*,const int*,const double*);
DeviceMesh build_runtime_index_mesh(const RuntimeProductView&,int,Real,const int*,int,int,int,Real*,Real*,const Real*,int=0);
void begin_runtime_index_group();
void end_runtime_index_group();
DeviceMesh build_runtime_physical_mesh(const RuntimeProductView&,int,Real,const int*,int,const int*,const int*,
  const int*,Real*,Real*,const Real*,std::int64_t,std::int64_t,int=0);
viskores::cont::DataSet clip_runtime_device_mesh(viskores::cont::DataSet,const RuntimeProductView&,int=3);
double runtime_product_clock();
void prepare_runtime_product_allocations(const RuntimeProductView&,int,int);
int publish_runtime_product(const char*,const char*,const char*,int,int,Real,RuntimeProductView&,DeviceMesh,std::string,double);
void clear_runtime_compatible_geometry();
std::shared_ptr<CompactMesh> retained_runtime_geometry();
struct StreamlineView {
  std::string id,velocity,seeds,color;
  std::array<Real,19> values;
};
const std::vector<StreamlineView>& streamline_views();
bool streamline_preview_active();
bool retain_quantitative_product_fields();
std::vector<double> device_display_palette(const std::string& color_map="diverging_blue_red");
DeviceMesh build_device_plane(int fcomm,int step,Real time,const int global[3],
  const int local[3],const int offset[3],Real* coordinates,Real* velocity,
  const Real origin[3],const Real normal[3],std::int64_t budget,
  std::int64_t reserve,std::int64_t retained,std::int64_t host_budget,bool speed_colors=false,Real speed_max=1);
int render_resident_products(const char* pipeline,const char* backend,const char* script,int fcomm,
  int step,Real time,const char* profile,std::vector<DeviceMesh> products,bool covered=false,bool complete=true,
  const RuntimeProductView* runtime=nullptr);
int render_compact_products(const char* backend,const char* script,int fcomm,
  int step,Real time,const char* profile,bool covered,Real duration,
  Real window_start,Real window_end,std::vector<CompactMesh> products,
  const RuntimeProductView* runtime=nullptr,std::shared_ptr<CompactMesh> frozen={});
}
#endif
