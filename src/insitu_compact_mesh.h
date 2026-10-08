#ifndef ASTR_INSITU_COMPACT_MESH_H
#define ASTR_INSITU_COMPACT_MESH_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace astr_insitu {
// Host-owned final geometry only; no solver or three-dimensional field buffer.
struct CompactMesh {
  std::string name,shape;
  std::vector<double> coordinates[3];
  std::vector<std::int64_t> connectivity,seed_ids,directions;
  std::map<std::string,std::vector<double>> point_fields;
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
std::vector<double> device_display_palette();
DeviceMesh build_device_plane(int fcomm,int step,double time,const int global[3],
  const int local[3],const int offset[3],double* coordinates,double* velocity,
  const double origin[3],const double normal[3],std::int64_t budget,
  std::int64_t reserve,std::int64_t retained,std::int64_t host_budget,bool speed_colors=false,double speed_max=1.);
int render_resident_products(const char* pipeline,const char* backend,const char* script,int fcomm,
  int step,double time,const char* profile,std::vector<DeviceMesh> products,bool covered=false);
int render_compact_products(const char* backend,const char* script,int fcomm,
  int step,double time,const char* profile,bool covered,double duration,
  double window_start,double window_end,std::vector<CompactMesh> products);
}
#endif
