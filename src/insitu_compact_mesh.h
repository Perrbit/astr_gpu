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
int render_resident_products(const char* pipeline,const char* backend,const char* script,int fcomm,
  int step,double time,const char* profile,std::vector<DeviceMesh> products,bool covered=false);
int render_compact_products(const char* backend,const char* script,int fcomm,
  int step,double time,const char* profile,bool covered,double duration,
  double window_start,double window_end,std::vector<CompactMesh> products);
}
#endif
