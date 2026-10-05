#ifndef ASTR_INSITU_COMPACT_MESH_H
#define ASTR_INSITU_COMPACT_MESH_H

#include <cstdint>
#include <map>
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
int render_compact_products(const char* backend,const char* script,int fcomm,
  int step,double time,const char* profile,bool covered,double duration,
  double window_start,double window_end,std::vector<CompactMesh> products);
}
#endif
