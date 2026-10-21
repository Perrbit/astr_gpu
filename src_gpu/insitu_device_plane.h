#ifndef ASTR_INSITU_DEVICE_PLANE_H
#define ASTR_INSITU_DEVICE_PLANE_H

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include "astr_precision.h"

#ifdef __CUDACC__
#define ASTR_PLANE_EXEC __host__ __device__
#else
#define ASTR_PLANE_EXEC
#endif

namespace astr_insitu {
using Real=astr::numerical_real;
namespace plane_geometry {
constexpr Real real_max=std::numeric_limits<Real>::max();
constexpr Real real_min=std::numeric_limits<Real>::min();
constexpr Real real_epsilon=std::numeric_limits<Real>::epsilon();
constexpr int product_min_exponent=std::numeric_limits<Real>::min_exponent+std::numeric_limits<Real>::digits;

struct Point {
  Real values[3];
  ASTR_PLANE_EXEC Real& operator[](int d) {return values[d];}
  ASTR_PLANE_EXEC Real operator[](int d) const {return values[d];}
};
struct Plane {Point origin,normal;};
struct Vertex {Point position;std::int64_t id;};
enum class Status : int {ok=0,invalid=1,range=2};
struct SignedDistance {Real value=0;int sign=0;Status status=Status::ok;bool exact=false;};
struct Key {
  std::int64_t first=0,second=0;
  ASTR_PLANE_EXEC bool operator==(const Key& other) const {
    return first==other.first && second==other.second;
  }
  ASTR_PLANE_EXEC bool operator<(const Key& other) const {
    return first<other.first || (first==other.first && second<other.second);
  }
};
struct Intersection {Point position{};Key key{};Real fraction=0;};
struct Cut {
  Intersection vertices[4]{};
  int indices[6]{},points=0,triangles=0;
  bool coincident_face=false;
  Status status=Status::ok;
};

inline Plane make_plane(const Point& origin,const Point& normal) {
  Real scale=0;int axis=0;
  for(int d=0;d<3;++d) {
    if(!std::isfinite(origin[d]) || !std::isfinite(normal[d]))
      throw std::invalid_argument("Nonfinite physical plane");
    if(std::abs(normal[d])>scale) {scale=std::abs(normal[d]);axis=d;}
  }
  if(scale==0) throw std::invalid_argument("Zero physical plane normal");
  Plane result{origin,{}};
  Real squared=0;
  for(int d=0;d<3;++d) {
    result.normal[d]=normal[d]/scale;
    if(normal[d]!=0 && result.normal[d]==0)
      throw std::invalid_argument("Physical plane normal component is not representable");
    squared+=result.normal[d]*result.normal[d];
  }
  const Real length=std::sqrt(squared),direction=normal[axis]<0?-1:1;
  for(int d=0;d<3;++d) {
    result.normal[d]=(result.normal[d]/length)*direction;
    if(normal[d]!=0 && result.normal[d]==0)
      throw std::invalid_argument("Unit plane normal component is not representable");
    if(result.normal[d]==0) result.normal[d]=0;
  }
  return result;
}

ASTR_PLANE_EXEC inline bool finite(Real x) {return x==x && std::abs(x)<=real_max;}
ASTR_PLANE_EXEC inline int sign(Real x) {return (x>0)-(x<0);}

// Error-free transforms require round-to-nearest, no fast-math or implicit FMA.
ASTR_PLANE_EXEC inline void two_sum(Real a,Real b,Real& sum,Real& error) {
  sum=a+b;
  const Real bv=sum-a,av=sum-bv,br=b-bv,ar=a-av;
  error=ar+br;
}
ASTR_PLANE_EXEC inline void two_diff(Real a,Real b,Real& difference,Real& error) {
  difference=a-b;
  const Real bv=a-difference,av=difference+bv,br=bv-b,ar=a-av;
  error=ar+br;
}

struct Expansion {
  Real values[32]{};
  int size=0;
  Status status=Status::ok;
  ASTR_PLANE_EXEC void add(Real term) {
    if(!finite(term)) {status=Status::range;return;}
    Real result[32]{},q=term;
    int count=0;
    for(int i=0;i<size;++i) {
      Real next,error;two_sum(q,values[i],next,error);
      if(!finite(next) || !finite(error)) {status=Status::range;return;}
      if(error!=0) {
        if(count==32) {status=Status::range;return;}
        result[count++]=error;
      }
      q=next;
    }
    if(q!=0) {
      if(count==32) {status=Status::range;return;}
      result[count++]=q;
    }
    size=count;for(int i=0;i<count;++i) values[i]=result[i];
  }
  ASTR_PLANE_EXEC void product(Real a,Real b,Real direction=1) {
    if(a==0 || b==0) return;
    int ea=0,eb=0;std::frexp(a,&ea);std::frexp(b,&eb);
    // The lowest product bit of two native significands must be representable.
    if(ea+eb<product_min_exponent) {status=Status::range;return;}
    const Real rounded=a*b;
    // A lost underflow residual cannot be certified by this native expansion.
    if(!finite(rounded) || std::abs(rounded)<real_min) {status=Status::range;return;}
    const Real error=std::fma(a,b,-rounded);
    if(!finite(error) || (error!=0 && std::abs(error)<real_min)) {status=Status::range;return;}
    add(direction*error);add(direction*rounded);
  }
  ASTR_PLANE_EXEC SignedDistance result() const {
    SignedDistance output{};output.status=status;output.exact=true;
    if(status!=Status::ok || size==0) return output;
    output.sign=sign(values[size-1]);
    for(int i=0;i<size;++i) output.value+=values[i];
    if(!finite(output.value)) {output.status=Status::range;return output;}
    if(output.value==0) output.value=values[size-1];
    return output;
  }
};

ASTR_PLANE_EXEC inline SignedDistance plane_distance(const Plane& plane,const Point& point) {
  SignedDistance result{};
  Real scale=0,value=0;bool underflow=false;
  for(int d=0;d<3;++d) {
    if(!finite(point[d]) || !finite(plane.origin[d]) || !finite(plane.normal[d])) {
      result.status=Status::invalid;return result;
    }
    const Real difference=point[d]-plane.origin[d],term=plane.normal[d]*difference;
    if(!finite(difference) || !finite(term)) {result.status=Status::range;return result;}
    underflow=underflow || (difference!=0 && plane.normal[d]!=0 && std::abs(term)<real_min);
    value+=term;
    scale+=std::abs(plane.normal[d])*std::abs(point[d])+std::abs(plane.normal[d])*std::abs(plane.origin[d]);
  }
  if(!finite(value) || !finite(scale)) {result.status=Status::range;return result;}
  const Real bound=16*real_epsilon*scale+4*real_min;
  if(!underflow && std::abs(value)>bound) {
    result.value=value;result.sign=sign(value);return result;
  }
  Expansion exact;
  for(int d=0;d<3;++d) {
    Real difference,error;two_diff(point[d],plane.origin[d],difference,error);
    exact.product(plane.normal[d],error);exact.product(plane.normal[d],difference);
  }
  return exact.result();
}

ASTR_PLANE_EXEC inline int dominant_axis(const Plane& plane) {
  int axis=0;
  for(int d=1;d<3;++d) if(std::abs(plane.normal[d])>std::abs(plane.normal[axis])) axis=d;
  return axis;
}

ASTR_PLANE_EXEC inline SignedDistance orientation(const Plane& plane,const Point& a,
    const Point& b,const Point& c) {
  const int axis=dominant_axis(plane),u=(axis+1)%3,v=(axis+2)%3;
  Real bu[2],bv[2],cu[2],cv[2];
  two_diff(b[u],a[u],bu[0],bu[1]);two_diff(b[v],a[v],bv[0],bv[1]);
  two_diff(c[u],a[u],cu[0],cu[1]);two_diff(c[v],a[v],cv[0],cv[1]);
  Expansion exact;
  for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
    exact.product(bu[i],cv[j]);exact.product(bv[i],cu[j],-1);
  }
  auto result=exact.result();
  if(plane.normal[axis]<0) {result.value=-result.value;result.sign=-result.sign;}
  return result;
}

ASTR_PLANE_EXEC inline Status tetrahedron_range(const Vertex* vertices,int* orientation=nullptr) {
  Real edge[3][3],scale=0;
  for(int i=0;i<4;++i) {
    if(vertices[i].id<0) return Status::invalid;
    for(int j=0;j<i;++j) if(vertices[i].id==vertices[j].id) return Status::invalid;
    for(int d=0;d<3;++d) if(!finite(vertices[i].position[d])) return Status::invalid;
  }
  for(int i=0;i<3;++i) for(int d=0;d<3;++d) {
    edge[i][d]=vertices[i+1].position[d]-vertices[0].position[d];
    if(!finite(edge[i][d])) return Status::range;
    if(std::abs(edge[i][d])>scale) scale=std::abs(edge[i][d]);
  }
  if(scale==0) return Status::invalid;
  for(int i=0;i<3;++i) for(int d=0;d<3;++d) edge[i][d]/=scale;
  const Real terms[6]={edge[0][0]*edge[1][1]*edge[2][2],edge[0][1]*edge[1][2]*edge[2][0],
    edge[0][2]*edge[1][0]*edge[2][1],edge[0][2]*edge[1][1]*edge[2][0],
    edge[0][1]*edge[1][0]*edge[2][2],edge[0][0]*edge[1][2]*edge[2][1]};
  Real determinant=0,permanent=0;
  for(int i=0;i<6;++i) {determinant+=(i<3?terms[i]:-terms[i]);permanent+=std::abs(terms[i]);}
  // Unresolved or collapsed cells are rejected, never silently repaired.
  if(std::abs(determinant)<=64*real_epsilon*permanent || !finite(determinant)) return Status::range;
  if(orientation) *orientation=sign(determinant);
  return Status::ok;
}

ASTR_PLANE_EXEC inline Status insert(Cut& cut,const Intersection& intersection) {
  for(int i=0;i<cut.points;++i) if(cut.vertices[i].key==intersection.key) return Status::ok;
  if(cut.points==4) return Status::invalid;
  cut.vertices[cut.points++]=intersection;return Status::ok;
}

ASTR_PLANE_EXEC inline Cut cut_tetrahedron(const Plane& plane,const Vertex* vertices) {
  Cut cut{};cut.status=tetrahedron_range(vertices);
  if(cut.status!=Status::ok) return cut;
  SignedDistance distance[4];
  for(int i=0;i<4;++i) {
    distance[i]=plane_distance(plane,vertices[i].position);
    if(distance[i].status!=Status::ok) {cut.status=distance[i].status;return cut;}
    if(distance[i].sign==0) {
      Intersection p{};p.position=vertices[i].position;p.key={vertices[i].id,vertices[i].id};
      cut.status=insert(cut,p);if(cut.status!=Status::ok) return cut;
    }
  }
  for(int i=0;i<4;++i) for(int j=i+1;j<4;++j) {
    if(distance[i].sign*distance[j].sign>=0) continue;
    const int lo=vertices[i].id<vertices[j].id?i:j,hi=lo==i?j:i;
    const Real a=std::abs(distance[lo].value),b=std::abs(distance[hi].value),scale=a>b?a:b;
    const Real denominator=a/scale+b/scale,t=(a/scale)/denominator;
    if(!finite(t) || t<=0 || t>=1) {cut.status=Status::range;return cut;}
    Intersection p{};p.key={vertices[lo].id,vertices[hi].id};p.fraction=t;
    for(int d=0;d<3;++d) {
      const Real delta=vertices[hi].position[d]-vertices[lo].position[d];
      p.position[d]=vertices[lo].position[d]+t*delta;
      if(!finite(delta) || !finite(p.position[d])) {cut.status=Status::range;return cut;}
    }
    cut.status=insert(cut,p);if(cut.status!=Status::ok) return cut;
  }
  if(cut.points<3) {
    if(cut.points==2 && cut.vertices[1].key<cut.vertices[0].key) {
      const auto p=cut.vertices[0];cut.vertices[0]=cut.vertices[1];cut.vertices[1]=p;
    }
    return cut;
  }
  const int axis=dominant_axis(plane),u=(axis+1)%3,v=(axis+2)%3;
  int pivot=0;
  for(int i=1;i<cut.points;++i) {
    const auto& a=cut.vertices[i];const auto& b=cut.vertices[pivot];
    if(a.position[u]<b.position[u] || (a.position[u]==b.position[u] &&
        (a.position[v]<b.position[v] || (a.position[v]==b.position[v] && a.key<b.key)))) pivot=i;
  }
  const auto first=cut.vertices[0];cut.vertices[0]=cut.vertices[pivot];cut.vertices[pivot]=first;
  for(int i=2;i<cut.points;++i) {
    const auto value=cut.vertices[i];int j=i;
    while(j>1) {
      const auto turn=orientation(plane,cut.vertices[0].position,cut.vertices[j-1].position,value.position);
      if(turn.status!=Status::ok || turn.sign==0) {cut.status=Status::range;return cut;}
      if(turn.sign>0) break;
      cut.vertices[j]=cut.vertices[j-1];--j;
    }
    cut.vertices[j]=value;
  }
  int smallest=0;
  for(int i=1;i<cut.points;++i) if(cut.vertices[i].key<cut.vertices[smallest].key) smallest=i;
  Intersection ordered[4];
  for(int i=0;i<cut.points;++i) ordered[i]=cut.vertices[(i+smallest)%cut.points];
  for(int i=0;i<cut.points;++i) cut.vertices[i]=ordered[i];
  cut.triangles=cut.points-2;
  for(int t=0;t<cut.triangles;++t) {
    const auto turn=orientation(plane,cut.vertices[0].position,cut.vertices[t+1].position,
      cut.vertices[t+2].position);
    if(turn.status!=Status::ok || turn.sign<=0) {cut.status=Status::range;cut.triangles=0;return cut;}
    cut.indices[3*t]=0;cut.indices[3*t+1]=t+1;cut.indices[3*t+2]=t+2;
  }
  cut.coincident_face=cut.points==3;
  for(int i=0;i<cut.points;++i)
    cut.coincident_face=cut.coincident_face && cut.vertices[i].key.first==cut.vertices[i].key.second;
  return cut;
}

ASTR_PLANE_EXEC inline int tetrahedron_corner(int tet,int corner) {
  // Freudenthal decomposition: shared quadrilateral faces use the same diagonal.
  const int corners[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},
    {0,6,4,7},{0,4,5,7},{0,5,1,7}};
  return corners[tet][corner];
}

struct Grid {
  std::int64_t cells[3];
};
inline Grid make_grid(std::int64_t nx,std::int64_t ny,std::int64_t nz) {
  Grid grid{{nx,ny,nz}};std::int64_t nodes=1,cells=1;
  for(auto n:grid.cells) {
    if(n<=0 || n==INT64_MAX || nodes>INT64_MAX/(n+1) || cells>INT64_MAX/6/n)
      throw std::invalid_argument("Invalid static slice grid extent");
    nodes*=n+1;cells*=n;
  }
  return grid;
}
ASTR_PLANE_EXEC inline std::int64_t node_id(const Grid& grid,std::int64_t i,
    std::int64_t j,std::int64_t k) {
  return i+(grid.cells[0]+1)*(j+(grid.cells[1]+1)*k);
}
ASTR_PLANE_EXEC inline std::int64_t cell_id(const Grid& grid,std::int64_t i,
    std::int64_t j,std::int64_t k) {
  return i+grid.cells[0]*(j+grid.cells[1]*k);
}
ASTR_PLANE_EXEC inline void sort_face(std::int64_t* ids) {
  for(int i=1;i<3;++i) {
    const auto value=ids[i];int j=i;
    while(j>0 && value<ids[j-1]) {ids[j]=ids[j-1];--j;}
    ids[j]=value;
  }
}

// Resolve an exactly coincident subface from global connectivity alone. The
// owner is independent of partition, normal direction and MPI rank numbering.
ASTR_PLANE_EXEC inline bool owns_coincident_face(const Grid& grid,const Cut& cut,
    std::int64_t this_tet) {
  std::int64_t ids[3],lower[3],upper[3];
  for(int p=0;p<3;++p) ids[p]=cut.vertices[p].key.first;
  sort_face(ids);
  for(int d=0;d<3;++d) {lower[d]=INT64_MAX;upper[d]=0;}
  for(int p=0;p<3;++p) {
    const std::int64_t index[3]={ids[p]%(grid.cells[0]+1),
      (ids[p]/(grid.cells[0]+1))%(grid.cells[1]+1),
      ids[p]/((grid.cells[0]+1)*(grid.cells[1]+1))};
    for(int d=0;d<3;++d) {
      if(index[d]<lower[d]) lower[d]=index[d];
      if(index[d]>upper[d]) upper[d]=index[d];
    }
  }
  std::int64_t first[3],last[3],owner=INT64_MAX;
  for(int d=0;d<3;++d) {
    first[d]=lower[d]-(lower[d]==upper[d]?1:0);
    if(first[d]<0) first[d]=0;
    last[d]=lower[d];if(last[d]>=grid.cells[d]) last[d]=grid.cells[d]-1;
  }
  for(auto k=first[2];k<=last[2];++k) for(auto j=first[1];j<=last[1];++j)
    for(auto i=first[0];i<=last[0];++i) {
      std::int64_t nodes[8];
      for(int corner=0;corner<8;++corner)
        nodes[corner]=node_id(grid,i+(corner&1),j+((corner>>1)&1),k+((corner>>2)&1));
      for(int tet=0;tet<6;++tet) for(int excluded=0;excluded<4;++excluded) {
        std::int64_t face[3];int n=0;
        for(int p=0;p<4;++p) if(p!=excluded) face[n++]=nodes[tetrahedron_corner(tet,p)];
        sort_face(face);
        if(face[0]==ids[0] && face[1]==ids[1] && face[2]==ids[2]) {
          const auto candidate=6*cell_id(grid,i,j,k)+tet;
          if(candidate<owner) owner=candidate;
        }
      }
    }
  return this_tet==owner;
}

ASTR_PLANE_EXEC inline Cut cut_structured_tetrahedron(const Plane& plane,const Grid& grid,
    const Vertex* hex,std::int64_t global_cell,int tet) {
  Cut cut{};
  if(tet<0 || tet>=6 || global_cell<0 ||
      global_cell>=grid.cells[0]*grid.cells[1]*grid.cells[2]) {
    cut.status=Status::invalid;return cut;
  }
  Vertex vertices[4];
  for(int p=0;p<4;++p) vertices[p]=hex[tetrahedron_corner(tet,p)];
  int oriented=0;
  cut.status=tetrahedron_range(vertices,&oriented);
  if(cut.status!=Status::ok) return cut;
  if(oriented<=0) {cut.status=Status::invalid;return cut;}
  cut=cut_tetrahedron(plane,vertices);
  if(cut.status==Status::ok && cut.coincident_face &&
      !owns_coincident_face(grid,cut,6*global_cell+tet)) return Cut{};
  return cut;
}

} // namespace plane_geometry
} // namespace astr_insitu

#undef ASTR_PLANE_EXEC
#endif
