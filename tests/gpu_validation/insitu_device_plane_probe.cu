#include "insitu_device_plane.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

using namespace astr_insitu::plane_geometry;

struct Fixture {
  Plane plane;
  Vertex vertices[4];
  int points,triangles;
};
struct Audit {
  int invalid=0,topology=0,winding=0,zero_area=0,identity=0,unchanged=0;
  double plane_error=0.,field_error=0.;
};

double affine(const Point& p) {return 2.*p[0]-3.*p[1]+4.*p[2]+1.;}

std::vector<Fixture> fixtures() {
  const Vertex nodes[4]={{{0.,0.,0.},0},{{1.,0.,0.},1},
    {{0.,1.,0.},2},{{0.,0.,1.},3}};
  struct Case {Point origin,normal;int points,triangles;};
  const Case cases[]={
    {{.25,0.,0.},{1.,0.,0.},3,1},
    {{.5,0.,0.},{1.,1.,0.},4,2},
    {{0.,0.,0.},{1.,1.,1.},1,0},
    {{0.,0.,0.},{1.,1.,0.},2,0},
    {{0.,0.,0.},{1.,0.,0.},3,1},
    {{2.,0.,0.},{1.,0.,0.},0,0},
    {{std::nextafter(1.,0.),0.,0.},{1.,0.,0.},3,1},
  };
  std::vector<Fixture> result;
  for(const auto& c:cases) {
    int permutation[4]={0,1,2,3};
    do {
      Fixture f{};f.plane=make_plane(c.origin,c.normal);
      f.points=c.points;f.triangles=c.triangles;
      for(int d=0;d<4;++d) f.vertices[d]=nodes[permutation[d]];
      result.push_back(f);
    } while(std::next_permutation(permutation,permutation+4));
  }
  return result;
}

void validate(const Fixture& f,const Cut& cut,const Cut& canonical,Audit& a) {
  if(cut.status!=Status::ok) {++a.invalid;return;}
  if(cut.points!=f.points || cut.triangles!=f.triangles) ++a.topology;
  if(cut.points!=canonical.points || cut.triangles!=canonical.triangles) {++a.identity;return;}
  for(int i=0;i<cut.points;++i) {
    const auto& p=cut.vertices[i];
    const auto& ref=canonical.vertices[i];
    if(!(p.key==ref.key) || p.fraction!=ref.fraction) ++a.identity;
    long double distance=0.;
    for(int d=0;d<3;++d) {
      if(p.position[d]!=ref.position[d]) ++a.identity;
      distance+=static_cast<long double>(f.plane.normal[d])*
        (static_cast<long double>(p.position[d])-f.plane.origin[d]);
    }
    a.plane_error=std::max(a.plane_error,double(std::abs(distance)));
    Point lo{},hi{};
    bool found_lo=false,found_hi=false;
    for(const auto& v:f.vertices) {
      if(v.id==p.key.first) {lo=v.position;found_lo=true;}
      if(v.id==p.key.second) {hi=v.position;found_hi=true;}
    }
    if(!found_lo || !found_hi) {++a.identity;continue;}
    const double interpolated=affine(lo)+p.fraction*(affine(hi)-affine(lo));
    a.field_error=std::max(a.field_error,std::abs(interpolated-affine(p.position)));
  }
  for(int t=0;t<cut.triangles;++t) {
    Point e1{},e2{};
    for(int d=0;d<3;++d) {
      e1[d]=cut.vertices[cut.indices[3*t+1]].position[d]-cut.vertices[cut.indices[3*t]].position[d];
      e2[d]=cut.vertices[cut.indices[3*t+2]].position[d]-cut.vertices[cut.indices[3*t]].position[d];
    }
    const long double x=static_cast<long double>(e1[1])*e2[2]-static_cast<long double>(e1[2])*e2[1];
    const long double y=static_cast<long double>(e1[2])*e2[0]-static_cast<long double>(e1[0])*e2[2];
    const long double z=static_cast<long double>(e1[0])*e2[1]-static_cast<long double>(e1[1])*e2[0];
    if(x==0. && y==0. && z==0.) ++a.zero_area;
    if(x*f.plane.normal[0]+y*f.plane.normal[1]+z*f.plane.normal[2]<=0.) ++a.winding;
    for(int j=0;j<3;++j) {
      if(cut.indices[3*t+j]<0 || cut.indices[3*t+j]>=cut.points) ++a.topology;
      if(cut.indices[3*t+j]!=canonical.indices[3*t+j]) ++a.identity;
    }
  }
}

__global__ void generate(const Fixture* input,Cut* output,int count) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count) output[i]=cut_tetrahedron(input[i].plane,input[i].vertices);
}
__global__ void classify(Plane plane,Point p,SignedDistance* output) {
  if(threadIdx.x==0 && blockIdx.x==0) *output=plane_distance(plane,p);
}
struct MeshTriangle {Intersection vertices[3]{};double field[3]{};};
struct MeshCell {MeshTriangle triangles[2]{};int count=0;Status status=Status::ok;};
__host__ __device__ Point mesh_position(const Grid& grid,std::int64_t id,bool curve) {
  const double x=double(id%(grid.cells[0]+1))/grid.cells[0];
  const double y=double(id/(grid.cells[0]+1)%(grid.cells[1]+1))/grid.cells[1];
  const double z=double(id/((grid.cells[0]+1)*(grid.cells[1]+1)))/grid.cells[2];
  return {x,y+(curve?.05*sin(2.*3.14159265358979323846*x)*sin(3.14159265358979323846*y):0.),z};
}
__global__ void cut_mesh(Plane plane,Grid grid,int axis,int rank,int ranks,bool curve,MeshCell* output,int count) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=count) return;
  int extent[3]={int(grid.cells[0]),int(grid.cells[1]),int(grid.cells[2])};extent[axis]/=ranks;
  const int cell=id/6,tet=id%6;
  std::int64_t i=cell%extent[0],j=cell/extent[0]%extent[1],k=cell/(extent[0]*extent[1]);
  if(axis==0) i+=rank*extent[0];
  if(axis==1) j+=rank*extent[1];
  if(axis==2) k+=rank*extent[2];
  Vertex hex[8];
  for(int c=0;c<8;++c) {
    hex[c].id=node_id(grid,i+(c&1),j+((c>>1)&1),k+((c>>2)&1));
    hex[c].position=mesh_position(grid,hex[c].id,curve);
  }
  const auto cut=cut_structured_tetrahedron(plane,grid,hex,cell_id(grid,i,j,k),tet);
  MeshCell result{};result.count=cut.triangles;result.status=cut.status;
  for(int t=0;t<cut.triangles;++t) for(int p=0;p<3;++p) {
    const auto vertex=cut.vertices[cut.indices[3*t+p]];
    result.triangles[t].vertices[p]=vertex;
    double lo=0.,hi=0.;
    for(int c=0;c<8;++c) {
      const auto& x=hex[c].position;const double value=2.*x[0]-3.*x[1]+4.*x[2]+1.;
      if(hex[c].id==vertex.key.first) lo=value;
      if(hex[c].id==vertex.key.second) hi=value;
    }
    result.triangles[t].field[p]=lo+vertex.fraction*(hi-lo);
  }
  output[id]=result;
}
void cuda_require(cudaError_t status,const char* operation) {
  if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}

bool domain_edge(const Grid& grid,const Key& a,const Key& b) {
  for(int d=0;d<3;++d) for(int side=0;side<2;++side) {
    bool on_face=true;
    for(auto id:{a.first,a.second,b.first,b.second}) {
      const std::int64_t index[3]={id%(grid.cells[0]+1),
        (id/(grid.cells[0]+1))%(grid.cells[1]+1),id/((grid.cells[0]+1)*(grid.cells[1]+1))};
      on_face=on_face && index[d]==(side?grid.cells[d]:0);
    }
    if(on_face) return true;
  }
  return false;
}

void mesh_acceptance(int rank,int ranks,int axis,bool curve,bool coincident) {
  const auto grid=make_grid(4,4,4);
  Plane plane{};
  if(rank==0) plane=make_plane({.5,0.,0.},coincident?Point{1.,0.,0.}:Point{1.,.25,-.125});
  MPI_Bcast(&plane,sizeof(plane),MPI_BYTE,0,MPI_COMM_WORLD);
  const int cells=int(grid.cells[0]*grid.cells[1]*grid.cells[2]*6)/ranks;
  MeshCell* device=nullptr;cuda_require(cudaMalloc(&device,cells*sizeof(MeshCell)),"mesh allocation");
  cut_mesh<<<(cells+63)/64,64>>>(plane,grid,axis,rank,ranks,curve,device,cells);
  cuda_require(cudaDeviceSynchronize(),"mesh synchronization");
  std::vector<MeshCell> local(cells),repeated(cells);
  cuda_require(cudaMemcpy(local.data(),device,cells*sizeof(MeshCell),cudaMemcpyDeviceToHost),"test-only mesh readback");
  cut_mesh<<<(cells+63)/64,64>>>(plane,grid,axis,rank,ranks,curve,device,cells);
  cuda_require(cudaDeviceSynchronize(),"mesh repeat synchronization");
  cuda_require(cudaMemcpy(repeated.data(),device,cells*sizeof(MeshCell),cudaMemcpyDeviceToHost),"test-only mesh repeat");
  cuda_require(cudaFree(device),"mesh release");
  int local_triangles=0;for(const auto& cell:local) local_triangles+=cell.count;
  if(ranks==2 && axis==0 && coincident && rank==1 && local_triangles!=0)
    throw std::runtime_error("coincident partition face must have one global owner");
  // A component oracle may gather bounded test geometry; production may not.
  std::vector<MeshCell> all(rank==0?cells*ranks:0),all_repeated(rank==0?cells*ranks:0);
  MPI_Gather(local.data(),cells*sizeof(MeshCell),MPI_BYTE,all.data(),cells*sizeof(MeshCell),MPI_BYTE,0,MPI_COMM_WORLD);
  MPI_Gather(repeated.data(),cells*sizeof(MeshCell),MPI_BYTE,all_repeated.data(),cells*sizeof(MeshCell),MPI_BYTE,0,MPI_COMM_WORLD);
  if(rank!=0) return;
  using Edge=std::array<Key,2>;using Triangle=std::array<Key,3>;
  std::map<Edge,int> edges,edge_directions;std::map<Triangle,int> triangles;
  std::map<Key,Intersection> points;std::map<Key,std::vector<Key>> adjacency,boundary_adjacency;
  Audit audit{};long double area=0.;int count=0;
  for(std::size_t c=0;c<all.size();++c) {
    const auto& cell=all[c];const auto& repeat=all_repeated[c];
    if(cell.status!=Status::ok) {++audit.invalid;continue;}
    if(cell.count!=repeat.count || cell.status!=repeat.status) ++audit.identity;
    for(int t=0;t<cell.count;++t) {
      ++count;const auto& tri=cell.triangles[t];Triangle keys{};
      Point e1{},e2{};
      for(int p=0;p<3;++p) {
        const auto& vertex=tri.vertices[p];const auto& again=repeat.triangles[t].vertices[p];
        keys[p]=vertex.key;long double distance=0.;
        if(!(vertex.key==again.key) || vertex.fraction!=again.fraction ||
            tri.field[p]!=repeat.triangles[t].field[p]) ++audit.identity;
        for(int d=0;d<3;++d) {
          if(vertex.position[d]!=again.position[d]) ++audit.identity;
          distance+=static_cast<long double>(plane.normal[d])*
            (static_cast<long double>(vertex.position[d])-plane.origin[d]);
          e1[d]=tri.vertices[1].position[d]-tri.vertices[0].position[d];
          e2[d]=tri.vertices[2].position[d]-tri.vertices[0].position[d];
        }
        audit.plane_error=std::max(audit.plane_error,double(std::abs(distance)));
        audit.field_error=std::max(audit.field_error,std::abs(tri.field[p]-affine(vertex.position)));
        const auto found=points.find(vertex.key);
        if(found!=points.end()) {
          if(found->second.fraction!=vertex.fraction) ++audit.identity;
          for(int d=0;d<3;++d) if(found->second.position[d]!=vertex.position[d]) ++audit.identity;
        } else points[vertex.key]=vertex;
      }
      std::sort(keys.begin(),keys.end());if(++triangles[keys]!=1) ++audit.topology;
      for(int e=0;e<3;++e) {
        Edge key={tri.vertices[e].key,tri.vertices[(e+1)%3].key};
        const int direction=key[1]<key[0]?-1:1;
        if(direction<0) std::swap(key[0],key[1]);++edges[key];edge_directions[key]+=direction;
      }
      const long double cross[3]={static_cast<long double>(e1[1])*e2[2]-static_cast<long double>(e1[2])*e2[1],
        static_cast<long double>(e1[2])*e2[0]-static_cast<long double>(e1[0])*e2[2],
        static_cast<long double>(e1[0])*e2[1]-static_cast<long double>(e1[1])*e2[0]};
      const long double squared=cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2];
      if(squared==0.) ++audit.zero_area;
      if(cross[0]*plane.normal[0]+cross[1]*plane.normal[1]+cross[2]*plane.normal[2]<=0.) ++audit.winding;
      area+=.5L*std::sqrt(squared);
    }
  }
  int boundary=0;
  for(const auto& edge:edges) {
    const auto& key=edge.first;
    adjacency[key[0]].push_back(key[1]);adjacency[key[1]].push_back(key[0]);
    if(edge.second==1) {
      ++boundary;if(!domain_edge(grid,key[0],key[1])) ++audit.topology;
      boundary_adjacency[key[0]].push_back(key[1]);boundary_adjacency[key[1]].push_back(key[0]);
    } else if(edge.second!=2 || edge_directions[key]!=0) ++audit.topology;
  }
  for(const auto& point:boundary_adjacency) if(point.second.size()!=2) ++audit.topology;
  for(const auto* graph:{&adjacency,&boundary_adjacency}) {
    if(graph->empty()) {++audit.topology;continue;}
    std::map<Key,bool> seen;std::vector<Key> pending{graph->begin()->first};
    while(!pending.empty()) {
      const auto point=pending.back();pending.pop_back();if(seen[point]) continue;seen[point]=true;
      for(const auto& neighbor:graph->at(point)) if(!seen[neighbor]) pending.push_back(neighbor);
    }
    if(seen.size()!=graph->size()) ++audit.topology;
  }
  if(!count || !boundary || int(points.size())-int(edges.size())+int(triangles.size())!=1) ++audit.topology;
  const long double expected=coincident?1.L:std::sqrt(1.L+.25L*.25L+.125L*.125L);
  const double area_error=double(std::abs(area-expected));
  std::printf("X4_PLANE_MESH ranks=%d axis=%d curve=%d coincident=%d triangles=%d points=%zu "
    "topology_errors=%d winding_errors=%d zero_area=%d identity_errors=%d invalid=%d "
    "plane_error=%.17g field_error=%.17g area_error=%.17g test_geometry_readback=1 runtime_admission=0\n",
    ranks,axis,int(curve),int(coincident),count,points.size(),audit.topology,audit.winding,audit.zero_area,
    audit.identity,audit.invalid,audit.plane_error,audit.field_error,area_error);
  if(audit.invalid || audit.topology || audit.winding || audit.zero_area || audit.identity ||
      audit.plane_error>2.e-10 || audit.field_error>2.e-10 || area_error>2.e-10)
    throw std::runtime_error("whole test mesh topology/geometry acceptance failed");
}

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank=0,ranks=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  try {
    int devices=0;cuda_require(cudaGetDeviceCount(&devices),"device count");
    if(devices<ranks || ranks>2) throw std::runtime_error("bounded plane probe requires NP=1/2, one GPU per rank");
    cuda_require(cudaSetDevice(rank),"device selection");
    Plane plane{};
    if(rank==0) plane=make_plane({.25,-2.,7.},{-3.,-3.,1.});
    MPI_Bcast(&plane,sizeof(plane),MPI_BYTE,0,MPI_COMM_WORLD);
    const auto opposite=make_plane({.25,-2.,7.},{3.,3.,-1.});
    for(int d=0;d<3;++d)
      if(plane.normal[d]!=opposite.normal[d] || plane.origin[d]!=opposite.origin[d])
        throw std::runtime_error("collective canonical plane differs");
    double squared=0.;for(double v:plane.normal.values) squared+=v*v;
    if(std::abs(squared-1.)>4.*std::numeric_limits<double>::epsilon() || plane.normal[0]<=0.)
      throw std::runtime_error("plane is not unit/canonical");
    for(const Point normal:{Point{0.,0.,0.},Point{std::numeric_limits<double>::infinity(),0.,0.},
        Point{std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min(),0.}}) {
      bool rejected=false;
      try {make_plane({0.,0.,0.},normal);} catch(const std::invalid_argument&) {rejected=true;}
      if(!rejected) throw std::runtime_error("invalid plane accepted");
    }
    const auto tiny=make_plane({0.,0.,0.},{0.,std::numeric_limits<double>::denorm_min(),0.});
    const auto huge=make_plane({0.,0.,0.},{std::numeric_limits<double>::max(),0.,0.});
    if(tiny.normal[1]!=1. || huge.normal[0]!=1.) throw std::runtime_error("scaled normalization failed");
    const auto input=fixtures();
    const std::size_t input_bytes=input.size()*sizeof(Fixture),output_bytes=input.size()*sizeof(Cut);
    if(input_bytes+output_bytes>2*1024*1024) throw std::runtime_error("probe buffer budget exceeded");
    Fixture* source=nullptr;Cut* destination=nullptr;
    cuda_require(cudaMalloc(&source,input_bytes),"input allocation");
    cuda_require(cudaMalloc(&destination,output_bytes),"output allocation");
    cuda_require(cudaMemcpy(source,input.data(),input_bytes,cudaMemcpyHostToDevice),"test upload");
    generate<<<(input.size()+63)/64,64>>>(source,destination,int(input.size()));
    cuda_require(cudaDeviceSynchronize(),"cut synchronization");
    std::vector<Cut> output(input.size()),repeat(input.size());
    cuda_require(cudaMemcpy(output.data(),destination,output_bytes,cudaMemcpyDeviceToHost),"test-only geometry readback");
    generate<<<(input.size()+63)/64,64>>>(source,destination,int(input.size()));
    cuda_require(cudaDeviceSynchronize(),"repeat synchronization");
    cuda_require(cudaMemcpy(repeat.data(),destination,output_bytes,cudaMemcpyDeviceToHost),"test-only repeat readback");
    std::vector<Fixture> restored(input.size());
    cuda_require(cudaMemcpy(restored.data(),source,input_bytes,cudaMemcpyDeviceToHost),"source verification readback");
    Audit audit{};
    if(std::memcmp(restored.data(),input.data(),input_bytes)!=0) ++audit.unchanged;
    for(std::size_t i=0;i<input.size();++i) {
      const auto canonical=cut_tetrahedron(input[(i/24)*24].plane,input[(i/24)*24].vertices);
      validate(input[i],output[i],canonical,audit);
      validate(input[i],repeat[i],output[i],audit);
      validate(input[i],cut_tetrahedron(input[i].plane,input[i].vertices),output[i],audit);
    }
    cuda_require(cudaFree(destination),"output release");cuda_require(cudaFree(source),"input release");
    SignedDistance* sign_d=nullptr;cuda_require(cudaMalloc(&sign_d,sizeof(SignedDistance)),"predicate allocation");
    const auto cancellation=make_plane({0.,0.,0.},{1.,1.,1.});
    classify<<<1,1>>>(cancellation,{9007199254740992.,1.,-9007199254740992.},sign_d);
    cuda_require(cudaDeviceSynchronize(),"predicate synchronization");
    SignedDistance sign{};
    cuda_require(cudaMemcpy(&sign,sign_d,sizeof(sign),cudaMemcpyDeviceToHost),"predicate scalar readback");
    if(sign.status!=Status::ok || sign.sign!=1 || !sign.exact) ++audit.invalid;
    classify<<<1,1>>>(make_plane({0.,0.,0.},{1.,1.e-300,0.}),{0.,1.e-100,0.},sign_d);
    cuda_require(cudaDeviceSynchronize(),"underflow synchronization");
    cuda_require(cudaMemcpy(&sign,sign_d,sizeof(sign),cudaMemcpyDeviceToHost),"underflow scalar readback");
    if(sign.status!=Status::range) ++audit.invalid;
    cuda_require(cudaFree(sign_d),"predicate release");
    const int failures=audit.invalid+audit.topology+audit.winding+audit.zero_area+audit.identity+audit.unchanged;
    std::printf("X4_PLANE_CORE rank=%d cases=%zu invalid=%d topology_errors=%d winding_errors=%d "
      "zero_area=%d identity_errors=%d source_errors=%d plane_error=%.17g field_error=%.17g "
      "test_geometry_readback=1 runtime_admission=0\n",rank,input.size(),audit.invalid,audit.topology,
      audit.winding,audit.zero_area,audit.identity,audit.unchanged,audit.plane_error,audit.field_error);
    if(failures || audit.plane_error>2.e-10 || audit.field_error>2.e-10)
      throw std::runtime_error("plane component acceptance failed");
    const int axis=argc>1?std::atoi(argv[1]):0;
    if(axis<0 || axis>2) throw std::invalid_argument("Invalid mesh partition axis");
    for(bool curve:{false,true}) for(bool coincident:{false,true})
      mesh_acceptance(rank,ranks,axis,curve,coincident);
    MPI_Finalize();return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"X4 plane rank %d: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
