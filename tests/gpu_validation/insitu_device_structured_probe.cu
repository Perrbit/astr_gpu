#include "insitu_device_structured.h"
#include <viskores/cont/Initialize.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <cstdio>
#include <math_constants.h>

using namespace astr_insitu;
using namespace structured_geometry;
using Key=std::array<std::int64_t,2>;
struct Triangle {
  Key keys[3];
  double xyz[3][3],velocity[3][3],q[3];
};
__host__ __device__ viskores::Vec3f position(int i,int j,int k,bool curve) {
  const double x=i/4.,y=j/4.,z=k/4.;
  return viskores::Vec3f(x,y+(curve?.05*sin(2.*3.14159265358979323846*x)*sin(3.14159265358979323846*y):0.),z);
}
__host__ __device__ viskores::Vec3f affine(const viskores::Vec3f& p) {
  return viskores::Vec3f(2.*p[0]-3.*p[1]+4.*p[2]+1.,p[0]+p[2],p[1]-2.*p[2]);
}
__host__ __device__ double scalar(const viskores::Vec3f& p) {return -p[0]+2.*p[1]+.5*p[2];}
__global__ void initialize(viskores::Vec3f* xyz,viskores::Vec3f* v,DeviceDiagnostics* diagnostics,
    Partition partition,bool curve) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=partition.local_nodes()) return;
  const auto nx=partition.cells[0]+1,ny=partition.cells[1]+1;
  const auto p=position(int(id%nx+partition.offset[0]),int(id/nx%ny+partition.offset[1]),
    int(id/(nx*ny)+partition.offset[2]),curve);
  xyz[id]=p;v[id]=affine(p);diagnostics[id]=DeviceDiagnostics(0.);diagnostics[id][9]=scalar(p);
}
__global__ void check_source(const viskores::Vec3f* xyz,const viskores::Vec3f* v,
    const DeviceDiagnostics* diagnostics,Partition partition,bool curve,int* bad) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=partition.local_nodes()) return;
  const auto nx=partition.cells[0]+1,ny=partition.cells[1]+1;
  const auto p=position(int(id%nx+partition.offset[0]),int(id/nx%ny+partition.offset[1]),
    int(id/(nx*ny)+partition.offset[2]),curve);
  if(xyz[id]!=p || v[id]!=affine(p)) atomicAdd(bad,1);
  for(int d=0;d<14;++d) if(diagnostics[id][d]!=(d==9?scalar(p):0.)) atomicAdd(bad,1);
}
__global__ void corrupt_source(viskores::Vec3f* xyz,viskores::Vec3f* v,bool geometry) {
  if(threadIdx.x==0 && blockIdx.x==0) {
    if(geometry) xyz[0][0]=2.;
    else v[2][0]=CUDART_INF;
  }
}
void cuda_require(cudaError_t status,const char* message) {
  if(status!=cudaSuccess) throw std::runtime_error(std::string(message)+": "+cudaGetErrorString(status));
}
struct Source {
  viskores::Vec3f *xyz=nullptr,*v=nullptr;
  DeviceDiagnostics* diagnostics=nullptr;
  int* bad=nullptr;
  explicit Source(viskores::Id n) {
    cuda_require(cudaMalloc(&xyz,n*sizeof(*xyz)),"source coordinate allocation");
    cuda_require(cudaMalloc(&v,n*sizeof(*v)),"source velocity allocation");
    cuda_require(cudaMalloc(&diagnostics,n*sizeof(*diagnostics)),"source diagnostic allocation");
    cuda_require(cudaMalloc(&bad,sizeof(int)),"source check allocation");
  }
  ~Source() {cudaFree(xyz);cudaFree(v);cudaFree(diagnostics);cudaFree(bad);}
};
struct Readback {
  std::vector<viskores::Vec3f> xyz,velocity;
  std::vector<double> q;
  std::vector<PlaneKey> keys;
  std::vector<viskores::Id> indices;
};
Readback test_readback(const PlaneProduct& product) {
  Readback result;
  DeviceGeometryOwner owner(product.data,4,.004);
  const auto& view=owner.get();
  result.xyz.resize(view.points);result.velocity.resize(view.points);result.q.resize(view.points);
  result.keys.resize(view.points);result.indices.resize(view.cells*3);
  if(view.points) {
    cuda_require(cudaMemcpy(result.xyz.data(),view.coordinates,view.points*sizeof(*view.coordinates),
      cudaMemcpyDeviceToHost),"test-only xyz download");
    cuda_require(cudaMemcpy(result.velocity.data(),view.velocity,view.points*sizeof(*view.velocity),
      cudaMemcpyDeviceToHost),"test-only velocity download");
    cuda_require(cudaMemcpy(result.q.data(),view.q,view.points*sizeof(*view.q),cudaMemcpyDeviceToHost),
      "test-only scalar download");
    cuda_require(cudaMemcpy(result.indices.data(),view.connectivity,view.cells*3*sizeof(*view.connectivity),
      cudaMemcpyDeviceToHost),"test-only indices download");
    const auto keys=product.keys.ReadPortal();
    for(viskores::Id i=0;i<view.points;++i) result.keys[i]=keys.Get(i);
  }
  return result;
}
template<class T> bool identical(const std::vector<T>& a,const std::vector<T>& b) {
  return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));
}
bool outer_edge(const Key& a,const Key& b) {
  for(int d=0;d<3;++d) for(int side:{0,4}) {
    bool on=true;
    for(auto id:{a[0],a[1],b[0],b[1]}) {
      const std::int64_t index[3]={id%5,id/5%5,id/25};
      on=on && index[d]==side;
    }
    if(on) return true;
  }
  return false;
}
void rejections(int rank) {
  const Partition partition(viskores::Id3(4),viskores::Id3(4),viskores::Id3(0));
  Source source(partition.local_nodes());
  const plane_geometry::Point origin{.5,0.,0.},normal{1.,0.,0.};
  const auto initialize_source=[&]() {
    initialize<<<(partition.local_nodes()+127)/128,128>>>(source.xyz,source.v,source.diagnostics,partition,false);
    synchronize_device_stage("rejection fixture");
  };
  const auto expect_reject=[&](const char* message,const plane_geometry::Point& n,std::uint64_t budget) {
    bool rejected=false;
    try {extract_plane(source.xyz,source.v,source.diagnostics,partition,origin,n,budget);}
    catch(const std::exception& error) {rejected=std::string(error.what()).find(message)!=std::string::npos;}
    if(!rejected) throw std::runtime_error(std::string("Missing rejection: ")+message);
  };
  initialize_source();
  expect_reject("Zero physical plane normal",{0.,0.,0.},2097152);
  expect_reject("controlled-array budget",normal,1);
  corrupt_source<<<1,1>>>(source.xyz,source.v,true);synchronize_device_stage("invalid geometry fixture");
  expect_reject("Invalid structured plane geometry/range",normal,2097152);
  initialize_source();
  corrupt_source<<<1,1>>>(source.xyz,source.v,false);synchronize_device_stage("invalid field fixture");
  expect_reject("Nonfinite structured plane field/intersection",normal,2097152);
  std::printf("X4_STRUCTURED_REJECTION rank=%d zero_normal=1 budget=1 inverted_cell=1 nonfinite_field=1\n",rank);
}
void acceptance(int rank,int ranks,int axis,bool curve,int plane_kind) {
  viskores::Id3 cells(4),offset(0);cells[axis]/=ranks;offset[axis]=rank*cells[axis];
  const Partition partition(viskores::Id3(4),cells,offset);
  Source source(partition.local_nodes());
  initialize<<<(partition.local_nodes()+127)/128,128>>>(source.xyz,source.v,source.diagnostics,partition,curve);
  synchronize_device_stage("structured fixture");
  const plane_geometry::Point origin={plane_kind==2?2.:.5,0.,0.};
  const plane_geometry::Point normal={1.,plane_kind==0?.25:0.,plane_kind==0?-.125:0.};
  const auto plane=plane_geometry::make_plane(origin,normal);
  const auto product=extract_plane(source.xyz,source.v,source.diagnostics,partition,origin,normal,2097152);
  const auto data=test_readback(product);
  const auto repeated=extract_plane(source.xyz,source.v,source.diagnostics,partition,origin,normal,2097152);
  const auto repeat=test_readback(repeated);
  if(!identical(data.xyz,repeat.xyz) || !identical(data.velocity,repeat.velocity) || !identical(data.q,repeat.q) ||
      !identical(data.keys,repeat.keys) || !identical(data.indices,repeat.indices))
    throw std::runtime_error("structured repeated output differs");
  for(std::size_t p=1;p<data.keys.size();++p)
    if(!KeyLess{}(data.keys[p-1],data.keys[p])) throw std::runtime_error("local plane keys are not unique/ordered");
  if(plane_kind==1 && ranks==2 && axis==0 && rank==1 && !data.indices.empty())
    throw std::runtime_error("coincident high-rank product should be empty");
  cuda_require(cudaMemset(source.bad,0,sizeof(int)),"source check reset");
  check_source<<<(partition.local_nodes()+127)/128,128>>>(source.xyz,source.v,source.diagnostics,partition,curve,source.bad);
  synchronize_device_stage("check source");
  int bad=0;cuda_require(cudaMemcpy(&bad,source.bad,sizeof(int),cudaMemcpyDeviceToHost),"source check scalar");
  if(bad) throw std::runtime_error("structured extraction modified its source");
  std::vector<Triangle> local(data.indices.size()/3);
  for(std::size_t t=0;t<local.size();++t) for(int p=0;p<3;++p) {
    const auto id=data.indices[3*t+p];
    if(id<0 || static_cast<std::size_t>(id)>=data.xyz.size()) throw std::runtime_error("invalid plane index");
    local[t].keys[p]={data.keys[id][0],data.keys[id][1]};local[t].q[p]=data.q[id];
    for(int d=0;d<3;++d) {local[t].xyz[p][d]=data.xyz[id][d];local[t].velocity[p][d]=data.velocity[id][d];}
  }
  int bytes=int(local.size()*sizeof(Triangle));
  std::vector<int> sizes(ranks),starts(ranks);
  MPI_Allgather(&bytes,1,MPI_INT,sizes.data(),1,MPI_INT,MPI_COMM_WORLD);
  int total=0;for(int r=0;r<ranks;++r) {starts[r]=total;total+=sizes[r];}
  std::vector<Triangle> all(total/sizeof(Triangle));
  MPI_Allgatherv(local.data(),bytes,MPI_BYTE,all.data(),sizes.data(),starts.data(),MPI_BYTE,MPI_COMM_WORLD);
  if(rank!=0) return;
  if(plane_kind==2) {
    if(!all.empty()) throw std::runtime_error("outside plane produced geometry");
    std::printf("X4_STRUCTURED ranks=%d axis=%d curve=%d plane=empty errors=0 device_only_before_oracle=1 "
      "test_geometry_readback=1 runtime_admission=0\n",ranks,axis,int(curve));return;
  }
  std::set<std::array<Key,3>> triangles;
  std::map<Key,std::array<double,7>> points;
  std::map<std::array<Key,2>,std::pair<int,int>> edges;
  std::map<Key,std::vector<Key>> graph,boundary;
  long double area=0.;double error=0.;
  for(const auto& triangle:all) {
    std::array<Key,3> tri;
    long double a[3],b[3];
    for(int p=0;p<3;++p) {
      tri[p]=triangle.keys[p];const auto* position=triangle.xyz[p];
      std::array<double,7> value{position[0],position[1],position[2],triangle.velocity[p][0],
        triangle.velocity[p][1],triangle.velocity[p][2],triangle.q[p]};
      if(points.count(tri[p]) && points[tri[p]]!=value) throw std::runtime_error("MPI shared key differs");
      points[tri[p]]=value;
      const viskores::Vec3f xyz(position[0],position[1],position[2]);const auto expected=affine(xyz);
      long double distance=0.;
      for(int d=0;d<3;++d) {
        distance+=plane.normal[d]*(static_cast<long double>(position[d])-plane.origin[d]);
        error=std::max(error,std::abs(triangle.velocity[p][d]-expected[d]));
      }
      error=std::max(error,double(std::abs(distance)));error=std::max(error,std::abs(triangle.q[p]-scalar(xyz)));
    }
    auto unordered=tri;std::sort(unordered.begin(),unordered.end());
    if(!triangles.insert(unordered).second) throw std::runtime_error("duplicate plane triangle");
    for(int d=0;d<3;++d) {a[d]=triangle.xyz[1][d]-triangle.xyz[0][d];b[d]=triangle.xyz[2][d]-triangle.xyz[0][d];}
    const long double cross[3]={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    const auto squared=cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2];
    if(squared==0. || cross[0]*plane.normal[0]+cross[1]*plane.normal[1]+cross[2]*plane.normal[2]<=0.)
      throw std::runtime_error("zero-area or reversed plane triangle");
    area+=.5L*std::sqrt(squared);
    for(int p=0;p<3;++p) {
      std::array<Key,2> edge{tri[p],tri[(p+1)%3]};const int sign=edge[1]<edge[0]?-1:1;
      if(sign<0) std::swap(edge[0],edge[1]);auto& e=edges[edge];++e.first;e.second+=sign;
    }
  }
  for(const auto& edge:edges) {
    const auto& pair=edge.first;graph[pair[0]].push_back(pair[1]);graph[pair[1]].push_back(pair[0]);
    if(edge.second.first==1) {
      if(!outer_edge(pair[0],pair[1])) throw std::runtime_error("unexpected interior boundary edge");
      boundary[pair[0]].push_back(pair[1]);boundary[pair[1]].push_back(pair[0]);
    } else if(edge.second.first!=2 || edge.second.second!=0) throw std::runtime_error("nonmanifold/inconsistent plane edge");
  }
  for(const auto& point:boundary) if(point.second.size()!=2) throw std::runtime_error("invalid boundary degree");
  for(const auto* g:{&graph,&boundary}) {
    if(g->empty()) throw std::runtime_error("empty plane connectivity graph");
    std::set<Key> seen;std::vector<Key> pending{g->begin()->first};
    while(!pending.empty()) {const auto p=pending.back();pending.pop_back();if(!seen.insert(p).second) continue;
      for(const auto& other:g->at(p)) if(!seen.count(other)) pending.push_back(other);}
    if(seen.size()!=g->size()) throw std::runtime_error("disconnected plane graph");
  }
  if(std::int64_t(points.size())-std::int64_t(edges.size())+std::int64_t(triangles.size())!=1)
    throw std::runtime_error("plane Euler characteristic differs");
  const long double expected=plane_kind==1?1.L:std::sqrt(1.L+.25L*.25L+.125L*.125L);
  const double area_error=double(std::abs(area-expected));
  if(error>2.e-10 || area_error>2.e-10) throw std::runtime_error("plane field/geometry reference differs");
  std::printf("X4_STRUCTURED ranks=%d axis=%d curve=%d plane=%s triangles=%zu points=%zu errors=0 "
    "max_error=%.17g area_error=%.17g device_only_before_oracle=1 test_geometry_readback=1 runtime_admission=0\n",
    ranks,axis,int(curve),plane_kind==1?"coincident":"oblique",triangles.size(),points.size(),error,area_error);
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);int rank=0,ranks=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  try {
    int devices=0;cuda_require(cudaGetDeviceCount(&devices),"CUDA device count");
    if(ranks>2 || devices<ranks) throw std::runtime_error("structured probe is bounded to NP=1/2 GPUs");
    cuda_require(cudaSetDevice(rank),"CUDA device selection");
    viskores::cont::Initialize(argc,argv,viskores::cont::InitializeOptions::None);
    const int axis=argc>1?std::atoi(argv[1]):0;
    if(axis<0 || axis>2) throw std::invalid_argument("axis outside x/y/z");
    for(bool curve:{false,true}) for(int kind=0;kind<3;++kind) acceptance(rank,ranks,axis,curve,kind);
    rejections(rank);
    if(rank==0) std::puts("PASS structured device scan, key welding, topology and immutable source");
  } catch(const std::exception& error) {
    std::fprintf(stderr,"rank=%d structured probe: %s\n",rank,error.what());MPI_Abort(MPI_COMM_WORLD,1);
  }
  MPI_Finalize();return 0;
}
