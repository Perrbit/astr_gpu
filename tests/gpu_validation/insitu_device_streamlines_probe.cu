#include "insitu_device_streamlines.h"
#include <viskores/cont/Initialize.h>
#include <cstdio>
#include <cstdlib>

__host__ __device__ viskores::Vec3f field(double x,double y,double z,bool constant,int axis) {
  if(constant) {viskores::Vec3f v(0.);v[axis]=1.;return v;}
  return viskores::Vec3f(sin(x)*cos(y)*cos(z),-cos(x)*sin(y)*cos(z),0.);
}
__global__ void generate_halo(double* fields,int nx,int ny,int nz,int ox,int oy,int oz,
    double h,bool constant,int axis) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int nodes=nx*ny*nz;
  if(i>=nodes) return;
  const auto v=field((i%nx+ox-3)*h,(i/nx%ny+oy-3)*h,(i/(nx*ny)+oz-3)*h,constant,axis);
  for(int d=0;d<3;++d) fields[i+d*nodes]=v[d];
}
__global__ void verify_halo(const double* fields,int nx,int ny,int nz,int ox,int oy,int oz,
    double h,bool constant,int axis,int* bad) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int nodes=nx*ny*nz;
  if(i>=nodes) return;
  const auto v=field((i%nx+ox-3)*h,(i/nx%ny+oy-3)*h,(i/(nx*ny)+oz-3)*h,constant,axis);
  for(int d=0;d<3;++d) if(fields[i+d*nodes]!=v[d]) atomicAdd(bad,1);
}
viskores::Vec3f interpolate(const astr_insitu::TraceVertex& point,double h,bool constant,int axis) {
  int base[3];double s[3];
  for(int d=0;d<3;++d) {base[d]=std::min(31,std::max(0,int(floor(point[d]/h))));s[d]=point[d]/h-base[d];}
  viskores::Vec3f result(0.);
  for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
    const double w=(x?s[0]:1.-s[0])*(y?s[1]:1.-s[1])*(z?s[2]:1.-s[2]);
    result+=w*field((base[0]+x)*h,(base[1]+y)*h,(base[2]+z)*h,constant,axis);
  }
  return result;
}
int run(int argc,char** argv,int rank,int ranks) {
  if(ranks!=1 && ranks!=2) throw std::runtime_error("NP=1/2 only");
  const int axis=argc>1?std::atoi(argv[1]):0;
  const bool constant=argc>2 && std::string(argv[2])=="constant";
  const bool forward_only=argc>3 && std::string(argv[3])=="forward";
  if(axis<0 || axis>2) throw std::runtime_error("invalid axis");
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<ranks || cudaSetDevice(rank)!=cudaSuccess)
    throw std::runtime_error("one physical GPU per rank required");
  viskores::cont::Initialize(argc,argv);
  viskores::Id3 extent(32,32,32),offset(0,0,0);
  extent[axis]=32/ranks;offset[axis]=rank*32/ranks;
  viskores::Id3 dims(extent[0]+7,extent[1]+7,extent[2]+7);
  const int nodes=dims[0]*dims[1]*dims[2];
  const double pi=std::acos(-1.),h=2.*pi/32.;
  double* source=nullptr;
  if(cudaMalloc(&source,3*nodes*sizeof(double))!=cudaSuccess) throw std::runtime_error("halo allocation failed");
  generate_halo<<<(nodes+255)/256,256>>>(source,dims[0],dims[1],dims[2],offset[0],offset[1],offset[2],h,constant,axis);
  if(cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("halo generation failed");
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto vectors=astr_insitu::pack_component_halo(source,dims);
  auto geometry=astr_insitu::trace_tgv_device(vectors,vectors,extent,offset,MPI_COMM_WORLD,
    64*1024*1024,constant,axis,forward_only);
  double error=0.;long long vertices=0,segments=geometry.segments.size();
  for(const auto& line:geometry.segments) for(std::size_t i=0;i<line.points.size();++i) {
    ++vertices;
    const auto& p=line.points[i];const auto v=interpolate(p,h,constant,axis);
    for(int d=0;d<3;++d) error=std::max(error,std::abs(v[d]-line.velocity[i][d]));
    if(i && !(p[3]>line.points[i-1][3])) throw std::runtime_error("repeated/reversed accepted length");
    if(constant) {
      viskores::Vec3f original(pi/2.,pi/8.+line.seed*(6.*pi/8.)/15.,pi/4.),seed;
      for(int d=0;d<3;++d) seed[d]=original[(d-axis+3)%3];
      for(int d=0;d<3;++d) error=std::max(error,std::abs(p[d]-seed[d]-(d==axis?line.direction*p[3]:0.)));
    }
  }
  for(int i=0;i<geometry.particles;++i) {
    const auto& s=geometry.final_state[i];
    for(int d=0;d<8;++d) if(!std::isfinite(s[d])) throw std::runtime_error("nonfinite particle state");
    if(constant && i<16) {
      if(s[7]!=astr_insitu::TraceLength) throw std::runtime_error("constant forward trace unfinished");
      error=std::max(error,std::abs(s[axis]-3*pi/2.));error=std::max(error,std::abs(s[4]-pi));
    }
  }
  astr_insitu::require_device_only(vectors);
  int* bad=nullptr,local_bad=-1,total_bad=-1;
  if(cudaMalloc(&bad,sizeof(int))!=cudaSuccess || cudaMemset(bad,0,sizeof(int))!=cudaSuccess)
    throw std::runtime_error("source check allocation failed");
  verify_halo<<<(nodes+255)/256,256>>>(source,dims[0],dims[1],dims[2],offset[0],offset[1],offset[2],h,constant,axis,bad);
  if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(&local_bad,bad,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("source check failed");
  cudaFree(bad);cudaFree(source);
  double worst=0.;long long local[3]={vertices,segments,geometry.transfers},totals[3];
  MPI_Allreduce(&error,&worst,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(local,totals,3,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&local_bad,&total_bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(!rank) std::printf("IS8 CUDA compact streamlines NP=%d axis=%d constant=%d max_error=%.17g vertices=%lld segments=%lld transfers=%lld rounds=%d input_host_mirror=0 source_errors=%d particles=%d\n",
    ranks,axis,constant,worst,totals[0],totals[1],totals[2],geometry.rounds,total_bad,geometry.particles);
  return std::isfinite(worst) && worst<=2e-10 && totals[0]>0 && total_bad==0?0:2;
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);int rank=0,ranks=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  int status=0;
  try {status=run(argc,argv,rank,ranks);} catch(const std::exception& e) {
    std::fprintf(stderr,"rank %d device streamline failure: %s\n",rank,e.what());MPI_Abort(MPI_COMM_WORLD,3);
  }
  MPI_Finalize();return status;
}
