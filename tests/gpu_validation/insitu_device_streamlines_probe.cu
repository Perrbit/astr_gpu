#include "insitu_device_streamlines.h"
#include <viskores/cont/Initialize.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <cstdio>
#include <cstdlib>

__host__ __device__ viskores::Vec3f mapped_position(double x,double y,double z,int mapping) {
  if(mapping==1) return viskores::Vec3f(x+.15*sin(x)*sin(y),
    y+.15*sin(y)*sin(z),z+.15*sin(z)*sin(x));
  return viskores::Vec3f(x,y+.15*sin(x)*sin(z),z);
}

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
__global__ void generate_physical_halo(viskores::Vec3f* coordinates,double* fields,
    int nx,int ny,int nz,int ox,int oy,int oz,double h,bool constant,int axis,int mapping) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int nodes=nx*ny*nz;
  if(i>=nodes) return;
  const auto p=mapped_position((i%nx+ox-3)*h,(i/nx%ny+oy-3)*h,(i/(nx*ny)+oz-3)*h,mapping);
  coordinates[i]=p;
  const auto v=field(p[0],p[1],p[2],constant,axis);
  for(int d=0;d<3;++d) fields[i+d*nodes]=v[d];
}
__global__ void verify_physical_halo(const viskores::Vec3f* coordinates,const double* fields,
    int nx,int ny,int nz,int ox,int oy,int oz,double h,bool constant,int axis,int mapping,int* bad) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int nodes=nx*ny*nz;
  if(i>=nodes) return;
  const auto p=mapped_position((i%nx+ox-3)*h,(i/nx%ny+oy-3)*h,(i/(nx*ny)+oz-3)*h,mapping);
  const auto v=field(p[0],p[1],p[2],constant,axis);
  for(int d=0;d<3;++d) if(coordinates[i][d]!=p[d] || fields[i+d*nodes]!=v[d]) atomicAdd(bad,1);
}

VISKORES_EXEC_CONT viskores::Vec3f affine_value(const viskores::Vec3f& p) {
  return viskores::Vec3f(2.*p[0]-3.*p[1]+4.*p[2]+1.,p[0]+p[2],p[1]-2.*p[2]);
}
struct AffineVelocity : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(const viskores::Vec3f& p,viskores::Vec3f& v) const {v=affine_value(p);}
};
struct CheckPhysicalCellInterpolation : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,ExecObject,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  viskores::Id3 dimensions;
  template<class Coordinates,class Grid>
  VISKORES_EXEC void operator()(viskores::Id id,const Coordinates& coords,const Grid& grid,double& error) const {
    const auto cells=dimensions-viskores::Id3(1);
    const viskores::Id3 cell(id%cells[0],id/cells[0]%cells[1],id/(cells[0]*cells[1]));
    viskores::Vec3f point(0.);
    const viskores::Vec3f parametric(.23,.37,.61);
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
      const auto index=cell+viskores::Id3(x,y,z);
      const auto weight=(x?parametric[0]:1.-parametric[0])*(y?parametric[1]:1.-parametric[1])*
        (z?parametric[2]:1.-parametric[2]);
      point+=weight*coords.Get(index[0]+dimensions[0]*(index[1]+dimensions[1]*index[2]));
    }
    viskores::VecVariable<viskores::Vec3f,2> values;
    const auto status=grid.Evaluate(point,0.,values);
    error=status.CheckFail()?INFINITY:viskores::Magnitude(values[0]-affine_value(point));
    const auto cached=grid.WithCache();
    cached.cell=id;
    const auto cached_status=cached.Evaluate(point,0.,values);
    error=viskores::Max(error,cached_status.CheckFail()?INFINITY:
      viskores::Magnitude(values[0]-affine_value(point)));
    // Exercise face/edge/corner ties, a cell jump, and an outside-domain cache miss.
    if(id<32) {
      viskores::Id previous=id;
      viskores::Vec3f previous_parametric(.5);
      for(int query=0;query<8;++query) {
        viskores::Vec3f target(.31,.43,.57);
        if(query<3) target[query]=0.;
        if(query==3) target=viskores::Vec3f(0.);
        if(query==4) target=viskores::Vec3f(1.);
        const auto selected=query==6?(id+cells[0]*cells[1])/2:id;
        const auto ids=grid.owner.PointIds(selected);
        double vertices[8][3];
        for(int i=0;i<8;++i) for(int d=0;d<3;++d) vertices[i][d]=coords.Get(ids[i])[d];
        viskores::Vec3f world;
        lcl::parametricToWorld(lcl::Hexahedron{},lcl::makeFieldAccessorNestedSOA(vertices,3),target,world);
        if(query==7) world=viskores::Vec3f(-100.);
        viskores::Id expected=-1;
        viskores::Vec3f expected_parametric;
        const auto reference=grid.owner.Find(world,expected,expected_parametric);
        const auto actual=grid.owner.FindCached(world,previous,previous_parametric);
        if(actual!=reference || (actual>=0 && expected!=previous)) error=INFINITY;
        if(actual>=0) error=viskores::Max(error,viskores::Magnitude(previous_parametric-expected_parametric));
      }
    }
  }
};

double check_physical_components(const astr_insitu::PhysicalTraceData& data,
    const viskores::Id3& extent,const viskores::Id3& offset,int axis,int mapping,int rank,int ranks) {
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::ArrayHandle<viskores::Vec3f> affine;
  invoke(AffineVelocity{},data.coordinates,affine);
  astr_insitu::synchronize_device_stage("AffineVelocityReference");
  using Field=viskores::worklet::flow::VelocityField<viskores::cont::ArrayHandle<viskores::Vec3f>>;
  astr_insitu::PhysicalTraceOwner owner(data,extent,offset,32);
  astr_insitu::PhysicalTraceGrid grid(owner,Field(affine));
  CheckPhysicalCellInterpolation check;check.dimensions=data.dimensions;
  const auto cells=data.dimensions-viskores::Id3(1);
  viskores::cont::ArrayHandle<double> errors;
  invoke(check,viskores::cont::ArrayHandleIndex(cells[0]*cells[1]*cells[2]),data.coordinates,grid,errors);
  astr_insitu::synchronize_device_stage("CheckPhysicalCellInterpolation");
  const auto error=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},errors,0.,viskores::Maximum{});
  astr_insitu::require_device_only(affine);astr_insitu::require_device_only(data.coordinates);
  if(mapping==2 && ranks==2 && axis==1) {
    const double pi=std::acos(-1.);
    // Both points lie above y=pi, but straddle the actual displaced rank face.
    std::array<astr_insitu::RK45TraceWorklet::State,2> queries;
    for(int i=0;i<2;++i) queries[i]=astr_insitu::RK45TraceWorklet::State(
      pi/4.,pi+(i?.1125:.0375),pi/4.,0.,0.,0.,0.,astr_insitu::TraceActive);
    viskores::cont::ArrayHandle<viskores::Id> located;
    invoke(astr_insitu::LocatePhysicalTraceOwner{},
      viskores::cont::make_ArrayHandle(queries.data(),2,viskores::CopyFlag::On),owner,located);
    astr_insitu::synchronize_device_stage("IndependentCurvedFaceOwnership");
    const auto result=located.ReadPortal();
    for(int i=0;i<2;++i) if((result.Get(i)>=0)!=(rank==i))
      throw std::runtime_error("Curved y-face owner incorrectly uses Cartesian bounds");
  }
  return error;
}
__host__ __device__ viskores::Vec3f interpolate(const astr_insitu::TraceVertex& point,double h,bool constant,int axis) {
  int base[3];double s[3];
  for(int d=0;d<3;++d) {base[d]=std::min(31,std::max(0,int(floor(point[d]/h))));s[d]=point[d]/h-base[d];}
  viskores::Vec3f result(0.);
  for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
    const double w=(x?s[0]:1.-s[0])*(y?s[1]:1.-s[1])*(z?s[2]:1.-s[2]);
    result+=w*field((base[0]+x)*h,(base[1]+y)*h,(base[2]+z)*h,constant,axis);
  }
  return result;
}
__global__ void inspect_resident(const astr_insitu::TraceVertex* points,const viskores::Vec3f* velocity,
    const viskores::Id* particle,viskores::Id count,double h,bool constant,int axis,unsigned long long* error) {
  const auto i=static_cast<viskores::Id>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=count) return;
  const auto p=points[i];const auto v=interpolate(p,h,constant,axis);
  double worst=0.;
  for(int d=0;d<3;++d) worst=fmax(worst,fabs(v[d]-velocity[i][d]));
  if(constant) {
    const double pi=acos(-1.);
    const viskores::Vec3f original(pi/2.,pi/8.+(particle[i]%16)*(6.*pi/8.)/15.,pi/4.);
    for(int d=0;d<3;++d) worst=fmax(worst,fabs(p[d]-original[(d-axis+3)%3]-
      (d==axis?(particle[i]<16?1.:-1.)*p[3]:0.)));
  }
  if(!isfinite(worst)) worst=INFINITY;
  atomicMax(error,static_cast<unsigned long long>(__double_as_longlong(worst)));
}
int run(int argc,char** argv,int rank,int ranks) {
  if(ranks!=1 && ranks!=2) throw std::runtime_error("NP=1/2 only");
  const int axis=argc>1?std::atoi(argv[1]):0;
  const bool constant=argc>2 && std::string(argv[2])=="constant";
  const bool forward_only=argc>3 && std::string(argv[3])=="forward";
  const bool resident=argc>4 && std::string(argv[4])=="resident";
  const std::string mapping_name=argc>5?argv[5]:"";
  const int mapping=mapping_name=="periodic"?1:(mapping_name=="y-wavy"?2:0);
  if(!mapping_name.empty() && !mapping) throw std::runtime_error("invalid physical mapping");
  if(mapping && !constant) throw std::runtime_error("Physical probe currently admits only the independent constant-flow oracle");
  if(axis<0 || axis>2) throw std::runtime_error("invalid axis");
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<ranks || cudaSetDevice(rank)!=cudaSuccess)
    throw std::runtime_error("one physical GPU per rank required");
  viskores::cont::Initialize(argc,argv);
  if(resident && viskores::cont::cuda::internal::CudaAllocator::UsingManagedMemory())
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
  viskores::Id3 extent(32,32,32),offset(0,0,0);
  extent[axis]=32/ranks;offset[axis]=rank*32/ranks;
  viskores::Id3 dims(extent[0]+7,extent[1]+7,extent[2]+7);
  const int nodes=dims[0]*dims[1]*dims[2];
  const double pi=std::acos(-1.),h=2.*pi/32.;
  double* source=nullptr;
  viskores::Vec3f* coordinates=nullptr;
  if(cudaMalloc(&source,3*nodes*sizeof(double))!=cudaSuccess) throw std::runtime_error("halo allocation failed");
  if(mapping) {
    if(!resident || cudaMalloc(&coordinates,nodes*sizeof(*coordinates))!=cudaSuccess)
      throw std::runtime_error("physical coordinate allocation failed or nonresident mode");
    generate_physical_halo<<<(nodes+255)/256,256>>>(coordinates,source,dims[0],dims[1],dims[2],
      offset[0],offset[1],offset[2],h,constant,axis,mapping);
  } else generate_halo<<<(nodes+255)/256,256>>>(source,dims[0],dims[1],dims[2],offset[0],offset[1],offset[2],h,constant,axis);
  if(cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("halo generation failed");
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto vectors=astr_insitu::pack_component_halo(source,dims);
  auto physical= mapping?astr_insitu::borrow_cuda_array(coordinates,nodes):viskores::cont::ArrayHandle<viskores::Vec3f>{};
  double interpolation_error=0.;
  if(mapping) interpolation_error=check_physical_components(
    astr_insitu::crop_physical_trace_halo(physical,vectors,vectors,extent,offset,32),extent,offset,axis,mapping,rank,ranks);
  if(!std::isfinite(interpolation_error) || interpolation_error>2e-10) {
    std::fprintf(stderr,"rank %d physical noncentral affine interpolation error=%.17g\n",rank,interpolation_error);
    throw std::runtime_error("Physical interpolation gate failed before trajectory integration");
  }
  auto geometry=astr_insitu::trace_tgv_device(vectors,vectors,extent,offset,MPI_COMM_WORLD,
    64*1024*1024,constant,axis,forward_only,false,32,resident,mapping?&physical:nullptr);
  double error=interpolation_error;long long vertices=0,segments=geometry.segments.size();
  if(resident && geometry.resident_geometry.GetNumberOfCoordinateSystems()) {
    auto accepted=geometry.resident_geometry.GetPointField("accepted").GetData().
      AsArrayHandle<viskores::cont::ArrayHandle<astr_insitu::TraceVertex>>();
    auto values=geometry.resident_geometry.GetPointField("velocity").GetData().
      AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f>>();
    auto particles=geometry.resident_geometry.GetPointField("particle").GetData().
      AsArrayHandle<viskores::cont::ArrayHandle<viskores::Id>>();
    vertices=accepted.GetNumberOfValues();segments=geometry.resident_geometry.GetCellSet().GetNumberOfCells();
    viskores::cont::Token token;
    const auto pointer=[&](const auto& array) {
      astr_insitu::require_device_only(array);
      return array.GetBuffers()[0].ReadPointerDevice(viskores::cont::DeviceAdapterTagCuda{},token);
    };
    unsigned long long* device_error=nullptr;
    if(cudaMalloc(&device_error,sizeof(double))!=cudaSuccess || cudaMemset(device_error,0,sizeof(double))!=cudaSuccess)
      throw std::runtime_error("Resident check allocation failed");
    inspect_resident<<<(vertices+255)/256,256>>>(static_cast<const astr_insitu::TraceVertex*>(pointer(accepted)),
      static_cast<const viskores::Vec3f*>(pointer(values)),static_cast<const viskores::Id*>(pointer(particles)),
      vertices,h,constant,axis,device_error);
    double trajectory_error=0.;
    if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(&trajectory_error,device_error,sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Resident check failed");
    error=std::max(error,trajectory_error);
    cudaFree(device_error);
  }
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
  if(mapping) verify_physical_halo<<<(nodes+255)/256,256>>>(coordinates,source,dims[0],dims[1],dims[2],
    offset[0],offset[1],offset[2],h,constant,axis,mapping,bad);
  else verify_halo<<<(nodes+255)/256,256>>>(source,dims[0],dims[1],dims[2],offset[0],offset[1],offset[2],h,constant,axis,bad);
  if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(&local_bad,bad,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("source check failed");
  cudaFree(bad);cudaFree(source);cudaFree(coordinates);
  double worst=0.;long long local[3]={vertices,segments,geometry.transfers},totals[3];
  MPI_Allreduce(&error,&worst,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(local,totals,3,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(&local_bad,&total_bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(!rank) std::printf("IS8 CUDA compact streamlines NP=%d axis=%d constant=%d max_error=%.17g vertices=%lld segments=%lld transfers=%lld rounds=%d input_host_mirror=0 source_errors=%d particles=%d resident=%d control_read_bytes=%llu owner_query_read_bytes=%llu curve_grid=%d mapping=%s\n",
    ranks,axis,constant,worst,totals[0],totals[1],totals[2],geometry.rounds,total_bad,geometry.particles,resident,
    static_cast<unsigned long long>(geometry.control_read_bytes),
    static_cast<unsigned long long>(geometry.owner_query_read_bytes),int(mapping!=0),mapping_name.c_str());
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
