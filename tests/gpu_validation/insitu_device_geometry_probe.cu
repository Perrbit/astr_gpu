#include "insitu_device_geometry.h"
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/Initialize.h>
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCell.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLPolyDataReader.h>
#include <nvtx3/nvToolsExt.h>

using Diagnostics=astr_insitu::DeviceDiagnostics;

__host__ __device__ viskores::Vec3f value(double x,double y,double z) {
  return viskores::Vec3f(sin(x)*cos(y)*cos(z),-cos(x)*sin(y)*cos(z),0.);
}
__host__ __device__ double q_value(double x,double y,double z) {
  return (sin(x)*sin(x)*sin(y)*sin(y)-cos(x)*cos(x)*cos(y)*cos(y))*cos(z)*cos(z);
}
__global__ void generate(viskores::Vec3f* velocity,Diagnostics* diagnostics,int nx,int ny,int nz,
                         int ox,int oy,int oz,double h) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=nx*ny*nz) return;
  double x=(i%nx+ox)*h,y=(i/nx%ny+oy)*h,z=(i/(nx*ny)+oz)*h;
  velocity[i]=value(x,y,z); diagnostics[i]=Diagnostics(0.);
  diagnostics[i][9]=q_value(x,y,z);
}
__global__ void verify_source(const viskores::Vec3f* velocity,const Diagnostics* diagnostics,
    int nx,int ny,int nz,int ox,int oy,int oz,double h,int* bad) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=nx*ny*nz) return;
  double x=(i%nx+ox)*h,y=(i/nx%ny+oy)*h,z=(i/(nx*ny)+oz)*h;
  const auto v=value(x,y,z);
  for(int d=0;d<3;++d) if(velocity[i][d]!=v[d]) atomicAdd(bad,1);
  for(int d=0;d<14;++d) if(diagnostics[i][d]!=(d==9?q_value(x,y,z):0.)) atomicAdd(bad,1);
}
struct Allocations {
  viskores::Vec3f* velocity=nullptr;
  Diagnostics* diagnostics=nullptr;
  explicit Allocations(int n) {
    if(cudaMalloc(&velocity,n*sizeof(*velocity))!=cudaSuccess ||
       cudaMalloc(&diagnostics,n*sizeof(*diagnostics))!=cudaSuccess)
      throw std::runtime_error("geometry probe allocation failed");
  }
  ~Allocations() {if(velocity) cudaFree(velocity); if(diagnostics) cudaFree(diagnostics);}
};

// This independent probe oracle interpolates only final product coordinates.
viskores::Vec3f interpolated(const viskores::Vec3f& p,double h,bool q,double& scalar) {
  int base[3]; double fraction[3];
  for(int d=0;d<3;++d) {
    base[d]=std::max(0,std::min(31,int(std::floor(p[d]/h))));
    fraction[d]=p[d]/h-base[d];
  }
  viskores::Vec3f v(0.); scalar=0.;
  for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
    double w=(x?fraction[0]:1.-fraction[0])*(y?fraction[1]:1.-fraction[1])*
      (z?fraction[2]:1.-fraction[2]);
    double px=(base[0]+x)*h,py=(base[1]+y)*h,pz=(base[2]+z)*h;
    v+=w*value(px,py,pz);
    if(q) scalar+=w*q_value(px,py,pz);
  }
  return v;
}

struct Measures { double error=0.,area=0.; long long cells=0,points=0; };
void roundtrip(const viskores::cont::DataSet& data,bool surface,const std::string& path) {
  vtkNew<vtkPolyData> mesh;
  vtkNew<vtkPoints> points;points->SetDataTypeToDouble();
  vtkNew<vtkDoubleArray> velocity,u;
  velocity->SetName("velocity");velocity->SetNumberOfComponents(3);
  u->SetName("u");
  vtkNew<vtkCellArray> polygons;
  if(data.GetNumberOfCoordinateSystems() && data.GetCoordinateSystem().GetNumberOfPoints()) {
    viskores::cont::ArrayHandle<viskores::Vec3f> coords;
    viskores::cont::ArrayCopy(data.GetCoordinateSystem().GetData(),coords);
    const auto p=coords.ReadPortal();
    const auto v=data.GetPointField("velocity").GetData().AsArrayHandle<decltype(coords)>().ReadPortal();
    const auto color=data.GetPointField("u").GetData().AsArrayHandle<viskores::cont::ArrayHandle<double>>().ReadPortal();
    for(viskores::Id i=0;i<coords.GetNumberOfValues();++i) {
      const auto xyz=p.Get(i),vector=v.Get(i);
      const double x[3]={xyz[0],xyz[1],xyz[2]},y[3]={vector[0],vector[1],vector[2]};
      points->InsertNextPoint(x);velocity->InsertNextTuple(y);u->InsertNextValue(color.Get(i));
    }
    if(surface) {
      const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
      const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{}).ReadPortal();
      for(viskores::Id i=0;i<cells.GetNumberOfCells();++i) {
        const vtkIdType ids[3]={indices.Get(3*i),indices.Get(3*i+1),indices.Get(3*i+2)};
        polygons->InsertNextCell(3,ids);
      }
      vtkNew<vtkDoubleArray> q;q->SetName("Q_rs");
      auto values=data.GetPointField("Q_rs").GetData().AsArrayHandle<viskores::cont::ArrayHandle<double>>().ReadPortal();
      for(viskores::Id i=0;i<coords.GetNumberOfValues();++i) q->InsertNextValue(values.Get(i));
      mesh->GetPointData()->AddArray(q);
    } else {
      const auto dimensions=data.GetCellSet().AsCellSet<viskores::cont::CellSetStructured<2>>().GetPointDimensions();
      for(viskores::Id j=0;j<dimensions[1]-1;++j) for(viskores::Id i=0;i<dimensions[0]-1;++i) {
        const vtkIdType a=j*dimensions[0]+i,ids[4]={a,a+1,a+dimensions[0]+1,a+dimensions[0]};
        polygons->InsertNextCell(4,ids);
      }
    }
  }
  mesh->SetPoints(points);mesh->SetPolys(polygons);
  mesh->GetPointData()->AddArray(velocity);mesh->GetPointData()->AddArray(u);
  vtkNew<vtkXMLPolyDataWriter> writer;writer->SetFileName(path.c_str());writer->SetInputData(mesh);
  if(!writer->Write() || writer->GetErrorCode()) throw std::runtime_error("Final geometry writer failed");
  vtkNew<vtkXMLPolyDataReader> reader;reader->SetFileName(path.c_str());reader->Update();
  const auto restored=reader->GetOutput();
  if(reader->GetErrorCode() || restored->GetNumberOfPoints()!=mesh->GetNumberOfPoints() ||
     restored->GetNumberOfCells()!=mesh->GetNumberOfCells()) throw std::runtime_error("VTK geometry sizes differ");
  for(vtkIdType i=0;i<mesh->GetNumberOfPoints();++i) {
    double a[3],b[3];mesh->GetPoint(i,a);restored->GetPoint(i,b);
    for(int d=0;d<3;++d) if(a[d]!=b[d]) throw std::runtime_error("VTK coordinate roundtrip differs");
    for(const char* name:{"u","velocity","Q_rs"}) {
      auto before=mesh->GetPointData()->GetArray(name);
      if(!before) continue;
      auto after=restored->GetPointData()->GetArray(name);
      if(!after || after->GetNumberOfComponents()!=before->GetNumberOfComponents()) throw std::runtime_error("VTK field missing");
      for(int d=0;d<before->GetNumberOfComponents();++d)
        if(before->GetComponent(i,d)!=after->GetComponent(i,d)) throw std::runtime_error("VTK field roundtrip differs");
    }
  }
  for(vtkIdType i=0;i<mesh->GetNumberOfCells();++i) {
    auto a=mesh->GetCell(i),b=restored->GetCell(i);
    if(a->GetNumberOfPoints()!=b->GetNumberOfPoints()) throw std::runtime_error("VTK cell arity differs");
    for(vtkIdType j=0;j<a->GetNumberOfPoints();++j)
      if(a->GetPointId(j)!=b->GetPointId(j)) throw std::runtime_error("VTK cell connectivity differs");
  }
}

Measures inspect(const viskores::cont::DataSet& data,bool surface,double iso) {
  Measures m;
  if(!data.GetNumberOfCoordinateSystems()) return m;
  m.points=data.GetCoordinateSystem().GetNumberOfPoints();
  m.cells=data.GetCellSet().GetNumberOfCells();
  if(!m.points) {if(m.cells) throw std::runtime_error("empty geometry has cells"); return m;}
  viskores::cont::ArrayHandle<viskores::Vec3f> coords;
  if(surface) coords=data.GetCoordinateSystem().GetData().AsArrayHandle<decltype(coords)>();
  else viskores::cont::ArrayCopy(data.GetCoordinateSystem().GetData(),coords);
  auto vectors=data.GetPointField("velocity").GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f>>();
  auto u=data.GetPointField("u").GetData().AsArrayHandle<viskores::cont::ArrayHandle<double>>();
  if(surface) {astr_insitu::require_device_only(coords);}
  astr_insitu::require_device_only(vectors); astr_insitu::require_device_only(u);
  const auto p=coords.ReadPortal(); const auto v=vectors.ReadPortal(); const auto color=u.ReadPortal();
  const double h=2.*std::acos(-1.)/32.;
  for(viskores::Id i=0;i<m.points;++i) {
    double q=0.; const auto expected=interpolated(p.Get(i),h,surface,q);
    for(int d=0;d<3;++d) {
      if(!std::isfinite(p.Get(i)[d]) || !std::isfinite(v.Get(i)[d])) throw std::runtime_error("nonfinite product");
      m.error=std::max(m.error,std::abs(v.Get(i)[d]-expected[d]));
    }
    m.error=std::max(m.error,std::abs(color.Get(i)-v.Get(i)[0]));
    m.error=std::max(m.error,surface?std::abs(q-iso):std::abs(p.Get(i)[2]-4*h));
  }
  if(surface) {
    auto q=data.GetPointField("Q_rs").GetData().AsArrayHandle<viskores::cont::ArrayHandle<double>>().ReadPortal();
    for(viskores::Id i=0;i<m.points;++i) m.error=std::max(m.error,std::abs(q.Get(i)-iso));
    const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
    auto connectivity=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{}).ReadPortal();
    for(viskores::Id c=0;c<m.cells;++c) {
      if(cells.GetNumberOfPointsInCell(c)!=3) throw std::runtime_error("nontriangular contour");
      const auto a=p.Get(connectivity.Get(3*c+1))-p.Get(connectivity.Get(3*c));
      const auto b=p.Get(connectivity.Get(3*c+2))-p.Get(connectivity.Get(3*c));
      const viskores::Vec3f cross(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]);
      m.area+=.5*std::sqrt(viskores::Dot(cross,cross));
    }
  } else m.area=m.cells*h*h;
  return m;
}

int run(int argc,char** argv,int rank,int ranks) {
  if(ranks!=1 && ranks!=2) throw std::runtime_error("NP=1/2 only");
  int axis=argc>1?std::atoi(argv[1]):0;
  bool empty=argc>2 && std::string(argv[2])=="empty";
  if(axis<0 || axis>2) throw std::runtime_error("invalid axis");
  int devices=0;
  if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices<ranks || cudaSetDevice(rank)!=cudaSuccess)
    throw std::runtime_error("one GPU per rank required");
  viskores::cont::Initialize(argc,argv);
  viskores::Id3 dims(33,33,33),offset(0,0,0);
  dims[axis]=32/ranks+1; offset[axis]=rank*32/ranks;
  int nodes=dims[0]*dims[1]*dims[2];
  Allocations a(nodes);
  double h=2.*std::acos(-1.)/32.,iso=empty?10.:.25;
  generate<<<(nodes+255)/256,256>>>(a.velocity,a.diagnostics,dims[0],dims[1],dims[2],offset[0],offset[1],offset[2],h);
  if(cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("field generation failed");
  astr_insitu::DeviceGeometryAudit audit;
  nvtxRangePushA("ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION");
  auto products=astr_insitu::extract_tgv_geometry(a.velocity,a.diagnostics,dims,offset,true,true,iso,&audit);
  nvtxRangePop();
  std::printf("IS8_GEOMETRY_INPUT rank=%d q=0x%llx u=0x%llx scalar_bytes=%llu velocity=0x%llx velocity_bytes=%llu diagnostics=0x%llx diagnostics_bytes=%llu\n",
    rank,static_cast<unsigned long long>(audit.q),static_cast<unsigned long long>(audit.u),
    static_cast<unsigned long long>(audit.scalar_bytes),reinterpret_cast<unsigned long long>(a.velocity),
    static_cast<unsigned long long>(nodes)*sizeof(*a.velocity),reinterpret_cast<unsigned long long>(a.diagnostics),
    static_cast<unsigned long long>(nodes)*sizeof(*a.diagnostics));
  nvtxRangePushA("ASTR_IS8_COMPACT_GEOMETRY_READ");
  auto slice=inspect(products.slice,false,iso),surface=inspect(products.surface,true,iso);
  if(argc>3) {
    const std::string prefix=argv[3];
    roundtrip(products.slice,false,prefix+".rank"+std::to_string(rank)+".slice.vtp");
    roundtrip(products.surface,true,prefix+".rank"+std::to_string(rank)+".surface.vtp");
    std::printf("IS8_GEOMETRY_ROUNDTRIP rank=%d exact=1\n",rank);
  }
  nvtxRangePop();
  double local[3]={std::max(slice.error,surface.error),slice.area,surface.area},global[3];
  MPI_Allreduce(local,global,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(local+1,global+1,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  long long counts[4]={slice.cells,surface.cells,slice.points,surface.points},totals[4];
  MPI_Allreduce(counts,totals,4,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
  int *bad=nullptr,source_errors=-1;
  if(cudaMalloc(&bad,sizeof(int))!=cudaSuccess || cudaMemset(bad,0,sizeof(int))!=cudaSuccess)
    throw std::runtime_error("source check allocation failed");
  verify_source<<<(nodes+255)/256,256>>>(a.velocity,a.diagnostics,dims[0],dims[1],dims[2],offset[0],offset[1],offset[2],h,bad);
  if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(&source_errors,bad,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("source check failed");
  cudaFree(bad);
  int source_total=0; MPI_Allreduce(&source_errors,&source_total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(!rank) std::printf("IS8 CUDA geometry axis=%d NP=%d empty=%d max_error=%.17g slice_area=%.17g surface_area=%.17g slice_cells=%lld surface_cells=%lld slice_points=%lld surface_points=%lld input_host_mirror=0\n",
    axis,ranks,empty,global[0],global[1],global[2],totals[0],totals[1],totals[2],totals[3]);
  if(!rank) std::printf("source_errors=%d\n",source_total);
  return source_total==0 && std::isfinite(global[0]) && global[0]<=2e-10 && totals[0]==1024 &&
    std::abs(global[1]-4*std::acos(-1.)*std::acos(-1.))<=2e-10 &&
    (empty?totals[1]==0:totals[1]>0) ? 0:2;
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv); int rank=0,ranks=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  int status=0;
  try {status=run(argc,argv,rank,ranks);} catch(const std::exception& e) {
    std::fprintf(stderr,"rank %d geometry failure: %s\n",rank,e.what()); MPI_Abort(MPI_COMM_WORLD,3);
  }
  MPI_Finalize(); return status;
}
