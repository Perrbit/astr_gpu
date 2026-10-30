#include "insitu_compact_mesh.h"
#include "insitu_device_geometry.h"
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using Real=astr_insitu::Real;
using Vector=astr_insitu::NativeVector;
extern "C" void astr_insitu_runtime_physical_group(int);

static void cuda_ok(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static Real physical_component(int rank,bool q_surface=false) {
  const int global[]={8,6,4},extent[]={5,7,5},offset[]={4*rank,0,0};
  const int ids[]={2,3,4,5,22},n=extent[0]*extent[1]*extent[2],nf=q_surface?5:4;
  const Real references[]={Real(2.),Real(100.),Real(300.)};
  std::vector<Vector> positions(n);
  std::vector<Real> fields(nf*n);
  for(int k=0;k<extent[2];++k) for(int j=0;j<extent[1];++j) for(int i=0;i<extent[0];++i) {
    const int node=i+extent[0]*(j+extent[1]*k);
    const Real x=Real(i+offset[0])/Real(global[0]),y=Real(j)/Real(global[1]),z=Real(k)/Real(global[2]);
    const Real curved_y=y+Real(.04)*std::sin(std::acos(-Real(1.))*x)*std::sin(std::acos(-Real(1.))*y);
    positions[node]=Vector(x,curved_y,z);
    fields[node]=Real(1.)+x;fields[n+node]=Real(2.)-curved_y;fields[2*n+node]=Real(.5)+Real(.2)*z;
    fields[3*n+node]=Real(100.)+Real(10.)*x+curved_y+z;
    if(q_surface) fields[4*n+node]=x;
  }
  Vector* xyz=nullptr;Real* values=nullptr;
  cuda_ok(cudaMalloc(&xyz,n*sizeof(Vector)));cuda_ok(cudaMalloc(&values,nf*n*sizeof(Real)));
  cuda_ok(cudaMemcpy(xyz,positions.data(),n*sizeof(Vector),cudaMemcpyHostToDevice));
  cuda_ok(cudaMemcpy(values,fields.data(),nf*n*sizeof(Real),cudaMemcpyHostToDevice));
  astr_insitu_runtime_physical_group(1);
  Real error=Real(0.);
  for(double iso:q_surface?std::vector<double>{.45,.65}:std::vector<double>{0.}) {
    std::shared_ptr<void> first;
    bool area_checked=false;
    for(const std::string quantity:{"velocity_x","velocity_x","p_pinf","velocity_magnitude"}) {
      astr_insitu::RuntimeProductView context{};context.id=quantity;context.labels[4]=quantity;
      context.labels[0]=q_surface?"q_surface":"slice";
      context.labels[6]="diverging_blue_red";context.reals[22]=0.;context.reals[23]=4.;
      const double selection[]={.5,.4,.3,.2,-1.,.3,.2,.8,.1,.9,0.,1.};
      std::copy_n(selection,12,context.reals.begin());
      context.reals[25]=iso;
      auto mesh=astr_insitu::build_runtime_physical_mesh(context,1,Real(.001),ids,nf,global,extent,offset,
        reinterpret_cast<Real*>(xyz),values,references,2LL*1024*1024*1024,n*7*sizeof(Real),quantity=="p_pinf"?5:2);
      if(!q_surface && (!mesh.draw.points || !mesh.draw.cells)) throw std::runtime_error("Physical affine plane is empty");
      if(quantity=="velocity_x") {
        if(first && first!=mesh.owner) throw std::runtime_error("Physical display change repeated extraction");
        first=mesh.owner;
      }
      auto owner=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(mesh.owner);
      const auto& view=owner->get();
      std::vector<Vector> points(view.points);
      std::vector<Real> scalar(view.points);
      std::vector<viskores::Id> indices(3*view.cells);
      if(view.points) {
        cuda_ok(cudaMemcpy(points.data(),view.coordinates,points.size()*sizeof(Vector),cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(scalar.data(),view.scalar,scalar.size()*sizeof(Real),cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(indices.data(),view.connectivity,indices.size()*sizeof(viskores::Id),cudaMemcpyDeviceToHost));
      }
      for(std::size_t i=0;i<points.size();++i) {
        const auto& p=points[i];
        const Real u=Real(1.)+p[0],v=Real(2.)-p[1],w=Real(.5)+Real(.2)*p[2];
        const Real expected=quantity=="p_pinf"?(Real(100.)+Real(10.)*p[0]+p[1]+p[2])/references[1]:
          (quantity=="velocity_magnitude"?std::sqrt(u*u+v*v+w*w):u);
        error=std::max(error,std::abs(scalar[i]-expected));
        error=std::max(error,q_surface?std::abs(p[0]-Real(iso)):
          std::abs(Real(.2)*(p[0]-Real(.5))-(p[1]-Real(.4))+Real(.3)*(p[2]-Real(.3))));
        if(!std::isfinite(scalar[i])) throw std::runtime_error("Nonfinite physical scalar");
        for(int d=0;d<3;++d) {
          error=std::max(error,std::max(Real(0.),Real(selection[6+2*d])-p[d]));
          error=std::max(error,std::max(Real(0.),p[d]-Real(selection[7+2*d])));
        }
      }
      Real area=0;
      for(std::size_t i=0;i<indices.size();i+=3) {
        for(int d=0;d<3;++d) if(indices[i+d]<0 || indices[i+d]>=view.points)
          throw std::runtime_error("Physical plane connectivity exceeds its points");
        const auto a=points[indices[i]],b=points[indices[i+1]],c=points[indices[i+2]];
        const auto cross=viskores::Cross(b-a,c-a);
        const Real squared=viskores::Dot(cross,cross);
        if(!(squared>Real(0.)) || (!q_surface &&
            !(viskores::Dot(cross,Vector(-Real(.2),Real(1.),-Real(.3)))>Real(0.))))
          throw std::runtime_error("Physical plane winding/area differs");
        area+=Real(.5)*std::sqrt(squared);
      }
      if(q_surface && !area_checked) {
        Real total=0;
        MPI_Allreduce(&area,&total,1,sizeof(Real)==8?MPI_DOUBLE:MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
        error=std::max(error,std::abs(total-Real(.8)));area_checked=true;
      }
    }
  }
  astr_insitu_runtime_physical_group(0);
  cuda_ok(cudaFree(xyz));cuda_ok(cudaFree(values));
  return error;
}

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank=0,devices=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  try {
    cuda_ok(cudaGetDeviceCount(&devices));
    if(!devices) throw std::runtime_error("A CUDA device is required");
    cuda_ok(cudaSetDevice(rank%devices));
    constexpr int nx=7,ny=5,n=nx*ny;
    std::vector<Vector> positions(n);
    std::vector<Real> fields(3*n);
    for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
      const auto x=Real(i)/Real(nx-1),y=Real(j)/Real(ny-1);
      positions[i+nx*j]=Vector(x,y,Real(.1)*x);
    }
    Vector* xyz=nullptr;Real* values=nullptr;
    cuda_ok(cudaMalloc(&xyz,n*sizeof(Vector)));cuda_ok(cudaMalloc(&values,3*n*sizeof(Real)));
    cuda_ok(cudaMemcpy(xyz,positions.data(),n*sizeof(Vector),cudaMemcpyHostToDevice));
    Real error=Real(0.);
    for(const std::string quantity:{"velocity_x","u_Uinf","velocity_magnitude","p_pinf"}) {
      for(int i=0;i<n;++i) {
        fields[i]=quantity=="p_pinf"?Real(100.)+Real(10.)*positions[i][0]+positions[i][1]:Real(1.)+positions[i][0];
        fields[n+i]=Real(2.)-positions[i][1];fields[2*n+i]=Real(.5);
      }
      cuda_ok(cudaMemcpy(values,fields.data(),3*n*sizeof(Real),cudaMemcpyHostToDevice));
      astr_insitu::RuntimeProductView definition{};
      definition.id="component";definition.source="instantaneous";
      definition.labels[4]=quantity;definition.labels[6]="diverging_blue_red";
      definition.reals[22]=-2.;definition.reals[23]=4.;
      const int ids[]={quantity=="p_pinf"?5:2,3,4};
      const Real references[]={Real(2.),Real(100.),Real(300.)};
      const int components=quantity=="velocity_magnitude"?3:1;
      auto mesh=astr_insitu::build_runtime_index_mesh(definition,0,Real(0.),ids,components,nx,ny,
        reinterpret_cast<Real*>(xyz),values,references);
      if(mesh.draw.points!=n || mesh.draw.cells!=2*(nx-1)*(ny-1))
        throw std::runtime_error("Runtime image topology count differs");
      auto owner=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(mesh.owner);
      const auto& view=owner->get();
      std::vector<Real> actual(n);
      std::vector<Vector> geometry(n);
      std::vector<viskores::Id> indices(3*mesh.draw.cells);
      // Diagnostic-only oracle; the production caller passes these arrays directly to the renderer.
      cuda_ok(cudaMemcpy(actual.data(),view.scalar,n*sizeof(Real),cudaMemcpyDeviceToHost));
      cuda_ok(cudaMemcpy(geometry.data(),view.coordinates,n*sizeof(Vector),cudaMemcpyDeviceToHost));
      cuda_ok(cudaMemcpy(indices.data(),view.connectivity,indices.size()*sizeof(viskores::Id),cudaMemcpyDeviceToHost));
      for(int i=0;i<n;++i) {
        Real expected=fields[i];
        if(quantity=="u_Uinf") expected/=references[0];
        else if(quantity=="p_pinf") expected/=references[1];
        else if(quantity=="velocity_magnitude") expected=std::sqrt(
          fields[i]*fields[i]+fields[n+i]*fields[n+i]+fields[2*n+i]*fields[2*n+i]);
        if(!std::isfinite(actual[i])) throw std::runtime_error("Nonfinite runtime image scalar");
        error=std::max(error,std::abs(actual[i]-expected));
        for(int d=0;d<3;++d) error=std::max(error,std::abs(geometry[i][d]-positions[i][d]));
      }
      for(const auto index:indices) if(index<0 || index>=n)
        throw std::runtime_error("Runtime image connectivity exceeds its points");
      for(std::size_t i=0;i<indices.size();i+=3) {
        const auto a=geometry[indices[i]],b=geometry[indices[i+1]],c=geometry[indices[i+2]];
        const Real cross=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
        if(!(cross>Real(0.))) throw std::runtime_error("Zero-area or reversed runtime image triangle");
      }
    }
    astr_insitu::begin_runtime_index_group();
    {
      astr_insitu::RuntimeProductView definition{};
      definition.id="reuse";definition.labels[4]="velocity_x";
      definition.labels[6]="diverging_blue_red";definition.reals[22]=-2.;definition.reals[23]=4.;
      const int ids[]={2,3,4};const Real references[]={Real(2.),Real(100.),Real(300.)};
      auto first=astr_insitu::build_runtime_index_mesh(definition,1,Real(.001),ids,3,nx,ny,
        reinterpret_cast<Real*>(xyz),values,references,2);
      definition.id="other-camera";definition.reals[12]=20.;definition.reals[23]=6.;
      auto second=astr_insitu::build_runtime_index_mesh(definition,1,Real(.001),ids,3,nx,ny,
        reinterpret_cast<Real*>(xyz),values,references,2);
      if(first.owner!=second.owner) throw std::runtime_error("Display-only change repeated extraction");
      definition.id="other-field";definition.labels[4]="velocity_y";
      auto third=astr_insitu::build_runtime_index_mesh(definition,1,Real(.001),ids,3,nx,ny,
        reinterpret_cast<Real*>(xyz),values,references,3);
      const auto a=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(first.owner);
      const auto b=std::static_pointer_cast<astr_insitu::DeviceGeometryOwner>(third.owner);
      if(a->get().connectivity!=b->get().connectivity)
        throw std::runtime_error("Same slice repeated native triangulation");
      std::vector<Real> scalar(n);
      cuda_ok(cudaMemcpy(scalar.data(),b->get().scalar,n*sizeof(Real),cudaMemcpyDeviceToHost));
      for(int i=0;i<n;++i) error=std::max(error,std::abs(scalar[i]-fields[n+i]));
    }
    astr_insitu::end_runtime_index_group();
    error=std::max(error,physical_component(rank));
    error=std::max(error,physical_component(rank,true));
    cuda_ok(cudaFree(xyz));cuda_ok(cudaFree(values));
    Real global=0;
    MPI_Allreduce(&error,&global,1,sizeof(Real)==8?MPI_DOUBLE:MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
    if(sizeof(Real)==8 && global>Real(2e-10)) throw std::runtime_error("Runtime native image scalar differs");
    if(!rank) std::printf("RUNTIME_IMAGE_COMPONENT_PASS source_bytes=%zu maxabs=%.17g\n",sizeof(Real),double(global));
  } catch(const std::exception& error) {
    std::fprintf(stderr,"runtime plane probe: %s\n",error.what());MPI_Abort(MPI_COMM_WORLD,1);
  }
  MPI_Finalize();return 0;
}
