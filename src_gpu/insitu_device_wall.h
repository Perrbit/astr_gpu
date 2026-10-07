#ifndef ASTR_INSITU_DEVICE_WALL_H
#define ASTR_INSITU_DEVICE_WALL_H
#include "insitu_device_geometry.h"

namespace astr_insitu {
struct PackWallField : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5);
  viskores::Id nodes,component;
  VISKORES_CONT PackWallField(viskores::Id n,int c):nodes(n),component(c) {}
  template<class Coordinates,class Fields>
  VISKORES_EXEC void operator()(viskores::Id id,const Coordinates& input,const Fields& fields,
      viskores::Vec3f& xyz,double& scalar) const {
    for(int d=0;d<3;++d) xyz[d]=input.Get(id+d*nodes);
    scalar=fields.Get(id+component*nodes);
  }
};
struct WallTriangleIndex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id im,km,nodes,normal_component;
  VISKORES_CONT WallTriangleIndex(viskores::Id x,viskores::Id z,viskores::Id n,int normal):
    im(x),km(z),nodes(n),normal_component(normal) {}
  template<class Fields>
  VISKORES_EXEC void operator()(viskores::Id id,const Fields& fields,viskores::Id& index) const {
    const auto cell=id/6,corner=id%6,wall=cell/(im*km),local=cell%(im*km);
    const auto base=local%im+(im+1)*(local/im+wall*(km+1));
    const viskores::Id positive[6]={0,im+1,im+2,0,im+2,1};
    const viskores::Id negative[6]={0,1,im+2,0,im+2,im+1};
    index=base+(fields.Get(base+normal_component*nodes)>0.?positive[corner]:negative[corner]);
  }
};
using WallAudit=viskores::Vec<double,3>;
struct CheckWallTriangles : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5);
  viskores::Id nodes,normal_component;
  VISKORES_CONT CheckWallTriangles(viskores::Id n,int normal):nodes(n),normal_component(normal) {}
  template<class Coordinates,class Indices,class Fields>
  VISKORES_EXEC void operator()(viskores::Id triangle,const Coordinates& xyz,const Indices& indices,
      const Fields& fields,WallAudit& result) const {
    const auto a=indices.Get(3*triangle),b=indices.Get(3*triangle+1),c=indices.Get(3*triangle+2);
    const auto normal=viskores::Cross(xyz.Get(b)-xyz.Get(a),xyz.Get(c)-xyz.Get(a));
    const double area=.5*viskores::Sqrt(viskores::Dot(normal,normal)),side=fields.Get(a+normal_component*nodes);
    const bool valid=viskores::IsFinite(area) && area>0. && viskores::IsFinite(side) &&
      side!=0. && viskores::Abs(side)<=1. && normal[1]*side>0. &&
      fields.Get(b+normal_component*nodes)*side>0. && fields.Get(c+normal_component*nodes)*side>0.;
    result=WallAudit(valid?area:0.,valid?0.:1.,side>0.?area:0.);
  }
};
using WallMeasure=viskores::Vec<double,2>;
struct CheckWallQuads : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id im,km;
  VISKORES_CONT CheckWallQuads(viskores::Id x,viskores::Id z):im(x),km(z) {}
  template<class Coordinates>
  VISKORES_EXEC void operator()(viskores::Id cell,const Coordinates& xyz,WallMeasure& result) const {
    const auto wall=cell/(im*km),local=cell%(im*km);
    const auto base=local%im+(im+1)*(local/im+wall*(km+1));
    const auto a=xyz.Get(base),b=xyz.Get(base+1),c=xyz.Get(base+im+1),d=xyz.Get(base+im+2);
    const double gauss[2]={.5-.5/viskores::Sqrt(3.),.5+.5/viskores::Sqrt(3.)};
    double area=0.;viskores::Vec3f reference(0.);
    for(int q=0;q<2;++q) for(int p=0;p<2;++p) {
      const double s=gauss[p],t=gauss[q];
      const auto ds=(1.-t)*(b-a)+t*(d-c),dt=(1.-s)*(c-a)+s*(d-b);
      const auto normal=viskores::Cross(ds,dt);
      const double measure=viskores::Sqrt(viskores::Dot(normal,normal));
      if(!viskores::IsFinite(measure) || measure<=0.) {result=WallMeasure(0.,1.);return;}
      if(p==0 && q==0) reference=normal;
      if(viskores::Dot(reference,normal)<=0.) {result=WallMeasure(0.,1.);return;}
      area+=measure/4.;
    }
    result=WallMeasure(area,viskores::IsFinite(area) && area>0.?0.:1.);
  }
};
struct WallProduct {
  viskores::cont::DataSet data;
  WallAudit audit{0.,0.,0.};
  WallMeasure measure{0.,0.};
};
inline WallProduct extract_bc41_wall(double* coordinates_pointer,double* fields_pointer,
    int im,int km,int nw,int component,const char* name,int field_count=4,int normal_component=3) {
  if(im<1 || km<1 || nw<0 || nw>2 || field_count<2 || field_count>18 || component<0 ||
      component>=field_count || normal_component<0 || normal_component>=field_count ||
      component==normal_component || !name)
    throw std::invalid_argument("Invalid device wall product descriptor");
  WallProduct result;
  const viskores::Id nodes=viskores::Id(im+1)*(km+1)*nw,cells=viskores::Id(im)*km*nw*2;
  if(!nodes) return result;
  auto source=borrow_cuda_array(coordinates_pointer,nodes*3);
  auto fields=borrow_cuda_array(fields_pointer,nodes*field_count);
  viskores::cont::ArrayHandle<viskores::Vec3f> xyz;
  viskores::cont::ArrayHandle<double> scalar;
  viskores::cont::ArrayHandle<viskores::Id> indices;
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  invoke(PackWallField(nodes,component),viskores::cont::ArrayHandleIndex(nodes),source,fields,xyz,scalar);
  synchronize_device_stage("PackWallField");
  invoke(WallTriangleIndex(im,km,nodes,normal_component),viskores::cont::ArrayHandleIndex(3*cells),fields,indices);
  synchronize_device_stage("WallTriangleIndex");
  viskores::cont::ArrayHandle<WallAudit> audit;
  invoke(CheckWallTriangles(nodes,normal_component),viskores::cont::ArrayHandleIndex(cells),xyz,indices,fields,audit);
  synchronize_device_stage("CheckWallTriangles");
  result.audit=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},audit,WallAudit(0.));
  synchronize_device_stage("WallTriangleReduce");
  if(result.audit[1]!=0.) throw std::runtime_error("Invalid wall triangle area/winding");
  viskores::cont::ArrayHandle<WallMeasure> measures;
  invoke(CheckWallQuads(im,km),viskores::cont::ArrayHandleIndex(cells/2),xyz,measures);
  synchronize_device_stage("CheckWallQuads");
  result.measure=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},measures,WallMeasure(0.));
  synchronize_device_stage("WallQuadReduce");
  if(result.measure[1]!=0.) throw std::runtime_error("Invalid wall bilinear quadrature");
  require_device_only(source);require_device_only(fields);require_device_only(xyz);
  require_device_only(scalar);require_device_only(indices);require_device_only(audit);
  require_device_only(measures);
  viskores::cont::CellSetSingleType<> connectivity;
  connectivity.Fill(nodes,viskores::CELL_SHAPE_TRIANGLE,3,indices);
  result.data.SetCellSet(connectivity);
  result.data.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",xyz));
  result.data.AddPointField(name,scalar);
  return result;
}
} // namespace astr_insitu
#endif
