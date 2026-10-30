#include "insitu_compact_mesh.h"
#include "insitu_product_failure.h"
#include "insitu_streamline_seeds.h"
#include <vtkAOSDataArrayTemplate.h>
#include <vtkSOADataArrayTemplate.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkClipPolyData.h>
#include <vtkContourFilter.h>
#include <vtkCutter.h>
#include <vtkDataSet.h>
#include <vtkErrorCode.h>
#include <vtkIdList.h>
#include <vtkNew.h>
#include <vtkPlane.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPStreamTracer.h>
#include <vtkRungeKutta45.h>
#include <vtkStructuredGrid.h>
#include <vtkTriangleFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>

extern "C" int astr_insitu_runtime_host_begin(const char*,const char*,int,int,astr_insitu::Real);
extern "C" void astr_insitu_runtime_failure(const char*);

namespace {
using Real=astr_insitu::Real;
using Array=vtkAOSDataArrayTemplate<Real>;
using View=astr_insitu::RuntimeProductView;
using Packet=std::shared_ptr<astr_insitu::CompactMesh>;
struct Extraction {
  std::string kind,sampling,seeds,direction;
  int steps;
  std::array<double,18> values;
  bool operator<(const Extraction& b) const {
    return std::tie(kind,sampling,seeds,direction,steps,values)<
      std::tie(b.kind,b.sampling,b.seeds,b.direction,b.steps,b.values);
  }
};
struct Quantity {
  std::string name;
  int field;
  Real factor;
  std::vector<int> fields;
  bool geometry;
  bool operator<(const Quantity& b) const {
    return std::tie(name,field,factor,fields,geometry)<std::tie(b.name,b.field,b.factor,b.fields,b.geometry);
  }
};
struct Geometry {
  vtkSmartPointer<vtkPolyData> data;
  std::map<Quantity,Packet> packets;
};
struct HostGroup {
  bool active=false,prepared=false,download_counted=false;
  Real* coordinates=nullptr;
  Real* values=nullptr;
  std::array<int,3> extent{};
  std::vector<int> fields;
  vtkSmartPointer<vtkDataSet> input;
  std::map<Extraction,Geometry> geometry;
  int extractions=0,integrations=0,reuse=0;
  std::uint64_t source_bytes=0;
} group;

void prepare(int axis,const int* extent,const int* fields,int count,Real* xyz,Real* values) {
  const std::array<int,3> shape{{extent[0],extent[1],extent[2]}};
  const std::vector<int> ids(fields,fields+count);
  if(group.prepared) {
    if(group.coordinates!=xyz || group.values!=values || group.extent!=shape || group.fields!=ids)
      throw std::logic_error("Host extraction group received another frozen source");
    return;
  }
  const vtkIdType nodes=vtkIdType(extent[0])*extent[1]*extent[2];
  vtkNew<Array> positions;positions->SetNumberOfComponents(3);
  if(nodes) positions->SetArray(xyz,3*nodes,1);
  vtkNew<vtkPoints> points;points->SetData(positions);
  if(axis) {
    int axes[2],n=0;
    for(int d=0;d<3;++d) if(d!=axis-1) axes[n++]=d;
    vtkNew<vtkPolyData> plane;plane->SetPoints(points);
    vtkNew<vtkCellArray> quads;
    for(int j=0;j<extent[axes[1]]-1;++j) for(int i=0;i<extent[axes[0]]-1;++i) {
      const vtkIdType a=i+vtkIdType(extent[axes[0]])*j;
      const vtkIdType ids[4]={a,a+1,a+extent[axes[0]]+1,a+extent[axes[0]]};
      quads->InsertNextCell(4,ids);
    }
    plane->SetPolys(quads);group.input=plane;
  } else {
    vtkNew<vtkStructuredGrid> volume;volume->SetDimensions(extent);volume->SetPoints(points);
    group.input=volume;
  }
  for(int c=0;c<count;++c) {
    vtkNew<Array> array;array->SetName(("raw"+std::to_string(fields[c])).c_str());
    if(nodes) array->SetArray(values+c*nodes,nodes,1);
    group.input->GetPointData()->AddArray(array);
  }
  if(std::find(ids.begin(),ids.end(),2)!=ids.end() && std::find(ids.begin(),ids.end(),3)!=ids.end() &&
      std::find(ids.begin(),ids.end(),4)!=ids.end()) {
    vtkNew<vtkSOADataArrayTemplate<Real>> velocity;velocity->SetName("velocity");velocity->SetNumberOfComponents(3);
    for(int d=0;d<3;++d) {
      const int slot=std::find(ids.begin(),ids.end(),d+2)-ids.begin();
      if(nodes) velocity->SetArray(d,values+slot*nodes,nodes,true,true);
    }
    group.input->GetPointData()->SetVectors(velocity);
  }
  group.source_bytes=std::uint64_t(nodes)*(count+3)*sizeof(Real);
  group.coordinates=xyz;group.values=values;group.extent=shape;group.fields=ids;group.prepared=true;
}

Extraction extraction(const View& view) {
  Extraction result{view.labels[0],view.labels[3],view.labels[7],view.labels[9],view.integers[6],{}};
  std::copy_n(view.reals.begin(),12,result.values.begin());
  std::copy_n(view.reals.begin()+25,6,result.values.begin()+12);
  return result;
}

vtkSmartPointer<vtkPolyData> extract(const View& view) {
  vtkSmartPointer<vtkPolyData> data;
  if(view.labels[0]=="streamlines") {
    vtkNew<vtkPolyData> seeds;vtkNew<vtkPoints> sites;sites->SetDataTypeToDouble();
    const auto layout=astr_insitu::tgv_streamline_seeds(view.labels[7]);
    for(const auto& point:layout) sites->InsertNextPoint(point[0],point[1],point[2]);
    seeds->SetPoints(sites);
    vtkNew<vtkPStreamTracer> trace;trace->SetInputData(group.input);trace->SetSourceData(seeds);
    trace->SetInputArrayToProcess(0,0,0,vtkDataObject::FIELD_ASSOCIATION_POINTS,"velocity");
    vtkNew<vtkRungeKutta45> integrator;trace->SetIntegrator(integrator);
    trace->SetIntegrationStepUnit(vtkStreamTracer::LENGTH_UNIT);
    trace->SetInitialIntegrationStep(view.reals[26]);trace->SetMinimumIntegrationStep(view.reals[27]);
    trace->SetMaximumIntegrationStep(view.reals[28]);trace->SetMaximumError(view.reals[29]);
    trace->SetMaximumPropagation(view.reals[30]);trace->SetMaximumNumberOfSteps(view.integers[6]);
    trace->SetIntegrationDirection(view.labels[9]=="both"?vtkStreamTracer::BOTH:
      (view.labels[9]=="forward"?vtkStreamTracer::FORWARD:vtkStreamTracer::BACKWARD));
    trace->SetUseLocalSeedSource(false);trace->SetComputeVorticity(false);
    trace->Update();
    if(trace->GetErrorCode()!=vtkErrorCode::NoError) throw std::runtime_error("Existing VTK parallel streamline failed");
    data=vtkSmartPointer<vtkPolyData>::New();data->ShallowCopy(trace->GetOutput());++group.integrations;
  } else if(view.labels[0]=="q_surface") {
    vtkNew<vtkContourFilter> filter;filter->SetInputData(group.input);
    filter->SetInputArrayToProcess(0,0,0,vtkDataObject::FIELD_ASSOCIATION_POINTS,"raw22");
    filter->SetValue(0,view.reals[25]);filter->ComputeNormalsOff();filter->ComputeGradientsOff();
    filter->SetOutputPointsPrecision(sizeof(Real)==8?vtkAlgorithm::DOUBLE_PRECISION:vtkAlgorithm::SINGLE_PRECISION);
    filter->Update();
    if(filter->GetErrorCode()!=vtkErrorCode::NoError) throw std::runtime_error("Existing VTK Q contour failed");
    data=vtkSmartPointer<vtkPolyData>::New();data->ShallowCopy(filter->GetOutput());
  } else if(view.labels[3]=="physical_plane") {
    vtkNew<vtkPlane> plane;plane->SetOrigin(view.reals.data());plane->SetNormal(view.reals.data()+3);
    vtkNew<vtkCutter> filter;filter->SetInputData(group.input);filter->SetCutFunction(plane);filter->GenerateTrianglesOn();
    filter->SetOutputPointsPrecision(sizeof(Real)==8?vtkAlgorithm::DOUBLE_PRECISION:vtkAlgorithm::SINGLE_PRECISION);
    filter->Update();
    if(filter->GetErrorCode()!=vtkErrorCode::NoError) throw std::runtime_error("Existing VTK physical cut failed");
    data=vtkSmartPointer<vtkPolyData>::New();data->ShallowCopy(filter->GetOutput());
  } else {
    data=vtkPolyData::SafeDownCast(group.input);
  }
  if(view.labels[3]!="grid_index") for(int d=0;d<3;++d) for(int side=0;side<2;++side) {
    vtkNew<vtkPlane> plane;double origin[3]={0.,0.,0.},normal[3]={0.,0.,0.};
    origin[d]=view.reals[6+2*d+side];normal[d]=side?-1.:1.;plane->SetOrigin(origin);plane->SetNormal(normal);
    vtkNew<vtkClipPolyData> clip;clip->SetInputData(data);clip->SetClipFunction(plane);clip->Update();
    auto selected=vtkSmartPointer<vtkPolyData>::New();selected->ShallowCopy(clip->GetOutput());data=selected;
  }
  vtkNew<vtkTriangleFilter> triangles;triangles->SetInputData(data);triangles->PassVertsOff();triangles->Update();
  auto result=vtkSmartPointer<vtkPolyData>::New();result->ShallowCopy(triangles->GetOutput());
  return result;
}

Quantity quantity(const View& view,int field,const Real* references) {
  Quantity result{view.labels[4],field,Real(1.),{},view.labels[1]=="geometry"};
  for(const auto& value:view.fields) result.fields.push_back(value.first);
  const auto& name=view.labels[4];
  if(name=="u_Uinf" || name=="w_Uinf" || name=="p_pinf" || name=="T_Tinf") {
    const Real reference=references[name=="p_pinf"?1:(name=="T_Tinf"?2:0)];
    if(!std::isfinite(reference) || reference==Real(0.)) throw std::invalid_argument("Host normalization requires a finite nonzero reference");
    result.factor=Real(1.)/reference;
    if(!std::isfinite(result.factor)) throw std::invalid_argument("Host reference reciprocal is not representable");
  }
  return result;
}

Packet packet(vtkPolyData* data,const View& view,const Quantity& selected) {
  auto result=std::make_shared<astr_insitu::CompactMesh>();result->name="runtime_product";
  const bool lines=view.labels[0]=="streamlines";
  result->shape=lines?"line":"tri";
  const auto nodes=data->GetNumberOfPoints();
  for(auto& axis:result->coordinates) axis.resize(nodes);
  for(vtkIdType i=0;i<nodes;++i) {
    double xyz[3];data->GetPoint(i,xyz);
    for(int d=0;d<3;++d) {
      if(!std::isfinite(xyz[d])) throw std::runtime_error("Nonfinite host product coordinate");
      result->coordinates[d][i]=Real(xyz[d]);
    }
  }
  auto copy_field=[&](const char* name,const char* source) {
    auto array=data->GetPointData()->GetArray(source);
    if(nodes && !array) throw std::runtime_error(std::string("Existing host geometry lacks ")+source);
    auto& values=result->point_fields[name];values.resize(nodes);
    for(vtkIdType i=0;i<nodes;++i) {
      values[i]=Real(array->GetComponent(i,0));
      if(!std::isfinite(values[i])) throw std::runtime_error(std::string("Nonfinite host product field ")+name);
    }
  };
  if(selected.geometry) {
    for(const auto& field:view.fields) copy_field(field.second.c_str(),("raw"+std::to_string(field.first)).c_str());
    if(lines) copy_field("IntegrationTime","IntegrationTime");
  } else {
    auto& values=result->point_fields["quantity"];values.resize(nodes);
    if(selected.name=="velocity_magnitude") {
      auto u=data->GetPointData()->GetArray("raw2"),v=data->GetPointData()->GetArray("raw3"),w=data->GetPointData()->GetArray("raw4");
      if(nodes && (!u || !v || !w)) throw std::runtime_error("Host geometry lacks velocity components");
      for(vtkIdType i=0;i<nodes;++i) {
        const Real a=Real(u->GetComponent(i,0)),b=Real(v->GetComponent(i,0)),c=Real(w->GetComponent(i,0));
        values[i]=std::sqrt(a*a+b*b+c*c);
        if(!std::isfinite(values[i])) throw std::runtime_error("Nonfinite host velocity magnitude");
      }
    } else {
      copy_field("quantity",("raw"+std::to_string(selected.field)).c_str());
      for(auto& value:values) {
        value*=selected.factor;
        if(!std::isfinite(value)) throw std::runtime_error("Nonfinite normalized host quantity");
      }
    }
  }
  vtkNew<vtkIdList> ids;
  auto cells=lines?data->GetLines():data->GetPolys();cells->InitTraversal();
  auto seeds=data->GetCellData()->GetArray("SeedIds");
  auto reasons=data->GetCellData()->GetArray("ReasonForTermination");
  auto times=data->GetPointData()->GetArray("IntegrationTime");
  vtkIdType cell=0;
  while(cells->GetNextCell(ids)) {
    if(lines) {
      if(!seeds || !reasons || !times) throw std::runtime_error("Existing VTK streamline lacks native identity/termination data");
      for(vtkIdType edge=0;edge+1<ids->GetNumberOfIds();++edge) {
        result->connectivity.insert(result->connectivity.end(),{ids->GetId(edge),ids->GetId(edge+1)});
        result->seed_ids.push_back(std::int64_t(seeds->GetComponent(cell,0)));
        const double t=times->GetComponent(ids->GetId(0),0)+times->GetComponent(ids->GetId(ids->GetNumberOfIds()-1),0);
        result->directions.push_back(t<0.?-1:1);
        if(selected.geometry) result->cell_fields["ReasonForTermination"].push_back(std::int64_t(reasons->GetComponent(cell,0)));
      }
    } else {
      if(ids->GetNumberOfIds()!=3) throw std::runtime_error("Existing host surface triangulation did not return triangles");
      for(int d=0;d<3;++d) result->connectivity.push_back(ids->GetId(d));
    }
    ++cell;
  }
  auto& controls=result->controls;
  controls["host_source"]=1.;controls["device_points"]=nodes;
  controls["device_cells"]=result->connectivity.size()/(lines?2:3);controls["device_arity"]=lines?2:3;
  controls["source_real_bytes"]=sizeof(Real);controls["geometry_download_bytes"]=0.;
  std::uint64_t bytes=(result->connectivity.size()+result->seed_ids.size()+result->directions.size())*sizeof(std::int64_t);
  for(const auto& field:result->point_fields) bytes+=field.second.size()*sizeof(Real);
  for(const auto& field:result->cell_fields) bytes+=field.second.size()*sizeof(std::int64_t);
  bytes+=std::uint64_t(nodes)*3*sizeof(Real);controls["geometry_host_bytes"]=double(bytes);
  if(nodes) {
    const double* bounds=data->GetBounds();for(int d=0;d<6;++d) controls["bound"+std::to_string(d)]=bounds[d];
    const auto& scalar=result->point_fields.begin()->second;
    const auto range=std::minmax_element(scalar.begin(),scalar.end());
    controls["actual_min"]=*range.first;controls["actual_max"]=*range.second;
  } else for(int d=0;d<6;++d) controls["bound"+std::to_string(d)]=0.;
  return result;
}
}

extern "C" void astr_insitu_runtime_host_group(int begin) {
  if(!begin) {
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    std::printf("ASTR_RUNTIME_HOST_GROUP rank=%d extractions=%d integrations=%d owner_reuse=%d retained_after=0\n",
      rank,group.extractions,group.integrations,group.reuse);
  }
  group=HostGroup{};group.active=begin!=0;
}

extern "C" int astr_insitu_render_host_product(const char* backend,const char* script,int fcomm,int step,Real time,
    const char* id,const char* source,const char* definition,const char* directory,std::int64_t version,
    const char* labels,const int* words,const double* settings,const int* field_ids,int nfields,const int* extent,
    int axis,Real* coordinates,Real* values,const Real* references,int provider,const Real* statistics,
    std::int64_t samples,int selected_field,std::int64_t host_budget,std::int64_t download) {
  const auto comm=MPI_Comm_f2c(fcomm);
  const double started=MPI_Wtime();
  View context;std::string error;Extraction key;Quantity color;
  try {
    if(!group.active) throw std::logic_error("Host product needs a frozen extraction group");
    context=astr_insitu::runtime_product_view(id,source,definition,directory,version,labels,words,settings);
    context.statistics_provider=provider;context.samples=samples;std::copy_n(statistics,3,context.statistics.begin());
    prepare(axis,extent,field_ids,nfields,coordinates,values);
    key=extraction(context);color=quantity(context,selected_field,references);
#ifdef ASTR_BUILD_TESTING
    if(const char* fault=std::getenv("ASTR_INSITU_TEST_RUNTIME_ALLOCATION_REFUSAL")) {
      int rank=0;MPI_Comm_rank(comm,&rank);
      if(std::string(fault)==context.id+":"+std::to_string(step)+":"+std::to_string(rank)+":host") host_budget=0;
    }
#endif
    if(group.source_bytes>std::uint64_t(host_budget)) throw std::runtime_error("Host source exceeds its explicit buffer budget");
  } catch(const std::exception& caught) {error=caught.what();}
  if(astr_insitu::consensus_product_failure(comm,error)) {astr_insitu_runtime_failure(error.c_str());return 1;}
  if(astr_insitu_runtime_host_begin(backend,script,fcomm,step,time)) return 1;
  auto found=group.geometry.find(key);
  if(found==group.geometry.end()) {
    try {
      Geometry value;value.data=extract(context);found=group.geometry.emplace(key,std::move(value)).first;++group.extractions;
    } catch(const std::exception& caught) {
      if(context.labels[0]=="streamlines") {
        std::fprintf(stderr,"Existing distributed VTK streamline stage failed: %s\n",caught.what());MPI_Abort(comm,1);return 1;
      }
      error=caught.what();
    }
  }
  if(astr_insitu::consensus_product_failure(comm,error)) {astr_insitu_runtime_failure(error.c_str());return 1;}
  Packet result;
  try {
    auto& geometry=found->second;auto previous=geometry.packets.find(color);
    if(previous!=geometry.packets.end()) {result=previous->second;++group.reuse;}
    else {
      result=packet(geometry.data,context,color);
      std::uint64_t bytes=group.source_bytes;
      for(const auto& item:group.geometry) for(const auto& entry:item.second.packets)
        bytes+=std::uint64_t(entry.second->controls.at("geometry_host_bytes"));
      if(bytes>std::uint64_t(host_budget) ||
          std::uint64_t(result->controls.at("geometry_host_bytes"))>std::uint64_t(host_budget)-bytes)
        throw std::runtime_error("Host extracted products exceed their explicit buffer budget");
      geometry.packets.emplace(color,result);
    }
    result->controls["source_field_download_bytes"]=group.download_counted?0.:double(download);group.download_counted=true;
  } catch(const std::exception& caught) {error=caught.what();}
  if(astr_insitu::consensus_product_failure(comm,error)) {astr_insitu_runtime_failure(error.c_str());return 1;}
  const double prepared=MPI_Wtime();
  const int status=astr_insitu::render_compact_products(backend,script,fcomm,step,time,"runtime_catalog",false,Real(0.),Real(0.),Real(0.),{},&context,result);
  int rank=0;MPI_Comm_rank(comm,&rank);
  std::printf("ASTR_RUNTIME_HOST_TIMING rank=%d id=%s step=%d prepare_inclusive_seconds=%.17g consumer_inclusive_seconds=%.17g status=%d\n",
    rank,id,step,prepared-started,MPI_Wtime()-prepared,status);
  return status;
}
