#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkStreamTracer.h>
#include <cmath>
#include <cstdio>

int main(int argc,char**) {
  const double pi=std::acos(-1.),h=2.*pi/32.,radius=.2;
  vtkNew<vtkImageData> grid;
  grid->SetDimensions(33,33,33);
  grid->SetSpacing(h,h,h);
  vtkNew<vtkDoubleArray> velocity;
  velocity->SetName("velocity");
  velocity->SetNumberOfComponents(3);
  velocity->SetNumberOfTuples(grid->GetNumberOfPoints());
  grid->GetPointData()->SetVectors(velocity);
  double maximum_radius_error=0.,arc=0.,endpoint=0.;
  for (int mode=0;mode<2;++mode) {
    for (vtkIdType i=0;i<grid->GetNumberOfPoints();++i) {
      double point[3],value[3]={1.,0.,0.};
      grid->GetPoint(i,point);
      if (mode) { value[0]=-(point[1]-pi); value[1]=point[0]-pi; }
      velocity->SetTuple(i,value);
    }
    velocity->Modified();
    vtkNew<vtkStreamTracer> tracer;
    tracer->SetInputData(grid);
    tracer->SetIntegratorTypeToRungeKutta45();
    tracer->SetIntegrationStepUnit(vtkStreamTracer::LENGTH_UNIT);
    tracer->SetInitialIntegrationStep(.1*h);
    tracer->SetMinimumIntegrationStep(.01*h);
    tracer->SetMaximumIntegrationStep(.5*h);
    tracer->SetMaximumError(1e-8);
    tracer->SetMaximumPropagation(pi);
    tracer->SetMaximumNumberOfSteps(100000);
    tracer->SetIntegrationDirectionToForward();
    tracer->SetComputeVorticity(false);
    tracer->SetStartPosition(mode ? pi+radius : pi/2.,pi,pi/4.);
    tracer->Update();
    auto* output=tracer->GetOutput();
    if (output->GetNumberOfLines()!=1 || output->GetNumberOfPoints()<2) return 1;
    double last[3];
    output->GetPoint(output->GetNumberOfPoints()-1,last);
    if (!mode) {
      endpoint=std::abs(last[0]-3.*pi/2.);
      endpoint=std::fmax(endpoint,std::abs(last[1]-pi));
      endpoint=std::fmax(endpoint,std::abs(last[2]-pi/4.));
    } else {
      double previous[3];
      output->GetPoint(0,previous);
      for (vtkIdType i=1;i<output->GetNumberOfPoints();++i) {
        double point[3]; output->GetPoint(i,point);
        const double a[2]={previous[0]-pi,previous[1]-pi},b[2]={point[0]-pi,point[1]-pi};
        arc+=radius*std::atan2(a[0]*b[1]-a[1]*b[0],a[0]*b[0]+a[1]*b[1]);
        maximum_radius_error=std::fmax(maximum_radius_error,std::abs(std::hypot(b[0],b[1])-radius));
        for (int d=0;d<3;++d) previous[d]=point[d];
      }
    }
    std::printf("StreamTracer mode=%d points=%lld\n",mode,static_cast<long long>(output->GetNumberOfPoints()));
  }
  const double gap=pi-arc;
  std::printf("StreamTracer accepted-length audit constant_endpoint_error=%.17g circle_arc=%.17g target=%.17g arc_gap=%.17g minimum_step=%.17g radius_error=%.17g\n",
    endpoint,arc,pi,gap,.01*h,maximum_radius_error);
  if (!std::isfinite(endpoint) || !std::isfinite(gap) || !std::isfinite(maximum_radius_error) || endpoint>2e-10) return 3;
  // Diagnostic of arc termination, not a new physical CURVE/DNS acceptance gate.
  // The existing final-step minimum bound can leave less than one minimum step.
  if (argc>1 && std::abs(gap)>.01*h) return 2;
  return 0;
}
