#include <vtkFunctionSet.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkRungeKutta45.h>
#include <cmath>
#include <cstdio>
#include <initializer_list>

class CircularTangent : public vtkFunctionSet {
public:
  static CircularTangent* New();
  vtkTypeMacro(CircularTangent,vtkFunctionSet);
  using vtkFunctionSet::FunctionValues;
  int FunctionValues(double* x,double* f) override {
    const double radius=std::sqrt(x[0]*x[0]+x[1]*x[1]);
    if (radius==0.) return 0;
    f[0]=-x[1]/radius; f[1]=x[0]/radius; f[2]=0.;
    return 1;
  }
protected:
  CircularTangent() { NumFuncs=3; NumIndepVars=4; }
};
vtkStandardNewMacro(CircularTangent);

// Audit only: demonstrates that VTK RK45's returned step is the NEXT suggestion,
// not necessarily the accepted length. It does not alter either installed VTK.
int main() {
  vtkNew<CircularTangent> field;
  vtkNew<vtkRungeKutta45> integrator;
  integrator->SetFunctionSet(field);
  const double h=2.*std::acos(-1.)/32.;
  bool distinct=false;
  for (double radius : {.05,.1,.2,.5,1.}) {
    double previous[3]={radius,0.,0.},next[3]={},step=.1*h,actual=0.,error=0.;
    const int status=integrator->ComputeNextStep(previous,nullptr,next,0.,step,actual,
                                                .01*h,.5*h,1e-8,error);
    if (status!=0 || !std::isfinite(actual) || !std::isfinite(step) || !std::isfinite(error)) return 1;
    const double mismatch=std::abs(std::abs(step)-std::abs(actual));
    std::printf("VTK length audit radius=%.3g accepted=%.17g suggested=%.17g mismatch=%.17g error=%.17g\n",
      radius,actual,step,mismatch,error);
    distinct=distinct || mismatch>2e-10;
  }
  return distinct ? 0 : 2;
}
