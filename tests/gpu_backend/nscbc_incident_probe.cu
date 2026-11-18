#include <cuda_runtime.h>
#include "nscbc_incident_math.cuh"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Real=astr::numerical_real;
constexpr int cases=9;
constexpr double tolerance=2e-10;
struct Input {
    Real parameters[3],face[45],external[45],right[225],left[225],metric[3],base[5],reference_rate[5];
    Real states[135],velocity[81],pressure[27],jacobian[27],line_metric[81],rest_jacobian[27],rest_metric[81];
};
struct Output {
    Real rate[45],lambda[45],rhs[45],normal[45],compatible[45],rest[45];
    Real preserved_rate[45],zero_rate[45],idempotent[45],late_rhs[45];
    int mask[45],status[45],failure_contract[9];
};

__global__ void evaluate(const Input *input,Output *output)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=cases) return;
    const Input &in=*input;Output &out=*output;
    const Real *face=in.face+5*c,*left=in.left+25*c,*right=in.right+25*c;
    const Real gamma=in.parameters[0],hn=in.parameters[1],jac=in.parameters[2];
    bool incoming[5];Real rate[5],lambda[5],rhs[5],normal[5],zero[5]={},rest_states[15];
    for(int k=0;k<5;++k) out.status[5*c+k]=-1;
    int status=astr::incident::incoming_rate(face,in.external+5*c,gamma,in.metric,hn,left,
        in.reference_rate,rate,lambda,incoming);
    out.status[5*c]=status;if(status) return;
    for(int m=0;m<5;++m) {
        out.rate[5*c+m]=rate[m];out.lambda[5*c+m]=lambda[m];out.mask[5*c+m]=incoming[m];rhs[m]=in.base[m];
    }
    status=astr::incident::apply_complete_rhs(rhs,jac,right,left,incoming,rate);
    out.status[5*c+1]=status;if(status) return;
    for(int m=0;m<5;++m) out.rhs[5*c+m]=rhs[m];
    status=astr::incident::apply_complete_rhs(rhs,jac,right,left,incoming,rate);
    if(status) { out.status[5*c+1]=status;return; }
    for(int m=0;m<5;++m) {
        out.idempotent[5*c+m]=rhs[m];
        rhs[m]=in.base[m]+Real(.1)*Real((m%2==0)?m+1:-(m+1));
    }
    status=astr::incident::apply_complete_rhs(rhs,jac,right,left,incoming,rate);
    if(status) { out.status[5*c+1]=status;return; }
    for(int m=0;m<5;++m) out.late_rhs[5*c+m]=rhs[m];
    status=astr::incident::normal_characteristics(in.states+15*c,in.velocity+9*c,in.pressure+3*c,
        in.jacobian+3*c,in.line_metric+9*c,left,normal);
    out.status[5*c+2]=status;if(status) return;
    for(int m=0;m<5;++m) { out.normal[5*c+m]=normal[m];rhs[m]=in.base[m]; }
    astr::incident::convective_compatibility(rhs,jac,left,normal,incoming,zero);
    status=astr::incident::apply_complete_rhs(rhs,jac,right,left,incoming,zero);
    out.status[5*c+3]=status;if(status) return;
    for(int m=0;m<5;++m) {
        out.compatible[5*c+m]=rhs[m];
        for(int j=0;j<3;++j) rest_states[m+5*j]=face[m];
    }
    status=astr::incident::normal_characteristics(rest_states,in.velocity+9*c,in.pressure+3*c,
        in.rest_jacobian+3*c,in.rest_metric+9*c,left,out.rest+5*c);
    out.status[5*c+4]=status;if(status) return;
    status=astr::incident::incoming_rate(face,face,gamma,in.metric,hn,left,in.reference_rate,rate,lambda,incoming);
    if(status) { out.status[5*c]=status;return; }
    for(int m=0;m<5;++m) { out.preserved_rate[5*c+m]=rate[m];zero[m]=0; }
    status=astr::incident::incoming_rate(face,face,gamma,in.metric,hn,left,zero,rate,lambda,incoming);
    if(status) { out.status[5*c]=status;return; }
    for(int m=0;m<5;++m) out.zero_rate[5*c+m]=rate[m];

    // Preserve the original status and no-write behavior on rejected inputs.
    int failures=0;
    status=astr::incident::incoming_rate(face,face,gamma,in.metric,Real(0),left,in.reference_rate,rate,lambda,incoming);
    if(status!=1) ++failures;
    for(int m=0;m<5;++m)
        if(rate[m]!=in.reference_rate[m]||lambda[m]!=0||incoming[m]) ++failures;
    for(int m=0;m<5;++m) rhs[m]=in.base[m];
    status=astr::incident::apply_complete_rhs(rhs,Real(0),right,left,incoming,rate);
    if(status!=1) ++failures;
    for(int m=0;m<5;++m) if(rhs[m]!=in.base[m]) ++failures;
    const Real invalid_jac[3]={Real(2),Real(0),Real(1)};
    status=astr::incident::normal_characteristics(in.states+15*c,in.velocity+9*c,in.pressure+3*c,
        invalid_jac,in.line_metric+9*c,left,normal);
    if(status!=1) ++failures;
    for(int m=0;m<5;++m) if(normal[m]!=0) ++failures;
    out.failure_contract[c]=failures;
}

template<class T> void read(std::ifstream &stream,T *values,size_t count)
{
    if(!stream.read(reinterpret_cast<char *>(values),sizeof(T)*count))
        throw std::runtime_error("truncated incident component packet");
}
void check(cudaError_t status,const char *operation)
{
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
double compare(const Real *actual,const Real *expected,int count)
{
    double error=0;
    for(int i=0;i<count;++i) {
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i])) throw std::runtime_error("nonfinite component result");
        error=std::max(error,std::abs(double(actual[i])-expected[i]));
    }
    if(error>tolerance) throw std::runtime_error("component difference exceeds 2e-10: "+std::to_string(error));
    return error;
}
double projection(const Real *matrix,int row,const Real *q)
{
    double value=0;for(int m=0;m<5;++m) value+=matrix[row+5*m]*q[m];return value;
}
}

int main(int argc,char **argv)
{
    try {
        if(argc!=2) throw std::runtime_error("usage: astr_backend_incident_math cpu_packet.bin");
        std::ifstream stream(argv[1],std::ios::binary);
        int32_t header[3];read(stream,header,3);
        if(header[0]!=1||header[1]!=sizeof(Real)||header[2]!=cases)
            throw std::runtime_error("incident packet kind/version/count mismatch");
        Input input{};Output expected{},actual{};
        read(stream,input.parameters,3);read(stream,input.face,45);read(stream,input.external,45);
        read(stream,input.right,225);read(stream,input.left,225);read(stream,input.metric,3);
        read(stream,input.base,5);read(stream,input.reference_rate,5);
        read(stream,input.states,135);read(stream,input.velocity,81);read(stream,input.pressure,27);
        read(stream,input.jacobian,27);read(stream,input.line_metric,81);
        read(stream,input.rest_jacobian,27);read(stream,input.rest_metric,81);
        read(stream,expected.rate,45);read(stream,expected.lambda,45);read(stream,expected.rhs,45);
        read(stream,expected.normal,45);read(stream,expected.compatible,45);read(stream,expected.rest,45);
        read(stream,expected.mask,45);
        if(stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("unexpected incident packet tail");
        Input *device_input=nullptr;Output *device_output=nullptr;
        check(cudaMalloc(&device_input,sizeof(Input)),"allocate input");
        check(cudaMalloc(&device_output,sizeof(Output)),"allocate output");
        check(cudaMemcpy(device_input,&input,sizeof(Input),cudaMemcpyHostToDevice),"upload frozen input");
        evaluate<<<1,32>>>(device_input,device_output);
        check(cudaGetLastError(),"launch incident component");check(cudaDeviceSynchronize(),"synchronize incident component");
        check(cudaMemcpy(&actual,device_output,sizeof(Output),cudaMemcpyDeviceToHost),"read component results");
        check(cudaFree(device_output),"free output");check(cudaFree(device_input),"free input");
        for(int c=0;c<cases;++c) {
            for(int k=0;k<5;++k) if(actual.status[5*c+k])
                throw std::runtime_error("valid incident case rejected: "+std::to_string(c)+" stage "+std::to_string(k));
            if(actual.failure_contract[c]) throw std::runtime_error("incident rejection contract changed");
        }
        double maximum=0;
        const Real *values[]={actual.rate,actual.lambda,actual.rhs,actual.normal,actual.compatible,actual.rest};
        const Real *references[]={expected.rate,expected.lambda,expected.rhs,expected.normal,expected.compatible,expected.rest};
        const char *names[]={"rate","lambda","rhs","normal","compatible","free_stream"};
        std::cout<<std::setprecision(17);
        for(int k=0;k<6;++k) {
            const double error=compare(values[k],references[k],45);maximum=std::max(maximum,error);
            std::cout<<"INCIDENT_"<<names[k]<<"_MAX_ABS "<<error<<'\n';
        }
        const int counts[9]={1,4,0,5,1,0,4,1,1};double invariant=0;
        for(int c=0;c<cases;++c) {
            int count=0;
            for(int m=0;m<5;++m) {
                if(actual.mask[5*c+m]!=expected.mask[5*c+m]) throw std::runtime_error("characteristic mask changed");
                count+=actual.mask[5*c+m];
                invariant=std::max(invariant,std::abs(double(actual.preserved_rate[5*c+m])-input.reference_rate[m]));
                if(actual.zero_rate[5*c+m]!=0) throw std::runtime_error("stationary reference changed");
                invariant=std::max(invariant,std::abs(double(actual.idempotent[5*c+m])-actual.rhs[5*c+m]));
                Real late[5];for(int n=0;n<5;++n) late[n]=input.base[n]+Real(.1)*Real((n%2==0)?n+1:-(n+1));
                const Real *left=input.left+25*c;const double jac=input.parameters[2];
                const double desired=actual.mask[5*c+m]?actual.rate[5*c+m]:projection(left,m,input.base)/jac;
                invariant=std::max(invariant,std::abs(projection(left,m,actual.rhs+5*c)/jac-desired));
                const double late_desired=actual.mask[5*c+m]?actual.rate[5*c+m]:projection(left,m,late)/jac;
                invariant=std::max(invariant,std::abs(projection(left,m,actual.late_rhs+5*c)/jac-late_desired));
            }
            if(count!=counts[c]) throw std::runtime_error("normal characteristic count changed");
        }
        if(!std::isfinite(invariant)||invariant>tolerance) throw std::runtime_error("incident characteristic invariant failed");
        std::cout<<"INCIDENT_INVARIANT_MAX_ABS "<<invariant<<'\n';
        std::cout<<"INCIDENT_CPP_COMPONENT_PASS cases=9 max_abs="<<maximum<<" failure_contract=0\n";
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
