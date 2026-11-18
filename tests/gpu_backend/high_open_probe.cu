#include <cuda_runtime.h>
#include "high_open_math.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Real=astr::numerical_real;
struct Input {
    Real samples[80],primitive[400],lambda[50],rhs[50],lodi[50],left[25];
    Real states[400],velocity[240],pressure[80],jacobian[80],metric[240];
};
struct Output {
    Real derivative[30],flux[60],filtered[30],rate[50],normal_d[50],normal_r[50];
    int positions[144];
};

__global__ void evaluate(const Input *in,Output *out)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=10) return;
    const Real *f=in->samples+8*c;
    for(int row=0;row<3;++row) {
        out->derivative[row+3*c]=astr::high_open::derivative6(f,row);
        for(int wind=0;wind<2;++wind)
            out->flux[row+3*wind+6*c]=astr::high_open::mp_flux6(f,row,wind==1);
        out->filtered[row+3*c]=astr::high_open::filter6(f,row);
    }
    bool incoming[5];for(int m=0;m<5;++m) incoming[m]=in->lambda[m+5*c]<0;
    astr::high_open::incoming_rate7(in->primitive+40*c,Real(1.4),Real(.025),in->left,in->lambda+5*c,
        incoming,in->rhs+5*c,Real(2),in->lodi+5*c,c==9,out->rate+5*c);
    astr::high_open::normal_terms6(in->states+40*c,in->velocity+24*c,in->pressure+8*c,
        in->jacobian+8*c,in->metric+24*c,out->normal_d+5*c,out->normal_r+5*c);
}

__global__ void positions(Output *out)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=72) return;
    int row,side;astr::high_open::interface_position(id%18-1,16,id/18+1,row,side);
    out->positions[2*id]=row;out->positions[2*id+1]=side;
}

__global__ void differentiate(const Real *f,Real *df,int n,Real h,const Real *metric)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>n) return;
    Real derivative;
    if(i<3) derivative=astr::high_open::derivative6(f,i);
    else if(i>n-3) {
        Real line[8];for(int j=0;j<8;++j) line[j]=f[n-j];
        derivative=-astr::high_open::derivative6(line,n-i);
    } else derivative=Real(.75)*(f[i+1]-f[i-1])-Real(.15)*(f[i+2]-f[i-2])+(f[i+3]-f[i-3])/Real(60);
    derivative/=h;
    if(metric) derivative/=metric[i];
    df[i]=derivative;
}

__global__ void viscous_flux(const Real *df,const Real *metric,const Real *mu,Real *flux,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) flux[i]=mu[i]*df[i]/metric[i];
}

template<class T> void read(std::ifstream &stream,T *values,size_t count)
{
    if(!stream.read(reinterpret_cast<char *>(values),sizeof(T)*count)) throw std::runtime_error("truncated high-open packet");
}
void check(cudaError_t status,const char *operation)
{
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void synchronized_launch()
{
    check(cudaGetLastError(),"kernel launch");check(cudaDeviceSynchronize(),"kernel synchronization");
}
double compare(const Real *actual,const Real *expected,int count,bool relative=false)
{
    double maximum=0;
    for(int i=0;i<count;++i) {
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i])) throw std::runtime_error("nonfinite high-open value");
        double error=std::abs(double(actual[i])-expected[i]);
        if(relative) error/=std::max(1.,std::abs(double(expected[i])));
        maximum=std::max(maximum,error);
    }
    if(maximum>2e-10) throw std::runtime_error("high-open component error: "+std::to_string(maximum));
    return maximum;
}
}

int main(int argc,char **argv)
{
    try {
        if(argc!=2) throw std::runtime_error("usage: astr_backend_high_open_math original_packet.bin");
        std::ifstream stream(argv[1],std::ios::binary);
        int32_t header[6];read(stream,header,6);
        if(header[0]!=1||header[1]!=sizeof(Real)||header[2]!=10||header[3]!=32||header[4]!=64||header[5]!=128)
            throw std::runtime_error("high-open packet identity/kind/grid mismatch");
        Input input{};Output expected{},actual{};
        read(stream,input.samples,80);read(stream,input.primitive,400);read(stream,input.lambda,50);
        read(stream,input.rhs,50);read(stream,input.lodi,50);read(stream,input.left,25);
        read(stream,input.states,400);read(stream,input.velocity,240);read(stream,input.pressure,80);
        read(stream,input.jacobian,80);read(stream,input.metric,240);
        read(stream,expected.derivative,30);read(stream,expected.flux,60);read(stream,expected.filtered,30);
        read(stream,expected.rate,50);read(stream,expected.normal_d,50);read(stream,expected.normal_r,50);
        read(stream,expected.positions,144);
        std::array<Real,129*6*3> composite;read(stream,composite.data(),composite.size());
        if(stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("unexpected high-open packet tail");
        Input *device_input=nullptr;Output *device_output=nullptr;
        check(cudaMalloc(&device_input,sizeof(Input)),"allocate input");check(cudaMalloc(&device_output,sizeof(Output)),"allocate output");
        check(cudaMemcpy(device_input,&input,sizeof(Input),cudaMemcpyHostToDevice),"upload frozen inputs");
        evaluate<<<1,32>>>(device_input,device_output);synchronized_launch();
        positions<<<1,128>>>(device_output);synchronized_launch();
        check(cudaMemcpy(&actual,device_output,sizeof(Output),cudaMemcpyDeviceToHost),"read high-open results");
        check(cudaFree(device_output),"release output");check(cudaFree(device_input),"release input");
        std::cout<<std::setprecision(17);
        const Real *values[]={actual.derivative,actual.flux,actual.filtered,actual.rate,actual.normal_d,actual.normal_r};
        const Real *references[]={expected.derivative,expected.flux,expected.filtered,expected.rate,expected.normal_d,expected.normal_r};
        const int counts[]={30,60,30,50,50,50};
        const char *names[]={"derivative6","flux6","filter6","incoming7","normal_derivative6","metric_rest6"};
        for(int k=0;k<6;++k) std::cout<<"HIGH_OPEN_"<<names[k]<<"_MAX_ABS "<<compare(values[k],references[k],counts[k])<<'\n';
        for(int i=0;i<144;++i) if(actual.positions[i]!=expected.positions[i]) throw std::runtime_error("boundary interface ownership changed");
        const double free_error=compare(actual.normal_d,actual.normal_r,5);
        for(int row=0;row<3;++row) {
            if(actual.derivative[row+3*8]!=0||actual.filtered[row+3*8]!=1) throw std::runtime_error("constant field changed");
            for(int wind=0;wind<2;++wind) if(actual.flux[row+3*wind+6*8]!=1) throw std::runtime_error("constant numerical flux changed");
        }
        std::cout<<"HIGH_OPEN_FREE_STREAM_MAX_ABS "<<free_error<<'\n';
        double errors[3];
        for(int grid=0;grid<3;++grid) {
            const int n=header[3+grid],count=n+1;const Real h=Real(1)/n;
            const Real *f=composite.data()+129*6*grid,*reference_df=f+129,*metric=f+258,*mu=f+387;
            const Real *reference_result=f+516,*analytic=f+645;
            Real *device=nullptr;check(cudaMalloc(&device,6*count*sizeof(Real)),"allocate composite workspace");
            check(cudaMemcpy(device,f,count*sizeof(Real),cudaMemcpyHostToDevice),"upload composite input");
            check(cudaMemcpy(device+count,metric,count*sizeof(Real),cudaMemcpyHostToDevice),"upload metric");
            check(cudaMemcpy(device+2*count,mu,count*sizeof(Real),cudaMemcpyHostToDevice),"upload viscosity");
            differentiate<<<(count+127)/128,128>>>(device,device+3*count,n,h,nullptr);synchronized_launch();
            viscous_flux<<<(count+127)/128,128>>>(device+3*count,device+count,device+2*count,device+4*count,count);synchronized_launch();
            differentiate<<<(count+127)/128,128>>>(device+4*count,device+5*count,n,h,device+count);synchronized_launch();
            std::vector<Real> derivative(count),result(count);
            check(cudaMemcpy(derivative.data(),device+3*count,count*sizeof(Real),cudaMemcpyDeviceToHost),"read gradient");
            check(cudaMemcpy(result.data(),device+5*count,count*sizeof(Real),cudaMemcpyDeviceToHost),"read composition");
            check(cudaFree(device),"release composite workspace");
            const double gradient_error=compare(derivative.data(),reference_df,count,true);
            const double composite_error=compare(result.data(),reference_result,count,true);
            errors[grid]=0;
            for(int i=0;i<count;++i) if(i<=5||i>=n-5) errors[grid]=std::max(errors[grid],std::abs(double(result[i])-analytic[i]));
            std::cout<<"HIGH_OPEN_COMPOSITE "<<n<<' '<<gradient_error<<' '<<composite_error<<' '<<errors[grid]<<'\n';
        }
        for(int i=0;i<2;++i) {
            const double order=std::log(errors[i]/errors[i+1])/std::log(2.);
            if(!std::isfinite(order)||order<5.5) throw std::runtime_error("high-open composite order below 5.5");
            std::cout<<"HIGH_OPEN_COMPOSITE_ORDER "<<order<<'\n';
        }
        std::cout<<"HIGH_OPEN_CPP_COMPONENT_PASS cases=10 positions=72 grids=32,64,128\n";
    } catch(const std::exception &error) { std::cerr<<error.what()<<'\n';return 1; }
}
