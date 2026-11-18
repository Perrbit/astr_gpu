#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "astr_precision.h"
#include "mp_positivity_math.cuh"

namespace {
using Real=astr::numerical_real;
void require(bool condition,const char *message)
{
    if(!condition) throw std::runtime_error(message);
}
void check(cudaError_t status)
{
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void evaluate(int n,const Real *base,const Real *correction,int *valid,Real *theta,int *limited_valid)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const Real *q=base+5*i,*delta=correction+5*i;
    valid[i]=astr::mp_admissible(q);
    theta[i]=astr::mp_face_budget(q,delta);
    Real trial[5];
    for(int c=0;c<5;++c) trial[c]=q[c]+theta[i]*delta[c];
    limited_valid[i]=astr::mp_admissible(trial);
}
}

int main(int argc,char **argv)
{
    try {
        require(argc==3,"expected original input and original result");
        std::ifstream input(argv[1],std::ios::binary),reference(argv[2],std::ios::binary);
        int32_t shape[2]{},expected_shape[2]{};
        input.read(reinterpret_cast<char *>(shape),sizeof(shape));
        reference.read(reinterpret_cast<char *>(expected_shape),sizeof(expected_shape));
        require(input.good() && reference.good() && shape[0]>0 && shape[1]==sizeof(Real) &&
            shape[0]==expected_shape[0] && shape[1]==expected_shape[1],"original layout/precision");
        const int n=shape[0];
        std::vector<Real> base(5*n),delta(5*n),expected(n),theta(n);
        std::vector<int> expected_valid(n),original_limited_valid(n),valid(n),limited_valid(n);
        input.read(reinterpret_cast<char *>(base.data()),base.size()*sizeof(Real));
        input.read(reinterpret_cast<char *>(delta.data()),delta.size()*sizeof(Real));
        reference.read(reinterpret_cast<char *>(expected_valid.data()),n*sizeof(int32_t));
        reference.read(reinterpret_cast<char *>(expected.data()),n*sizeof(Real));
        reference.read(reinterpret_cast<char *>(original_limited_valid.data()),n*sizeof(int32_t));
        require(input.good() && reference.good() && input.peek()==std::char_traits<char>::eof() &&
            reference.peek()==std::char_traits<char>::eof(),"original payload length");
        Real *q=nullptr,*d=nullptr,*t=nullptr;
        int *v=nullptr,*lv=nullptr;
        check(cudaSetDevice(0));
        check(cudaMalloc(&q,base.size()*sizeof(Real)));
        check(cudaMalloc(&d,delta.size()*sizeof(Real)));
        check(cudaMalloc(&t,n*sizeof(Real)));
        check(cudaMalloc(&v,n*sizeof(int)));
        check(cudaMalloc(&lv,n*sizeof(int)));
        check(cudaMemcpy(q,base.data(),base.size()*sizeof(Real),cudaMemcpyHostToDevice));
        check(cudaMemcpy(d,delta.data(),delta.size()*sizeof(Real),cudaMemcpyHostToDevice));
        evaluate<<<(n+127)/128,128>>>(n,q,d,v,t,lv);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        check(cudaMemcpy(theta.data(),t,n*sizeof(Real),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(valid.data(),v,n*sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(limited_valid.data(),lv,n*sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaFree(q));check(cudaFree(d));check(cudaFree(t));check(cudaFree(v));check(cudaFree(lv));
        double error=0;
        int limited=0,inadmissible=0,original_inadmissible=0,classification_mismatches=0;
        for(int i=0;i<n;++i) {
            if(valid[i]!=expected_valid[i]) {
                std::cerr<<"MP_ADMISSIBILITY_DIAGNOSTIC case="<<i<<" real_bytes="<<sizeof(Real)
                    <<" actual="<<valid[i]<<" original="<<expected_valid[i]<<'\n';
                ++classification_mismatches;
            }
            require(std::isfinite(theta[i]) && theta[i]>=0 && theta[i]<=1,"nonfinite/out-of-range face budget");
            error=std::max(error,std::abs(double(theta[i])-double(expected[i])));
            if(valid[i] && limited_valid[i]!=1) {
                std::cerr<<std::setprecision(17)<<"MP_BUDGET_DIAGNOSTIC case="<<i
                    <<" theta="<<theta[i]<<" original_theta="<<expected[i]
                    <<" original_limited_admissible="<<original_limited_valid[i]<<" base=";
                for(int c=0;c<5;++c) std::cerr<<base[5*i+c]<<' ';
                std::cerr<<"correction=";
                for(int c=0;c<5;++c) std::cerr<<delta[5*i+c]<<' ';
                std::cerr<<'\n';
                ++inadmissible;
            }
            original_inadmissible+=expected_valid[i] && !original_limited_valid[i];
            limited+=valid[i] && theta[i]<1;
        }
        if(sizeof(Real)==8) require(error<=2e-10,"face budget differs from original by more than 2e-10");
        std::cout<<std::setprecision(17)<<"CPP_MP_BUDGET_COMPARISON cases="<<n
            <<" max_abs_theta="<<error<<" invalid_returned="<<inadmissible
            <<" original_invalid_returned="<<original_inadmissible
            <<" classification_mismatches="<<classification_mismatches<<'\n';
        require(classification_mismatches==0,"admissibility differs from original device function");
        require(inadmissible==0,"returned face budget is not admissible");
        std::cout<<std::setprecision(17)<<"CPP_MP_BUDGET_PASS cases="<<n<<" limited="<<limited<<" real_bytes="<<sizeof(Real)
            <<" max_abs_theta="<<error<<'\n';
    } catch(const std::exception &error) {
        std::cerr<<"CPP_MP_BUDGET_FAILURE "<<error.what()<<'\n';
        return 1;
    }
}
