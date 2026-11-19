#include <cuda_runtime.h>
#include "chemistry_characteristic_math.cuh"
#include "air5_nr4_fixed_acoustic.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void nr4_rhs(int n,const double *input,double *output,int *status)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double *x=input+96*i;double *out=output+67*i;
    using namespace astr::air5;
    const bool preserve=x[93]!=0.;double p,sound,gradient[11];
    status[i]=characteristic::pressure_differential(x,preserve,p,sound,gradient);
    if(status[i])return;
    const double v=x[2]/x[0];bool actual[2];int region;
    const bool expected[2]={x[77]!=0.,x[78]!=0.};
    nr4_fixed::classify(x,preserve,actual,region);
    out[66]=(v-sound<0.)!=expected[0] || (v+sound<0.)!=expected[1];
    const double remaining[11]={};
    for(int phase=0;phase<2;++phase) {
        double normal[11];
        if(!characteristic::normal_convection(x,x+55,x+66,x[88],p,gradient,normal,phase!=0)) {status[i]=1;return;}
        for(int mode=0;mode<3;++mode) {
            double *rhs=out+22*mode+11*phase;
            const bool *mask=mode==0?nullptr:mode==1?actual:expected;
            if(!characteristic::transport_rhs(x,x+11,p,sound,gradient,normal,remaining,x[91],rhs,x[92],mask)) {
                status[i]=2;return;
            }
        }
    }
}

void check(cudaError_t error)
{if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}}

int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    FILE *f=std::fopen(argv[1],"rb");int n;
    if(!f || std::fread(&n,4,1,f)!=1 || n<=0)return 2;
    std::vector<double> input(96*n),output(67*n);std::vector<int> status(n);
    if(std::fread(input.data(),8,input.size(),f)!=input.size() || std::fclose(f))return 2;
    double *x,*out;int *st;
    check(cudaMalloc(&x,input.size()*8));check(cudaMalloc(&out,output.size()*8));check(cudaMalloc(&st,n*4));
    check(cudaMemcpy(x,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    nr4_rhs<<<(n+63)/64,64>>>(n,x,out,st);check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),out,output.size()*8,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status.data(),st,n*4,cudaMemcpyDeviceToHost));
    check(cudaFree(x));check(cudaFree(out));check(cudaFree(st));
    f=std::fopen(argv[2],"wb");if(!f)return 2;
    bool ok=std::fwrite(&n,4,1,f)==1 && std::fwrite(status.data(),4,status.size(),f)==status.size() &&
            std::fwrite(output.data(),8,output.size(),f)==output.size();
    return std::fclose(f)==0 && ok?0:2;
}
