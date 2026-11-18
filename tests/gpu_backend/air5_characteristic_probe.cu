#include "chemistry_characteristic_math.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(int status) {if(status) {std::fprintf(stderr,"CUDA status%d\n",status);std::exit(1);}}
__global__ void evaluate_all(const double *input,double *output,int *flags,int n)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double *x=input+96*i;double *out=output+66*i;int *status=flags+5*i;
    for(int c=0;c<66;++c)out[c]=0.;for(int c=0;c<5;++c)status[c]=0;
    using namespace astr::air5::characteristic;
    status[0]=normal_convection(x,x+55,x+66,x[88],x[89],x+22,out,false);
    status[1]=normal_convection(x,x+55,x+66,x[88],x[89],x+22,out+11,true);
    status[2]=transport_rhs(x,x+11,x[89],x[90],x+22,x+33,x+44,x[91],out+22,x[92]);
    y_flux_differential(x,x[89],x+22,x+77,out+33);
    if(status[0])status[3]=transport_rhs(x,x+11,x[89],x[90],x+22,out,x+44,x[91],out+44,x[92]);
    if(status[1])status[4]=transport_rhs(x,x+11,x[89],x[90],x+22,out+11,x+44,x[91],out+55,x[92]);
}
int main(int argc,char **argv)
{
    if(argc!=3)return 2;FILE *f=std::fopen(argv[1],"rb");if(!f)return 2;int n;
    if(std::fread(&n,4,1,f)!=1 || n<=0)return 2;
    std::vector<double> x(96*n),out(66*n);std::vector<int> flags(5*n);
    if(std::fread(x.data(),8,x.size(),f)!=x.size())return 2;if(std::fclose(f)!=0)return 2;
    double *xd,*od;int *sd;check(cudaMalloc(&xd,x.size()*8));check(cudaMalloc(&od,out.size()*8));check(cudaMalloc(&sd,flags.size()*4));
    check(cudaMemcpy(xd,x.data(),x.size()*8,cudaMemcpyHostToDevice));evaluate_all<<<(n+127)/128,128>>>(xd,od,sd,n);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(out.data(),od,out.size()*8,cudaMemcpyDeviceToHost));check(cudaMemcpy(flags.data(),sd,flags.size()*4,cudaMemcpyDeviceToHost));
    check(cudaFree(xd));check(cudaFree(od));check(cudaFree(sd));
    f=std::fopen(argv[2],"wb");if(!f)return 2;
    const bool written=std::fwrite(&n,4,1,f)==1 && std::fwrite(flags.data(),4,flags.size(),f)==flags.size() &&
        std::fwrite(out.data(),8,out.size(),f)==out.size();if(std::fclose(f)!=0 || !written)return 2;return 0;
}
