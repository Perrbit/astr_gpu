#include <cuda_runtime.h>
#include "chemistry_properties_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>

__global__ void evaluate(int n,const double *input,double *output,int *statuses,bool preserve_all)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    using namespace astr::air5;
    const double *x=input+41*i;
    double *out=output+114*i;
    int *st=statuses+7*i;
    for(int j=0;j<114;++j) out[j]=0;
    double z[5],p;
    for(int s=0;s<5;++s) z[s]=x[0]*x[s+5];
    st[0]=pressure(z,x[4],p);
    st[1]=primitive_to_conservative(x[0],x+1,x[4],x+5,x[10],out);
    st[2]=conservative_to_primitive(out,preserve_all,out[11],out+12,out[15],out+16,out[21],out[22]);
    out[23]=p;
    st[3]=temperature_derivatives(z,x[4],out+24,out[29]);
    st[4]=tv_derivatives(z,x[10],out+30,out[35]);
    st[5]=transport_properties(x[4],x[10],p,x+5,out[36],out[37],out[38],out+39,out+44,out+69);
    st[6]=diffusive_flux(x[0],x+1,x[4],x[10],p,x+5,x+11,x+20,x+23,x+26,
                         out+74,out+83,out+98,out+101,out+104,out+109);
}

static void check(cudaError_t status)
{
    if(status==cudaSuccess) return;
    std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1);
}
int main(int argc,char **argv)
{
    if(argc!=4) return 2;
    FILE *f=std::fopen(argv[1],"rb");
    if(!f) return 2;
    int n=0;
    if(std::fread(&n,sizeof(n),1,f)!=1 || n<=0) return 2;
    std::vector<double> input(41*n),output(114*n);
    std::vector<int> statuses(7*n);
    if(std::fread(input.data(),sizeof(double),input.size(),f)!=input.size()) return 2;
    std::fclose(f);
    double *xd,*od;int *sd;
    check(cudaMalloc(&xd,input.size()*sizeof(double)));
    check(cudaMalloc(&od,output.size()*sizeof(double)));
    check(cudaMalloc(&sd,statuses.size()*sizeof(int)));
    check(cudaMemcpy(xd,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice));
    evaluate<<<(n+127)/128,128>>>(n,xd,od,sd,std::strcmp(argv[3],"all")==0);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),od,output.size()*sizeof(double),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(statuses.data(),sd,statuses.size()*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaFree(xd));check(cudaFree(od));check(cudaFree(sd));
    f=std::fopen(argv[2],"wb");
    if(!f) return 2;
    bool written=std::fwrite(&n,sizeof(n),1,f)==1;
    written=written && std::fwrite(statuses.data(),sizeof(int),statuses.size(),f)==statuses.size();
    written=written && std::fwrite(output.data(),sizeof(double),output.size(),f)==output.size();
    return std::fclose(f)==0 && written ? 0 : 2;
}
