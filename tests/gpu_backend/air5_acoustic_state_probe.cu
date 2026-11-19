#include <cuda_runtime.h>
#include "chemistry_characteristic_math.cuh"
#include "chemistry_properties_math.cuh"
#include "air5_nr4_fixed_acoustic.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void observe(int n,const double *q,double *value,int *status)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const double *state=q+12*i;
    double *out=value+10*i;
    double velocity[3];
    status[i]=astr::air5::conservative_to_primitive(state,state[11]!=0.,out[7],velocity,
        out[0],out+2,out[1],out[9]);
    out[8]=velocity[1];
}

__global__ void observe_characteristics(int n,const double *q,double *value,int *status,bool fixed)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    const double *state=q+12*i;
    double *out=value+10*i;
    double pressure,sound,gradient[11];
    for(int j=0;j<10;++j)out[j]=0.;
    status[i]=astr::air5::characteristic::pressure_differential(state,state[11]!=0.,pressure,sound,gradient);
    if(status[i])return;
    const double velocity=state[2]/state[0];
    out[0]=pressure;out[1]=sound;out[2]=velocity-sound;out[3]=velocity+sound;
    if(fixed) {
        bool incoming[2];int region;
        out[6]=astr::air5::nr4_fixed::classify(state,state[11]!=0.,incoming,region);
        out[4]=incoming[0];out[5]=incoming[1];out[7]=region;
    }
}

void check(cudaError_t status)
{
    if(status!=cudaSuccess) {std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status));std::exit(1);}
}

int main(int argc,char **argv)
{
    const bool observe_top=argc==4 && std::strcmp(argv[3],"characteristics")==0;
    const bool fixed=argc==4 && std::strcmp(argv[3],"fixed-classification")==0;
    if(argc!=3 && !observe_top && !fixed)return 2;
    FILE *input=std::fopen(argv[1],"rb");
    int n=0;
    if(!input || std::fread(&n,sizeof(n),1,input)!=1 || n<=0)return 2;
    std::vector<double> q(12*n),value(10*n);
    std::vector<int> status(n);
    if(std::fread(q.data(),sizeof(double),q.size(),input)!=q.size() || std::fclose(input)!=0)return 2;
    double *dq,*dv;int *ds;
    check(cudaMalloc(&dq,q.size()*sizeof(double)));
    check(cudaMalloc(&dv,value.size()*sizeof(double)));
    check(cudaMalloc(&ds,status.size()*sizeof(int)));
    check(cudaMemcpy(dq,q.data(),q.size()*sizeof(double),cudaMemcpyHostToDevice));
    if(observe_top || fixed)observe_characteristics<<<(n+127)/128,128>>>(n,dq,dv,ds,fixed);
    else observe<<<(n+127)/128,128>>>(n,dq,dv,ds);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(value.data(),dv,value.size()*sizeof(double),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status.data(),ds,status.size()*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaFree(ds));check(cudaFree(dv));check(cudaFree(dq));
    FILE *output=std::fopen(argv[2],"wb");if(!output)return 2;
    bool written=std::fwrite(&n,sizeof(n),1,output)==1 &&
        std::fwrite(status.data(),sizeof(int),status.size(),output)==status.size() &&
        std::fwrite(value.data(),sizeof(double),value.size(),output)==value.size();
    return std::fclose(output)==0 && written?0:2;
}
