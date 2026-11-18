#include <cuda_runtime.h>
#include "chemistry_kinetics_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void evaluate(int n,const double *input,int mode,double *output,int *statuses)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    using namespace astr::air5;
    const double *x=input+53*i;
    double *out=output+132*i;int *st=statuses+11*i;
    for(int j=0;j<132;++j) out[j]=0;
    for(int j=0;j<11;++j) st[j]=-1;
    if(mode==99) {
        for(int s=0;s<5;++s) out[s]=species_vibrational_difference(s,x[11],x[12]);
        double dt[6],dtv[6];
        st[0]=temperature_derivatives(x+5,x[11],dt,dt[5]);
        st[1]=tv_derivatives(x+5,x[12],dtv,dtv[5]);
        st[2]=vt_source<false>(x+5,x[11],x[12],nullptr,nullptr,out[5],nullptr);
        if(st[0]!=0 || st[1]!=0) return;
        st[3]=vt_source<true>(x+5,x[11],x[12],dt,dtv,out[6],out+7);
        for(int v=0;v<6;++v) { out[13+v]=dt[v];out[19+v]=dtv[v]; }
        return;
    }
    st[0]=instantaneous_source<true>(x[0],x+1,x[4],x+5,out,out+6,out+42,mode);
    double progress[12];
    st[1]=instantaneous_source<false>(x[0],x+1,x[4],x+5,out+54,nullptr,progress,mode);
    ActivityData d;
    st[2]=instantaneous_source<false,true>(x[0],x+1,x[4],x+5,out+60,nullptr,progress,mode,&d);
    for(int s=0;s<5;++s) { out[66+s]=d.production[s];out[71+s]=d.loss[s]; }
    for(int j=0;j<2;++j) { out[76+j]=d.energy[j];out[78+j]=d.temperatures[j]; }
    if(st[2]==0) out[89]=activity_index(x[0],x+5,1.e-10,d,out+80);
    int pivots[6]={-1,-1,-1,-1,-1,-1};
    st[3]=lu_factor_6(x+11,out+90,pivots);
    for(int j=0;j<6;++j) st[5+j]=pivots[j]+1;
    if(st[3]==0) st[4]=lu_solve_6(out+90,pivots,x+47,out+126);
}
static void check(cudaError_t status)
{
    if(status!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(1); }
}
int main(int argc,char **argv)
{
    if(argc!=4) return 2;
    FILE *f=std::fopen(argv[1],"rb");if(!f) return 2;
    int n=0;if(std::fread(&n,sizeof(n),1,f)!=1 || n<=0) return 2;
    std::vector<double> input(53*n),output(132*n);std::vector<int> statuses(11*n);
    if(std::fread(input.data(),sizeof(double),input.size(),f)!=input.size()) return 2;
    std::fclose(f);
    double *xd,*od;int *sd;
    check(cudaMalloc(&xd,input.size()*sizeof(double)));check(cudaMalloc(&od,output.size()*sizeof(double)));
    check(cudaMalloc(&sd,statuses.size()*sizeof(int)));
    check(cudaMemcpy(xd,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice));
    evaluate<<<(n+63)/64,64>>>(n,xd,std::atoi(argv[3]),od,sd);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),od,output.size()*sizeof(double),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(statuses.data(),sd,statuses.size()*sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaFree(xd));check(cudaFree(od));check(cudaFree(sd));
    f=std::fopen(argv[2],"wb");if(!f) return 2;
    bool written=std::fwrite(&n,sizeof(n),1,f)==1;
    written=written && std::fwrite(statuses.data(),sizeof(int),statuses.size(),f)==statuses.size();
    written=written && std::fwrite(output.data(),sizeof(double),output.size(),f)==output.size();
    return std::fclose(f)==0 && written?0:2;
}
