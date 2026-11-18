#include <cuda_runtime.h>
#include "chemistry_flux_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void evaluate(const double *input,double *output,int *statuses,int n,bool preserve_all)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double *x=input+80*i;double *out=output+17*i;
    for(int c=0;c<17;++c)out[c]=0;
    using namespace astr::air5;
    int status=0;
    switch(static_cast<int>(x[79])) {
    case 0:status=close_species_flux(x,x+5,x+10,x[15],out);break;
    case 1:status=close_species_flux_moment(x,x+5,x+10,x[15],x+16,x[21],out);break;
    case 2:out[0]=interior_ratio(x[0]);out[1]=transport_species_ratio(x[1],x[2],x[3]);break;
    case 3: {
        int mask,species;
        out[0]=admissible_face_ratio(x,x+11,x[22]!=0,mask,species,x[23]!=0,x+70);
        out[1]=mask;out[2]=species;out[3]=translational_margin(x);out[4]=vibrational_excess(x[10],x+5,x+70);
        double trial[11];for(int c=0;c<11;++c)trial[c]=x[c]+6.*x[11+c];
        out[5]=x[23];out[6]=state_admissible(trial,x+70);out[7]=face_state_admissible(trial,x+70,x[23]!=0);
        for(int s=0;s<5;++s)out[8+s]=trial[5+s];
        break;
    }
    case 4:status=symmetric_face_trial(x,x+11,x+22,x+33,x+44,x+55,x[66],preserve_all,
        x[67]!=0,x[68]!=0,x+70,out,out[11]);break;
    default:status=-1;
    }
    statuses[i]=status;
}
void check(cudaError_t error) {if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}}
int main(int argc,char **argv)
{
    if(argc!=4)return 2;
    FILE *file=std::fopen(argv[1],"rb");if(!file)return 2;int n;
    if(std::fread(&n,4,1,file)!=1)return 2;std::vector<double> input(80*n),output(17*n);std::vector<int> status(n);
    if(std::fread(input.data(),8,input.size(),file)!=input.size())return 2;std::fclose(file);
    double *x,*out;int *st;
    check(cudaMalloc(&x,input.size()*8));check(cudaMalloc(&out,output.size()*8));check(cudaMalloc(&st,n*4));
    check(cudaMemcpy(x,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    evaluate<<<(n+31)/32,32>>>(x,out,st,n,std::atoi(argv[3])!=0);check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),out,output.size()*8,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status.data(),st,n*4,cudaMemcpyDeviceToHost));check(cudaFree(x));check(cudaFree(out));check(cudaFree(st));
    file=std::fopen(argv[2],"wb");if(!file)return 2;
    bool written=std::fwrite(&n,4,1,file)==1 && std::fwrite(status.data(),4,n,file)==static_cast<size_t>(n);
    written=written && std::fwrite(output.data(),8,output.size(),file)==output.size();
    return std::fclose(file)==0 && written?0:2;
}
