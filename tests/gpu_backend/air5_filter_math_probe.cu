#include <cuda_runtime.h>
#include "chemistry_filter_math.cuh"
#ifdef ASTR_TEST_NR2_CAPACITY
#include "air5_nr2_capacity_math.cuh"
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void evaluate(const double *input,double *output,int *statuses,int n)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double *x=input+256*i;double *out=output+48*i;
    for(int c=0;c<48;++c)out[c]=0.;int status=0;
    using namespace astr::air5;
    switch(static_cast<int>(x[255])) {
    case 0:
        out[0]=filter::lower_energy_admissible(x,x+11,x+16);
        out[1]=filter::pressure_admissible(x);out[2]=filter::lower_admissible(x,x+11,x+16);break;
    case 1:
        out[0]=filter::pressure_represented_admissible(x,x+11);filter::state_magnitude(x,x+11,out+1);break;
    case 2:out[0]=filter::pair_box_safe(x,x+11,x+22,x[33]!=0,x+34,x+39);break;
    case 3:out[0]=filter::feasible_mass(x,x+5,x+10,x[15]);break;
    case 4:filter::high_face(x,x+110,static_cast<int>(x[220]),static_cast<int>(x[221]),x[222]!=0,x[223]!=0,out);break;
    case 5:
        filter::face_bounds(x,x+11,x+22,out,out+5,out[10]);
#ifdef ASTR_TEST_NR2_CAPACITY
        status=nr2_capacity::project(x+27,out,out+5,out[10],out+11);break;
#else
        status=close_species_flux(x+27,out,out+5,out[10],out+11);break;
#endif
    case 6:
        status=filter::thermal_limit(x,x+11,x+22,x+33,x+38,x+43,x+54,x[65]!=0,out,out[11]);
        if(status==0)out[12]=filter::pair_admissible(x,x+11,out,x+33,x+38,x+43,x+54,x[65]!=0);
        break;
    default:status=-1;
    }
    statuses[i]=status;
}
void check(cudaError_t error) {if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}}
int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    FILE *file=std::fopen(argv[1],"rb");if(!file)return 2;int n;
    if(std::fread(&n,4,1,file)!=1)return 2;
    std::vector<double> input(256*n),output(48*n);std::vector<int> statuses(n);
    if(std::fread(input.data(),8,input.size(),file)!=input.size())return 2;std::fclose(file);
    double *x,*out;int *status;
    check(cudaMalloc(&x,input.size()*8));check(cudaMalloc(&out,output.size()*8));check(cudaMalloc(&status,n*4));
    check(cudaMemcpy(x,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    evaluate<<<(n+31)/32,32>>>(x,out,status,n);check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(),out,output.size()*8,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(statuses.data(),status,n*4,cudaMemcpyDeviceToHost));
    check(cudaFree(x));check(cudaFree(out));check(cudaFree(status));
    file=std::fopen(argv[2],"wb");if(!file)return 2;
    bool written=std::fwrite(&n,4,1,file)==1 && std::fwrite(statuses.data(),4,n,file)==static_cast<size_t>(n);
    written=written && std::fwrite(output.data(),8,output.size(),file)==output.size();
    return std::fclose(file)==0 && written?0:2;
}
