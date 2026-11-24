#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int nodes=10,mu_nodes=8,faces=7,rows=4,channels=faces+rows;

struct Table {
    double interpolation[faces][mu_nodes],gradient[faces][nodes],divergence[rows][faces];
};

// Test-only evaluation. Viscous flux points into the domain at both walls.
__host__ __device__ void evaluate(const Table &table,const double *q,const double *mu,int intervals,int side,double *out)
{
    for(int f=0;f<faces;++f) {
        double gradient=0.,viscosity=0.;
        for(int j=0;j<mu_nodes;++j) viscosity+=table.interpolation[f][j]*mu[side ? intervals-j : j];
        for(int j=0;j<nodes;++j) gradient+=table.gradient[f][j]*q[side ? intervals-j : j];
        out[f]=viscosity*gradient;
    }
    for(int j=0;j<rows;++j) {
        double value=0.;
        for(int f=0;f<faces;++f) value+=table.divergence[j][f]*out[f];
        out[faces+j]=value;
    }
}

__global__ void probe(Table table,const double *q,const double *mu,int intervals,int count,double *out)
{
    const int task=blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=2*count) return;
    evaluate(table,q+(task/2)*(intervals+1),mu,intervals,task%2,out+task*channels);
}

void check(cudaError_t status)
{
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

int main(int argc,char **argv)
{
    if(argc!=3) return 2;
    try {
        int dims[2];
        Table table;
        static_assert(sizeof(Table)==(faces*mu_nodes+faces*nodes+rows*faces)*sizeof(double),"packed probe table");
        std::ifstream input(argv[1],std::ios::binary);
        if(!input.read(reinterpret_cast<char *>(dims),sizeof(dims)) ||
           !input.read(reinterpret_cast<char *>(&table),sizeof(table)) || dims[0]<nodes-1 || dims[1]<=0)
            throw std::runtime_error("invalid ME6 wall10 fixture header");
        const int intervals=dims[0],count=dims[1];
        std::vector<double> mu(intervals+1),q(count*(intervals+1)),cpu(2*count*channels),gpu(2*count*channels);
        if(!input.read(reinterpret_cast<char *>(mu.data()),mu.size()*sizeof(double)) ||
           !input.read(reinterpret_cast<char *>(q.data()),q.size()*sizeof(double)))
            throw std::runtime_error("truncated ME6 wall10 fixture");
        for(int i=0;i<count;++i) for(int side=0;side<2;++side)
            evaluate(table,q.data()+i*(intervals+1),mu.data(),intervals,side,cpu.data()+(2*i+side)*channels);
        double *q_d=nullptr,*mu_d=nullptr,*out_d=nullptr;
        const auto qbytes=q.size()*sizeof(double),mbytes=mu.size()*sizeof(double),rbytes=gpu.size()*sizeof(double);
        check(cudaMalloc(&q_d,qbytes));
        check(cudaMalloc(&mu_d,mbytes));
        check(cudaMalloc(&out_d,rbytes));
        check(cudaMemcpy(q_d,q.data(),qbytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(mu_d,mu.data(),mbytes,cudaMemcpyHostToDevice));
        probe<<<(2*count+127)/128,128>>>(table,q_d,mu_d,intervals,count,out_d);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(gpu.data(),out_d,rbytes,cudaMemcpyDeviceToHost));
        check(cudaFree(out_d));
        check(cudaFree(mu_d));
        check(cudaFree(q_d));
        std::ofstream output(argv[2],std::ios::binary);
        output.write(reinterpret_cast<const char *>(cpu.data()),rbytes);
        output.write(reinterpret_cast<const char *>(gpu.data()),rbytes);
        return output ? 0 : 1;
    } catch(const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
