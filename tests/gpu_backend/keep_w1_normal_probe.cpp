#include "keep_w1_tensor_math.h"
#include <array>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using astr::w1::NormalResult;
using astr::w1::NormalTable;
using astr::w1::TensorFlux;
using astr::w1::TensorGrid;
using astr::w1::TensorResult;
using astr::w1::TensorState;
using astr::w1::TensorTable;

extern "C" int astr_w1_normal_probe_gpu(const NormalTable *,const double *,const double *,const double *,
    NormalResult *,int,int,double,double,double);
extern "C" int astr_w1_tensor_probe_gpu(const TensorTable *,TensorGrid,TensorState,TensorFlux *,TensorResult *);

static void check(cudaError_t status)
{
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<typename T> static void read(std::istream &in,T *data,std::size_t count)
{
    if(!in.read(reinterpret_cast<char *>(data),sizeof(T)*count)) throw std::runtime_error("truncated W1 fixture");
}

template<typename T> static T *upload(const T *data,std::size_t count)
{
    void *device=nullptr;
    check(cudaMalloc(&device,sizeof(T)*count));
    check(cudaMemcpy(device,data,sizeof(T)*count,cudaMemcpyHostToDevice));
    return static_cast<T *>(device);
}

static void read_table(std::istream &in,NormalTable &table)
{
    read(in,table.begin,astr::w1::normal_rows+1);
    read(in,table.value_index,astr::w1::normal_terms);
    read(in,table.coefficient_index,astr::w1::normal_terms);
    read(in,table.term,astr::w1::normal_terms);
    read(in,table.h0,6);
    read(in,&table.borrowing,1);
}

static int tensor_probe(std::istream &in,const char *path)
{
    TensorTable table;
    TensorGrid grid{};
    read_table(in,table.normal);
    for(auto &row:table.dleft) read(in,row,9);
    read(in,grid.n,3);read(in,grid.inv_h,3);
    const int count=grid.n[0]*grid.n[1]*grid.n[2];
    std::array<std::vector<double>,7> fields;
    std::array<double *,7> on_gpu;
    for(int m=0;m<7;++m) {
        fields[m].resize(count);
        read(in,fields[m].data(),count);
        on_gpu[m]=upload(fields[m].data(),count);
    }
    const TensorState state{{fields[0].data(),fields[1].data(),fields[2].data(),fields[3].data()},
                            fields[4].data(),fields[5].data(),fields[6].data()};
    const TensorState state_d{{on_gpu[0],on_gpu[1],on_gpu[2],on_gpu[3]},on_gpu[4],on_gpu[5],on_gpu[6]};
    std::vector<TensorFlux> flux(count);
    std::vector<TensorResult> host(count),device(count);
    auto visit=[&](auto fill) {
        for(int k=0;k<grid.n[2];++k) for(int j=0;j<grid.n[1];++j) for(int i=0;i<grid.n[0];++i)
            fill(astr::w1::tensor_offset(grid,i,j,k),i,j,k);
    };
    visit([&](int n,int i,int j,int k){flux[n]=astr::w1::tensor_flux(table,grid,state,i,j,k);});
    visit([&](int n,int i,int j,int k){host[n]=astr::w1::tensor_rhs(table,grid,state,astr::w1::PackedTensorFlux{flux.data()},i,j,k);});

    const auto table_d=upload(&table,1);
    TensorFlux *flux_d=nullptr;
    TensorResult *out_d=nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&flux_d),sizeof(TensorFlux)*count));
    check(cudaMalloc(reinterpret_cast<void **>(&out_d),sizeof(TensorResult)*count));
    check(static_cast<cudaError_t>(astr_w1_tensor_probe_gpu(table_d,grid,state_d,flux_d,out_d)));
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(device.data(),out_d,sizeof(TensorResult)*count,cudaMemcpyDeviceToHost));
    check(cudaFree(out_d));check(cudaFree(flux_d));check(cudaFree(table_d));
    for(auto ptr:on_gpu) check(cudaFree(ptr));
    static_assert(sizeof(TensorResult)==8*sizeof(double),"packed W1 tensor result");
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char *>(host.data()),sizeof(TensorResult)*count);
    out.write(reinterpret_cast<const char *>(device.data()),sizeof(TensorResult)*count);
    return out ? 0 : 1;
}

int main(int argc,char **argv)
{
    if(argc!=3 && (argc!=4 || std::string(argv[3])!="tensor")) return 2;
    try {
        std::ifstream in(argv[1],std::ios::binary);
        if(argc==4) return tensor_probe(in,argv[2]);
        NormalTable table;
        read_table(in,table);
        int dims[2];double parameters[3];
        read(in,dims,2);read(in,parameters,3);
        const int nodes=dims[0],lines=dims[1],count=nodes*lines;
        std::vector<double> f(count),b(count),jac(count);
        read(in,f.data(),count);read(in,b.data(),count);read(in,jac.data(),count);
        std::vector<NormalResult> host(count),device(count);
        for(int j=0;j<nodes;++j) for(int lane=0;lane<lines;++lane)
            host[j*lines+lane]=astr::w1::normal_point(table,f.data()+lane,b.data()+lane,jac.data()+lane,lines,
                nodes-1,j,parameters[0],parameters[1],parameters[2]);

        const auto table_d=upload(&table,1);
        const auto f_d=upload(f.data(),count),b_d=upload(b.data(),count), jac_d=upload(jac.data(),count);
        NormalResult *out_d=nullptr;
        check(cudaMalloc(reinterpret_cast<void **>(&out_d),sizeof(NormalResult)*count));
        check(static_cast<cudaError_t>(astr_w1_normal_probe_gpu(table_d,f_d,b_d,jac_d,out_d,nodes,lines,
                                                               parameters[0],parameters[1],parameters[2])));
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(device.data(),out_d,sizeof(NormalResult)*count,cudaMemcpyDeviceToHost));
        check(cudaFree(out_d));check(cudaFree(jac_d));check(cudaFree(b_d));check(cudaFree(f_d));check(cudaFree(table_d));
        static_assert(sizeof(NormalResult)==3*sizeof(double),"packed W1 result");
        std::ofstream out(argv[2],std::ios::binary);
        out.write(reinterpret_cast<const char *>(host.data()),sizeof(NormalResult)*count);
        out.write(reinterpret_cast<const char *>(device.data()),sizeof(NormalResult)*count);
        return out ? 0 : 1;
    } catch(const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
