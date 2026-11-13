#define OMPI_SKIP_MPICXX 1
#include <hdf5.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs=std::filesystem;
void require(bool condition,const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Layout { int global[3],origin[3],upper[3],halo,topology[3]; };

std::map<int,Layout> layouts(const fs::path &directory)
{
    std::ifstream input(directory/"run.log");
    require(input.good(),"open solver layout log");
    std::map<int,Layout> result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.rfind("ASTR_GPU_CPP_LAYOUT ",0)!=0) continue;
        Layout l{};int rank;
        const int read=std::sscanf(line.c_str(),
            "ASTR_GPU_CPP_LAYOUT rank=%d global=%d%d%d origin=%d%d%d upper=%d%d%d halo=%d topology=%d%d%d",
            &rank,&l.global[0],&l.global[1],&l.global[2],&l.origin[0],&l.origin[1],&l.origin[2],
            &l.upper[0],&l.upper[1],&l.upper[2],&l.halo,&l.topology[0],&l.topology[1],&l.topology[2]);
        require(read==14 && result.emplace(rank,l).second,"solver layout record");
    }
    require(!result.empty(),"missing solver layouts");
    const auto &first=result.at(0);
    const int count=first.topology[0]*first.topology[1]*first.topology[2];
    require(static_cast<int>(result.size())==count,"missing rank layouts");
    for (int rank=0;rank<count;++rank) require(result.count(rank)==1,"rank layout identity");
    return result;
}

void field(const fs::path &reference,const fs::path &directory)
{
    const auto layout=layouts(directory);
    const hid_t file=H5Fopen(reference.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    require(file>=0,"open frozen reference");
    std::array<std::vector<double>,5> expected;
    const auto &global=layout.at(0).global;
    for (int c=0;c<5;++c) {
        const std::string name="q000"+std::to_string(c+1);
        const hid_t dataset=H5Dopen2(file,name.c_str(),H5P_DEFAULT),space=H5Dget_space(dataset);
        hsize_t dims[3]{};
        require(space>=0 && H5Sget_simple_extent_ndims(space)==3,"reference dimensions");
        require(H5Sget_simple_extent_dims(space,dims,nullptr)==3,"read reference dimensions");
        for (int a=0;a<3;++a) require(dims[2-a]==static_cast<hsize_t>(global[a]+1),"global reference shape");
        expected[c].resize(dims[0]*dims[1]*dims[2]);
        require(H5Dread(dataset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,expected[c].data())>=0,"read reference");
        H5Sclose(space);H5Dclose(dataset);
    }
    require(H5Fclose(file)>=0,"close reference");
    for (const auto &[rank,l]:layout) {
        char filename[128];
        std::snprintf(filename,sizeof(filename),"rhs.cpp_final.step00000004.rk01.rank%08d.bin",rank);
        std::ifstream input(directory/"outdat"/filename,std::ios::binary);
        std::array<int32_t,5> header{};
        input.read(reinterpret_cast<char *>(header.data()),sizeof(header));
        require(input.good() && header[3]==l.halo && header[4]==5,"snapshot header");
        size_t n[3];
        for (int a=0;a<3;++a) {
            require(header[a]==l.upper[a] && l.origin[a]>=0 && l.origin[a]+l.upper[a]<=global[a],"snapshot layout");
            n[a]=l.upper[a]+2*l.halo+1;
        }
        std::vector<double> actual(n[0]*n[1]*n[2]*5);
        input.read(reinterpret_cast<char *>(actual.data()),actual.size()*sizeof(double));
        require(input.good() && input.peek()==std::char_traits<char>::eof(),"snapshot payload length");
        for (int c=0;c<5;++c) {
            double error=0,scale=1;
            for (int k=0;k<=l.upper[2];++k) for (int j=0;j<=l.upper[1];++j) for (int i=0;i<=l.upper[0];++i) {
                const double value=actual[((c*n[2]+k+l.halo)*n[1]+j+l.halo)*n[0]+i+l.halo];
                const double ref=expected[c][((static_cast<size_t>(k+l.origin[2])*(global[1]+1)+j+l.origin[1])
                                              *(global[0]+1)+i+l.origin[0])];
                require(std::isfinite(value) && std::isfinite(ref),"nonfinite physical field");
                error=std::max(error,std::abs(value-ref));scale=std::max(scale,std::abs(ref));
            }
            std::cout<<"FIELD rank="<<rank<<" component="<<c+1<<" scaled_error="<<error/scale<<'\n';
            require(error/scale<=2e-10,"scaled field difference exceeds 2e-10");
        }
    }
}

void exact(const fs::path &a,const fs::path &b)
{
    const hid_t fa=H5Fopen(a.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),fb=H5Fopen(b.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    require(fa>=0 && fb>=0,"open checkpoint pair");
    H5G_info_t ga{},gb{};
    require(H5Gget_info(fa,&ga)>=0 && H5Gget_info(fb,&gb)>=0 && ga.nlinks==gb.nlinks,"dataset count");
    for (hsize_t item=0;item<ga.nlinks;++item) {
        const ssize_t length=H5Lget_name_by_idx(fa,".",H5_INDEX_NAME,H5_ITER_INC,item,nullptr,0,H5P_DEFAULT);
        require(length>=0,"dataset name length");
        std::vector<char> name(length+1);
        require(H5Lget_name_by_idx(fa,".",H5_INDEX_NAME,H5_ITER_INC,item,name.data(),name.size(),H5P_DEFAULT)==length,
                "dataset name");
        const hid_t da=H5Dopen2(fa,name.data(),H5P_DEFAULT),db=H5Dopen2(fb,name.data(),H5P_DEFAULT);
        require(da>=0 && db>=0,"checkpoint dataset identity");
        const hid_t sa=H5Dget_space(da),sb=H5Dget_space(db),ta=H5Dget_type(da),tb=H5Dget_type(db);
        require(H5Sextent_equal(sa,sb)>0 && H5Tequal(ta,tb)>0,"checkpoint dataset shape/type");
        const hssize_t points=H5Sget_simple_extent_npoints(sa);
        require(points>=0,"checkpoint dataspace");
        std::vector<unsigned char> left(points*H5Tget_size(ta)),right(left.size());
        if (points>0) {
            require(H5Dread(da,ta,H5S_ALL,H5S_ALL,H5P_DEFAULT,left.data())>=0 &&
                    H5Dread(db,tb,H5S_ALL,H5S_ALL,H5P_DEFAULT,right.data())>=0,"read checkpoint data");
            require(left==right,std::string("checkpoint words differ: ")+name.data());
        }
        H5Sclose(sa);H5Sclose(sb);H5Tclose(ta);H5Tclose(tb);H5Dclose(da);H5Dclose(db);
    }
    require(H5Fclose(fa)>=0 && H5Fclose(fb)>=0,"close checkpoint pair");
    std::cout<<"EXACT datasets="<<ga.nlinks<<" all words equal\n";
}

void physical(const fs::path &a,const fs::path &b)
{
    const hid_t fa=H5Fopen(a.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),fb=H5Fopen(b.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT);
    require(fa>=0 && fb>=0,"open physical checkpoint pair");
    for (int c=1;c<=5;++c) {
        const std::string name="q000"+std::to_string(c);
        const hid_t da=H5Dopen2(fa,name.c_str(),H5P_DEFAULT),db=H5Dopen2(fb,name.c_str(),H5P_DEFAULT);
        require(da>=0 && db>=0,"physical checkpoint component");
        const hid_t sa=H5Dget_space(da),sb=H5Dget_space(db);
        require(H5Sextent_equal(sa,sb)>0,"physical checkpoint shape");
        const hssize_t count=H5Sget_simple_extent_npoints(sa);
        require(count>0,"physical checkpoint points");
        std::vector<double> left(count),right(count);
        require(H5Dread(da,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,left.data())>=0 &&
                H5Dread(db,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,right.data())>=0,"read physical checkpoints");
        double error=0,scale=1;
        for (size_t i=0;i<left.size();++i) {
            require(std::isfinite(left[i]) && std::isfinite(right[i]),"nonfinite physical checkpoint");
            error=std::max(error,std::abs(left[i]-right[i]));scale=std::max(scale,std::abs(left[i]));
        }
        std::cout<<"PHYSICAL component="<<c<<" scaled_error="<<error/scale<<'\n';
        require(error/scale<=2e-10,"physical checkpoint difference exceeds 2e-10");
        H5Sclose(sa);H5Sclose(sb);H5Dclose(da);H5Dclose(db);
    }
    require(H5Fclose(fa)>=0 && H5Fclose(fb)>=0,"close physical checkpoint pair");
}
}

int main(int argc,char **argv)
{
    try {
        require(argc==4,"usage: astr_backend_compare field REFERENCE_H5 RUN_DIRECTORY | exact|physical STATE_A STATE_B");
        const std::string mode=argv[1];
        if (mode=="field") field(argv[2],argv[3]);
        else if (mode=="exact") exact(argv[2],argv[3]);
        else if (mode=="physical") physical(argv[2],argv[3]);
        else throw std::runtime_error("mode must be field, exact or physical");
        return 0;
    } catch (const std::exception &error) {
        std::cerr<<"COMPARE FAILED: "<<error.what()<<'\n';return 1;
    }
}
