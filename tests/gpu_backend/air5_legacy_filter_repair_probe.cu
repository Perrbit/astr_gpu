#include "device_state.h"
#include "chemistry_coupling.h"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void check(int status)
{
    if(status) {std::fprintf(stderr,"legacy repair test status %d\n",status);std::exit(1);}
}

int main(int argc,char **argv)
{
    if(argc!=4)return 2;
    FILE *data=std::fopen(argv[1],"rb"),*floor_input=std::fopen(argv[2],"rb"),*out=std::fopen(argv[3],"wb");
    int rows;double floor[5];
    if(!data || !floor_input || !out || std::fread(&rows,4,1,data)!=1 ||
       std::fread(floor,8,5,floor_input)!=5 || std::fclose(floor_input))return 2;
    check(astr_device_bind(0));
    void *handle[2]{};
    for(int comp=0;comp<2;++comp) {
        check(astr_device_state_create(0,0,0,0,1,comp,&handle[comp]));
        check(astr_device_air5_prepare(handle[comp]));
        if(comp) {
            const AstrAir5ChemistryOptions options{1,0,0,1,1,32,0,1e-9,1e-13,0.};
            check(astr_device_air5_chemistry_prepare(handle[comp],&options));
        }
    }
    AstrDeviceView v{};check(astr_device_state_view(handle[0],ASTR_DEVICE_Q,&v));
    const size_t center=v.stride[0]+v.stride[1]+v.stride[2];
    std::vector<double> q(v.stride[4]*v.extent[4],-777.),got(q.size());
    for(int row=0;row<rows;++row) {
        int comp;double pre[11],raw[11],carry[11],rhs[11],saved[11],after[11];
        if(std::fread(&comp,4,1,data)!=1 || comp<0 || comp>1 || std::fread(pre,8,11,data)!=11 ||
           std::fread(raw,8,11,data)!=11 || std::fread(carry,8,11,data)!=11)return 2;
        auto h=handle[comp];
        for(int c=0;c<11;++c) {rhs[c]=987.+c;q[center+c*v.stride[3]]=pre[c];}
        check(astr_device_state_upload(h,ASTR_DEVICE_Q,q.data(),q.size()*8));
        check(astr_device_state_upload(h,ASTR_DEVICE_RHS,rhs,sizeof(rhs)));
        check(astr_device_air5_legacy_filter_save(h));
        check(astr_device_state_download(h,ASTR_DEVICE_RHS,saved,sizeof(saved)));
        check(astr_device_state_download(h,ASTR_DEVICE_Q,got.data(),got.size()*8));
        if(std::memcmp(q.data(),got.data(),q.size()*8))return 3;
        for(int c=0;c<11;++c) {
            if(saved[c]!=(c==0 || (c>=5 && c<10)?pre[c]:rhs[c]))return 3;
            q[center+c*v.stride[3]]=raw[c];
        }
        check(astr_device_state_upload(h,ASTR_DEVICE_Q,q.data(),q.size()*8));
        if(comp)check(astr_device_state_upload(h,ASTR_DEVICE_AIR5_CARRY,carry,sizeof(carry)));
        check(astr_device_air5_legacy_filter_repair(h,floor));
        int physics;check(astr_device_air5_check(h,&physics));
        check(astr_device_state_download(h,ASTR_DEVICE_Q,got.data(),got.size()*8));
        check(astr_device_state_download(h,ASTR_DEVICE_RHS,rhs,sizeof(rhs)));
        if(std::memcmp(rhs,saved,sizeof(rhs)))return 3;
        for(int c=0;c<11;++c) {
            const size_t id=center+c*v.stride[3];after[c]=got[id];
            if(!comp && (c<5 || c==10) && std::memcmp(&got[id],&raw[c],8))return 3;
            got[id]=raw[c];
        }
        if(std::memcmp(q.data(),got.data(),q.size()*8))return 3;
        if(comp)check(astr_device_state_download(h,ASTR_DEVICE_AIR5_CARRY,carry,sizeof(carry)));
        if(std::fwrite(&physics,4,1,out)!=1 || std::fwrite(after,8,11,out)!=11 || std::fwrite(carry,8,11,out)!=11)return 2;
    }
    for(auto &h:handle)check(astr_device_state_destroy(&h));
    if(std::fclose(data) || std::fclose(out))return 2;
}
