#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include "device_state_internal.cuh"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include "halo_transport.h"
#include "chemistry_face_math.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <string>
#include <vector>

namespace {
void check(int code) {if(code){std::fprintf(stderr,"operation status %d\n",code);std::exit(1);}}
size_t size(AstrDeviceView v) {return v.stride[4]*v.extent[4];}
size_t offset(AstrDeviceView v,int i,int j,int k,int c=1,int d=1)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3]+(d-1)*v.stride[4];}
void upload(void *h,int f,const std::vector<double> &v) {check(astr_device_state_upload(h,f,v.data(),v.size()*8));}
std::vector<double> download(void *h,int f,AstrDeviceView v)
{std::vector<double> data(size(v));check(astr_device_state_download(h,f,data.data(),data.size()*8));return data;}
int physics(void *h) {int status;check(astr_device_air5_check(h,&status));return status;}
bool write(FILE *f,const void *data,size_t width,size_t count) {return std::fwrite(data,width,count,f)==count;}
struct Capture {int p[3],upper[3],types[3],axis,top,capturing;double cdt,floor[5];};
__global__ void capture_point(astr::air5::FaceFields fields,AstrDeviceView base,Capture data,const int8_t *mask,double *out)
{
    using namespace astr::air5;for(int c=0;c<80;++c)out[c]=0;
    const int a=data.axis;int right[3]={data.p[0],data.p[1],data.p[2]};++right[a];
    ConvectiveLine line{fields,{data.p[0],data.p[1],data.p[2]},a};int status=0;
    const bool shock=data.capturing && air5_shock_face(mask,data.upper,data.p,a,data.types[a]);
    convective_face(line,data.p[a],data.upper[a],data.types[a],shock,false,true,out,status);
    if(data.top && a==0 && data.p[0]==data.upper[0] && (data.types[0]==2 || data.types[0]==4))line.load(data.p[0],out+11);
    else convective_face(line,data.p[a],data.upper[a],data.types[a],false,true,true,out+11,status);
    for(int c=0;c<11;++c) {
        out[22+c]=face_value(base,data.p,c+1);out[33+c]=face_value(base,right,c+1);
        out[44+c]=face_value(fields.q,data.p,c+1);out[55+c]=face_value(fields.q,right,c+1);
    }
    out[66]=data.cdt;out[67]=!((data.types[a]==1 || data.types[a]==4) && data.p[a]==0);
    out[68]=!(right[a]==data.upper[a] && (data.types[a]==2 || data.types[a]==4));
    if(data.top)out[68]=!(a==0 && data.p[0]==data.upper[0] && (data.types[0]==2 || data.types[0]==4));
    for(int s=0;s<5;++s)out[70+s]=data.floor[s];out[75]=data.p[0];out[76]=data.p[1];out[77]=data.p[2];out[78]=a+1;out[79]=4;
}
}
int main(int argc,char **argv)
{
    if(argc!=4)return 2;FILE *file=std::fopen(argv[1],"rb");if(!file)return 2;
    int head[4];if(std::fread(head,4,4,file)!=4)return 2;
    const int im=head[0],jm=head[1],km=head[2],cases=head[3],halo=5;
    const size_t total=static_cast<size_t>(im+11)*(jm+11)*(km+11),physical=static_cast<size_t>(im+1)*(jm+1)*(km+1);
    const size_t plane_count=11*static_cast<size_t>(im+1)*(km+1),face_count=12*static_cast<size_t>(im+2)*(jm+2)*(km+2);
    std::vector<int> jobs(cases*8),mask(physical);double floor[5],dt;
    std::vector<double> table(total*31),initial(physical*22),top(plane_count);
    if(std::fread(jobs.data(),4,jobs.size(),file)!=jobs.size() || std::fread(floor,8,5,file)!=5 || std::fread(&dt,8,1,file)!=1 ||
        std::fread(table.data(),8,table.size(),file)!=table.size() || std::fread(initial.data(),8,initial.size(),file)!=initial.size() ||
        std::fread(top.data(),8,top.size(),file)!=top.size() || std::fread(mask.data(),4,mask.size(),file)!=mask.size())return 2;
    std::fclose(file);bool with_top=false,capturing=false;
    for(int j=0;j<cases;++j) {with_top=with_top || jobs[8*j+6]!=0;capturing=capturing || jobs[8*j+4]!=0;}
    check(astr_device_bind(0));check(MPI_Init(&argc,&argv));void *handle=nullptr;
    check(astr_device_state_create(0,im,jm,km,halo,std::atoi(argv[3]),&handle));check(astr_device_air5_prepare(handle));
    check(astr_device_air5_geometry_prepare(handle));check(astr_device_air5_rk_prepare(handle));check(astr_device_air5_faces_prepare(handle));
    check(astr_device_air5_symmetric_prepare(handle,floor,with_top));
    const int fields[]={ASTR_DEVICE_Q,ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES,
        ASTR_DEVICE_METRIC,ASTR_DEVICE_JACOBIAN};
    const int components[]={11,1,3,1,5,3,1},directions[]={1,1,1,1,1,3,1};
    std::vector<std::vector<double>> original;std::vector<AstrDeviceView> views;int column=0;
    for(int f=0;f<7;++f) {
        AstrDeviceView v;check(astr_device_state_view(handle,fields[f],&v));views.push_back(v);std::vector<double> data(size(v));
        for(int d=1;d<=directions[f];++d)for(int c=1;c<=components[f];++c) {
            size_t id=0;for(int k=-halo;k<=km+halo;++k)for(int j=-halo;j<=jm+halo;++j)for(int i=-halo;i<=im+halo;++i,++id)
                data[offset(v,i,j,k,c,d)]=table[31*id+column];
            ++column;
        }
        upload(handle,fields[f],data);original.push_back(std::move(data));
    }
    auto &state=*static_cast<astr::State *>(handle);
    if(capturing) {
        check(cudaMalloc(&state.shock_mask,physical));std::vector<int8_t> markers(mask.begin(),mask.end());
        check(cudaMemcpy(state.shock_mask,markers.data(),physical,cudaMemcpyHostToDevice));
    }
    AstrDeviceView saved,rhs,base,faces,top_view;
    check(astr_device_state_view(handle,ASTR_DEVICE_SAVE,&saved));check(astr_device_state_view(handle,ASTR_DEVICE_RHS,&rhs));
    check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,&base));check(astr_device_state_view(handle,ASTR_DEVICE_INTERFACE_FLUX,&faces));
    if(with_top)check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,&top_view));
    else if(astr_device_state_view(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,&top_view)!=ASTR_DEVICE_NOT_ALLOCATED)return 3;
    std::vector<double> saved_data(size(saved)),rhs_data(size(rhs));
    for(size_t id=0;id<physical;++id)for(int c=0;c<11;++c) {saved_data[id+c*physical]=initial[22*id+c];rhs_data[id+c*physical]=initial[22*id+11+c];}
    upload(handle,ASTR_DEVICE_SAVE,saved_data);if(with_top)upload(handle,ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH,top);
    int period[3],nulls[3]={MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL},sizes[3]={1,1,1};
    for(int a=0;a<3;++a)period[a]=jobs[a]==3;
    check(astr_device_transport_prepare(handle,static_cast<int>(MPI_Comm_c2f(MPI_COMM_SELF)),nulls,nulls,sizes,11,0,0,period));
    void *buffers[4];size_t capacity;check(astr_device_air5_symmetric_buffers(handle,buffers,&capacity));
    for(int slot=1;slot<4;++slot)if(static_cast<double *>(buffers[slot])-static_cast<double *>(buffers[0])!=static_cast<ptrdiff_t>(slot*capacity))return 4;
    file=std::fopen(argv[2],"wb");if(!file)return 2;bool written=write(file,&cases,4,1);
    const double a[]={1.,.75,1./3.},b[]={0.,.25,2./3.},c[]={1.,.25,2./3.};
    for(int job=0;job<cases;++job) {
        const int *option=jobs.data()+8*job;const int stage=option[3]-1,top_only=option[6];const double cdt=c[stage]*dt;
        for(int axis=0;axis<3;++axis)if(option[axis]!=jobs[axis])return 5;
        upload(handle,ASTR_DEVICE_RHS,rhs_data);
        upload(handle,ASTR_DEVICE_SAVE,saved_data);
        check(astr_device_air5_symmetric_base(handle,option,a[stage],b[stage],cdt,stage==0,top_only));
        if(physics(handle))return 6;
        for(int axis=0;axis<3;++axis)check(astr_device_exchange_axis(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,axis,0));
        const auto budget=download(handle,ASTR_DEVICE_AIR5_CONVECTION_BASE,base);
        if(top_only) {check(astr_device_air5_symmetric_top_normal(handle,option,cdt,option[5]));if(physics(handle))return 7;}
        const auto starting_rhs=download(handle,ASTR_DEVICE_RHS,rhs);
        std::vector<double> snapshots(3*face_count,0.);int final_status=0;
        for(int axis=0;axis<3;++axis) {
            if(top_only && axis==1)continue;
            check(astr_device_air5_symmetric_faces(handle,option,axis+1,option[4],cdt,top_only,option[5]));
            const int face_status=physics(handle);
            if(face_status) {
                const auto values=download(handle,ASTR_DEVICE_INTERFACE_FLUX,faces);
                std::fprintf(stderr,"owned face failure job=%d axis=%d status=%d stage=%d preserve=%d top=%d\n",job,axis+1,face_status,stage+1,option[5],top_only);
                bool captured=false;
                for(int k=0;k<=km && !captured;++k)for(int j=0;j<=jm && !captured;++j)for(int i=0;i<=im && !captured;++i)
                    if(values[offset(faces,i,j,k,12)]<0) {
                        Capture input{{i,j,k},{im,jm,km},{option[0],option[1],option[2]},axis,top_only,option[4],cdt,{}};
                        for(int s=0;s<5;++s)input.floor[s]=floor[s];double *device,out[80];check(cudaMalloc(&device,sizeof(out)));
                        capture_point<<<1,1,0,state.stream>>>(astr::air5::face_fields(state),base,input,state.shock_mask,device);
                        check(cudaGetLastError());check(cudaMemcpyAsync(out,device,sizeof(out),cudaMemcpyDeviceToHost,state.stream));
                        check(cudaStreamSynchronize(state.stream));check(cudaFree(device));
                        FILE *capture=std::fopen((std::string(argv[2])+".point.bin").c_str(),"wb");int one=1;
                        if(!capture)return 15;write(capture,&one,4,1);write(capture,out,8,80);std::fclose(capture);
                        std::fprintf(stderr,"captured face i/j/k=%d/%d/%d\n",i,j,k);captured=true;
                    }
                return 8;
            }
            check(astr_device_air5_symmetric_plane(handle,axis+1,0,period[axis],period[axis]));
            if(period[axis])check(astr_device_air5_symmetric_copy_local_planes(handle,axis+1));
            check(astr_device_air5_symmetric_plane(handle,axis+1,1,period[axis],period[axis]));
            if(option[7] && axis==0) {
                auto corrupt=download(handle,ASTR_DEVICE_INTERFACE_FLUX,faces);corrupt[offset(faces,0,1,1,1)]=-1e12;
                upload(handle,ASTR_DEVICE_INTERFACE_FLUX,corrupt);
            }
            const auto values=download(handle,ASTR_DEVICE_INTERFACE_FLUX,faces);size_t id=axis*face_count;
            for(int component=1;component<=12;++component)for(int k=-1;k<=km;++k)for(int j=-1;j<=jm;++j)for(int i=-1;i<=im;++i)
                snapshots[id++]=values[offset(faces,i,j,k,component)];
            check(astr_device_air5_symmetric_apply(handle,option,axis+1,cdt,top_only));final_status=physics(handle);
            if(option[7]) {if(final_status!=202)return 9;break;}
            if(final_status)return 10;
        }
        if(!option[7]) {check(astr_device_air5_symmetric_final(handle,option,a[stage],b[stage],cdt,stage==0,top_only));final_status=physics(handle);if(final_status)return 11;}
        const auto result=download(handle,ASTR_DEVICE_RHS,rhs);AstrAir5TransportFailure failure{};
        check(astr_device_air5_symmetric_failure(handle,&failure));
        if(option[7] && (failure.failure[0]!=1 || failure.failure[1]!=1 || failure.failure[2]!=1 || failure.failure[3]!=1 ||
            failure.failure[4]!=1 || failure.failure[5]!=202 || failure.failure[6]!=0 || failure.cdt!=cdt))return 12;
        const auto untouched_q=download(handle,ASTR_DEVICE_Q,views[0]);
        if(std::memcmp(untouched_q.data(),original[0].data(),untouched_q.size()*8))return 13;
        const auto untouched_save=download(handle,ASTR_DEVICE_SAVE,saved);
        if(std::memcmp(untouched_save.data(),saved_data.data(),untouched_save.size()*8))return 14;
        if(!option[7]) {
            int first[3]={0,0,0},last[3]={im,jm,km};
            for(int axis=0;axis<3;++axis) {
                if(option[axis]==1 || option[axis]==4)first[axis]=1;
                if(option[axis]==2 || option[axis]==4)--last[axis];
            }
            if(top_only) {first[1]=last[1]=jm;last[0]=im;first[2]=0;last[2]=km;}
            check(astr_device_air5_rk_update(handle,first,last,dt,a[stage],b[stage],c[stage],stage==0));
        }
        const auto advanced=download(handle,ASTR_DEVICE_Q,views[0]);std::vector<double> rk_state(11*physical);size_t id=0;
        for(int component=1;component<=11;++component)for(int k=0;k<=km;++k)for(int j=0;j<=jm;++j)for(int i=0;i<=im;++i)
            rk_state[id++]=advanced[offset(views[0],i,j,k,component)];
        written=written && write(file,option,4,8) && write(file,&final_status,4,1) && write(file,budget.data(),8,budget.size()) &&
            write(file,starting_rhs.data(),8,starting_rhs.size()) && write(file,snapshots.data(),8,snapshots.size()) && write(file,result.data(),8,result.size()) &&
            write(file,rk_state.data(),8,rk_state.size()) && write(file,failure.failure,4,7) && write(file,failure.values,8,66) && write(file,&failure.cdt,8,1);
        // Each job is an independent frozen stage, not a failed-step retry.
        upload(handle,ASTR_DEVICE_Q,original[0]);upload(handle,ASTR_DEVICE_SAVE,saved_data);
    }
    for(int f=0;f<7;++f) {const auto after=download(handle,fields[f],views[f]);if(std::memcmp(after.data(),original[f].data(),after.size()*8))return 13;}
    const auto after=download(handle,ASTR_DEVICE_SAVE,saved);if(std::memcmp(after.data(),saved_data.data(),after.size()*8))return 14;
    check(astr_device_state_destroy(&handle));check(MPI_Finalize());
    return std::fclose(file)==0 && written?0:2;
}
