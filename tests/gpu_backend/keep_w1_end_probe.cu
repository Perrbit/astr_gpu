#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include "device_state.h"
#include "fluid_explicit.h"
#include "halo_transport.h"
#include "keep_w1_operator.h"

namespace {
constexpr int im=6,km=7,ja=23,hm=5;
void check(int status,const char *operation)
{
    if(status==0) return;
    std::fprintf(stderr,"%s: %d\n",operation,status);
    MPI_Abort(MPI_COMM_WORLD,1);
}

std::vector<double> apply(int device,int first,int span,MPI_Comm comm,int before,int after,int mode,
    const int send[2],const int recv[2])
{
    void *s=nullptr;
    check(astr_device_state_create(device,im,span,km,hm,0,&s),"state");
    check(astr_device_fluid_prepare(s,0),"primitives");
    check(astr_device_fluid_rhs_prepare(s,0),"RHS and geometry");
    check(astr_keep_w1_prepare(s),"W1 tables and fluxes");
    int size;
    MPI_Comm_size(comm,&size);
    const int lower[]={MPI_PROC_NULL,before,MPI_PROC_NULL},upper[]={MPI_PROC_NULL,after,MPI_PROC_NULL};
    const int dims[]={1,size,1},periodic[]={1,0,1},split[]={0,size>1,0};
    check(astr_device_transport_prepare(s,MPI_Comm_c2f(comm),lower,upper,dims,14,mode,0,periodic),"halo transport");
    const int nx=im+1+2*hm,ny=span+1+2*hm,nz=km+1+2*hm,halo_count=nx*ny*nz;
    const int nodes=(im+1)*(span+1)*(km+1);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    std::vector<double> vel(3*halo_count,nan),temp(halo_count,nan),met(9*halo_count,nan),jac(halo_count,nan);
    std::vector<double> rhs(5*nodes,0.);
    for(int k=0;k<=km;++k) for(int j=-hm;j<=span+hm;++j) for(int i=0;i<=im;++i) {
        const int y=first+j;
        if(y<0 || y>ja) continue;
        const int n=i+hm+nx*(j+hm+ny*(k+hm));
        const double xx=.31*(i%im),zz=.37*(k%km),yy=double(y)/ja;
        vel[n]=.012+.17*std::sin(xx+yy)*std::cos(zz);
        vel[n+halo_count]=-.014+.13*std::sin(xx+zz-yy);
        vel[n+2*halo_count]=.02+.19*std::cos(xx-zz+2*yy);
        temp[n]=1.3+.07*std::cos(xx+zz-3*yy);
        jac[n]=1.2-.6*yy;
        met[n+4*halo_count]=1./jac[n];
    }
    const auto upload=[&](int field,const std::vector<double> &v) {
        check(astr_device_state_upload(s,field,v.data(),v.size()*sizeof(double)),"upload field");
    };
    upload(ASTR_DEVICE_VELOCITY,vel);
    upload(ASTR_DEVICE_TEMPERATURE,temp);
    upload(ASTR_DEVICE_METRIC,met);
    upload(ASTR_DEVICE_JACOBIAN,jac);
    upload(ASTR_DEVICE_RHS,rhs);
    const double inverse[]={4.,1.,3.},wall[]={1.1,1.05};
    // Reuse the same slots across two asynchronous RHS calls.
    for(int step=0;step<2;++step)
        check(astr_keep_w1_diffusion(s,first,ja,split,inverse,wall,3000.,.72,.036,.4,1.4,send,recv),"W1 diffusion");
    check(astr_device_state_download(s,ASTR_DEVICE_RHS,rhs.data(),rhs.size()*sizeof(double)),"download RHS");
    check(astr_device_state_destroy(&s),"release state");
    return rhs;
}
}

int main(int argc,char **argv)
{
    MPI_Init(&argc,&argv);
    int rank,size,devices;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    check(size!=3 || argc!=2 || (std::strcmp(argv[1],"pinned") && std::strcmp(argv[1],"device")),
        "usage: mpiexec -n 3 astr_backend_keep_w1_end pinned|device");
    const int mode=std::strcmp(argv[1],"device")==0 ? ASTR_HALO_DEVICE : ASTR_HALO_PINNED;
    check(astr_device_count(&devices),"device count");
    check(devices<2,"test needs two GPUs");
    const int first[]={0,7,15},span[]={7,8,8},none[]={0,0};
    const int send[]={rank==0 ? 2 : 0,rank==2 ? 3 : 0},recv[]={rank==1 ? 2 : 0,rank==1 ? 3 : 0};
    const auto serial=apply(rank%2,0,ja,MPI_COMM_SELF,MPI_PROC_NULL,MPI_PROC_NULL,mode,none,none);
    const auto slab=apply(rank%2,first[rank],span[rank],MPI_COMM_WORLD,
        rank>0 ? rank-1 : MPI_PROC_NULL,rank<2 ? rank+1 : MPI_PROC_NULL,mode,send,recv);
    const int nx=im+1,nz=km+1,ny=span[rank]+1;
    double error=0.;
    for(int m=0;m<5;++m) for(int k=0;k<nz;++k) for(int j=0;j<ny;++j) for(int i=0;i<nx;++i) {
        const double x=slab[i+nx*(j+ny*(k+nz*m))];
        const double ref=serial[i+nx*(first[rank]+j+(ja+1)*(k+nz*m))];
        check(!std::isfinite(x) || !std::isfinite(ref),"nonfinite RHS");
        error=std::max(error,std::abs(x-ref)/std::max(1.,std::abs(ref)));
    }
    double global;
    MPI_Allreduce(&error,&global,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if(rank==0) std::printf("W1 %s: 7+8+8, middle rank receives 2+3 end planes, max scaled RHS difference %.17g\n",
        argv[1],global);
    check(global>2.e-10,"RHS mismatch");
    MPI_Finalize();
}
