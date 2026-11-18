#include "chemistry_face_math.cuh"
#include "chemistry_flux_math.cuh"
#include "chemistry_transport.h"

namespace {
using namespace astr::air5;
constexpr int threads=256;
struct Region {
    int first[3],width[3],upper[3],ntype[3];
    size_t count;
};
struct Stage {double floor[5],a,b,cdt;bool first;};
__device__ double &at(AstrDeviceView v,const int p[3],int c=1)
{return static_cast<double *>(v.data)[astr::offset(v,p[0],p[1],p[2],c)];}
__device__ void point(size_t id,Region region,int p[3])
{
    p[0]=region.first[0]+id%region.width[0];
    p[1]=region.first[1]+(id/region.width[0])%region.width[1];
    p[2]=region.first[2]+id/(static_cast<size_t>(region.width[0])*region.width[1]);
}
__device__ double rhs_sign(int c) {return c>=5 && c<10 ? -1. : 1.;}
__device__ double face_ratio(AstrDeviceView ratio,const int p[3],int axis,int face)
{
    int x[3]={p[0],p[1],p[2]};x[axis]=face;
    const double left=at(ratio,x);++x[axis];
    return fmin(left,at(ratio,x));
}
__device__ void face(FaceFields fields,const int p[3],Region region,int axis,int index,double flux[11],int channel,bool right)
{
    const DiffusiveLine line{fields,{p[0],p[1],p[2]},axis,channel,true};
    const int last=region.upper[axis],type=region.ntype[axis];
    const bool lower=(type==1 || type==4) && index<=1;
    const bool upper=(type==2 || type==4) && index>=last-2;
    const int first=lower?0:(upper?last-5:index-2);
    double v[6][11],jac[6];
    int point[3]={p[0],p[1],p[2]};
    for(int r=0;r<6;++r) {
        point[axis]=first+r; jac[r]=face_value(fields.jacobian,point);
        line.load(first+r,v[r],false);
    }
    // NVHPC uses opposite Jacobian-product contractions for the two
    // centered faces in a node RHS. The physical closures use their own rows.
    const bool reverse=right && !lower && !upper;
    for(int c=0;c<11;++c) {
        double s[6];for(int r=0;r<6;++r)s[r]=jac[r]*v[r][c];
        const double middle=reverse?fma(jac[2],v[2][c],s[3]):fma(jac[3],v[3][c],s[2]);
        const double near=reverse?fma(jac[1],v[1][c],s[4]):fma(jac[4],v[4][c],s[1]);
        const double outer=reverse?fma(jac[0],v[0][c],s[5]):fma(jac[5],v[5][c],s[0]);
        const double anchor=fma(1./60.,outer,fma(37.*(1./60.),middle,(-2./15.)*near));
        if(lower) {
            const double d2=fma(2./3.,fma(jac[3],v[3][c],-s[1]),(-1./12.)*fma(jac[4],v[4][c],-s[0]));
            const double d1=.5*fma(jac[2],v[2][c],-s[0]);
            const double d0=fma(-1.5,s[0],fma(-.5*jac[2],v[2][c],2.*s[1]));
            flux[c]=index==1?anchor-d2:(index==0?anchor-d2-d1:anchor-d2-d1-d0);
        } else if(upper) {
            const double d0=fma(2./3.,fma(jac[4],v[4][c],-s[2]),(-1./12.)*fma(jac[5],v[5][c],-s[1]));
            const double d1=.5*fma(jac[5],v[5][c],-s[3]);
            const double d2=fma(1.5,s[5],fma(.5*jac[3],v[3][c],-2.*s[4]));
            const int depth=index-(last-3);
            flux[c]=depth==1?anchor+d0:(depth==2?anchor+d0+d1:anchor+d0+d1+d2);
        } else flux[c]=anchor;
    }
}
__device__ void layered_face(FaceFields fields,AstrDeviceView ratio,const int p[3],Region region,int axis,int index,
                            double flux[11],bool right)
{
    const double theta=face_ratio(ratio,p,axis,index);
    for(int c=5;c<10;++c)flux[c]*=theta;
    if(theta==1.)return;
    double carried[11];face(fields,p,region,axis,index,carried,1,right);
    flux[4]=fma(theta-1.,carried[4],flux[4]);
    flux[10]=fma(theta-1.,carried[10],flux[10]);
}
__device__ void record(AstrAir5DiffusionReport *report,int *failure,const int p[3],int variable,
    double origin,double current,double increment,double base,double negative)
{
    atomicMax(failure,invalid_composition);
    if(atomicCAS(&report->claimed,0,1)!=0)return;
    for(int a=0;a<3;++a)report->location[a]=p[a];
    report->location[3]=variable;
    report->values[0]=origin;report->values[1]=current;report->values[2]=increment;
    report->values[3]=base;report->values[4]=negative;
}
__global__ void reset_budget(double *species,double *energy,size_t count,int *failure,AstrAir5DiffusionReport *report)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<count) {
        species[id]=1.;
        if(energy)energy[id]=1.;
    }
    if(id==0) {*failure=0;*report={};report->minimum_bits=__double_as_longlong(DBL_MAX);}
}

template<bool energy>
__global__ void diffusion_budget(FaceFields fields,AstrDeviceView save,AstrDeviceView rhs,AstrDeviceView species,
    AstrDeviceView energy_ratio,Region region,Stage stage,bool diagnostics,int *failure,AstrAir5DiffusionReport *report)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=region.count)return;
    int p[3];point(id,region,p);
    double left[3][11],right[3][11],q[11],origin[11],base[11],residual[11];
    const double jac=at(fields.jacobian,p);
    for(int c=0;c<11;++c) {
        q[c]=at(fields.q,p,c+1);
        origin[c]=stage.first?q[c]*jac:at(save,p,c+1);
        residual[c]=at(rhs,p,c+1);
        // Match the original NVHPC RK budget contraction, without changing
        // the device library's global contraction mode.
        base[c]=fma(stage.cdt,residual[c],fma(stage.a,origin[c],stage.b*q[c]*jac));
    }
    for(int d=0;d<3;++d) {
        face(fields,p,region,d,p[d]-1,left[d],0,false);face(fields,p,region,d,p[d],right[d],0,true);
        if(energy) {
            layered_face(fields,species,p,region,d,p[d]-1,left[d],false);
            layered_face(fields,species,p,region,d,p[d],right[d],true);
        }
    }
    double ratio=1.;
    if(!energy) {
        if(!state_admissible<true>(base,stage.floor))
            record(report,failure,p,7,base[0],translational_margin<true>(base),0.,0.,0.);
        for(int c=5;c<10;++c) {
            double negative=0.,flux_scale=0.;
            for(int d=0;d<3;++d) {
                negative=negative+fmin(0.,stage.cdt*left[d][c])+fmin(0.,-stage.cdt*right[d][c]);
                flux_scale=flux_scale+fabs(left[d][c])+fabs(right[d][c]);
            }
            if(base[c]<0.)record(report,failure,p,c-4,origin[c],q[c]*jac,stage.cdt*residual[c],base[c],negative);
            else if(negative<0.) {
                const double scale=fabs(stage.a*origin[c])+fabs(stage.b*q[c]*jac)+stage.cdt*(fabs(residual[c])+flux_scale);
                ratio=fmin(ratio,transport_species_ratio(base[c],negative,scale));
            }
        }
        if(!energy_ratio.data) {
            double negative=0.;
            for(int d=0;d<3;++d) {
                double l=left[d][10],r=right[d][10];
                for(int c=5;c<10;++c) {l+=stage.floor[c-5]*left[d][c];r+=stage.floor[c-5]*right[d][c];}
                negative=negative+fmin(0.,-stage.cdt*l)+fmin(0.,stage.cdt*r);
            }
            const double current=vibrational_excess<true>(q[10],q+5,stage.floor)*jac;
            const double initial=stage.first?current:vibrational_excess<true>(origin[10],origin+5,stage.floor);
            const double increment=stage.cdt*vibrational_excess<true>(residual[10],residual+5,stage.floor);
            const double available=stage.a*initial+stage.b*current+increment;
            if(available<0.)record(report,failure,p,6,initial,current,increment,available,negative);
            else if(negative<0.)ratio=fmin(ratio,interior_ratio(available/(-negative)));
        }
    }
    if(energy || !energy_ratio.data)for(int d=0;d<3;++d)for(int side=0;side<2;++side) {
        double correction[11]={};int mask,species_mask;
        const double factor=side ? stage.cdt : -stage.cdt;
        for(int c=1;c<11;++c)correction[c]=factor*rhs_sign(c)*(side ? right[d][c] : left[d][c]);
        ratio=fmin(ratio,admissible_face_ratio<true>(base,correction,false,mask,species_mask,energy,stage.floor));
    }
    if(energy)at(energy_ratio,p)=ratio;
    else {
        ratio=fmax(0.,fmin(1.,ratio));at(species,p)=ratio;
        if(diagnostics && ratio<1.-1.e-14) {
            atomicAdd(&report->limited,1);
            atomicMin(reinterpret_cast<unsigned long long *>(&report->minimum_bits),__double_as_longlong(ratio));
        }
    }
}
__global__ void diffusion_rhs(FaceFields fields,AstrDeviceView species,AstrDeviceView energy,AstrDeviceView rhs,
    Region region,int axis)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=region.count)return;
    int p[3];point(id,region,p);
    const int index=p[axis];
    double l[11],r[11];
    face(fields,p,region,axis,index-1,l,0,false);face(fields,p,region,axis,index,r,0,true);
    if(energy.data) {
        layered_face(fields,species,p,region,axis,index-1,l,false);
        layered_face(fields,species,p,region,axis,index,r,true);
    }
    const AstrDeviceView ratio=energy.data ? energy : species;
    const double low=face_ratio(ratio,p,axis,index-1),high=face_ratio(ratio,p,axis,index);
    for(int c=1;c<11;++c)at(rhs,p,c+1)+=rhs_sign(c)*fma(high,r[c],-low*l[c]);
}
int make_region(const astr::State &s,const int *ntypes,const int *first,const int *last,Region &region)
{
    if(!ntypes || !first || !last)return ASTR_DEVICE_INVALID_ARGUMENT;
    region.count=1;
    for(int d=0;d<3;++d) {
        if(first[d]<0 || last[d]>s.upper[d] || first[d]>last[d] || ntypes[d]<1 || ntypes[d]>4)
            return ASTR_DEVICE_INVALID_ARGUMENT;
        region.first[d]=first[d];region.width[d]=last[d]-first[d]+1;
        region.upper[d]=s.upper[d];region.ntype[d]=ntypes[d];region.count*=region.width[d];
    }
    return 0;
}
}

extern "C" int astr_device_air5_diffusion_prepare(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    auto &ratio=s.fields[ASTR_DEVICE_AIR5_DIFFUSION_RATIO];
    if(!s.viscous_ready || !s.air5_ready || !s.rhs_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.air5_diffusion_report)return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=ratio.view.data?0:astr::allocate_field(ratio,s.upper,s.halo,3);
    if(!status && s.fields[ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX].view.data)
        status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_DIFFUSION_ENERGY_RATIO],s.upper,s.halo,3);
    if(status)return status;
    return static_cast<int>(cudaMalloc(&s.air5_diffusion_report,sizeof(AstrAir5DiffusionReport)));
}
extern "C" int astr_device_air5_diffusion_begin(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_diffusion_report)return ASTR_DEVICE_NOT_ALLOCATED;
    const auto &ratio=s.fields[ASTR_DEVICE_AIR5_DIFFUSION_RATIO];
    const size_t count=ratio.view.stride[3];
    reset_budget<<<(count+threads-1)/threads,threads,0,s.stream>>>(static_cast<double *>(ratio.view.data),
        static_cast<double *>(s.fields[ASTR_DEVICE_AIR5_DIFFUSION_ENERGY_RATIO].view.data),count,s.air5_failure,s.air5_diffusion_report);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_diffusion_budget(void *handle,const int ntypes[3],const int first[3],const int last[3],
    const double floor[5],double a,double b,double cdt,int first_stage,int energy,int diagnostics)
{
    if(!handle || !floor || energy<0 || energy>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const auto e=s.fields[ASTR_DEVICE_AIR5_DIFFUSION_ENERGY_RATIO].view;
    if(!s.air5_diffusion_report || (energy && !e.data))return ASTR_DEVICE_NOT_ALLOCATED;
    Region region{};
    const int status=make_region(s,ntypes,first,last,region);
    if(status)return status;
    const Stage stage{{floor[0],floor[1],floor[2],floor[3],floor[4]},a,b,cdt,first_stage!=0};
    const auto save=s.fields[ASTR_DEVICE_SAVE].view,rhs=s.fields[ASTR_DEVICE_RHS].view;
    const auto ratio=s.fields[ASTR_DEVICE_AIR5_DIFFUSION_RATIO].view;
    const size_t grid=(region.count+threads-1)/threads;
    if(energy)diffusion_budget<true><<<grid,threads,0,s.stream>>>(
        face_fields(s),save,rhs,ratio,e,region,stage,false,s.air5_failure,s.air5_diffusion_report);
    else diffusion_budget<false><<<grid,threads,0,s.stream>>>(
        face_fields(s),save,rhs,ratio,e,region,stage,diagnostics!=0,s.air5_failure,s.air5_diffusion_report);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_diffusion_rhs(void *handle,const int ntypes[3],const int first[3],const int last[3],int axis)
{
    if(!handle || axis<1 || axis>3)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_diffusion_report)return ASTR_DEVICE_NOT_ALLOCATED;
    Region region{};
    const int status=make_region(s,ntypes,first,last,region);
    if(status)return status;
    diffusion_rhs<<<(region.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),
        s.fields[ASTR_DEVICE_AIR5_DIFFUSION_RATIO].view,s.fields[ASTR_DEVICE_AIR5_DIFFUSION_ENERGY_RATIO].view,
        s.fields[ASTR_DEVICE_RHS].view,region,axis-1);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_diffusion_report(void *handle,AstrAir5DiffusionReport *report)
{
    if(!handle || !report)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_diffusion_report)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemcpyAsync(report,s.air5_diffusion_report,sizeof(*report),cudaMemcpyDeviceToHost,s.stream));
    return status ? status : static_cast<int>(cudaStreamSynchronize(s.stream));
}
