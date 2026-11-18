#include "chemistry_face_math.cuh"
#include "chemistry_flux_math.cuh"
#include "chemistry_transport.h"

namespace {
using namespace astr::air5;
constexpr int threads=256;
struct Region {int first[3],width[3],upper[3],type[3];size_t count;};
struct Stage {double floor[5],a,b,cdt;bool first,capturing,species_budget,collect;int phase;};
struct NodeFaces {double l[3][11],r[3][11],ll[3][11],lr[3][11];};

__device__ double &at(AstrDeviceView v,const int p[3],int c=1)
{return static_cast<double *>(v.data)[astr::offset(v,p[0],p[1],p[2],c)];}
__device__ void point(size_t id,Region region,int p[3])
{
    p[0]=region.first[0]+id%region.width[0];
    p[1]=region.first[1]+(id/region.width[0])%region.width[1];
    p[2]=region.first[2]+id/(static_cast<size_t>(region.width[0])*region.width[1]);
}
__device__ double face_ratio(AstrDeviceView ratio,const int p[3],int d,int face)
{
    int x[3]={p[0],p[1],p[2]};x[d]=face;
    const double value=at(ratio,x);++x[d];
    return fmin(value,at(ratio,x));
}
__device__ double retained_species(const double f[11])
{
    double sum=0.;
    for(int c=6;c<10;++c)sum+=f[c];
    return sum;
}
__device__ void consistent_face(double high[11],double low[11],int phase,double theta)
{
    if(phase==1) {
        for(int c=5;c<10;++c)high[c]=low[c];
    } else {
        for(int c=0;c<11;++c)if(c<5 || c==10) {
            low[c]=theta==1.?high[c]:fma(theta,high[c]-low[c],low[c]);
            high[c]=low[c];
        }
        low[5]=low[0]-retained_species(low);
    }
    high[5]=high[0]-retained_species(high);
}
template<bool fused=false>
__device__ int high_face(ConvectiveLine line,Region shape,const int8_t *mask,int d,int index,
                        bool capturing,bool symmetric,double out[11],bool right=false)
{
    int status=0,p[3]={line.p[0],line.p[1],line.p[2]};p[d]=index;
    line.right=right;
    const bool shock=capturing && air5_shock_face(mask,shape.upper,p,d,shape.type[d]);
    convective_face<fused>(line,index,shape.upper[d],shape.type[d],shock,false,symmetric,out,status);
    return status;
}
__device__ int node_faces(FaceFields fields,AstrDeviceView fluid,const int8_t *mask,Region shape,
                         const int p[3],bool capturing,int phase,NodeFaces &f)
{
    int status=0;
    for(int d=0;d<3;++d) {
        const ConvectiveLine line{fields,{p[0],p[1],p[2]},d,true,false,false,capturing};
        const ConvectiveLine right_line{fields,{p[0],p[1],p[2]},d,true,true,false,capturing};
        const int face=p[d];
        int low_l=0,low_r=0;
        status=max(status,high_face<true>(line,shape,mask,d,face-1,capturing,false,f.l[d]));
        status=max(status,high_face<true>(line,shape,mask,d,face,capturing,false,f.r[d],true));
        convective_face<true>(line,face-1,shape.upper[d],shape.type[d],false,true,false,f.ll[d],low_l);
        convective_face<true>(right_line,face,shape.upper[d],shape.type[d],false,true,false,f.lr[d],low_r);
        status=max(status,max(low_l,low_r));
        if(phase) {
            consistent_face(f.l[d],f.ll[d],phase,phase==2?face_ratio(fluid,p,d,face-1):1.);
            consistent_face(f.r[d],f.lr[d],phase,phase==2?face_ratio(fluid,p,d,face):1.);
        }
    }
    return status;
}
__global__ void raw_kernel(FaceFields fields,AstrDeviceView rhs,const int8_t *mask,int *failure,
                           Region shape,int d,bool capturing,bool symmetric)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=shape.count)return;
    int p[3];point(id,shape,p);
    const ConvectiveLine line{fields,{p[0],p[1],p[2]},d,capturing,false,true,capturing};
    if(capturing) {
        double l[11]={},r[11]={};
        const int low=high_face<true>(line,shape,mask,d,p[d]-1,true,symmetric,l);
        const int high=high_face<true>(line,shape,mask,d,p[d],true,symmetric,r,true);
        if(low || high)atomicMax(failure,max(low,high));
        for(int c=0;c<11;++c)at(rhs,p,c+1)=at(rhs,p,c+1)+l[c]-r[c];
    } else {
        const int type=shape.type[d],last=shape.upper[d],index=p[d];
        const int depth=(type==1 || type==4) && index<=2?index:
                        ((type==2 || type==4) && index>=last-2?last-index:3);
        const int reach=depth==1?1:(depth==2?2:3);
        double l[3][11],r[3][11];
        for(int j=0;j<reach;++j) {line.load(index-j-1,l[j]);line.load(index+j+1,r[j]);}
        for(int c=0;c<11;++c) {
            const double d1=r[0][c]-l[0][c];
            const double derivative=reach==1?.5*d1:(reach==2?(2./3.)*d1-(1./12.)*(r[1][c]-l[1][c]):
                                             .75*d1-.15*(r[1][c]-l[1][c])+(1./60.)*(r[2][c]-l[2][c]));
            at(rhs,p,c+1)-=derivative;
        }
    }
}
__device__ void failure_at(AstrAir5ConvectionReport *report,int *status,const int p[3],int variable,
                           double a,double b,double c=0.,double d=0.)
{
    atomicMax(status,invalid_composition);
    if(atomicCAS(&report->claimed,0,1)!=0)return;
    for(int n=0;n<3;++n)report->location[n]=p[n];
    report->location[3]=variable;
    report->values[0]=a;report->values[1]=b;report->values[2]=c;report->values[3]=d;
}
__device__ void minimum_at(AstrAir5ConvectionReport *report,int entry,double ratio)
{
    atomicAdd(&report->active[entry],1);
    atomicMin(reinterpret_cast<unsigned long long *>(&report->minimum_bits[entry]),
              static_cast<unsigned long long>(__double_as_longlong(ratio)));
}
__global__ void reset_budget(double *ratio,double *fluid,size_t count,int *failure,AstrAir5ConvectionReport *report)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<count) {
        ratio[id]=1.;
        if(fluid)fluid[id]=1.;
    }
    if(id==0) {
        *failure=0;*report={};
        for(int n=0;n<10;++n)report->minimum_bits[n]=__double_as_longlong(DBL_MAX);
    }
}
__global__ void budget_kernel(FaceFields fields,AstrDeviceView save,AstrDeviceView ratio,AstrDeviceView fluid,
    const int8_t *mask,Region shape,Stage step,int *failure,AstrAir5ConvectionReport *report)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=shape.count)return;
    int p[3];point(id,shape,p);
    NodeFaces f{};
    const int status=node_faces(fields,fluid,mask,shape,p,step.capturing,step.phase,f);
    if(status)atomicMax(failure,status);
    double low[11],origin[11],current[11];
    const double jac=at(fields.jacobian,p);
    for(int c=0;c<11;++c) {
        double rhs=0.;
        for(int d=0;d<3;++d)rhs=rhs+f.ll[d][c]-f.lr[d][c];
        current[c]=at(fields.q,p,c+1)*jac;
        origin[c]=step.first?current[c]:at(save,p,c+1);
        low[c]=step.a*origin[c]+step.b*current[c]+step.cdt*rhs;
    }
    const double excess=vibrational_excess(low[10],low+5,step.floor);
    if(!state_admissible(low,step.floor)) {
        double minimum=low[5];for(int c=6;c<10;++c)minimum=fmin(minimum,low[c]);
        failure_at(report,failure,p,7,low[0],minimum,excess,translational_margin(low));
    }
    double minimum[10];for(int i=0;i<10;++i)minimum[i]=1.;
    for(int c=5;c<10;++c) {
        double pminus=0.;
        for(int d=0;d<3;++d)
            pminus=pminus+fmin(0.,step.cdt*(f.l[d][c]-f.ll[d][c]))+fmin(0.,-step.cdt*(f.r[d][c]-f.lr[d][c]));
        if(low[c]<0.)failure_at(report,failure,p,c-4,low[c],pminus);
        else if(pminus<0.) {
            double flux_scale=0.;
            for(int d=0;d<3;++d) {
                flux_scale=flux_scale+2.*(fabs(f.ll[d][c])+fabs(f.lr[d][c]))+fabs(f.l[d][c])+fabs(f.r[d][c]);
                if(step.phase && c==5) {
                    flux_scale=flux_scale+fabs(f.ll[d][0])+fabs(f.lr[d][0])+fabs(f.l[d][0])+fabs(f.r[d][0]);
                    const double *tables[]={f.ll[d],f.lr[d],f.l[d],f.r[d]};
                    for(const double *flux:tables) {
                        double retained=0.;for(int s=6;s<10;++s)retained+=fabs(flux[s]);
                        flux_scale+=retained;
                    }
                }
            }
            double scale=fabs(step.a*origin[c])+fabs(step.b*current[c])+step.cdt*flux_scale;
            if(step.phase)scale*=3.;
            const double candidate=transport_species_ratio(low[c],pminus,scale);
            minimum[0]=fmin(minimum[0],candidate);
            minimum[1]=fmin(minimum[1],candidate);
            minimum[c]=candidate;
        }
    }
    double pminus=0.;
    for(int d=0;d<3;++d) {
        double l=f.l[d][10]-f.ll[d][10],r=f.r[d][10]-f.lr[d][10];
        for(int s=0;s<5;++s) {
            l-=step.floor[s]*(f.l[d][s+5]-f.ll[d][s+5]);
            r-=step.floor[s]*(f.r[d][s+5]-f.lr[d][s+5]);
        }
        pminus=pminus+fmin(0.,step.cdt*l)+fmin(0.,-step.cdt*r);
    }
    if(excess<0.)failure_at(report,failure,p,6,excess,pminus);
    else if(pminus<0.) {
        minimum[2]=interior_ratio(excess/(-pminus));
        minimum[0]=fmin(minimum[0],minimum[2]);
    }
    for(int d=0;d<3;++d)for(int side=0;side<2;++side) {
        double correction[11];int mask_bits,species_bits;
        for(int c=0;c<11;++c)correction[c]=side?-step.cdt*(f.r[d][c]-f.lr[d][c]):step.cdt*(f.l[d][c]-f.ll[d][c]);
        const double candidate=admissible_face_ratio(low,correction,step.collect,mask_bits,species_bits,step.species_budget,step.floor);
        minimum[0]=fmin(minimum[0],candidate);
        if(step.collect && candidate<1.) {
            for(int n=0;n<4;++n)if(mask_bits&(1<<n))minimum[1+n]=fmin(minimum[1+n],candidate);
            for(int s=0;s<5;++s)if(species_bits&(1<<s))minimum[5+s]=fmin(minimum[5+s],candidate);
        }
    }
    at(ratio,p)=minimum[0];
    if(step.collect) {
        if(minimum[0]<1.-1.e-14)minimum_at(report,0,minimum[0]);
        for(int i=1;i<10;++i)if(minimum[i]<1.)minimum_at(report,i,minimum[i]);
    }
}
__global__ void rhs_kernel(FaceFields fields,AstrDeviceView rhs,AstrDeviceView ratio,AstrDeviceView fluid,
    const int8_t *mask,Region shape,bool capturing,int phase,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=shape.count)return;
    int p[3];point(id,shape,p);
    NodeFaces f{};
    const int status=node_faces(fields,fluid,mask,shape,p,capturing,phase,f);
    if(status) {atomicMax(failure,status);return;}
    double l[3],r[3];
    for(int d=0;d<3;++d) {
        l[d]=face_ratio(ratio,p,d,p[d]-1);
        r[d]=face_ratio(ratio,p,d,p[d]);
    }
    for(int c=0;c<11;++c) {
        double value=0.;
        for(int d=0;d<3;++d) {
            value=fma(l[d],f.l[d][c]-f.ll[d][c],value+f.ll[d][c]-f.lr[d][c]);
            value=fma(-r[d],f.r[d][c]-f.lr[d][c],value);
        }
        at(rhs,p,c+1)=value;
    }
}
int region_for(const astr::State &s,const int *ntypes,const int *first,const int *last,int capturing,Region &shape)
{
    if(!ntypes || !first || !last || capturing<0 || capturing>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.air5_ready || !s.fluid_ready || !s.rhs_ready || (capturing && !s.shock_mask))return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.halo<(capturing?4:3))return ASTR_DEVICE_INVALID_ARGUMENT;
    shape.count=1;
    for(int d=0;d<3;++d) {
        if(first[d]<0 || last[d]>s.upper[d] || first[d]>last[d] || ntypes[d]<1 || ntypes[d]>4 ||
           (ntypes[d]!=3 && s.upper[d]<5))return ASTR_DEVICE_INVALID_ARGUMENT;
        shape.first[d]=first[d];shape.width[d]=last[d]-first[d]+1;
        shape.upper[d]=s.upper[d];shape.type[d]=ntypes[d];shape.count*=shape.width[d];
    }
    return 0;
}
int ratio_field(int phase)
{return phase==1 ? ASTR_DEVICE_AIR5_CONVECTION_FLUID_RATIO : ASTR_DEVICE_AIR5_DIFFUSION_RATIO;}
}

extern "C" int astr_device_air5_raw_convection(void *handle,const int ntypes[3],const int first[3],const int last[3],
    int capturing,int symmetric)
{
    if(!handle || symmetric<0 || symmetric>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    Region shape{};int status=region_for(s,ntypes,first,last,capturing,shape);
    if(status)return status;
    const auto &rhs=s.fields[ASTR_DEVICE_RHS];
    status=static_cast<int>(cudaMemsetAsync(rhs.view.data,0,rhs.bytes,s.stream));if(status)return status;
    status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    for(int d=0;d<3;++d) {
        raw_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(
            face_fields(s),rhs.view,s.shock_mask,s.air5_failure,shape,d,capturing!=0,symmetric!=0);
        status=astr::finish_launch(s);if(status)return status;
    }
    return 0;
}
extern "C" int astr_device_air5_convection_budget_prepare(void *handle,int consistent)
{
    if(!handle || consistent<0 || consistent>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fluid_ready || !s.rhs_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.air5_convection_report)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &ratio=s.fields[ASTR_DEVICE_AIR5_DIFFUSION_RATIO];
    int status=ratio.view.data?0:astr::allocate_field(ratio,s.upper,s.halo,3);
    if(!status && consistent)status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_CONVECTION_FLUID_RATIO],s.upper,s.halo,3);
    if(status)return status;
    return static_cast<int>(cudaMalloc(&s.air5_convection_report,sizeof(AstrAir5ConvectionReport)));
}
extern "C" int astr_device_air5_convection_budget_begin(void *handle,int phase)
{
    if(!handle || phase<0 || phase>2)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const auto &ratio=s.fields[ASTR_DEVICE_AIR5_DIFFUSION_RATIO];
    const auto &fluid=s.fields[ASTR_DEVICE_AIR5_CONVECTION_FLUID_RATIO];
    if(!s.air5_convection_report || !ratio.view.data || (phase && !fluid.view.data))return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=phase==2?0:ratio.bytes/sizeof(double);
    const size_t blocks=count?(count+threads-1)/threads:1;
    reset_budget<<<blocks,threads,0,s.stream>>>(static_cast<double *>(ratio.view.data),
        phase==1?static_cast<double *>(fluid.view.data):nullptr,count,s.air5_failure,s.air5_convection_report);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_convection_budget(void *handle,const int ntypes[3],const int first[3],const int last[3],
    const double floor[5],double a,double b,double cdt,int first_stage,int capturing,int species_budget,int phase,int collect)
{
    if(!handle || !floor || phase<0 || phase>2)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const auto fluid=s.fields[ASTR_DEVICE_AIR5_CONVECTION_FLUID_RATIO].view, ratio=s.fields[ratio_field(phase)].view;
    if(!s.air5_convection_report || !ratio.data || (phase==2 && !fluid.data))return ASTR_DEVICE_NOT_ALLOCATED;
    Region shape{};const int status=region_for(s,ntypes,first,last,capturing,shape);
    if(status)return status;
    const Stage step{{floor[0],floor[1],floor[2],floor[3],floor[4]},a,b,cdt,first_stage!=0,capturing!=0,species_budget!=0,collect!=0,phase};
    budget_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),
        s.fields[ASTR_DEVICE_SAVE].view,ratio,fluid,s.shock_mask,shape,step,s.air5_failure,s.air5_convection_report);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_convection_rhs(void *handle,const int ntypes[3],const int first[3],const int last[3],
    int capturing,int phase)
{
    if(!handle || (phase!=0 && phase!=2))return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const auto fluid=s.fields[ASTR_DEVICE_AIR5_CONVECTION_FLUID_RATIO].view, ratio=s.fields[ratio_field(phase)].view;
    if(!s.air5_convection_report || !ratio.data || (phase==2 && !fluid.data))return ASTR_DEVICE_NOT_ALLOCATED;
    Region shape{};const int status=region_for(s,ntypes,first,last,capturing,shape);
    if(status)return status;
    rhs_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),
        s.fields[ASTR_DEVICE_RHS].view,ratio,fluid,s.shock_mask,shape,capturing!=0,phase,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_convection_read_report(void *handle,AstrAir5ConvectionReport *report)
{
    if(!handle || !report)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_convection_report)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemcpyAsync(report,s.air5_convection_report,sizeof(*report),cudaMemcpyDeviceToHost,s.stream));
    return status?status:static_cast<int>(cudaStreamSynchronize(s.stream));
}
